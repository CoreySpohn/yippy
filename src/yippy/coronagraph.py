"""Base coronagraph class."""

from pathlib import Path

import astropy.io.fits as pyfits
import astropy.units as u
import jax.numpy as jnp
import numpy as np
from lod_unit import lod
from scipy.interpolate import make_interp_spline
from tqdm import tqdm

from .header import HeaderData
from .jax_funcs import enable_x64, set_host_device_count, set_platform
from .logger import logger
from .offax import OffAx
from .offjax import OffJAX
from .sky_trans import SkyTrans
from .stellar_intens import StellarIntens
from .util import (
    convert_to_pix,
    extract_and_oversample_subarray,
    load_coro_performance_from_fits,
    measure_flux_in_oversampled_aperture,
    save_coro_performance_to_fits,
)


class Coronagraph:
    """Primary object for simulating a coronagraph.

    The Coronagraph object manages the coronagraph response for both on-axis
    and off-axis sources. It is primarily called with either
    `:ref:pydata:Coronagraph.offax(x,y)`, to get the off-axis response at a
    given (x,y) offset from the star, or `Coronagraph.stellar(r)` to get the
    coronagraph response at a given stellar angular diameter r.
    """

    def __init__(
        self,
        yip_path: Path,
        use_jax: bool = True,
        use_x64: bool = False,
        stellar_intens_file: str = "stellar_intens.fits",
        stellar_diam_file: str = "stellar_intens_diam_list.fits",
        offax_data_file: str = "offax_psf.fits",
        offax_offsets_file: str = "offax_psf_offset_list.fits",
        sky_trans_file: str = "sky_trans.fits",
        performance_file: str = "coro_perf.fits",
        x_symmetric: bool = True,
        y_symmetric: bool = False,
        cpu_cores: int = 1,
        platform: str = "cpu",
    ):
        """Initialize the Coronagraph object.

        Loads the coronagraph data from the given yield input package and creates
        the interpolation functions for the stellar intensity and off-axis PSF.

        Args:
            yip_path (Path):
                Yield input package directory. Must have fits files
                    offax_psf_offset_list - The off-axis PSF list
                    offax_psf - PSF of off-axis sources
                    sky_trans - Sky transmission data
            use_jax (bool):
                Whether to use JAX for optimized computation. Default is False.
            use_x64 (bool):
                Whether to use 64-bit floating point precision. Default is False.
            stellar_intens_file (str):
                Name of the stellar intensity file. Default is stellar_intens.fits
            stellar_diam_file (str):
                Name of the stellar intensity diameter list file. Default is
                stellar_intens_diam_list.fits
            offax_data_file (str):
                Name of the off-axis PSF file. Default is offax_psf.fits
            offax_offsets_file (str):
                Name of the off-axis PSF offset list file. Default is
                offax_psf_offset_list.fits
            sky_trans_file (str):
                Name of the sky transmission data file. Default is sky_trans.fits.
            performance_file (str):
                Name of the coronagraph performance (contrast, throughput) fits file.
            x_symmetric (bool):
                Whether off-axis PSFs are symmetric about the x-axis. Default is True.
            y_symmetric (bool):
                Whether off-axis PSFs are symmetric about the y-axis. Default is False.
            cpu_cores (int):
                Number of CPU cores to use. Default is 1.
            platform (str):
                Platform to use for JAX computation. Default is "cpu". Options are
                "cpu", "gpu", "tpu".
        """
        ###################
        # Read input data #
        ###################
        yip_path = Path(yip_path)
        self.yip_path = yip_path

        logger.info(f"Creating {yip_path.stem} coronagraph")

        self.name = yip_path.stem
        # Get header and calculate the lambda/D value
        stellar_intens_header = pyfits.getheader(Path(yip_path, stellar_intens_file), 0)

        # Get pixel scale with units
        self.header = HeaderData.from_fits_header(stellar_intens_header)
        self.pixel_scale = self.header.pixscale
        self.frac_obscured = self.header.obscured

        # Stellar intensity of the star being observed as function of stellar
        # angular diameter (unitless)
        self.stellar_intens = StellarIntens(
            yip_path, stellar_intens_file, stellar_diam_file
        )

        # Offaxis PSF of the planet as function of separation from the star
        if use_jax:
            # Apply JAX settings
            if use_x64:
                enable_x64()
            if platform != "cpu":
                set_platform(platform)
            elif cpu_cores > 1:
                set_host_device_count(cpu_cores)
            self.offax = OffJAX(
                yip_path,
                offax_data_file,
                offax_offsets_file,
                self.pixel_scale,
                x_symmetric,
                y_symmetric,
                cpu_cores,
                platform,
            )
        else:
            self.offax = OffAx(
                yip_path,
                offax_data_file,
                offax_offsets_file,
                self.pixel_scale,
                x_symmetric,
                y_symmetric,
            )

        # Get the sky_trans mask
        self.sky_trans = SkyTrans(yip_path, sky_trans_file)

        # PSF datacube here is a 4D array of PSFs at each pixel (x psf offset,
        # y psf offset, x, y). Given the computational cost of generating this
        # datacube, it is only generated when needed.
        self.has_psf_datacube = False

        # Shape of the images in the PSFs
        self.psf_shape = np.array([self.header.naxis1, self.header.naxis2])
        assert self.psf_shape[0] == self.psf_shape[1], "PSF must be square"
        self.npixels = self.psf_shape[0]

        # Get the contrast and throughput
        perf_path = Path(self.yip_path, performance_file)
        if perf_path.exists():
            sep, throughput, raw_contrast = load_coro_performance_from_fits(
                performance_file, self.yip_path
            )
        else:
            logger.info("No precomputed performance file found. Computing now...")
            sep_throughput, throughput = self.get_throughput_curve(plot=False)
            sep_contrast, raw_contrast = self.get_contrast_curve(plot=False)

            assert np.all(
                sep_throughput == sep_contrast
            ), "Mismatch in separations for performance parameters"
            sep = sep_throughput

            # Save to fits
            save_coro_performance_to_fits(
                sep, throughput, raw_contrast, performance_file, self.yip_path
            )

        # Create splines
        self.throughput_interp = make_interp_spline(sep, throughput, k=3)
        self.raw_contrast_interp = make_interp_spline(sep, raw_contrast, k=3)

        logger.info(f"Created {yip_path.stem}")

    def create_psf_datacube(self, batch_size=128):
        """Load the PSF datacube from a file or generate it if it doesn't exist.

        The PSF datacube is a 4D array of PSFs at each pixel (x psf offset,
        y psf offset, x, y). Given the computational cost of generating this
        datacube, it is only generated when needed and saved to a numpy binary
        file in the yip_path directory.

        Args:
            batch_size (int):
                Number of PSFs to generate in each batch. Default is 128.
        """
        datacube_path = Path(self.yip_path, "psf_datacube.npy")
        if datacube_path.exists():
            logger.info(f"Loading PSF datacube from {datacube_path}.")
            psfs = jnp.load(datacube_path)
            self.has_psf_datacube = True
        else:
            # Create data cube of spatially dependent PSFs.
            psfs_shape = (*self.psf_shape, *self.psf_shape)
            psfs = np.zeros(psfs_shape, dtype=np.float32)
            pixel_lod = (
                (np.arange(self.npixels) - ((self.npixels - 1) // 2))
                * u.pixel
                * self.pixel_scale
            ).value

            # Get the pixel coordinates for the PSF evaluations
            x_lod, y_lod = np.meshgrid(pixel_lod, pixel_lod, indexing="xy")
            points = np.column_stack((x_lod.flatten(), y_lod.flatten()))
            n_points = points.shape[0]

            logger.info(
                "Calculating data cube of spatially dependent PSFs, please hold..."
            )
            with tqdm(total=n_points, desc="Computing PSFs") as pb:
                for i in range(0, n_points, batch_size):
                    # Select the current batch
                    batch_points = points[i : i + batch_size]
                    batch_psfs = self.offax.create_psfs_parallel(
                        batch_points[:, 0], batch_points[:, 1]
                    )

                    # Store the batch in the data cube
                    psfs.reshape((-1, self.npixels, self.npixels))[
                        i : i + batch_size
                    ] = batch_psfs
                    pb.update(batch_points.shape[0])
            jnp.save(datacube_path, psfs)
            logger.info(f"PSF datacube saved to {datacube_path}.")

        self.psf_datacube = psfs

    def __repr__(self):
        """String representation of the Coronagraph object."""
        base_str = f"Coronagraph {self.name} ({self.yip_path})\n"
        base_str += f"{self.offax.type} off-axis PSFs, {self.offax.n_psfs} provided"
        if self.has_psf_datacube:
            base_str += f"\n{base_str}\nPSF datacube loaded"
        return base_str

    def get_throughput_curve(self, aperture_radius_lod=0.7, oversample=4, plot=True):
        """Creates the coronagraph throughput curve.

        Compute the coronagraph throughput vs. separation using ONLY the
        provided planet off-axis PSFs (no interpolation).
        We define throughput as the fraction of the total flux
        (planet PSF normalized to 1) that lands inside a photometric aperture.
        """
        separations = []
        throughputs = []

        # Loop through all provided offsets
        for i, x_lod_val in enumerate(np.array(self.offax.x_offsets)):
            for j, y_lod_val in enumerate(np.array(self.offax.y_offsets)):
                psf_img = self.offax.reshaped_psfs[i, j]
                r = np.sqrt(x_lod_val**2 + y_lod_val**2)

                # Aperture radius in pixel units
                radius_pix = aperture_radius_lod / self.pixel_scale.value

                # Planet coords in pixel space
                px = convert_to_pix(
                    x_lod_val, self.offax.center_x, self.pixel_scale
                ).value.astype(int)
                py = convert_to_pix(
                    y_lod_val, self.offax.center_y, self.pixel_scale
                ).value.astype(int)

                # Extract & oversample
                subarr_oversamp, px_os, py_os, radius_os, subarr_orig = (
                    extract_and_oversample_subarray(
                        psf_img, px, py, radius_pix, oversample
                    )
                )

                # Measure flux in aperture
                planet_flux_in_ap = measure_flux_in_oversampled_aperture(
                    subarr_oversamp, px_os, py_os, radius_os, subarr_orig
                )

                # planet_flux_in_ap is the throughput if psf_img sums to 1
                throughput = planet_flux_in_ap
                separations.append(r)
                throughputs.append(throughput)

        # Sort by separation
        separations = np.array(separations)
        throughputs = np.array(throughputs)
        idx_sort = np.argsort(separations)
        separations = separations[idx_sort]
        throughputs = throughputs[idx_sort]

        if plot:
            import matplotlib.pyplot as plt

            plt.figure()
            plt.plot(separations, throughputs, "o-", ms=6)
            plt.xlabel("Separation [λ/D]")
            plt.ylabel("Throughput")
            plt.title(f"{self.name} Throughput")
            plt.grid(True)
            plt.show()

        return separations, throughputs

    def get_contrast_curve(
        self, stellar_diam=0 * lod, aperture_radius_lod=0.7, oversample=4, plot=True
    ):
        """Creates the raw contrast curve.

        Compute a photometric aperture–based contrast curve vs. separation,
        using the provided offsets.

        Args:
            stellar_diam (Quantity):
                The stellar diameter used for the star's PSF.
            aperture_radius_lod (float):
                The aperture radius in lambda/D.
            oversample (int):
                The oversampling factor for interpolation.
            plot (bool):
                Whether to plot the contrast curve.
        """
        # Grab star PSF for given stellar diameter
        star_psf = self.stellar_intens(stellar_diam)

        separations = []
        contrasts = []

        for i, x_lod_val in enumerate(np.array(self.offax.x_offsets)):
            for j, y_lod_val in enumerate(np.array(self.offax.y_offsets)):
                # Planet PSF & separation
                planet_psf = self.offax.reshaped_psfs[i, j]
                r = np.sqrt(x_lod_val**2 + y_lod_val**2)

                # Aperture radius in pixel units
                radius_pix = aperture_radius_lod / self.pixel_scale.value

                # Planet flux in aperture
                px = convert_to_pix(
                    x_lod_val, self.offax.center_x, self.pixel_scale
                ).value.astype(int)
                py = convert_to_pix(
                    y_lod_val, self.offax.center_y, self.pixel_scale
                ).value.astype(int)
                subarr_oversamp_p, px_os, py_os, radius_os, subarr_orig_p = (
                    extract_and_oversample_subarray(
                        planet_psf, px, py, radius_pix, oversample
                    )
                )
                planet_flux_in_ap = measure_flux_in_oversampled_aperture(
                    subarr_oversamp_p, px_os, py_os, radius_os, subarr_orig_p
                )

                # Star flux in aperture at same offset
                subarr_oversamp_s, sx_os, sy_os, radius_os_s, subarr_orig_s = (
                    extract_and_oversample_subarray(
                        star_psf, px, py, radius_pix, oversample
                    )
                )
                star_flux_in_ap = measure_flux_in_oversampled_aperture(
                    subarr_oversamp_s, sx_os, sy_os, radius_os_s, subarr_orig_s
                )

                # Contrast
                if star_flux_in_ap > 0:
                    contrast_val = star_flux_in_ap / planet_flux_in_ap
                else:
                    contrast_val = np.nan

                separations.append(r)
                contrasts.append(contrast_val)

        # Sort by separation
        separations = np.array(separations)
        contrasts = np.array(contrasts)
        idx_sort = np.argsort(separations)
        separations = separations[idx_sort]
        contrasts = contrasts[idx_sort]

        if plot:
            import matplotlib.pyplot as plt

            plt.figure()
            plt.plot(separations, contrasts, "o-")
            plt.xlabel("Separation [λ/D]")
            plt.ylabel("Raw Contrast")
            plt.title(f"{self.name} Raw Contrast")
            plt.yscale("log")
            plt.grid(True)
            plt.show()

        return separations, contrasts
