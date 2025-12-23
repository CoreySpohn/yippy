"""Base coronagraph class."""

import json
from pathlib import Path

import astropy.io.fits as pyfits
import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
from lod_unit import lod
from scipy.interpolate import make_interp_spline
from scipy.optimize import root_scalar
from tqdm import tqdm

from ._version import __version__
from .header import HeaderData
from .jax_funcs import (
    enable_x64,
    set_host_device_count,
    set_platform,
)
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
        y_symmetric: bool = True,
        cpu_cores: int = 4,
        platform: str = "cpu",
        use_quarter_psf_datacube: bool = False,
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
            use_quarter_psf_datacube (bool):
                Whether to compute the PSF datacube in only the first quadrant.
                This is faster and uses less memory, but may not be accurate for
                all coronagraphs. Default is False.
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

        # Store platform and use_jax for later use
        self.platform = platform
        self.use_jax = use_jax

        # PSF datacube here is a 4D array of PSFs at each pixel (x psf offset,
        # y psf offset, x, y). Given the computational cost of generating this
        # datacube, it is only generated when needed.
        self.has_psf_datacube = False
        self.use_quarter_psf_datacube = use_quarter_psf_datacube

        # Shape of the images in the PSFs
        self.psf_shape = np.array([self.header.naxis1, self.header.naxis2])
        assert self.psf_shape[0] == self.psf_shape[1], "PSF must be square"
        self.npixels = self.psf_shape[0]

        # Append the version number to the performance file name
        performance_file = f"{performance_file}_v{__version__}.fits"

        # Get the contrast and throughput
        if self.offax.type == "1d":
            perf_path = Path(self.yip_path, performance_file)
            if perf_path.exists():
                # Performance file exists - load it
                logger.info(f"Loading performance metrics from {performance_file}")
                self.compute_all_performance_curves(
                    aperture_radius_lod=0.7,
                    save_to_fits=False,
                    load_from_file=performance_file,
                    plot=False,
                )
            else:
                # No performance file - compute all metrics
                logger.info(
                    "No precomputed performance file found. "
                    "Computing all performance metrics..."
                )
                self.compute_all_performance_curves(
                    aperture_radius_lod=0.7,
                    save_to_fits=True,
                    performance_file=performance_file,
                    plot=False,
                )
        else:
            logger.warning("2d contrast/throughput not supported currently")

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
        ext = "_quarter" if self.use_quarter_psf_datacube else ""
        datacube_path = Path(self.yip_path, f"psf_datacube{ext}.npy")
        if datacube_path.exists():
            logger.info(f"Loading PSF datacube from {datacube_path}.")
            psfs = jnp.load(datacube_path)
        else:
            # Create data cube of spatially dependent PSFs.
            psfs_shape = (*self.psf_shape, *self.psf_shape)
            psfs = np.zeros(psfs_shape, dtype=np.float32)
            if not self.use_quarter_psf_datacube:
                pixel_lod = (
                    (np.arange(self.npixels) - ((self.npixels - 1) // 2))
                    * u.pixel
                    * self.pixel_scale
                ).value
            else:
                center_idx = (self.npixels - 1) // 2
                pixel_lod = (
                    (np.arange(center_idx, self.npixels) - center_idx)
                    * u.pixel
                    * self.pixel_scale
                ).value
            n_src = len(pixel_lod)
            psfs_shape = (n_src, n_src, *self.psf_shape)
            psfs = np.zeros(psfs_shape, dtype=np.float32)

            # Get the pixel coordinates for the PSF evaluations
            x_lod, y_lod = np.meshgrid(pixel_lod, pixel_lod, indexing="xy")
            points = np.column_stack((x_lod.flatten(), y_lod.flatten()))
            n_points = points.shape[0]

            logger.info(
                f"Calculating {'quarter' if ext else 'full'} data cube of "
                f"spatially dependent PSFs ({n_points} points), please hold..."
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

        # Move datacube to GPU/TPU device if conditions are met
        if (
            self.use_quarter_psf_datacube
            and self.use_jax
            and self.platform in ("gpu", "tpu")
        ):
            logger.info(
                f"Moving PSF datacube to {self.platform.upper()} device "
                "(quarter symmetric datacube)"
            )
            # Convert to JAX array and place on device
            try:
                psfs = jax.device_put(jnp.array(psfs))
                logger.info(
                    f"Successfully moved PSF datacube to {self.platform.upper()} device"
                )
            except (MemoryError, RuntimeError) as e:
                logger.warning(
                    f"Failed to move PSF datacube to {self.platform.upper()} device "
                    f"(insufficient memory): {e}. Keeping datacube on CPU."
                )
                # psfs remains as numpy array on CPU

        self.has_psf_datacube = True
        self.psf_datacube = psfs

    def __repr__(self):
        """String representation of the Coronagraph object."""
        base_str = f"Coronagraph {self.name} ({self.yip_path})\n"
        base_str += f"{self.offax.type} off-axis PSFs, {self.offax.n_psfs} provided"

        # Add information about performance metrics
        interp_info = []
        if hasattr(self, "throughput_interp"):
            interp_info.append("throughput")
        if hasattr(self, "raw_contrast_interp"):
            interp_info.append("raw contrast")
        if hasattr(self, "occ_trans_interp"):
            interp_info.append("occulter transmission")
        if hasattr(self, "core_area_interp"):
            interp_info.append("core area")
        if hasattr(self, "core_intensity_interp"):
            interp_info.append("core mean intensity")

        if interp_info:
            base_str += f"\nInterpolators available: {', '.join(interp_info)}"

        if hasattr(self, "IWA"):
            base_str += f"\nInner Working Angle: {self.IWA:.2f}"

        if hasattr(self, "OWA"):
            base_str += f"\nOuter Working Angle: {self.OWA:.2f}"

        if self.has_psf_datacube:
            base_str += "\nPSF datacube loaded"

        return base_str

    def _compute_radial_average(self, image, center=None, nbins=None):
        """Compute radial average of a 2D image.

        Args:
            image (numpy.ndarray):
                2D image to compute radial average for
            center (list, optional):
                [x, y] pixel coordinates of center to compute average about.
                If None, use center of image.
            nbins (int, optional):
                Number of radial bins. If None, defaults to floor(max_dimension/2).

        Returns:
            tuple:
                separations_pix (numpy.ndarray):
                    Bin centers in pixel units
                radial_profile (numpy.ndarray):
                    Radial profile values
        """
        # Find the center of the image
        if center is None:
            center_x = (image.shape[1] - 1) / 2
            center_y = (image.shape[0] - 1) / 2

            # Try to get center from header - HeaderData might store this differently
            if hasattr(self.header, "xcenter"):
                center_x = self.header.xcenter
            if hasattr(self.header, "ycenter"):
                center_y = self.header.ycenter

            center = [center_x, center_y]

        # Create distance array
        y, x = np.indices(image.shape)
        r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

        # Define bins for radial averaging
        if nbins is None:
            nbins = int(np.floor(np.max(image.shape) / 2))

        max_radius = np.max(r)
        bins = np.linspace(0, max_radius, nbins + 1)
        bin_centers = (bins[1:] + bins[:-1]) / 2

        # Using the same approach as radialfun.py
        # Digitize the distances and compute means in each bin
        inds = np.digitize(r.ravel(), bins)
        # Max value will be in its own bin, put all matching pixels in the last
        # valid bin
        inds[inds == nbins + 1] = nbins

        # Compute means in each bin
        means = np.zeros(nbins)
        image_flat = image.ravel()
        for j in range(1, nbins + 1):
            means[j - 1] = np.nanmean(image_flat[inds == j])

        return bin_centers, means

    def _plot_performance_curve(
        self, x, y, title, xlabel, ylabel, marker="o-", log_scale=False, ms=4
    ):
        """Helper method to plot performance curves.

        Args:
            x (numpy.ndarray):
                X values (typically separations)
            y (numpy.ndarray):
                Y values (the performance metric)
            title (str):
                Plot title
            xlabel (str):
                X-axis label
            ylabel (str):
                Y-axis label
            marker (str, optional):
                Marker style. Default is 'o-'
            log_scale (bool, optional):
                Whether to use log scale for y-axis. Default is False.
            ms (int, optional):
                Marker size. Default is 4.
        """
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(x, y, marker, ms=ms)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        if log_scale:
            plt.yscale("log")
        plt.grid(True)
        plt.show()

    def compute_all_performance_curves(
        self,
        aperture_radius_lod=0.7,
        stellar_diam=None,
        fit_gaussian_for_core_area=False,
        use_phot_aperture_as_min=False,
        oversample=2,
        save_to_fits=True,
        performance_file="coro_perf.fits",
        load_from_file=None,
        plot=False,
    ):
        """Compute all coronagraph performance curves at once.

        This method computes the throughput, raw contrast, occulter transmission,
        core area, and core mean intensity curves and optionally saves them to files.
        If load_from_file is provided, it will load throughput and contrast from the
        specified file instead of computing them.

        Args:
            aperture_radius_lod (float):
                Aperture radius in lambda/D for throughput and contrast calculations.
            stellar_diam (Quantity, optional):
                Stellar diameter for contrast calculation. If None, uses the first
                available diameter (typically 0.0 * lod).
            fit_gaussian_for_core_area (bool):
                Whether to fit Gaussian for core area calculation.
            use_phot_aperture_as_min (bool):
                Whether to use aperture_radius_lod as minimum area if fitting Gaussian.
            oversample (int):
                Oversampling factor for PSF extraction.
            save_to_fits (bool):
                Whether to save the results to FITS files.
            performance_file (str):
                Filename for saving throughput and contrast.
            load_from_file (str, optional):
                If provided, load throughput and contrast from this file
                instead of computing them.
            plot (bool):
                Whether to plot the performance curves.

        Returns:
            dict:
                Dictionary containing all performance curves.
        """
        # If stellar_diam is None, use the first available diameter
        if stellar_diam is None:
            stellar_diam = self.stellar_intens.diams[0]

        # Check if we should load throughput and contrast from file
        if load_from_file is not None:
            logger.info(f"Loading throughput and contrast from {load_from_file}")
            try:
                sep, throughput, raw_contrast = load_coro_performance_from_fits(
                    load_from_file, self.yip_path
                )
                logger.info(
                    f"Successfully loaded performance data from {load_from_file}"
                )
                # Create splines for throughput and contrast
                self.throughput_interp = make_interp_spline(sep, throughput, k=3)
                self.raw_contrast_interp = make_interp_spline(sep, raw_contrast, k=3)
            except Exception as e:
                logger.warning(f"Error loading from {load_from_file}: {e}")
                logger.info("Computing throughput and contrast from scratch")
                load_from_file = None

        # If not loading from file, compute throughput and contrast
        if load_from_file is None:
            logger.info("Computing all performance metrics...")

            # Compute all off-axis PSF dependent metrics in one pass
            logger.info("Computing throughput, contrast, and core area curves...")
            metrics = self._compute_performance_metrics(
                stellar_diam=stellar_diam,
                aperture_radius_lod=aperture_radius_lod,
                fit_gaussian_for_core_area=fit_gaussian_for_core_area,
                use_phot_aperture_as_min=use_phot_aperture_as_min,
                oversample=oversample,
            )

            sep = metrics["separations"]
            throughput = metrics["throughput"]
            raw_contrast = metrics["raw_contrast"]
            core_area = metrics["core_area"]

            # Create splines for throughput and contrast
            self.throughput_interp = make_interp_spline(sep, throughput, k=3)
            self.raw_contrast_interp = make_interp_spline(sep, raw_contrast, k=3)

            # Save to FITS if requested
            if save_to_fits:
                save_coro_performance_to_fits(
                    sep, throughput, raw_contrast, performance_file, self.yip_path
                )
        else:
            # If loading from file, we still need to compute core area
            logger.info("Computing core area curve...")
            # Use the shared implementation to calculate the core area only
            core_metrics = self._compute_performance_metrics(
                stellar_diam=stellar_diam,
                aperture_radius_lod=aperture_radius_lod,
                fit_gaussian_for_core_area=fit_gaussian_for_core_area,
                use_phot_aperture_as_min=use_phot_aperture_as_min,
                oversample=oversample,
                compute_throughput=False,
                compute_contrast=False,
            )

            sep = core_metrics["separations"]
            core_area = core_metrics["core_area"]

        # Compute occulter transmission (independent of off-axis PSFs)
        logger.info("Computing occulter transmission curve...")
        sep_occ_trans, occ_trans = self.occulter_transmission_curve(plot=plot)

        # Compute core mean intensity (independent of off-axis PSFs)
        logger.info("Computing core mean intensity curve...")
        sep_core_intensity, core_intensities = self.core_mean_intensity_curve(
            stellar_diam_values=None,  # Use all available diameters
            plot=plot,
        )

        # Create remaining spline interpolators
        self.occ_trans_interp = make_interp_spline(sep_occ_trans, occ_trans, k=3)
        self.core_area_interp = make_interp_spline(sep, core_area, k=3)
        self.core_intensity_interp = make_interp_spline(
            sep_core_intensity,
            core_intensities[stellar_diam],  # Use the specified stellar diameter
            k=3,
        )
        # Store the complete intensity dict for potential later use
        self.core_intensity_dict = core_intensities

        # Compute Inner Working Angle
        valid_mask = throughput > 0
        half_max_throughput = max(throughput[valid_mask]) / 2
        _closest_ind = np.searchsorted(throughput[valid_mask], half_max_throughput)
        # Map that to the full throughput array
        closest_ind = np.where(throughput == throughput[valid_mask][_closest_ind])[0]

        def iwa_func(x):
            return self.throughput_interp(x) - half_max_throughput

        self.IWA = (
            root_scalar(iwa_func, bracket=[sep[closest_ind - 1], sep[closest_ind]]).root
            * lod
        )

        # Compute OWA using max_offset_in_image as the physical limit
        if hasattr(self.offax, "max_offset_in_image"):
            self.OWA = self.offax.max_offset_in_image
            logger.info(
                f"OWA set to max_offset_in_image: {self.OWA.to(u.lod).value:.2f} λ/D"
            )
        else:
            # Fallback to maximum separation if max_offset_in_image not available
            self.OWA = np.max(sep) * lod
            logger.warning(
                "max_offset_in_image not available, using maximum separation as OWA"
            )

        # Plot if requested
        if plot:
            self._plot_performance_curve(
                sep,
                throughput,
                title=f"{self.name} Throughput",
                xlabel="Separation [λ/D]",
                ylabel="Throughput",
                ms=6,
            )

            self._plot_performance_curve(
                sep,
                raw_contrast,
                title=f"{self.name} Raw Contrast",
                xlabel="Separation [λ/D]",
                ylabel="Raw Contrast",
                log_scale=True,
            )

            title_suffix = (
                " (Gaussian fit)" if fit_gaussian_for_core_area else " (fixed aperture)"
            )
            self._plot_performance_curve(
                sep,
                core_area,
                title=f"{self.name} Core Area{title_suffix}",
                xlabel="Separation [λ/D]",
                ylabel="Core Area [(λ/D)²]",
            )

        # Return all performance curves
        return {
            "separations": sep,
            "throughput": throughput,
            "raw_contrast": raw_contrast,
            "separations_occ_trans": sep_occ_trans,
            "occ_trans": occ_trans,
            "separations_core_area": sep,  # Same as separations
            "core_area": core_area,
            "separations_core_intensity": sep_core_intensity,
            "core_intensities": core_intensities,
            "IWA": self.IWA,
            "OWA": self.OWA,
        }

    def _compute_performance_metrics(
        self,
        stellar_diam=0.0 * lod,
        aperture_radius_lod=0.7,
        fit_gaussian_for_core_area=False,
        use_phot_aperture_as_min=False,
        oversample=2,
        compute_throughput=True,
        compute_contrast=True,
        compute_core_area=True,
    ):
        """Compute throughput, contrast, and core area metrics in a single pass.

        This internal method processes all off-axis PSF positions once, computing
        multiple metrics at each position to avoid redundant calculations.

        Args:
            stellar_diam (Quantity):
                The stellar diameter used for contrast calculations.
            aperture_radius_lod (float):
                The aperture radius in lambda/D.
            fit_gaussian_for_core_area (bool):
                Whether to fit a 2D Gaussian to determine core area.
            use_phot_aperture_as_min (bool):
                Whether to use aperture_radius_lod as a minimum area if fitting
                Gaussian.
            oversample (int):
                The oversampling factor for interpolation.
            compute_throughput (bool):
                Whether to compute throughput.
            compute_contrast (bool):
                Whether to compute contrast.
            compute_core_area (bool):
                Whether to compute core area.

        Returns:
            dict:
                Dictionary containing calculated metrics.
        """
        # Set up for Gaussian fitting if needed
        if fit_gaussian_for_core_area and compute_core_area:
            from scipy.optimize import curve_fit

            def gaussian_2d(coords, amplitude, x0, y0, sigma_x, sigma_y):
                x, y = coords
                x_term = ((x - x0) / sigma_x) ** 2
                y_term = ((y - y0) / sigma_y) ** 2
                return amplitude * np.exp(-(x_term + y_term) / 2)

        # Get stellar PSF for contrast calculation if needed
        star_psf = None
        if compute_contrast:
            star_psf = self.stellar_intens(stellar_diam)

        # Storage for results
        separations = []
        throughputs = []
        contrasts = []
        core_areas = []

        # Loop through all provided offsets
        for i, x_lod_val in enumerate(np.array(self.offax.x_offsets)):
            for j, y_lod_val in enumerate(np.array(self.offax.y_offsets)):
                # Calculate radial separation
                r = np.sqrt(x_lod_val**2 + y_lod_val**2)

                # Skip separations that exceed the maximum offset where PSF is
                # within image
                if hasattr(self.offax, "max_offset_in_image"):
                    max_sep = self.offax.max_offset_in_image.to(u.lod).value
                    if r > max_sep:
                        logger.debug(
                            f"Skipping separation {r:.2f} λ/D "
                            f"(exceeds max_offset_in_image {max_sep:.2f} λ/D)"
                        )
                        continue

                # Get planet PSF
                planet_psf = self.offax.reshaped_psfs[i, j]

                # Pixel coordinates are needed for all computations
                px = convert_to_pix(
                    x_lod_val, self.offax.center_x, self.pixel_scale
                ).value.astype(int)
                py = convert_to_pix(
                    y_lod_val, self.offax.center_y, self.pixel_scale
                ).value.astype(int)

                # Aperture radius in pixel units
                radius_pix = aperture_radius_lod / self.pixel_scale.value

                # Calculate throughput and contrast (both need same PSF extraction)
                if compute_throughput or compute_contrast:
                    # Extract & oversample for throughput/contrast
                    subarr_oversamp_p, px_os, py_os, radius_os, subarr_orig_p = (
                        extract_and_oversample_subarray(
                            planet_psf, px, py, radius_pix, oversample
                        )
                    )

                    # Measure planet flux in aperture (used for both throughput
                    # and contrast)
                    planet_flux_in_ap = measure_flux_in_oversampled_aperture(
                        subarr_oversamp_p, px_os, py_os, radius_os, subarr_orig_p
                    )

                    # Throughput = planet flux in aperture (PSFs are normalized
                    # to sum to 1)
                    if compute_throughput:
                        throughput = planet_flux_in_ap
                        throughputs.append(throughput)

                    # Calculate contrast using star flux in same aperture
                    if compute_contrast:
                        subarr_oversamp_s, sx_os, sy_os, radius_os_s, subarr_orig_s = (
                            extract_and_oversample_subarray(
                                star_psf, px, py, radius_pix, oversample
                            )
                        )
                        star_flux_in_ap = measure_flux_in_oversampled_aperture(
                            subarr_oversamp_s, sx_os, sy_os, radius_os_s, subarr_orig_s
                        )

                        # Contrast = star flux / planet flux
                        contrast_val = (
                            star_flux_in_ap / planet_flux_in_ap
                            if star_flux_in_ap > 0
                            else 0
                        )
                        contrasts.append(contrast_val)

                # Calculate core area
                if compute_core_area:
                    if fit_gaussian_for_core_area:
                        # Extract a larger subarray for Gaussian fitting
                        subarr_oversamp_g, px_os_g, py_os_g, _, _ = (
                            extract_and_oversample_subarray(
                                planet_psf,
                                px,
                                py,
                                aperture_radius_lod / self.pixel_scale.value * 3,
                                oversample,
                            )
                        )

                        # Create coordinate grids
                        y_grid, x_grid = np.indices(subarr_oversamp_g.shape)

                        # Compute initial guess for Gaussian parameters
                        amplitude = subarr_oversamp_g.max()
                        x0, y0 = px_os_g, py_os_g
                        sigma_x = sigma_y = 1.0 * oversample
                        initial_guess = [amplitude, x0, y0, sigma_x, sigma_y]

                        # Fit 2D Gaussian
                        popt, _ = curve_fit(
                            gaussian_2d,
                            (x_grid, y_grid),
                            subarr_oversamp_g.ravel(),
                            p0=initial_guess,
                        )

                        # Extract sigma values (standard deviations of the
                        # Gaussian fit)
                        _, _, _, sigma_x, sigma_y = popt

                        # Convert from pixel units to lambda/D
                        sigma_x_lod = sigma_x / oversample * self.pixel_scale.value
                        sigma_y_lod = sigma_y / oversample * self.pixel_scale.value

                        # Compute core area from Gaussian parameters (pi *
                        # FWHM_x * FWHM_y / 4)
                        # FWHM = 2.355 * sigma for a Gaussian
                        fwhm_x = 2.355 * sigma_x_lod
                        fwhm_y = 2.355 * sigma_y_lod
                        core_area = np.pi * fwhm_x * fwhm_y / 4

                        # Apply minimum area if requested
                        if use_phot_aperture_as_min:
                            min_area = np.pi * aperture_radius_lod**2
                            core_area = max(core_area, min_area)
                    else:
                        # Fixed core area based on aperture radius
                        core_area = np.pi * aperture_radius_lod**2

                    # Add to results
                    core_areas.append(core_area)

                # Store separation for all calculations
                separations.append(r)

        # Convert to arrays and sort by separation
        separations = np.array(separations)
        result = {"separations": separations}

        # Convert and sort each result array if it was computed
        if compute_throughput:
            throughputs = np.array(throughputs)
            result["throughput"] = throughputs[np.argsort(separations)]

        if compute_contrast:
            contrasts = np.array(contrasts)
            result["raw_contrast"] = contrasts[np.argsort(separations)]

        if compute_core_area:
            core_areas = np.array(core_areas)
            result["core_area"] = core_areas[np.argsort(separations)]

        # Sort the separations themselves last
        result["separations"] = np.sort(separations)

        return result

    def occulter_transmission_curve(self, plot=True):
        """Creates the occulter transmission (sky transmission) curve.

        Compute the radial profile of the sky transmission mask.

        Args:
            plot (bool):
                Whether to plot the occulter transmission curve.

        Returns:
            tuple:
                separations (numpy.ndarray):
                    Separations in lambda/D
                occ_trans_vals (numpy.ndarray):
                    Occulter transmission values at each separation
        """
        # Get sky transmission data
        sky_trans_data = self.sky_trans()

        # Compute radial average
        bin_centers, occ_trans_vals = self._compute_radial_average(sky_trans_data)

        # Convert bin centers from pixels to lambda/D
        separations = bin_centers * self.pixel_scale.value

        if plot:
            self._plot_performance_curve(
                separations,
                occ_trans_vals,
                title=f"{self.name} Occulter Transmission",
                xlabel="Separation [λ/D]",
                ylabel="Occulter Transmission",
            )

        return separations, occ_trans_vals

    def core_mean_intensity_curve(self, stellar_diam_values=None, plot=True):
        """Creates the core mean intensity curves for different stellar diameters.

        Computes the radial profile of the stellar intensity for different stellar
        diameters, providing the core mean intensity at each radius.

        Args:
            stellar_diam_values (list of Quantity, optional):
                List of stellar diameters to compute the core mean intensity for.
                If None, uses the stellar diameters provided in the stellar_diam_file.
            plot (bool):
                Whether to plot the core mean intensity curves.

        Returns:
            tuple:
                separations (numpy.ndarray):
                    Separations in lambda/D
                intensities (dict):
                    Dictionary mapping stellar diameter values to arrays of core mean
                    intensity values at each separation
        """
        # Get all available stellar diameters from StellarIntens
        available_diams = self.stellar_intens.diams

        if stellar_diam_values is None:
            # Use all available diameters
            stellar_diam_values = available_diams
        else:
            # Ensure requested diameters are available
            for diam in stellar_diam_values:
                if diam not in available_diams:
                    raise ValueError(
                        f"Requested stellar diameter {diam} not"
                        " found in available diameters"
                    )

        # Find center of stellar intensity image
        center = [self.stellar_intens.center_x, self.stellar_intens.center_y]

        # Create storage for results
        intensities = {}
        separations = None

        # Process the first diameter to get the bin centers
        stellar_diam = stellar_diam_values[0]
        stellar_psf = self.stellar_intens(stellar_diam)

        # Compute radial average using similar method as in EXOSIMS
        # This ensures we match the original implementation
        dims = stellar_psf.shape
        nbins = int(np.floor(np.max(dims) / 2))

        # Create distance array from center
        y, x = np.indices(stellar_psf.shape)
        r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

        # Define bins and compute bin centers
        max_radius = np.max(r)
        bins = np.linspace(0, max_radius, nbins + 1)
        bin_centers = (bins[1:] + bins[:-1]) / 2

        # Store all profiles
        for stellar_diam in stellar_diam_values:
            # Get the stellar intensity PSF for this diameter
            stellar_psf = self.stellar_intens(stellar_diam)
            stellar_psf_flat = stellar_psf.ravel()
            r_flat = r.ravel()

            # Compute radial average using bins
            radial_profile = np.zeros(nbins)
            for i in range(nbins):
                # Select pixels in the current bin
                mask = (r_flat >= bins[i]) & (r_flat < bins[i + 1])
                if np.any(mask):
                    radial_profile[i] = np.nanmean(stellar_psf_flat[mask])
                else:
                    # If no pixels in this bin, use the previous value
                    radial_profile[i] = radial_profile[i - 1] if i > 0 else 0

            # Store this profile
            intensities[stellar_diam] = radial_profile

        # Convert bin centers from pixels to lambda/D
        separations = bin_centers * self.pixel_scale.value

        if plot:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 6))
            for diam, profile in intensities.items():
                plt.plot(
                    separations, profile, "-", label=f"Diam = {diam.value:.1f} λ/D"
                )

            plt.xlabel("Separation [λ/D]")
            plt.ylabel("Core Mean Intensity")
            plt.title(f"{self.name} Core Mean Intensity")
            plt.yscale("log")
            plt.grid(True)
            plt.legend()
            plt.show()

        return separations, intensities

    def _convert_separation_to_lod(self, separation):
        """Convert separation value(s) to lambda/D units (scalar or array).

        Args:
            separation (float, Quantity, or array-like):
                The separation value(s), either as scalar(s) in lambda/D or Quantity.
                Can be a single value or array-like.

        Returns:
            numpy.ndarray or float:
                The separation value(s) in lambda/D. Returns a float for a single value
                or a numpy array for array-like inputs.

        Raises:
            ValueError:
                If the separation has units that are not lambda/D.
        """
        # Handle Quantity objects with units
        if hasattr(separation, "unit"):
            if separation.unit == lod:
                separation_val = separation.value
            else:
                raise ValueError(
                    f"Separation must be in lambda/D, not {separation.unit}"
                )
        else:
            # Handle non-Quantity values (assumed to be in lambda/D)
            separation_val = separation

        # Convert to numpy array to handle both scalar and array inputs consistently
        return np.atleast_1d(separation_val)

    def throughput(self, separation):
        """Return the throughput at the given separation(s).

        Args:
            separation (float, Quantity, or array-like):
                The separation(s) at which to evaluate the throughput, in lambda/D.
                Can be a single value or array-like.

        Returns:
            float or numpy.ndarray:
                The throughput at the given separation(s). Returns a float for single
                inputs or a numpy array for array-like inputs.
        """
        sep_values = self._convert_separation_to_lod(separation)
        result = self.throughput_interp(sep_values)

        # Return a scalar if the input was a scalar
        if (
            len(sep_values) == 1
            and np.isscalar(separation)
            or (hasattr(separation, "shape") and len(separation.shape) == 0)
        ):
            return float(result[0])
        return result

    def raw_contrast(self, separation):
        """Return the raw contrast at the given separation(s).

        Args:
            separation (float, Quantity, or array-like):
                The separation(s) at which to evaluate the contrast, in lambda/D.
                Can be a single value or array-like.

        Returns:
            float or numpy.ndarray:
                The raw contrast at the given separation(s). Returns a float for single
                inputs or a numpy array for array-like inputs.
        """
        sep_values = self._convert_separation_to_lod(separation)
        result = self.raw_contrast_interp(sep_values)

        # Return a scalar if the input was a scalar
        if (
            len(sep_values) == 1
            and np.isscalar(separation)
            or (hasattr(separation, "shape") and len(separation.shape) == 0)
        ):
            return float(result[0])
        return result

    def occulter_transmission(self, separation):
        """Return the occulter transmission at the given separation(s).

        Args:
            separation (float, Quantity, or array-like):
                The separation(s) at which to evaluate the occulter
                transmission, in lambda/D. Can be a single value or array-like.

        Returns:
            float or numpy.ndarray:
                The occulter transmission at the given separation(s). Returns a
                float for single inputs or a numpy array for array-like inputs.
        """
        sep_values = self._convert_separation_to_lod(separation)
        result = self.occ_trans_interp(sep_values)

        # Return a scalar if the input was a scalar
        if (
            len(sep_values) == 1
            and np.isscalar(separation)
            or (hasattr(separation, "shape") and len(separation.shape) == 0)
        ):
            return float(result[0])
        return result

    def core_area(self, separation):
        """Return the core area at the given separation(s).

        Args:
            separation (float, Quantity, or array-like):
                The separation(s) at which to evaluate the core area, in lambda/D.
                Can be a single value or array-like.

        Returns:
            float or numpy.ndarray:
                The core area at the given separation(s), in (lambda/D)^2.
                Returns a float for single inputs or a numpy array for
                array-like inputs.
        """
        sep_values = self._convert_separation_to_lod(separation)
        result = self.core_area_interp(sep_values)

        # Return a scalar if the input was a scalar
        if (
            len(sep_values) == 1
            and np.isscalar(separation)
            or (hasattr(separation, "shape") and len(separation.shape) == 0)
        ):
            return float(result[0])
        return result

    def core_mean_intensity(self, separation, stellar_diam=0.0 * lod):
        """Returns core mean intensity at the given separation(s) and stellar diameter.

        Args:
            separation (float, Quantity, or array-like):
                The separation(s) at which to evaluate the core intensity, in lambda/D.
                Can be a single value or array-like.
            stellar_diam (Quantity, optional):
                The stellar diameter for which to evaluate the core intensity.
                Currently only 0.0 * lod is supported for interpolation.

        Returns:
            float or numpy.ndarray:
                The core mean intensity at the given separation(s) and stellar diameter.
                Returns a float for single inputs or a numpy array for
                array-like inputs.
        """
        sep_values = self._convert_separation_to_lod(separation)

        if stellar_diam != 0.0 * lod:
            logger.warning(
                "Only stellar_diam=0.0*lod is currently supported for interpolation"
            )

        result = self.core_intensity_interp(sep_values)

        # Return a scalar if the input was a scalar
        if (
            len(sep_values) == 1
            and np.isscalar(separation)
            or (hasattr(separation, "shape") and len(separation.shape) == 0)
        ):
            return float(result[0])
        return result

    def throughput_curve(self, aperture_radius_lod=0.7, oversample=1, plot=True):
        """Creates the coronagraph throughput curve.

        Compute the coronagraph throughput vs. separation using ONLY the
        provided planet off-axis PSFs (no interpolation).
        We define throughput as the fraction of the total flux
        (planet PSF normalized to 1) that lands inside a photometric aperture.

        Args:
            aperture_radius_lod (float):
                The aperture radius in lambda/D.
            oversample (int):
                The oversampling factor for interpolation.
            plot (bool):
                Whether to plot the throughput curve.

        Returns:
            tuple:
                separations (numpy.ndarray):
                    Separations in lambda/D
                throughputs (numpy.ndarray):
                    Throughput values at each separation
        """
        # Use the shared implementation to calculate throughput only
        metrics = self._compute_performance_metrics(
            aperture_radius_lod=aperture_radius_lod,
            oversample=oversample,
            compute_contrast=False,
            compute_core_area=False,
        )

        separations = metrics["separations"]
        throughputs = metrics["throughput"]

        if plot:
            self._plot_performance_curve(
                separations,
                throughputs,
                title=f"{self.name} Throughput",
                xlabel="Separation [λ/D]",
                ylabel="Throughput",
                ms=6,
            )

        return separations, throughputs

    def raw_contrast_curve(
        self, stellar_diam=0 * lod, aperture_radius_lod=0.7, oversample=2, plot=True
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

        Returns:
            tuple:
                separations (numpy.ndarray):
                    Separations in lambda/D
                contrasts (numpy.ndarray):
                    Raw contrast values at each separation
        """
        # Use the shared implementation to calculate contrast only
        metrics = self._compute_performance_metrics(
            stellar_diam=stellar_diam,
            aperture_radius_lod=aperture_radius_lod,
            oversample=oversample,
            compute_throughput=False,
            compute_core_area=False,
        )

        separations = metrics["separations"]
        contrasts = metrics["raw_contrast"]

        if plot:
            self._plot_performance_curve(
                separations,
                contrasts,
                title=f"{self.name} Raw Contrast",
                xlabel="Separation [λ/D]",
                ylabel="Raw Contrast",
                log_scale=True,
            )

        return separations, contrasts

    def core_area_curve(
        self,
        aperture_radius_lod=0.7,
        fit_gaussian=False,
        use_phot_aperture_as_min=False,
        oversample=2,
        plot=True,
    ):
        """Creates the core area curve for the coronagraph.

        The core area represents the effective area of the point spread function core.
        If fit_gaussian is True, it computes the area by fitting a 2D Gaussian to
        the PSF at each separation. Otherwise, it uses a fixed aperture size.

        Args:
            aperture_radius_lod (float):
                The aperture radius in lambda/D. This is used as the fixed core area
                radius if fit_gaussian is False, or as a minimum area if
                use_phot_aperture_as_min is True.
            fit_gaussian (bool):
                Whether to fit a 2D Gaussian to the PSF at each separation to determine
                the core area. Default is False, which uses a fixed aperture radius.
            use_phot_aperture_as_min (bool):
                Whether to use aperture_radius_lod as a minimum area if fit_gaussian
                is True. Only used if fit_gaussian is True.
            oversample (int):
                The oversampling factor for interpolation if fitting Gaussians.
            plot (bool):
                Whether to plot the core area curve.

        Returns:
            tuple:
                separations (numpy.ndarray):
                    Separations in lambda/D
                core_areas (numpy.ndarray):
                    Core areas at each separation in (lambda/D)^2
        """
        # Use the shared implementation to calculate core area only
        metrics = self._compute_performance_metrics(
            aperture_radius_lod=aperture_radius_lod,
            fit_gaussian_for_core_area=fit_gaussian,
            use_phot_aperture_as_min=use_phot_aperture_as_min,
            oversample=oversample,
            compute_throughput=False,
            compute_contrast=False,
        )

        separations = metrics["separations"]
        core_areas = metrics["core_area"]

        if plot:
            title_suffix = " (Gaussian fit)" if fit_gaussian else " (fixed aperture)"
            self._plot_performance_curve(
                separations,
                core_areas,
                title=f"{self.name} Core Area{title_suffix}",
                xlabel="Separation [λ/D]",
                ylabel="Core Area [(λ/D)²]",
            )

        return separations, core_areas

    def _save_to_exosims_format(
        self,
        sep,
        throughput,
        raw_contrast,
        core_area,
        sep_occ_trans,
        occ_trans,
        sep_core_intensity,
        core_intensities,
        aperture_radius_lod,
        fit_gaussian_for_core_area,
        use_phot_aperture_as_min,
        units,
    ):
        """Save performance curves in EXOSIMS format.

        Args:
            sep (numpy.ndarray):
                Separations in lambda/D for throughput/contrast/core_area
            throughput (numpy.ndarray):
                Throughput values at each separation
            raw_contrast (numpy.ndarray):
                Raw contrast values at each separation
            core_area (numpy.ndarray):
                Core areas at each separation in (lambda/D)^2
            sep_occ_trans (numpy.ndarray):
                Separations in lambda/D for occulter transmission
            occ_trans (numpy.ndarray):
                Occulter transmission values at each separation
            sep_core_intensity (numpy.ndarray):
                Separations in lambda/D for core mean intensity
            core_intensities (dict):
                Dictionary mapping stellar diameter values to arrays of core mean
                intensity values at each separation
            aperture_radius_lod (float):
                Aperture radius in lambda/D for throughput and contrast calculations
            fit_gaussian_for_core_area (bool):
                Whether Gaussian fitting was used for core area calculation
            use_phot_aperture_as_min (bool):
                Whether aperture_radius_lod was used as minimum area if fitting Gaussian
            units (str):
                Units for the EXOSIMS files
        """
        # Create EXOSIMS subdirectory if it doesn't exist
        exosims_dir = Path(self.yip_path, "exosims")
        exosims_dir.mkdir(exist_ok=True)

        # Create base header from coronagraph parameters
        base_header = pyfits.Header()
        base_header["PIXSCALE"] = (self.pixel_scale.value, "Angular pixel scale")
        base_header["LAMBDA"] = (self.header.lambda0.value, "Wavelength in micrometers")
        base_header["D"] = (self.header.diameter.value, "Telescope diameter in meters")
        base_header["OBSCURED"] = (self.header.obscured, "Obscuration fraction")
        if self.header.maxlam is not None and self.header.minlam is not None:
            base_header["DELTALAM"] = (
                (self.header.maxlam - self.header.minlam).value,
                "Bandpass width in micrometers",
            )
        base_header["UNITS"] = (units, "Angular units")

        # 1. Save occulter transmission (occ_trans.fits format)
        occ_trans_header = base_header.copy()
        occ_trans_data = np.vstack((sep_occ_trans, occ_trans)).transpose()
        hdul = pyfits.HDUList(
            [pyfits.PrimaryHDU(occ_trans_data, header=occ_trans_header)]
        )
        occ_trans_file = exosims_dir / "occ_trans.fits"
        hdul.writeto(occ_trans_file, overwrite=True)

        # 2. Save core throughput (core_thruput.fits format)
        core_thruput_header = base_header.copy()
        if fit_gaussian_for_core_area:
            core_thruput_header["PHOTAPER"] = "Gaussian"
            if use_phot_aperture_as_min:
                core_thruput_header["MINAPER"] = aperture_radius_lod
            else:
                core_thruput_header["MINAPER"] = 0
        else:
            core_thruput_header["PHOTAPER"] = aperture_radius_lod

        core_thruput_data = np.vstack((sep, throughput)).transpose()
        hdul = pyfits.HDUList(
            [pyfits.PrimaryHDU(core_thruput_data, header=core_thruput_header)]
        )
        core_thruput_file = exosims_dir / "core_thruput.fits"
        hdul.writeto(core_thruput_file, overwrite=True)

        # 3. Save core area (core_area.fits format) - only if Gaussian fitting was used
        # For fixed aperture, this would be a scalar value like in
        # process_opticalsys_package.py
        if fit_gaussian_for_core_area:
            core_area_header = core_thruput_header.copy()  # Same header as throughput
            core_area_data = np.vstack((sep, core_area)).transpose()
            hdul = pyfits.HDUList(
                [pyfits.PrimaryHDU(core_area_data, header=core_area_header)]
            )
            core_area_file = exosims_dir / "core_area.fits"
            hdul.writeto(core_area_file, overwrite=True)
        else:
            # For fixed aperture, the core area is just the scalar value
            logger.info(
                f"Fixed aperture core area: {aperture_radius_lod**2 * np.pi:.6f} (λ/D)²"
            )

        # 4. Save core mean intensity (core_mean_intensity.fits format)
        core_intensity_header = base_header.copy()

        # Add stellar diameter information to header (DIAM000, DIAM001, etc.)
        stellar_diams = list(core_intensities.keys())
        for j, diam in enumerate(stellar_diams):
            core_intensity_header[f"DIAM{j:03d}"] = (
                diam.value,
                f"Stellar diameter {j} in lambda/D",
            )

        # Stack all intensity profiles with separations as first row
        # Format: first row is separations, subsequent rows are intensities for each
        # diameter
        intensity_array = np.zeros((len(stellar_diams), len(sep_core_intensity)))
        for j, diam in enumerate(stellar_diams):
            intensity_array[j] = core_intensities[diam]

        # Stack separations and intensities:
        # [separations, intens_diam0, intens_diam1, ...]
        core_intensity_data = np.vstack(
            (sep_core_intensity, intensity_array)
        ).transpose()

        hdul = pyfits.HDUList(
            [pyfits.PrimaryHDU(core_intensity_data, header=core_intensity_header)]
        )
        core_intensity_file = exosims_dir / "core_mean_intensity.fits"
        hdul.writeto(core_intensity_file, overwrite=True)

        # 5. Save raw contrast for completeness (not part of original EXOSIMS format
        # but useful for analysis)
        contrast_header = core_thruput_header.copy()  # Same parameters as throughput
        contrast_data = np.vstack((sep, raw_contrast)).transpose()
        hdul = pyfits.HDUList(
            [pyfits.PrimaryHDU(contrast_data, header=contrast_header)]
        )
        contrast_file = exosims_dir / "raw_contrast.fits"
        hdul.writeto(contrast_file, overwrite=True)

        logger.info(f"EXOSIMS format files saved to {exosims_dir}/")
        logger.info("Files created:")
        logger.info("  - occ_trans.fits (occulter transmission)")
        logger.info("  - core_thruput.fits (throughput)")
        if fit_gaussian_for_core_area:
            logger.info("  - core_area.fits (core area from Gaussian fits)")
        logger.info("  - core_mean_intensity.fits (stellar intensity)")
        logger.info("  - raw_contrast.fits (raw contrast)")
        logger.info("  - specs.json (EXOSIMS specification file)")

    def to_exosims(
        self,
        aperture_radius_lod=0.7,
        fit_gaussian_for_core_area=False,
        use_phot_aperture_as_min=False,
        units="LAMBDA/D",
    ):
        """Save performance curves in EXOSIMS format.

        This method saves the coronagraph's performance curves (throughput, contrast,
        occulter transmission, core area, and core mean intensity) to individual FITS
        files in the same format as used by process_opticalsys_package.py for use
        with EXOSIMS.

        Args:
            aperture_radius_lod (float):
                Aperture radius in lambda/D used for throughput and contrast
                calculations. Default is 0.7.
            fit_gaussian_for_core_area (bool):
                Whether Gaussian fitting was used for core area calculation.
                Default is False (fixed aperture).
            use_phot_aperture_as_min (bool):
                Whether aperture_radius_lod was used as minimum area if fitting
                Gaussian. Default is False.
            units (str):
                Units for the angular separations in EXOSIMS files.
                Default is "LAMBDA/D".

        Raises:
            ValueError:
                If performance curves have not been computed yet.
        """
        # Check that performance curves have been computed
        required_attrs = [
            "throughput_interp",
            "raw_contrast_interp",
            "occ_trans_interp",
            "core_area_interp",
            "core_intensity_interp",
        ]

        missing_attrs = [attr for attr in required_attrs if not hasattr(self, attr)]
        if missing_attrs:
            raise ValueError(
                f"Performance curves not computed yet. Missing: {missing_attrs}. "
                "Call compute_all_performance_curves() first."
            )

        # Get the separations from the off-axis PSF data
        separations = []
        for i, x_lod_val in enumerate(np.array(self.offax.x_offsets)):
            for j, y_lod_val in enumerate(np.array(self.offax.y_offsets)):
                r = np.sqrt(x_lod_val**2 + y_lod_val**2)

                # Skip separations that exceed the maximum offset where PSF is
                # within image
                if hasattr(self.offax, "max_offset_in_image"):
                    max_sep = self.offax.max_offset_in_image.to(u.lod).value
                    if r > max_sep:
                        logger.debug(
                            f"Skipping separation {r:.2f} λ/D "
                            f"(exceeds max_offset_in_image {max_sep:.2f} λ/D)"
                        )
                        continue

                separations.append(r)

        separations = np.sort(np.unique(separations))

        # Log filtering results
        if hasattr(self.offax, "max_offset_in_image"):
            max_sep = self.offax.max_offset_in_image.to(u.lod).value
            logger.info(
                f"Using {len(separations)} separations within max_offset_in_image "
                f"({max_sep:.2f} λ/D)"
            )
            logger.info(
                f"Separation range: {np.min(separations):.2f} - "
                f"{np.max(separations):.2f} λ/D"
            )
        else:
            logger.info(
                f"Using all {len(separations)} separations "
                "(no max_offset_in_image filtering)"
            )

        # Evaluate interpolators at these separations
        throughput = self.throughput_interp(separations)
        raw_contrast = self.raw_contrast_interp(separations)
        core_area = self.core_area_interp(separations)

        # Validate results to ensure no extrapolation artifacts
        if np.any(throughput > 1):
            raise ValueError(
                f"Found {np.sum(throughput > 1)}"
                f" throughput values > 1. Max value: {np.max(throughput):.3f}"
            )
        if np.any(throughput < 0):
            raise ValueError(
                f"Found {np.sum(throughput < 0)}"
                f" negative throughput values. Min value: {np.min(throughput):.3f}"
            )

        # Get occulter transmission data
        # Use the stored data from the SkyTrans object
        sky_trans_data = self.sky_trans()
        bin_centers, occ_trans = self._compute_radial_average(sky_trans_data)
        sep_occ_trans = bin_centers * self.pixel_scale.value

        # Get core mean intensity data
        # Use the stored stellar intensity dict
        if hasattr(self, "core_intensity_dict"):
            core_intensities = self.core_intensity_dict
        else:
            # Fallback: compute it on the fly
            _, core_intensities = self.core_mean_intensity_curve(plot=False)

        # Get separations for core intensity (first diameter's data)
        first_diam = list(core_intensities.keys())[0]
        stellar_psf = self.stellar_intens(first_diam)
        dims = stellar_psf.shape
        nbins = int(np.floor(np.max(dims) / 2))
        center = [self.stellar_intens.center_x, self.stellar_intens.center_y]

        # Create distance array from center
        y, x = np.indices(stellar_psf.shape)
        r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        max_radius = np.max(r)
        bins = np.linspace(0, max_radius, nbins + 1)
        bin_centers = (bins[1:] + bins[:-1]) / 2
        sep_core_intensity = bin_centers * self.pixel_scale.value

        # Call the existing save method
        self._save_to_exosims_format(
            sep=separations,
            throughput=throughput,
            raw_contrast=raw_contrast,
            core_area=core_area,
            sep_occ_trans=sep_occ_trans,
            occ_trans=occ_trans,
            sep_core_intensity=sep_core_intensity,
            core_intensities=core_intensities,
            aperture_radius_lod=aperture_radius_lod,
            fit_gaussian_for_core_area=fit_gaussian_for_core_area,
            use_phot_aperture_as_min=use_phot_aperture_as_min,
            units=units,
        )

        # Use the stored IWA and OWA values that were computed in
        # compute_all_performance_curves
        IWA = self.IWA
        OWA = self.OWA

        # Convert units if needed (to arcseconds if requested)
        to_arcsec = units.lower() == "arcsec"
        if to_arcsec:
            # Convert lambda/D to arcseconds using the telescope diameter and wavelength
            angunit = ((self.header.lambda0) / (self.header.diameter)).to(
                u.arcsec, equivalencies=u.dimensionless_angles()
            )
            IWA_output = (IWA * angunit).to(u.arcsec).value
            OWA_output = (OWA * angunit).to(u.arcsec).value
        else:
            IWA_output = IWA.to_value(u.lod)
            OWA_output = OWA.to_value(u.lod)

        # Determine core area filename/value
        if fit_gaussian_for_core_area:
            core_area_fname = "core_area.fits"
        else:
            # For fixed aperture, it's just the scalar value
            core_area_fname = aperture_radius_lod**2 * np.pi

        # Calculate deltaLam
        if self.header.maxlam is not None and self.header.minlam is not None:
            deltaLam = (self.header.maxlam - self.header.minlam).to(u.nm).value
        else:
            deltaLam = None

        # Create the EXOSIMS specs dictionary in the same format as
        # process_opticalsys_package.py
        outdict = {
            "pupilDiam": self.header.diameter.to(u.m).value,
            "obscurFac": self.header.obscured,
            "starlightSuppressionSystems": [
                {
                    "name": self.name,
                    "lam": self.header.lambda0.to(u.nm).value,
                    "deltaLam": deltaLam,
                    "occ_trans": "occ_trans.fits",
                    "core_thruput": "core_thruput.fits",
                    "core_mean_intensity": "core_mean_intensity.fits",
                    "core_area": core_area_fname,
                    "IWA": IWA_output,
                    "OWA": OWA_output,
                    "input_angle_units": units,
                }
            ],
        }

        # Save the JSON specs file
        exosims_dir = Path(self.yip_path, "exosims")
        specs_file = exosims_dir / "specs.json"
        with open(specs_file, "w") as f:
            json.dump(outdict, f, indent=2)

        logger.info(f"EXOSIMS specs saved to {specs_file}")
        return outdict
