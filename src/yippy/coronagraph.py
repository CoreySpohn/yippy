"""Base coronagraph class."""

from pathlib import Path

import astropy.io.fits as pyfits
import astropy.units as u
import numpy as np
import xarray as xr
from scipy.ndimage import rotate
from tqdm import tqdm

from .header import HeaderData
from .logger import logger
from .offax_base import OffAx
from .sky_trans import SkyTrans
from .stellar_intens import StellarIntens


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
        stellar_intens_file: str = "stellar_intens.fits",
        stellar_diam_file: str = "stellar_intens_diam_list.fits",
        offax_data_file: str = "offax_psf.fits",
        offax_offsets_file: str = "offax_psf_offset_list.fits",
        sky_trans_file: str = "sky_trans.fits",
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
        """
        ###################
        # Read input data #
        ###################
        yip_path = Path(yip_path)

        logger.info(f"Creating {yip_path.stem} coronagraph")

        self.name = yip_path.stem
        # Get header and calculate the lambda/D value
        stellar_intens_header = pyfits.getheader(Path(yip_path, stellar_intens_file), 0)

        # Get pixel scale with units
        self.header = HeaderData.from_fits_header(stellar_intens_header)
        self.pixel_scale = self.header.pixscale

        # Stellar intensity of the star being observed as function of stellar
        # angular diameter (unitless)
        self.stellar_intens = StellarIntens(
            yip_path, stellar_intens_file, stellar_diam_file
        )

        # Offaxis PSF of the planet as function of separation from the star
        self.offax = OffAx(
            yip_path, offax_data_file, offax_offsets_file, self.pixel_scale
        )

        # Get the sky_trans mask
        self.sky_trans = SkyTrans(yip_path, sky_trans_file)

        # PSF datacube here is a 4D array of PSFs at each pixel (x psf offset,
        # y psf offset, x, y). Given the computational cost of generating this
        # datacube, it is only generated when needed.
        self.has_psf_datacube = False
        logger.info(f"Created {yip_path.stem}")

    def get_disk_psfs(self):
        """Load the disk image from a file or generate it if it doesn't exist."""
        # Load data cube of spatially dependent PSFs.
        disk_dir = Path(".cache/disks/")
        if not disk_dir.exists():
            disk_dir.mkdir(parents=True, exist_ok=True)
        path = Path(
            disk_dir,
            self.yip_path.name + ".nc",
        )

        coords = {
            "x psf offset (pix)": np.arange(self.npixels),
            "y psf offset (pix)": np.arange(self.npixels),
            "x (pix)": np.arange(self.npixels),
            "y (pix)": np.arange(self.npixels),
        }
        dims = ["x psf offset (pix)", "y psf offset (pix)", "x (pix)", "y (pix)"]
        if path.exists():
            logger.info("Loading data cube of spatially dependent PSFs, please hold...")
            psfs_xr = xr.open_dataarray(path)
        else:
            logger.info(
                "Calculating data cube of spatially dependent PSFs, please hold..."
            )
            # Compute pixel grid.
            # lambda/D
            pixel_lod = (
                (np.arange(self.npixels) - ((self.npixels - 1) // 2))
                * u.pixel
                * self.pixel_scale
            )

            x_lod, y_lod = np.meshgrid(pixel_lod, pixel_lod, indexing="xy")

            # lambda/D
            pixel_dist_lod = np.sqrt(x_lod**2 + y_lod**2)

            # deg
            pixel_angle = np.arctan2(y_lod, x_lod)

            # Compute pixel grid contrast.
            psfs_shape = (
                pixel_dist_lod.shape[0],
                pixel_dist_lod.shape[1],
                self.npixels,
                self.npixels,
            )
            psfs = np.zeros(psfs_shape, dtype=np.float32)
            npsfs = np.prod(pixel_dist_lod.shape)

            pbar = tqdm(
                total=npsfs, desc="Computing datacube of PSFs at every pixel", delay=0.5
            )

            radially_symmetric_psf = "1d" in self.type
            # Get the PSF (npixel, npixel) of a source at every pixel

            # Note: intention is that i value maps to x offset and j value maps
            # to y offset
            for i in range(pixel_dist_lod.shape[0]):
                for j in range(pixel_dist_lod.shape[1]):
                    # Basic structure here is to get the distance in lambda/D,
                    # determine whether the psf has to be rotated (if the
                    # coronagraph is defined in 1 dimension), evaluate
                    # the offaxis psf at the distance, then rotate the
                    # image
                    if self.type == "1d":
                        psf_eval_dists = pixel_dist_lod[i, j]
                        rotate_angle = pixel_angle[i, j]
                    elif self.type == "1dno0":
                        psf_eval_dists = np.sqrt(
                            pixel_dist_lod[i, j] ** 2 - self.offax_psf_offset_x[0] ** 2
                        )
                        rotate_angle = pixel_angle[i, j] + np.arcsin(
                            self.offax_psf_offset_x[0] / pixel_dist_lod[i, j]
                        )
                    elif self.type == "2dq":
                        # lambda/D
                        temp = np.array([y_lod[i, j], x_lod[i, j]])
                        psf = self.offax_psf_interp(np.abs(temp))[0]
                        if y_lod[i, j] < 0.0:
                            # lambda/D
                            psf = psf[::-1, :]
                        if x_lod[i, j] < 0.0:
                            # lambda/D
                            psf = psf[:, ::-1]
                    else:
                        # lambda/D
                        temp = np.array([y_lod[i, j], x_lod[i, j]])
                        psf = self.offax_psf_interp(temp)[0]

                    if radially_symmetric_psf:
                        psf = self.ln_offax_psf_interp(psf_eval_dists)
                        temp = np.exp(
                            rotate(
                                psf,
                                -rotate_angle.to(u.deg).value,
                                reshape=False,
                                mode="nearest",
                                order=5,
                            )
                        )
                    psfs[i, j] = temp
                    pbar.update(1)

            # Save data cube of spatially dependent PSFs.
            psfs_xr = xr.DataArray(
                psfs,
                coords=coords,
                dims=dims,
            )
            psfs_xr.to_netcdf(path)
        self.has_psf_datacube = True
        self.psf_datacube = np.ascontiguousarray(psfs_xr)
