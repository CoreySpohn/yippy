"""Base coronagraph class."""

from itertools import product
from pathlib import Path

import astropy.io.fits as pyfits
import astropy.units as u
import numpy as np
import xarray as xr
from tqdm import tqdm

from .header import HeaderData
from .logger import logger
from .offax import OffAx
from .offjax import OffJAX
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
        use_jax: bool = False,
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
            use_jax (bool):
                Whether to use JAX for optimized computation. Default is False.
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
        self.yip_path = yip_path

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
        if not use_jax:
            self.offax = OffAx(
                yip_path, offax_data_file, offax_offsets_file, self.pixel_scale
            )
        else:
            self.offax = OffJAX(
                yip_path, offax_data_file, offax_offsets_file, self.pixel_scale
            )

        # Get the sky_trans mask
        self.sky_trans = SkyTrans(yip_path, sky_trans_file)

        # PSF datacube here is a 4D array of PSFs at each pixel (x psf offset,
        # y psf offset, x, y). Given the computational cost of generating this
        # datacube, it is only generated when needed.
        self.has_psf_datacube = False
        logger.info(f"Created {yip_path.stem}")

        # Shape of the images in the PSFs
        self.psf_shape = np.array([self.header.naxis1, self.header.naxis2])
        assert self.psf_shape[0] == self.psf_shape[1], "PSF must be square"
        self.npixels = self.psf_shape[0]
        logger.info(f"Created {yip_path.stem}")

    def create_offax_datacube(self):
        """Load the disk image from a file or generate it if it doesn't exist."""
        # Load data cube of spatially dependent PSFs.
        path = self.yip_path / "offax_datacube.nc"

        coords = {
            "x psf offset (pix)": np.arange(self.psf_shape[0]),
            "y psf offset (pix)": np.arange(self.psf_shape[1]),
            "x (pix)": np.arange(self.psf_shape[0]),
            "y (pix)": np.arange(self.psf_shape[1]),
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
            # Compute pixel grid contrast.
            psfs_shape = (*self.psf_shape, *self.psf_shape)
            psfs = np.zeros(psfs_shape, dtype=np.float32)
            pixel_lod = (
                (np.arange(self.npixels) - ((self.npixels - 1) // 2))
                * u.pixel
                * self.pixel_scale
            ).value

            # x_lod, y_lod = np.meshgrid(pixel_lod, pixel_lod, indexing="xy")
            npsfs = np.prod(self.psf_shape)
            pb = tqdm(total=npsfs, desc="Computing datacube of PSFs at every pixel")

            # Note: intention is that i value maps to x offset and j value maps
            # to y offset

            pix_inds = np.arange(self.npixels)

            for (i, x), (j, y) in product(
                zip(pix_inds, pixel_lod), zip(pix_inds, pixel_lod)
            ):
                psfs[i, j] = self.offax(x, y)
                pb.update(1)

            # # Calculate PSFs only for the first quadrant (where x >= 0 and y >= 0)
            # half_size = len(pix_inds) // 2
            # pbar = tqdm(
            #     total=half_size**2,
            #     desc="Computing datacube of PSFs at every pixel",
            #     delay=0.5,
            # )
            #
            # for i in range(half_size, len(pix_inds)):
            #     for j in range(half_size, len(pix_inds)):
            #         psfs[i, j] = self.offax(pixel_lod[i], pixel_lod[j])
            #         pbar.update(1)
            #
            # # Mirror the first quadrant to the other quadrants
            # # Mirror over y-axis (flip left to right)
            # psfs[half_size:, :half_size] = np.fliplr(psfs[half_size:, half_size:])
            #
            # # Mirror over x-axis (flip up to down)
            # psfs[:half_size, half_size:] = np.flipud(psfs[half_size:, half_size:])
            #
            # # Mirror over both axes (flip both horizontally and vertically)
            # psfs[:half_size, :half_size] = np.flipud(
            #     np.fliplr(psfs[half_size:, half_size:])
            # )

            # Save data cube of spatially dependent PSFs.
            psfs_xr = xr.DataArray(
                psfs,
                coords=coords,
                dims=dims,
            )
            psfs_xr.to_netcdf(path)
        self.has_psf_datacube = True
        self.psf_datacube = np.ascontiguousarray(psfs_xr)
        breakpoint()
