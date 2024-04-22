"""Base class for all offax_psfs.fits files."""

from pathlib import Path

import astropy.io.fits as pyfits
import astropy.units as u
import numpy as np
from astropy.units import Quantity
from lod_unit import lod
from numpy.typing import NDArray

from yippy.offax_psf import OneD, QuarterSymmetric, TwoD
from yippy.util import convert_to_lod

from .logger import logger


class OffAx:
    """Base class for handling off-axis PSFs.

    This class loads and processes PSF data from the yield input package (YIP).
    It currently supports oneD and quater symmetric PSF YIPs. The primary use
    is to interpolate the PSF data to a given x/y position. This is done by
    calling the OffAx object with the x/y position as arguments, which itself
    calls the psf object after converting units.

    Attributes:
        pixel_scale (Quantity):
            Pixel scale of the PSF data in lambda/D.
        center_x (Quantity):
            Central x position in the PSF data.
        center_y (Quantity):
            Central y position in the PSF data.
        psf:
            Instance of the appropriate PSF class (e.g., OneD, TwoD) based on input YIP.

    Args:
        yip_dir (Path):
            Path to the directory containing PSF and offset data.
        logger (Logger):
            Logger for logging events and information.
        offax_data_file (str):
            Name of the file containing the PSF data.
        offax_offsets_file (str):
            Name of the file containing the offsets data.
        pixel_scale (Quantity):
            Pixel scale of the PSF data in lambda/D.
    """

    def __init__(
        self,
        yip_dir: Path,
        offax_data_file: str,
        offax_offsets_file: str,
        pixel_scale: Quantity,
    ) -> None:
        """Initializes the OffAx class by loading PSF and offset data from YIP.

        Determines the type of coronagraph based on the symmetry and structure of the
        offsets and chooses the correct PSF class (OneD, QuarterSymmetric) accordingly.
        """
        # Pixel scale in lambda/D
        self.pixel_scale = pixel_scale

        # Load off-axis PSF data (e.g. the planet) (unitless intensity maps)
        psfs = pyfits.getdata(Path(yip_dir, offax_data_file), 0)

        # Save the center of the pixel array, which is used for converting to
        # lambda/D when the x/y positions are in pixels.
        self.center_x = psfs.shape[1] / 2 * u.pix
        self.center_y = psfs.shape[2] / 2 * u.pix

        # Load the offset list, which is in units of lambda/D
        offsets = pyfits.getdata(Path(yip_dir, offax_offsets_file), 0) * lod

        # Check whether offsets are given as 1D or 2D
        one_d_offsets = len(offsets.shape) == 1
        if one_d_offsets:
            # Add a second dimension if the offsets are 1D
            offsets = np.vstack((offsets, np.zeros_like(offsets)))

        if len(offsets.shape) > 1:
            if (offsets.shape[1] != 2) and (offsets.shape[0] == 2):
                # This condition occurs when the offsets is transposed
                # from the expected format
                offsets = offsets.T
        assert (
            len(offsets) == psfs.shape[0]
        ), "Offsets and PSFs do not have the same number of elements"

        ########################################################################
        # Determine the format of the input coronagraph files so we can handle #
        # the coronagraph correctly (e.g. radially symmetric in x direction)   #
        ########################################################################

        # Get the unique values of the offset list so that we can format the
        # data into
        offsets_x = np.unique(offsets[:, 0])
        offsets_y = np.unique(offsets[:, 1])

        if len(offsets_x) == 1:
            logger.info("Coronagraph is radially symmetric")
            type = "1d"
            # Instead of handling angles for 1dy, swap the x and y
            offsets_x, offsets_y = (offsets_y, offsets_x)
            raise NotImplementedError(
                (
                    "Verify that the PSFs are correct for this case!"
                    " I don't have a test file for this case yet but I think they"
                    " probably need to be rotated by 90 degrees."
                )
            )
        elif len(offsets_y) == 1:
            logger.info("Coronagraph is radially symmetric")
            type = "1d"
        elif np.min(offsets) >= 0 * lod:
            logger.info("Coronagraph is quarterly symmetric")
            type = "2dq"
        else:
            logger.info("Coronagraph response is full 2D")
            type = "2df"

        # interpolate planet data depending on type
        if "1" in type:
            # Always set up to interpolate along the x axis
            self.psf = OneD(psfs, offsets_x)
        elif type == "2dq":
            self.psf = QuarterSymmetric(psfs, offsets)
        elif type == "2df":
            self.psf = TwoD(psfs, offsets_x, offsets_y)

    def __call__(
        self, x: Quantity, y: Quantity, lam=None, D=None, dist=None
    ) -> NDArray:
        """Return the PSF at the given x/y position.

        This function (via util.convert_to_lod) has the following assumptions
        on the x/y values provided:
            - If units are pixels, they follow the 00LL convention. As in the
              (0,0) point is the lower left corner of the image.
            - If the x/y values are in lambda/D, angular, or length units the
              (0,0) point is the center of the image, where the star is
              (hopefully) located.

        Args:
            x (astropy.units.Quantity):
                x position. Can be either units of pixel, lod, an angular
                unit (e.g. arcsec), or a length unit (e.g. AU)
            y (astropy.units.Quantity):
                y position. Can be either units of pixel, lod, an angular
                unit (e.g. arcsec), or a length unit (e.g. AU)
            lam (astropy.units.Quantity):
                Wavelength of the observation
            D (astropy.units.Quantity):
                Diameter of the telescope
            dist (astropy.units.Quantity):
                Distance to the system

        Returns:
            NDArray:
                The PSF at the given x/y position
        """
        # Convert the x and y positions to lambda/D if they are in pixels
        if x.unit != lod:
            x = convert_to_lod(x, self.center_x, self.pixel_scale, lam, D, dist)
        if y.unit != lod:
            y = convert_to_lod(y, self.center_y, self.pixel_scale, lam, D, dist)

        return self.psf(x, y)
