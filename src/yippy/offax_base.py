"""Base class for all offax_psfs.fits files."""

from pathlib import Path
from typing import TYPE_CHECKING

import astropy.io.fits as pyfits
import astropy.units as u
import numpy as np
from astropy.units import Quantity
from lod_unit import lod
from numpy.typing import NDArray

from yippy.offax_psf import OneD
from yippy.util import convert_to_lod

if TYPE_CHECKING:
    from logging import Logger


class OffAx:
    """Base class for all off-axis classes."""

    def __init__(
        self,
        yip_dir: Path,
        logger: "Logger",
        offax_data_file: str,
        offax_offsets_file: str,
        pixel_scale: Quantity,
    ) -> None:
        """Initialize the OffAx class."""
        # Pixel scale in lambda/D
        self.pixel_scale = pixel_scale

        # Load off-axis PSF data (e.g. the planet) (unitless intensity maps)
        psfs = pyfits.getdata(Path(yip_dir, offax_data_file), 0)

        # The offset list here is in units of lambda/D
        offsets = pyfits.getdata(Path(yip_dir, offax_offsets_file), 0) * lod
        # Check whether offsets is 1D or 2D
        one_d_offsets = len(offsets.shape) == 1
        # Add a second dimension if the offsets are 1D
        if one_d_offsets:
            offsets = np.vstack((offsets, np.zeros_like(offsets)))

        if len(offsets.shape) > 1:
            if (offsets.shape[1] != 2) and (offsets.shape[0] == 2):
                # This condition occurs when the offsets is transposed
                # from the expected format for radially symmetric coronagraphs
                offsets = offsets.T
        assert (
            len(offsets) == psfs.shape[0]
        ), "Offsets and PSFs do not have the same number of elements"

        self.center_x = psfs.shape[1] / 2 * u.pix
        self.center_y = psfs.shape[2] / 2 * u.pix

        ########################################################################
        # Determine the format of the input coronagraph files so we can handle #
        # the coronagraph correctly (e.g. radially symmetric in x direction)   #
        ########################################################################

        # Check that we have both x and y offset information (even if there
        # is only one axis with multiple values)

        # Get the unique values of the offset list so that we can format the
        # data into
        offsets_x = np.unique(offsets[:, 0])
        offsets_y = np.unique(offsets[:, 1])

        if (len(offsets_x) == 1) and (offsets_x[0] == 0 * lod):
            logger.info("Coronagraph is radially symmetric")
            type = "1d"
            # Instead of handling angles for 1dy, swap the x and y
            offsets_x, offsets_y = (offsets_y, offsets_x)
        elif (len(offsets_y) == 1) and (offsets_y[0] == 0 * lod):
            logger.info("Coronagraph is radially symmetric")
            type = "1d"
        elif len(offsets_x) == 1:
            # 1 dimensional with offset (e.g. no offset=0)
            logger.info("Coronagraph is radially symmetric")
            type = "1dno0"
            offsets_x, offsets_y = (offsets_y, offsets_x)
        elif len(offsets_y) == 1:
            logger.info("Coronagraph is radially symmetric")
            type = "1dno0"
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
        else:
            pass
            # zz_temp = psfs.reshape(
            #     offsets_x.shape[0],
            #     offsets_y.shape[0],
            #     psfs.shape[1],
            #     psfs.shape[2],
            # )
        #     if type == "2dq":
        #         # Reflect PSFs to cover the x = 0 and y = 0 axes.
        #         offsets_x = np.append(
        #             -offsets_x[0], offsets_x
        #         )
        #         offsets_y = np.append(
        #             -offsets_y[0], offsets_y
        #         )
        #         zz = np.pad(zz_temp, ((1, 0), (1, 0), (0, 0), (0, 0)))
        #         zz[0, 1:] = zz_temp[0, :, ::-1, :]
        #         zz[1:, 0] = zz_temp[:, 0, :, ::-1]
        #         zz[0, 0] = zz_temp[0, 0, ::-1, ::-1]
        #
        #         ln_offax_psf_interp = RegularGridInterpolator(
        #             (offsets_x, offsets_y),
        #             np.log(zz),
        #             method="linear",
        #             bounds_error=False,
        #             fill_value=fill,
        #         )
        #     else:
        #         # This section included references to non-class attributes for
        #         # offsets_x and offsets_y. I think it meant
        #         # to be the class attributes
        #         ln_offax_psf_interp = RegularGridInterpolator(
        #             (offsets_x, offsets_y),
        #             np.log(zz_temp),
        #             method="linear",
        #             bounds_error=False,
        #             fill_value=fill,
        #         )
        # offax_psf_interp = lambda coordinate: np.exp(ln_offax_psf_interp(coordinate))
        #

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
