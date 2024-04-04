"""Base class for all offax_psfs.fits files."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import astropy.io.fits as pyfits
from astropy.units import Quantity
from lod_unit import lod, lod_eq
from numpy.typing import NDArray
from yippy.offax_psf import OneD

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
    ) -> None:
        """Initialize the OffAx class."""
        # Load off-axis PSF data (e.g. the planet) (unitless intensity maps)
        psfs = pyfits.getdata(Path(yip_dir, offax_data_file), 0)

        # The offset list here is in units of lambda/D
        offsets = pyfits.getdata(Path(yip_dir, offax_offsets_file), 0) * lod

        assert (
            len(offsets) == psfs.shape[0]
        ), "Offsets and PSFs do not have the same number of elements"

        ########################################################################
        # Determine the format of the input coronagraph files so we can handle #
        # the coronagraph correctly (e.g. radially symmetric in x direction)   #
        ########################################################################

        # Check whether offsets is 1D or 2D
        if len(offsets.shape) == 1:
            type = "1d"

        if len(offsets.shape) > 1:
            if (offsets.shape[1] != 2) and (offsets.shape[0] == 2):
                # This condition occurs when the offsets is transposed
                # from the expected format for radially symmetric coronagraphs
                offsets = offsets.T

        # Check that we have both x and y offset information (even if there
        # is only one axis with multiple values)
        if offsets.shape[1] != 2:
            raise UserWarning("Array offsets should have 2 columns")

        # Get the unique values of the offset list so that we can format the
        # data into
        offax_psf_offset_x = np.unique(offsets[:, 0])
        offax_psf_offset_y = np.unique(offsets[:, 1])

        if (len(offax_psf_offset_x) == 1) and (offax_psf_offset_x[0] == 0 * lod):
            type = "1d"
            # Instead of handling angles for 1dy, swap the x and y
            offax_psf_offset_x, offax_psf_offset_y = (
                offax_psf_offset_y,
                offax_psf_offset_x,
            )

            # self.offax_psf_base_angle = 90.0 * u.deg
            logger.info("Coronagraph is radially symmetric")
        elif (len(offax_psf_offset_y) == 1) and (offax_psf_offset_y[0] == 0 * lod):
            type = "1d"
            # self.offax_psf_base_angle = 0.0 * u.deg
            logger.info("Coronagraph is radially symmetric")
        elif len(offax_psf_offset_x) == 1:
            # 1 dimensional with offset (e.g. no offset=0)
            type = "1dno0"
            offax_psf_offset_x, offax_psf_offset_y = (
                offax_psf_offset_y,
                offax_psf_offset_x,
            )
            # self.offax_psf_base_angle = 90.0 * u.deg
            logger.info("Coronagraph is radially symmetric")
        elif len(offax_psf_offset_y) == 1:
            type = "1dno0"
            # self.offax_psf_base_angle = 0.0 * u.deg
            logger.info("Coronagraph is radially symmetric")
        elif np.min(offsets) >= 0 * lod:
            type = "2dq"
            # self.offax_psf_base_angle = 0.0 * u.deg
            # self.logger.info(
            #     f"Quarterly symmetric response --> reflecting PSFs ({self.type})"
            # )
            logger.info("Coronagraph is quarterly symmetric")
        else:
            type = "2df"
            # self.offax_psf_base_angle = 0.0 * u.deg
            logger.info("Coronagraph response is full 2D")

        # interpolate planet data depending on type
        if "1" in type:
            # Always set up to interpolate along the x axis
            self.offax = OneD(psfs, offax_psf_offset_x)
        else:
            zz_temp = offax_psf.reshape(
                offax_psf_offset_x.shape[0],
                offax_psf_offset_y.shape[0],
                offax_psf.shape[1],
                offax_psf.shape[2],
            )
            if type == "2dq":
                # Reflect PSFs to cover the x = 0 and y = 0 axes.
                offax_psf_offset_x = np.append(
                    -offax_psf_offset_x[0], offax_psf_offset_x
                )
                offax_psf_offset_y = np.append(
                    -offax_psf_offset_y[0], offax_psf_offset_y
                )
                zz = np.pad(zz_temp, ((1, 0), (1, 0), (0, 0), (0, 0)))
                zz[0, 1:] = zz_temp[0, :, ::-1, :]
                zz[1:, 0] = zz_temp[:, 0, :, ::-1]
                zz[0, 0] = zz_temp[0, 0, ::-1, ::-1]

                ln_offax_psf_interp = RegularGridInterpolator(
                    (offax_psf_offset_x, offax_psf_offset_y),
                    np.log(zz),
                    method="linear",
                    bounds_error=False,
                    fill_value=fill,
                )
            else:
                # This section included references to non-class attributes for
                # offax_psf_offset_x and offax_psf_offset_y. I think it meant
                # to be the class attributes
                ln_offax_psf_interp = RegularGridInterpolator(
                    (offax_psf_offset_x, offax_psf_offset_y),
                    np.log(zz_temp),
                    method="linear",
                    bounds_error=False,
                    fill_value=fill,
                )
        offax_psf_interp = lambda coordinate: np.exp(ln_offax_psf_interp(coordinate))

    def __call__(self, x: float, y: float) -> NDArray:
        """Return the PSF at the given x/y position.

        Args:
            x(float):
                x position in lambda/D
            y(float):
                y position in lambda/D
        Returns:
            NDArray:
                The PSF at the given x/y position
        """
        return self.psf(x, y)
