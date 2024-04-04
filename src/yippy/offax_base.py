"""Base class for all offax_psfs.fits files."""

from numpy.typing import NDArray
import astropy.io.fits as pyfits
from astropy.units import Quantity
from lod_unit import lod, lod_eq


class OffAx:
    """Base class for all off-axis classes."""

    def __init__(
        self,
        yip_dir: Path,
        logger: logging.Logger,
        offax_file_name: str = "offax_psf.fits",
        offax_offset_file_name: str = "offax_psf_offset_list.fits",
    ) -> None:
        """Initialize the OffAx class."""
        # Load off-axis PSF data (e.g. the planet) (unitless intensity maps)
        offax_psf_data = pyfits.getdata(Path(yip_path, "offax_psf.fits"), 0)

        # The offset list here is in units of lambda/D
        offax_psf_offset_list = (
            pyfits.getdata(Path(yip_path, "offax_psf_offset_list.fits"), 0) * lod
        )

        ########################################################################
        # Determine the format of the input coronagraph files so we can handle #
        # the coronagraph correctly (e.g. radially symmetric in x direction)   #
        ########################################################################
        if len(offax_psf_offset_list.shape) > 1:
            if (offax_psf_offset_list.shape[1] != 2) and (
                offax_psf_offset_list.shape[0] == 2
            ):
                # This condition occurs when the offax_psf_offset_list is transposed
                # from the expected format for radially symmetric coronagraphs
                offax_psf_offset_list = offax_psf_offset_list.T

        # Check that we have both x and y offset information (even if there
        # is only one axis with multiple values)
        if offax_psf_offset_list.shape[1] != 2:
            raise UserWarning("Array offax_psf_offset_list should have 2 columns")

        # Get the unique values of the offset list so that we can format the
        # data into
        offax_psf_offset_x = np.unique(offax_psf_offset_list[:, 0])
        offax_psf_offset_y = np.unique(offax_psf_offset_list[:, 1])

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
        elif np.min(offax_psf_offset_list) >= 0 * lod:
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

    def __call__(self, x: Quantity, y: Quantity) -> NDArray:
        """Return the PSF at the given off-axis position."""
        return self.interp(x, y)
