"""This module handles one dimensional offax_psfs.fits files."""

from .offax_base import OffAxBase
import astropy.io.fits as pyfits
from lod_unit import lod, lod_eq


class OneDOffAx(OffAxBase):
    """Class for one dimensional off-axis PSFs."""

    def __init__(
        self,
        yip_dir,
        offax_file_name="offax_psf.fits",
        offax_offset_file_name="offax_psf_offset_list.fits",
    ):
        """Initialize the OneDOffAx class."""
        # Load off-axis data (e.g. the planet) (unitless intensity maps)
        self.offax_psf = pyfits.getdata(Path(yip_dir, offax_file_name), 0)

        # The offset list here is in units of lambda/D
        self.offax_psf_offset_list = (
            pyfits.getdata(Path(yip_dir, offax_offset_file_name), 0) * lod
        )
