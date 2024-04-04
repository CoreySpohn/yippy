"""This module handles one dimensional offax_psfs.fits files."""

from .offax_base import OffAxBase


class OneDOffAx(OffAxBase):
    """Class for one dimensional off-axis PSFs."""

    def __init__(self, offax_psf_dir):
        """Initialize the OneDOffAx class."""
