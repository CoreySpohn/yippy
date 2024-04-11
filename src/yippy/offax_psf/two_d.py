"""This module handles two dimensional offax_psf.fits files."""

import astropy.units as u
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator


class TwoD:
    """This class manages the interpolation for two dimensional offaxis PSFs."""

    def __init__(self, psfs: NDArray, x_offsets: NDArray, y_offsets: NDArray) -> None:
        """Set up the interpolants from the input files."""
        # Check where the offsets begin
        breakpoint()
        self.offset_x_range = u.Quantity([x_offsets[0], x_offsets[-1]])
        self.offset_y_range = u.Quantity([y_offsets[0], y_offsets[-1]])
        self.psf_shape = psfs.shape[1:]

        zz_temp = self.offax_psf.reshape(
            self.offax_psf_offset_x.shape[0],
            self.offax_psf_offset_y.shape[0],
            self.offax_psf.shape[1],
            self.offax_psf.shape[2],
        )
        # Interpolate the PSFs in log space to avoid negative values
        self.ln_offax_psf_interp = RegularGridInterpolator(
            (self.offax_psf_offset_x, self.offax_psf_offset_y),
            np.log(zz_temp),
            bounds_error=False,
            fill_value=-100,
        )
