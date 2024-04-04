"""This module handles one dimensional offax_psfs.fits files."""

from pathlib import Path

import astropy.io.fits as pyfits
from lod_unit import lod, lod_eq
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from scipy.ndimage import rotate


class OneD:
    """Class for one dimensional off-axis PSFs."""

    def __init__(self, psfs: NDArray, offsets: NDArray) -> None:
        """Initialize the OneD class.

        Args:
            psfs (np.ndarray):
                The off-axis PSFs. Shape is (n, xpix, ypix).
            offsets (np.ndarray):
                Array of length n with the offsets of the off-axis PSFs in lambda/D.
        """
        ln_interp = interp1d(
            offsets,
            np.log(psfs),
            kind="cubic",
            axis=0,
            bounds_error=False,
            fill_value=np.log(1e-100),
        )
        psf_shape = psfs.shape
        self.star_pos = np.array([psfs.shape[1], psfs.shape[2]]) / 2
        breakpoint()
        self.one_d_interp = lambda x: np.exp(ln_interp(x))

    def interp(self, x: float, y: float):
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
        # Get the angular separation
        sep = x**2 + y**2
        psf = self.one_d_interp(sep)

        # Get the rotation angle
        rot_angle = np.arctan2(y, x)

        # Rotate the PSF
        psf = rotate(psf, rot_angle)
        return self.interp((x, y))
