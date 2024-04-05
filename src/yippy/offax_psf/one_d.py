"""This module handles one dimensional offax_psfs.fits files."""

import astropy.units as u
import numpy as np
from astropy.units import Quantity
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
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
        # Check where the offsets begin
        self.offset_range = u.Quantity([offsets[0], offsets[-1]])
        self.psf_shape = psfs.shape[1:]

        # Interpolate the PSFs in log space to avoid negative values
        self.log_interp = CubicSpline(offsets, np.log(psfs))

        # Define the one-d interpolation function
        self.one_d_interp = lambda x: np.exp(self.log_interp(x))

    def __call__(self, x: Quantity, y: Quantity):
        """Return the PSF at the given x/y position.

        Calculates the separation of the position and determines. If the x/y
        position is outside the range of the PSFs, it will return zeros.

        Args:
            x(Quantity):
                x position in lambda/D
            y(Qunatity):
                y position in lambda/D
        Returns:
            NDArray:
                The PSF at the given x/y position
        """
        sep = np.sqrt(x**2 + y**2)
        if sep < self.offset_range[0] or sep > self.offset_range[1]:
            # If the separation is outside the range of the PSFs, return zeros
            return np.zeroslike(self.psf_shape)

        # Get the rotation angle
        rot_angle = np.arctan2(y, x)

        # Interpolate the PSF to the given separation
        one_d_psf = self.one_d_interp(sep)

        # Rotate the PSF
        psf = np.exp(
            rotate(
                np.log(one_d_psf),
                -rot_angle.to(u.deg).value,
                reshape=False,
                mode="nearest",
                order=5,
            )
        )
        return psf
