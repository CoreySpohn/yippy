"""This module handles one dimensional offax_psf.fits files."""

import astropy.units as u
import numpy as np
from astropy.units import Quantity
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.ndimage import rotate


class OneD:
    """Handles interpolation and rotation of one-dimensional off-axis PSFs.

    This class enables the calculation of the PSF at a specified position using
    interpolated one-dimensional PSF data. If the requested position is outside
    the interpolation range, a zero array is returned. This class should not be
    interacted with directly, but rather by calling Coronagraph.offax(x,y)
    which then calls this class when appropriate.

    Attributes:
        min_offset (float):
            The minimum offset value in lambda/D.
        max_offset (float):
            The maximum offset value in lambda/D.
        psf_shape (tuple):
            Shape of the PSF arrays, typically (xpix, ypix).

    Args:
        psfs (NDArray):
            The off-axis PSFs, shaped (n, xpix, ypix).
        offsets (NDArray):
            Array of offsets in lambda/D corresponding to each PSF.
    """

    def __init__(self, psfs: NDArray, offsets: NDArray) -> None:
        """Initializes the OneD class by creating the log interpolant.

        The PSFs are interpolated in logarithmic space to ensure positive values
        throughout the interpolation range.
        """
        # Check where the offsets begin
        self.min_offset = offsets[0]
        self.max_offset = offsets[-1]
        self.psf_shape = psfs.shape[1:]

        # Interpolate the PSFs in log space to avoid negative values
        self.log_interp = CubicSpline(offsets, np.log(psfs))

        # Define the one-d interpolation function
        self.one_d_interp = lambda x: np.exp(self.log_interp(x))

    def __call__(self, x: Quantity, y: Quantity):
        """Calculates and returns the PSF at the specified x, y position in lambda/D.

        If the computed separation from the origin exceeds the interpolation range,
        a zero-filled array matching the PSF shape is returned. Otherwise, the PSF is
        interpolated and rotated to the correct angle based on its position.

        Args:
            x (Quantity):
                x position in lambda/D.
            y (Quantity):
                y position in lambda/D.

        Returns:
            NDArray:
                The interpolated and possibly rotated PSF array at the given position.
        """
        sep = np.sqrt(x**2 + y**2)
        if sep < self.min_offset or sep > self.max_offset:
            # If the separation is outside the range of the PSFs, return zeros
            return np.zeros(self.psf_shape)

        # Get the rotation angle
        rot_angle = np.arctan2(y, x)

        # Interpolate the PSF to the given separation
        one_d_psf = self.one_d_interp(sep)

        # Check if we need to rotate the PSF
        if rot_angle.value != 0.0:
            psf = np.exp(
                rotate(
                    np.log(one_d_psf),
                    -rot_angle.to(u.deg).value,
                    reshape=False,
                    mode="nearest",
                    order=5,
                )
            )
        else:
            psf = one_d_psf
        return psf
