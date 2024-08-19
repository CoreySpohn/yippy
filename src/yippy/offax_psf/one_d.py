"""This module handles one dimensional offax_psf.fits files."""

import numpy as np
from astropy.units import Quantity
from numpy.typing import NDArray

from yippy.logger import logger


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

        self.offsets = offsets
        self.psfs = psfs

    def get_psf_and_transform(self, x: Quantity, y: Quantity):
        """For a given position, return the nearest PSF and interpolation information.

        This function computes the nearest PSF based on the provided (x, y)
        coordinates. It calculates the separation and retrieves the closest PSF
        available in the dataset. Additionally, it returns the necessary shift
        in x, y (with y always being 0 for one-dimensional PSF offsets), and
        the rotation angle required to align the PSF correctly for the given
        coordinates.

        Args:
            x (Quantity):
                The x-coordinate of the position in astropy units.
            y (Quantity):
                The y-coordinate of the position in astropy units.

        Returns:
            Tuple[np.ndarray, float, float, float]:
                - image (np.ndarray):
                    The nearest PSF image corresponding to the given separation.
                - x_shift (float):
                    The shift in the x direction required to align the PSF to
                    the desired position.
                - y_shift (float):
                    The shift in the y direction (always 0 in this implementation).
                - rot_angle (float):
                    The angle by which the PSF should be rotated to match the
                    (x, y) coordinates.

        Notes:
            If the separation is outside the valid range of PSFs, the function
            returns a blank image with the same shape as the PSFs, along with
            shifts and rotation angles set to 0.

        """
        # Get the closest PSF to the given separation
        sep = np.sqrt(x**2 + y**2)
        if sep < self.min_offset or sep > self.max_offset:
            # If the separation is outside the range of the PSFs, return zeros
            logger.warning(
                f"Requested PSF separation ({sep:.2f}) "
                "is outside the provided PSF offsets "
                f"({self.min_offset:.2f}, {self.max_offset:.2f})."
                " Returning a blank image."
            )
            return np.zeros(self.psf_shape), 0, 0, 0

        # Get the distance to shift the PSF in lam/D
        offset_diffs = sep - self.offsets

        nearest_offset = np.argmin(np.abs(offset_diffs))
        x_shift = offset_diffs[nearest_offset]
        image = self.psfs[nearest_offset]

        rot_angle = np.arctan2(y, x)

        # y shift is always 0 for the one dimensional PSF offsets
        return image, x_shift, 0, rot_angle
