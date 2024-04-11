"""This module handles quarter symmetric offax_psf.fits files."""

import numpy as np
from astropy.units import Quantity
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator


class QuarterSymmetric:
    """Class for quarter symmetric offax_psf data."""

    def __init__(self, psfs: NDArray, offsets: Quantity) -> None:
        """Initialize the QuarterSymmetric class.

        In this situation the PSFs are only in the first quadrant and the other
        quadrants are symmetric. This class handles the interpolation of the
        PSFs.

        Args:
            psfs (np.ndarray):
                The off-axis PSFs. Shape is (n_x_offsets*n_y_offsets, xpix, ypix).
            offsets (astropy.units.Quantity):
                The offsets of the PSFs in lambda/D in a
                (n_x_offsets*n_y_offsets, 2) array.
        """
        x_offsets = np.unique(offsets[:, 0])
        y_offsets = np.unique(offsets[:, 1])
        # Initialize an empty array for the reshaped PSFs with an extra
        # dimension for symmetry
        reshaped_psfs = np.zeros(
            (len(x_offsets), len(y_offsets), psfs.shape[1], psfs.shape[2])
        )

        # Create a mapping from offset pairs to PSFs
        offset_to_psf = {
            (ox.value, oy.value): psf for (ox, oy), psf in zip(offsets, psfs)
        }

        # Populate the reshaped PSFs array using the mapping
        for i, ox in enumerate(x_offsets):
            for j, oy in enumerate(y_offsets):
                try:
                    reshaped_psfs[i, j] = offset_to_psf[(ox.value, oy.value)]
                except KeyError:
                    # Handle missing PSFs for some offset pairs if necessary
                    raise ValueError("Missing PSF for offset pair ({ox}, {oy})")

        # TODO: Figure out how this should be handled. Should 0 values be undefined if
        # not in the data? Even for something like x=10, y=0?

        # Basically adding a flipped version of the nearest PSF to the x and y axes
        # (say 0.1 lam/D) to the the -0.1 lam/D position. This lets us easily use
        # 0 offsets in the interpolation even though we usually are not given values
        # for 0.

        # Add an extra dimension to the PSFs to allow for 0 offsets
        padded_psfs = np.pad(reshaped_psfs, ((1, 0), (1, 0), (0, 0), (0, 0)))
        padded_psfs[0, 1:] = reshaped_psfs[0, :, :, ::-1]
        padded_psfs[1:, 0] = reshaped_psfs[:, 0, ::-1, :]
        padded_psfs[0, 0] = reshaped_psfs[0, 0, ::-1, ::-1]

        # Extend offsets to include the negative of the smallest offset
        extended_x_offsets = np.append(-x_offsets[0], x_offsets)
        extended_y_offsets = np.append(-y_offsets[0], y_offsets)

        # Log transform the PSF values (avoid negative values in interpolation)
        log_psfs = np.log(padded_psfs)

        # Setting fill_value to -np.inf to return zeros for out of range values
        fill_value = -np.inf

        self.log_interp = RegularGridInterpolator(
            (extended_x_offsets, extended_y_offsets),
            log_psfs,
            method="linear",
            bounds_error=False,
            fill_value=fill_value,
        )

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
        # Translate the x, y values to the first quadrant
        x_val_first_quadrant = abs(x)
        y_val_first_quadrant = abs(y)

        # Use the interpolator to get the PSF in the first quadrant
        psf = np.exp(
            self.log_interp(Quantity([x_val_first_quadrant, y_val_first_quadrant]))
        )[0]

        # Flip the PSF back to the original quadrant if necessary
        if x < 0:
            psf = np.flip(psf, axis=1)  # Flip horizontally
        if y < 0:
            psf = np.flip(psf, axis=0)  # Flip vertically
        return psf
