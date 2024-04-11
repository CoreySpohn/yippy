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
        # Reshape the PSFs to be (n_x_offsets, n_y_offsets, xpix, ypix)
        # reshaped_psfs = psfs.reshape(
        #     len(x_offsets),
        #     len(y_offsets),
        #     psfs.shape[1],
        #     psfs.shape[2],
        # )
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

        # Extend offsets to include their negatives (for quarter symmetry)
        extended_x_offsets = np.append(-x_offsets[:1], x_offsets)
        extended_y_offsets = np.append(-y_offsets[:1], y_offsets)

        # Pad and reflect the reshaped PSFs to cover the full plane
        symmetric_psfs = np.pad(reshaped_psfs, ((1, 0), (1, 0), (0, 0), (0, 0)))

        # Reflect across y-axis
        symmetric_psfs[0, 1:] = reshaped_psfs[0, :, ::-1, :]
        # Reflect across x-axis
        symmetric_psfs[1:, 0] = reshaped_psfs[:, 0, :, ::-1]
        # Reflect the first quadrant to cover the origin
        symmetric_psfs[0, 0] = reshaped_psfs[0, 0, ::-1, ::-1]

        # Log transform the PSF values (avoid negative values in interpolation)
        log_symmetric_psfs = np.log(symmetric_psfs)

        # Choose an appropriate fill_value for log space (e.g., log of the
        # minimum positive PSF value)
        fill_value = np.min(log_symmetric_psfs[log_symmetric_psfs > -np.inf])

        self.log_interp = RegularGridInterpolator(
            (extended_x_offsets, extended_y_offsets),
            log_symmetric_psfs,
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
        psf = np.exp(self.log_interp(Quantity([x, y])))[0]
        return psf
