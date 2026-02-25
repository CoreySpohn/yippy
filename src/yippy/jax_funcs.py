"""JAX functions for image processing operations."""

from functools import partial

import jax.numpy as jnp
from jax import lax


def get_pad_info(image, pad_factor):
    """Get the padding information for an image.

    Args:
        image (jax.numpy.ndarray):
            The input image to be shifted.
        pad_factor (float):
            The factor by which to pad the image.

    Returns:
        int:
            The number of pixels in the original image.
        int:
            The number of pixels to pad the image.
        int:
            The edge of the image after padding.
        int:
            The number of pixels in the final image.
    """
    n_pixels_orig = image.shape[0]
    n_pad = int(pad_factor * n_pixels_orig)
    img_edge = n_pad + n_pixels_orig
    n_pixels_final = int(2 * n_pixels_orig * pad_factor + n_pixels_orig)
    return n_pixels_orig, n_pad, img_edge, n_pixels_final


def fft_shift_x(image, shift_pixels, phasor):
    """Apply a Fourier shift to an image along the x axis.

    A traceable version of the `fft_shift_1d` function that applies a Fourier
    shift to an image along the x-axis. This allows the function to be traced
    and compiled by JAX.

    Args:
        image (jax.numpy.ndarray):
            The input 2D image to be shifted.
        shift_pixels (float):
            The number of pixels by which to shift the image along the specified axis.
        phasor (jax.numpy.ndarray):
            Precomputed components for the Fourier shift (exp(-2j * pi * fft_freqs)).

    Returns:
        jax.numpy.ndarray:
            The shifted image after applying the Fourier transform and removing
            the padding.
    """
    _n_pixels_orig, n_pad, img_edge, _n_pixels_final = get_pad_info(image, 1.5)

    # Pad the image with zeros
    padded = lax.pad(image, 0.0, [(n_pad, n_pad, 0), (n_pad, n_pad, 0)])

    # Take the 1D Fourier transform along the specified axis
    padded = jnp.fft.fft(padded, axis=1)

    # Tile the phasor to match the dimensions of the padded image
    # Horizontal shift (x-axis)
    phasor = jnp.tile(phasor**shift_pixels, (padded.shape[0], 1))

    # Apply the phasor along the specified axis
    padded = padded * phasor

    # Reconstruct the image using the inverse Fourier transform along the specified axis
    padded = jnp.real(jnp.fft.ifft(padded, axis=1))

    # Unpad the image to return to the original size
    image = padded[n_pad:img_edge, n_pad:img_edge]

    # Cut any negative values to zero. This occurs in the region with no
    # information in the original image (e.g. the left pixels when moving
    # a PSF rightwards)
    return jnp.maximum(image, 0.0)


def fft_shift_y(image, shift_pixels, phasor):
    """Apply a Fourier shift to an image along the y axis.

    A traceable version of the `fft_shift_1d` function that applies a Fourier
    shift to an image along the y-axis. This allows the function to be traced
    and compiled by JAX.

    Args:
        image (jax.numpy.ndarray):
            The input 2D image to be shifted.
        shift_pixels (float):
            The number of pixels by which to shift the image along the specified axis.
        phasor (jax.numpy.ndarray):
            Precomputed components for the Fourier shift (exp(-2j * pi * fft_freqs)).

    Returns:
        jax.numpy.ndarray:
            The shifted image after applying the Fourier transform and removing
            the padding.
    """
    _n_pixels_orig, n_pad, img_edge, _n_pixels_final = get_pad_info(image, 1.5)

    # Pad the image with zeros
    padded = lax.pad(image, 0.0, [(n_pad, n_pad, 0), (n_pad, n_pad, 0)])

    # Take the 1D Fourier transform along the specified axis
    padded = jnp.fft.fft(padded, axis=0)

    # Tile the phasor to match the dimensions of the padded image
    # Vertical shift (y-axis)
    phasor = jnp.tile(phasor**shift_pixels, (padded.shape[1], 1)).T

    # Apply the phasor along the specified axis
    padded = padded * phasor

    # Reconstruct the image using the inverse Fourier transform along the specified axis
    padded = jnp.real(jnp.fft.ifft(padded, axis=0))

    # Unpad the image to return to the original size
    image = padded[n_pad:img_edge, n_pad:img_edge]

    # Cut any negative values to zero. This occurs in the region with no
    # information in the original image (e.g. the left pixels when moving
    # a PSF rightwards)
    return jnp.maximum(image, 0.0)


def create_shift_mask(psf, shift_x, shift_y, x_grid, y_grid, fill_val=1):
    """Create a mask to identify valid pixels to average.

    This function is useful because when the PSF is shifted there are empty
    pixels, since they were outside the initial image, and should not be
    included in the final average. This version avoids lax.cond for better
    performance under JIT.

    Args:
        psf (jax.numpy.ndarray):
            The PSF image to shift.
        shift_x (float):
            The shift in the x direction.
        shift_y (float):
            The shift in the y direction.
        x_grid (jax.numpy.ndarray):
            The x-coordinate grid.
        y_grid (jax.numpy.ndarray):
            The y-coordinate grid.
        fill_val (float, optional):
            The value to fill the mask with.

    Returns:
        jax.numpy.ndarray:
            The mask to identify valid pixels to average.
    """
    height, width = psf.shape

    # Create masks for x and y shifts using branchless arithmetic.
    # This is much more efficient than using lax.cond.

    # For positive x shift, valid pixels are where x_grid >= ceil(shift_x).
    # For negative x shift, valid pixels are where x_grid < width - ceil(-shift_x).
    shift_x_pos = jnp.maximum(0, shift_x)
    shift_x_neg = jnp.maximum(0, -shift_x)
    n_x_pos = jnp.ceil(shift_x_pos).astype(jnp.int32)
    n_x_neg = jnp.ceil(shift_x_neg).astype(jnp.int32)
    mask_x = (x_grid >= n_x_pos) & (x_grid < (width - n_x_neg))

    # For positive y shift, valid pixels are where y_grid >= ceil(shift_y).
    # For negative y shift, valid pixels are where y_grid < height - ceil(-shift_y).
    shift_y_pos = jnp.maximum(0, shift_y)
    shift_y_neg = jnp.maximum(0, -shift_y)
    n_y_pos = jnp.ceil(shift_y_pos).astype(jnp.int32)
    n_y_neg = jnp.ceil(shift_y_neg).astype(jnp.int32)
    mask_y = (y_grid >= n_y_pos) & (y_grid < (height - n_y_neg))

    # Combine the masks and multiply by the fill value
    mask = mask_x & mask_y
    return mask.astype(psf.dtype) * fill_val


def get_near_inds_offsets_1D(
    x_offsets: jnp.ndarray, y_offsets: jnp.ndarray, _x: float, _y: float
):
    """Computes the nearest indices and offsets for the given _x position in 1D.

    Args:
        x_offsets (jnp.ndarray): 1D array of x offsets, sorted in ascending order.
        y_offsets (jnp.ndarray): 1D array of y offsets.
        _x (float): The x-coordinate for which to find nearby offsets.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]:
            - near_inds: Array of shape (2,) containing the indices of the two
              surrounding points.
            - near_offsets: Array of shape (2,) containing the corresponding x offsets.
    """
    # Find insertion index for _x
    _x_ind = jnp.searchsorted(x_offsets, _x, side="left")

    # Handle boundary conditions
    x_ind_low = jnp.clip(_x_ind - 1, 0, x_offsets.size - 1)
    x_ind_high = jnp.clip(_x_ind, 0, x_offsets.size - 1)

    y_ind = 0

    # Collect the two nearest indices
    near_inds = jnp.array([[x_ind_low, y_ind], [x_ind_high, y_ind]])

    # Extract the corresponding x offset values
    x_vals_low = x_offsets[x_ind_low]
    x_vals_high = x_offsets[x_ind_high]
    y_val = y_offsets[y_ind]

    near_offsets = jnp.array([[x_vals_low, y_val], [x_vals_high, y_val]])

    return near_inds, near_offsets


def create_avg_psf_1D(
    x: float,
    y: float,
    pixel_scale: float,
    x_offsets: jnp.ndarray,
    y_offsets: jnp.ndarray,
    x_grid: jnp.ndarray,
    y_grid: jnp.ndarray,
    x_phasor: jnp.ndarray,
    y_phasor: jnp.ndarray,
    reshaped_psfs: jnp.ndarray,
):
    """Creates and returns the PSF at the specified off-axis position using JAX."""
    # The core logic is similar to OffAx, but uses JAX operations
    _x, _y = convert_xy_1D(x, y)

    near_inds, near_offsets = get_near_inds_offsets_1D(x_offsets, y_offsets, _x, _y)

    near_psfs = reshaped_psfs[near_inds[:, 0], near_inds[:, 1]]

    near_shifts = (jnp.array([_x, _y]) - near_offsets) / pixel_scale
    near_diffs = jnp.linalg.norm(near_shifts, axis=1)
    sigma = 0.25
    # Adding a small value to avoid division by zero
    weights = jnp.exp(-(near_diffs**2) / (2 * sigma**2)) + 1e-16
    weights /= weights.sum()

    # Manually shift each PSF
    shifted_psf1, mask1 = shift_and_mask(
        near_psfs[0],
        near_shifts[0, 0],
        near_shifts[0, 1],
        weights[0],
        x_grid,
        y_grid,
        x_phasor,
        y_phasor,
    )
    shifted_psf2, mask2 = shift_and_mask(
        near_psfs[1],
        near_shifts[1, 0],
        near_shifts[1, 1],
        weights[1],
        x_grid,
        y_grid,
        x_phasor,
        y_phasor,
    )

    # Accumulate the weighted PSFs and the weight masks
    psf = shifted_psf1 + shifted_psf2
    weight_array = mask1 + mask2

    # Normalize the PSF
    safe_reciprocal = jnp.where(weight_array != 0, 1.0 / weight_array, 0.0)
    psf = psf * safe_reciprocal

    return psf


def create_avg_psf_2DQ(
    x: float,
    y: float,
    pixel_scale: float,
    x_offsets: jnp.ndarray,
    y_offsets: jnp.ndarray,
    x_grid: jnp.ndarray,
    y_grid: jnp.ndarray,
    x_phasor: jnp.ndarray,
    y_phasor: jnp.ndarray,
    reshaped_psfs: jnp.ndarray,
):
    """Creates and returns the PSF at the specified off-axis position using JAX."""
    # The core logic is similar to OffAx, but uses JAX operations
    _x, _y = convert_xy_2DQ(x, y)

    near_inds, near_offsets = get_near_inds_offsets_2D(x_offsets, y_offsets, _x, _y)

    near_psfs = reshaped_psfs[near_inds[:, 0], near_inds[:, 1]]

    near_shifts = (jnp.array([_x, _y]) - near_offsets) / pixel_scale
    near_diffs = jnp.linalg.norm(near_shifts, axis=1)
    sigma = 0.25
    # Adding a small value to avoid division by zero
    weights = jnp.exp(-(near_diffs**2) / (2 * sigma**2)) + 1e-16
    weights /= weights.sum()

    # Manually shift each PSF
    shifted_psf1, mask1 = shift_and_mask(
        near_psfs[0],
        near_shifts[0, 0],
        near_shifts[0, 1],
        weights[0],
        x_grid,
        y_grid,
        x_phasor,
        y_phasor,
    )
    shifted_psf2, mask2 = shift_and_mask(
        near_psfs[1],
        near_shifts[1, 0],
        near_shifts[1, 1],
        weights[1],
        x_grid,
        y_grid,
        x_phasor,
        y_phasor,
    )
    shifted_psf3, mask3 = shift_and_mask(
        near_psfs[2],
        near_shifts[2, 0],
        near_shifts[2, 1],
        weights[2],
        x_grid,
        y_grid,
        x_phasor,
        y_phasor,
    )
    shifted_psf4, mask4 = shift_and_mask(
        near_psfs[3],
        near_shifts[3, 0],
        near_shifts[3, 1],
        weights[3],
        x_grid,
        y_grid,
        x_phasor,
        y_phasor,
    )

    # Accumulate the weighted PSFs and the weight masks
    psf = shifted_psf1 + shifted_psf2 + shifted_psf3 + shifted_psf4
    weight_array = mask1 + mask2 + mask3 + mask4

    safe_reciprocal = jnp.where(weight_array != 0, 1.0 / weight_array, 0.0)
    psf = psf * safe_reciprocal

    return psf


def get_near_inds_offsets_2D(
    x_offsets: jnp.ndarray, y_offsets: jnp.ndarray, _x: float, _y: float
):
    """Computes the nearest indices and offsets for the given (_x, _y) position in 2D.

    Args:
        x_offsets (jnp.ndarray): 1D array of x offsets, sorted in ascending order.
        y_offsets (jnp.ndarray): 1D array of y offsets, sorted in ascending order.
        _x (float): The x-coordinate for which to find nearby offsets.
        _y (float): The y-coordinate for which to find nearby offsets.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]:
            - near_inds: Array of shape (4, 2) containing the indices of the
              four surrounding points.
            - near_offsets: Array of shape (4, 2) containing the corresponding
              (x, y) offsets.
    """
    # Find insertion indices for _x and _y
    _x_ind = jnp.searchsorted(x_offsets, _x, side="left")
    _y_ind = jnp.searchsorted(y_offsets, _y, side="left")

    # Handle boundary conditions for x indices
    x_ind_low = jnp.clip(_x_ind - 1, 0, x_offsets.size - 1)
    x_ind_high = jnp.clip(_x_ind, 0, x_offsets.size - 1)

    # Handle boundary conditions for y indices
    y_ind_low = jnp.clip(_y_ind - 1, 0, y_offsets.size - 1)
    y_ind_high = jnp.clip(_y_ind, 0, y_offsets.size - 1)

    # Collect the two nearest indices for x and y
    x_inds = jnp.array([x_ind_low, x_ind_high])
    y_inds = jnp.array([y_ind_low, y_ind_high])

    # Manually create the four combinations without using meshgrid
    near_inds = jnp.array(
        [
            [x_inds[0], y_inds[0]],
            [x_inds[0], y_inds[1]],
            [x_inds[1], y_inds[0]],
            [x_inds[1], y_inds[1]],
        ]
    )

    # Extract the corresponding x and y offset values
    x_vals_low = x_offsets[x_inds[0]]
    x_vals_high = x_offsets[x_inds[1]]
    y_vals_low = y_offsets[y_inds[0]]
    y_vals_high = y_offsets[y_inds[1]]

    # Manually create the corresponding (x, y) offset combinations
    near_offsets = jnp.array(
        [
            [x_vals_low, y_vals_low],
            [x_vals_low, y_vals_high],
            [x_vals_high, y_vals_low],
            [x_vals_high, y_vals_high],
        ]
    )
    return near_inds, near_offsets


def shift_and_mask(
    near_psf, shift_x, shift_y, weight, x_grid, y_grid, x_phasor, y_phasor
):
    """Shifts the PSF in x and y directions and applies a weight mask.

    Args:
        near_psf (jax.numpy.ndarray): The PSF image to shift.
        shift_x (float): Shift in the x-direction.
        shift_y (float): Shift in the y-direction.
        weight (float): Weight for the PSF.
        x_grid (jax.numpy.ndarray): The x-coordinate grid.
        y_grid (jax.numpy.ndarray): The y-coordinate grid.
        x_phasor (jax.numpy.ndarray): Precomputed components for the Fourier shift.
        y_phasor (jax.numpy.ndarray): Precomputed components for the Fourier shift.

    Returns:
        Tuple[jax.numpy.ndarray, jax.numpy.ndarray]:
            - Weighted shifted PSF.
            - Weight mask.
    """
    # Shift the PSF in x and y directions
    shifted_psf = fft_shift_x(near_psf, shift_x, x_phasor)
    shifted_psf = fft_shift_y(shifted_psf, shift_y, y_phasor)

    # Create the weight mask
    weight_mask = create_shift_mask(
        near_psf, shift_x, shift_y, x_grid, y_grid, fill_val=weight
    )

    # Apply the weight mask
    weighted_psf = weight_mask * shifted_psf

    return weighted_psf, weight_mask


def convert_xy_1D(x, y):
    """Converts x and y to 1D coordinates."""
    return jnp.sqrt(x**2 + y**2), 0


def convert_xy_2DQ(x, y):
    """Converts x and y to 2D quarter symmetric coordinates."""
    return jnp.abs(x), jnp.abs(y)


def convert_xy_2D(x, y):
    """Converts x and y to 2D coordinates."""
    return x, y


def basic_shift_val(input_val, converted_val, pixel_scale):
    """Calculates the shift in pixels for a basic shift."""
    return (input_val - converted_val) / pixel_scale


def sym_shift_val(input_val, converted_val, pixel_scale):
    """Calculates the shift in pixels for a symmetric shift."""
    sign = jnp.where(input_val >= 0, 1.0, -1.0)
    return (input_val - sign * converted_val) / pixel_scale


def x_basic_shift(input_val, converted_val, PSF, pixel_scale, x_phasor):
    """Shifts the PSF to the specified x position."""
    shift = basic_shift_val(input_val, converted_val, pixel_scale)
    return fft_shift_x(PSF, shift, x_phasor)


def y_basic_shift(input_val, converted_val, PSF, pixel_scale, y_phasor):
    """Shifts the PSF to the specified y position."""
    shift = basic_shift_val(input_val, converted_val, pixel_scale)
    return fft_shift_y(PSF, shift, y_phasor)


def x_symmetric_shift(input_val, converted_val, PSF, pixel_scale, x_phasor):
    """Shifts the PSF to the specified position assuming symmetry about x=0."""
    # Apply a horizontal flip if the input value is negative
    _PSF = jnp.where(input_val < 0, jnp.fliplr(PSF), PSF)

    # Calculate the distance to shift the PSF
    shift = sym_shift_val(input_val, converted_val, pixel_scale)
    # Apply the shift
    return fft_shift_x(_PSF, shift, x_phasor)


def y_symmetric_shift(input_val, converted_val, PSF, pixel_scale, y_phasor):
    """Shifts the PSF to the specified position assuming symmetry about y=0."""
    # Apply a vertical flip if the input value is negative
    _PSF = jnp.where(input_val < 0, jnp.flipud(PSF), PSF)

    # Get the distance to shift the PSF
    shift = sym_shift_val(input_val, converted_val, pixel_scale)
    # Apply the shift
    return fft_shift_y(_PSF, shift, y_phasor)


def fft_rotate_jax(image, rot_deg):
    """Rotate an image by a specified angle using Fourier-based shear operations.

    This function performs an image rotation by decomposing the rotation into
    three sequential shear operations in the Fourier domain. For more details
    see Larkin et al. (1997).

    Args:
        image (jax.numpy.ndarray):
            The input image to be rotated.
        rot_deg (float):
            The rotation angle in degrees. Positive values rotate the image
            counterclockwise, and negative values rotate it clockwise.

    Returns:
        jax.numpy.ndarray:
            The rotated image.
    """
    # To rotate counterclockwise, with the origin in the lower left, we use the
    # negative of the angle
    rot_deg = -rot_deg

    # Cut the angle to (-45, 45] and a number of 90-degree rotations
    rot_deg, n_rot = decompose_angle_jax(rot_deg)

    image = rot90_helper_jax(image, n_rot)

    image = lax.cond(
        rot_deg != 0.0,
        lambda x: rotate_with_shear(image, x),
        lambda x: image,
        rot_deg,
    )

    return image


def rotate_with_shear(image, rot_deg):
    """Rotate an image by a specified angle using Fourier-based shear operations.

    This is a helper function that simplifies the fft_rotate_jax function by
    simplifying the lambda function used in the lax.cond call.

    Args:
        image (jax.numpy.ndarray):
            The input image to be rotated.
        rot_deg (float):
            The rotation angle in degrees.

    Returns:
        jax.numpy.ndarray:
            The rotated image.
    """
    theta = jnp.deg2rad(rot_deg)
    a = jnp.tan(theta / 2)
    b = -jnp.sin(theta)

    x_freqs, x_dists, y_freqs, y_dists = fft_shear_setup(image)
    # Rotate using three shears
    # s_x
    image = fft_shear_x(image, a, x_freqs, x_dists)

    # s_yx
    image = fft_shear_y(image, b, y_freqs, y_dists)

    # s_xyx
    image = fft_shear_x(image, a, x_freqs, x_dists)
    return image


def fft_shear_setup(image):
    """Perform a shear operation in the Fourier domain.

    Args:
        image (jax.numpy.ndarray):
            The input image to be sheared.

    Returns:
        tuple:
            - jax.numpy.ndarray: x frequencies used for the Fourier transform.
            - jax.numpy.ndarray: x distances from the center of the image.
            - jax.numpy.ndarray: y frequencies used for the Fourier transform.
            - jax.numpy.ndarray: y distances from the center of the image
    """
    # Calculate padding size based on the image dimensions
    n_pixels = image.shape[0]
    n_pad = int(1.5 * n_pixels)

    # Pad the image with zeros
    padded = jnp.pad(image, n_pad, mode="constant")

    # Calculate the coordinate array for the padded image
    padded_height, padded_width = padded.shape
    center_y, center_x = (jnp.array(padded.shape) - 1) / 2
    grid_y, grid_x = jnp.mgrid[0:padded_height, 0:padded_width]

    # Array of distances from the center of the image along the shear axis
    # if axis == 1:
    # Shearing along the horizontal axis
    x_dists = grid_x - center_x
    x_perpendicular_axis = 1
    # Compute the Fourier frequencies for the dimension perpendicular to the shear axis
    x_freqs = jnp.fft.fftfreq(x_dists.shape[x_perpendicular_axis])
    x_freqs = jnp.fft.fftshift(x_freqs)

    # Tile the shifted frequencies to match the dimensions of the padded image
    x_freqs = jnp.tile(x_freqs, (x_dists.shape[1], 1)).T

    # Shearing along the vertical axis
    y_dists = grid_y - center_y

    # Determine the perpendicular axis to the shear direction
    y_perpendicular_axis = 0

    y_freqs = jnp.fft.fftfreq(y_dists.shape[y_perpendicular_axis])
    y_freqs = jnp.fft.fftshift(y_freqs)
    y_freqs = jnp.tile(y_freqs, (y_dists.shape[0], 1))

    return x_freqs, x_dists, y_freqs, y_dists


def fft_shear_x(image, shear_factor, x_freqs, x_dists):
    """Perform a shear operation in the Fourier domain along the x-axis.

    Uses JAX functions to perform the shear operation in the Fourier domain
    along the x-axis.

    Args:
        image (jax.numpy.ndarray):
            The input image to be sheared.
        shear_factor (float):
            The shear factor.
        x_freqs (jax.numpy.ndarray):
            x frequencies used for the Fourier transform.
        x_dists (jax.numpy.ndarray):
            x distances from the center of the image.

    Returns:
        jax.numpy.ndarray:
            The sheared image with the zero padding removed.
    """
    # Calculate padding size based on the image dimensions
    n_pixels = image.shape[0]
    n_pad = int(1.5 * n_pixels)
    img_edge = n_pad + n_pixels

    # Pad the image with zeros
    padded = jnp.pad(image, n_pad, mode="constant")
    padded = jnp.fft.fftshift(padded)
    padded = jnp.fft.fftshift(jnp.fft.fft(padded, axis=1))

    # Apply the phase shift (shear) in the Fourier domain
    padded = jnp.exp(-2j * jnp.pi * shear_factor * x_freqs * x_dists) * padded

    # Shift back and apply the inverse Fourier transform along the specified axis
    padded = jnp.fft.fftshift(padded)
    padded = jnp.fft.ifft(padded, axis=1)
    padded = jnp.fft.fftshift(padded)

    # Unpad the image to return to the original size
    image = jnp.real(padded[n_pad:img_edge, n_pad:img_edge])

    return image


def fft_shear_y(image, shear_factor, y_freqs, y_dists):
    """Perform a shear operation in the Fourier domain along the y-axis.

    Uses JAX operations.

    Args:
        image (jax.numpy.ndarray):
            The input image to be sheared.
        shear_factor (float):
            The shear factor.
        y_freqs (jax.numpy.ndarray):
            y frequencies used for the Fourier transform.
        y_dists (jax.numpy.ndarray):
            y distances from the center of the image.

    Returns:
        jax.numpy.ndarray:
            The sheared image with the zero padding removed.
    """
    # Calculate padding size based on the image dimensions
    n_pixels = image.shape[0]
    n_pad = int(1.5 * n_pixels)
    img_edge = n_pad + n_pixels

    # Pad the image with zeros
    padded = jnp.pad(image, n_pad, mode="constant")
    padded = jnp.fft.fftshift(padded)
    padded = jnp.fft.fftshift(jnp.fft.fft(padded, axis=0))

    # Apply the phase shift (shear) in the Fourier domain
    padded = jnp.exp(-2j * jnp.pi * shear_factor * y_freqs * y_dists) * padded

    # Shift back and apply the inverse Fourier transform along the specified axis
    padded = jnp.fft.fftshift(padded)
    padded = jnp.fft.ifft(padded, axis=0)
    padded = jnp.fft.fftshift(padded)

    # Unpad the image to return to the original size
    image = jnp.real(padded[n_pad:img_edge, n_pad:img_edge])

    return image


def decompose_angle_jax(angle):
    """Decompose an angle from [0, 360) to (-45, 45] and 90 degree rotations.

    Args:
        angle (float):
            The input angle in degrees.

    Returns:
        tuple:
            - float: The rotation angle in the range (-45, 45] degrees.
            - int: The number of 90-degree rotations to apply.
    """
    # Normalize the angle to [0, 360)
    angle = angle % 360

    # Determine the number of 90-degree rotations
    n_rot = (angle // 90).astype(int)

    # Cut the angle to [0, 90)
    adjusted_angle = angle % 90

    # Adjust the angle to the range (-45, 45]
    adjusted_angle, n_rot = lax.cond(
        adjusted_angle > 45,
        lambda x: (x - 90, n_rot + 1),
        lambda x: (x, n_rot),
        adjusted_angle,
    )
    # if n_rot is 4, set it to 0 to avoid unnecessary rotations
    # this occurs when the angle is in the range (315, 360)
    n_rot = lax.cond(
        n_rot == 4,
        lambda x: 0,
        lambda x: x,
        n_rot,
    )

    return adjusted_angle, n_rot


def rot90_traceable(m, k=1, axes=(0, 1)):
    """Rotate an array by 90 degrees in the plane specified by axes.

    This function is a traceable version of `numpy.rot90` taken from the
    jax GitHub issues.

    Args:
        m (jax.numpy.ndarray):
            The input array to be rotated.
        k (int, optional):
            The number of 90-degree rotations to apply.
        axes (tuple, optional):
            The axes to rotate the array in.

    Returns:
        jax.numpy.ndarray:
            The rotated array.
    """
    k %= 4
    return lax.switch(k, [partial(jnp.rot90, m, k=i, axes=axes) for i in range(4)])


def rot90_helper_jax(image, n_rot):
    """Rotate an image by 90 degrees a specified number of times.

    Ensures that the image is of odd dimensions before rotating to avoid
    incorrect centering. Uses JAX operations.

    Args:
        image (jax.numpy.ndarray):
            The input image to be rotated.
        n_rot (int):
            The number of 90-degree rotations to apply.

    Returns:
        jax.numpy.ndarray:
            The rotated image.
    """
    # Check if the image needs padding for even dimensions
    needs_padding = jnp.logical_or(image.shape[0] % 2 == 0, image.shape[1] % 2 == 0)

    def pad_and_rotate(img, n_rot):
        padded = jnp.zeros((img.shape[0] + 1, img.shape[1] + 1))
        padded = padded.at[:-1, :-1].set(img)
        rotated = rot90_traceable(padded, k=n_rot)
        return rotated[:-1, :-1]

    def rotate_only(img, n_rot):
        return rot90_traceable(img, k=n_rot)

    _img = lax.cond(
        needs_padding,
        lambda img: pad_and_rotate(img, n_rot),
        lambda img: rotate_only(img, n_rot),
        image,
    )

    return _img


def synthesize_psf_separable(
    x,
    y,
    pixel_scale,
    flat_psfs,
    x_offsets,
    y_offsets,
    offset_to_flat_idx,
    kx,
    ky,
    n_pad,
    x_symmetric,
    y_symmetric,
    input_type,
    max_offset,
):
    """Synthesizes PSF using separable 1D FFTs.

    Optimized for speed and stability with even-sized padding.
    Uses flat_psfs with offset_to_flat_idx mapping for memory efficiency.

    Args:
        x (float):
            X coordinate in lambda/D.
        y (float):
            Y coordinate in lambda/D.
        pixel_scale (float):
            Pixel scale in lambda/D per pixel.
        flat_psfs (jnp.ndarray):
            Flat array of PSFs with shape (N_psfs, H, W).
        x_offsets (jnp.ndarray):
            Sorted array of unique x offsets.
        y_offsets (jnp.ndarray):
            Sorted array of unique y offsets.
        offset_to_flat_idx (jnp.ndarray):
            2D array mapping (x_idx, y_idx) -> flat_psfs index.
        kx (jnp.ndarray):
            FFT frequencies for x-axis.
        ky (jnp.ndarray):
            FFT frequencies for y-axis.
        n_pad (int):
            Number of padding pixels.
        x_symmetric (bool):
            Whether PSFs are symmetric about x=0.
        y_symmetric (bool):
            Whether PSFs are symmetric about y=0.
        input_type (str):
            Type of input data ("1d", "2dq", or "2df").
        max_offset (float):
            Maximum valid offset distance (-1 to disable bounds check).

    Returns:
        jnp.ndarray:
            Synthesized PSF at the requested (x, y) position.
    """
    # Neighbor lookup
    if input_type == "1d":
        _x, _y = convert_xy_1D(x, y)
        inds, offsets = get_near_inds_offsets_1D(x_offsets, y_offsets, _x, _y)
    else:
        _x, _y = convert_xy_2DQ(x, y)
        inds, offsets = get_near_inds_offsets_2D(x_offsets, y_offsets, _x, _y)

    # Use the index mapping to get flat_psfs indices from (x_idx, y_idx) pairs
    flat_indices = offset_to_flat_idx[inds[:, 0], inds[:, 1]]
    neighbors = flat_psfs[flat_indices]

    # We calculate weights based on the relative position of _x/_y
    # within the bounding box of the neighbors found.
    if input_type == "1d":
        # Neighbors are: [x_low, 0], [x_high, 0]
        x0 = offsets[0, 0]
        x1 = offsets[1, 0]

        # Span of the current grid cell
        span_x = x1 - x0

        # Avoid division by zero if query lands exactly on a node or grid is 1 point
        span_x = jnp.where(span_x == 0, 1.0, span_x)

        # Normalize distance (0.0 to 1.0)
        u = (_x - x0) / span_x

        # Linear weights: w0 = (1-u), w1 = u
        # shape (2,)
        weights = jnp.array([1.0 - u, u])

        # Safety: If span was 0, force weight to [1, 0]
        weights = jnp.where(x1 == x0, jnp.array([1.0, 0.0]), weights)

    else:  # 2D Case
        # Neighbors order from get_near_inds_offsets_2D is:
        # 0: (x0, y0), 1: (x0, y1), 2: (x1, y0), 3: (x1, y1)
        x0 = offsets[0, 0]
        x1 = offsets[3, 0]  # dim 0 is x
        y0 = offsets[0, 1]
        y1 = offsets[3, 1]  # dim 1 is y

        span_x = x1 - x0
        span_y = y1 - y0

        # Avoid div by zero
        span_x = jnp.where(span_x == 0, 1.0, span_x)
        span_y = jnp.where(span_y == 0, 1.0, span_y)

        # Normalize coordinates (0.0 to 1.0)
        u = (_x - x0) / span_x
        v = (_y - y0) / span_y

        # Bilinear Weights
        # w00 = (1-u)(1-v)
        # w01 = (1-u)v
        # w10 = u(1-v)
        # w11 = uv
        w0 = (1.0 - u) * (1.0 - v)
        w1 = (1.0 - u) * v
        w2 = u * (1.0 - v)
        w3 = u * v

        weights = jnp.array([w0, w1, w2, w3])

        # Safety for coincident nodes
        weights = jnp.where(
            (x1 == x0) & (y1 == y0), jnp.array([1.0, 0.0, 0.0, 0.0]), weights
        )

    # Symmetry logic
    eff_offsets_x = offsets[:, 0]
    eff_offsets_y = offsets[:, 1]

    if x_symmetric:
        eff_offsets_x = jnp.where(x < 0, -offsets[:, 0], offsets[:, 0])
        neighbors = lax.cond(
            x < 0, lambda p: jnp.flip(p, axis=2), lambda p: p, neighbors
        )
    if y_symmetric:
        eff_offsets_y = jnp.where(y < 0, -offsets[:, 1], offsets[:, 1])
        neighbors = lax.cond(
            y < 0, lambda p: jnp.flip(p, axis=1), lambda p: p, neighbors
        )

    # Calculate Shift (in pixels)
    shift_x = (x - eff_offsets_x) / pixel_scale
    shift_y = (y - eff_offsets_y) / pixel_scale

    # Pad (using consistent n_pad from init)
    # (K, H, W) -> (K, H_pad, W_pad)
    pad_width = ((0, 0), (n_pad, n_pad), (n_pad, n_pad))
    neighbors = jnp.pad(neighbors, pad_width)

    # Capture width for robust reconstruction
    _, _h_pad, w_pad = neighbors.shape

    # FFT X (Real-to-Complex, Axis 2)
    # Output: (K, H_pad, W_pad//2 + 1)
    spec = jnp.fft.rfft(neighbors, axis=2)

    # Apply X-Shift
    # kx: (W_freq,). shift_x: (K,).
    # Broadcast: (K, 1, W_freq)
    phasor_x = jnp.exp(-2j * jnp.pi * shift_x[:, None, None] * kx[None, None, :])
    spec = spec * phasor_x

    # FFT Y (Complex-to-Complex, Axis 1)
    spec = jnp.fft.fft(spec, axis=1)

    # Apply Y-Shift
    # ky: (H_pad,). shift_y: (K,).
    # Broadcast: (K, H_pad, 1)
    phasor_y = jnp.exp(-2j * jnp.pi * shift_y[:, None, None] * ky[None, :, None])
    spec = spec * phasor_y

    # Weighted Sum (Collapse Neighbors)
    # (K, H_freq, W_freq) -> (H_freq, W_freq)
    summed_spec = jnp.sum(spec * weights[:, None, None], axis=0)

    # IFFT Y
    res = jnp.fft.ifft(summed_spec, axis=0)

    # IFFT X (Complex-to-Real)
    # Pass n=w_pad to ensure we reconstruct to the exact even size
    res = jnp.fft.irfft(res, n=w_pad, axis=1)

    # Unpad
    h_orig = flat_psfs.shape[-2]
    psf = res[n_pad : n_pad + h_orig, n_pad : n_pad + h_orig]

    psf = jnp.maximum(psf, 0.0)

    # Mask if out of bounds
    # Only currently implemented for 1D (radial symmetry)
    # max_offset must be provided (> 0) to enable masking
    psf = lax.cond(
        (input_type == "1d") & (max_offset > 0),
        lambda p: jnp.where(jnp.sqrt(x**2 + y**2) > max_offset, 0.0, p),
        lambda p: p,
        psf,
    )

    return psf


def synthesize_psf_idw(
    x,
    y,
    pixel_scale,
    flat_psfs,
    flat_x_offsets,
    flat_y_offsets,
    kx,
    ky,
    n_pad,
    x_symmetric,
    y_symmetric,
    input_type,
    k_neighbors=4,
):
    """Synthesizes PSF using Inverse Distance Weighting (IDW) for irregular grids."""
    # Handle Symmetry (Map to 1st Quadrant if 2DQ)
    if input_type == "2dq":
        _x, _y = convert_xy_2DQ(x, y)
    elif input_type == "1d":
        _x, _y = convert_xy_1D(x, y)
    else:
        _x, _y = x, y

    # Compute Distances to ALL samples
    # flat_x_offsets shape: (N_samples,)
    d2 = (flat_x_offsets - _x) ** 2 + (flat_y_offsets - _y) ** 2
    dists = jnp.sqrt(d2)

    # Find K Nearest Neighbors
    # We use top_k on negative distance to find the smallest distances
    neg_dists, inds = lax.top_k(-dists, k_neighbors)
    near_dists = -neg_dists

    # Get Neighbors
    neighbors = flat_psfs[inds]

    # Calculate Weights (Inverse Distance)
    # Add epsilon to avoid divide by zero
    weights = 1.0 / (near_dists + 1e-10)
    weights = weights / jnp.sum(weights)

    # Apply Symmetry Flips to neighbors
    eff_offsets_x = flat_x_offsets[inds]
    eff_offsets_y = flat_y_offsets[inds]

    if x_symmetric:
        # Check against original x
        neighbors = lax.cond(
            x < 0, lambda p: jnp.flip(p, axis=2), lambda p: p, neighbors
        )
        eff_offsets_x = jnp.where(x < 0, -eff_offsets_x, eff_offsets_x)

    if y_symmetric:
        # Check against original y
        neighbors = lax.cond(
            y < 0, lambda p: jnp.flip(p, axis=1), lambda p: p, neighbors
        )
        eff_offsets_y = jnp.where(y < 0, -eff_offsets_y, eff_offsets_y)

    # Calculate Shift (in pixels)
    shift_x = (x - eff_offsets_x) / pixel_scale
    shift_y = (y - eff_offsets_y) / pixel_scale

    # Pad and FFT (Standard Shift Logic)
    pad_width = ((0, 0), (n_pad, n_pad), (n_pad, n_pad))
    neighbors = jnp.pad(neighbors, pad_width)

    # FFT X
    spec = jnp.fft.rfft(neighbors, axis=2)
    phasor_x = jnp.exp(-2j * jnp.pi * shift_x[:, None, None] * kx[None, None, :])
    spec = spec * phasor_x

    # FFT Y
    spec = jnp.fft.fft(spec, axis=1)
    phasor_y = jnp.exp(-2j * jnp.pi * shift_y[:, None, None] * ky[None, :, None])
    spec = spec * phasor_y

    # Weighted Sum
    summed_spec = jnp.sum(spec * weights[:, None, None], axis=0)

    # IFFT
    res = jnp.fft.ifft(summed_spec, axis=0)
    res = jnp.fft.irfft(res, n=neighbors.shape[-1], axis=1)

    # Unpad
    h_orig = flat_psfs.shape[-2]
    psf = res[n_pad : n_pad + h_orig, n_pad : n_pad + h_orig]

    return jnp.maximum(psf, 0.0)
