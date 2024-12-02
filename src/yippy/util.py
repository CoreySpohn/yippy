"""Utility functions for the yippy package."""

from functools import partial

import astropy.units as u
import jax.numpy as jnp
import numpy as np
from astropy.units import Quantity
from jax import lax
from lod_unit import lod, lod_eq


def convert_to_lod(
    x: Quantity, center_pix=None, pixel_scale=None, lam=None, D=None, dist=None
) -> Quantity:
    """Convert the x/y position to lambda/D.

    This function has the following assumptions on the x/y values provided:
        - If units are pixels, they follow the 00LL convention. As in the (0,0)
          point is the lower left corner of the image.
        - If the x/y values are in lambda/D, angular, or length units the
            (0,0) point is the center of the image, where the star is
            (hopefully) located.

    Args:
        x (astropy.units.Quantity):
            Position. Can be units of pixel, an angular unit (e.g. arcsec),
            or a length unit (e.g. AU)
        center_pix (astropy.units.Quantity):
            Center of the image in pixels (for the relevant axis)
        pixel_scale (astropy.units.Quantity):
            Pixel scale of in
        lam (astropy.units.Quantity):
            Wavelength of the observation
        D (astropy.units.Quantity):
            Diameter of the telescope
        dist (astropy.units.Quantity):
            Distance to the system
    """
    if x.unit == "pixel":
        assert (
            center_pix is not None
        ), "Center pixel must be provided to convert pixel to lod."
        assert (
            pixel_scale is not None
        ), "Pixel scale must be provided to convert pixel to lod."
        assert pixel_scale.unit == (
            lod / u.pix
        ), f"Pixel scale must be in units of lod/pix, not {pixel_scale.unit}."

        x = x - center_pix
        x = x * pixel_scale
        # Center the x position
    elif x.unit.physical_type == "angle":
        assert lam is not None, (
            f"Wavelength must be provided to convert {x.unit.physical_type}" f" to lod."
        )
        assert D is not None, (
            f"Telescope diameter must be provided to convert {x.unit.physical_type}"
            f" to lod."
        )
        x = x.to(lod, lod_eq(lam, D))
    elif x.unit.physical_type == "length":
        # If the distance to the system is not provided, raise an error
        assert dist is not None, (
            f"Distance to system must be provided to convert {x.unit.physical_type}"
            f" to {lod}."
        )
        x_angular = np.arctan(x.to(u.m).value / dist.to(u.m).value) * u.rad
        x = x_angular.to(lod, lod_eq(lam, D))
    else:
        raise ValueError(f"No conversion implemented for {x.unit.physical_type}")
    return x


def fft_shift(image, x=0, y=0):
    """Apply a Fourier shift to an image along the x and/or y axes.

    This function performs a 1D Fourier shift along the x-axis and/or y-axis
    of the input image. If a shift of 0 is specified for either axis, no
    operation is performed along that axis.

    The function uses the `fft_shift_1d` method to apply the shift in the
    Fourier domain, allowing for subpixel accuracy in the shift. The image
    is padded with zeros during the process, and the padding is removed before
    returning the shifted image.

    Args:
        image (numpy.ndarray):
            The input 2D image to be shifted.
        x (float):
            The number of pixels by which to shift the image along the x-axis.
        y (float):
            The number of pixels by which to shift the image along the y-axis.

    Returns:
        numpy.ndarray:
            The shifted image after applying the Fourier transform-based shift
            along the specified axes.
    """
    assert x != 0 or y != 0, "One of x or y must be non-zero."

    if x != 0:
        # Horizontal shift
        image = fft_shift_1d(image, x, axis=1)
    if y != 0:
        # Vertical shift
        image = fft_shift_1d(image, y, axis=0)

    return image


def fft_shift_1d(image, shift_pixels, axis):
    """Apply a Fourier shift to an image along a specified axis.

    This function pads the input image with zeros, performs a 1D Fourier
    transform along the specified axis, shifts the image by a specified
    number of pixels using a phasor, and reconstructs the image after
    applying the shift. The padded regions are removed before returning
    the shifted image.

    Args:
        image (numpy.ndarray):
            The input 2D image to be shifted.
        shift_pixels (float):
            The number of pixels by which to shift the image along the specified axis.
        axis (int):
            The axis to shift (0 for vertical, 1 for horizontal).

    Returns:
        numpy.ndarray:
            The shifted image after applying the Fourier transform and
            removing the padding.
    """
    n_pixels = image.shape[0]
    n_pad = int(1.5 * n_pixels)
    img_edge = n_pad + n_pixels

    # Pad the image with zeros
    padded = np.pad(image, n_pad, mode="constant")

    # Take the 1D Fourier transform along the specified axis
    padded = np.fft.fft(padded, axis=axis)

    # Get the frequencies used for the Fourier transform along the specified axis
    freqs = np.fft.fftfreq(4 * n_pixels)

    # Create the phasor
    phasor = np.exp(-2j * np.pi * freqs * shift_pixels)

    # Tile the phasor to match the dimensions of the padded image
    if axis == 1:
        phasor = np.tile(phasor, (padded.shape[0], 1))  # Horizontal shift (x-axis)
    else:
        phasor = np.tile(phasor, (padded.shape[1], 1)).T  # Vertical shift (y-axis)

    # Apply the phasor along the specified axis
    padded = padded * phasor

    # Reconstruct the image using the inverse Fourier transform along the specified axis
    padded = np.real(np.fft.ifft(padded, axis=axis))

    # Unpad the image to return to the original size
    image = padded[n_pad:img_edge, n_pad:img_edge]

    return image


def fft_rotate(image, rot_deg):
    """Rotate an image by a specified angle using Fourier-based shear operations.

    This function performs an image rotation by decomposing the rotation into
    three sequential shear operations in the Fourier domain. For more details
    see Larkin et al. (1997).

    Args:
        image (numpy.ndarray):
            The input image to be rotated.
        rot_deg (float):
            The rotation angle in degrees. Positive values rotate the image
            counterclockwise, and negative values rotate it clockwise.

    Returns:
        numpy.ndarray:
            The rotated image.
    """
    # To rotate counterclockwise, with the origin in the lower left, we use the
    # negative of the angle
    rot_deg = -rot_deg

    # Cut the angle to (-45, 45] and a number of 90-degree rotations
    rot_deg, n_rot = decompose_angle(rot_deg)

    image = rot90_helper(image, n_rot)

    if rot_deg != 0.0:
        theta = np.deg2rad(rot_deg)
        a = np.tan(theta / 2)
        b = -np.sin(theta)

        # Rotate using three shears
        # s_x
        image = fft_shear(image, a, axis=1)

        # s_yx
        image = fft_shear(image, b, axis=0)

        # s_xyx
        image = fft_shear(image, a, axis=1)

    return image


def fft_shear(image, shear_factor, axis):
    """Perform a shear operation in the Fourier domain.

    Args:
        image (numpy.ndarray):
            The input image to be sheared.
        shear_factor (float):
            The shear factor.
        axis (int):
            The axis to shear (0 for vertical, 1 for horizontal).

    Returns:
        numpy.ndarray:
            The sheared image with the zero padding removed.
    """
    # Calculate padding size based on the image dimensions
    n_pixels = image.shape[0]
    n_pad = int(1.5 * n_pixels)
    img_edge = n_pad + n_pixels

    # Pad the image with zeros
    padded = np.pad(image, n_pad, mode="constant")

    # Calculate the coordinate array for the padded image
    padded_height, padded_width = padded.shape
    center_y, center_x = (np.array(padded.shape) - 1) / 2
    grid_y, grid_x = np.mgrid[0:padded_height, 0:padded_width]

    # Array of distances from the center of the image along the shear axis
    if axis == 1:
        # Shearing along the horizontal axis
        dists = grid_x - center_x
    else:
        # Shearing along the vertical axis
        dists = grid_y - center_y

    # Determine the perpendicular axis to the shear direction
    perpendicular_axis = 1 - axis % 2

    # Compute the Fourier frequencies for the dimension perpendicular to the shear axis
    freqs = np.fft.fftfreq(dists.shape[perpendicular_axis])
    freqs = np.fft.fftshift(freqs)

    # Tile the shifted frequencies to match the dimensions of the padded image
    freqs = np.tile(freqs, (dists.shape[axis], 1))

    # Transpose the frequency array if the shear is applied along the horizontal axis
    if axis == 1:
        freqs = freqs.T

    # Shift the padded image to center the zero-frequency component
    padded = np.fft.fftshift(padded)

    # Apply the Fourier transform along the specified axis
    padded = np.fft.fft(padded, axis=axis)
    padded = np.fft.fftshift(padded)

    # Apply the phase shift (shear) in the Fourier domain
    padded = np.exp(-2j * np.pi * shear_factor * freqs * dists) * padded

    # Shift back and apply the inverse Fourier transform along the specified axis
    padded = np.fft.fftshift(padded)
    padded = np.fft.ifft(padded, axis=axis)
    padded = np.fft.fftshift(padded)

    # Unpad the image to return to the original size
    image = np.real(padded[n_pad:img_edge, n_pad:img_edge])

    return image


def rot90_helper(image, n_rot):
    """Rotate an image by 90 degrees a specified number of times.

    Ensures that the image is of odd dimensions before rotating to avoid
    incorrect centering.

    Args:
        image (numpy.ndarray):
            The input image to be rotated.
        n_rot (int):
            The number of 90-degree rotations to apply.

    Returns:
        numpy.ndarray:
            The rotated image.
    """
    # Add padding if the image has even dimensions before rotating
    # by 90 degrees to avoid incorrect centering
    needs_padding = not image.shape[0] % 2 or not image.shape[1] % 2
    if needs_padding:
        _img = np.zeros([image.shape[0] + 1, image.shape[1] + 1])
        _img[:-1, :-1] = image.copy()
    else:
        _img = image.copy()

    # Apply the rotation
    _img = np.rot90(_img, k=n_rot)

    # Remove padding if it was added
    if needs_padding:
        _img = _img[:-1, :-1]

    return _img


def fft_shift_jax(image, x=0, y=0):
    """Apply a Fourier shift to an image along the x and/or y axes using JAX.

    This is the JAX version of the `fft_shift` function.

    Args:
        image (jax.numpy.ndarray):
            The input 2D image to be shifted.
        x (float, optional):
            The number of pixels by which to shift the image along the x-axis.
        y (float, optional):
            The number of pixels by which to shift the image along the y-axis.

    Returns:
        jax.numpy.ndarray:
            The shifted image after applying the Fourier transform-based shift
            along the specified axes.
    """
    image = lax.cond(
        x != 0,
        lambda x: fft_shift_x(image, x),
        lambda x: image,
        x,
    )
    image = lax.cond(
        y != 0,
        lambda y: fft_shift_y(image, y),
        lambda y: image,
        y,
    )
    return image


def fft_shift_x(image, shift_pixels):
    """Apply a Fourier shift to an image along the x axis.

    A traceable version of the `fft_shift_1d` function that applies a Fourier
    shift to an image along the x-axis. This allows the function to be traced
    and compiled by JAX.

    Args:
        image (jax.numpy.ndarray):
            The input 2D image to be shifted.
        shift_pixels (float):
            The number of pixels by which to shift the image along the specified axis.

    Returns:
        jax.numpy.ndarray:
            The shifted image after applying the Fourier transform and removing
            the padding.
    """
    n_pixels = image.shape[0]
    n_pad = int(1.5 * n_pixels)
    img_edge = n_pad + n_pixels

    # Pad the image with zeros
    padded = jnp.pad(image, n_pad, mode="constant")

    # Get the frequencies used for the Fourier transform along the specified axis
    freqs = jnp.fft.fftfreq(4 * n_pixels)

    # Take the 1D Fourier transform along the specified axis
    padded = jnp.fft.fft(padded, axis=1)

    # Create the phasor
    phasor = jnp.exp(-2j * jnp.pi * freqs * shift_pixels)

    # Tile the phasor to match the dimensions of the padded image
    phasor = jnp.tile(phasor, (padded.shape[0], 1))  # Horizontal shift (x-axis)

    # Apply the phasor along the specified axis
    padded = padded * phasor

    # Reconstruct the image using the inverse Fourier transform along the specified axis
    padded = jnp.real(jnp.fft.ifft(padded, axis=1))

    # Unpad the image to return to the original size
    image = padded[n_pad:img_edge, n_pad:img_edge]

    return image


def fft_shift_y(image, shift_pixels):
    """Apply a Fourier shift to an image along the y axis.

    A traceable version of the `fft_shift_1d` function that applies a Fourier
    shift to an image along the y-axis. This allows the function to be traced
    and compiled by JAX.

    Args:
        image (jax.numpy.ndarray):
            The input 2D image to be shifted.
        shift_pixels (float):
            The number of pixels by which to shift the image along the specified axis.

    Returns:
        jax.numpy.ndarray:
            The shifted image after applying the Fourier transform and removing
            the padding.
    """
    n_pixels = image.shape[0]
    n_pad = int(1.5 * n_pixels)
    img_edge = n_pad + n_pixels

    # Pad the image with zeros
    padded = jnp.pad(image, n_pad, mode="constant")

    # Take the 1D Fourier transform along the specified axis
    padded = jnp.fft.fft(padded, axis=0)

    # Get the frequencies used for the Fourier transform along the specified axis
    freqs = jnp.fft.fftfreq(4 * n_pixels)

    # Create the phasor
    phasor = jnp.exp(-2j * jnp.pi * freqs * shift_pixels)

    # Tile the phasor to match the dimensions of the padded image
    # Vertical shift (y-axis)
    phasor = jnp.tile(phasor, (padded.shape[1], 1)).T

    # Apply the phasor along the specified axis
    padded = padded * phasor

    # Reconstruct the image using the inverse Fourier transform along the specified axis
    padded = jnp.real(jnp.fft.ifft(padded, axis=0))

    # Unpad the image to return to the original size
    image = padded[n_pad:img_edge, n_pad:img_edge]

    return image


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


def decompose_angle(angle):
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
    n_rot = int(angle // 90)

    # Cut the angle to [0, 90)
    adjusted_angle = angle % 90

    # Adjust the angle to the range (-45, 45]
    if adjusted_angle > 45:
        adjusted_angle -= 90
        n_rot += 1

    # if n_rot is 4, set it to 0 to avoid unnecessary rotations
    # this occurs when the angle is in the range (315, 360)
    if n_rot == 4:
        n_rot = 0

    return adjusted_angle, n_rot


def create_shift_mask(psf, shift_x, shift_y, fill_val=1):
    """Create a mask to identify valid pixels to average.

    This function is useful because when the PSF is shifted there are empty
    pixels, since they were outside the initial image, and should not be
    included in the final average.

    Args:
        psf (np.ndarray):
            The PSF image to shift.
        shift_x (float):
            The shift in the x direction.
        shift_y (float):
            The shift in the y direction.
        fill_val (float, optional):
            The value to fill the mask with.

    Returns:
        np.ndarray:
            The mask to identify valid pixels to average.
    """
    mask = np.full_like(psf, fill_val)

    # Handle x-direction shifting
    if shift_x > 0:
        # Zero out the left side
        mask[:, : int(np.ceil(shift_x))] = 0
    elif shift_x < 0:
        # Zero out the right side
        mask[:, int(np.floor(shift_x)) :] = 0

    # Handle y-direction shifting
    if shift_y > 0:
        # Zero out the bottom side
        mask[: int(np.ceil(shift_y)), :] = 0
    elif shift_y < 0:
        # Zero out the top side
        mask[int(np.floor(shift_y)) :, :] = 0

    return mask


def handle_shift_x_positive(mask, shift_x, x_grid):
    """Zero out the left side of the mask for positive x shifts."""
    n = jnp.ceil(shift_x).astype(jnp.int32)
    n = jnp.clip(n, 0, mask.shape[1])
    # Set mask to 0 where x_grid < n
    return jnp.where(x_grid < n, 0.0, mask)


def handle_shift_x_negative(mask, shift_x, x_grid):
    """Zero out the right side of the mask for negative x shifts."""
    n = jnp.ceil(-shift_x).astype(jnp.int32)
    n = jnp.clip(n, 0, mask.shape[1])
    # Set mask to 0 where x_grid >= (width - n)
    return jnp.where(x_grid >= (mask.shape[1] - n), 0.0, mask)


def handle_shift_y_positive(mask, shift_y, y_grid):
    """Zero out the top side of the mask for positive y shifts."""
    n = jnp.ceil(shift_y).astype(jnp.int32)
    n = jnp.clip(n, 0, mask.shape[0])
    # Set mask to 0 where y_grid < n
    return jnp.where(y_grid < n, 0.0, mask)


def handle_shift_y_negative(mask, shift_y, y_grid):
    """Zero out the bottom side of the mask for negative y shifts."""
    n = jnp.ceil(-shift_y).astype(jnp.int32)
    n = jnp.clip(n, 0, mask.shape[0])
    # Set mask to 0 where y_grid >= (height - n)
    return jnp.where(y_grid >= (mask.shape[0] - n), 0.0, mask)


def create_shift_mask_jax(psf, shift_x, shift_y, x_grid, y_grid, fill_val=1):
    """Create a mask to identify valid pixels to average.

    This function is useful because when the PSF is shifted there are empty
    pixels, since they were outside the initial image, and should not be
    included in the final average.

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
    mask = jnp.full_like(psf, fill_val)

    mask = lax.cond(
        jnp.sign(shift_x) == 1,
        # Zero out the left side
        lambda x: handle_shift_x_positive(x, shift_x, x_grid),
        # Zero out the right side
        lambda x: handle_shift_x_negative(x, shift_x, x_grid),
        mask,
    )

    mask = lax.cond(
        jnp.sign(shift_y) == 1,
        # Zero out the bottom side
        lambda x: handle_shift_y_positive(x, shift_y, y_grid),
        # Zero out the top side
        lambda x: handle_shift_y_negative(x, shift_y, y_grid),
        mask,
    )

    return mask


def fft_zoom(image, zoom_factor):
    """Zoom/rescale an image using Fourier interpolation.

    Args:
        image (np.ndarray):
            The input 2D image to be zoomed.
        zoom_factor (float):
            The factor by which to zoom the image.

    Returns:
        np.ndarray: The resampled image.
    """
    # Perform 2D Fourier transform on the input image
    f_transform = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image)))

    # Get the current shape of the image
    old_shape = np.array(image.shape)

    # Calculate the new shape based on the zoom factor
    new_shape = (np.array(image.shape) * zoom_factor).astype(int)

    # Initialize the output Fourier transform with zeros
    f_transform_new = np.zeros(new_shape, dtype=complex)

    # Determine the size of the region to copy from the old Fourier transform
    min_shape = np.minimum(old_shape, new_shape)

    # Define the start and end indices for copying the Fourier transform data
    start_old = (old_shape - min_shape) // 2
    end_old = start_old + min_shape

    start_new = (new_shape - min_shape) // 2
    end_new = start_new + min_shape

    # Copy the relevant portion of the old Fourier transform into the new one
    f_transform_new[start_new[0] : end_new[0], start_new[1] : end_new[1]] = f_transform[
        start_old[0] : end_old[0], start_old[1] : end_old[1]
    ]

    # Perform the inverse Fourier transform to get the zoomed image
    zoomed_image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(f_transform_new)))

    # Return the real part of the resampled image
    return np.real(zoomed_image)


def centered_fft_resample(image, image_pix_scale, new_shape, new_pix_scale):
    """Resample an image from one pixel scale to another using Fourier interpolation.

    This function adjusts the image to a new pixel scale and shape based on the
    given scales and desired output shape, while keeping the image centered.

    Args:
        image (np.ndarray):
            The single image to be resampled.
        image_pix_scale (float):
            The pixel scale of the input image (e.g., arcseconds per pixel).
        new_shape (tuple):
            The desired shape of the output image (height, width).
        new_pix_scale (float):
            The pixel scale of the output image (e.g., arcseconds per pixel).

    Returns:
        np.ndarray: The resampled single image.
    """
    # Calculate the zoom factor for resampling
    zoom_factor = image_pix_scale / new_pix_scale

    # Resample the image using Fourier interpolation
    scaled_image = fft_zoom(image, zoom_factor)

    # Calculate how much the resampled image needs to be cropped or padded
    # to match the desired detector shape
    scaled_shape = np.array(scaled_image.shape)
    new_shape = np.array(new_shape)
    center_offset = (scaled_shape - new_shape) // 2

    # Ensure symmetric padding/cropping
    pad_amount = np.abs(center_offset).astype(int)
    final_image = np.zeros(new_shape, dtype=scaled_image.dtype)

    # Apply padding or cropping based on center_offset
    if np.any(center_offset < 0):
        # Pad the image symmetrically
        padded_image = np.pad(
            scaled_image,
            ((pad_amount[0], pad_amount[0]), (pad_amount[1], pad_amount[1])),
            mode="constant",
        )
        final_image = padded_image[: new_shape[0], : new_shape[1]]
    else:
        # Crop the image symmetrically
        final_image = scaled_image[
            center_offset[0] : center_offset[0] + new_shape[0],
            center_offset[1] : center_offset[1] + new_shape[1],
        ]

    return final_image


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
    weights = jnp.exp(-(near_diffs**2) / (2 * sigma**2))
    weights /= weights.sum()

    # Manually shift each PSF
    shifted_psf1, mask1 = shift_and_mask(
        near_psfs[0], near_shifts[0, 0], near_shifts[0, 1], weights[0], x_grid, y_grid
    )
    shifted_psf2, mask2 = shift_and_mask(
        near_psfs[1], near_shifts[1, 0], near_shifts[1, 1], weights[1], x_grid, y_grid
    )

    # Accumulate the weighted PSFs and the weight masks
    psf = shifted_psf1 + shifted_psf2
    weight_array = mask1 + mask2

    # Normalize the PSF
    safe_reciprocal = jnp.where(weight_array != 0, 1.0 / weight_array, 0.0)
    psf = psf * safe_reciprocal
    # temp1 = psf / mask1
    # temp2 = psf / mask2
    # psf = psf / weight_array

    return psf


def create_psf_1D_no_symmetry(
    x: float,
    y: float,
    pixel_scale: float,
    x_offsets: jnp.ndarray,
    y_offsets: jnp.ndarray,
    x_grid: jnp.ndarray,
    y_grid: jnp.ndarray,
    reshaped_psfs: jnp.ndarray,
):
    """Create the radially symmetric PSF with no symmetry."""
    psf = create_avg_psf_1D(
        x, y, pixel_scale, x_offsets, y_offsets, x_grid, y_grid, reshaped_psfs
    )
    _x, _y = convert_xy_1D(x, y)
    psf = x_basic_shift(x, _x, psf, pixel_scale)
    psf = y_basic_shift(y, _y, psf, pixel_scale)
    return psf


def create_psf_1D_x_symmetry(
    x: float,
    y: float,
    pixel_scale: float,
    x_offsets: jnp.ndarray,
    y_offsets: jnp.ndarray,
    x_grid: jnp.ndarray,
    y_grid: jnp.ndarray,
    reshaped_psfs: jnp.ndarray,
):
    """Create the radially symmetric PSF with x symmetry."""
    psf = create_avg_psf_1D(
        x, y, pixel_scale, x_offsets, y_offsets, x_grid, y_grid, reshaped_psfs
    )
    _x, _y = convert_xy_1D(x, y)
    psf = x_symmetric_shift(x, _x, psf, pixel_scale)
    psf = y_basic_shift(y, _y, psf, pixel_scale)
    return psf


def create_psf_1D_y_symmetry(
    x: float,
    y: float,
    pixel_scale: float,
    x_offsets: jnp.ndarray,
    y_offsets: jnp.ndarray,
    x_grid: jnp.ndarray,
    y_grid: jnp.ndarray,
    reshaped_psfs: jnp.ndarray,
):
    """Create the radially symmetric PSF with y symmetry."""
    psf = create_avg_psf_1D(
        x, y, pixel_scale, x_offsets, y_offsets, x_grid, y_grid, reshaped_psfs
    )
    _x, _y = convert_xy_1D(x, y)
    psf = x_basic_shift(x, _x, psf, pixel_scale)
    psf = y_symmetric_shift(y, _y, psf, pixel_scale)
    return psf


def create_psf_1D_xy_symmetry(
    x: float,
    y: float,
    pixel_scale: float,
    x_offsets: jnp.ndarray,
    y_offsets: jnp.ndarray,
    x_grid: jnp.ndarray,
    y_grid: jnp.ndarray,
    reshaped_psfs: jnp.ndarray,
):
    """Create the radially symmetric PSF with x and y symmetry."""
    psf = create_avg_psf_1D(
        x, y, pixel_scale, x_offsets, y_offsets, x_grid, y_grid, reshaped_psfs
    )
    _x, _y = convert_xy_1D(x, y)
    psf = x_symmetric_shift(x, _x, psf, pixel_scale)
    psf = y_symmetric_shift(y, _y, psf, pixel_scale)
    return psf


def create_avg_psf_2DQ(
    x: float,
    y: float,
    pixel_scale: float,
    x_offsets: jnp.ndarray,
    y_offsets: jnp.ndarray,
    x_grid: jnp.ndarray,
    y_grid: jnp.ndarray,
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
    weights = jnp.exp(-(near_diffs**2) / (2 * sigma**2))
    weights /= weights.sum()

    # Manually shift each PSF
    shifted_psf1, mask1 = shift_and_mask(
        near_psfs[0], near_shifts[0, 0], near_shifts[0, 1], weights[0], x_grid, y_grid
    )
    shifted_psf2, mask2 = shift_and_mask(
        near_psfs[1], near_shifts[1, 0], near_shifts[1, 1], weights[1], x_grid, y_grid
    )
    shifted_psf3, mask3 = shift_and_mask(
        near_psfs[2], near_shifts[2, 0], near_shifts[2, 1], weights[2], x_grid, y_grid
    )
    shifted_psf4, mask4 = shift_and_mask(
        near_psfs[3], near_shifts[3, 0], near_shifts[3, 1], weights[3], x_grid, y_grid
    )

    # Accumulate the weighted PSFs and the weight masks
    psf = shifted_psf1 + shifted_psf2 + shifted_psf3 + shifted_psf4
    weight_array = mask1 + mask2 + mask3 + mask4

    # Normalize the PSF
    psf = psf / weight_array

    return psf


def create_psf_2DQ_no_symmetry(
    x: float,
    y: float,
    pixel_scale: float,
    x_offsets: jnp.ndarray,
    y_offsets: jnp.ndarray,
    x_grid: jnp.ndarray,
    y_grid: jnp.ndarray,
    reshaped_psfs: jnp.ndarray,
):
    """Creates the 2D quarter symmetric PSF without any symmetry."""
    psf = create_avg_psf_2DQ(
        x, y, pixel_scale, x_offsets, y_offsets, x_grid, y_grid, reshaped_psfs
    )
    _x, _y = convert_xy_2DQ(x, y)
    psf = x_basic_shift(x, _x, psf, pixel_scale)
    psf = y_basic_shift(y, _y, psf, pixel_scale)
    return psf


def create_psf_2DQ_x_symmetry(
    x: float,
    y: float,
    pixel_scale: float,
    x_offsets: jnp.ndarray,
    y_offsets: jnp.ndarray,
    x_grid: jnp.ndarray,
    y_grid: jnp.ndarray,
    reshaped_psfs: jnp.ndarray,
):
    """Create a 2D quarter symmetric PSF with x symmetry."""
    psf = create_avg_psf_2DQ(
        x, y, pixel_scale, x_offsets, y_offsets, x_grid, y_grid, reshaped_psfs
    )
    _x, _y = convert_xy_2DQ(x, y)
    psf = x_symmetric_shift(x, _x, psf, pixel_scale)
    psf = y_basic_shift(y, _y, psf, pixel_scale)
    return psf


def create_psf_2DQ_y_symmetry(
    x: float,
    y: float,
    pixel_scale: float,
    x_offsets: jnp.ndarray,
    y_offsets: jnp.ndarray,
    x_grid: jnp.ndarray,
    y_grid: jnp.ndarray,
    reshaped_psfs: jnp.ndarray,
):
    """Create a 2D quarter symmetric PSF with y symmetry."""
    psf = create_avg_psf_2DQ(
        x, y, pixel_scale, x_offsets, y_offsets, x_grid, y_grid, reshaped_psfs
    )
    _x, _y = convert_xy_2DQ(x, y)
    psf = x_basic_shift(x, _x, psf, pixel_scale)
    psf = y_symmetric_shift(y, _y, psf, pixel_scale)
    return psf


def create_psf_2DQ_xy_symmetry(
    x: float,
    y: float,
    pixel_scale: float,
    x_offsets: jnp.ndarray,
    y_offsets: jnp.ndarray,
    x_grid: jnp.ndarray,
    y_grid: jnp.ndarray,
    reshaped_psfs: jnp.ndarray,
):
    """Creates the PSF at the specified off-axis position with x and y symmetry."""
    psf = create_avg_psf_2DQ(
        x, y, pixel_scale, x_offsets, y_offsets, x_grid, y_grid, reshaped_psfs
    )
    _x, _y = convert_xy_2DQ(x, y)
    psf = x_symmetric_shift(x, _x, psf, pixel_scale)
    psf = y_symmetric_shift(y, _y, psf, pixel_scale)
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


def shift_and_mask(near_psf, shift_x, shift_y, weight, x_grid, y_grid):
    """Shifts the PSF in x and y directions and applies a weight mask.

    Args:
        near_psf (jax.numpy.ndarray): The PSF image to shift.
        shift_x (float): Shift in the x-direction.
        shift_y (float): Shift in the y-direction.
        weight (float): Weight for the PSF.
        x_grid (jax.numpy.ndarray): The x-coordinate grid.
        y_grid (jax.numpy.ndarray): The y-coordinate grid.

    Returns:
        Tuple[jax.numpy.ndarray, jax.numpy.ndarray]:
            - Weighted shifted PSF.
            - Weight mask.
    """
    # Shift the PSF in x and y directions
    shifted_psf = fft_shift_x(near_psf, shift_x)
    shifted_psf = fft_shift_y(shifted_psf, shift_y)

    # Create the weight mask
    weight_mask = create_shift_mask_jax(
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


def x_symmetric_shift(input_val, converted_val, PSF, pixel_scale):
    """Shifts the PSF to the specified position assuming symmetry about x=0."""
    flip = jnp.sign(input_val) == -1
    # Conditional flipping
    _PSF = lax.cond(
        flip,
        lambda _: jnp.fliplr(PSF),
        lambda _: PSF,
        operand=None,
    )
    # Conditional shift calculation
    shift = lax.cond(
        flip,
        lambda _: (input_val + converted_val) / pixel_scale,
        lambda _: (input_val - converted_val) / pixel_scale,
        operand=None,
    )
    # Perform shift
    # shifted_psf = fft_shift_x(_PSF, shift)
    return fft_shift_x(_PSF, shift)


def x_basic_shift(input_val, converted_val, PSF, pixel_scale):
    """Shifts the PSF to the specified x position."""
    shift = (input_val - converted_val) / pixel_scale
    return fft_shift_x(PSF, shift)


def y_symmetric_shift(input_val, converted_val, PSF, pixel_scale):
    """Shifts the PSF to the specified position assuming symmetry about x=0."""
    return lax.cond(
        jnp.sign(input_val) == -1,
        lambda _: fft_shift_y(
            jnp.flipud(PSF), (input_val + converted_val) / pixel_scale
        ),
        lambda _: fft_shift_y(PSF, (input_val - converted_val) / pixel_scale),
        None,
    )


def y_basic_shift(input_val, converted_val, PSF, pixel_scale):
    """Shifts the PSF to the specified y position."""
    shift = (input_val - converted_val) / pixel_scale
    return fft_shift_y(PSF, shift)
