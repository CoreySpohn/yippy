"""Utility functions for the yippy package."""

import astropy.units as u
import numpy as np
from astropy.units import Quantity
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


def fft_shift_2d(image, x=0, y=0):
    """Apply a 2D Fourier shift to an image along both the x and y axes.

    This function performs a 2D Fourier shift, allowing for subpixel accuracy
    in both x and y directions simultaneously. The image is transformed to the
    frequency domain, shifted by applying a phasor, and then transformed back
    to the spatial domain.

    Args:
        image (numpy.ndarray):
            The input 2D image to be shifted.
        x (float):
            The number of pixels by which to shift the image along the x-axis.
        y (float):
            The number of pixels by which to shift the image along the y-axis.

    Returns:
        numpy.ndarray:
            The shifted image after applying the 2D Fourier transform-based shift.
    """
    n_pixels = image.shape[0]
    n_pad = int(1.5 * n_pixels)
    img_edge = n_pad + n_pixels

    # Pad the image with zeros
    image = np.pad(image, n_pad, mode="constant")

    # Compute the 2D Fourier transform of the image
    image_ft = np.fft.fft2(image)

    # Get the shape of the image
    n_rows, n_cols = image.shape

    # Create frequency grids for both axes
    ky = np.fft.fftfreq(n_rows)
    kx = np.fft.fftfreq(n_cols)

    # Create meshgrids for frequencies
    # Kx, Ky = np.meshgrid(kx, ky)

    # Create the combined phasor for both shifts
    # phasor = np.exp(-2j * np.pi * (Kx * x + Ky * y))
    # Create 1D phasors
    exp_kx = np.exp(-2j * np.pi * kx * x)
    exp_ky = np.exp(-2j * np.pi * ky * y)
    # Compute outer product for phasor without full meshgrid
    phasor = np.outer(exp_ky, exp_kx)

    # Apply the phasor to the Fourier transformed image
    image_ft_shifted = image_ft * phasor

    # Compute the inverse 2D Fourier transform
    shifted_image = np.fft.ifft2(image_ft_shifted)

    # Return the real part of the shifted image
    shifted_image = np.real(shifted_image)

    return shifted_image[n_pad:img_edge, n_pad:img_edge]


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
