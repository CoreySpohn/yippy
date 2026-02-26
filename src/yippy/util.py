"""Utility functions for the yippy package."""

from pathlib import Path

import astropy.io.fits as fits
import astropy.units as u
import jax.numpy as jnp
import numpy as np
from astropy.units import Quantity
from hwoutils.transforms import resample_flux
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
        assert center_pix is not None, (
            "Center pixel must be provided to convert pixel to lod."
        )
        assert pixel_scale is not None, (
            "Pixel scale must be provided to convert pixel to lod."
        )
        assert pixel_scale.unit == (lod / u.pix), (
            f"Pixel scale must be in units of lod/pix, not {pixel_scale.unit}."
        )

        x = x - center_pix
        x = x * pixel_scale
        # Center the x position
    elif x.unit.physical_type == "angle":
        assert lam is not None, (
            f"Wavelength must be provided to convert {x.unit.physical_type} to lod."
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


def convert_to_pix(
    x: Quantity, center_pix, pixel_scale, lam=None, D=None, dist=None
) -> Quantity:
    """Convert the x/y position from lambda/D to pixel units.

    This function has the following assumptions on the x/y values provided:
        - If the desired output is in pixels, the (0,0) point is the lower left
          corner of the image.
        - If the x/y values are in lambda/D, angular, or length units, the
          (0,0) point is the center of the image, where the star is
          (hopefully) located.

    Args:
        x (astropy.units.Quantity):
            Position to convert. Should be in units of lambda/D, an angular unit
            (e.g., arcsec), or a length unit (e.g., AU).
        center_pix (astropy.units.Quantity, optional):
            Center of the image in pixels (for the relevant axis). Required if
            converting to pixel units.
        pixel_scale (astropy.units.Quantity, optional):
            Pixel scale in units of lambda/D per pixel. Required if converting to
            pixel units.
        lam (astropy.units.Quantity, optional):
            Wavelength of the observation. Required if converting from angular or
            length units to pixel units.
        D (astropy.units.Quantity, optional):
            Diameter of the telescope. Required if converting from angular or
            length units to pixel units.
        dist (astropy.units.Quantity, optional):
            Distance to the system. Required if converting from length units to
            pixel units.

    Returns:
        astropy.units.Quantity:
            Position in pixel units.

    Raises:
        AssertionError:
            If required parameters for the conversion are not provided or have
            incorrect units.
        ValueError:
            If the input unit type is not supported for conversion.
    """
    if isinstance(x, float) or isinstance(x, np.floating):
        # Assume x is a float in lambda/D
        x_pixels = x * lod / pixel_scale + center_pix
    elif x.unit == lod:
        # Center the x position
        x_pixels = x / pixel_scale + center_pix

    elif x.unit.physical_type == "angle":
        # Conversion from angle to pixels
        assert lam is not None, (
            "Wavelength must be provided to convert angle to pixels."
        )
        assert D is not None, (
            "Telescope diameter must be provided to convert angle to pixels."
        )

        # Convert angle to lambda/D
        x_lod = x.to(u.rad, lod_eq(lam, D))
        # Now convert lambda/D to pixels
        x_pixels = x_lod / pixel_scale + center_pix

    elif x.unit.physical_type == "length":
        # Conversion from length to pixels
        assert lam is not None, (
            "Wavelength must be provided to convert length to pixels."
        )
        assert D is not None, (
            "Telescope diameter must be provided to convert length to pixels."
        )
        assert dist is not None, (
            "Distance to system must be provided to convert length to pixels."
        )

        # Convert length to angle
        x_angle = np.arctan(x.to(u.m).value / dist.to(u.m).value) * u.rad
        # Convert angle to lambda/D
        x_lod = x_angle.to(lod, lod_eq(lam, D))
        # Now convert lambda/D to pixels
        x_pixels = x_lod / pixel_scale + center_pix

    else:
        raise ValueError(f"No conversion implemented for {x.unit.physical_type}")

    return x_pixels


def save_coro_performance_to_fits(
    sep: np.ndarray,
    throughput: np.ndarray,
    raw_contrast: np.ndarray,
    filename: str,
    outdir: Path,
    overwrite=True,
):
    """Save coronagraph performance (throughput, raw_contrast) to a FITS file."""
    sort_idx = np.argsort(sep)
    col_sep = fits.Column(name="separation_lamD", format="E", array=sep[sort_idx])
    col_thr = fits.Column(name="throughput", format="E", array=throughput[sort_idx])
    col_con = fits.Column(name="raw_contrast", format="E", array=raw_contrast[sort_idx])
    cols = fits.ColDefs([col_sep, col_thr, col_con])
    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.name = "CORO_PERFORMANCE"

    primary_hdu = fits.PrimaryHDU()
    hdul = fits.HDUList([primary_hdu, tbhdu])
    outpath = outdir / filename
    hdul.writeto(outpath, overwrite=overwrite)


def load_coro_performance_from_fits(filename: str, indir: Path):
    """Load (separation, throughput, raw_contrast) from a FITS file."""
    inpath = indir / filename
    with fits.open(inpath) as hdul:
        data = hdul["CORO_PERFORMANCE"].data
        sep = data["separation_lamD"]
        thr = data["throughput"]
        con = data["raw_contrast"]
    return sep, thr, con


def extract_and_oversample_subarray(
    psf_img: np.ndarray,
    center_x: float,
    center_y: float,
    radius_pix: float,
    oversample: int,
):
    """Get oversampled subarray of the PSF image around a given center.

    Extract a subarray of `psf_img` around (center_x, center_y),
    then oversample that subarray by the specified factor.

    Args:
        psf_img (np.ndarray):
            The input PSF image
        center_x (float):
            Position of the center in the x direction
        center_y (float):
            Position of the center in the y direction
        radius_pix (float):
            The radius of the subarray in pixels
        oversample (int):
            The oversampling factor

    Returns:
        subarr_oversamp (np.ndarray):
            The oversampled subarray
        center_x_os (float):
            center_x in oversampled subarray coords
        center_y_os (float):
            center_y in oversampled subarray coords
        radius_os (float):
            radius_pix * oversample
        subarr (np.ndarray):
            the original subarray (for flux renormalization)
    """
    ny, nx = psf_img.shape
    margin = int(np.ceil(radius_pix * 3))
    xmin = max(0, int(np.floor(center_x - margin)))
    xmax = min(nx - 1, int(np.ceil(center_x + margin)))
    ymin = max(0, int(np.floor(center_y - margin)))
    ymax = min(ny - 1, int(np.ceil(center_y + margin)))

    subarr = psf_img[ymin : ymax + 1, xmin : xmax + 1]

    # Flux-conserving oversample using resample_flux
    ny_os = subarr.shape[0] * oversample
    nx_os = subarr.shape[1] * oversample
    subarr_oversamp = np.asarray(
        resample_flux(
            jnp.asarray(np.asarray(subarr, dtype=np.float64)),
            1.0,
            1.0 / oversample,
            (ny_os, nx_os),
        )
    )

    center_x_os = (center_x - xmin) * oversample
    center_y_os = (center_y - ymin) * oversample
    radius_os = radius_pix * oversample

    return subarr_oversamp, center_x_os, center_y_os, radius_os, subarr


def measure_flux_in_oversampled_aperture(
    subarr_oversamp: np.ndarray,
    center_x_os: float,
    center_y_os: float,
    radius_os: float,
    subarr_original: np.ndarray,
) -> float:
    """Get flux in a circular aperture of radius `radius_os` in the oversampled array.

    Returns:
        flux_in_ap (float): total flux inside the circular mask
    """
    yy_os, xx_os = np.indices(subarr_oversamp.shape)
    rr_os = np.sqrt((xx_os - center_x_os) ** 2 + (yy_os - center_y_os) ** 2)
    ap_mask = rr_os <= radius_os

    flux_in_ap = subarr_oversamp[ap_mask].sum()

    return flux_in_ap


def crop_around_peak(arr, radius):
    """Crop a 2D array to a square region centered on the peak pixel.

    The output is always square with side length ``2 * r`` where ``r``
    is the largest feasible radius that fits within the array bounds
    (capped at the requested *radius*). This function is mostly used
    in the documentation animations.

    Args:
        arr (np.ndarray): 2D input array.
        radius (int): Desired half-width of the output crop in pixels.

    Returns:
        np.ndarray: Square cropped subarray centered on the peak.
    """
    peak_y, peak_x = np.unravel_index(arr.argmax(), arr.shape)
    ny, nx = arr.shape
    # Feasible half-widths in each direction from the peak
    r = min(radius, peak_y, ny - peak_y, peak_x, nx - peak_x)
    return arr[peak_y - r : peak_y + r, peak_x - r : peak_x + r]


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
