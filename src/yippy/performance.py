"""Coronagraph performance metric computation.

This module provides standalone functions for computing coronagraph performance
curves: throughput, raw contrast, core area, occulter transmission, and core
mean intensity. Each metric has its own compute function for clarity, plus a
``compute_all_performance_curves`` orchestrator.

All functions operate on a Coronagraph instance passed as the first argument,
keeping the Coronagraph class itself slim while preserving backward-compatible
delegation methods on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from hwoutils.constants import GAUSSIAN_FWHM_FACTOR
from hwoutils.map_coordinates import map_coordinates
from hwoutils.radial import radial_profile
from hwoutils.transforms import resample_flux
from lod_unit import lod
from scipy.interpolate import make_interp_spline
from scipy.optimize import root_scalar

from .logger import logger
from .util import (
    convert_to_pix,
    extract_and_oversample_subarray,
    load_coro_performance_from_fits,
    measure_flux_in_oversampled_aperture,
    save_coro_performance_to_fits,
)

if TYPE_CHECKING:
    from .coronagraph import Coronagraph


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class OffAxisPosition:
    """Data for a single off-axis PSF position along the performance axis."""

    separation: float  # |x| in lam/D
    psf: np.ndarray  # the off-axis PSF image
    px: int  # pixel x position in the image
    py: int  # pixel y position in the image


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _iter_xaxis_positions(coro: Coronagraph):
    """Yield :class:`OffAxisPosition` for each valid x-offset (y ~= 0).

    Iterates over each valid x-offset position along the performance-curve
    axis. All native YIP offsets are included, even those where the nominal
    point-source location lies outside the image, because the truncation-ratio
    throughput remains meaningful as long as the translated PSF retains
    significant in-image power.
    """
    y_offsets = np.array(coro.offax.y_offsets)
    y_idx = int(np.argmin(np.abs(y_offsets)))
    y_lod_val = y_offsets[y_idx]

    if y_lod_val != 0:
        logger.warning(f"No PSF at y=0, using closest y offset: {y_lod_val:.3f} lam/D")

    for i, x_lod_val in enumerate(np.array(coro.offax.x_offsets)):
        if x_lod_val < 0:
            continue
        r = abs(x_lod_val)
        planet_psf = coro.offax.get_psf_by_offset_idx(i, y_idx)
        if planet_psf is None:
            continue
        px = convert_to_pix(
            x_lod_val, coro.offax.center_x, coro.pixel_scale_arcsec
        ).value.astype(int)
        py = convert_to_pix(
            y_lod_val, coro.offax.center_y, coro.pixel_scale_arcsec
        ).value.astype(int)
        yield OffAxisPosition(separation=r, psf=planet_psf, px=px, py=py)


def _collect_and_sort(separations: list, values: list) -> tuple[np.ndarray, np.ndarray]:
    """Convert lists to sorted arrays by separation."""
    sep = np.array(separations)
    val = np.array(values)
    order = np.argsort(sep)
    return sep[order], val[order]


def _point_source_in_image(pos: OffAxisPosition) -> bool:
    """Return True if the nominal point-source pixel lies inside the PSF image."""
    ny, nx = pos.psf.shape
    return 0 <= pos.px < nx and 0 <= pos.py < ny


def _oversample_psf(
    psf: np.ndarray, pixel_scale_arcsec: float, oversample: int
) -> np.ndarray:
    """Oversample a PSF using area-scaled cubic interpolation.

    Uses ``resample_flux`` from hwoutils which converts to surface brightness
    before interpolation and back to flux units afterward. This is smooth
    center sampling, not exact target-pixel integration. Aperture estimates
    require an oversampling convergence check; clipping can also change flux.

    Args:
        psf: 2D PSF image.
        pixel_scale_arcsec: Pixel scale of the input PSF (lam/D per pixel).
        oversample: Oversampling factor.

    Returns:
        Oversampled PSF in target-pixel flux units, with negative values clamped.
    """
    os_pix = pixel_scale_arcsec / oversample
    ny_os = psf.shape[0] * oversample
    nx_os = psf.shape[1] * oversample
    psf_os = np.asarray(
        resample_flux(
            jnp.asarray(np.asarray(psf, dtype=np.float64)),
            pixel_scale_arcsec,
            os_pix,
            (ny_os, nx_os),
        )
    )
    return np.maximum(psf_os, 0.0)


def resolve_oversample(coro: Coronagraph, oversample: int | None) -> int:
    """Return the effective oversampling factor for a calculation.

    ``None`` selects AYO's rule, ``ceil(pixscale / 0.05)`` for the realized
    pixel scale in lambda/D; any other value is used as given.
    """
    if oversample is None:
        return int(np.ceil(coro.pixel_scale_arcsec.value / 0.05))
    return int(oversample)


def _threshold_mask(psf_os: np.ndarray, trunc_ratio: float) -> np.ndarray:
    """Create boolean mask of pixels exceeding ``trunc_ratio * peak``.

    Falls back to peak-only mask if no pixels exceed the threshold.
    """
    peak = psf_os.max()
    mask = psf_os > trunc_ratio * peak
    if not mask.any():
        mask = psf_os == peak
    return mask


def _compute_iwa_owa(
    coro: Coronagraph, sep: np.ndarray, throughput: np.ndarray
) -> None:
    """Compute IWA and OWA from throughput curve, storing on *coro*."""
    valid_mask = throughput > 0
    half_max_throughput = max(throughput[valid_mask]) / 2

    def iwa_func(x):
        return coro.throughput_interp(x) - half_max_throughput

    iwa_bracket = None
    for i in range(1, len(sep)):
        if iwa_func(sep[i - 1]) * iwa_func(sep[i]) < 0:
            iwa_bracket = [sep[i - 1], sep[i]]
            break

    if iwa_bracket is not None:
        coro.IWA = root_scalar(iwa_func, bracket=iwa_bracket).root * lod
    else:
        first_valid_idx = np.argmax(valid_mask)
        coro.IWA = sep[first_valid_idx] * lod
        logger.warning(f"Could not find IWA bracket, using first valid sep: {coro.IWA}")

    if hasattr(coro.offax, "max_offset_in_image"):
        coro.OWA = coro.offax.max_offset_in_image
        logger.info(
            f"OWA set to max_offset_in_image: {coro.OWA.to(lod).value:.2f} lam/D"
        )
    else:
        coro.OWA = np.max(sep) * lod
        logger.warning(
            "max_offset_in_image not available, using maximum separation as OWA"
        )


# ---------------------------------------------------------------------------
# Public utility
# ---------------------------------------------------------------------------


def compute_radial_average(
    image: np.ndarray,
    pixel_scale_value: float,
    center: tuple[float, float] | None = None,
    nbins: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute radial average of a 2D image.

    Thin wrapper around :func:`hwoutils.radial.radial_profile` for
    backward compatibility.

    Args:
        image: 2D numpy array.
        pixel_scale_value: Pixel scale in lambda/D per pixel (float).
        center: ``(cy, cx)`` pixel coordinates of center. Defaults to
            image center.
        nbins: Number of radial bins. Defaults to ``floor(max_dim / 2)``.

    Returns:
        ``(separations_lod, radial_profile)`` - 1-D arrays.
    """
    kwargs = {}
    if center is not None:
        # Incoming convention is [x, y]; radial_profile expects (cy, cx)
        if isinstance(center, list):
            kwargs["center"] = (center[1], center[0])
        else:
            kwargs["center"] = center
    if nbins is not None:
        kwargs["nbins"] = nbins
    seps, prof = radial_profile(
        jnp.asarray(np.asarray(image, dtype=np.float64)),
        pixel_scale_arcsec=pixel_scale_value,
        **kwargs,
    )
    return np.asarray(seps), np.asarray(prof)


def plot_performance_curve(
    x, y, title, xlabel, ylabel, marker="o-", log_scale=False, ms=4
):
    """Plot a single performance curve."""
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(x, y, marker, ms=ms)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if log_scale:
        plt.yscale("log")
    plt.grid(True)
    plt.show()


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------


def compute_throughput_curve(
    coro: Coronagraph,
    aperture_radius_lod: float = 0.7,
    oversample: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute coronagraph throughput vs separation.

    Throughput is the fraction of the total flux (planet PSF normalised to 1)
    that lands inside a photometric aperture at each off-axis position.

    Args:
        coro: Coronagraph instance.
        aperture_radius_lod: Aperture radius in lam/D.
        oversample: Oversampling factor.

    Returns:
        ``(separations, throughputs)`` - sorted 1-D arrays.
    """
    separations, throughputs = [], []
    radius_pix = aperture_radius_lod / coro.pixel_scale_arcsec.value

    for pos in _iter_xaxis_positions(coro):
        if not _point_source_in_image(pos):
            separations.append(pos.separation)
            throughputs.append(0.0)
            continue
        sub_os, px_os, py_os, r_os, sub_orig = extract_and_oversample_subarray(
            pos.psf, pos.px, pos.py, radius_pix, oversample
        )
        # Center aperture on PSF peak to match AYO's behavior
        peak_ij = np.unravel_index(sub_os.argmax(), sub_os.shape)
        py_os, px_os = float(peak_ij[0]), float(peak_ij[1])
        flux = measure_flux_in_oversampled_aperture(
            sub_os, px_os, py_os, r_os, sub_orig
        )
        separations.append(pos.separation)
        throughputs.append(flux)

    return _collect_and_sort(separations, throughputs)


def compute_raw_contrast_curve(
    coro: Coronagraph,
    stellar_diam=0.0 * lod,
    aperture_radius_lod: float = 0.7,
    oversample: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute raw contrast curve vs separation.

    Contrast is the ratio of stellar flux to planet flux within the same
    photometric aperture at each off-axis position.

    Args:
        coro: Coronagraph instance.
        stellar_diam: Stellar angular diameter for the on-axis PSF.
        aperture_radius_lod: Aperture radius in lam/D.
        oversample: Oversampling factor.

    Returns:
        ``(separations, contrasts)`` - sorted 1-D arrays.
    """
    star_psf = coro.stellar_intens(stellar_diam)
    separations, contrasts = [], []
    radius_pix = aperture_radius_lod / coro.pixel_scale_arcsec.value

    for pos in _iter_xaxis_positions(coro):
        if not _point_source_in_image(pos):
            # Raw contrast is undefined outside the image. Use a large finite
            # sentinel so downstream spline construction stays well-defined;
            # the point lies beyond OWA (max_offset_in_image) so it will not
            # be queried in physically meaningful regimes.
            separations.append(pos.separation)
            contrasts.append(1e20)
            continue
        # Planet flux
        sub_os_p, px_os_p, py_os_p, r_os_p, sub_orig_p = (
            extract_and_oversample_subarray(
                pos.psf, pos.px, pos.py, radius_pix, oversample
            )
        )
        planet_flux = measure_flux_in_oversampled_aperture(
            sub_os_p, px_os_p, py_os_p, r_os_p, sub_orig_p
        )
        # Star flux
        sub_os_s, sx_os, sy_os, r_os_s, sub_orig_s = extract_and_oversample_subarray(
            star_psf, pos.px, pos.py, radius_pix, oversample
        )
        star_flux = measure_flux_in_oversampled_aperture(
            sub_os_s, sx_os, sy_os, r_os_s, sub_orig_s
        )
        contrast_val = star_flux / planet_flux if planet_flux > 0 else np.inf
        separations.append(pos.separation)
        contrasts.append(contrast_val)

    return _collect_and_sort(separations, contrasts)


def compute_core_area_curve(
    coro: Coronagraph,
    aperture_radius_lod: float = 0.7,
    fit_gaussian: bool = False,
    use_phot_aperture_as_min: bool = False,
    oversample: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute core area vs separation.

    The core area represents the effective area of the PSF core.  If
    *fit_gaussian* is True a 2-D Gaussian is fitted to each PSF; otherwise a
    fixed circular aperture is used.

    Args:
        coro: Coronagraph instance.
        aperture_radius_lod: Aperture radius in lam/D.
        fit_gaussian: Whether to fit a 2-D Gaussian.
        use_phot_aperture_as_min: Use aperture area as a floor when fitting.
        oversample: Oversampling factor.

    Returns:
        ``(separations, core_areas)`` - sorted 1-D arrays, area in (lam/D)**2.
    """
    if fit_gaussian:
        from scipy.optimize import curve_fit

        def gaussian_2d(coords, amplitude, x0, y0, sigma_x, sigma_y):
            x, y = coords
            return (
                amplitude
                * np.exp(-(((x - x0) / sigma_x) ** 2 + ((y - y0) / sigma_y) ** 2) / 2)
            ).ravel()

    separations, core_areas = [], []

    for pos in _iter_xaxis_positions(coro):
        if fit_gaussian and not _point_source_in_image(pos):
            separations.append(pos.separation)
            core_areas.append(np.pi * aperture_radius_lod**2)
            continue
        if fit_gaussian:
            sub_os, px_os, py_os, _, _ = extract_and_oversample_subarray(
                pos.psf,
                pos.px,
                pos.py,
                aperture_radius_lod / coro.pixel_scale_arcsec.value * 3,
                oversample,
            )
            y_grid, x_grid = np.indices(sub_os.shape)
            amplitude = sub_os.max()
            sigma_init = 1.0 * oversample
            popt, _ = curve_fit(
                gaussian_2d,
                (x_grid, y_grid),
                sub_os.ravel(),
                p0=[amplitude, px_os, py_os, sigma_init, sigma_init],
            )
            _, _, _, sigma_x, sigma_y = popt
            sigma_x_lod = sigma_x / oversample * coro.pixel_scale_arcsec.value
            sigma_y_lod = sigma_y / oversample * coro.pixel_scale_arcsec.value
            fwhm_x = GAUSSIAN_FWHM_FACTOR * sigma_x_lod
            fwhm_y = GAUSSIAN_FWHM_FACTOR * sigma_y_lod
            area = np.pi * fwhm_x * fwhm_y / 4
            if use_phot_aperture_as_min:
                area = max(area, np.pi * aperture_radius_lod**2)
        else:
            area = np.pi * aperture_radius_lod**2

        separations.append(pos.separation)
        core_areas.append(area)

    return _collect_and_sort(separations, core_areas)


def compute_truncation_throughput_curve(
    coro: Coronagraph,
    psf_trunc_ratio: float = 0.5,
    oversample: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute throughput using a PSF-truncation-ratio aperture.

    Instead of a fixed circular aperture, selects all pixels where the
    oversampled PSF exceeds ``psf_trunc_ratio * peak``.  Throughput is the
    sum of those pixels (after area-scaled cubic interpolation).  This
    matches the ``photap_frac`` calculation in AYO's ``load_coronagraph.pro``.

    Args:
        coro: Coronagraph instance.
        psf_trunc_ratio: Fraction of the PSF peak used as threshold
            (e.g. 0.5 keeps all pixels > 50% of the peak).
        oversample: Oversampling factor.  ``None`` uses AYO's rule:
            ``ceil(pixscale / 0.05)``.

    Returns:
        ``(separations, throughputs)`` - sorted 1-D arrays.
    """
    pix_lod = coro.pixel_scale_arcsec.value
    oversample = resolve_oversample(coro, oversample)

    separations, throughputs = [], []

    for pos in _iter_xaxis_positions(coro):
        psf_os = _oversample_psf(pos.psf, pix_lod, oversample)
        mask = _threshold_mask(psf_os, psf_trunc_ratio)

        throughput = psf_os[mask].sum()
        separations.append(pos.separation)
        throughputs.append(throughput)

    return _collect_and_sort(separations, throughputs)


def compute_truncation_core_area_curve(
    coro: Coronagraph,
    psf_trunc_ratio: float = 0.5,
    oversample: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute core area using a PSF-truncation-ratio aperture.

    The core area is the solid angle (in (lam/D)**2) of all oversampled pixels
    that exceed ``psf_trunc_ratio * peak``.  This matches AYO's
    ``omega_lod`` calculation.

    Args:
        coro: Coronagraph instance.
        psf_trunc_ratio: Fraction of PSF peak used as threshold.
        oversample: Oversampling factor.  ``None`` uses AYO's rule.

    Returns:
        ``(separations, core_areas)`` - sorted 1-D arrays, area in (lam/D)^2.
    """
    pix_lod = coro.pixel_scale_arcsec.value
    oversample = resolve_oversample(coro, oversample)

    # Solid angle of one oversampled pixel in (lam/D)**2
    os_pix_lod = pix_lod / oversample
    pix_solid_angle = os_pix_lod**2

    separations, core_areas = [], []

    for pos in _iter_xaxis_positions(coro):
        psf_os = _oversample_psf(pos.psf, pix_lod, oversample)
        mask = _threshold_mask(psf_os, psf_trunc_ratio)

        area = mask.sum() * pix_solid_angle
        separations.append(pos.separation)
        core_areas.append(area)

    return _collect_and_sort(separations, core_areas)


def compute_occ_trans_curve(
    coro: Coronagraph,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute occulter (sky) transmission curve.

    This is the radial profile of the sky transmission mask.

    Args:
        coro: Coronagraph instance.

    Returns:
        ``(separations_lod, occ_trans)`` - 1-D arrays.
    """
    sky_trans_data = coro.sky_trans()
    return compute_radial_average(sky_trans_data, coro.pixel_scale_arcsec.value)


def stellar_axis_index(coro: Coronagraph) -> tuple[float, float]:
    """Pixel-center index ``(row, column)`` of the optical axis in the stellar maps.

    A package's ``XCENTER`` and ``YCENTER`` (or ``npix / 2`` when they are
    absent) count pixels from the array's outer corner. With integer indices
    labeling pixel centers the axis therefore sits half a pixel earlier: a
    256-pixel map with ``XCENTER = 128`` is point-symmetric about index 127.5,
    between the four central pixels.

    Args:
        coro: Coronagraph instance.

    Returns:
        ``(row, column)`` of the optical axis as 0-based pixel-center indices.
    """
    return (
        coro.stellar_intens.center_y - 0.5,
        coro.stellar_intens.center_x - 0.5,
    )


PROFILE_STEP_PIX = 0.25
"""Radial spacing, in pixels, of the tabulated core mean intensity profile."""


def profile_radii(shape: tuple[int, int], center: tuple[float, float]) -> np.ndarray:
    """Radii, in pixels, at which the core mean intensity profile is tabulated.

    The grid starts on the axis and steps by :data:`PROFILE_STEP_PIX` out to
    the farthest pixel center of an image of ``shape`` measured from
    ``center``, so the whole map is covered.

    Args:
        shape: ``(ny, nx)`` of the stellar intensity map.
        center: ``(row, column)`` of the optical axis as pixel-center indices.

    Returns:
        1-D array of radii in pixels.
    """
    ny, nx = shape
    cy, cx = center
    r_max = max(np.hypot(y - cy, x - cx) for y in (0, ny - 1) for x in (0, nx - 1))
    return np.arange(int(np.floor(r_max / PROFILE_STEP_PIX)) + 1) * PROFILE_STEP_PIX


def azimuthal_mean_profile(image, center, radii_pix, n_theta=None):
    """Azimuthal mean of an image on circles about ``center``.

    Each circle is sampled at ``n_theta`` equally spaced position angles with
    Keys cubic interpolation (:func:`hwoutils.map_coordinates.map_coordinates`)
    and the samples that fall inside the image are averaged. This is the mean
    intensity at exactly that radius. Averaging the pixels whose centers fall
    in an annulus instead mixes a range of radii, which biases the profile by
    several percent wherever it is curved on the scale of the annulus width.

    Args:
        image: 2-D array.
        center: ``(row, column)`` of the circle centers as pixel-center
            indices.
        radii_pix: 1-D array of circle radii in pixels.
        n_theta: Position angles per circle. Defaults to twice the larger
            image dimension, more than the highest azimuthal harmonic a map
            sampled at or above Nyquist carries at its corner radius.

    Returns:
        1-D JAX array, the mean of each circle's in-image samples (0 for a
        circle with none).
    """
    image = jnp.asarray(image)
    ny, nx = image.shape
    if n_theta is None:
        n_theta = 2 * max(ny, nx)
    theta = jnp.arange(n_theta) * (2.0 * jnp.pi / n_theta)
    radii = jnp.asarray(radii_pix, dtype=image.dtype)[:, None]
    rows = center[0] + radii * jnp.sin(theta)[None, :]
    cols = center[1] + radii * jnp.cos(theta)[None, :]
    inside = (rows >= 0) & (rows <= ny - 1) & (cols >= 0) & (cols <= nx - 1)
    values = map_coordinates(image, [rows, cols], order=3, mode="nearest")
    count = jnp.sum(inside, axis=1)
    total = jnp.sum(jnp.where(inside, values, 0.0), axis=1)
    return jnp.where(count > 0, total / jnp.maximum(count, 1), 0.0)


def core_mean_intensity_profile(coro: Coronagraph, stellar_diam) -> np.ndarray:
    """Core mean intensity profile of the stellar map at ``stellar_diam``.

    The profile is the azimuthal mean of ``coro.stellar_intens(stellar_diam)``
    about the optical axis, on the radii of :func:`profile_radii`. Because it
    is computed from the map itself, the curve at any diameter, tabulated or
    not, is the azimuthal mean of the map an image simulator renders at that
    diameter.

    Args:
        coro: Coronagraph instance.
        stellar_diam: Stellar angular diameter (Quantity in lambda/D).

    Returns:
        1-D array, per pixel, on the radii of :func:`profile_radii`.
    """
    psf = np.asarray(coro.stellar_intens(stellar_diam), dtype=np.float64)
    center = stellar_axis_index(coro)
    radii = profile_radii(psf.shape, center)
    return np.asarray(azimuthal_mean_profile(jnp.asarray(psf), center, radii))


def compute_core_mean_intensity_curve(
    coro: Coronagraph,
    stellar_diam_values=None,
) -> tuple[np.ndarray, dict]:
    """Compute core mean intensity curves for different stellar diameters.

    Args:
        coro: Coronagraph instance.
        stellar_diam_values:
            List of stellar diameters.  ``None`` -> use all available.

    Returns:
        ``(separations_lod, intensities_dict)`` where *intensities_dict*
        maps each stellar diameter to its radial intensity profile: the
        azimuthal mean, about the optical axis (:func:`stellar_axis_index`),
        of the fraction of the star's light falling on each pixel, at each
        separation (:func:`core_mean_intensity_profile`). The values are per
        pixel, not per unit solid angle; divide by the squared pixel scale in
        lambda/D for an intensity per (lambda/D)^2.
    """
    available_diams = coro.stellar_intens.diams
    if stellar_diam_values is None:
        stellar_diam_values = available_diams
    else:
        for diam in stellar_diam_values:
            if diam not in available_diams:
                raise ValueError(
                    f"Requested stellar diameter {diam}"
                    f" not found in available diameters"
                )

    shape = coro.stellar_intens.psfs.shape[-2:]
    radii = profile_radii(shape, stellar_axis_index(coro))
    separations = radii * coro.pixel_scale_arcsec.value
    intensities = {
        diam: core_mean_intensity_profile(coro, diam) for diam in stellar_diam_values
    }
    return separations, intensities


# ---------------------------------------------------------------------------
# Interpolator builder (shared by compute_all_performance_curves and
# Coronagraph.set_psf_trunc_ratio)
# ---------------------------------------------------------------------------


def _build_perf_interps(
    coro, sep, throughput, raw_contrast, sep_ca, core_area, interp_order
):
    """Assign throughput, contrast, and core area spline interpolators to *coro*."""
    coro.interp_order = interp_order
    coro.throughput_interp = make_interp_spline(sep, throughput, k=interp_order)

    log_contrast = np.log10(np.abs(raw_contrast) + 1e-20)
    coro._log_contrast_interp = make_interp_spline(sep, log_contrast, k=interp_order)
    coro.raw_contrast_interp = make_interp_spline(sep, raw_contrast, k=interp_order)

    coro.core_area_interp = make_interp_spline(sep_ca, core_area, k=interp_order)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def compute_all_performance_curves(
    coro: Coronagraph,
    aperture_radius_lod: float = 0.7,
    stellar_diam=None,
    fit_gaussian_for_core_area: bool = False,
    use_phot_aperture_as_min: bool = False,
    oversample: int | None = 2,
    save_to_fits: bool = True,
    performance_file: str = "coro_perf.fits",
    load_from_file: str | None = None,
    cache_dir: Path | None = None,
    plot: bool = False,
    psf_trunc_ratio: float | None = None,
    interp_order: int = 1,
) -> dict:
    """Compute (or load) all coronagraph performance curves.

    Stores interpolators on *coro*.
    This is the main entrypoint called during ``Coronagraph.__init__``.  It
    delegates to the individual ``compute_*`` helpers above, builds spline
    interpolators, and computes IWA / OWA.

    ``load_from_file`` is an explicit import: the named table supplies the
    separation, throughput and raw contrast as-is, whatever settings produced
    it, and nothing is written. Automatic cache reuse in ``Coronagraph``
    validates a table's identity metadata before passing it here. A table
    written with ``save_to_fits`` records the identity of the settings that
    actually produced it (when *coro* provides ``_perf_identity``), so a table
    saved under non-default settings is never accepted as an automatic entry
    for different ones. The stored contrast is always unfloored; the floor is
    applied after loading.

    Returns a dict of all curve data for convenience.
    """
    if stellar_diam is None:
        stellar_diam = coro.stellar_intens.diams[0]
    oversample = resolve_oversample(coro, oversample)

    # Resolve the directory for save/load operations
    io_dir = cache_dir if cache_dir is not None else coro.yip_path

    # ------------------------------------------------------------------
    # Throughput + contrast: load from file or compute
    # ------------------------------------------------------------------
    loaded = False
    if load_from_file is not None:
        logger.info(f"Loading throughput and contrast from {load_from_file}")
        try:
            sep, throughput, raw_contrast = load_coro_performance_from_fits(
                load_from_file, io_dir
            )
            loaded = True
            logger.info(f"Successfully loaded performance data from {load_from_file}")
        except Exception as e:
            logger.warning(f"Error loading from {load_from_file}: {e}")
            logger.info("Computing throughput and contrast from scratch")

    if not loaded:
        if psf_trunc_ratio is not None:
            logger.info(
                f"Computing throughput curve (PSF trunc ratio = {psf_trunc_ratio})..."
            )
            sep, throughput = compute_truncation_throughput_curve(
                coro, psf_trunc_ratio=psf_trunc_ratio, oversample=oversample
            )
        else:
            logger.info("Computing throughput curve...")
            sep, throughput = compute_throughput_curve(
                coro, aperture_radius_lod=aperture_radius_lod, oversample=oversample
            )
        logger.info("Computing raw contrast curve...")
        sep_c, raw_contrast = compute_raw_contrast_curve(
            coro,
            stellar_diam=stellar_diam,
            aperture_radius_lod=aperture_radius_lod,
            oversample=oversample,
        )
        # sep and sep_c should match since they iterate the same positions
        assert np.allclose(sep, sep_c), "Throughput and contrast separations differ"

        if save_to_fits:
            identity = None
            if hasattr(coro, "_perf_identity"):
                identity = coro._perf_identity(
                    aperture_radius_lod=aperture_radius_lod,
                    psf_trunc_ratio=psf_trunc_ratio,
                    oversample=oversample,
                    stellar_diam=stellar_diam,
                )
            save_coro_performance_to_fits(
                sep,
                throughput,
                raw_contrast,
                performance_file,
                io_dir,
                identity=identity,
            )

    # Apply contrast floor
    if coro.contrast_floor is not None:
        raw_contrast = np.maximum(np.abs(raw_contrast), coro.contrast_floor)
        logger.info(f"Applied contrast floor of {coro.contrast_floor:.1e}")

    # ------------------------------------------------------------------
    # Core area
    # ------------------------------------------------------------------
    if psf_trunc_ratio is not None:
        logger.info(
            f"Computing core area curve (PSF trunc ratio = {psf_trunc_ratio})..."
        )
        sep_ca, core_area = compute_truncation_core_area_curve(
            coro, psf_trunc_ratio=psf_trunc_ratio, oversample=oversample
        )
    else:
        logger.info("Computing core area curve...")
        sep_ca, core_area = compute_core_area_curve(
            coro,
            aperture_radius_lod=aperture_radius_lod,
            fit_gaussian=fit_gaussian_for_core_area,
            use_phot_aperture_as_min=use_phot_aperture_as_min,
            oversample=oversample,
        )

    _build_perf_interps(
        coro, sep, throughput, raw_contrast, sep_ca, core_area, interp_order
    )

    # ------------------------------------------------------------------
    # Occulter transmission
    # ------------------------------------------------------------------
    logger.info("Computing occulter transmission curve...")
    sep_occ_trans, occ_trans = compute_occ_trans_curve(coro)
    coro.occ_trans_interp = make_interp_spline(sep_occ_trans, occ_trans, k=interp_order)

    # ------------------------------------------------------------------
    # Core mean intensity
    # ------------------------------------------------------------------
    logger.info("Computing core mean intensity curve...")
    sep_core_intensity, core_intensities = compute_core_mean_intensity_curve(
        coro, stellar_diam_values=None
    )
    coro.core_intensity_dict = core_intensities
    coro.core_intensity_separations = sep_core_intensity
    # Profiles at other diameters are computed from the interpolated map on
    # first use, so they always agree with it (Coronagraph.core_mean_intensity).
    coro._core_intensity_cache = {}

    # 1D spline for the default stellar diameter (backward compatibility
    # and fast path for the common single-diameter case)
    coro.core_intensity_interp = make_interp_spline(
        sep_core_intensity,
        core_intensities[stellar_diam],
        k=interp_order,
    )

    # ------------------------------------------------------------------
    # IWA / OWA
    # ------------------------------------------------------------------
    _compute_iwa_owa(coro, sep, throughput)

    # ------------------------------------------------------------------
    # Optional plots
    # ------------------------------------------------------------------
    if plot:
        plot_performance_curve(
            sep,
            throughput,
            title=f"{coro.name} Throughput",
            xlabel="Separation [lam/D]",
            ylabel="Throughput",
            ms=6,
        )
        plot_performance_curve(
            sep,
            raw_contrast,
            title=f"{coro.name} Raw Contrast",
            xlabel="Separation [lam/D]",
            ylabel="Raw Contrast",
            log_scale=True,
        )
        suffix = (
            " (Gaussian fit)" if fit_gaussian_for_core_area else " (fixed aperture)"
        )
        plot_performance_curve(
            sep_ca,
            core_area,
            title=f"{coro.name} Core Area{suffix}",
            xlabel="Separation [lam/D]",
            ylabel="Core Area [(lam/D)**2]",
        )

    return {
        "separations": sep,
        "throughput": throughput,
        "raw_contrast": raw_contrast,
        "separations_occ_trans": sep_occ_trans,
        "occ_trans": occ_trans,
        "separations_core_area": sep_ca,
        "core_area": core_area,
        "separations_core_intensity": sep_core_intensity,
        "core_intensities": core_intensities,
        "IWA": coro.IWA,
        "OWA": coro.OWA,
    }
