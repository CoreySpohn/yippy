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

from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
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
# Utility helpers
# ---------------------------------------------------------------------------


def compute_radial_average(image, pixel_scale_value, center=None, nbins=None):
    """Compute radial average of a 2D image.

    Args:
        image:
            2D numpy array.
        pixel_scale_value:
            Pixel scale in lambda/D per pixel (float).
        center:
            ``[x, y]`` pixel coordinates of centre.  Defaults to image centre.
        nbins:
            Number of radial bins.  Defaults to ``floor(max_dim / 2)``.

    Returns:
        ``(separations_lod, radial_profile)`` – 1-D arrays.
    """
    if center is None:
        center = [(image.shape[1] - 1) / 2, (image.shape[0] - 1) / 2]

    y, x = np.indices(image.shape)
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    if nbins is None:
        nbins = int(np.floor(np.max(image.shape) / 2))

    max_radius = np.max(r)
    bins = np.linspace(0, max_radius, nbins + 1)
    bin_centers = (bins[1:] + bins[:-1]) / 2

    inds = np.digitize(r.ravel(), bins)
    inds[inds == nbins + 1] = nbins

    means = np.zeros(nbins)
    image_flat = image.ravel()
    for j in range(1, nbins + 1):
        means[j - 1] = np.nanmean(image_flat[inds == j])

    separations_lod = bin_centers * pixel_scale_value
    return separations_lod, means


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
# Helper: iterate over x-axis offsets (y=0) and yield per-position data
# ---------------------------------------------------------------------------


def _iter_xaxis_positions(coro: "Coronagraph"):
    """Yield ``(i, x_lod, y_lod, y_idx, planet_psf, px, py)`` for each
    valid x-offset position along the performance-curve axis (y ≈ 0).
    """
    y_offsets = np.array(coro.offax.y_offsets)
    y_idx = int(np.argmin(np.abs(y_offsets)))
    y_lod_val = y_offsets[y_idx]

    if y_lod_val != 0:
        logger.warning(
            f"No PSF at y=0, using closest y offset: {y_lod_val:.3f} λ/D"
        )

    max_sep = None
    if hasattr(coro.offax, "max_offset_in_image"):
        max_sep = coro.offax.max_offset_in_image.to(u.lod).value

    for i, x_lod_val in enumerate(np.array(coro.offax.x_offsets)):
        r = abs(x_lod_val)
        if max_sep is not None and r > max_sep:
            continue
        planet_psf = coro.offax.get_psf_by_offset_idx(i, y_idx)
        if planet_psf is None:
            continue
        px = convert_to_pix(
            x_lod_val, coro.offax.center_x, coro.pixel_scale
        ).value.astype(int)
        py = convert_to_pix(
            y_lod_val, coro.offax.center_y, coro.pixel_scale
        ).value.astype(int)
        yield i, r, x_lod_val, y_lod_val, y_idx, planet_psf, px, py


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------


def compute_throughput_curve(
    coro: "Coronagraph",
    aperture_radius_lod: float = 0.7,
    oversample: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute coronagraph throughput vs separation.

    Throughput is the fraction of the total flux (planet PSF normalised to 1)
    that lands inside a photometric aperture at each off-axis position.

    Args:
        coro: Coronagraph instance.
        aperture_radius_lod: Aperture radius in λ/D.
        oversample: Oversampling factor.

    Returns:
        ``(separations, throughputs)`` – sorted 1-D arrays.
    """
    separations, throughputs = [], []
    radius_pix = aperture_radius_lod / coro.pixel_scale.value

    for _, r, *_, planet_psf, px, py in _iter_xaxis_positions(coro):
        sub_os, px_os, py_os, r_os, sub_orig = extract_and_oversample_subarray(
            planet_psf, px, py, radius_pix, oversample
        )
        flux = measure_flux_in_oversampled_aperture(
            sub_os, px_os, py_os, r_os, sub_orig
        )
        separations.append(r)
        throughputs.append(flux)

    separations = np.array(separations)
    throughputs = np.array(throughputs)
    order = np.argsort(separations)
    return separations[order], throughputs[order]


def compute_raw_contrast_curve(
    coro: "Coronagraph",
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
        aperture_radius_lod: Aperture radius in λ/D.
        oversample: Oversampling factor.

    Returns:
        ``(separations, contrasts)`` – sorted 1-D arrays.
    """
    star_psf = coro.stellar_intens(stellar_diam)
    separations, contrasts = [], []
    radius_pix = aperture_radius_lod / coro.pixel_scale.value

    for _, r, *_, planet_psf, px, py in _iter_xaxis_positions(coro):
        # Planet flux
        sub_os_p, px_os_p, py_os_p, r_os_p, sub_orig_p = (
            extract_and_oversample_subarray(
                planet_psf, px, py, radius_pix, oversample
            )
        )
        planet_flux = measure_flux_in_oversampled_aperture(
            sub_os_p, px_os_p, py_os_p, r_os_p, sub_orig_p
        )
        # Star flux
        sub_os_s, sx_os, sy_os, r_os_s, sub_orig_s = (
            extract_and_oversample_subarray(
                star_psf, px, py, radius_pix, oversample
            )
        )
        star_flux = measure_flux_in_oversampled_aperture(
            sub_os_s, sx_os, sy_os, r_os_s, sub_orig_s
        )
        contrast_val = star_flux / planet_flux if star_flux > 0 else 0
        separations.append(r)
        contrasts.append(contrast_val)

    separations = np.array(separations)
    contrasts = np.array(contrasts)
    order = np.argsort(separations)
    return separations[order], contrasts[order]


def compute_core_area_curve(
    coro: "Coronagraph",
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
        aperture_radius_lod: Aperture radius in λ/D.
        fit_gaussian: Whether to fit a 2-D Gaussian.
        use_phot_aperture_as_min: Use aperture area as a floor when fitting.
        oversample: Oversampling factor.

    Returns:
        ``(separations, core_areas)`` – sorted 1-D arrays, area in (λ/D)².
    """
    if fit_gaussian:
        from scipy.optimize import curve_fit

        def gaussian_2d(coords, amplitude, x0, y0, sigma_x, sigma_y):
            x, y = coords
            return amplitude * np.exp(
                -(((x - x0) / sigma_x) ** 2 + ((y - y0) / sigma_y) ** 2) / 2
            )

    separations, core_areas = [], []
    radius_pix = aperture_radius_lod / coro.pixel_scale.value

    for _, r, *_, planet_psf, px, py in _iter_xaxis_positions(coro):
        if fit_gaussian:
            sub_os, px_os, py_os, _, _ = extract_and_oversample_subarray(
                planet_psf,
                px,
                py,
                aperture_radius_lod / coro.pixel_scale.value * 3,
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
            sigma_x_lod = sigma_x / oversample * coro.pixel_scale.value
            sigma_y_lod = sigma_y / oversample * coro.pixel_scale.value
            fwhm_x = 2.355 * sigma_x_lod
            fwhm_y = 2.355 * sigma_y_lod
            area = np.pi * fwhm_x * fwhm_y / 4
            if use_phot_aperture_as_min:
                area = max(area, np.pi * aperture_radius_lod**2)
        else:
            area = np.pi * aperture_radius_lod**2

        separations.append(r)
        core_areas.append(area)

    separations = np.array(separations)
    core_areas = np.array(core_areas)
    order = np.argsort(separations)
    return separations[order], core_areas[order]


def compute_occ_trans_curve(
    coro: "Coronagraph",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute occulter (sky) transmission curve.

    This is the radial profile of the sky transmission mask.

    Args:
        coro: Coronagraph instance.

    Returns:
        ``(separations_lod, occ_trans)`` – 1-D arrays.
    """
    sky_trans_data = coro.sky_trans()
    return compute_radial_average(sky_trans_data, coro.pixel_scale.value)


def compute_core_mean_intensity_curve(
    coro: "Coronagraph",
    stellar_diam_values=None,
) -> tuple[np.ndarray, dict]:
    """Compute core mean intensity curves for different stellar diameters.

    Args:
        coro: Coronagraph instance.
        stellar_diam_values:
            List of stellar diameters.  ``None`` → use all available.

    Returns:
        ``(separations_lod, intensities_dict)`` where *intensities_dict*
        maps each stellar diameter to its radial intensity profile.
    """
    available_diams = coro.stellar_intens.diams
    if stellar_diam_values is None:
        stellar_diam_values = available_diams
    else:
        for diam in stellar_diam_values:
            if diam not in available_diams:
                raise ValueError(
                    f"Requested stellar diameter {diam} not found in available diameters"
                )

    center = [coro.stellar_intens.center_x, coro.stellar_intens.center_y]
    intensities: dict = {}

    # Get grid from the first diameter
    stellar_psf = coro.stellar_intens(stellar_diam_values[0])
    dims = stellar_psf.shape
    nbins = int(np.floor(np.max(dims) / 2))

    y, x = np.indices(stellar_psf.shape)
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    max_radius = np.max(r)
    bins = np.linspace(0, max_radius, nbins + 1)
    bin_centers = (bins[1:] + bins[:-1]) / 2

    for diam in stellar_diam_values:
        psf = coro.stellar_intens(diam)
        psf_flat = psf.ravel()
        r_flat = r.ravel()
        profile = np.zeros(nbins)
        for i in range(nbins):
            mask = (r_flat >= bins[i]) & (r_flat < bins[i + 1])
            if np.any(mask):
                profile[i] = np.nanmean(psf_flat[mask])
            else:
                profile[i] = profile[i - 1] if i > 0 else 0
        intensities[diam] = profile

    separations = bin_centers * coro.pixel_scale.value
    return separations, intensities


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def compute_all_performance_curves(
    coro: "Coronagraph",
    aperture_radius_lod: float = 0.7,
    stellar_diam=None,
    fit_gaussian_for_core_area: bool = False,
    use_phot_aperture_as_min: bool = False,
    oversample: int = 2,
    save_to_fits: bool = True,
    performance_file: str = "coro_perf.fits",
    load_from_file: str | None = None,
    plot: bool = False,
) -> dict:
    """Compute (or load) all coronagraph performance curves and store
    interpolators on *coro*.

    This is the main entrypoint called during ``Coronagraph.__init__``.  It
    delegates to the individual ``compute_*`` helpers above, builds spline
    interpolators, and computes IWA / OWA.

    Returns a dict of all curve data for convenience.
    """
    if stellar_diam is None:
        stellar_diam = coro.stellar_intens.diams[0]

    # ------------------------------------------------------------------
    # Throughput + contrast: load from file or compute
    # ------------------------------------------------------------------
    loaded = False
    if load_from_file is not None:
        logger.info(f"Loading throughput and contrast from {load_from_file}")
        try:
            sep, throughput, raw_contrast = load_coro_performance_from_fits(
                load_from_file, coro.yip_path
            )
            loaded = True
            logger.info(
                f"Successfully loaded performance data from {load_from_file}"
            )
        except Exception as e:
            logger.warning(f"Error loading from {load_from_file}: {e}")
            logger.info("Computing throughput and contrast from scratch")

    if not loaded:
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
        assert np.allclose(sep, sep_c), (
            "Throughput and contrast separations differ"
        )

        if save_to_fits:
            save_coro_performance_to_fits(
                sep, throughput, raw_contrast, performance_file, coro.yip_path
            )

    # Apply contrast floor
    if coro.contrast_floor is not None:
        raw_contrast = np.maximum(np.abs(raw_contrast), coro.contrast_floor)
        logger.info(f"Applied contrast floor of {coro.contrast_floor:.1e}")

    # Splines for throughput and contrast
    coro.throughput_interp = make_interp_spline(sep, throughput, k=3)
    log_contrast = np.log10(np.abs(raw_contrast) + 1e-20)
    coro._log_contrast_interp = make_interp_spline(sep, log_contrast, k=3)
    coro.raw_contrast_interp = make_interp_spline(sep, raw_contrast, k=3)

    # ------------------------------------------------------------------
    # Core area
    # ------------------------------------------------------------------
    logger.info("Computing core area curve...")
    sep_ca, core_area = compute_core_area_curve(
        coro,
        aperture_radius_lod=aperture_radius_lod,
        fit_gaussian=fit_gaussian_for_core_area,
        use_phot_aperture_as_min=use_phot_aperture_as_min,
        oversample=oversample,
    )
    coro.core_area_interp = make_interp_spline(sep_ca, core_area, k=3)

    # ------------------------------------------------------------------
    # Occulter transmission
    # ------------------------------------------------------------------
    logger.info("Computing occulter transmission curve...")
    sep_occ_trans, occ_trans = compute_occ_trans_curve(coro)
    coro.occ_trans_interp = make_interp_spline(sep_occ_trans, occ_trans, k=3)

    # ------------------------------------------------------------------
    # Core mean intensity
    # ------------------------------------------------------------------
    logger.info("Computing core mean intensity curve...")
    sep_core_intensity, core_intensities = compute_core_mean_intensity_curve(
        coro, stellar_diam_values=None
    )
    coro.core_intensity_interp = make_interp_spline(
        sep_core_intensity, core_intensities[stellar_diam], k=3
    )
    coro.core_intensity_dict = core_intensities

    # ------------------------------------------------------------------
    # IWA / OWA
    # ------------------------------------------------------------------
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
        logger.warning(
            f"Could not find IWA bracket, using first valid sep: {coro.IWA}"
        )

    if hasattr(coro.offax, "max_offset_in_image"):
        coro.OWA = coro.offax.max_offset_in_image
        logger.info(
            f"OWA set to max_offset_in_image: {coro.OWA.to(u.lod).value:.2f} λ/D"
        )
    else:
        coro.OWA = np.max(sep) * lod
        logger.warning(
            "max_offset_in_image not available, using maximum separation as OWA"
        )

    # ------------------------------------------------------------------
    # Optional plots
    # ------------------------------------------------------------------
    if plot:
        plot_performance_curve(
            sep, throughput,
            title=f"{coro.name} Throughput",
            xlabel="Separation [λ/D]", ylabel="Throughput", ms=6,
        )
        plot_performance_curve(
            sep, raw_contrast,
            title=f"{coro.name} Raw Contrast",
            xlabel="Separation [λ/D]", ylabel="Raw Contrast",
            log_scale=True,
        )
        suffix = (
            " (Gaussian fit)" if fit_gaussian_for_core_area else " (fixed aperture)"
        )
        plot_performance_curve(
            sep_ca, core_area,
            title=f"{coro.name} Core Area{suffix}",
            xlabel="Separation [λ/D]", ylabel="Core Area [(λ/D)²]",
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
