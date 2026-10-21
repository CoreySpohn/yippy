"""Band-limited aperture sums against the clear circular pupil's Airy pattern.

The intensity and encircled energy follow Born and Wolf, Principles of Optics,
7th edition, section 8.5.2 (diffraction at a circular aperture):
I(r) = (pi / 4) [2 J1(pi r) / (pi r)]^2 and
E(r) = 1 - J0(pi r)^2 - J1(pi r)^2, with r in lambda/D.
The intensity integrates to one over the infinite image; do not normalize the
finite crop, which would fold the missing Airy wings into the core.
"""

from types import SimpleNamespace

import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from lod_unit import lod
from scipy.optimize import brentq
from scipy.special import j0, j1

from yippy.performance import (
    circular_aperture,
    compute_core_area_curve,
    compute_raw_contrast_curve,
    compute_throughput_curve,
    core_window_half_width_pix,
    truncation_core,
)


def airy_psf(pixel_lod, shift_pix=0.0, size=256):
    """Sample a unit-flux Airy intensity without renormalizing its finite crop."""
    rows, cols = np.indices((size, size))
    radius = np.hypot(rows - size // 2, cols - size // 2 - shift_pix) * pixel_lod
    x = np.pi * radius
    amplitude = 2 * j1(x) / np.where(x == 0, 1.0, x)
    amplitude[x == 0] = 1.0
    return np.pi / 4 * amplitude**2 * pixel_lod**2


def image_coro(psf, pixel_lod, star=None):
    """Expose one image through the same data interface as a loaded package."""
    center_y, center_x = np.array(psf.shape) // 2
    return SimpleNamespace(
        pixel_scale_arcsec=pixel_lod * lod / u.pix,
        offax=SimpleNamespace(
            x_offsets=np.array([0.0]) * lod,
            y_offsets=np.array([0.0]) * lod,
            center_x=center_x * u.pix,
            center_y=center_y * u.pix,
            get_psf_by_offset_idx=lambda i, j: psf,
        ),
        stellar_intens=lambda diam: psf if star is None else star,
    )


def encircled_energy(radius_lod):
    """Infinite-image Airy encircled energy from Born and Wolf, section 8.5.2."""
    x = np.pi * radius_lod
    return 1 - j0(x) ** 2 - j1(x) ** 2


@pytest.mark.parametrize("pixel_lod", [0.25, 0.5])
@pytest.mark.parametrize("shift_pix", [0.0, 0.3])
@pytest.mark.parametrize(
    "radius_lod",
    [
        pytest.param(
            0.7,
            marks=pytest.mark.xfail(
                strict=True,
                reason="10x center counting exceeds 0.1% continuum error at r=0.7",
            ),
        ),
        1.0,
        1.22,
    ],
)
def test_circle_airy_encircled_energy(pixel_lod, shift_pix, radius_lod):
    """A circular sum recovers the Airy encircled energy at each sampling phase."""
    coro = image_coro(airy_psf(pixel_lod, shift_pix), pixel_lod)
    measured = compute_throughput_curve(coro, aperture_radius_lod=radius_lod)[1][0]
    assert measured == pytest.approx(encircled_energy(radius_lod), rel=1e-3)


@pytest.mark.parametrize("pixel_lod", [0.25, 0.5])
@pytest.mark.parametrize("shift_pix", [0.0, 0.3])
@pytest.mark.parametrize("metric", ["area", "throughput"])
@pytest.mark.xfail(
    strict=True, reason="10x whole-cell Airy half-maximum core exceeds 0.1% error"
)
def test_truncation_airy_half_maximum(pixel_lod, shift_pix, metric):
    """The half-maximum core has the Airy FWHM area and encircled energy."""
    radius = brentq(lambda r: (2 * j1(np.pi * r) / (np.pi * r)) ** 2 - 0.5, 0.1, 1)
    throughput, area = truncation_core(
        airy_psf(pixel_lod, shift_pix),
        0.5,
        pixel_lod,
        half_width_pix=core_window_half_width_pix(pixel_lod),
    )
    if metric == "area":
        assert float(area) == pytest.approx(np.pi * radius**2, rel=1e-3)
    else:
        assert float(throughput) == pytest.approx(encircled_energy(radius), rel=1e-3)


@pytest.mark.parametrize("pixel_lod", [0.25, 0.5])
def test_circle_is_invariant_to_subpixel_shift(pixel_lod):
    """A 0.3-pixel source shift preserves the peak-centered encircled flux."""
    fluxes = [
        compute_throughput_curve(image_coro(airy_psf(pixel_lod, shift), pixel_lod))[1][
            0
        ]
        for shift in (0.0, 0.3)
    ]
    assert fluxes[0] == pytest.approx(fluxes[1], rel=1e-3)


def test_flat_image_flux_is_the_counted_circle_area():
    """A flat stellar image carries intensity times the measured aperture area."""
    psf = airy_psf(0.25, 0.3, size=64)
    coro = image_coro(psf, 0.25, star=np.full_like(psf, 3.0))
    flux = compute_throughput_curve(coro)[1][0]
    contrast = compute_raw_contrast_curve(coro)[1][0]
    area = compute_core_area_curve(coro)[1][0]
    assert contrast * flux == pytest.approx(3 * area / 0.25**2, rel=1e-6)


def test_circle_counts_centers_on_its_boundary():
    """The four axial centers at radius 28 subpixels belong to the circle."""
    coro = image_coro(airy_psf(0.25, size=64), 0.25)
    # Integer lattice points in a closed disk of radius 28: 2453 cells.
    area = compute_core_area_curve(coro)[1][0]
    assert area == pytest.approx(2453 * 0.025**2, rel=1e-6)


@pytest.mark.parametrize("pixel_lod", [0.25, 0.5])
@pytest.mark.parametrize("shift_pix", [0.0, 0.3])
@pytest.mark.parametrize("radius_lod", [0.7, 1.0, 1.22])
def test_circle_matches_independently_sampled_airy(pixel_lod, shift_pix, radius_lod):
    """The DFT sum agrees with exact Airy values at the same subpixel centers."""
    # Relative to the brightest subpixel the source is half a subpixel away
    # on each axis. A 0.3-pixel shift moves it by exactly three subpixels.
    half = int(np.ceil(radius_lod / pixel_lod * 10))
    rows, cols = np.mgrid[-half : half + 1, -half : half + 1]
    radius = np.hypot(rows + 0.5, cols + 0.5) * pixel_lod / 10
    x = np.pi * radius
    intensity = np.pi / 4 * (2 * j1(x) / x) ** 2
    inside = rows**2 + cols**2 <= (radius_lod / pixel_lod * 10) ** 2 + 1e-10
    expected = intensity[inside].sum() * (pixel_lod / 10) ** 2
    coro = image_coro(airy_psf(pixel_lod, shift_pix), pixel_lod)
    measured = compute_throughput_curve(coro, aperture_radius_lod=radius_lod)[1][0]
    assert measured == pytest.approx(expected, rel=2e-6)


@pytest.mark.parametrize("pixel_lod", [0.25, 0.5])
@pytest.mark.parametrize("shift_pix", [0.0, 0.3])
def test_truncation_matches_independently_sampled_airy(pixel_lod, shift_pix):
    """Threshold membership and flux agree with directly sampled Airy values."""
    rows, cols = np.mgrid[-100:100, -100:100] + 0.5
    radius = np.hypot(rows, cols) * pixel_lod / 10
    x = np.pi * radius
    intensity = np.pi / 4 * (2 * j1(x) / x) ** 2
    above = intensity > 0.5 * intensity.max()
    expected_area = above.sum() * (pixel_lod / 10) ** 2
    expected_flux = intensity[above].sum() * (pixel_lod / 10) ** 2
    flux, area = truncation_core(
        airy_psf(pixel_lod, shift_pix),
        0.5,
        pixel_lod,
        half_width_pix=core_window_half_width_pix(pixel_lod),
    )
    assert float(area) == pytest.approx(expected_area, rel=1e-6)
    assert float(flux) == pytest.approx(expected_flux, rel=2e-6)


@pytest.mark.parametrize(
    "aperture",
    [
        pytest.param(
            "circle",
            marks=pytest.mark.xfail(
                strict=True,
                reason="10x circular center counting exceeds 0.1% convergence at r=0.7",
            ),
        ),
        "truncation",
    ],
)
@pytest.mark.parametrize("metric", ["throughput", "area"])
def test_shipped_ten_against_forty_subpixels(shipped_coro, aperture, metric):
    """Quantify 10x boundary quadrature against 40x on a native shipped PSF."""
    coro = shipped_coro
    # Use the native PSF nearest 20 lambda/D, in the unobscured science region.
    nodes = np.asarray(coro.offax.x_offsets)
    iy = int(np.argmin(np.abs(np.asarray(coro.offax.y_offsets))))
    valid = [
        i
        for i in range(len(nodes))
        if coro.offax.get_psf_by_offset_idx(i, iy) is not None
    ]
    ix = min(valid, key=lambda i: abs(nodes[i] - 20))
    psf = coro.offax.get_psf_by_offset_idx(ix, iy)
    pixel_lod = float(coro.pixel_scale_arcsec.value)
    values = []
    for factor in (10, 40):
        if aperture == "circle":
            sampled = image_coro(psf, pixel_lod)
            result = (
                compute_throughput_curve(sampled, oversample=factor)[1][0],
                compute_core_area_curve(sampled, oversample=factor)[1][0],
            )
        else:
            result = truncation_core(
                psf,
                0.3,
                pixel_lod,
                oversample=factor,
                half_width_pix=core_window_half_width_pix(pixel_lod),
            )
        values.append(float(result[0 if metric == "throughput" else 1]))
    assert values[0] == pytest.approx(values[1], rel=1e-3)


def test_shipped_contrast_is_repeatable_for_faint_stellar_sums(shipped_coro):
    """Faint stellar DFT sums are stable despite cancellation of bright modes."""
    values = np.stack([compute_raw_contrast_curve(shipped_coro)[1] for _ in range(20)])
    np.testing.assert_allclose(
        values, np.broadcast_to(values[0], values.shape), rtol=1e-6
    )


def test_circular_aperture_composes_under_jit():
    """The numerical kernel supports enclosing JIT with the active JAX precision."""
    with jax.enable_x64(False):
        flux, area, star_flux = jax.jit(
            lambda psf: circular_aperture(psf, 2.0, 0.25, half_width_pix=3)
        )(jnp.zeros((8, 8)))
    assert float(flux) == 0
    assert float(star_flux) == 0
    # A radius-20-subpixel closed quadrant has 335 centers.
    assert float(area) == pytest.approx(335 * 0.025**2, rel=1e-6)
