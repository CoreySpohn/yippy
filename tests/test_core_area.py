"""The core area is the area of the PSF core the image simulator renders.

With a PSF truncation ratio ``r`` the core of an off-axis PSF is the region
where its intensity exceeds ``r`` times its peak, and ``Coronagraph.core_area``
is the solid angle of that region in (lambda/D)^2. It is the photometric
aperture an exposure-time calculator multiplies the leaked starlight and the
background by, so at every separation, on or between the tabulated offsets, it
must be the core area of the PSF yippy synthesizes for a source there.

The references here do not come from the code under test. A circular Gaussian
of standard deviation sigma has the core area ``2 pi sigma^2 ln(1 / r)``. Between
two tabulated offsets yippy synthesizes the PSF as the two neighboring PSFs
shifted onto the source and mixed with weights linear in separation; for
concentric Gaussians that mixture is radially symmetric and its core radius is
the root of a one-dimensional equation. On a shipped package the reference is
the synthesized PSF's band-limited reconstruction, thresholded over the whole
image.
"""

import astropy.io.fits as pyfits
import numpy as np
import pytest
from conftest import REFERENCE_DATA_REQUIRED
from scipy.optimize import brentq

from yippy import Coronagraph

# Separations (lambda/D) probed, on and between the tabulated offsets.
GAUSSIAN_SEPARATIONS_LOD = (2.1, 2.37, 3.0, 3.62, 4.13, 5.05)
TRUNC_RATIOS = (0.3, 0.5)

# Offset grids of the Gaussian package: the default one puts every PSF center
# on a pixel center (0.5 lambda/D steps on 0.25 lambda/D pixels), the other
# puts them at fractions of a pixel.
OFFSET_GRIDS = {"on_pixels": (0.0, 0.5), "off_pixels": (0.07, 0.37)}

# Relative tolerance on the core area of a smooth, well-sampled PSF. The area
# is counted in cells a tenth of a pixel (0.025 lambda/D) on a side; a core of
# radius 0.6 lambda/D centered on a pixel, where whole rings of symmetric
# cells cross the threshold together, sits within 0.8 percent of the closed
# form, and off-center cores sit closer.
AREA_TOL = 1e-2

# A package whose PSF width grows with separation, tabulated every 2 lambda/D,
# so a source between offsets gets a PSF whose core is neither neighbor's.
GROWTH_OFFSET_STEP_LOD = 2.0
GROWTH_SIGMA0_LOD = 0.35
GROWTH_SIGMA_SLOPE = 0.1


def gaussian_core_area(sigma_lod, ratio):
    """Core area of a circular Gaussian above ``ratio`` times its peak."""
    return 2.0 * np.pi * sigma_lod**2 * np.log(1.0 / ratio)


@pytest.fixture(scope="module", params=sorted(OFFSET_GRIDS))
def gaussian_package(request, tmp_path_factory):
    """The radially symmetric analytic package: identical Gaussians."""
    from synthetic_yip import write_synthetic_yip

    first, step = OFFSET_GRIDS[request.param]
    return write_synthetic_yip(
        tmp_path_factory.mktemp("core") / request.param,
        min_offset_lod=first,
        offset_step_lod=step,
    )


@pytest.mark.parametrize("ratio", TRUNC_RATIOS)
def test_core_area_of_a_gaussian_is_its_closed_form(gaussian_package, ratio):
    """The core area of a Gaussian PSF is its closed form wherever it sits.

    Counting whole pixels above the threshold, even on a twice-oversampled
    grid, puts the area on a lattice of (0.125 lambda/D)^2 cells; for a core
    centered on a pixel the cells cross the threshold in symmetric rings and
    the area is off by 9 percent at a ratio of 0.5.
    """
    coro = Coronagraph(gaussian_package.path, psf_trunc_ratio=ratio)
    expected = gaussian_core_area(gaussian_package.psf_sigma_lod, ratio)
    got = np.array([coro.core_area(s) for s in GAUSSIAN_SEPARATIONS_LOD])
    err = np.abs(got / expected - 1.0)
    assert err.max() <= AREA_TOL, (
        f"core area departs from 2 pi sigma^2 ln(1/r) = {expected:.5f} by "
        f"{err.max():.4f} at {GAUSSIAN_SEPARATIONS_LOD[int(np.argmax(err))]} "
        f"lambda/D: {np.round(got, 5)}"
    )


def growth_sigma(separation_lod):
    """PSF standard deviation of the growth package at a tabulated offset."""
    return GROWTH_SIGMA0_LOD + GROWTH_SIGMA_SLOPE * separation_lod


@pytest.fixture(scope="module")
def growth_package(tmp_path_factory):
    """A package whose Gaussian PSFs widen with separation."""
    from synthetic_yip import _gaussian_psf, write_synthetic_yip

    package = write_synthetic_yip(
        tmp_path_factory.mktemp("core") / "growth",
        offset_step_lod=GROWTH_OFFSET_STEP_LOD,
    )
    with pyfits.open(package.path / "offax_psf.fits") as hdul:
        header = hdul[0].header.copy()
    psfs = np.stack(
        [
            _gaussian_psf(
                package.npix,
                package.pixscale,
                package.center,
                x,
                y,
                growth_sigma(np.hypot(x, y)),
                package.psf_flux * package.occulter(np.hypot(x, y)),
            )
            for x, y in package.offsets
        ]
    )
    pyfits.PrimaryHDU(psfs, header=header).writeto(
        package.path / "offax_psf.fits", overwrite=True
    )
    return package


def mixture_core_area(package, separation_lod, ratio):
    """Core area of the PSF yippy synthesizes between two Gaussian offsets.

    The two neighboring PSFs are shifted onto the source and mixed with
    weights ``1 - u`` and ``u``, ``u`` the fractional position between them,
    so the PSF is a sum of two concentric Gaussians of known widths and
    fluxes.
    """
    nodes = np.sort(package.offsets[:, 0])
    lo = nodes[nodes <= separation_lod].max()
    hi = nodes[nodes > separation_lod].min()
    u = (separation_lod - lo) / (hi - lo)
    terms = []
    for node, weight in ((lo, 1.0 - u), (hi, u)):
        sigma = growth_sigma(node)
        flux = weight * package.psf_flux * package.occulter(node)
        terms.append((flux / (2.0 * np.pi * sigma**2), sigma))

    def profile(rho):
        return sum(a * np.exp(-(rho**2) / (2.0 * s**2)) for a, s in terms)

    radius = brentq(lambda rho: profile(rho) - ratio * profile(0.0), 0.0, 10.0)
    return np.pi * radius**2


@pytest.mark.parametrize("ratio", TRUNC_RATIOS)
def test_core_area_follows_the_synthesized_psf_between_offsets(growth_package, ratio):
    """Between tabulated offsets the core area is that of the rendered PSF.

    Interpolating the tabulated core areas linearly in separation instead
    overstates it by 4 to 9 percent here, because the core of the mixed PSF
    is not the mix of the two cores.
    """
    coro = Coronagraph(growth_package.path, psf_trunc_ratio=ratio)
    separations = np.array([2.5, 3.0, 3.37, 3.5, 4.6, 5.25])
    expected = np.array(
        [mixture_core_area(growth_package, s, ratio) for s in separations]
    )
    got = np.array([coro.core_area(s) for s in separations])
    err = np.abs(got / expected - 1.0)
    assert err.max() <= AREA_TOL, (
        f"core area departs from the synthesized PSF's by {err.max():.4f} at "
        f"{separations[int(np.argmax(err))]} lambda/D: got {np.round(got, 4)}, "
        f"expected {np.round(expected, 4)}"
    )


def fourier_upsampled_core_area(psf, pixel_lod, ratio, factor):
    """Core area of a PSF from its band-limited (zero-padded FFT) upsampling.

    A sampled PSF at or above Nyquist determines the continuous intensity,
    and zero-padding its spectrum evaluates that intensity on a grid
    ``factor`` times finer, so counting the finer cells above threshold
    converges to the continuous core area without a spatial kernel.
    """
    psf = np.asarray(psf, dtype=np.float64)
    n = psf.shape[0]
    big = n * factor
    spec = np.fft.fftshift(np.fft.fft2(psf))
    padded = np.zeros((big, big), dtype=complex)
    start = (big - n) // 2
    padded[start : start + n, start : start + n] = spec
    fine = np.real(np.fft.ifft2(np.fft.ifftshift(padded)))
    above = fine > ratio * fine.max()
    return above.sum() * (pixel_lod / factor) ** 2


SHIPPED_RATIO = 0.3


@pytest.fixture(scope="module")
def shipped_trunc_coro():
    """The shipped AAVC package at truncation ratio 0.3, or a skip that says why."""
    from yippy import fetch_yip

    try:
        yip_path = fetch_yip("eac1_aavc_2d")
    except Exception as exc:
        message = (
            f"data-absent: eac1_aavc_2d is not cached and could not be fetched ({exc})"
        )
        if REFERENCE_DATA_REQUIRED:
            pytest.fail(message)
        pytest.skip(message)
    return Coronagraph(yip_path, psf_trunc_ratio=SHIPPED_RATIO)


@pytest.mark.parametrize("separation_lod", (3.3, 4.7, 6.1, 9.4))
def test_shipped_core_area_matches_the_band_limited_psf(
    shipped_trunc_coro, separation_lod
):
    """On a shipped package the curve is the core area of the rendered PSF.

    The reference counts above-threshold cells of the synthesized PSF's
    band-limited reconstruction over the whole image, eight times finer than
    the package pixels.
    """
    from lod_unit import lod

    coro = shipped_trunc_coro
    pixel_lod = float(coro.pixel_scale_arcsec.value)
    y_offsets = np.asarray(coro.offax.y_offsets, dtype=float)
    y0 = float(y_offsets[np.argmin(np.abs(y_offsets))])
    psf = coro.offax(separation_lod * lod, y0 * lod)
    expected = fourier_upsampled_core_area(psf, pixel_lod, SHIPPED_RATIO, factor=8)
    got = coro.core_area(separation_lod)
    assert got == pytest.approx(expected, rel=AREA_TOL)
