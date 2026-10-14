"""The core area is the area of the PSF core the image simulator renders.

With a PSF truncation ratio ``r`` the core of an off-axis PSF is the region
where its intensity exceeds ``r`` times its peak, and ``Coronagraph.core_area``
is the solid angle of that region in (lambda/D)^2. It is the photometric
aperture an exposure-time calculator multiplies the leaked starlight and the
background by, so at every separation, on or between the tabulated offsets, it
must be the core area of the PSF yippy synthesizes for a source there.

The references here do not come from the code under test. A circular Gaussian
of standard deviation sigma has the core area ``2 pi sigma^2 ln(1 / r)``.
"""

import numpy as np
import pytest

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
