"""What interpolation between tabulated offsets does to flux, and where the axis is.

The reader synthesizes a PSF at any position by combining the tabulated ones,
and a consumer treats the result as a photometric quantity. Two things have to
hold for that to be safe: the combination must not create or destroy light
relative to the PSFs it combined, and every path through the reader must agree
about where the optical axis sits.

The second is currently not true, and the disagreement is recorded here rather
than papered over: the separation map places the axis at the header's centre
while off-axis synthesis mirrors about half a pixel away from it.
"""

import numpy as np
import pytest
from lod_unit import lod

# Positions between tabulated knots, where the reader has to interpolate. The
# knot spacing of the synthetic package is 0.5 lambda/D.
KNOT_SPACING_LOD = 0.5
BETWEEN_KNOTS_LOD = (1.25, 2.25, 3.75, 4.25)


def _sum(coro, x_lod, y_lod=0.0):
    """Total flux of the synthesized PSF at a sky position."""
    return float(np.asarray(coro.offax(x_lod * lod, y_lod * lod)).sum())


def test_interpolated_flux_is_bracketed_by_its_two_neighbours(syn_coro_1d):
    """A PSF between two knots holds between their two fluxes.

    The radial path combines the two bracketing PSFs with weights that sum to
    one, so the result cannot hold more light than the brighter neighbour or
    less than the dimmer one. A weighting bug that renormalized, dropped a
    neighbour, or let the weights sum to something other than one shows up
    here and nowhere else, because every shape and ratio test in the suite is
    blind to an overall scale.

    The audit found the original version of this check executing zero
    assertions, so the loop counts what it did.
    """
    coro = syn_coro_1d
    checked = 0

    for separation in BETWEEN_KNOTS_LOD:
        low = np.floor(separation / KNOT_SPACING_LOD) * KNOT_SPACING_LOD
        high = np.ceil(separation / KNOT_SPACING_LOD) * KNOT_SPACING_LOD
        assert low < separation < high, "the probe landed on a knot, not between two"

        lower_sum = _sum(coro, low)
        upper_sum = _sum(coro, high)
        interpolated = _sum(coro, separation)

        assert min(lower_sum, upper_sum) <= interpolated <= max(lower_sum, upper_sum), (
            f"the PSF at {separation} holds {interpolated:.6f}, outside the "
            f"[{min(lower_sum, upper_sum):.6f}, {max(lower_sum, upper_sum):.6f}] "
            "its two bracketing knots hold"
        )
        checked += 1

    assert checked == len(BETWEEN_KNOTS_LOD), (
        "the bracketing loop skipped separations, so it asserted less than it claims"
    )


def test_flux_grows_with_separation_as_the_occulter_opens(syn_coro_1d, synthetic_1d):
    """More light gets through further out, monotonically.

    The occulter transmission is monotone in separation by construction, and
    nothing in the reader should reverse that. This is the cheapest check that
    the separation is being used with the right sign, and unlike the direction
    anchors it is insensitive to where the blob lands.
    """
    separations = np.arange(0.5, 6.0, 0.5)
    sums = np.array([_sum(syn_coro_1d, float(s)) for s in separations])

    assert np.all(np.diff(sums) > 0), (
        f"synthesized flux is not monotone in separation: {sums}"
    )
    expected_ratio = float(synthetic_1d.occulter(5.0) / synthetic_1d.occulter(1.0))
    got_ratio = _sum(syn_coro_1d, 5.0) / _sum(syn_coro_1d, 1.0)
    assert got_ratio == pytest.approx(expected_ratio, rel=0.02), (
        "the ratio of synthesized fluxes does not follow the occulter law"
    )


def test_a_tabulated_offset_is_returned_without_interpolation(
    syn_coro_1d, synthetic_1d
):
    """At a knot the reader should hand back what the package holds.

    A query that lands exactly on a tabulated offset needs no combination, and
    the total flux there is the package's own value for that PSF. This
    separates an interpolation error from a loading error: if this fails, the
    problem is upstream of any weighting.
    """
    package = synthetic_1d
    for separation in (1.0, 2.0, 3.0):
        expected = float(package.occulter(separation) * package.psf_flux)
        got = _sum(syn_coro_1d, separation)
        # The array truncates the Gaussian's wings, so the sum is slightly
        # below the analytic total flux; well inside the field that deficit is
        # far smaller than a percent.
        assert got == pytest.approx(expected, rel=0.01), (
            f"the PSF tabulated at {separation} holds {got:.6f}, not the "
            f"{expected:.6f} the package was written with"
        )


def test_the_separation_map_puts_zero_at_the_declared_centre(syn_coro_1d, synthetic_1d):
    """The separation map reads zero exactly at the header's optical axis."""
    centre = int(synthetic_1d.center)
    separation_map = np.asarray(syn_coro_1d.separation_map())

    assert separation_map[centre, centre] == 0.0, (
        f"the separation map reads {separation_map[centre, centre]} at the "
        f"declared centre ({centre}, {centre}), not zero"
    )
    row, column = np.unravel_index(np.argmin(separation_map), separation_map.shape)
    assert (row, column) == (centre, centre), (
        f"the separation map's minimum is at ({row}, {column}), not at the "
        f"declared centre ({centre}, {centre})"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Two paths through the reader disagree about the optical axis. The "
        "separation map places it at the header's XCENTER, npix / 2, and reads "
        "exactly zero there. Off-axis synthesis mirrors with a whole-array "
        "flip, which is symmetric about (npix - 1) / 2 instead, so a source at "
        "-s lands one pixel closer to the axis than one at +s. Both cannot be "
        "right, and a consumer converting a sky position to a pixel has no way "
        "to tell which path it inherited. Same defect as the anchor in "
        "test_direction_anchor.py, asserted here through the map so that "
        "whichever convention is chosen has to make both paths agree."
    ),
)
def test_both_paths_agree_about_where_the_axis_is(syn_coro_1d, synthetic_1d):
    """The centre the map uses is the centre synthesis mirrors about."""
    separation_map = np.asarray(syn_coro_1d.separation_map())
    map_centre = np.unravel_index(np.argmin(separation_map), separation_map.shape)[1]

    columns = []
    for separation in (2.0, 3.0):
        for sign in (+1.0, -1.0):
            psf = np.asarray(syn_coro_1d.offax(sign * separation * lod, 0.0 * lod))
            columns.append(np.unravel_index(np.argmax(psf), psf.shape)[1])
    synthesis_centre = float(np.mean(columns))

    assert synthesis_centre == pytest.approx(float(map_centre)), (
        f"the separation map is centred on column {map_centre} while off-axis "
        f"synthesis is centred on {synthesis_centre}"
    )


# The level of the four pixels that straddle the optical axis in the
# axis-concentrated stellar map, and the faint level everywhere else. Both are
# positive because the stellar-intensity reader interpolates in log space.
AXIS_BLOCK_LEVEL = 1.0
AXIS_BACKGROUND_LEVEL = 1e-4


@pytest.fixture(scope="module")
def axis_concentrated_coro(tmp_path_factory):
    """A package whose stellar leakage sits on the four pixels around the axis.

    The header follows the shipped packages: ``XCENTER = YCENTER = npix / 2``
    counts pixels from the array's outer corner, so with integer indices
    labelling pixel centres the optical axis is at index ``(npix - 1) / 2``,
    between four pixels. Every shipped stellar map is point-symmetric about
    that index and not about ``npix / 2``.
    """
    import astropy.io.fits as pyfits
    from synthetic_yip import write_synthetic_yip

    from yippy import Coronagraph

    package = write_synthetic_yip(tmp_path_factory.mktemp("axis") / "axis_concentrated")
    path = package.path / "stellar_intens.fits"
    with pyfits.open(path) as hdul:
        header = hdul[0].header.copy()
        maps = np.full_like(hdul[0].data, AXIS_BACKGROUND_LEVEL)
    axis = (package.npix - 1) // 2
    maps[:, axis : axis + 2, axis : axis + 2] = AXIS_BLOCK_LEVEL
    pyfits.PrimaryHDU(maps, header=header).writeto(path, overwrite=True)
    return Coronagraph(package.path)


def test_core_mean_intensity_is_measured_about_the_optical_axis(
    axis_concentrated_coro,
):
    """The innermost core mean intensity is the light right at the axis.

    The four pixels around the axis are the only ones within one pixel of it,
    so the azimuthal mean intensity at the innermost tabulated separation is
    exactly their level, and the next separation outward sees only the
    background. A profile centred half a pixel off the axis on each axis mixes
    the block with the background in its first bin and reads well below the
    block level.
    """
    coro = axis_concentrated_coro
    separations, profiles = coro.core_mean_intensity_curve(plot=False)
    profile = profiles[next(iter(profiles))]

    assert float(coro.core_mean_intensity(separations[0])) == pytest.approx(
        AXIS_BLOCK_LEVEL, rel=1e-6
    ), "the innermost mean intensity is not the level of the pixels at the axis"
    assert profile[0] == pytest.approx(AXIS_BLOCK_LEVEL, rel=1e-6)
    assert profile[1] == pytest.approx(AXIS_BACKGROUND_LEVEL, rel=1e-6), (
        "light from the pixels at the axis leaked into the second radial bin"
    )
