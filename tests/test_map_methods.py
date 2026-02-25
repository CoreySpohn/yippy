"""Tests for Coronagraph 2D map projection methods.

Uses the session-scoped ``coro`` fixture from conftest.py.
"""

import numpy as np
import pytest


class TestSeparationMap:
    """Tests for Coronagraph.separation_map."""

    def test_shape(self, coro):
        """Output shape matches (npix, npix)."""
        r = coro.separation_map()
        assert r.shape == (coro.npixels, coro.npixels)

    def test_center_is_zero(self, coro):
        """Separation at the grid center is zero."""
        r = coro.separation_map()
        cy, cx = round(coro.header.ycenter), round(coro.header.xcenter)
        assert r[cy, cx] == pytest.approx(0, abs=coro.pixel_scale.value)

    def test_radially_symmetric(self, coro):
        """Values at equal pixel distance from center should be equal."""
        r = coro.separation_map()
        cy, cx = round(coro.header.ycenter), round(coro.header.xcenter)
        d = 10
        assert r[cy, cx + d] == pytest.approx(r[cy + d, cx], rel=1e-10)
        assert r[cy, cx + d] == pytest.approx(r[cy, cx - d], rel=1e-10)

    def test_all_nonnegative(self, coro):
        """All separations are non-negative."""
        r = coro.separation_map()
        assert np.all(r >= 0)


class TestCoreMeanIntensityMap:
    """Tests for Coronagraph.core_mean_intensity_map."""

    def test_shape(self, coro):
        """Output shape matches (npix, npix)."""
        m = coro.core_mean_intensity_map()
        assert m.shape == (coro.npixels, coro.npixels)

    def test_radially_symmetric(self, coro):
        """Output should be perfectly radially symmetric."""
        m = coro.core_mean_intensity_map()
        cy, cx = round(coro.header.ycenter), round(coro.header.xcenter)
        d = 10
        assert m[cy, cx + d] == pytest.approx(m[cy + d, cx], rel=1e-10)
        assert m[cy, cx + d] == pytest.approx(m[cy - d, cx], rel=1e-10)

    def test_positive_values(self, coro):
        """All values are non-negative."""
        m = coro.core_mean_intensity_map()
        assert np.all(m >= 0)

    def test_matches_1d(self, coro):
        """Map values should match point-query at the same separation."""
        r = coro.separation_map()
        m = coro.core_mean_intensity_map()
        cy, cx = round(coro.header.ycenter), round(coro.header.xcenter)
        d = 15
        sep = r[cy, cx + d]
        expected = coro.core_mean_intensity(sep)
        assert m[cy, cx + d] == pytest.approx(expected, rel=1e-10)


class TestNoiseFloorAyoMap:
    """Tests for Coronagraph.noise_floor_ayo_map."""

    def test_shape(self, coro):
        """Output shape matches (npix, npix)."""
        m = coro.noise_floor_ayo_map(ppf=30.0)
        assert m.shape == (coro.npixels, coro.npixels)

    def test_ppf_scaling(self, coro):
        """Result should scale inversely with ppf."""
        m30 = coro.noise_floor_ayo_map(ppf=30.0)
        m60 = coro.noise_floor_ayo_map(ppf=60.0)
        np.testing.assert_allclose(m30, 2.0 * m60, rtol=1e-12)

    def test_equals_istar_over_ppf(self, coro):
        """Should be exactly core_mean_intensity_map / ppf."""
        ppf = 25.0
        istar = coro.core_mean_intensity_map()
        nf = coro.noise_floor_ayo_map(ppf=ppf)
        np.testing.assert_allclose(nf, istar / ppf, rtol=1e-12)


class TestThroughputMap:
    """Tests for Coronagraph.throughput_map."""

    def test_shape(self, coro):
        """Output shape matches (npix, npix)."""
        m = coro.throughput_map()
        assert m.shape == (coro.npixels, coro.npixels)

    def test_bounded(self, coro):
        """All values are in [0, 1]."""
        m = coro.throughput_map()
        assert np.all(m >= 0)
        assert np.all(m <= 1.0)

    def test_radially_symmetric(self, coro):
        """Equidistant points from center have equal throughput."""
        m = coro.throughput_map()
        cy, cx = round(coro.header.ycenter), round(coro.header.xcenter)
        d = 10
        assert m[cy, cx + d] == pytest.approx(m[cy + d, cx], rel=1e-10)

    def test_matches_1d(self, coro):
        """Map values match the 1D interpolator at the same separation."""
        r = coro.separation_map()
        m = coro.throughput_map()
        cy, cx = round(coro.header.ycenter), round(coro.header.xcenter)
        d = 15
        sep = r[cy, cx + d]
        expected = coro.throughput(sep)
        assert m[cy, cx + d] == pytest.approx(expected, rel=1e-10)


class TestCoreAreaMap:
    """Tests for Coronagraph.core_area_map."""

    def test_shape(self, coro):
        """Output shape matches (npix, npix)."""
        m = coro.core_area_map()
        assert m.shape == (coro.npixels, coro.npixels)

    def test_positive(self, coro):
        """All core areas are non-negative."""
        m = coro.core_area_map()
        assert np.all(m >= 0)

    def test_matches_1d(self, coro):
        """Map values match the 1D interpolator at the same separation."""
        r = coro.separation_map()
        m = coro.core_area_map()
        cy, cx = round(coro.header.ycenter), round(coro.header.xcenter)
        d = 15
        sep = r[cy, cx + d]
        expected = coro.core_area(sep)
        assert m[cy, cx + d] == pytest.approx(expected, rel=1e-10)
