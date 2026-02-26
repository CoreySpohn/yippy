"""Tests for scipy-to-interpax interpolation in EqxCoronagraph.

Verifies that ``_scipy_to_interpax`` faithfully converts scipy splines and
that all EqxCoronagraph performance methods agree with the original
scipy-based Coronagraph interpolators.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import make_interp_spline

from yippy.eqx_coronagraph import _scipy_to_interpax

# =============================================================================
# Unit tests for _scipy_to_interpax helper
# =============================================================================


class TestScipyToInterpax:
    """Unit tests for _scipy_to_interpax with synthetic data."""

    @pytest.fixture()
    def sine_spline(self):
        """Create a scipy cubic spline of sin(x) on [0, 2π]."""
        x = np.linspace(0, 2 * np.pi, 50)
        y = np.sin(x)
        return make_interp_spline(x, y, k=3), x, y

    def test_agrees_at_original_knots(self, sine_spline):
        """Interpax spline should match scipy at the data points."""
        scipy_spl, x, y = sine_spline
        interpax_spl = _scipy_to_interpax(scipy_spl)

        for xi, yi in zip(x, y, strict=True):
            val = float(interpax_spl(jnp.float32(xi)))
            np.testing.assert_allclose(val, yi, rtol=1e-4, atol=1e-6)

    def test_agrees_at_midpoints(self, sine_spline):
        """Interpax spline should match scipy at midpoints."""
        scipy_spl, x, _ = sine_spline
        interpax_spl = _scipy_to_interpax(scipy_spl)

        mids = (x[:-1] + x[1:]) / 2.0
        for xi in mids:
            expected = float(scipy_spl(xi))
            actual = float(interpax_spl(jnp.float32(xi)))
            np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-6)

    def test_edge_values(self, sine_spline):
        """First and last knot values should be reproduced."""
        scipy_spl, x, y = sine_spline
        interpax_spl = _scipy_to_interpax(scipy_spl)

        np.testing.assert_allclose(
            float(interpax_spl(jnp.float32(x[0]))),
            y[0],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            float(interpax_spl(jnp.float32(x[-1]))),
            y[-1],
            atol=1e-5,
        )

    def test_monotonic_data(self):
        """Conversion should work for monotonically increasing data."""
        x = np.linspace(1.0, 10.0, 30)
        y = np.log(x)
        scipy_spl = make_interp_spline(x, y, k=3)
        interpax_spl = _scipy_to_interpax(scipy_spl)

        test_x = np.linspace(1.5, 9.5, 20)
        for xi in test_x:
            expected = float(scipy_spl(xi))
            actual = float(interpax_spl(jnp.float32(xi)))
            np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=1e-6)


# =============================================================================
# Integration tests: EqxCoronagraph vs Coronagraph (real data)
# =============================================================================


class TestEqxCoronagraphInterpolation:
    """Compare EqxCoronagraph to Coronagraph scipy interpolators."""

    @pytest.fixture()
    def test_separations(self, coro):
        """20 test separations between IWA and OWA."""
        iwa = float(coro.IWA.value)
        owa = float(coro.OWA.value)
        return np.linspace(iwa, owa, 20)

    def test_throughput(self, coro, eqx_coro, test_separations):
        """Throughput should match scipy throughput_interp."""
        for sep in test_separations:
            expected = float(coro.throughput_interp(sep))
            actual = float(eqx_coro.throughput(sep))
            np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-8)

    def test_raw_contrast_at_knots(self, coro, eqx_coro):
        """Raw contrast should agree at original spline data knots.

        ``_scipy_to_interpax`` re-evaluates the scipy spline at its
        interior knots and builds a new interpax spline from those
        (x, y) pairs. Agreement at these points verifies the
        conversion captured the correct values.
        """
        scipy_spl = coro._log_contrast_interp
        k = scipy_spl.k
        knots = np.unique(scipy_spl.t[k:-k])

        original_floor = eqx_coro.contrast_floor
        object.__setattr__(eqx_coro, "contrast_floor", None)

        try:
            for x in knots:
                expected = float(10.0 ** scipy_spl(x))
                actual = float(eqx_coro.raw_contrast(x))
                np.testing.assert_allclose(
                    actual,
                    expected,
                    rtol=5e-3,
                )
        finally:
            object.__setattr__(eqx_coro, "contrast_floor", original_floor)

    def test_occulter_transmission(self, coro, eqx_coro, test_separations):
        """Occulter transmission should match scipy."""
        for sep in test_separations:
            expected = float(coro.occ_trans_interp(sep))
            actual = float(eqx_coro.occulter_transmission(sep))
            np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-8)

    def test_core_area(self, coro, eqx_coro, test_separations):
        """Core area should match scipy core_area_interp."""
        for sep in test_separations:
            expected = float(coro.core_area_interp(sep))
            actual = float(eqx_coro.core_area(sep))
            np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-8)

    def test_core_mean_intensity(self, coro, eqx_coro, test_separations):
        """Core mean intensity should match scipy."""
        for sep in test_separations:
            expected = float(coro.core_intensity_interp(sep))
            actual = float(eqx_coro.core_mean_intensity(sep))
            np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-8)
