"""Tests for scipy-to-interpax interpolation in EqxCoronagraph.

Verifies that ``_scipy_to_interpax`` faithfully converts scipy splines and
that all EqxCoronagraph performance methods agree with the original
scipy-based Coronagraph interpolators.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import requires_available_yip
from scipy.interpolate import make_interp_spline

from yippy import Coronagraph, EqxCoronagraph
from yippy.eqx_coronagraph import _scipy_to_interpax
from yippy.performance import _build_perf_interps

# =============================================================================
# Unit tests for _scipy_to_interpax helper
# =============================================================================


class TestScipyToInterpax:
    """Unit tests for _scipy_to_interpax with synthetic data."""

    @pytest.fixture()
    def sine_spline(self):
        """Create a scipy cubic spline of sin(x) on [0, 2*pi]."""
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


@requires_available_yip
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
        """Raw contrast should agree at the original performance table nodes."""
        scipy_spl = coro._log_contrast_interp
        knots = scipy_spl.pchip.x

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


@pytest.mark.parametrize("order", [None, 1])
def test_synthetic_performance_agrees_in_float64(synthetic_1d, order):
    """Default and legacy curves agree eagerly and under jit, including tails."""
    with jax.enable_x64():
        kwargs = {} if order is None else {"interp_order": order}
        coro = Coronagraph(synthetic_1d.path, **kwargs)
        eqx_coro = EqxCoronagraph(yippy_coro=coro)
        grid = np.array([-2.0, -0.1, 0.0, 0.13, 1.37, 3.29, 6.0, 10.0, 15.0])
        for name in [
            "throughput",
            "raw_contrast",
            "core_area",
            "occulter_transmission",
            "core_mean_intensity",
        ]:
            expected = getattr(coro, name)(grid)
            method = getattr(eqx_coro, name)
            np.testing.assert_allclose(method(grid), expected, rtol=1e-12, atol=1e-24)
            np.testing.assert_allclose(
                jax.jit(lambda x, method=method: method(x))(grid),
                expected,
                rtol=1e-12,
                atol=1e-24,
            )
        for diameter in [0.05, 0.1, 0.3]:
            expected = coro.core_mean_intensity(grid, stellar_diam=diameter)
            actual = jax.jit(lambda x, d: eqx_coro.core_mean_intensity(x, d))(
                grid, diameter
            )
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-24)
        derivative = jax.grad(lambda x: eqx_coro.throughput(x))(1.37)
        assert np.isfinite(derivative)
        interp = coro.throughput_interp
        expected_derivative = (interp.pchip if order is None else interp)(1.37, nu=1)
        np.testing.assert_allclose(derivative, expected_derivative, rtol=1e-12)


def test_pchip_conversion_uses_original_nodes():
    """Nonuniform PCHIP data survives conversion without a cubic spline refit."""
    with jax.enable_x64():
        nodes = np.array([0.0, 0.1, 0.3, 0.7, 1.5, 4.0, 8.0])
        values = np.array([0.0, 0.01, 0.03, 0.5, 0.9, 0.95, 1.0])
        coro = SimpleNamespace()
        _build_perf_interps(coro, nodes, values, nodes, values, nodes, values, None)
        scipy_interp = coro.throughput_interp
        converted = _scipy_to_interpax(scipy_interp)
        grid = np.linspace(nodes[0], nodes[-1], 101)
        np.testing.assert_allclose(
            converted(grid), scipy_interp(grid), rtol=1e-12, atol=1e-16
        )
        np.testing.assert_allclose(
            jax.jit(lambda x: converted(x))(grid),
            scipy_interp(grid),
            rtol=1e-12,
            atol=1e-16,
        )

        endpoints = nodes[[0, -1]]
        derivatives = jax.vmap(jax.grad(lambda x: converted(x)))(endpoints)
        np.testing.assert_allclose(
            derivatives, scipy_interp.pchip(endpoints, nu=1), rtol=1e-12
        )
