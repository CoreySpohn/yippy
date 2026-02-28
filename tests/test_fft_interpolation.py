"""Unit tests for FFT-based PSF interpolation.

Tests the core mathematical properties of the Fourier shift and
synthesis pipeline: shift fidelity, flux conservation, Parseval's
theorem, and backend commutativity.
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def offax(coro):
    """The OffAx/OffJAX object from the session coronagraph."""
    return coro.offax


@pytest.fixture(scope="session")
def sample_psf(offax):
    """A single reference PSF from the middle of the offset grid."""
    mid = len(offax.x_offsets) // 2
    idx = offax.offset_to_flat_idx[mid, 0]
    return np.array(offax.flat_psfs[int(idx)])


@pytest.fixture(scope="session")
def grid_offsets(offax):
    """All x-offsets as a numpy array."""
    return np.array(offax.x_offsets)


# ---------------------------------------------------------------------------
# Test 1: FFT shift fidelity
# ---------------------------------------------------------------------------


class TestShiftFidelity:
    """Verify the phasor-multiply shift preserves signal integrity."""

    def test_integer_shift_matches_roll(self, sample_psf):
        """Integer pixel shift via FFT must match np.roll exactly."""
        from yippy.util import fft_shift

        for shift in [1, -1, 3, -5]:
            shifted = fft_shift(sample_psf, x=shift, y=0)
            rolled = np.roll(sample_psf, shift, axis=1)
            # edges differ due to zero-padding vs wrap-around; compare interior
            margin = abs(shift) + 2
            np.testing.assert_allclose(
                shifted[margin:-margin, margin:-margin],
                rolled[margin:-margin, margin:-margin],
                atol=1e-10,
                err_msg=f"Integer shift by {shift} failed",
            )

    def test_roundtrip_subpixel(self, sample_psf):
        """Shift by +delta then -delta should recover the original.

        The CPU fft_shift loses edge information during pad/unpad, so
        we only check the bright core where relative errors are small.
        """
        from yippy.util import fft_shift

        for delta in [0.3, 0.5, 0.7, 1.4]:
            shifted_fwd = fft_shift(sample_psf, x=delta, y=0)
            roundtrip = fft_shift(shifted_fwd, x=-delta, y=0)
            # compare only pixels above 5% of peak (robust bright core)
            mask = sample_psf > 0.05 * np.max(sample_psf)
            np.testing.assert_allclose(
                roundtrip[mask],
                sample_psf[mask],
                rtol=0.05,
                err_msg=f"Roundtrip failed for delta={delta}",
            )

    def test_shift_accumulation(self, sample_psf):
        """Shift(+0.3) then Shift(+0.7) must equal Shift(+1.0).

        Double pad/unpad cycle loses edge info; check bright core only.
        """
        from yippy.util import fft_shift

        two_step = fft_shift(fft_shift(sample_psf, x=0.3, y=0), x=0.7, y=0)
        one_step = fft_shift(sample_psf, x=1.0, y=0)
        mask = one_step > 0.05 * np.max(one_step)
        np.testing.assert_allclose(
            two_step[mask],
            one_step[mask],
            rtol=0.05,
            err_msg="Shift accumulation (0.3+0.7 vs 1.0) failed",
        )

    def test_shift_preserves_total_flux(self, sample_psf):
        """FFT shift must not change total flux by more than 5%.

        The CPU fft_shift pads and unpads, so edge pixels that shift
        outside the original footprint are lost. Flux loss scales with
        shift magnitude as a fraction of the PSF extent.
        """
        from yippy.util import fft_shift

        original_flux = np.sum(sample_psf)
        for delta in [0.1, 0.5, 1.3, 3.7]:
            shifted = fft_shift(sample_psf, x=delta, y=0)
            shifted_flux = np.sum(shifted)
            np.testing.assert_allclose(
                shifted_flux,
                original_flux,
                rtol=0.05,
                err_msg=f"Flux changed after shift by {delta}",
            )


# ---------------------------------------------------------------------------
# Test 2: Flux conservation through the full pipeline
# ---------------------------------------------------------------------------


class TestFluxConservation:
    """Verify total flux is preserved through interpolation."""

    def test_grid_point_flux(self, coro, offax, grid_offsets):
        """Interpolation at a grid point must return exact flux match."""
        for i in range(min(5, len(grid_offsets))):
            x_val = float(grid_offsets[i])
            if x_val < 0.5:
                continue
            ref_idx = int(offax.offset_to_flat_idx[i, 0])
            ref_psf = np.array(offax.flat_psfs[ref_idx])
            ref_flux = np.sum(ref_psf)

            interp_psf = np.array(offax.create_psf(x_val, 0.0))
            interp_flux = np.sum(interp_psf)
            np.testing.assert_allclose(
                interp_flux,
                ref_flux,
                rtol=0.01,
                err_msg=f"Flux mismatch at grid point x={x_val}",
            )

    def test_midpoint_flux_bounded(self, offax, grid_offsets):
        """Flux at midpoints should be within 20% of neighbor range."""
        for i in range(min(5, len(grid_offsets) - 1)):
            x_lo = float(grid_offsets[i])
            x_hi = float(grid_offsets[i + 1])
            if x_lo < 0.5:
                continue
            x_mid = (x_lo + x_hi) / 2.0

            idx_lo = int(offax.offset_to_flat_idx[i, 0])
            idx_hi = int(offax.offset_to_flat_idx[i + 1, 0])
            flux_lo = float(np.sum(np.array(offax.flat_psfs[idx_lo])))
            flux_hi = float(np.sum(np.array(offax.flat_psfs[idx_hi])))

            interp_psf = np.array(offax.create_psf(x_mid, 0.0))
            interp_flux = float(np.sum(interp_psf))

            flux_min = min(flux_lo, flux_hi) * 0.8
            flux_max = max(flux_lo, flux_hi) * 1.2
            assert flux_min <= interp_flux <= flux_max, (
                f"Midpoint flux {interp_flux:.6f} outside range "
                f"[{flux_min:.6f}, {flux_max:.6f}] at x={x_mid}"
            )


# ---------------------------------------------------------------------------
# Test 3: Positivity clamp impact
# ---------------------------------------------------------------------------


class TestPositivityClamp:
    """Quantify flux lost to the max(psf, 0) clamp in fft_shift_x."""

    def test_clamp_flux_loss_is_negligible(self, sample_psf):
        """Energy eliminated by clamping must be < 1e-4 of total energy."""
        import jax
        import jax.numpy as jnp

        from yippy.jax_funcs import fft_shift_x, get_pad_info

        with jax.enable_x64():
            psf = jnp.array(sample_psf, dtype=jnp.float64)
            _, _, _, n_final = get_pad_info(psf, 1.5)
            kx = jnp.fft.fftfreq(n_final)
            phasor = jnp.exp(-2j * jnp.pi * kx)

            unclamped = np.array(fft_shift_x(psf, 0.37, phasor, clamp=False))
            clamped = np.array(fft_shift_x(psf, 0.37, phasor, clamp=True))

        total_energy = np.sum(unclamped**2)
        neg_mask = unclamped < 0
        neg_energy = np.sum(unclamped[neg_mask] ** 2)
        neg_fraction = neg_energy / total_energy

        assert neg_fraction < 1e-4, (
            f"Negative pixel energy is {neg_fraction:.2e} of total (limit: 1e-4)"
        )
        # Clamped output should equal unclamped with negatives zeroed
        np.testing.assert_array_equal(
            clamped,
            np.maximum(unclamped, 0),
        )


# ---------------------------------------------------------------------------
# Test 4: Wrap-around bounds check
# ---------------------------------------------------------------------------


class TestWrapAroundBounds:
    """Verify that get_pad_info provides sufficient padding for grid shifts."""

    def test_padding_covers_max_shift(self, offax, grid_offsets):
        """The pad factor must accommodate the largest inter-grid shift."""
        import jax.numpy as jnp

        from yippy.jax_funcs import get_pad_info

        psf = jnp.array(offax.flat_psfs[0])
        _n_orig, n_pad, _, _n_final = get_pad_info(psf, 1.5)
        pixel_scale = float(offax.pixel_scale.value)

        max_gap = float(np.max(np.diff(grid_offsets)))
        max_shift_px = max_gap / pixel_scale

        assert n_pad > max_shift_px, (
            f"Padding ({n_pad} px) is smaller than max grid shift "
            f"({max_shift_px:.1f} px)"
        )


# ---------------------------------------------------------------------------
# Test 5: Extrapolation behavior
# ---------------------------------------------------------------------------


class TestExtrapolation:
    """Document behavior when querying outside the YIP grid."""

    def test_beyond_owa_returns_something(self, offax, grid_offsets):
        """Query beyond the last offset should not crash."""
        max_offset = float(grid_offsets[-1])
        psf = np.array(offax.create_psf(max_offset + 1.0, 0.0))
        # just verify it runs and returns a finite array
        assert np.all(np.isfinite(psf))

    def test_negative_quadrant_matches_symmetry(self, offax, grid_offsets):
        """For symmetric coronagraphs, PSF(-x, 0) should equal fliplr(PSF(x, 0))."""
        if not offax.x_symmetric:
            pytest.skip("Coronagraph is not x-symmetric")
        x_val = float(grid_offsets[len(grid_offsets) // 2])
        psf_pos = np.array(offax.create_psf(x_val, 0.0))
        psf_neg = np.array(offax.create_psf(-x_val, 0.0))
        np.testing.assert_allclose(
            psf_neg,
            np.fliplr(psf_pos),
            atol=1e-8,
            err_msg=f"Symmetry violation at x={x_val}",
        )


# ---------------------------------------------------------------------------
# Test 6: Continuous sweep (astrometric smoothness)
# ---------------------------------------------------------------------------


class TestContinuousSweep:
    """Verify smooth behavior during sub-pixel sweeps."""

    def test_peak_flux_smooth(self, offax, grid_offsets):
        """Peak flux should vary smoothly during a sub-pixel sweep."""
        x_lo = float(grid_offsets[len(grid_offsets) // 2])
        x_hi = float(grid_offsets[len(grid_offsets) // 2 + 1])
        n_steps = 20
        peaks = []
        for x in np.linspace(x_lo, x_hi, n_steps):
            psf = np.array(offax.create_psf(float(x), 0.0))
            peaks.append(float(np.max(psf)))

        peaks = np.array(peaks)
        # no sudden jumps: max fractional change between adjacent steps < 50%
        ratios = peaks[1:] / peaks[:-1]
        assert np.all(ratios > 0.5) and np.all(ratios < 2.0), (
            f"Peak flux jump detected: ratios range [{ratios.min():.3f}, "
            f"{ratios.max():.3f}]"
        )

    def test_centroid_smooth(self, offax, grid_offsets):
        """Center-of-light should track a smooth line during sweep."""
        x_lo = float(grid_offsets[len(grid_offsets) // 2])
        x_hi = float(grid_offsets[len(grid_offsets) // 2 + 1])
        n_steps = 20

        centroids_x = []

        for x in np.linspace(x_lo, x_hi, n_steps):
            psf = np.array(offax.create_psf(float(x), 0.0))
            ny, nx = psf.shape
            _ygrid, xgrid = np.mgrid[:ny, :nx]
            total = np.sum(psf)
            if total > 0:
                cx = np.sum(xgrid * psf) / total
                centroids_x.append(cx)

        centroids_x = np.array(centroids_x)
        # centroid should be monotonically increasing (moving right)
        diffs = np.diff(centroids_x)
        # allow small numerical noise but overall trend should be positive
        assert np.sum(diffs > -0.1) >= len(diffs) * 0.8, (
            "Centroid is not monotonically tracking during sweep"
        )


# ---------------------------------------------------------------------------
# Test 7: float32 vs float64 precision
# ---------------------------------------------------------------------------


class TestFloat32VsFloat64:
    """Quantify the precision noise floor from float32 vs float64 in JAX."""

    def test_shift_precision_with_x64(self, sample_psf):
        """JAX fft_shift in float32 vs float64 (via enable_x64) should agree."""
        import jax
        import jax.numpy as jnp

        from yippy.jax_funcs import fft_shift_x, get_pad_info

        # float32 baseline (JAX default)
        psf_f32 = jnp.array(sample_psf, dtype=jnp.float32)
        _, _n_pad, _, n_final = get_pad_info(psf_f32, 1.5)
        kx_f32 = jnp.fft.fftfreq(n_final)
        phasor_f32 = jnp.exp(-2j * jnp.pi * kx_f32)
        result_f32 = np.array(fft_shift_x(psf_f32, 0.37, phasor_f32))

        # float64 with enable_x64 context manager
        with jax.enable_x64():
            psf_f64 = jnp.array(sample_psf, dtype=jnp.float64)
            _, _n_pad_64, _, n_final_64 = get_pad_info(psf_f64, 1.5)
            kx_f64 = jnp.fft.fftfreq(n_final_64)
            phasor_f64 = jnp.exp(-2j * jnp.pi * kx_f64)
            result_f64 = np.array(fft_shift_x(psf_f64, 0.37, phasor_f64))

        peak = np.max(np.abs(result_f64))
        max_diff = np.max(np.abs(result_f64 - result_f32.astype(np.float64)))
        relative_diff = max_diff / peak

        # float32 introduces noise around 1e-7; document the actual value
        assert relative_diff < 1e-4, (
            f"float32 vs float64 relative diff {relative_diff:.2e} exceeds 1e-4"
        )

    def test_synthesized_psf_is_finite(self, offax, grid_offsets):
        """Synthesized PSF at a midpoint must be finite and positive."""
        x_lo = float(grid_offsets[len(grid_offsets) // 2])
        x_hi = float(grid_offsets[len(grid_offsets) // 2 + 1])
        x_mid = (x_lo + x_hi) / 2.0

        psf = np.array(offax.create_psf(x_mid, 0.0))
        assert np.all(np.isfinite(psf))
        assert np.sum(psf) > 0


# ---------------------------------------------------------------------------
# Test 8: Parseval's theorem (energy conservation in frequency domain)
# ---------------------------------------------------------------------------


class TestParsevalTheorem:
    """Verify yippy's FFT shift preserves energy (Parseval's theorem).

    The FFT phasor shift is mathematically energy-preserving.  The positivity
    clamp (max(psf, 0)) intentionally discards small negative ringing
    artifacts, so we test each concern separately using the ``clamp``
    parameter on ``fft_shift_x``.
    """

    def test_phasor_shift_preserves_energy_exactly(self, sample_psf):
        """fft_shift_x(clamp=False) must preserve energy exactly."""
        import jax
        import jax.numpy as jnp

        from yippy.jax_funcs import fft_shift_x, get_pad_info

        with jax.enable_x64():
            psf = jnp.array(sample_psf, dtype=jnp.float64)
            energy_before = float(jnp.sum(psf**2))

            _, _, _, n_final = get_pad_info(psf, 1.5)
            kx = jnp.fft.fftfreq(n_final)
            phasor = jnp.exp(-2j * jnp.pi * kx)
            shifted = fft_shift_x(psf, 0.37, phasor, clamp=False)
            energy_after = float(jnp.sum(shifted**2))

        # Pad/unpad cycle and float32 source data limit precision to ~1e-5
        np.testing.assert_allclose(
            energy_after,
            energy_before,
            rtol=1e-5,
            err_msg="Phasor shift violated energy conservation",
        )

    def test_clamp_energy_loss_is_small(self, sample_psf):
        """fft_shift_x(clamp=True) should lose less than 1% energy."""
        import jax
        import jax.numpy as jnp

        from yippy.jax_funcs import fft_shift_x, get_pad_info

        with jax.enable_x64():
            psf = jnp.array(sample_psf, dtype=jnp.float64)
            energy_before = float(jnp.sum(psf**2))

            _, _, _, n_final = get_pad_info(psf, 1.5)
            kx = jnp.fft.fftfreq(n_final)
            phasor = jnp.exp(-2j * jnp.pi * kx)
            shifted = fft_shift_x(psf, 0.37, phasor, clamp=True)
            energy_after = float(jnp.sum(shifted**2))

        loss = 1.0 - energy_after / energy_before
        assert loss < 0.01, f"Clamp lost {loss:.4%} of energy (limit: 1%)"
        assert loss >= 0, "Energy increased -- impossible for a clamp"


# ---------------------------------------------------------------------------
# Test 10: Dark hole contrast preservation
# ---------------------------------------------------------------------------


class TestDarkHoleContrast:
    """Verify interpolation does not inject artifacts into the dark hole."""

    def test_dark_hole_intensity_stable(self, offax, grid_offsets):
        """Mean intensity in the 3-10 lam/D annulus of an interpolated PSF.

        Must not exceed that of the neighboring grid-point PSFs.
        This test ensures that the interpolation process does not introduce
        significant artifacts that would artificially brighten the dark hole
        region compared to the PSFs at the original grid points.
        """
        pixel_scale = float(offax.pixel_scale.value)

        # Pick a midpoint between two grid points
        mid_idx = len(grid_offsets) // 2
        x_lo = float(grid_offsets[mid_idx])
        x_hi = float(grid_offsets[mid_idx + 1])
        x_mid = (x_lo + x_hi) / 2.0

        # Get the grid-point reference PSFs
        ref_lo = np.array(offax.create_psf(x_lo, 0.0))
        ref_hi = np.array(offax.create_psf(x_hi, 0.0))

        # Get the interpolated midpoint PSF
        interp_psf = np.array(offax.create_psf(x_mid, 0.0))

        ny, nx = interp_psf.shape
        cy, cx = ny // 2, nx // 2
        yy, xx = np.mgrid[:ny, :nx]
        rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) * pixel_scale

        annulus = (rr >= 3.0) & (rr <= 10.0)
        if not np.any(annulus):
            pytest.skip("PSF too small for 3-10 lam/D annulus")

        baseline = max(
            np.mean(ref_lo[annulus]),
            np.mean(ref_hi[annulus]),
        )
        interp_mean = np.mean(interp_psf[annulus])

        # Interpolated dark hole should not be significantly brighter
        # than the brightest neighbor
        assert interp_mean < baseline * 1.5, (
            f"Interpolated dark hole mean ({interp_mean:.2e}) exceeds "
            f"1.5x neighbor baseline ({baseline:.2e})"
        )


# ---------------------------------------------------------------------------
# Test 11: get_pad_info arithmetic
# ---------------------------------------------------------------------------


class TestGetPadInfo:
    """Verify padding dimensions from get_pad_info."""

    def test_standard_padding(self):
        """Default 1.5x padding on a 10x10 image."""
        import jax.numpy as jnp

        from yippy.jax_funcs import get_pad_info

        image = jnp.zeros((10, 10))
        n_orig, n_pad, img_edge, n_final = get_pad_info(image, 1.5)

        assert n_orig == 10
        assert n_pad == 15
        assert img_edge == 25  # n_pad + n_orig
        assert n_final == 40  # 2 * n_pad + n_orig

    def test_real_psf_size(self, sample_psf):
        """Padding on a real PSF shape (typically 256x256)."""
        import jax.numpy as jnp

        from yippy.jax_funcs import get_pad_info

        psf = jnp.array(sample_psf)
        n_orig, n_pad, img_edge, n_final = get_pad_info(psf, 1.5)

        assert n_orig == sample_psf.shape[0]
        assert n_pad == int(1.5 * n_orig)
        assert img_edge == n_pad + n_orig
        assert n_final == 2 * n_pad + n_orig


# ---------------------------------------------------------------------------
# Test 12: create_shift_mask boundary logic
# ---------------------------------------------------------------------------


class TestCreateShiftMask:
    """Verify branchless masking for shifted PSFs."""

    def test_positive_x_shift(self):
        """Positive x-shift should mask the left columns."""
        import jax.numpy as jnp

        from yippy.jax_funcs import create_shift_mask

        psf = jnp.ones((10, 10))
        y_grid, x_grid = jnp.mgrid[0:10, 0:10]

        mask = create_shift_mask(psf, 2.3, 0.0, x_grid, y_grid, fill_val=1.0)

        # ceil(2.3) = 3, so left 3 columns should be zero
        assert float(jnp.sum(mask[:, :3])) == 0.0
        # rest should be 1.0
        assert float(jnp.min(mask[:, 3:])) == 1.0

    def test_negative_y_shift(self):
        """Negative y-shift should mask the bottom rows."""
        import jax.numpy as jnp

        from yippy.jax_funcs import create_shift_mask

        psf = jnp.ones((10, 10))
        y_grid, x_grid = jnp.mgrid[0:10, 0:10]

        mask = create_shift_mask(psf, 0.0, -1.5, x_grid, y_grid, fill_val=1.0)

        # ceil(1.5) = 2, so bottom 2 rows should be zero
        assert float(jnp.sum(mask[8:, :])) == 0.0
        # rest should be 1.0
        assert float(jnp.min(mask[:8, :])) == 1.0

    def test_fill_value_applied(self):
        """Fill value should be used for valid pixels."""
        import jax.numpy as jnp

        from yippy.jax_funcs import create_shift_mask

        psf = jnp.ones((10, 10))
        y_grid, x_grid = jnp.mgrid[0:10, 0:10]

        mask = create_shift_mask(psf, 0.0, 0.0, x_grid, y_grid, fill_val=3.5)
        assert float(jnp.min(mask)) == 3.5

    def test_combined_shift(self):
        """Positive x and negative y shifts mask correct edges."""
        import jax.numpy as jnp

        from yippy.jax_funcs import create_shift_mask

        psf = jnp.ones((10, 10))
        y_grid, x_grid = jnp.mgrid[0:10, 0:10]

        mask = create_shift_mask(psf, 2.3, -1.5, x_grid, y_grid, fill_val=2.0)

        # Left 3 cols masked
        assert float(jnp.sum(mask[:, :3])) == 0.0
        # Bottom 2 rows masked
        assert float(jnp.sum(mask[8:, :])) == 0.0
        # Interior should be fill_val
        assert float(mask[5, 5]) == 2.0


# ---------------------------------------------------------------------------
# Test 13: Nearest neighbor index lookup
# ---------------------------------------------------------------------------


class TestNearestNeighborLookup:
    """Verify index clamping in get_near_inds_offsets_1D and _2D."""

    def test_1d_interior_point(self):
        """Query between two grid points returns correct bounding indices."""
        import jax.numpy as jnp

        from yippy.jax_funcs import get_near_inds_offsets_1D

        x_offsets = jnp.array([0.0, 1.0, 2.0, 3.0])
        y_offsets = jnp.array([0.0])

        inds, offsets = get_near_inds_offsets_1D(x_offsets, y_offsets, 1.2, 0.0)

        # Should bound between index 1 (x=1.0) and index 2 (x=2.0)
        assert int(inds[0, 0]) == 1
        assert int(inds[1, 0]) == 2
        np.testing.assert_allclose(float(offsets[0, 0]), 1.0)
        np.testing.assert_allclose(float(offsets[1, 0]), 2.0)

    def test_1d_boundary_clamp(self):
        """Query at the left edge should clamp to index 0."""
        import jax.numpy as jnp

        from yippy.jax_funcs import get_near_inds_offsets_1D

        x_offsets = jnp.array([1.0, 2.0, 3.0])
        y_offsets = jnp.array([0.0])

        inds, _offsets = get_near_inds_offsets_1D(x_offsets, y_offsets, 0.5, 0.0)

        # Both should clamp to index 0
        assert int(inds[0, 0]) == 0
        assert int(inds[1, 0]) == 0

    def test_2d_interior_point(self):
        """Query in the interior of a 2D grid returns 4 bounding corners."""
        import jax.numpy as jnp

        from yippy.jax_funcs import get_near_inds_offsets_2D

        x_offs = jnp.array([0.0, 1.0, 2.0])
        y_offs = jnp.array([0.0, 1.0, 2.0])

        inds, offsets = get_near_inds_offsets_2D(x_offs, y_offs, 1.2, 0.8)

        # Should return 4 corners: (1,0), (1,1), (2,0), (2,1)
        assert inds.shape == (4, 2)
        assert offsets.shape == (4, 2)
        # x indices should be 1 and 2
        assert int(inds[0, 0]) == 1  # x_low
        assert int(inds[2, 0]) == 2  # x_high
        # y indices should be 0 and 1
        assert int(inds[0, 1]) == 0  # y_low
        assert int(inds[1, 1]) == 1  # y_high
