"""
Tests for src/response_propagator.py.

Validates:
  - OU response kernel: causality, normalisation, decay rate
  - Rough response kernel: causality, singularity structure, H-scaling
  - Bucketed vega (bump-and-revalue): shape and sign
  - Bucketed vega (response propagator): shape and consistency with BAV
"""

import sys

sys.path.insert(0, '/workspace/project')

import numpy as np
import pytest

from src.response_propagator import (
    ou_response_kernel,
    rough_response_kernel,
    compute_bucketed_vega_bump,
    compute_bucketed_vega_response,
)


# ---------------------------------------------------------------------------
# OU response kernel
# ---------------------------------------------------------------------------

class TestOUResponseKernel:
    def test_causality_zero_before_source(self):
        """R(t, s) = 0 for t ≤ s (strict causality)."""
        t_arr = np.array([0.0, 0.3, 0.5])
        s = 0.6
        R = ou_response_kernel(t_arr, s, theta=1.0)
        np.testing.assert_array_equal(R, 0.0)

    def test_unity_just_after_source(self):
        """R(t, s) → 1 as t → s+ (no lag attenuation)."""
        s = 0.4
        t_close = np.array([s + 1e-8])
        R = ou_response_kernel(t_close, s, theta=2.0)
        np.testing.assert_allclose(R, [1.0], atol=1e-5)

    def test_exponential_decay(self):
        """R(t, s) = exp(-theta*(t-s)) for t > s."""
        s = 0.0
        theta = 2.0
        t_arr = np.array([0.5, 1.0, 2.0])
        R = ou_response_kernel(t_arr, s, theta)
        expected = np.exp(-theta * t_arr)
        np.testing.assert_allclose(R, expected, rtol=1e-12)

    def test_larger_theta_faster_decay(self):
        """Higher theta → faster exponential decay."""
        t_arr = np.linspace(0.01, 1.0, 50)
        R1 = ou_response_kernel(t_arr, 0.0, theta=1.0)
        R2 = ou_response_kernel(t_arr, 0.0, theta=5.0)
        assert R2[-1] < R1[-1]

    def test_shape_preserved(self):
        t_arr = np.ones((3, 4))
        R = ou_response_kernel(t_arr, 0.0, theta=1.0)
        assert R.shape == (3, 4)


# ---------------------------------------------------------------------------
# Rough response kernel
# ---------------------------------------------------------------------------

class TestRoughResponseKernel:
    def test_causality_zero_before_source(self):
        """R(t, s) = 0 for t ≤ s."""
        t_arr = np.array([0.0, 0.2, 0.5])
        s = 0.6
        R = rough_response_kernel(t_arr, s, eta=1.0, H=0.1)
        np.testing.assert_array_equal(R, 0.0)

    def test_positive_after_source(self):
        """R(t, s) > 0 for t > s."""
        t_arr = np.array([0.7, 0.8, 1.0])
        s = 0.6
        R = rough_response_kernel(t_arr, s, eta=1.0, H=0.1)
        assert np.all(R > 0.0)

    def test_singular_integrable_at_H_below_half(self):
        """For H < 0.5, kernel ~ (t-s)^{H-0.5} is integrable near t=s+."""
        s = 0.0
        t_arr = np.linspace(1e-6, 1.0, 5000)
        R = rough_response_kernel(t_arr, s, eta=1.0, H=0.1)
        integral = np.trapz(R, t_arr)
        # integral of (t-s)^(H-0.5) from 0 to 1 = 1/(H+0.5) (for eta*sqrt(2H) factor)
        assert np.isfinite(integral) and integral > 0.0

    def test_scaling_with_eta(self):
        """Kernel scales linearly with eta."""
        t_arr = np.array([0.5, 1.0])
        s = 0.1
        R1 = rough_response_kernel(t_arr, s, eta=1.0, H=0.2)
        R2 = rough_response_kernel(t_arr, s, eta=2.0, H=0.2)
        np.testing.assert_allclose(R2, 2.0 * R1, rtol=1e-12)

    def test_H_scaling(self):
        """Higher H → slower power-law decay (kernel larger at large t-s)."""
        t_arr = np.array([2.0])
        s = 0.0
        R_low_H  = rough_response_kernel(t_arr, s, eta=1.0, H=0.1)
        R_high_H = rough_response_kernel(t_arr, s, eta=1.0, H=0.4)
        # (t-s)^(H-0.5): H=0.4 → exponent -0.1 (larger than H=0.1 → exponent -0.4)
        assert R_high_H[0] > R_low_H[0]

    def test_formula_values(self):
        """Check explicit formula: R = eta*sqrt(2H)*(t-s)^{H-0.5}."""
        eta, H, s = 1.5, 0.2, 0.3
        t_arr = np.array([0.8, 1.5])
        R = rough_response_kernel(t_arr, s, eta, H)
        expected = eta * np.sqrt(2 * H) * (t_arr - s) ** (H - 0.5)
        np.testing.assert_allclose(R, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# Bucketed vega — bump-and-revalue
# ---------------------------------------------------------------------------

class TestBumpAndRevalue:
    def _make_price_func(self):
        """Simple quadratic price function for unit testing."""
        # P(vol_multipliers) = sum_k (vol_k^2) / n_buckets
        # dP/dvol_k = 2 * vol_k_base / n_buckets = 2/n_buckets at base
        def price_func(mults):
            return float(np.mean(mults ** 2))
        return price_func

    def test_output_shape(self):
        pf = self._make_price_func()
        vegas, elapsed = compute_bucketed_vega_bump(pf, n_buckets=8, T=1.0, eps=1e-4)
        assert vegas.shape == (8,)
        assert elapsed >= 0.0

    def test_sign_positive_for_increasing_price(self):
        """When price increases with each bucket vol, all vegas positive."""
        pf = self._make_price_func()
        vegas, _ = compute_bucketed_vega_bump(pf, n_buckets=5, T=1.0, eps=1e-4)
        assert np.all(vegas > 0.0)

    def test_uniform_price_equal_vegas(self):
        """Symmetric price function → all bucketed vegas equal."""
        pf = self._make_price_func()
        vegas, _ = compute_bucketed_vega_bump(pf, n_buckets=6, T=1.0, eps=1e-5)
        np.testing.assert_allclose(vegas, vegas[0], rtol=1e-4)

    def test_finite_difference_accuracy(self):
        """BAV with eps=1e-4 vs analytic derivative of known function."""
        # P = sum(m_k) / n; dP/dm_k = 1/n for all k
        n = 4
        pf = lambda m: float(np.sum(m) / n)
        vegas, _ = compute_bucketed_vega_bump(pf, n_buckets=n, T=1.0, eps=1e-4)
        np.testing.assert_allclose(vegas, np.full(n, 1.0 / n), rtol=1e-4)


# ---------------------------------------------------------------------------
# Bucketed vega — response propagator
# ---------------------------------------------------------------------------

class TestResponsePropagatorVega:
    T = 1.0
    N_GRID = 500
    t_grid = np.linspace(0.0, T, N_GRID)
    N_BUCKETS = 8
    bucket_times = np.linspace(T / N_BUCKETS / 2, T - T / N_BUCKETS / 2, N_BUCKETS)
    theta = 2.0

    def test_output_shape(self):
        def sens(t): return np.ones_like(t)
        def kern(t, s): return ou_response_kernel(t, s, self.theta)
        vegas, elapsed = compute_bucketed_vega_response(
            sens, kern, self.t_grid, self.bucket_times
        )
        assert vegas.shape == (self.N_BUCKETS,)
        assert elapsed >= 0.0

    def test_all_finite(self):
        def sens(t): return np.ones_like(t)
        def kern(t, s): return ou_response_kernel(t, s, self.theta)
        vegas, _ = compute_bucketed_vega_response(
            sens, kern, self.t_grid, self.bucket_times
        )
        assert np.all(np.isfinite(vegas))

    def test_positive_for_positive_kernel_and_sensitivity(self):
        """Positive kernel + positive sensitivity → positive vegas."""
        def sens(t): return np.ones_like(t)
        def kern(t, s): return ou_response_kernel(t, s, self.theta)
        vegas, _ = compute_bucketed_vega_response(
            sens, kern, self.t_grid, self.bucket_times
        )
        assert np.all(vegas >= 0.0)

    def test_decreasing_with_ou_kernel(self):
        """
        For constant sensitivity and OU kernel, later source buckets have
        shorter residual horizon → smaller integral → smaller vega.
        """
        def sens(t): return np.ones_like(t)
        def kern(t, s): return ou_response_kernel(t, s, self.theta)
        bucket_times_uniform = np.linspace(0.05, 0.95, 10)
        vegas, _ = compute_bucketed_vega_response(
            sens, kern, self.t_grid, bucket_times_uniform
        )
        # Vegas should be non-increasing across time
        assert np.all(np.diff(vegas) <= 0.01)  # allow small numerical noise

    def test_rough_kernel_positive(self):
        """Rough kernel with H=0.1 and valid t_grid → all vegas positive."""
        t_grid_pos = np.linspace(1e-5, self.T, self.N_GRID)
        def sens(t): return np.ones_like(t)
        def kern(t, s): return rough_response_kernel(t, s, eta=1.5, H=0.1)
        bucket_times_inner = np.linspace(0.1, 0.9, 6)
        vegas, _ = compute_bucketed_vega_response(
            sens, kern, t_grid_pos, bucket_times_inner
        )
        assert np.all(np.isfinite(vegas))
        assert np.all(vegas >= 0.0)
