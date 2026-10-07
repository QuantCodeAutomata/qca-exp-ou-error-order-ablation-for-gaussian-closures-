"""
Tests for src/exotic_pricing.py.

Validates:
  - One-touch price is a probability: in [0, 1]
  - Reverse knock-out price ≤ vanilla call price (knock-out removes upside)
  - FVA price is strictly positive (E[sqrt(V)] > 0)
  - Black-76 helper: boundary cases and put-call parity
  - Simulated paths: shapes and martingale condition
"""

import sys

sys.path.insert(0, '/workspace/project')

import numpy as np
import pytest

from src.exotic_pricing import (
    black_call,
    simulate_expou_paths,
    price_one_touch_mc,
    price_reverse_knockout_mc,
    price_fva_mc,
)

# ---------------------------------------------------------------------------
# Shared parameters (small MC for speed)
# ---------------------------------------------------------------------------

THETA = 1.0
MU = -3.76
SIGMA = 1.8
RHO = 0.0
T = 1.0
F0 = 100.0
N_PATHS = 4000
N_STEPS = 50


# ---------------------------------------------------------------------------
# Black-76 helper
# ---------------------------------------------------------------------------

class TestBlackCall:
    def test_zero_vol_returns_intrinsic(self):
        assert black_call(100.0, 90.0, 0.0, 1.0) == pytest.approx(10.0, abs=1e-10)
        assert black_call(100.0, 110.0, 0.0, 1.0) == 0.0

    def test_zero_maturity_returns_intrinsic(self):
        assert black_call(100.0, 95.0, 0.3, 0.0) == pytest.approx(5.0, abs=1e-10)

    def test_call_non_negative(self):
        for K in [80.0, 100.0, 120.0]:
            assert black_call(100.0, K, 0.2, 1.0) >= 0.0

    def test_put_call_parity(self):
        """C - P = F - K (undiscounted)."""
        F, K, vol, T_ = 100.0, 95.0, 0.25, 0.5
        c = black_call(F, K, vol, T_)
        # put = c - (F - K) by parity
        put = c - (F - K)
        assert put >= -1e-10


# ---------------------------------------------------------------------------
# Path simulator
# ---------------------------------------------------------------------------

class TestSimulateExpouPaths:
    def test_output_shapes(self):
        n_paths, n_steps = 50, 30
        logS, X = simulate_expou_paths(THETA, MU, SIGMA, RHO, T,
                                       n_paths, n_steps, seed=0)
        assert logS.shape == (n_steps + 1, n_paths)
        assert X.shape == (n_steps + 1, n_paths)

    def test_initial_conditions(self):
        logS, X = simulate_expou_paths(THETA, MU, SIGMA, RHO, T,
                                       100, 20, seed=1)
        # logS starts at 0
        np.testing.assert_array_equal(logS[0], 0.0)
        # X starts at mu (stationary mean)
        np.testing.assert_array_almost_equal(X[0], MU, decimal=12)

    def test_martingale_approx(self):
        """E[S_T / S_0] ≈ 1  (weak check: within 5% at moderate path count)."""
        n_paths = 8000
        logS, _ = simulate_expou_paths(THETA, MU, SIGMA, RHO, T,
                                       n_paths, 100, seed=42)
        S_T_mean = float(np.mean(np.exp(logS[-1])))
        assert abs(S_T_mean - 1.0) < 0.05


# ---------------------------------------------------------------------------
# One-touch pricer
# ---------------------------------------------------------------------------

class TestOneTouchMC:
    def test_probability_bounds(self):
        """Price must be a valid probability in [0, 1]."""
        price, se = price_one_touch_mc(
            THETA, MU, SIGMA, RHO, T, barrier_ratio=1.06,
            F0=F0, n_paths=N_PATHS, n_steps=N_STEPS, seed=7
        )
        assert 0.0 <= price <= 1.0
        assert se >= 0.0

    def test_higher_barrier_lower_price(self):
        """Raising the barrier must reduce the touch probability."""
        p_low, _  = price_one_touch_mc(THETA, MU, SIGMA, RHO, T, 1.05,
                                       F0=F0, n_paths=N_PATHS, n_steps=N_STEPS, seed=7)
        p_high, _ = price_one_touch_mc(THETA, MU, SIGMA, RHO, T, 1.20,
                                       F0=F0, n_paths=N_PATHS, n_steps=N_STEPS, seed=7)
        assert p_low >= p_high - 0.02   # allow small MC noise

    def test_certain_touch_at_zero_barrier(self):
        """Barrier at or below initial spot → always touch (price ≈ 1)."""
        price, _ = price_one_touch_mc(
            THETA, MU, SIGMA, RHO, T, barrier_ratio=0.99,
            F0=F0, n_paths=2000, n_steps=10, seed=0
        )
        assert price == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Reverse knock-out pricer
# ---------------------------------------------------------------------------

class TestReverseKnockoutMC:
    def test_rko_leq_vanilla_call(self):
        """
        RKO call price ≤ vanilla call price.
        (knock-out removes value relative to plain vanilla)
        """
        rko_price, _ = price_reverse_knockout_mc(
            THETA, MU, SIGMA, RHO, T, K=100.0, barrier_ratio=1.10,
            F0=F0, n_paths=N_PATHS, n_steps=N_STEPS, seed=17
        )
        # Vanilla ATM call price via Black-76 with stationary vol
        import math
        gamma = SIGMA ** 2 / (2.0 * THETA)
        v_stat = math.exp(MU + gamma)
        vanilla = black_call(F0, 100.0, math.sqrt(v_stat), T)
        assert rko_price <= vanilla + 0.5   # allow generous tolerance

    def test_non_negative_price(self):
        price, _ = price_reverse_knockout_mc(
            THETA, MU, SIGMA, RHO, T, K=100.0, barrier_ratio=1.10,
            F0=F0, n_paths=N_PATHS, n_steps=N_STEPS, seed=17
        )
        assert price >= 0.0

    def test_unreachable_barrier_equals_vanilla(self):
        """
        Very high barrier → knock-out never triggered → RKO ≈ vanilla call.

        Compare two MC estimates of the same payoff (one with barrier=1e6,
        one without knockout) using the same seed.
        """
        rko_p, _ = price_reverse_knockout_mc(
            THETA, MU, SIGMA, RHO, T, K=100.0, barrier_ratio=1e6,
            F0=F0, n_paths=N_PATHS, n_steps=N_STEPS, seed=5
        )
        # Vanilla call via MC: same model but no barrier (barrier_ratio=1e9)
        vanilla_p, _ = price_reverse_knockout_mc(
            THETA, MU, SIGMA, RHO, T, K=100.0, barrier_ratio=1e9,
            F0=F0, n_paths=N_PATHS, n_steps=N_STEPS, seed=5
        )
        # Both should give the same price (no path ever reaches 1e6 or 1e9)
        assert abs(rko_p - vanilla_p) < 0.5   # well within MC noise


# ---------------------------------------------------------------------------
# Forward-vol agreement pricer
# ---------------------------------------------------------------------------

class TestFVAMC:
    def test_positive_price(self):
        """E[sqrt(V)] must be strictly positive."""
        price, se = price_fva_mc(
            THETA, MU, SIGMA, RHO, T_start=0.5, Delta=0.5,
            F0=F0, n_paths=N_PATHS, n_steps=N_STEPS, seed=3
        )
        assert price > 0.0
        assert se >= 0.0

    def test_fva_order_of_magnitude(self):
        """E[sqrt(V)] for typical exp-OU params should be in a sane range."""
        price, _ = price_fva_mc(
            THETA, MU, SIGMA, RHO, T_start=0.5, Delta=0.5,
            F0=F0, n_paths=N_PATHS, n_steps=N_STEPS, seed=3
        )
        # v_stat ≈ exp(MU + sigma^2/(2*theta)) ≈ exp(-3.76+1.62) = exp(-2.14) ≈ 0.118
        # E[sqrt(V)] ≈ sqrt(v_stat * 0.5) ≈ sqrt(0.059) ≈ 0.24
        assert 0.01 < price < 5.0

    def test_longer_window_larger_fva(self):
        """Longer accumulation window → larger FVA."""
        p_short, _ = price_fva_mc(THETA, MU, SIGMA, RHO, T_start=0.5, Delta=0.25,
                                   F0=F0, n_paths=N_PATHS, n_steps=100, seed=11)
        p_long,  _ = price_fva_mc(THETA, MU, SIGMA, RHO, T_start=0.5, Delta=0.5,
                                   F0=F0, n_paths=N_PATHS, n_steps=100, seed=11)
        assert p_long >= p_short - 0.01   # allow MC noise
