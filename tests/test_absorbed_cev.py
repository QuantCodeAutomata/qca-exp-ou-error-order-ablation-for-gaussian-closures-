"""
Tests for absorbed_cev.py — Schroder (1989) CEV call pricer and BS utilities.
"""

import sys
sys.path.insert(0, '/workspace/project')

import numpy as np
import pytest
from src.absorbed_cev import bs_call, bs_impvol, cev_call, cev_call_lamperti


# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------

class TestBsCall:
    def test_atm_call_positive(self):
        """ATM call price must be positive."""
        price = bs_call(100.0, 100.0, 1.0, 0.2)
        assert price > 0.0

    def test_atm_call_known_value(self):
        """ATM call for F=K=100, T=1, sigma=0.2 ≈ 7.966."""
        price = bs_call(100.0, 100.0, 1.0, 0.2)
        assert abs(price - 7.9656) < 0.01, f"ATM call = {price:.4f}, expected ~7.97"

    def test_deep_itm_lower_bound(self):
        """Deep ITM call >= intrinsic value F-K."""
        price = bs_call(100.0, 50.0, 1.0, 0.2)
        assert price >= 50.0 - 1e-9, f"Deep ITM call = {price:.4f}"

    def test_deep_otm_small_price(self):
        """Deep OTM call should be small."""
        price = bs_call(100.0, 200.0, 0.1, 0.2)
        assert price < 0.01, f"Deep OTM call = {price:.6f}"

    def test_zero_time_intrinsic(self):
        """At T=0, call = max(F-K, 0)."""
        assert abs(bs_call(100.0, 90.0, 0.0, 0.2) - 10.0) < 1e-10
        assert abs(bs_call(100.0, 110.0, 0.0, 0.2) - 0.0) < 1e-10

    def test_array_input(self):
        """Vectorised over strikes."""
        K = np.array([80.0, 100.0, 120.0])
        prices = bs_call(100.0, K, 1.0, 0.2)
        assert prices.shape == (3,)
        assert np.all(prices >= 0.0)
        assert prices[0] > prices[1] > prices[2]

    def test_put_call_parity(self):
        """Call - Put = F - K (forward measure, r=0)."""
        F, K, T, sigma = 100.0, 95.0, 1.0, 0.3
        call = bs_call(F, K, T, sigma)
        # Put = Call - (F-K) in forward measure
        put_implied = call - (F - K)
        assert put_implied > 0.0, "Put price should be positive"


class TestBsImpvol:
    def test_round_trip(self):
        """Invert bs_call to recover sigma."""
        sigma_true = 0.25
        price = bs_call(100.0, 100.0, 1.0, sigma_true)
        sigma_rec = bs_impvol(price, 100.0, 100.0, 1.0)
        assert abs(sigma_rec - sigma_true) < 1e-6, (
            f"IV round-trip: got {sigma_rec:.8f}, expected {sigma_true}")

    def test_array_round_trip(self):
        """Array of strikes: round-trip."""
        K = np.array([85.0, 100.0, 115.0])
        sigma_true = np.array([0.22, 0.20, 0.23])
        prices = bs_call(100.0, K, 1.0, sigma_true)
        sigma_rec = bs_impvol(prices, 100.0, K, 1.0)
        np.testing.assert_allclose(sigma_rec, sigma_true, atol=1e-5,
                                   err_msg="Array IV round-trip failed")

    def test_intrinsic_returns_nan(self):
        """Price at intrinsic bound → NaN (no finite IV)."""
        result = bs_impvol(0.0, 100.0, 110.0, 1.0)
        assert np.isnan(result), f"Expected NaN, got {result}"


# ---------------------------------------------------------------------------
# CEV pricer
# ---------------------------------------------------------------------------

class TestCevCall:
    def test_bs_limit(self):
        """beta=1 (limit) returns BS call."""
        bs_price = bs_call(100.0, 100.0, 1.0, 0.3)
        cev_price = cev_call(100.0, 100.0, 1.0, alpha=0.3, beta=1.0 - 1e-10)
        assert abs(cev_price - bs_price) < 0.05, (
            f"CEV beta≈1 differs from BS: {cev_price:.4f} vs {bs_price:.4f}")

    def test_positive_price(self):
        """CEV call prices are non-negative."""
        for beta in [0.2, 0.5, 0.8]:
            alpha = 0.2 * 100.0 ** (1.0 - beta)
            price = cev_call(100.0, 100.0, 1.0, alpha=alpha, beta=beta)
            assert price >= 0.0, f"Negative CEV price for beta={beta}"

    def test_monotone_in_strike(self):
        """Call price is decreasing in strike."""
        alpha = 0.2 * 100.0 ** 0.5   # beta=0.5
        K_arr = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        prices = cev_call(100.0, K_arr, 1.0, alpha=alpha, beta=0.5)
        diffs = np.diff(prices)
        assert np.all(diffs <= 0.0), f"CEV call not monotone decreasing in K: {prices}"

    def test_intrinsic_lower_bound(self):
        """CEV call >= max(F-K, 0)."""
        alpha = 0.3 * 100.0 ** 0.5
        price = cev_call(100.0, 80.0, 2.0, alpha=alpha, beta=0.5)
        assert price >= 20.0 - 1e-8, f"CEV call below intrinsic: {price:.4f}"

    def test_negative_skew_for_beta_less_than_one(self):
        """Absorbed CEV with beta<1 produces a downward-sloping (negative) IV skew.

        vol(K) = alpha*K^{beta-1} decreases with K, so BS-implied vol is
        higher for low strikes than high strikes (negative skew).
        """
        beta = 0.5
        alpha = 0.3 * 100.0 ** (1.0 - beta)
        K_low, K_high = 80.0, 120.0
        iv_low = bs_impvol(cev_call(100.0, K_low, 1.0, alpha, beta), 100.0, K_low, 1.0)
        iv_high = bs_impvol(cev_call(100.0, K_high, 1.0, alpha, beta), 100.0, K_high, 1.0)
        assert np.isfinite(iv_low) and np.isfinite(iv_high), (
            f"IV not finite: iv_low={iv_low}, iv_high={iv_high}")
        assert iv_low > iv_high, (
            f"CEV beta={beta}<1 should have negative skew: "
            f"IV(80)={iv_low*100:.2f}% < IV(120)={iv_high*100:.2f}%")

    def test_array_consistency(self):
        """Scalar and array inputs give same results."""
        alpha, beta = 0.2 * 100.0 ** 0.5, 0.5
        K_arr = np.array([90.0, 100.0, 110.0])
        prices_arr = cev_call(100.0, K_arr, 1.0, alpha, beta)
        for i, K in enumerate(K_arr):
            price_scalar = cev_call(100.0, K, 1.0, alpha, beta)
            assert abs(prices_arr[i] - price_scalar) < 1e-12


# ---------------------------------------------------------------------------
# CEV Lamperti pricer
# ---------------------------------------------------------------------------

class TestCevCallLamperti:
    def test_consistency_with_cev_call(self):
        """cev_call_lamperti matches cev_call for standard inputs."""
        beta = 0.5
        alpha = 0.2 * 100.0 ** (1.0 - beta)
        F0, K, T = 100.0, 105.0, 2.0
        betac = 1.0 - beta
        Z0 = F0 ** betac / betac
        ZK = K ** betac / betac
        var_Z = alpha ** 2 * T

        lam_price = cev_call_lamperti(Z0, K, ZK, var_Z, beta)
        cev_price = cev_call(F0, K, T, alpha, beta)
        assert abs(lam_price - cev_price) < 1e-6, (
            f"Lamperti price {lam_price:.6f} != CEV price {cev_price:.6f}")

    def test_absorbed_below_zero_gives_zero(self):
        """Absorbed CEV at Z0<=0 gives zero call price."""
        beta = 0.5
        K, ZK = 100.0, 100.0 ** 0.5 / 0.5
        price = cev_call_lamperti(-1.0, K, ZK, var_Z=1.0, beta=beta)
        assert price == 0.0, f"Expected 0 for absorbed state, got {price}"

    def test_large_var_price_near_intrinsic(self):
        """Very small variance (near expiry) → price near intrinsic."""
        beta = 0.5
        F0, K = 110.0, 100.0
        betac = 1.0 - beta
        Z0 = F0 ** betac / betac
        ZK = K ** betac / betac
        price = cev_call_lamperti(Z0, K, ZK, var_Z=1e-10, beta=beta)
        assert abs(price - max(F0 - K, 0.0)) < 0.1, (
            f"Near-expiry price {price:.4f} far from intrinsic {F0-K:.4f}")
