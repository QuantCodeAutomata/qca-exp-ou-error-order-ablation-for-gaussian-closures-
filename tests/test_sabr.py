"""
Tests for sabr_stack.py — Hagan SABR formula, routed stack, and MC benchmark.
"""

import sys
sys.path.insert(0, '/workspace/project')

import numpy as np
import pytest
from src.sabr_stack import hagan_impvol, sabr_stack_call, sabr_stack_impvol, sabr_mc


# ---------------------------------------------------------------------------
# Hagan SABR implied-vol formula
# ---------------------------------------------------------------------------

class TestHaganImpvol:
    """Tests for the Hagan (2002) SABR implied-vol formula."""

    def test_positive_iv(self):
        """All IVs are positive for a valid SABR configuration."""
        K = np.linspace(70.0, 130.0, 15)
        iv = hagan_impvol(K, F=100.0, T=1.0, alpha=0.04, beta=0.5, rho=-0.3, nu=0.4)
        assert np.all(iv > 0.0), f"Non-positive Hagan IV: {iv}"

    def test_atm_equals_alpha_for_beta1(self):
        """For beta=1, ATM Hagan IV → alpha (up to small correction from T*nu^2 term)."""
        alpha, beta, rho, nu, T = 0.20, 1.0, 0.0, 0.01, 1.0
        iv_atm = hagan_impvol(np.array([100.0]), F=100.0, T=T, alpha=alpha,
                               beta=beta, rho=rho, nu=nu)
        # For nu->0, Hagan -> alpha; small nu gives small correction
        assert abs(iv_atm[0] - alpha) < 0.005, (
            f"ATM IV={iv_atm[0]:.5f} should be ≈ alpha={alpha:.5f}")

    def test_atm_limit_at_zero_nu(self):
        """For nu≈0, SABR reduces to CEV: ATM IV ≈ alpha / F^(1-beta)."""
        beta, alpha, F, T = 0.5, 0.2 * 100.0 ** 0.5, 100.0, 1.0
        iv_atm = hagan_impvol(np.array([F]), F=F, T=T, alpha=alpha, beta=beta,
                               rho=0.0, nu=1e-8)
        expected = alpha / F ** (1.0 - beta)
        assert abs(iv_atm[0] - expected) < 0.002, (
            f"ATM IV={iv_atm[0]:.5f} ≠ {expected:.5f}")

    def test_skew_from_rho(self):
        """Negative rho → negative skew: IV(K<F) > IV(K>F)."""
        K = np.array([90.0, 100.0, 110.0])
        iv = hagan_impvol(K, F=100.0, T=1.0, alpha=0.04, beta=0.5, rho=-0.4, nu=0.4)
        assert iv[0] > iv[2], (
            f"Expected left IV > right IV for rho<0, got {iv[0]:.4f} < {iv[2]:.4f}")

    def test_positive_rho_right_skew(self):
        """Positive rho → positive skew: IV(K>F) > IV(K<F)."""
        K = np.array([90.0, 110.0])
        iv = hagan_impvol(K, F=100.0, T=1.0, alpha=0.04, beta=0.5, rho=0.4, nu=0.4)
        assert iv[1] > iv[0], (
            f"Expected right IV > left IV for rho>0, got {iv[1]:.4f} < {iv[0]:.4f}")

    def test_atm_safe_for_very_close_K(self):
        """ATM formula is numerically safe when K is very close to F."""
        K = np.array([100.0, 100.0 + 1e-9, 100.0 - 1e-9])
        iv = hagan_impvol(K, F=100.0, T=1.0, alpha=0.04, beta=0.5, rho=0.0, nu=0.4)
        assert np.all(np.isfinite(iv)), f"Non-finite Hagan IV near ATM: {iv}"
        # All three should be very close to each other
        assert np.max(np.abs(iv - iv[0])) < 1e-4, f"ATM IV unstable: {iv}"

    def test_array_vs_scalar(self):
        """Array and scalar calls give consistent results."""
        K_arr = np.array([90.0, 100.0, 110.0])
        iv_arr = hagan_impvol(K_arr, F=100.0, T=1.0, alpha=0.04, beta=0.5, rho=0.0, nu=0.4)
        for i, K in enumerate(K_arr):
            iv_s = hagan_impvol(K, F=100.0, T=1.0, alpha=0.04, beta=0.5, rho=0.0, nu=0.4)
            assert abs(iv_arr[i] - iv_s) < 1e-12, (
                f"Array/scalar mismatch at K={K}: {iv_arr[i]:.8f} vs {iv_s:.8f}")


# ---------------------------------------------------------------------------
# SABR routed stack
# ---------------------------------------------------------------------------

class TestSabrStack:
    """Tests for the GH-quadrature-based deterministic SABR stack."""

    def test_positive_prices(self):
        """All stack call prices should be positive."""
        K = np.linspace(70.0, 130.0, 10)
        prices = sabr_stack_call(K, F0=100.0, T=2.0, alpha0=0.04,
                                  beta=0.5, rho=0.0, nu=0.4)
        assert np.all(prices >= 0.0), f"Negative stack prices: {prices}"

    def test_monotone_in_strike(self):
        """Call prices are decreasing in strike."""
        K = np.linspace(70.0, 130.0, 13)
        prices = sabr_stack_call(K, F0=100.0, T=2.0, alpha0=0.04,
                                  beta=0.5, rho=-0.3, nu=0.4)
        assert np.all(np.diff(prices) <= 1e-10), (
            f"Stack call not monotone in K: {prices}")

    def test_iv_in_range(self):
        """Stack implied vols are in a physically reasonable range [5%, 100%]."""
        beta = 0.5
        alpha0 = 0.2 * 100.0 ** (1.0 - beta)   # CEV backbone ≈ 20% ATM vol
        K = np.linspace(80.0, 120.0, 11)
        iv = sabr_stack_impvol(K, F0=100.0, T=1.0, alpha0=alpha0,
                                beta=beta, rho=0.0, nu=0.4)
        valid = iv[np.isfinite(iv)]
        assert len(valid) > 0, "All stack IVs are NaN"
        assert np.all(valid > 0.05), f"Stack IV below 5%: {valid}"
        assert np.all(valid < 1.0), f"Stack IV above 100%: {valid}"

    def test_atm_close_to_hagan_small_nu(self):
        """For small nu, stack ATM ≈ Hagan ATM."""
        beta, nu = 0.5, 0.05
        alpha0 = 0.2 * 100.0 ** (1.0 - beta)
        K_atm = np.array([100.0])
        iv_stack = sabr_stack_impvol(K_atm, F0=100.0, T=1.0,
                                      alpha0=alpha0, beta=beta, rho=0.0, nu=nu)
        iv_hagan = hagan_impvol(K_atm, F=100.0, T=1.0,
                                 alpha=alpha0, beta=beta, rho=0.0, nu=nu)
        assert np.isfinite(iv_stack[0]), "Stack ATM IV is NaN"
        assert abs(iv_stack[0] - iv_hagan[0]) < 0.005, (
            f"Stack ATM {iv_stack[0]:.5f} vs Hagan {iv_hagan[0]:.5f} differ by "
            f"{abs(iv_stack[0]-iv_hagan[0])*1e4:.0f}bp (expected < 50bp for small nu)")

    def test_rho_skew_direction(self):
        """Stack with rho<0 produces left-skewed smile."""
        K = np.array([80.0, 100.0, 120.0])
        iv = sabr_stack_impvol(K, F0=100.0, T=2.0, alpha0=0.04,
                                beta=0.5, rho=-0.5, nu=0.4)
        assert iv[0] > iv[2], (
            f"Expected left skew for rho<0: IV(80)={iv[0]:.4f}, IV(120)={iv[2]:.4f}")

    def test_scalar_input(self):
        """Scalar strike input works without error."""
        price = sabr_stack_call(100.0, F0=100.0, T=1.0, alpha0=0.04,
                                 beta=0.5, rho=0.0, nu=0.4)
        p = float(np.ravel(price)[0])
        assert np.isfinite(p) and p >= 0.0


# ---------------------------------------------------------------------------
# Monte Carlo benchmark
# ---------------------------------------------------------------------------

class TestSabrMC:
    """Tests for the Euler-Maruyama SABR Monte Carlo."""

    def test_prices_positive(self):
        """MC call prices are non-negative."""
        K = np.array([90.0, 100.0, 110.0])
        prices, _ = sabr_mc(K, F0=100.0, T=1.0, alpha0=0.04,
                             beta=0.5, rho=0.0, nu=0.4,
                             n_paths=2048, n_steps_per_year=64, seed=0)
        assert np.all(prices >= 0.0), f"Negative MC prices: {prices}"

    def test_monotone_in_strike(self):
        """MC prices are (weakly) decreasing in strike."""
        K = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        prices, _ = sabr_mc(K, F0=100.0, T=1.0, alpha0=0.04,
                             beta=0.5, rho=0.0, nu=0.4,
                             n_paths=4096, n_steps_per_year=128, seed=1)
        assert np.all(np.diff(prices) <= 1.0), (
            f"MC prices not monotone: {prices}")

    def test_atm_price_in_range(self):
        """ATM MC price is within 10% of BS formula with sigma=alpha."""
        beta = 0.5
        alpha0 = 0.2 * 100.0 ** (1.0 - beta)  # so BS-equiv sigma ≈ 0.2
        prices, se = sabr_mc(np.array([100.0]), F0=100.0, T=1.0,
                              alpha0=alpha0, beta=beta, rho=0.0, nu=0.1,
                              n_paths=8192, n_steps_per_year=256, seed=7)
        from src.absorbed_cev import bs_call
        bs_ref = bs_call(100.0, 100.0, 1.0, 0.2)
        assert abs(prices[0] - bs_ref) < 3.0 * float(se[0]) + 1.0, (
            f"MC ATM price {prices[0]:.4f} far from BS {bs_ref:.4f}")

    def test_deterministic_seed(self):
        """Same seed gives same result."""
        K = np.array([100.0])
        p1, _ = sabr_mc(K, F0=100.0, T=0.5, alpha0=0.04,
                         beta=0.5, rho=0.0, nu=0.4,
                         n_paths=1024, n_steps_per_year=64, seed=123)
        p2, _ = sabr_mc(K, F0=100.0, T=0.5, alpha0=0.04,
                         beta=0.5, rho=0.0, nu=0.4,
                         n_paths=1024, n_steps_per_year=64, seed=123)
        assert abs(p1[0] - p2[0]) < 1e-12, "MC with same seed gave different results"

    def test_scalar_return(self):
        """Scalar K returns scalar price and std_err."""
        price, se = sabr_mc(100.0, F0=100.0, T=0.5, alpha0=0.04,
                             beta=0.5, rho=0.0, nu=0.4,
                             n_paths=512, n_steps_per_year=32, seed=5)
        assert isinstance(price, float) and isinstance(se, float)
        assert price >= 0.0 and se >= 0.0
