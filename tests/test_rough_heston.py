"""
Tests for rough_heston.py — fractional Riccati solver and COS pricer.
"""

import sys
sys.path.insert(0, '/workspace/project')

import numpy as np
import pytest
from src.rough_heston import fractional_riccati, rough_heston_log_cf, rough_heston_call_cos, bs_impvol
from src.absorbed_cev import bs_call


# ---------------------------------------------------------------------------
# Standard parameters
# ---------------------------------------------------------------------------

PARAMS = dict(kappa=1.0, theta=0.04, lam=0.5, H=0.1, v0=0.04, rho_hv=0.0)
F0 = 100.0
K_ATM = np.array([100.0])
K_SMILE = F0 * np.linspace(0.8, 1.2, 11)


# ---------------------------------------------------------------------------
# Fractional Riccati solver
# ---------------------------------------------------------------------------

class TestFractionalRiccati:
    def test_initial_condition(self):
        """h(:, 0) = 0 always."""
        u = np.array([1.0, 5.0, 10.0])
        h = fractional_riccati(u, T=1.0, N=50, **{k: v for k, v in PARAMS.items()
                                                   if k in ['kappa', 'lam', 'H', 'rho_hv']})
        np.testing.assert_array_equal(h[:, 0], 0.0 + 0.0j)

    def test_shape(self):
        """Output shape is (len(u), N+1)."""
        u = np.arange(1.0, 6.0)
        N = 30
        h = fractional_riccati(u, T=1.0, kappa=1.0, lam=0.5, H=0.1, N=N)
        assert h.shape == (5, N + 1), f"Shape mismatch: {h.shape}"

    def test_finite_values(self):
        """All h values are finite for moderate u."""
        u = np.linspace(0.1, 15.0, 20)
        h = fractional_riccati(u, T=1.0, kappa=1.0, lam=0.5, H=0.1, N=100)
        assert np.all(np.isfinite(h)), "Non-finite values in fractional Riccati solution"

    def test_negative_real_part(self):
        """Re(h) ≤ 0 for all time steps (ensures |CF| ≤ 1)."""
        u = np.linspace(0.1, 20.0, 30)
        h = fractional_riccati(u, T=1.0, kappa=1.0, lam=0.5, H=0.1, N=100)
        # Skip h[:, 0] = 0; check h[:, 1:]
        assert np.all(np.real(h[:, 1:]) <= 1e-10), (
            f"Positive Re(h) values found: max={np.max(np.real(h[:, 1:])):.4f}")

    def test_lam0_recovers_linear(self):
        """For lam=0, h solves a linear fractional equation (no quadratic term)."""
        u = np.array([1.0])
        h = fractional_riccati(u, T=1.0, kappa=1.0, lam=0.0, H=0.499, N=500)
        # h[:, N] should be approximately g*(1-e^{-T}) for alpha→1
        g = -0.5 * (1.0 ** 2 + 1j * 1.0)
        h_ode = g * (1.0 - np.exp(-1.0))
        assert abs(h[0, -1] - h_ode) < 0.02, (
            f"lam=0 Riccati: h(T)={h[0,-1]:.4f}, ODE={h_ode:.4f}")


# ---------------------------------------------------------------------------
# Log characteristic function
# ---------------------------------------------------------------------------

class TestRoughHestonLogCF:
    def test_cf_at_zero_is_one(self):
        """phi(0) = 1: log_cf(0) = 0."""
        log_cf = rough_heston_log_cf(np.array([0.0]), T=1.0, **PARAMS, N=50)
        assert abs(log_cf[0]) < 1e-10, f"log_cf(0) = {log_cf[0]}"

    def test_nonpositive_real_part(self):
        """Re(log_cf) ≤ 0 for all u (characteristic function bound |phi|≤1)."""
        u = np.linspace(0.1, 20.0, 20)
        log_cf = rough_heston_log_cf(u, T=1.0, **PARAMS, N=200)
        assert np.all(np.real(log_cf) <= 1e-8), (
            f"Positive Re(log_cf): max={np.max(np.real(log_cf)):.5f}")

    def test_martingale_lam0(self):
        """lam=0, v0=theta: CF(-i) = exp(v0*h(T,-i) + kappa*theta*int_h(-i)) ≈ 1."""
        # We verify via prices: rough Heston (lam=0) ATM call should match BS
        # This implicitly tests E[S_T] = S_0 (martingale)
        p = rough_heston_call_cos(K_ATM, F0, T=1.0,
                                  kappa=1.0, theta=0.04, lam=0.0, H=0.499, v0=0.04,
                                  rho_hv=0.0, N_riccati=500, N_cos=64)
        bs_ref = bs_call(F0, K_ATM[0], 1.0, 0.2)  # sigma = sqrt(theta) = 0.2
        assert abs(p[0] - bs_ref) < 0.5, (
            f"lam=0 ATM price {p[0]:.4f} vs BS {bs_ref:.4f} (should be <0.5 apart)")

    def test_bs_limit_flat_smile(self):
        """lam=0, v0=theta (H=0.499): all strikes give sigma≈sqrt(theta)=20%."""
        p = rough_heston_call_cos(K_SMILE, F0, T=1.0,
                                  kappa=1.0, theta=0.04, lam=0.0, H=0.499, v0=0.04,
                                  rho_hv=0.0, N_riccati=500, N_cos=64)
        iv = bs_impvol(p, F0, K_SMILE, 1.0)
        valid = iv[np.isfinite(iv)]
        assert len(valid) >= 7, f"Too few valid IVs: {iv}"
        np.testing.assert_allclose(valid, 0.20, atol=0.005,
                                   err_msg="lam=0 IV should be ≈20% everywhere")

    def test_finite_values_moderate_u(self):
        """log_cf is finite for moderate u values."""
        u = np.linspace(0.1, 15.0, 20)
        log_cf = rough_heston_log_cf(u, T=1.0, **PARAMS, N=100)
        assert np.all(np.isfinite(log_cf)), "Non-finite log_cf values"


# ---------------------------------------------------------------------------
# COS pricer
# ---------------------------------------------------------------------------

class TestRoughHestonCallCOS:
    def test_prices_nonnegative(self):
        """All prices are non-negative (enforced by max(price, 0))."""
        prices = rough_heston_call_cos(K_SMILE, F0, T=1.0, **PARAMS,
                                       N_riccati=200, N_cos=64)
        assert np.all(prices >= 0.0), f"Negative prices: {prices}"

    def test_prices_below_forward(self):
        """All prices are at most F0 (call can't exceed spot)."""
        prices = rough_heston_call_cos(K_SMILE, F0, T=1.0, **PARAMS,
                                       N_riccati=200, N_cos=64)
        assert np.all(prices <= F0 + 1e-8), f"Prices exceed F0={F0}: {prices}"

    def test_monotone_in_strike(self):
        """Call prices are (weakly) decreasing in strike."""
        prices = rough_heston_call_cos(K_SMILE, F0, T=1.0, **PARAMS,
                                       N_riccati=200, N_cos=64)
        assert np.all(np.diff(prices) <= 1e-6), (
            f"Prices not monotone decreasing: {prices}")

    def test_iv_in_range(self):
        """Implied vols are in [5%, 100%] for the standard parameters."""
        prices = rough_heston_call_cos(K_SMILE, F0, T=1.0, **PARAMS,
                                       N_riccati=200, N_cos=64)
        iv = bs_impvol(prices, F0, K_SMILE, 1.0)
        valid = iv[np.isfinite(iv)]
        assert len(valid) >= 5, f"Too few valid IVs: {iv}"
        assert np.all(valid > 0.05), f"IV below 5%: {valid}"
        assert np.all(valid < 1.0), f"IV above 100%: {valid}"

    def test_n200_closer_than_n50_to_reference(self):
        """N=200 solution is closer to N=500 reference than N=50."""
        ref = rough_heston_call_cos(K_SMILE, F0, T=1.0, **PARAMS,
                                     N_riccati=500, N_cos=64)
        p50 = rough_heston_call_cos(K_SMILE, F0, T=1.0, **PARAMS,
                                     N_riccati=50, N_cos=64)
        p200 = rough_heston_call_cos(K_SMILE, F0, T=1.0, **PARAMS,
                                      N_riccati=200, N_cos=64)

        err50 = np.nanmean(np.abs(p50 - ref))
        err200 = np.nanmean(np.abs(p200 - ref))
        assert err200 < err50, (
            f"N=200 error ({err200:.4f}) should be < N=50 error ({err50:.4f})")

    def test_scalar_strike_input(self):
        """Scalar strike input runs without error and returns a non-negative value."""
        price = rough_heston_call_cos(100.0, F0, T=1.0, **PARAMS,
                                       N_riccati=50, N_cos=32)
        assert float(np.ravel(price)[0]) >= 0.0

    def test_multiple_maturities_consistent(self):
        """Longer maturities give higher ATM call prices (positive time value)."""
        prices = [
            rough_heston_call_cos(np.array([F0]), F0, T=T, **PARAMS,
                                   N_riccati=200, N_cos=64)[0]
            for T in [0.5, 1.0, 2.0]
        ]
        assert prices[0] < prices[1] < prices[2], (
            f"ATM prices not increasing in T: {prices}")

    def test_symmetric_smile_zero_rho(self):
        """For rho_hv=0, IV smile is approximately symmetric around ATM.

        True symmetry holds at the CF level; the COS truncation introduces
        O(10 bp) asymmetry for extreme wings, hence the generous tolerance.
        """
        K_sym = F0 * np.array([0.90, 0.95, 1.00, 1.05, 1.10])
        p = rough_heston_call_cos(K_sym, F0, T=1.0, **PARAMS,
                                   N_riccati=200, N_cos=64)
        iv = bs_impvol(p, F0, K_sym, 1.0)
        # iv[0] ~ iv[4], iv[1] ~ iv[3]
        for i in range(2):
            j = 4 - i
            if np.isfinite(iv[i]) and np.isfinite(iv[j]):
                assert abs(iv[i] - iv[j]) < 0.008, (
                    f"Asymmetric smile at K/F={K_sym[i]/F0:.2f} vs {K_sym[j]/F0:.2f}: "
                    f"{iv[i]*100:.2f}% vs {iv[j]*100:.2f}%")
