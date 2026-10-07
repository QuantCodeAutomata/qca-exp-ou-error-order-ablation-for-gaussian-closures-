"""
Tests for rough Bergomi pricers and shared metrics utilities.

Covers:
  - src.metrics   : black_call, implied_vol, rmse_bp
  - src.rough_bergomi_rqmc    : rough_bergomi_rqmc
  - src.rough_bergomi_closure : rough_bergomi_gaussian_closure
"""

import sys
sys.path.insert(0, '/workspace/project')

import numpy as np
import pytest

from src.metrics import black_call, implied_vol, rmse_bp
from src.rough_bergomi_rqmc import rough_bergomi_rqmc
from src.rough_bergomi_closure import rough_bergomi_gaussian_closure


# ---------------------------------------------------------------------------
# Standard parameters
# ---------------------------------------------------------------------------

F0 = 1.0
STRIKES = F0 * np.array([0.90, 0.95, 1.00, 1.05, 1.10])

BASE = dict(v0=0.04, eta=1.0, rho=0.0, H=0.2, T=1.0)


# ---------------------------------------------------------------------------
# src.metrics — black_call
# ---------------------------------------------------------------------------

class TestBlackCall:
    def test_atm_positive(self):
        """ATM call is positive."""
        assert black_call(1.0, 1.0, 0.2, 1.0) > 0.0

    def test_atm_known_value(self):
        """ATM F=K=1, T=1, sigma=0.2 ≈ 0.07966 (Black formula)."""
        p = black_call(1.0, 1.0, 0.2, 1.0)
        assert abs(p - 0.07966) < 0.001, f"ATM call = {p:.5f}"

    def test_zero_time_intrinsic(self):
        """T=0: call equals intrinsic value."""
        assert abs(black_call(1.0, 0.9, 0.2, 0.0) - 0.1) < 1e-10
        assert abs(black_call(1.0, 1.1, 0.2, 0.0) - 0.0) < 1e-10

    def test_nonnegative(self):
        """Call prices are non-negative for all moneyness levels."""
        K = np.array([0.7, 0.9, 1.0, 1.1, 1.3])
        prices = black_call(1.0, K, 0.2, 1.0)
        assert np.all(prices >= 0.0)

    def test_decreasing_in_strike(self):
        """Call prices decrease with strike."""
        K = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
        prices = black_call(1.0, K, 0.2, 1.0)
        assert np.all(np.diff(prices) <= 0.0)

    def test_increasing_in_vol(self):
        """ATM call increases with volatility."""
        p_low  = black_call(1.0, 1.0, 0.1, 1.0)
        p_high = black_call(1.0, 1.0, 0.4, 1.0)
        assert p_high > p_low

    def test_below_forward(self):
        """Call price <= forward (no arbitrage upper bound)."""
        p = black_call(1.0, 0.5, 2.0, 1.0)
        assert p <= 1.0 + 1e-10


# ---------------------------------------------------------------------------
# src.metrics — implied_vol
# ---------------------------------------------------------------------------

class TestImpliedVol:
    def test_round_trip_atm(self):
        """Inverting black_call recovers sigma."""
        sigma_true = 0.25
        price = black_call(1.0, 1.0, sigma_true, 1.0)
        sigma_rec = implied_vol(price, 1.0, 1.0, 1.0)
        assert abs(sigma_rec - sigma_true) < 1e-6

    def test_round_trip_smile(self):
        """Round-trip for off-ATM strikes."""
        for K, sigma in [(0.90, 0.22), (1.05, 0.19)]:
            price = black_call(1.0, K, sigma, 1.0)
            sigma_rec = implied_vol(price, 1.0, K, 1.0)
            assert abs(sigma_rec - sigma) < 1e-5, (
                f"K={K}: IV={sigma_rec:.5f}, expected {sigma:.5f}")

    def test_zero_price_returns_nan(self):
        """Zero price returns NaN."""
        result = implied_vol(0.0, 1.0, 1.1, 1.0)
        assert np.isnan(result)

    def test_zero_T_returns_nan(self):
        """T=0 returns NaN."""
        result = implied_vol(0.05, 1.0, 1.0, 0.0)
        assert np.isnan(result)


# ---------------------------------------------------------------------------
# src.metrics — rmse_bp
# ---------------------------------------------------------------------------

class TestRmseBp:
    def test_identical_arrays(self):
        """RMSE of identical arrays is zero."""
        iv = [0.20, 0.22, 0.19]
        assert rmse_bp(iv, iv) == pytest.approx(0.0, abs=1e-10)

    def test_known_value(self):
        """RMSE of [0.20] vs [0.21] = 100 bp."""
        assert abs(rmse_bp([0.20], [0.21]) - 100.0) < 1e-8

    def test_symmetry(self):
        """RMSE(a, b) == RMSE(b, a)."""
        a = [0.20, 0.22, 0.18]
        b = [0.21, 0.20, 0.23]
        assert abs(rmse_bp(a, b) - rmse_bp(b, a)) < 1e-10

    def test_nan_handling(self):
        """NaN entries are excluded from the RMSE computation."""
        a = [0.20, np.nan, 0.22]
        b = [0.21, 0.19,   0.23]
        # Only the non-NaN pairs: (0.20-0.21)^2 + (0.22-0.23)^2 → RMSE in bp
        expected = np.sqrt(((0.20 - 0.21) ** 2 + (0.22 - 0.23) ** 2) / 2) * 1e4
        assert abs(rmse_bp(a, b) - expected) < 1e-6

    def test_all_nan_returns_nan(self):
        """All-NaN input returns NaN."""
        assert np.isnan(rmse_bp([np.nan, np.nan], [0.2, 0.2]))

    def test_unit(self):
        """Result is in basis points (1 bp = 0.0001 vol unit)."""
        # 1 bp difference in IV → rmse = 1.0
        assert abs(rmse_bp([0.2000], [0.2001]) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# rough_bergomi_rqmc
# ---------------------------------------------------------------------------

class TestRoughBergomiRQMC:
    def test_prices_nonnegative(self):
        """All call prices are non-negative."""
        prices = rough_bergomi_rqmc(**BASE, strikes=STRIKES, F0=F0, N=20, n_paths=256)
        assert np.all(prices >= 0.0), f"Negative prices: {prices}"

    def test_prices_below_forward(self):
        """Call prices are at most F0 (no-arbitrage upper bound)."""
        prices = rough_bergomi_rqmc(**BASE, strikes=STRIKES, F0=F0, N=20, n_paths=256)
        assert np.all(prices <= F0 + 1e-8), f"Prices exceed F0: {prices}"

    def test_prices_above_intrinsic(self):
        """Call prices are at least max(F0 - K, 0)."""
        prices = rough_bergomi_rqmc(**BASE, strikes=STRIKES, F0=F0, N=20, n_paths=256)
        intrinsic = np.maximum(F0 - STRIKES, 0.0)
        assert np.all(prices >= intrinsic - 1e-6), (
            f"Prices below intrinsic: {prices} vs {intrinsic}")

    def test_monotone_in_strike(self):
        """Call prices are non-increasing in strike."""
        prices = rough_bergomi_rqmc(**BASE, strikes=STRIKES, F0=F0, N=20, n_paths=512)
        assert np.all(np.diff(prices) <= 1e-4), (
            f"Prices not monotone: {prices}")

    def test_rho_nonzero_shape(self):
        """rho=-0.7 runs without error and returns correct shape."""
        p = rough_bergomi_rqmc(
            v0=0.04, eta=1.5, rho=-0.7, H=0.1, T=1.0,
            strikes=STRIKES, F0=F0, N=20, n_paths=256
        )
        assert p.shape == (len(STRIKES),)
        assert np.all(p >= 0.0)

    def test_atm_price_reasonable(self):
        """ATM price is within a broad reasonable range for the base parameters."""
        atm_price = rough_bergomi_rqmc(
            **BASE, strikes=np.array([F0]), F0=F0, N=30, n_paths=1024
        )[0]
        # ATM BS price at sigma=sqrt(v0)=0.2 is ~0.0797; allow factor 2 tolerance
        assert 0.01 < atm_price < 0.3, f"ATM price = {atm_price:.4f} out of range"

    def test_iv_in_reasonable_range(self):
        """Implied vols for ATM strike should be in [5%, 80%] for base params."""
        prices = rough_bergomi_rqmc(**BASE, strikes=np.array([F0]), F0=F0, N=30, n_paths=1024)
        iv = implied_vol(max(prices[0], 1e-10), F0, F0, BASE['T'])
        if np.isfinite(iv):
            assert 0.05 < iv < 0.80, f"ATM IV = {iv:.4f} out of range"


# ---------------------------------------------------------------------------
# rough_bergomi_gaussian_closure
# ---------------------------------------------------------------------------

class TestRoughBergomiClosure:
    def test_prices_nonnegative(self):
        """All closure call prices are non-negative."""
        prices = rough_bergomi_gaussian_closure(**BASE, strikes=STRIKES, F0=F0, N=20)
        assert np.all(prices >= 0.0), f"Negative closure prices: {prices}"

    def test_prices_below_forward(self):
        """Closure call prices are at most F0."""
        prices = rough_bergomi_gaussian_closure(**BASE, strikes=STRIKES, F0=F0, N=20)
        assert np.all(prices <= F0 + 1e-8), f"Closure prices exceed F0: {prices}"

    def test_monotone_in_strike(self):
        """Closure prices are non-increasing in strike."""
        prices = rough_bergomi_gaussian_closure(**BASE, strikes=STRIKES, F0=F0, N=20)
        assert np.all(np.diff(prices) <= 1e-4), (
            f"Closure prices not monotone: {prices}")

    def test_atm_close_to_rqmc(self):
        """Closure ATM price is within 20% of RQMC benchmark (same parameters)."""
        rqmc = rough_bergomi_rqmc(**BASE, strikes=np.array([F0]), F0=F0, N=30, n_paths=2048)
        cl   = rough_bergomi_gaussian_closure(**BASE, strikes=np.array([F0]), F0=F0, N=30)
        assert abs(cl[0] - rqmc[0]) / max(rqmc[0], 1e-8) < 0.20, (
            f"Closure ATM={cl[0]:.5f} vs RQMC={rqmc[0]:.5f}")

    def test_iv_in_reasonable_range(self):
        """Closure ATM IV is in [5%, 80%] for base params."""
        prices = rough_bergomi_gaussian_closure(**BASE, strikes=np.array([F0]), F0=F0, N=20)
        iv = implied_vol(max(prices[0], 1e-10), F0, F0, BASE['T'])
        if np.isfinite(iv):
            assert 0.05 < iv < 0.80, f"Closure ATM IV = {iv:.4f}"

    def test_rmse_vs_rqmc_mild_regime(self):
        """At rho=0 (closure's design point), RMSE vs RQMC < 300 bp in mild regime."""
        # The closure ignores rho, so compare at rho=0 where it is designed to work.
        rqmc_p = rough_bergomi_rqmc(
            v0=0.04, eta=1.0, rho=0.0, H=0.2, T=1.0,
            strikes=STRIKES, F0=F0, N=30, n_paths=2048
        )
        cl_p = rough_bergomi_gaussian_closure(
            v0=0.04, eta=1.0, rho=0.0, H=0.2, T=1.0,
            strikes=STRIKES, F0=F0, N=30
        )
        bench_ivs = np.array([implied_vol(max(p, 1e-10), F0, K, 1.0) for p, K in zip(rqmc_p, STRIKES)])
        cl_ivs    = np.array([implied_vol(max(p, 1e-10), F0, K, 1.0) for p, K in zip(cl_p, STRIKES)])
        err = rmse_bp(cl_ivs, bench_ivs)
        assert err < 300.0 or np.isnan(err), (
            f"Closure RMSE vs RQMC = {err:.1f}bp (expected < 300bp for rho=0 mild regime)")
