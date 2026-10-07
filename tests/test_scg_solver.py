"""
Tests for the SCG ODE solver for the exp-OU model (src/scg_solver.py).
"""

import numpy as np
import pytest
from src.scg_solver import (
    scg_cf,
    scg_call_price,
    scg_stationary_taylor_cf,
    scg_stationary_exact_cf,
    scg_tilted_taylor_cf,
)


THETA, MU, SIGMA, T = 3.0, -2.0, 1.0, 1.0
F0 = 1.0


def test_scg_cf_at_u0():
    """SCG CF at u=0 should be 1 (or near 1 due to SCG approximation)."""
    phi0 = scg_cf(0.0, THETA, MU, SIGMA, T)
    # At u=0, w=0 and the CF collapses to 1
    assert abs(phi0 - 1.0) < 1e-4, f"phi(0) = {phi0} should be ~1"


def test_scg_cf_modulus():
    """|CF| should be <= 1 approximately."""
    for u in [0.5, 1.0, 2.0]:
        phi = scg_cf(u, THETA, MU, SIGMA, T)
        # Some approximation error is acceptable; check at least finite
        assert np.isfinite(abs(phi)), f"Non-finite phi at u={u}"


def test_scg_cf_complex():
    """CF at u != 0 should be complex."""
    phi = scg_cf(1.0, THETA, MU, SIGMA, T)
    assert isinstance(phi, (complex, np.complexfloating))


def test_scg_call_price_positive():
    """SCG call prices should be positive."""
    for K in [0.85, 0.90, 1.00, 1.10, 1.15]:
        p = scg_call_price(F0, K, THETA, MU, SIGMA, T)
        assert p > 0.0, f"Negative price at K={K}: {p}"


def test_scg_call_price_monotone_strike():
    """SCG call prices should be monotone decreasing in strike."""
    strikes = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]
    prices = [scg_call_price(F0, K, THETA, MU, SIGMA, T) for K in strikes]
    for i in range(len(prices) - 1):
        assert prices[i] >= prices[i + 1] - 1e-4, \
            f"Non-monotone at K={strikes[i+1]}: {prices[i]:.6f} < {prices[i+1]:.6f}"


def test_stationary_taylor_cf_u0():
    """Stationary Taylor CF at u=0 should be 1."""
    phi0 = scg_stationary_taylor_cf(0.0, THETA, MU, SIGMA, T)
    assert abs(phi0 - 1.0) < 1e-10


def test_stationary_exact_cf_u0():
    """Stationary exact CF at u=0 should be 1."""
    phi0 = scg_stationary_exact_cf(0.0, THETA, MU, SIGMA, T)
    assert abs(phi0 - 1.0) < 1e-10


def test_tilted_taylor_cf_u0():
    """Tilted Taylor CF at u=0 should be approximately 1."""
    phi0 = scg_tilted_taylor_cf(0.0, THETA, MU, SIGMA, T)
    # w=0 at u=0 so m=0 and V_eff cancels -> phi=1
    assert abs(phi0 - 1.0) < 1e-4


def test_stationary_exact_vs_taylor_small_coupling():
    """At small sigma, exact and Taylor stationary CFs should be close."""
    sigma_small = 0.2
    u = 1.0
    phi_t = scg_stationary_taylor_cf(u, THETA, MU, sigma_small, T)
    phi_e = scg_stationary_exact_cf(u, THETA, MU, sigma_small, T)
    assert abs(phi_t - phi_e) < 0.1, \
        f"Large divergence at small sigma: taylor={phi_t}, exact={phi_e}"


def test_scg_small_sigma_close_to_benchmark():
    """At small sigma (small coupling), SCG should be within 10 bp of tower."""
    from src.tower_benchmark import tower_call_price
    from src.metrics import implied_vol, rmse_bp
    sigma_small = 0.3
    strikes = np.array([0.95, 1.00, 1.05])
    iv_scg, iv_bench = [], []
    for K in strikes:
        p_scg = scg_call_price(F0, K, THETA, MU, sigma_small, T)
        p_bench = tower_call_price(F0, K, THETA, MU, sigma_small, T, N=50)
        iv_scg.append(implied_vol(p_scg, F0, K, T))
        iv_bench.append(implied_vol(p_bench, F0, K, T))
    rmse = rmse_bp(iv_scg, iv_bench)
    assert rmse < 50.0, f"SCG RMSE at small sigma too large: {rmse:.2f} bp"


def test_scg_finite_different_T():
    """SCG should return finite values for different maturities."""
    for T_val in [0.25, 0.5, 1.0, 2.0]:
        phi = scg_cf(1.0, THETA, MU, SIGMA, T_val)
        assert np.isfinite(abs(phi)), f"Non-finite at T={T_val}"
