"""
Tests for the exp-OU tower recursion benchmark (src/tower_benchmark.py).
"""

import numpy as np
import pytest
from src.tower_benchmark import tower_cf, tower_cf_vec, tower_call_price


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
THETA, MU, SIGMA, T = 3.0, -2.0, 1.0, 1.0
F0 = 1.0


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------
def test_tower_cf_normalisation():
    """CF at u=0 should equal 1 (martingale condition)."""
    phi0 = tower_cf(0.0, THETA, MU, SIGMA, T, N=50)
    assert abs(phi0 - 1.0) < 1e-3, f"phi(0)={phi0} should be ~1"


def test_tower_cf_complex_valued():
    """CF at u != 0 should be complex."""
    phi = tower_cf(0.5, THETA, MU, SIGMA, T, N=50)
    assert isinstance(phi, complex) or isinstance(phi, np.complexfloating)


def test_tower_cf_modulus_le_one():
    """|phi(u)| <= 1 for all real u (CF is bounded by 1)."""
    for u in [0.5, 1.0, 2.0, 5.0]:
        phi = tower_cf(u, THETA, MU, SIGMA, T, N=50)
        assert abs(phi) <= 1.0 + 1e-6, f"|phi({u})|={abs(phi):.4f} > 1"


def test_tower_cf_vec_consistency():
    """tower_cf_vec should agree with per-call tower_cf."""
    u_arr = np.array([0.0, 0.5, 1.0, 2.0])
    phi_vec = tower_cf_vec(u_arr, THETA, MU, SIGMA, T, N=30)
    for u, phi_v in zip(u_arr, phi_vec):
        phi_s = tower_cf(u, THETA, MU, SIGMA, T, N=30)
        assert abs(phi_v - phi_s) < 1e-10, f"Vec vs scalar mismatch at u={u}"


def test_tower_call_price_positive():
    """Call price should be positive for all strikes."""
    for K in [0.85, 0.90, 1.00, 1.10, 1.15]:
        p = tower_call_price(F0, K, THETA, MU, SIGMA, T, N=30)
        assert p > 0.0, f"Negative call price at K={K}: {p}"


def test_tower_call_price_decreasing_in_strike():
    """Call price should be monotone decreasing in strike."""
    strikes = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]
    prices = [tower_call_price(F0, K, THETA, MU, SIGMA, T, N=30) for K in strikes]
    for i in range(len(prices) - 1):
        assert prices[i] >= prices[i + 1] - 1e-6, \
            f"Non-monotone at K={strikes[i+1]}: {prices[i]:.6f} < {prices[i+1]:.6f}"


def test_tower_call_price_lower_bound():
    """Call price >= max(F - K, 0) (intrinsic value)."""
    for K in [0.85, 1.00, 1.15]:
        p = tower_call_price(F0, K, THETA, MU, SIGMA, T, N=30)
        lb = max(F0 - K, 0.0)
        assert p >= lb - 1e-6, f"Below intrinsic at K={K}: price={p:.6f}, lb={lb:.6f}"


def test_tower_refinement_convergence():
    """Finer tower should not drastically differ from coarse (convergence check)."""
    K = 1.0
    p_coarse = tower_call_price(F0, K, THETA, MU, SIGMA, T, N=20)
    p_fine = tower_call_price(F0, K, THETA, MU, SIGMA, T, N=80)
    assert abs(p_coarse - p_fine) < 0.05, \
        f"Refinement error too large: {abs(p_coarse - p_fine):.6f}"


def test_tower_small_sigma_reduces_variance():
    """Smaller sigma should give smaller variance (implied vol closer to flat)."""
    K = 1.05
    p_small = tower_call_price(F0, K, THETA, MU, 0.3, T, N=30)
    p_large = tower_call_price(F0, K, THETA, MU, 1.5, T, N=30)
    # Larger sigma = more vol = higher call price for OTM
    assert p_large >= p_small - 1e-4


def test_tower_atm_price_in_range():
    """ATM call price should be in a reasonable range (0 to F0)."""
    p = tower_call_price(F0, F0, THETA, MU, SIGMA, T, N=50)
    assert 0.0 < p < F0, f"ATM price out of range: {p:.6f}"
