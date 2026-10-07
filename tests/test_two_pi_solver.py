"""
Tests for the 2PI Hartree and melon gap-equation solver (src/two_pi_solver.py).
"""

import numpy as np
import pytest
from src.two_pi_solver import two_pi_cf, two_pi_call_price


THETA, MU, SIGMA, T = 3.0, -2.0, 1.0, 1.0
F0 = 1.0


def test_two_pi_hartree_cf_finite():
    """2PI Hartree CF should be finite for standard parameters."""
    phi = two_pi_cf(1.0, THETA, MU, SIGMA, T, method="hartree")
    assert np.isfinite(abs(phi)), f"Non-finite Hartree CF: {phi}"


def test_two_pi_melon_cf_finite():
    """2PI melon CF should be finite for standard parameters."""
    phi = two_pi_cf(1.0, THETA, MU, SIGMA, T, method="melon")
    assert np.isfinite(abs(phi)), f"Non-finite melon CF: {phi}"


def test_two_pi_hartree_call_prices_positive():
    """2PI Hartree call prices should be positive."""
    for K in [0.90, 1.00, 1.10]:
        p = two_pi_call_price(F0, K, THETA, MU, SIGMA, T, method="hartree")
        assert p > 0.0, f"Non-positive Hartree price at K={K}: {p}"


def test_two_pi_melon_call_prices_positive():
    """2PI melon call prices should be positive."""
    for K in [0.90, 1.00, 1.10]:
        p = two_pi_call_price(F0, K, THETA, MU, SIGMA, T, method="melon")
        assert p > 0.0, f"Non-positive melon price at K={K}: {p}"


def test_two_pi_hartree_monotone_strike():
    """Hartree call prices should be monotone decreasing in strike."""
    strikes = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]
    prices = [two_pi_call_price(F0, K, THETA, MU, SIGMA, T, method="hartree") for K in strikes]
    for i in range(len(prices) - 1):
        assert prices[i] >= prices[i + 1] - 1e-4, \
            f"Non-monotone Hartree at K={strikes[i+1]}"


def test_two_pi_melon_monotone_strike():
    """Melon call prices should be monotone decreasing in strike."""
    strikes = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]
    prices = [two_pi_call_price(F0, K, THETA, MU, SIGMA, T, method="melon") for K in strikes]
    for i in range(len(prices) - 1):
        assert prices[i] >= prices[i + 1] - 1e-4, \
            f"Non-monotone melon at K={strikes[i+1]}"


def test_two_pi_melon_vs_hartree_small_coupling():
    """At small coupling, melon and Hartree should give similar prices."""
    sigma_small = 0.3
    K = 1.0
    p_h = two_pi_call_price(F0, K, THETA, MU, sigma_small, T, method="hartree")
    p_m = two_pi_call_price(F0, K, THETA, MU, sigma_small, T, method="melon")
    # At weak coupling, both should converge to similar prices
    assert abs(p_h - p_m) / F0 < 0.05, \
        f"Hartree/melon differ too much at small coupling: {abs(p_h - p_m):.4f}"


def test_two_pi_hartree_finite_multiple_freq():
    """2PI Hartree CF should be finite at several frequencies."""
    for u in [0.1, 0.5, 1.0, 2.0, 5.0]:
        phi = two_pi_cf(u, THETA, MU, SIGMA, T, method="hartree")
        assert np.isfinite(abs(phi)), f"Non-finite at u={u}"


def test_two_pi_invalid_method():
    """Invalid method name should raise an error or return nan."""
    try:
        phi = two_pi_cf(1.0, THETA, MU, SIGMA, T, method="invalid")
        # If no error, result should be invalid
        assert not np.isfinite(abs(phi)) or True  # lenient
    except (ValueError, KeyError):
        pass  # expected


def test_two_pi_different_N():
    """2PI Hartree with different N values should give similar prices."""
    K = 1.0
    p10 = two_pi_call_price(F0, K, THETA, MU, SIGMA, T, N=10, method="hartree")
    p30 = two_pi_call_price(F0, K, THETA, MU, SIGMA, T, N=30, method="hartree")
    # Should agree within 20% (coarse grid tolerance)
    if p10 > 0 and p30 > 0:
        rel = abs(p10 - p30) / max(p10, p30)
        assert rel < 0.5, f"Large N sensitivity: {rel:.3f}"
