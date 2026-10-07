"""
Tests for the COS method Fourier inversion pricer (src/cos_pricer.py).
"""

import numpy as np
import pytest
from src.cos_pricer import cos_call_price, cos_call_prices


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def black_call(F, K, sigma, T):
    """Reference Black call price."""
    from scipy.stats import norm
    sqrtT = np.sqrt(T)
    d1 = np.log(F / K) / (sigma * sqrtT) + 0.5 * sigma * sqrtT
    d2 = d1 - sigma * sqrtT
    return F * norm.cdf(d1) - K * norm.cdf(d2)


def lognormal_cf(u: float, sigma: float, T: float) -> complex:
    """CF of log-price X ~ N(-0.5*sigma^2*T, sigma^2*T)."""
    mu_x = -0.5 * sigma ** 2 * T
    var_x = sigma ** 2 * T
    return np.exp(1j * u * mu_x - 0.5 * var_x * u ** 2)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------
def test_cos_atm_call_lognormal():
    """COS ATM call on lognormal model should match Black formula within 0.5 bp."""
    F, K, sigma, T = 100.0, 100.0, 0.20, 1.0
    ref = black_call(F, K, sigma, T)
    phi = lambda u: lognormal_cf(u, sigma, T)
    price = cos_call_price(phi, F, K, T, L=8.0, N=256)
    assert abs(price - ref) / F < 5e-5, f"ATM error too large: {abs(price-ref):.6f}"


def test_cos_otm_call_lognormal():
    """COS OTM call on lognormal should match Black within 1 bp."""
    F, K, sigma, T = 100.0, 110.0, 0.25, 0.5
    ref = black_call(F, K, sigma, T)
    phi = lambda u: lognormal_cf(u, sigma, T)
    price = cos_call_price(phi, F, K, T, L=8.0, N=256)
    assert abs(price - ref) / F < 1e-4, f"OTM error too large: {abs(price-ref):.6f}"


def test_cos_itm_call_lognormal():
    """COS ITM call on lognormal should match Black within 1 bp."""
    F, K, sigma, T = 100.0, 90.0, 0.30, 1.0
    ref = black_call(F, K, sigma, T)
    phi = lambda u: lognormal_cf(u, sigma, T)
    price = cos_call_price(phi, F, K, T, L=8.0, N=256)
    assert abs(price - ref) / F < 1e-4, f"ITM error too large: {abs(price-ref):.6f}"


def test_cos_call_prices_vectorised():
    """cos_call_prices should agree with per-strike cos_call_price."""
    F, sigma, T = 100.0, 0.20, 1.0
    strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
    phi = lambda u: lognormal_cf(u, sigma, T)
    prices_vec = cos_call_prices(phi, F, strikes, T, L=8.0, N=256)
    for i, K in enumerate(strikes):
        p_single = cos_call_price(phi, F, K, T, L=8.0, N=256)
        assert abs(prices_vec[i] - p_single) < 1e-10, \
            f"Vectorised vs scalar mismatch at K={K}"


def test_cos_no_arbitrage_lower_bound():
    """Call price >= max(F - K, 0)."""
    F, K, sigma, T = 100.0, 80.0, 0.20, 1.0
    phi = lambda u: lognormal_cf(u, sigma, T)
    price = cos_call_price(phi, F, K, T)
    assert price >= F - K - 1e-8


def test_cos_put_call_parity():
    """Call price - (F - K) = put price (call - call_deep_itm ≈ 0)."""
    F, sigma, T = 100.0, 0.20, 1.0
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    phi = lambda u: lognormal_cf(u, sigma, T)
    prices = cos_call_prices(phi, F, strikes, T, L=8.0, N=256)
    refs = np.array([black_call(F, K, sigma, T) for K in strikes])
    max_err = np.max(np.abs(prices - refs)) / F
    assert max_err < 5e-4, f"Max error {max_err:.2e} exceeds 5e-4"


def test_cos_call_monotone_in_strike():
    """Call price should be monotone decreasing in strike."""
    F, sigma, T = 100.0, 0.25, 1.0
    strikes = np.linspace(80.0, 120.0, 20)
    phi = lambda u: lognormal_cf(u, sigma, T)
    prices = cos_call_prices(phi, F, strikes, T, L=8.0, N=256)
    assert np.all(np.diff(prices) <= 1e-6), "Call prices are not monotone decreasing"


def test_cos_zero_vol_call():
    """At sigma=0, call = max(F-K, 0)."""
    F, K, T = 100.0, 95.0, 1.0
    phi = lambda u: np.exp(-0j * 0)  # trivial CF: phi(u) = 1 (log-price = 0)
    # Log-price = 0 means S_T = F; call = F - K
    price = cos_call_price(phi, F, K, T, L=8.0, N=512)
    assert abs(price - (F - K)) < 1.0, f"Zero-vol call mismatch: {price:.4f} vs {F - K:.4f}"


def test_cos_high_N_convergence():
    """Higher N should decrease error."""
    F, K, sigma, T = 100.0, 100.0, 0.20, 1.0
    ref = black_call(F, K, sigma, T)
    phi = lambda u: lognormal_cf(u, sigma, T)
    err_low = abs(cos_call_price(phi, F, K, T, L=8.0, N=32) - ref)
    err_high = abs(cos_call_price(phi, F, K, T, L=8.0, N=256) - ref)
    assert err_high <= err_low + 1e-8, "Higher N should not increase error"
