"""
Shared pricing and metric utilities.

Provides:
  black_call  — Black (forward) call price
  implied_vol — Black implied vol from call price
  rmse_bp     — RMSE between two IV arrays in basis points
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def black_call(F0, K, sigma, T):
    """
    Black (forward) call price.

    Parameters
    ----------
    F0    : forward
    K     : strike
    sigma : Black implied vol
    T     : time to expiry

    Returns
    -------
    Call price (scalar or array matching broadcast of inputs).
    """
    F0, K, sigma, T = map(np.asarray, (F0, K, sigma, T))
    sqrtT = np.sqrt(np.maximum(T, 0.0))
    safe_sqrtT = np.where(sqrtT > 0, sqrtT, 1.0)
    d1 = np.log(F0 / K) / (sigma * safe_sqrtT) + 0.5 * sigma * safe_sqrtT
    d2 = d1 - sigma * safe_sqrtT
    price = np.where(
        sqrtT > 0,
        F0 * norm.cdf(d1) - K * norm.cdf(d2),
        np.maximum(F0 - K, 0.0),
    )
    return float(price) if price.ndim == 0 else price


def implied_vol(price, F0, K, T, tol=1e-8):
    """
    Black implied vol from call price (scalar).

    Returns NaN when the price is outside the no-arbitrage band or T <= 0.

    Parameters
    ----------
    price : call price
    F0    : forward
    K     : strike
    T     : time to expiry

    Returns
    -------
    Implied vol (float), or NaN if not invertible.
    """
    p, f, k, t = float(price), float(F0), float(K), float(T)
    lo = max(f - k, 0.0) + 1e-12
    hi = f - 1e-12
    if p <= lo or p >= hi or t <= 0:
        return np.nan
    try:
        return brentq(lambda s: black_call(f, k, s, t) - p, 1e-6, 20.0, xtol=tol)
    except ValueError:
        return np.nan


def rmse_bp(iv_method, iv_bench):
    """
    Root-mean-squared error between two IV arrays in basis points (1bp = 0.01%).

    Parameters
    ----------
    iv_method : array-like of implied vols (in decimal, e.g. 0.20 for 20%)
    iv_bench  : array-like of benchmark implied vols

    Returns
    -------
    RMSE in basis points (float).
    """
    iv_m = np.asarray(iv_method, dtype=float)
    iv_b = np.asarray(iv_bench, dtype=float)
    mask = np.isfinite(iv_m) & np.isfinite(iv_b)
    if not mask.any():
        return np.nan
    return float(np.sqrt(np.mean(((iv_m[mask] - iv_b[mask]) * 1e4) ** 2)))
