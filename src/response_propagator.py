"""
Response-propagator bucketed vega engine.

Implements two response kernels (OU and rough fractional) and provides
both bump-and-revalue and kernel-contraction methods for bucketed vega.

Reference:
  In a SV model with log-vol field X_t and OU/fBm dynamics the response
  function R(t,s) = dX_t / dX_s propagates a unit perturbation at time s
  forward to time t.  The bucketed vega for source bucket s_k is then

      vega_k = ∫_0^T R(t, s_k) · (dP/dX_t) dt
             = ∫_0^T R(t, s_k) · (dP/dv_t) · v_t dt
"""

import time
import numpy as np


# ---------------------------------------------------------------------------
# Response kernels
# ---------------------------------------------------------------------------

def ou_response_kernel(t_array, s, theta):
    """
    OU response kernel R(t, s) = exp(-theta*(t-s)) for t > s, else 0.

    Parameters
    ----------
    t_array : array-like  evaluation times
    s       : float       source time
    theta   : float       OU mean-reversion speed

    Returns
    -------
    R : ndarray, same shape as t_array
    """
    t_array = np.asarray(t_array, dtype=float)
    return np.where(t_array > s, np.exp(-theta * (t_array - s)), 0.0)


def rough_response_kernel(t_array, s, eta, H):
    """
    Rough free-field response kernel R(t, s) = eta*sqrt(2H)*(t-s)^(H-0.5) for t > s.

    For H < 0.5 the kernel is singular (integrable) at t → s+.

    Parameters
    ----------
    t_array : array-like  evaluation times
    s       : float       source time
    eta     : float       vol-of-vol scale
    H       : float       Hurst exponent in (0, 0.5)

    Returns
    -------
    R : ndarray, same shape as t_array
    """
    t_array = np.asarray(t_array, dtype=float)
    result = np.zeros_like(t_array, dtype=float)
    mask = t_array > s
    result[mask] = eta * np.sqrt(2.0 * H) * (t_array[mask] - s) ** (H - 0.5)
    return result


# ---------------------------------------------------------------------------
# Bucketed vega — bump-and-revalue
# ---------------------------------------------------------------------------

def compute_bucketed_vega_bump(price_func, n_buckets, T, eps=1e-4):
    """
    Bump-and-revalue bucketed vega.

    Calls `price_func(vol_multipliers)` where vol_multipliers is a
    (n_buckets,) array of per-bucket vol scaling factors (base = ones).

    Parameters
    ----------
    price_func : callable  price_func(vol_multipliers) → float
    n_buckets  : int       number of equal-width time buckets over [0, T]
    T          : float     option maturity (used only for timing reference)
    eps        : float     bump size

    Returns
    -------
    vegas : (n_buckets,) array  dP/d(vol_k) for each bucket k
    elapsed : float  wall-clock seconds
    """
    t0 = time.perf_counter()
    base_price = price_func(np.ones(n_buckets))
    vegas = np.zeros(n_buckets)
    for k in range(n_buckets):
        bump = np.ones(n_buckets)
        bump[k] += eps
        bumped_price = price_func(bump)
        vegas[k] = (bumped_price - base_price) / eps
    elapsed = time.perf_counter() - t0
    return vegas, elapsed


# ---------------------------------------------------------------------------
# Bucketed vega — response-propagator contraction
# ---------------------------------------------------------------------------

def compute_bucketed_vega_response(sensitivity_func, kernel_func,
                                   t_grid, bucket_times):
    """
    Response-propagator bucketed vega via kernel contraction.

    Computes:
        vega_k = ∫_0^T kernel(t, s_k) · (dP/dv_t) dt   for each bucket k

    Parameters
    ----------
    sensitivity_func : callable  t_grid → (n_t,)  pointwise dP/dv_t
    kernel_func      : callable  (t_grid, s_k) → (n_t,)  R(t, s_k)
    t_grid           : (n_t,)   fine time grid
    bucket_times     : (n_buckets,)  representative times for each bucket

    Returns
    -------
    vegas   : (n_buckets,) array
    elapsed : float  wall-clock seconds
    """
    t0 = time.perf_counter()
    dp_dv = sensitivity_func(t_grid)          # (n_t,)
    vegas = np.zeros(len(bucket_times))
    for k, s_k in enumerate(bucket_times):
        R = kernel_func(t_grid, s_k)          # (n_t,)
        vegas[k] = np.trapz(R * dp_dv, t_grid)
    elapsed = time.perf_counter() - t0
    return vegas, elapsed
