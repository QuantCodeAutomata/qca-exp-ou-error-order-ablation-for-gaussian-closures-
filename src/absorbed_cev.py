"""
Absorbed CEV option pricing via Schroder (1989).

Model: dF = alpha * F^beta * dW,  F absorbed at 0 (beta < 1).
Lamperti transform: Z = F^(1-beta) / (1-beta),  dZ = alpha * dW (leading order).
"""

import numpy as np
from scipy.stats import ncx2, norm
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------

def bs_call(F0, K, T, sigma):
    """Black (forward) call price: dF = sigma*F*dW."""
    F0, K, T, sigma = map(np.asarray, (F0, K, T, sigma))
    sqrtT = np.sqrt(T)
    # Use a non-zero sqrtT in the denominator; np.where discards the invalid branch.
    safe_sqrtT = np.where(sqrtT > 0, sqrtT, 1.0)
    d1 = np.log(F0 / K) / (sigma * safe_sqrtT) + 0.5 * sigma * safe_sqrtT
    d2 = d1 - sigma * safe_sqrtT
    return np.where(
        sqrtT > 0,
        F0 * norm.cdf(d1) - K * norm.cdf(d2),
        np.maximum(F0 - K, 0.0),
    )


def bs_impvol(price, F0, K, T, tol=1e-8):
    """
    Black implied vol from call price (scalar or array).
    Returns NaN for prices outside the no-arbitrage band.
    """
    price = np.asarray(price, dtype=float)
    F0 = np.asarray(F0, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    scalar = price.ndim == 0
    price, F0, K, T = np.broadcast_arrays(price, F0, K, T)
    out = np.full(price.shape, np.nan)
    for idx in np.ndindex(price.shape):
        p, f, k, t = price[idx], F0[idx], K[idx], T[idx]
        lo = max(f - k, 0.0) + 1e-12
        hi = f
        if p <= lo or p >= hi or t <= 0:
            continue
        try:
            out[idx] = brentq(lambda s: bs_call(f, k, t, s) - p, 1e-6, 20.0, xtol=tol)
        except ValueError:
            pass
    return out[()] if scalar else out


# ---------------------------------------------------------------------------
# CEV pricer (Schroder 1989, Theorem 5)
# ---------------------------------------------------------------------------

def cev_call(F0, K, T, alpha, beta):
    """
    Absorbed CEV call price.

    Parameters
    ----------
    F0, K : forward and strike (positive scalars or arrays)
    T     : time to expiry
    alpha : CEV coefficient (dF = alpha * F^beta * dW)
    beta  : elasticity in [0, 1)

    Returns
    -------
    Call price (same shape as broadcast of inputs).
    """
    F0 = np.asarray(F0, dtype=float)
    K = np.asarray(K, dtype=float)

    if np.abs(beta - 1.0) < 1e-9:
        # Limiting case: Black-Scholes with sigma = alpha
        return bs_call(F0, K, T, alpha)

    betac = 1.0 - beta
    betac_inv = 1.0 / betac

    # Total variance of Lamperti coordinate Z = F^betac / betac
    var_Z = alpha ** 2 * T          # Var(Z_T) when dZ ≈ alpha * dW

    z0 = (F0 ** betac / betac) ** 2 / var_Z   # Z0^2 / var_Z
    zK = (K  ** betac / betac) ** 2 / var_Z   # ZK^2 / var_Z

    call = F0 * ncx2.sf(zK, 2.0 + betac_inv, z0) - K * ncx2.cdf(z0, betac_inv, zK)
    return np.maximum(call, 0.0)


def cev_call_lamperti(Z0_eff, K, ZK, var_Z, beta):
    """
    Absorbed CEV call from Lamperti-space coordinates.

    This is the core routine used by the SABR stack after the bridge
    displacement and variance-clock rescaling.

    Parameters
    ----------
    Z0_eff : effective Lamperti coordinate of the forward
             (= F0^(1-beta)/(1-beta) + bridge correction; may be negative)
    K      : option strike in F-space
    ZK     : Lamperti coordinate of the strike = K^(1-beta)/(1-beta)
    var_Z  : residual variance of Z: (1-rho^2) * int_0^T alpha_t^2 dt
    beta   : CEV elasticity in [0, 1)

    Returns
    -------
    Call price (scalar or array matching inputs).
    """
    Z0_eff = np.asarray(Z0_eff, dtype=float)
    K = np.asarray(K, dtype=float)
    ZK = np.asarray(ZK, dtype=float)
    var_Z = np.asarray(var_Z, dtype=float)

    betac = 1.0 - beta
    betac_inv = 1.0 / betac

    # Effective forward in F-space (zero if absorbed)
    F0_eff = np.where(Z0_eff > 0.0,
                      (betac * np.maximum(Z0_eff, 0.0)) ** betac_inv,
                      0.0)

    var_Z = np.maximum(var_Z, 1e-15)
    z0 = np.where(Z0_eff > 0.0, Z0_eff ** 2 / var_Z, 0.0)
    zK = ZK ** 2 / var_Z

    call = np.where(
        Z0_eff > 0.0,
        F0_eff * ncx2.sf(zK, 2.0 + betac_inv, z0) - K * ncx2.cdf(z0, betac_inv, zK),
        0.0,
    )
    return np.maximum(call, 0.0)


def cev_impvol(price, F0, K, T, alpha, beta):
    """Black implied vol from CEV call price (convenience wrapper)."""
    return bs_impvol(np.asarray(price), np.asarray(F0), np.asarray(K), np.asarray(T))
