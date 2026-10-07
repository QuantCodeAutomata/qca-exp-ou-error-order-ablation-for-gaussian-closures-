"""
SABR vanilla pricing.

Includes:
  - Hagan (2002) asymptotic implied-vol formula
  - Routed deterministic stack: outer Gauss-Hermite quadrature over alpha_T,
    inner absorbed-CEV pricing with bridge displacement and variance clock
  - Euler-Maruyama Monte Carlo benchmark
"""

import numpy as np
from scipy.special import roots_hermitenorm

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from absorbed_cev import cev_call_lamperti, bs_call, bs_impvol


# ---------------------------------------------------------------------------
# Hagan (2002) SABR implied-vol formula
# ---------------------------------------------------------------------------

def hagan_impvol(K, F, T, alpha, beta, rho, nu):
    """
    Hagan (2002) SABR Black implied volatility.

    Model: dF = alpha * F^beta * dW,  d(alpha) = nu * alpha * dZ,  corr = rho.

    Parameters
    ----------
    K     : strike (scalar or array)
    F     : forward (scalar)
    T     : time to expiry
    alpha : initial SABR alpha
    beta  : CEV elasticity
    rho   : correlation
    nu    : vol-of-vol

    Returns
    -------
    Black implied vol (same shape as K).
    """
    K = np.asarray(K, dtype=float)
    scalar = K.ndim == 0
    K = np.atleast_1d(K)

    betac = 1.0 - beta
    FK = F * K
    FK_mid = FK ** (betac / 2.0)           # (F*K)^{(1-beta)/2}
    FK_betac = FK ** betac                 # (F*K)^{1-beta}

    # log(F/K), handle ATM separately
    log_FK = np.log(F / K)
    atm = np.abs(log_FK) < 1e-8

    # Expansion coefficient A (moneyness correction to ATM)
    A = 1.0 + (betac ** 2 / 24.0) * log_FK ** 2 + (betac ** 4 / 1920.0) * log_FK ** 4

    # z-factor and chi-function (mapping to normal vol scale)
    if nu > 1e-10:
        z = (nu / alpha) * FK_mid * log_FK
    else:
        z = np.zeros_like(K)

    # chi(z) = log((sqrt(1-2*rho*z+z^2) + z - rho) / (1-rho))
    disc = np.sqrt(np.maximum(1.0 - 2.0 * rho * z + z ** 2, 0.0))
    denom = disc + z - rho
    # Guard against log(0) or negative argument from numerics
    denom = np.maximum(denom, 1e-14)
    chi_z = np.log(denom / (1.0 - rho))

    # z/chi(z): limit is 1 at z=0; use series 1 + rho*z/2 + ... near zero
    safe_ratio = np.where(np.abs(z) > 1e-6, z / np.where(np.abs(chi_z) > 1e-14, chi_z, 1e-14),
                          1.0 - rho * z / 2.0)
    z_over_chi = safe_ratio

    # Time-to-expiry correction B
    B = (
        betac ** 2 * alpha ** 2 / (24.0 * FK_betac)
        + 0.25 * rho * beta * nu * alpha / FK_mid
        + (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2
    )

    # Backbone: sigma_B = alpha / (F*K)^{(1-beta)/2}
    backbone = alpha / FK_mid

    sigma = backbone * z_over_chi * A * (1.0 + B * T)

    # ATM shortcut (numerically safe for K very close to F)
    atm_backbone = alpha / F ** betac
    atm_B = (
        betac ** 2 * alpha ** 2 / (24.0 * F ** (2.0 * betac))
        + 0.25 * rho * beta * nu * alpha / F ** betac
        + (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2
    )
    sigma_atm = atm_backbone * (1.0 + atm_B * T)
    sigma = np.where(atm, sigma_atm, sigma)

    return sigma[()] if scalar else sigma


# ---------------------------------------------------------------------------
# SABR routed stack: GH quadrature over alpha_T
# ---------------------------------------------------------------------------

def sabr_stack_call(K, F0, T, alpha0, beta, rho, nu, n_gh=32):
    """
    SABR call price via deterministic routed stack.

    Outer Gauss-Hermite quadrature integrates over alpha_T ~ LogNormal.
    For each node: bridge-displaced Lamperti coordinate + variance-clock
    rescaling feed an absorbed-CEV pricer.

    Parameters
    ----------
    K      : strike or array of strikes
    F0     : initial forward
    T      : time to expiry
    alpha0 : initial SABR alpha
    beta   : CEV elasticity in [0, 1)
    rho    : SABR correlation
    nu     : vol-of-vol (> 0)
    n_gh   : number of Gauss-Hermite quadrature nodes

    Returns
    -------
    Call price(s) with shape matching K.
    """
    K = np.asarray(K, dtype=float)
    scalar = K.ndim == 0
    K = np.atleast_1d(K)

    betac = 1.0 - beta

    # Lamperti coordinates
    Z0 = F0 ** betac / betac
    ZK = K ** betac / betac       # shape (n_strikes,)

    # Gauss-Hermite nodes & weights for N(0,1).
    # roots_hermitenorm returns weights for int f(x)*exp(-x^2/2) dx, so we
    # divide by sqrt(2*pi) to get the N(0,1) expectation.
    gh_nodes, gh_weights = roots_hermitenorm(n_gh)
    gh_weights = gh_weights / np.sqrt(2.0 * np.pi)

    # alpha_T nodes: LogNormal(log(alpha0) - 0.5*nu^2*T, nu*sqrt(T))
    alpha_T_nodes = alpha0 * np.exp(nu * np.sqrt(T) * gh_nodes - 0.5 * nu ** 2 * T)

    # Bridge displacement in Lamperti space (conditioned on alpha_T)
    # From Brownian bridge: E[Z_T | alpha_T] ≈ Z0 + (rho/nu)*(alpha_T - alpha0)
    Z0_eff_nodes = Z0 + (rho / nu) * (alpha_T_nodes - alpha0)   # shape (n_gh,)

    # Residual variance of Lamperti Z after conditioning on alpha_T:
    # Var[Z_T | alpha_T] ≈ (1-rho^2) * 0.5*(alpha0^2 + alpha_T^2) * T
    var_Z_nodes = (1.0 - rho ** 2) * 0.5 * (alpha0 ** 2 + alpha_T_nodes ** 2) * T

    # Price each (node, strike) combination
    # Z0_eff_nodes: (n_gh,), ZK: (n_strikes,)
    Z0_eff_2d = Z0_eff_nodes[:, np.newaxis]       # (n_gh, 1)
    var_Z_2d = var_Z_nodes[:, np.newaxis]          # (n_gh, 1)
    K_2d = K[np.newaxis, :]                        # (1, n_strikes)
    ZK_2d = ZK[np.newaxis, :]                      # (1, n_strikes)

    # CEV call prices for each (node, strike)
    prices_nodes = cev_call_lamperti(Z0_eff_2d, K_2d, ZK_2d, var_Z_2d, beta)
    # prices_nodes shape: (n_gh, n_strikes)

    # Quadrature: sum_k w_k * price_k
    call_prices = gh_weights @ prices_nodes        # (n_strikes,)

    return call_prices[()] if scalar else call_prices


def sabr_stack_impvol(K, F0, T, alpha0, beta, rho, nu, n_gh=32):
    """Black implied vol from SABR routed stack call prices."""
    prices = sabr_stack_call(K, F0, T, alpha0, beta, rho, nu, n_gh=n_gh)
    return bs_impvol(prices, F0, K, T)


# ---------------------------------------------------------------------------
# Monte Carlo benchmark
# ---------------------------------------------------------------------------

def sabr_mc(
    K,
    F0,
    T,
    alpha0,
    beta,
    rho,
    nu,
    n_paths=16384,
    n_steps_per_year=512,
    seed=42,
):
    """
    SABR call price(s) via Euler-Maruyama Monte Carlo.

    Uses log-Euler for alpha (preserves positivity) and Euler for F
    (absorbed at 0).

    Parameters
    ----------
    K               : strike or array of strikes
    F0, T, alpha0   : SABR parameters
    beta, rho, nu   : SABR parameters
    n_paths         : number of MC paths (default 2^14 = 16384)
    n_steps_per_year: time steps per year (default 512)
    seed            : RNG seed

    Returns
    -------
    (prices, std_errors) each with shape matching K.
    """
    K = np.asarray(K, dtype=float)
    scalar = K.ndim == 0
    K = np.atleast_1d(K)

    n_steps = max(int(np.ceil(n_steps_per_year * T)), 1)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    sqrt_1mrho2 = np.sqrt(max(1.0 - rho ** 2, 0.0))

    rng = np.random.default_rng(seed)

    F = np.full(n_paths, F0)
    alpha = np.full(n_paths, alpha0)

    for _ in range(n_steps):
        Z1 = rng.standard_normal(n_paths)
        Z2 = rng.standard_normal(n_paths)
        dW_alpha = Z1
        dW_F = rho * Z1 + sqrt_1mrho2 * Z2

        # Log-Euler for alpha (exact for GBM)
        alpha = alpha * np.exp(nu * sqrt_dt * dW_alpha - 0.5 * nu ** 2 * dt)

        # Euler for F with absorption at 0
        F = np.maximum(F + alpha * np.maximum(F, 0.0) ** beta * sqrt_dt * dW_F, 0.0)

    # Payoffs for each strike
    payoffs = np.maximum(F[:, np.newaxis] - K[np.newaxis, :], 0.0)  # (n_paths, n_K)
    prices = payoffs.mean(axis=0)
    std_errs = payoffs.std(axis=0) / np.sqrt(n_paths)

    if scalar:
        return float(prices[0]), float(std_errs[0])
    return prices, std_errs


def sabr_mc_impvol(K, F0, T, alpha0, beta, rho, nu, n_paths=16384, n_steps_per_year=512, seed=42):
    """Black implied vol from SABR MC call prices."""
    prices, _ = sabr_mc(K, F0, T, alpha0, beta, rho, nu,
                        n_paths=n_paths, n_steps_per_year=n_steps_per_year, seed=seed)
    return bs_impvol(prices, F0, K, T)
