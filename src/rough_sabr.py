"""
Rough SABR call pricer.

Model:
  alpha_t = alpha0 * exp(eta * Y_t - 0.5 * eta^2 * t^{2H})
  dF = alpha_t * F^beta * dW_t,  corr(dW, dY) = rho

where Y_t is a fractional Brownian motion with Hurst exponent H in (0, 0.5)
and F is absorbed at zero.

Provides:
  rough_sabr_rqmc  — RQMC pricer using absorbed-CEV pricing conditional on field
  rough_sabr_mc    — Euler-Maruyama Monte Carlo benchmark
"""

import numpy as np
from scipy.stats import norm

from src.rough_field import build_rough_covariance, sample_rough_field, sample_rough_field_sobol
from src.absorbed_cev import cev_call_lamperti, bs_impvol


def rough_sabr_rqmc(
    alpha0, eta, rho, H, beta, T, strikes, F0=100.0, N=30, n_paths=1024, n_scrambles=4
):
    """
    Rough SABR call prices via RQMC with absorbed-CEV path conditioning.

    Strategy:
      1. Sample rough field Y via PCA-ordered Sobol sequences.
      2. Compute alpha_t = alpha0 * exp(eta*Y_t - 0.5*eta^2*t^{2H}).
      3. Compute integrated Lamperti variance V_Z = ∫ alpha_t^2 dt.
      4. Apply bridge displacement for rho != 0:
           Z0_eff = Z0 + rho * ∫ alpha_t dY_t  (stochastic integral approximation)
      5. Price each path via absorbed-CEV Lamperti formula.

    Parameters
    ----------
    alpha0          : initial SABR alpha
    eta             : vol-of-vol for rough field
    rho             : correlation between dW and fBm driving noise
    H               : Hurst exponent
    beta            : CEV elasticity in [0, 1)
    T               : maturity
    strikes         : array of strikes
    F0              : initial forward
    N               : time discretisation steps
    n_paths         : Sobol points per scramble
    n_scrambles     : number of independent scrambles

    Returns
    -------
    prices : array of call prices, shape (len(strikes),)
    """
    strikes = np.asarray(strikes, dtype=float)
    betac = 1.0 - beta
    betac_inv = 1.0 / betac

    K_cov, t_grid = build_rough_covariance(T, N, H)
    dt = T / N

    Y_paths = sample_rough_field_sobol(K_cov, n_paths, n_scrambles)  # (N, n_total)
    n_total = Y_paths.shape[1]

    # Vol-of-vol paths: alpha_t = alpha0 * exp(eta*Y_t - 0.5*eta^2*t^{2H})
    log_alpha = eta * Y_paths - 0.5 * eta ** 2 * t_grid[:, None] ** (2 * H)
    alpha_paths = alpha0 * np.exp(log_alpha)  # (N, n_total)

    # Integrated Lamperti variance: V_Z = ∫ alpha_t^2 dt
    V_Z = np.sum(alpha_paths ** 2, axis=0) * dt  # (n_total,)

    # Lamperti coordinates
    Z0 = F0 ** betac / betac
    ZK_arr = strikes ** betac / betac

    if abs(rho) > 1e-10:
        # Stochastic integral J = ∫ alpha_t dW̃_t using the causal Cholesky scheme.
        # Y = L @ epsilon → epsilon_k = (L⁻¹ Y)_k ~ N(0,1).
        # Causal alpha: alpha^causal_k uses Y^causal_k = Y_k - L_kk*epsilon_k
        # (past-only field), so alpha^causal_k is predictable w.r.t. epsilon.
        L = np.linalg.cholesky(K_cov + 1e-10 * np.eye(N))
        epsilon = np.linalg.solve(L, Y_paths)               # (N, n_total)
        L_diag = np.diag(L)                                 # (N,)
        Y_causal = Y_paths - L_diag[:, None] * epsilon      # (N, n_total)
        log_alpha_c = eta * Y_causal - 0.5 * eta ** 2 * t_grid[:, None] ** (2 * H)
        alpha_causal = alpha0 * np.exp(log_alpha_c)
        J = np.sqrt(dt) * np.sum(alpha_causal * epsilon, axis=0)    # (n_total,)
        # Recompute V_Z with causal alpha
        V_Z = np.sum(alpha_causal ** 2, axis=0) * dt
        Z0_eff = Z0 + rho * J
        var_Z = np.maximum((1.0 - rho ** 2) * V_Z, 1e-14)
    else:
        Z0_eff = np.full(n_total, Z0)
        var_Z = np.maximum(V_Z, 1e-14)

    prices = np.zeros(len(strikes))
    for i, (K, ZK) in enumerate(zip(strikes, ZK_arr)):
        # Absorbed-CEV Lamperti pricing for each path, then average
        payoffs = cev_call_lamperti(Z0_eff, K, ZK, var_Z, beta)
        prices[i] = np.mean(payoffs)

    return prices


def rough_sabr_mc(
    alpha0, eta, rho, H, beta, T, strikes, F0=100.0, N=200, n_paths=4096, seed=42
):
    """
    Rough SABR call prices via Euler-Maruyama Monte Carlo (benchmark).

    Uses log-Euler for alpha (preserves positivity) and Euler for F
    (absorbed at 0 via max(F, 0)).

    The rough field Y is sampled exactly from the fBm covariance using Cholesky,
    then used path-by-path for alpha; F is evolved with correlated increments.

    Parameters
    ----------
    alpha0  : initial SABR alpha
    eta     : vol-of-vol for rough field
    rho     : correlation between dW_F and fBm increments
    H       : Hurst exponent
    beta    : CEV elasticity in [0, 1)
    T       : maturity
    strikes : array of strikes
    F0      : initial forward
    N       : time discretisation steps
    n_paths : number of Monte Carlo paths
    seed    : random seed

    Returns
    -------
    prices : array of call prices, shape (len(strikes),)
    """
    strikes = np.asarray(strikes, dtype=float)
    K_cov, t_grid = build_rough_covariance(T, N, H)
    dt = T / N
    sqrt_1mrho2 = np.sqrt(max(1.0 - rho ** 2, 0.0))

    # Sample entire rough field paths: (N, n_paths)
    Y_paths = sample_rough_field(K_cov, n_paths, seed=seed)

    # Cholesky innovations: Y = L @ epsilon, epsilon_k ~ i.i.d. N(0,1).
    L = np.linalg.cholesky(K_cov + 1e-10 * np.eye(N))
    epsilon = np.linalg.solve(L, Y_paths)          # (N, n_paths)
    L_diag = np.diag(L)                            # (N,)

    # Causal field: Y^causal_k = Y_k - L_kk * epsilon_k  (past-only, predictable)
    Y_causal = Y_paths - L_diag[:, None] * epsilon  # (N, n_paths)

    # Compute alpha from causal field so alpha_k is predictable w.r.t. epsilon_k
    log_alpha = eta * Y_causal - 0.5 * eta ** 2 * t_grid[:, None] ** (2 * H)
    alpha_paths = alpha0 * np.exp(log_alpha)         # (N, n_paths)

    # Simulate F using Euler steps with properly correlated noise.
    # epsilon_k ~ N(0,1) independent BM innovation; dZ_k is independent.
    rng = np.random.default_rng(seed + 1)
    F = np.full(n_paths, F0)
    sqrt_dt = np.sqrt(dt)

    for k in range(N):
        alpha_k = alpha_paths[k, :]          # (n_paths,)
        dW_F = rho * epsilon[k, :] * sqrt_dt + sqrt_1mrho2 * rng.standard_normal(n_paths) * sqrt_dt
        F = np.maximum(F + alpha_k * np.maximum(F, 0.0) ** beta * dW_F, 0.0)

    payoffs = np.maximum(F[:, None] - strikes[None, :], 0.0)  # (n_paths, n_K)
    prices = payoffs.mean(axis=0)
    return prices
