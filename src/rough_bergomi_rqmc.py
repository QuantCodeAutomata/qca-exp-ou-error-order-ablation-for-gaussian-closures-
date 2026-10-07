"""
Rough Bergomi call pricer via randomised quasi-Monte Carlo (RQMC).

Model:
  v_t = v0 * exp(eta * Y_t - 0.5 * eta^2 * t^{2H})
  dS/S = sqrt(v_t) dW_t,  corr(dW, dY) = rho

where Y_t is a fractional Brownian motion with Hurst exponent H in (0, 0.5).

Strategy:
  1. Sample rough field Y via PCA-ordered Sobol sequences.
  2. Compute integrated variance V = ∫ v_t dt (Euler sum).
  3. For rho != 0: form the conditional log-return distribution using
     J = ∫ sqrt(v_t) dY_t ≈ sum_k sqrt(v_k) * ΔY_k.
  4. Analytical Black pricing of the Gaussian conditional on each path.
"""

import numpy as np
from scipy.stats import norm

from src.rough_field import build_rough_covariance, sample_rough_field_sobol
from src.metrics import black_call, implied_vol


def rough_bergomi_rqmc(
    v0, eta, rho, H, T, strikes, F0=1.0, N=50, n_paths=4096, n_scrambles=4
):
    """
    Rough Bergomi call prices via RQMC with Gaussian path conditioning.

    Parameters
    ----------
    v0, eta, rho, H : rough Bergomi parameters
    T               : maturity
    strikes         : array of strikes
    F0              : initial forward (default 1.0)
    N               : time discretisation steps
    n_paths         : Sobol points per scramble
    n_scrambles     : number of independent scrambles

    Returns
    -------
    prices : array of call prices, shape (len(strikes),)
    """
    strikes = np.asarray(strikes, dtype=float)
    K_cov, t_grid = build_rough_covariance(T, N, H)
    dt = T / N

    # Sample rough field: (N, n_total)
    Y_paths = sample_rough_field_sobol(K_cov, n_paths, n_scrambles)
    n_total = Y_paths.shape[1]

    # Variance process v_t = v0 * exp(eta*Y_t - 0.5*eta^2*t^{2H})
    log_v = eta * Y_paths - 0.5 * eta ** 2 * t_grid[:, None] ** (2 * H)
    v_paths = v0 * np.exp(log_v)  # (N, n_total)

    # Integrated variance V = ∫₀ᵀ v_t dt ≈ dt * Σ v_k
    V = np.sum(v_paths, axis=0) * dt  # (n_total,)

    if abs(rho) > 1e-10:
        # Stochastic integral J = ∫ sqrt(v_t) dW̃_t.
        #
        # Cholesky factor: K_cov = L @ L^T  (L lower-triangular).
        # Innovations:     epsilon_k = (L⁻¹ Y)_k  ~  N(0,1), i.i.d.
        #
        # Causal field:   Y^causal_k = Y_k - L_kk * epsilon_k
        #                            = Σ_{j<k} L_kj * epsilon_j
        # This depends only on PAST innovations, so v^causal_k is predictable.
        # Then E[sqrt(v^causal_k) * epsilon_k | epsilon_{<k}] = 0, preserving
        # the stochastic-exponential martingale E[S_T] = F0.
        L = np.linalg.cholesky(K_cov + 1e-10 * np.eye(N))
        epsilon = np.linalg.solve(L, Y_paths)          # (N, n_total)
        L_diag = np.diag(L)                            # (N,)
        Y_causal = Y_paths - L_diag[:, None] * epsilon # (N, n_total)
        log_v_c = eta * Y_causal - 0.5 * eta ** 2 * t_grid[:, None] ** (2 * H)
        v_causal = v0 * np.exp(log_v_c)               # (N, n_total)
        V = np.sum(v_causal, axis=0) * dt             # recompute with causal v
        J = np.sqrt(dt) * np.sum(np.sqrt(v_causal) * epsilon, axis=0)   # (n_total,)
        mu_cond = -0.5 * V + rho * J
        s2_cond = np.maximum((1.0 - rho ** 2) * V, 1e-12)
    else:
        mu_cond = -0.5 * V
        s2_cond = np.maximum(V, 1e-12)

    # Analytical Black pricing for each path:
    # X_T | path ~ N(mu_cond, s2_cond)
    # E[(F0*exp(X) - K)+] = F0*exp(mu+s²/2)*N(d1) - K*N(d2)
    prices = np.zeros(len(strikes))
    for i, K in enumerate(strikes):
        x = np.log(K / F0)          # log-moneyness
        s = np.sqrt(s2_cond)
        d1 = (mu_cond + s2_cond - x) / s
        d2 = d1 - s
        payoff = (
            F0 * np.exp(mu_cond + 0.5 * s2_cond) * norm.cdf(d1)
            - K * norm.cdf(d2)
        )
        prices[i] = np.mean(payoff)

    return prices
