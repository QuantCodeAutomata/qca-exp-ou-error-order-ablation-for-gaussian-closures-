"""
Rough Bergomi call pricer via two-cumulant Gaussian / log-normal closure.

Strategy (rho=0 exact, rho!=0 approximate):
  1. Compute E[V] and Var[V] analytically from the fBm covariance matrix.
  2. Fit a log-normal distribution to V: log V ~ N(mu_V, sigma_V^2).
  3. Gauss-Hermite quadrature over log(V) → Black formula for each node.

For rho != 0 the closure ignores the correlation between the log-return and
the field (i.e., treats the smile as coming entirely from stochastic variance),
which introduces a systematic error in the skew. Use rough_bergomi_rqmc for
accurate rho != 0 pricing.
"""

import numpy as np

from src.rough_field import build_rough_covariance
from src.metrics import black_call


def rough_bergomi_gaussian_closure(
    v0, eta, rho, H, T, strikes, F0=1.0, N=50, n_gh=32
):
    """
    Rough Bergomi call prices via log-normal closure on integrated variance.

    Parameters
    ----------
    v0, eta, rho, H : rough Bergomi parameters
    T               : maturity
    strikes         : array of strikes
    F0              : initial forward (default 1.0)
    N               : time grid for covariance (number of steps)
    n_gh            : Gauss-Hermite quadrature order

    Returns
    -------
    prices : array of call prices, shape (len(strikes),)
    """
    strikes = np.asarray(strikes, dtype=float)
    K_cov, t_grid = build_rough_covariance(T, N, H)
    dt = T / N

    # --- First cumulant: E[V] ---
    # v_t = v0*exp(eta*W_t - 0.5*eta^2*t^{2H}) is a martingale: E[v_t] = v0.
    EV = v0 * T

    # --- Second cumulant: Var[V] ---
    # E[v_t * v_s] = v0^2 * exp(eta^2 * K(t,s))
    # Cov[v_t, v_s] = v0^2 * (exp(eta^2*K_{ts}) - 1)
    Ev_ts = v0 ** 2 * np.exp(eta ** 2 * K_cov)
    EV2 = np.sum(Ev_ts) * dt ** 2
    VarV = EV2 - EV ** 2

    # --- Log-normal fit to V ---
    # log(1 + VarV/EV^2) gives the log-variance; floor at 0 for stability.
    sig2_V = np.log(1.0 + max(VarV / EV ** 2, 1e-14))
    mu_V = np.log(EV) - 0.5 * sig2_V

    # --- Gauss-Hermite quadrature: ∫ f(x) exp(-x^2) dx ≈ Σ w_k f(x_k) ---
    nodes, weights = np.polynomial.hermite.hermgauss(n_gh)
    # Change variable: log V = mu_V + sqrt(2*sig2_V) * t
    V_nodes = np.exp(mu_V + np.sqrt(2.0 * sig2_V) * nodes)  # (n_gh,)

    prices = np.zeros(len(strikes))
    for i, K in enumerate(strikes):
        # Black vol from integrated variance at each node
        sigma_bs = np.sqrt(np.maximum(V_nodes / T, 1e-14))
        payoffs = np.array([black_call(F0, K, s, T) for s in sigma_bs])
        # Normalise by sqrt(pi) (change-of-variable factor from Gauss-Hermite)
        prices[i] = np.dot(weights, payoffs) / np.sqrt(np.pi)

    return prices
