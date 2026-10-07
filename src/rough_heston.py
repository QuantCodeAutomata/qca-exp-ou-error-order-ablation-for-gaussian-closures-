"""
Rough Heston option pricing.

Model (El Euch & Rosenbaum 2019):
  dS/S = sqrt(v_t) dW_t
  v_t via fractional Volterra equation with Hurst index H in (0, 1/2).

The log characteristic function (forward measure, r=0):
  log E[exp(iu * log(S_T/S_0))] = v0 * h(u, T) + kappa*theta * int_0^T h(u,t) dt

where h(u, T) is the endpoint value of h solving the fractional Volterra equation:
  h(t) = I^alpha[F(h) + g](t),   alpha = H + 0.5
  F(h) = (lambda^2/2)*h^2 + (i*u*rho_hv*lambda - kappa)*h
  g(u) = -(u^2 + i*u)/2   [Ito correction: E[S_T]=S_0]

Discretization: Adams-Bashforth predictor + implicit quadratic corrector.
"""

import numpy as np
from scipy.special import gamma as gamma_func
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Fractional Riccati solver (vectorised over u)
# ---------------------------------------------------------------------------

def fractional_riccati(u_arr, T, kappa, lam, H, rho_hv=0.0, N=100):
    """
    Solve the fractional Volterra equation for h(u, t_n), n = 0..N.

    h(t) = I^alpha[ F(h) + g ](t)
    F(h) = (lam^2/2)*h^2 + (i*u*rho_hv*lam - kappa)*h
    g(u) = (i*u - u^2)/2
    alpha = H + 0.5

    Uses Adams-Bashforth predictor followed by one Adams-Moulton corrector.
    Stability clamping prevents blow-up for large |u|.

    Parameters
    ----------
    u_arr   : 1-D array of real Fourier arguments
    T       : maturity
    kappa   : mean-reversion speed
    lam     : vol-of-vol in rough Heston
    H       : Hurst exponent (0, 0.5)
    rho_hv  : vol-price correlation (0 for exp_4)
    N       : number of time steps

    Returns
    -------
    h : complex array of shape (len(u_arr), N+1),  h[k, 0] = 0
    """
    u_arr = np.asarray(u_arr, dtype=float)
    n_u = len(u_arr)
    dt = T / N
    alpha = H + 0.5

    iu = 1j * u_arr                                    # (n_u,)
    # g(u) = -(u^2 + iu)/2  [Ito drift + variance forcing; preserves E[S_T]=S_0]
    g = -0.5 * (u_arr ** 2 + iu)                       # (n_u,)  complex constant
    lin_coef = iu * rho_hv * lam - kappa               # (n_u,)
    quad_coef = 0.5 * lam ** 2                         # scalar

    # Quadrature weights w_q = dt^alpha/Gamma(alpha+1)*(q^alpha - (q-1)^alpha)
    q_all = np.arange(1, N + 2, dtype=float)
    w = (dt ** alpha / gamma_func(alpha + 1.0)) * (q_all ** alpha - (q_all - 1.0) ** alpha)
    # w[q-1] is weight for distance q  (q = 1..N+1)

    # Additional corrector weight for the implicit self-term
    w0_corr = dt ** alpha / gamma_func(alpha + 2.0)

    # g * I^alpha[1](t_n) = g * n^alpha * dt^alpha / Gamma(alpha+1)
    n_idx = np.arange(1, N + 1, dtype=float)
    g_cumulative = (dt ** alpha / gamma_func(alpha + 1.0)) * n_idx ** alpha  # (N,)

    h = np.zeros((n_u, N + 1), dtype=complex)
    # h[:, 0] = 0

    # Cache F(h_j) for each j: F_cache[:, j] = F(h[:, j]) = quad*h_j^2 + lin*h_j
    F_cache = np.zeros((n_u, N + 1), dtype=complex)  # F_cache[:, 0] = 0

    # --- Implicit corrector via exact quadratic formula ---
    # The predictor (left-endpoint rule):
    #   h_n^pred = g_cumulative[n-1]*g + sum_{q=1}^{n-1} w[q-1] * F_cache[:, n-q]
    # Note: g appears ONLY via g_cumulative (no extra g in the F_cache sum).
    #
    # The corrector adds the implicit self-term for F only:
    #   h_n = h_n^pred + w0_corr * F(h_n)
    # which gives: w0_corr*quad*h_n^2 + (1 - w0_corr*lin)*h_n - h_n^pred = 0
    A_coef = w0_corr * quad_coef                          # scalar
    B_base = 1.0 - w0_corr * lin_coef                    # (n_u,) complex

    for n in range(1, N + 1):
        # Predictor: explicit sum of F over h[0..n-1] plus g via g_cumulative
        S = g * g_cumulative[n - 1]                      # I^alpha[g](t_n)
        for q in range(1, n):
            S = S + w[q - 1] * F_cache[:, n - q]         # I^alpha[F(h)](t_n) excluding h_n

        h_pred = S   # pure explicit predictor

        # Implicit corrector: h_n = S + w0_corr * F(h_n)
        # => A_coef * h_n^2 + B_base * h_n - S = 0
        # Linear case (lam=0, A_coef=0): h_n = S / B_base = h_pred / B_base
        if A_coef == 0.0:
            h_n = S / B_base
            h_n = np.where(np.isfinite(h_n), h_n, h_pred)
        else:
            C_coef = -S                                   # (n_u,)
            discriminant = B_base ** 2 - 4.0 * A_coef * C_coef
            sqrt_disc = np.sqrt(discriminant + 0j)

            # Two candidate roots
            r1 = (-B_base + sqrt_disc) / (2.0 * A_coef)
            r2 = (-B_base - sqrt_disc) / (2.0 * A_coef)

            # Primary: Re(h) ≤ 0 (ensures |CF| ≤ 1).
            # Secondary: closest to predictor for branch continuity.
            r1_neg = np.real(r1) <= 0.0
            r2_neg = np.real(r2) <= 0.0
            both_neg = r1_neg & r2_neg
            neither_neg = (~r1_neg) & (~r2_neg)
            close_r1 = np.abs(r1 - h_pred) <= np.abs(r2 - h_pred)
            use_r1 = (r1_neg & ~r2_neg) | (both_neg & close_r1) | (neither_neg & close_r1)
            h_n = np.where(use_r1, r1, r2)

            # Safety: fallback to predictor if corrector diverged
            h_n = np.where(np.isfinite(h_n), h_n, h_pred)
        h[:, n] = h_n
        F_cache[:, n] = quad_coef * h_n ** 2 + lin_coef * h_n

    return h


# ---------------------------------------------------------------------------
# Log characteristic function
# ---------------------------------------------------------------------------

def rough_heston_log_cf(u_arr, T, kappa, theta, lam, H, v0, rho_hv=0.0, N=100):
    """
    Log characteristic function of log(S_T/S_0) under rough Heston.

    From El Euch & Rosenbaum (2019), the log-CF is:
      log E[exp(iu * log(S_T/S_0))] = v0 * h(u, T) + kappa*theta * int_0^T h(u,t) dt

    where h(u, T) is the Riccati solution evaluated at the endpoint T, and
    int_0^T h(u,t)dt is the standard Riemann integral of h over [0, T].

    Parameters
    ----------
    u_arr : 1-D array of (real) Fourier arguments
    T, kappa, theta, lam, H, v0, rho_hv : model parameters
    N     : Riccati discretisation steps

    Returns
    -------
    log_cf : complex array of shape (len(u_arr),)
    """
    h = fractional_riccati(u_arr, T, kappa, lam, H, rho_hv=rho_hv, N=N)
    # h shape: (n_u, N+1), h[:, n] = h(u, n*dt)

    dt = T / N

    # h(u, T) = h[:, N]  (endpoint value)
    h_T = h[:, N]

    # int_0^T h(u,t) dt ≈ dt * (h[:,0]/2 + h[:,1] + ... + h[:,N-1] + h[:,N]/2)  (trapezoid)
    integral_h = dt * (0.5 * h[:, 0] + h[:, 1:N].sum(axis=1) + 0.5 * h[:, N])

    log_cf = v0 * h_T + kappa * theta * integral_h
    return log_cf


def rough_heston_call_cos(
    K_arr, F0, T, kappa, theta, lam, H, v0, rho_hv=0.0, N_riccati=100, N_cos=128, L=12
):
    """
    Rough Heston call prices via COS method.

    Parameters
    ----------
    K_arr     : strikes (array)
    F0        : initial forward (spot assumed)
    T         : maturity
    kappa, theta, lam, H, v0, rho_hv : rough Heston parameters
    N_riccati : time steps for fractional Riccati solver
    N_cos     : COS truncation terms
    L         : log-price truncation range multiplier

    Returns
    -------
    prices : array of call prices, shape matching K_arr
    """
    K_arr = np.asarray(K_arr, dtype=float)
    scalar = K_arr.ndim == 0
    K_arr = np.atleast_1d(K_arr)

    # Truncation interval for log-return x = log(S_T/F0)
    # Rough Heston variance ≈ theta*T for long maturities
    approx_var = max(v0 * T, theta * T, 0.04 * T)
    approx_std = np.sqrt(approx_var)
    a = -L * approx_std
    b = L * approx_std
    c = b - a

    # COS Fourier arguments: u_k = k*pi/(b-a), k = 0..N_cos-1
    k_arr = np.arange(N_cos, dtype=float)
    u_arr = k_arr * np.pi / c

    # Log characteristic function at each u_k
    log_cf = rough_heston_log_cf(u_arr, T, kappa, theta, lam, H, v0,
                                  rho_hv=rho_hv, N=N_riccati)

    # CF = exp(log_cf - i*u*a), set k=0 exactly to 1
    phi = np.exp(log_cf - 1j * u_arr * a)
    phi[0] = 1.0
    # Replace any NaN/Inf (CF blow-up for large u) with 0 (truncation)
    phi = np.where(np.isfinite(phi), phi, 0.0)

    # Payoff coefficients V_k for each strike (Fang & Oosterlee 2008)
    # Variable: y = log(S_T/K) = x - x0, x0 = log(K/F0)
    # Density of y on [a-x0, b-x0]; payoff K*(e^y-1)^+ for y in [max(a-x0,0), b-x0]
    # COS density coefficients: Re[phi(u_k)*exp(-i*u_k*a)] = Re[phi[k]] (no K-shift needed)
    prices = np.empty(len(K_arr))
    kpi_c = k_arr * np.pi / c
    phi_real = np.real(phi)   # density Fourier coefficients (K-independent)

    for i, K in enumerate(K_arr):
        x0 = np.log(K / F0)       # log-moneyness (positive for OTM call)
        a_k = a - x0              # shifted left bound in y-space
        b_k = b - x0              # shifted right bound
        lo = max(a_k, 0.0)        # payoff lower limit (y >= 0 to pay off)
        hi = b_k

        if lo >= hi:
            prices[i] = max(F0 - K, 0.0)
            continue

        # chi_k = int_{lo}^{hi} e^y cos(k*pi*(y - a_k)/c) dy
        chi = np.where(
            k_arr == 0,
            np.exp(hi) - np.exp(lo),
            (np.exp(hi) * (np.cos(kpi_c * (hi - a_k))
                           + kpi_c * np.sin(kpi_c * (hi - a_k)))
             - np.exp(lo) * (np.cos(kpi_c * (lo - a_k))
                             + kpi_c * np.sin(kpi_c * (lo - a_k))))
            / (1.0 + kpi_c ** 2),
        )

        # psi_k = int_{lo}^{hi} cos(k*pi*(y - a_k)/c) dy
        safe_kpi_c = np.where(k_arr == 0, 1.0, kpi_c)   # avoid 0/0; k=0 branch unused
        psi = np.where(
            k_arr == 0,
            hi - lo,
            (np.sin(safe_kpi_c * (hi - a_k)) - np.sin(safe_kpi_c * (lo - a_k))) / safe_kpi_c,
        )

        V_k = (2.0 / c) * (chi - psi)

        # COS sum: price = K * sum' Re[phi_k] * V_k
        terms = phi_real * V_k
        terms[0] *= 0.5
        prices[i] = K * terms.sum()

    prices = np.maximum(prices, 0.0)
    return prices[()] if scalar else prices


# ---------------------------------------------------------------------------
# Implied vol utility
# ---------------------------------------------------------------------------

def bs_impvol(price, F0, K, T):
    """Black implied vol from call price."""
    from scipy.optimize import brentq
    price = np.asarray(price, dtype=float)
    F0_s = float(F0)
    K = np.asarray(K, dtype=float)
    T_s = float(T)
    scalar = price.ndim == 0
    price, K = np.broadcast_arrays(price, K)
    out = np.full(price.shape, np.nan)

    def _bs(sigma, f, k, t):
        sqrtT = np.sqrt(t)
        d1 = np.log(f / k) / (sigma * sqrtT) + 0.5 * sigma * sqrtT
        d2 = d1 - sigma * sqrtT
        return f * norm.cdf(d1) - k * norm.cdf(d2)

    for idx in np.ndindex(price.shape):
        p, k = float(price[idx]), float(K[idx])
        lo_p = max(F0_s - k, 0.0) + 1e-12
        hi_p = F0_s - 1e-12
        if p <= lo_p or p >= hi_p or T_s <= 0:
            continue
        try:
            out[idx] = brentq(lambda s: _bs(s, F0_s, k, T_s) - p, 1e-6, 20.0, xtol=1e-8)
        except ValueError:
            pass
    return out[()] if scalar else out
