"""
Exotic option pricing under stochastic-volatility models.

Covers:
  - exp-OU stochastic volatility: one-touch, reverse knock-out call, FVA
  - Black-76 helper for flat-vol benchmarks
"""

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Black-76 helper
# ---------------------------------------------------------------------------

def black_call(F, K, vol, T):
    """Black-76 call price (forward F, strike K, flat vol, maturity T)."""
    if vol <= 0 or T <= 0:
        return max(float(F - K), 0.0)
    sqrtT = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * vol ** 2 * T) / (vol * sqrtT)
    return float(F * norm.cdf(d1) - K * norm.cdf(d1 - vol * sqrtT))


# ---------------------------------------------------------------------------
# exp-OU path simulator
# ---------------------------------------------------------------------------

def simulate_expou_paths(theta, mu, sigma, rho, T, n_paths, n_steps, seed=42):
    """
    Simulate exp-OU stochastic-volatility paths.

    Model:
        dX_t = -theta*(X_t - mu)*dt + sigma*dW1_t
        dlogS_t = -0.5*v_t*dt + sqrt(v_t)*dW2_t
        v_t = exp(X_t)
        corr(dW1, dW2) = rho

    X is initialised at the stationary mean mu.

    Parameters
    ----------
    theta, mu, sigma : OU mean-reversion speed, long-run mean, vol-of-vol
    rho   : spot-vol correlation
    T     : maturity
    n_paths, n_steps : MC dimensions
    seed  : RNG seed

    Returns
    -------
    logS_hist : (n_steps+1, n_paths)  log-spot path (logS[0]=0)
    X_hist    : (n_steps+1, n_paths)  log-vol (OU field) path
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    rho_perp = np.sqrt(1.0 - rho ** 2)

    X = np.full(n_paths, mu, dtype=float)
    logS = np.zeros(n_paths)

    X_hist = [X.copy()]
    logS_hist = [logS.copy()]

    for _ in range(n_steps):
        Z1 = rng.standard_normal(n_paths)
        Z2 = rho * Z1 + rho_perp * rng.standard_normal(n_paths)
        X = X - theta * (X - mu) * dt + sigma * sqrt_dt * Z1
        v = np.exp(X)
        logS = logS - 0.5 * v * dt + np.sqrt(v * dt) * Z2
        X_hist.append(X.copy())
        logS_hist.append(logS.copy())

    return np.array(logS_hist), np.array(X_hist)


# ---------------------------------------------------------------------------
# Exotic pricers
# ---------------------------------------------------------------------------

def price_one_touch_mc(theta, mu, sigma, rho, T, barrier_ratio,
                       F0=100.0, n_paths=10000, n_steps=100, seed=42):
    """
    Monte Carlo price for an upside one-touch barrier option.

    Pays 1 if max_{0<=t<=T} S_t >= B = F0 * barrier_ratio.

    Returns
    -------
    price : float  (touch probability)
    se    : float  (MC standard error)
    """
    B = F0 * barrier_ratio
    logS_hist, _ = simulate_expou_paths(theta, mu, sigma, rho, T,
                                        n_paths, n_steps, seed)
    S_hist = F0 * np.exp(logS_hist)          # (n_steps+1, n_paths)
    touched = np.any(S_hist >= B, axis=0)
    price = float(np.mean(touched))
    se = float(np.std(touched.astype(float)) / np.sqrt(n_paths))
    return price, se


def price_reverse_knockout_mc(theta, mu, sigma, rho, T, K, barrier_ratio,
                              F0=100.0, n_paths=10000, n_steps=100, seed=42):
    """
    Monte Carlo price for a reverse knock-out call option.

    Vanilla call (strike K) that is knocked out if S_t >= B = F0*barrier_ratio
    at any monitoring date.

    Returns
    -------
    price : float
    se    : float  (MC standard error)
    """
    B = F0 * barrier_ratio
    logS_hist, _ = simulate_expou_paths(theta, mu, sigma, rho, T,
                                        n_paths, n_steps, seed)
    S_hist = F0 * np.exp(logS_hist)
    alive = np.all(S_hist < B, axis=0)      # path survives entire monitoring
    S_T = S_hist[-1]
    payoff = alive * np.maximum(S_T - K, 0.0)
    price = float(np.mean(payoff))
    se = float(np.std(payoff) / np.sqrt(n_paths))
    return price, se


def price_fva_mc(theta, mu, sigma, rho, T_start, Delta,
                 F0=100.0, n_paths=10000, n_steps=100, seed=42):
    """
    Forward-Volatility Agreement (FVA) price.

    Approximated as E[sqrt(int_{T_start}^{T_start+Delta} v_t dt)].

    Parameters
    ----------
    T_start : start of the accumulation window
    Delta   : length of the accumulation window

    Returns
    -------
    price : float  E[sqrt(realised vol in [T_start, T_start+Delta])]
    se    : float  (MC standard error)
    """
    T_full = T_start + Delta
    rng = np.random.default_rng(seed)
    dt = T_full / n_steps
    sqrt_dt = np.sqrt(dt)
    rho_perp = np.sqrt(max(1.0 - rho ** 2, 0.0))

    X = np.full(n_paths, mu, dtype=float)
    V_accum = np.zeros(n_paths)
    t = 0.0

    for _ in range(n_steps):
        Z1 = rng.standard_normal(n_paths)
        X = X - theta * (X - mu) * dt + sigma * sqrt_dt * Z1
        t += dt
        if t > T_start:
            V_accum += np.exp(X) * dt

    sqrtV = np.sqrt(np.maximum(V_accum, 0.0))
    price = float(np.mean(sqrtV))
    se = float(np.std(sqrtV) / np.sqrt(n_paths))
    return price, se
