"""
Experiment 9: Rough Bergomi exotic option pricing.

Model:
    v_t = v0 * exp(eta * W^H_t - 0.5*eta^2 * t^{2H})
where W^H is a fractional Brownian motion with Hurst index H.

Parameters: v0=0.04, eta=1.8, rho=-0.7, T=1

Priced instruments for H in {0.1, 0.3}:
  1. One-touch barrier      (B/S0 = 1.06)
  2. Reverse knock-out call (K = 100, B/S0 = 1.10)
  3. Forward-vol agreement  (FVA: E[sqrt(int_{0.5}^{1.0} v_t dt)])
  4. Forward-start smile    (reset t1=0.5, horizon T=1,
                             K/S_t1 in linspace(0.8,1.2,9))

fBm covariance built from the exact covariance kernel; spot simulated with
the correlating BM increment extracted from the Cholesky factor.

Saves results/exp9_rbergomi_exotics.csv and results/exp9_rbergomi_exotics.png.
"""

import sys
import os

sys.path.insert(0, '/workspace/project')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

os.makedirs('/workspace/project/results', exist_ok=True)


# ---------------------------------------------------------------------------
# Rough Bergomi path simulator
# ---------------------------------------------------------------------------

def simulate_rough_bergomi_paths(v0, eta, rho, H, T, n_paths, N, seed=42):
    """
    Simulate rough Bergomi spot and variance paths.

    The fBm W^H is represented on the grid t_i = i*T/N, i=1..N, using its
    exact Gaussian covariance matrix and a Cholesky factorisation.

    Parameters
    ----------
    v0, eta, rho, H : model parameters
    T               : maturity
    n_paths, N      : MC paths, time steps
    seed            : RNG seed

    Returns
    -------
    logS_hist : (N+1, n_paths)  log-spot (logS[0] = 0)
    v_paths   : (N, n_paths)   instantaneous variance at each t_i
    t_grid    : (N,)            time grid t_i = i*T/N
    """
    t_grid = np.linspace(T / N, T, N)              # t_1 .. t_N

    # Exact fBm covariance: Cov(W^H_s, W^H_t) = 0.5*(s^{2H}+t^{2H}-|t-s|^{2H})
    K_cov = 0.5 * (
        t_grid[:, None] ** (2 * H)
        + t_grid[None, :] ** (2 * H)
        - np.abs(t_grid[:, None] - t_grid[None, :]) ** (2 * H)
    )
    K_cov += 1e-10 * np.eye(N)                     # numerical regularisation
    L = np.linalg.cholesky(K_cov)                  # lower-triangular factor

    rng = np.random.default_rng(seed)
    dt = T / N

    # Draw i.i.d. N(0,1) and map to correlated fBm realisations
    Z_vol = rng.standard_normal((N, n_paths))
    Y = L @ Z_vol                                   # (N, n_paths): W^H values

    # log-variance and instantaneous variance
    log_v = eta * Y - 0.5 * eta ** 2 * t_grid[:, None] ** (2 * H)
    v_paths = v0 * np.exp(log_v)                   # (N, n_paths)

    # Spot dynamics: dlogS = -0.5*v dt + rho*sqrt(v)*dW^H + sqrt(1-rho^2)*sqrt(v)*dW_perp
    Z_perp = rng.standard_normal((N, n_paths))
    dW_v = np.diff(np.vstack([np.zeros((1, n_paths)), Y]), axis=0)  # (N, n_paths)

    logS = np.zeros(n_paths)
    logS_hist = [logS.copy()]
    rho_perp = np.sqrt(max(1.0 - rho ** 2, 0.0))
    for i in range(N):
        sqrt_v = np.sqrt(v_paths[i])
        dlogS = (
            -0.5 * v_paths[i] * dt
            + rho * sqrt_v * dW_v[i]
            + rho_perp * sqrt_v * np.sqrt(dt) * Z_perp[i]
        )
        logS = logS + dlogS
        logS_hist.append(logS.copy())

    return np.array(logS_hist), v_paths, t_grid


# ---------------------------------------------------------------------------
# Black-76 scalar helpers
# ---------------------------------------------------------------------------

def _black_call_scalar(F, K, vol, T):
    if vol <= 0 or T <= 0 or K <= 0:
        return max(float(F - K), 0.0)
    sqrtT = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * vol ** 2 * T) / (vol * sqrtT)
    return float(F * norm.cdf(d1) - K * norm.cdf(d1 - vol * sqrtT))


def _bs_impvol_scalar(price, F, K, T, tol=1e-8, max_iter=150):
    """Newton-Raphson implied-vol solver (scalar)."""
    intrinsic = max(float(F - K), 0.0)
    if price <= intrinsic + 1e-14 or T <= 0:
        return np.nan
    sigma = np.sqrt(2.0 * abs(np.log(F / K)) / T) if F != K else 0.5
    sigma = max(sigma, 0.01)
    for _ in range(max_iter):
        p = _black_call_scalar(F, K, sigma, T)
        sqrtT = np.sqrt(T)
        d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrtT)
        vega = F * norm.pdf(d1) * sqrtT
        if vega < 1e-14:
            break
        sigma -= (p - price) / vega
        sigma = max(sigma, 1e-6)
        if abs(p - price) < tol:
            break
    return sigma if sigma > 1e-6 else np.nan


# ---------------------------------------------------------------------------
# Exotic pricers (path-based)
# ---------------------------------------------------------------------------

def price_one_touch(logS_hist, F0, barrier_ratio):
    """Upside one-touch: pays 1 if max S_t >= B."""
    B = F0 * barrier_ratio
    S_hist = F0 * np.exp(logS_hist)
    touched = np.any(S_hist >= B, axis=0)
    price = float(np.mean(touched))
    se = float(np.std(touched.astype(float)) / np.sqrt(touched.shape[0]))
    return price, se


def price_reverse_knockout(logS_hist, F0, K, barrier_ratio):
    """Reverse knock-out call: alive if S_t < B throughout."""
    B = F0 * barrier_ratio
    S_hist = F0 * np.exp(logS_hist)
    alive = np.all(S_hist < B, axis=0)
    payoff = alive * np.maximum(S_hist[-1] - K, 0.0)
    price = float(np.mean(payoff))
    se = float(np.std(payoff) / np.sqrt(payoff.shape[0]))
    return price, se


def price_fva(v_paths, t_grid, T_start, Delta):
    """FVA: E[sqrt(int_{T_start}^{T_start+Delta} v_t dt)]."""
    dt = t_grid[1] - t_grid[0] if len(t_grid) > 1 else t_grid[0]
    mask = (t_grid > T_start) & (t_grid <= T_start + Delta)
    V_accum = np.sum(v_paths[mask, :], axis=0) * dt
    sqrtV = np.sqrt(np.maximum(V_accum, 0.0))
    return float(np.mean(sqrtV)), float(np.std(sqrtV) / np.sqrt(sqrtV.shape[0]))


# ---------------------------------------------------------------------------
# Forward-start smile
# ---------------------------------------------------------------------------

def compute_forward_start_smile(v0, eta, rho, H, T1, Delta, F0,
                                K_ratios, n_paths, N_total, seed=42):
    """
    Forward-start implied-vol smile for rough Bergomi.

    Simulates a single path ensemble of length T1+Delta.
    At reset t1=T1, the option pays max(S_T/S_{t1} - K_ratio, 0) per unit
    of S_{t1}, so we compute the normalised call on the return ratio.
    Implied vol extracted via Black(F=1, K=K_ratio, IV, Delta).

    Returns implied vols (array, same length as K_ratios).
    """
    T_full = T1 + Delta
    logS_hist, _, _ = simulate_rough_bergomi_paths(
        v0, eta, rho, H, T_full, n_paths, N_total, seed=seed
    )
    # Locate the reset index corresponding to t1=T1
    N1 = int(round(N_total * T1 / T_full))
    N1 = max(1, min(N1, N_total - 1))

    # Normalised return over [T1, T1+Delta]
    log_ratio = logS_hist[-1] - logS_hist[N1]   # log(S_T / S_{t1})
    ratio = np.exp(log_ratio)                    # (n_paths,)

    # Use the empirical forward to absorb any discretisation drift
    F_empirical = float(np.mean(ratio))

    ivs = []
    for K_ratio in K_ratios:
        payoff = np.maximum(ratio - K_ratio, 0.0)
        price = float(np.mean(payoff))
        # Black(F_empirical, K_ratio, IV, Delta) → IV
        iv = _bs_impvol_scalar(price, F_empirical, K_ratio, Delta)
        ivs.append(iv)

    return np.array(ivs)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

V0    = 0.04
ETA   = 1.8
RHO   = -0.7
T     = 1.0
F0    = 100.0
N     = 252          # daily steps (as per AGENTS.md N=200, we use 252 for calendar)
N_PATHS = 10000

BARRIER_RATIO_OT  = 1.06
K_RKO             = 100.0
BARRIER_RATIO_RKO = 1.10

T_START_FVA = 0.5
DELTA_FVA   = 0.5

H_VALUES     = [0.1, 0.3]
K_RATIOS_FS  = np.linspace(0.8, 1.2, 9)

# ---------------------------------------------------------------------------
# Main loop over H
# ---------------------------------------------------------------------------

records    = []
fs_results = {}
exotic_results = {H: {} for H in H_VALUES}

print("=== EXP 9 — Rough Bergomi exotic option pricing ===")
print(f"v0={V0}, eta={ETA}, rho={RHO}, T={T}, N={N}, n_paths={N_PATHS}\n")

for H in H_VALUES:
    print(f"--- H = {H} ---")

    logS_hist, v_paths, t_grid = simulate_rough_bergomi_paths(
        V0, ETA, RHO, H, T, N_PATHS, N, seed=42
    )

    ot_p,  ot_se  = price_one_touch(logS_hist, F0, BARRIER_RATIO_OT)
    rko_p, rko_se = price_reverse_knockout(logS_hist, F0, K_RKO, BARRIER_RATIO_RKO)
    fva_p, fva_se = price_fva(v_paths, t_grid, T_START_FVA, DELTA_FVA)

    print(f"  One-Touch (B/S0={BARRIER_RATIO_OT}):             {ot_p:.4f} ± {ot_se:.4f}")
    print(f"  RKO Call  (K={K_RKO}, B/S0={BARRIER_RATIO_RKO}): {rko_p:.4f} ± {rko_se:.4f}")
    print(f"  FVA E[√V]:                                 {fva_p:.4f} ± {fva_se:.4f}")

    exotic_results[H] = dict(ot_p=ot_p, ot_se=ot_se,
                              rko_p=rko_p, rko_se=rko_se,
                              fva_p=fva_p, fva_se=fva_se)

    for prod, p, se in [('one_touch', ot_p, ot_se),
                        ('rko_call',  rko_p, rko_se),
                        ('fva',       fva_p, fva_se)]:
        records.append(dict(H=H, product=prod, price=p, se=se))

    # Forward-start smile
    ivs = compute_forward_start_smile(
        V0, ETA, RHO, H, T_START_FVA, DELTA_FVA, F0,
        K_RATIOS_FS, N_PATHS, N_total=N, seed=43,
    )
    fs_results[H] = ivs
    print(f"  Forward-start smile IVs (%):")
    for kr, iv in zip(K_RATIOS_FS, ivs):
        iv_str = f"{iv*100:.2f}" if np.isfinite(iv) else "NaN"
        print(f"    K/S_t1={kr:.2f}  IV={iv_str}%")
        records.append(dict(H=H,
                            product=f'fs_smile_K{kr:.2f}',
                            price=iv if np.isfinite(iv) else np.nan,
                            se=np.nan))
    print()

# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------

df = pd.DataFrame(records)
df.to_csv('/workspace/project/results/exp9_rbergomi_exotics.csv',
          index=False, float_format='%.6f')
print("Saved: results/exp9_rbergomi_exotics.csv")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    'EXP 9: Rough Bergomi Exotic Pricing\n'
    fr'$v_0={V0}$,  $\eta={ETA}$,  $\rho={RHO}$,  $T={T}$,  n_paths={N_PATHS}',
    fontsize=11,
)

# Left panel: bar chart of three exotic prices per H
ax = axes[0]
products = ['one_touch', 'rko_call', 'fva']
bar_labels = [
    f'One-Touch\n(B/S₀={BARRIER_RATIO_OT})',
    f'RKO Call\n(K={K_RKO}, B/S₀={BARRIER_RATIO_RKO})',
    'FVA\nE[√V]',
]
x = np.arange(len(products))
width = 0.3
colors = ['steelblue', 'coral']

for i, H in enumerate(H_VALUES):
    res = exotic_results[H]
    prices_H = [res['ot_p'],  res['rko_p'],  res['fva_p']]
    ses_H    = [res['ot_se'], res['rko_se'], res['fva_se']]
    offset = (i - 0.5) * width
    ax.bar(x + offset, prices_H, width,
           label=f'H={H}', color=colors[i], alpha=0.85,
           yerr=[2 * s for s in ses_H], capsize=4)

ax.set_xticks(x)
ax.set_xticklabels(bar_labels, fontsize=9)
ax.set_ylabel('Price / Probability')
ax.set_title('Exotic Prices by Hurst exponent H')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.35)

# Right panel: forward-start smile
ax2 = axes[1]
ls_cycle = ['-o', '-s']
for i, H in enumerate(H_VALUES):
    ivs = fs_results[H]
    mask = np.isfinite(ivs)
    ax2.plot(K_RATIOS_FS[mask], ivs[mask] * 100, ls_cycle[i],
             lw=2, ms=6, label=f'H={H}')
ax2.set_xlabel(r'Strike ratio $K/S_0$ (forward-start at $t_1=0.5$)', fontsize=10)
ax2.set_ylabel('Implied Vol (%)')
ax2.set_title(fr'Forward-Start Smile  ($t_1={T_START_FVA}$, $\Delta={DELTA_FVA}$)')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.35)

plt.tight_layout()
plt.savefig('/workspace/project/results/exp9_rbergomi_exotics.png',
            dpi=120, bbox_inches='tight')
plt.close()
print("Saved: results/exp9_rbergomi_exotics.png")
