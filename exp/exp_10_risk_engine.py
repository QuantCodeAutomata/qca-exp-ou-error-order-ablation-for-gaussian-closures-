"""
Experiment 10: Bucketed-vega risk engine — bump-and-revalue vs response propagator.

For a vanilla call in the exp-OU stochastic-vol model, two methods compute
the bucketed vega dP/d(vol_k) across n_buckets = 10 equal time buckets:

  Method A — Bump-and-revalue:
    Each bucket's vol is scaled by (1+eps); the effective variance changes by
    eps * bar_v * dt_bucket / T; price is re-evaluated analytically via
    the Black-Scholes formula with updated sigma_eff.

  Method B — Response propagator:
    The pointwise sensitivity dP/dv_t = vega_BS / (2*sigma_eff*T) is contracted
    against the OU response kernel R(t,s) = exp(-theta*(t-s)).

Both methods give dP/d(delta_v) for a LOCAL bump.  The response-propagator
additionally demonstrates the "memory vega" interpretation, where a perturbation
in the log-vol OU field at time s_k propagates to all later times.

Exp-OU parameters: theta=1, mu=-3.76, sigma=1.8
Rough parameters (rough kernel):  H=0.1, eta=1.5

Saves results/exp10_risk_engine.csv and results/exp10_risk_engine.png.
"""

import sys
import os
import time

sys.path.insert(0, '/workspace/project')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from src.response_propagator import (
    ou_response_kernel,
    rough_response_kernel,
    compute_bucketed_vega_bump,
    compute_bucketed_vega_response,
)

os.makedirs('/workspace/project/results', exist_ok=True)


# ---------------------------------------------------------------------------
# Black-76 helpers
# ---------------------------------------------------------------------------

def black_call_and_vega(F, K, sigma, T):
    """Return (call price, Black vega = dP/dsigma) for scalar inputs."""
    if sigma <= 0 or T <= 0:
        return max(float(F - K), 0.0), 0.0
    sqrtT = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrtT)
    price = float(F * norm.cdf(d1) - K * norm.cdf(d1 - sqrtT * sigma))
    vega  = float(F * norm.pdf(d1) * sqrtT)
    return price, vega


# ---------------------------------------------------------------------------
# exp-OU model: deterministic expected-vol term structure
# ---------------------------------------------------------------------------

def expou_expected_v(t_array, theta, mu, sigma):
    """
    E[v_t] = exp(mu + sigma^2/(2*theta) * (1 - exp(-2*theta*t)))
    for X_0 = mu (stationary initialisation).
    """
    gamma_t = (sigma ** 2 / (2.0 * theta)) * (1.0 - np.exp(-2.0 * theta * np.asarray(t_array)))
    return np.exp(mu + gamma_t)


def effective_sigma(t_grid, v_t):
    """sigma_eff = sqrt(mean(v_t)) via trapezoidal integration over [0,T]."""
    T = t_grid[-1]
    int_v = np.trapz(v_t, t_grid)
    return float(np.sqrt(int_v / T))


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

THETA = 1.0
MU    = -3.76
SIGMA = 1.8
F0    = 100.0
K     = 100.0
T     = 1.0
N_BUCKETS = 10

# Rough kernel parameters
H_ROUGH   = 0.1
ETA_ROUGH = 1.5

# Fine time grid for integrals
N_FINE = 2000
t_grid = np.linspace(1e-6, T, N_FINE)          # avoid t=0 for rough kernel
dt_bucket = T / N_BUCKETS
bucket_edges = np.linspace(0.0, T, N_BUCKETS + 1)
bucket_times = 0.5 * (bucket_edges[:-1] + bucket_edges[1:])  # bucket centres

# ---------------------------------------------------------------------------
# Baseline: exp-OU expected-vol path → effective sigma → base price + vega
# ---------------------------------------------------------------------------

v_t = expou_expected_v(t_grid, THETA, MU, SIGMA)          # (N_FINE,)
sigma_eff = effective_sigma(t_grid, v_t)

p_base, vega_bs = black_call_and_vega(F0, K, sigma_eff, T)

print("=== EXP 10 — Bucketed-vega risk engine ===")
print(f"exp-OU: theta={THETA}, mu={MU}, sigma={SIGMA}")
print(f"Option: F0={F0}, K={K}, T={T}")
print(f"sigma_eff = {sigma_eff:.5f},  base price = {p_base:.5f},  vega_BS = {vega_bs:.5f}")
print(f"Buckets: {N_BUCKETS}  (width {dt_bucket:.2f})\n")

# ---------------------------------------------------------------------------
# Method A: bump-and-revalue
# ---------------------------------------------------------------------------

eps = 1e-4

def price_func_bav(vol_multipliers):
    """
    Price a Black-76 call after scaling the vol in each bucket.

    vol_multipliers : (n_buckets,) array — base = ones
    The effective variance after bucket-k bump by eps is:
        sigma_eff^2_new = sigma_eff^2 + eps * mean(v_t in bucket k) * dt_bucket / T
    """
    # Build modified v(t) by applying multipliers bucket-by-bucket
    v_modified = v_t.copy()
    for k, m in enumerate(vol_multipliers):
        lo, hi = bucket_edges[k], bucket_edges[k + 1]
        mask = (t_grid >= lo) & (t_grid < hi)
        v_modified[mask] *= m
    sigma_mod = effective_sigma(t_grid, v_modified)
    p, _ = black_call_and_vega(F0, K, sigma_mod, T)
    return p


print("--- Method A: bump-and-revalue ---")
vegas_bav, t_bav = compute_bucketed_vega_bump(price_func_bav, N_BUCKETS, T, eps=eps)
print(f"  Runtime: {t_bav*1e3:.2f} ms")
print(f"  Vegas:   {np.round(vegas_bav, 6)}")
print(f"  Sum:     {vegas_bav.sum():.6f}  (≈ total vega {vega_bs:.6f})\n")

# ---------------------------------------------------------------------------
# Method B: response-propagator (OU kernel) — LOCAL sensitivity
# ---------------------------------------------------------------------------

# For a local bump of v_t in [s_k, s_k+dt_bucket], the sensitivity is
#   dP/d(bump_k) = integral_{s_k}^{s_k+dt_bucket} (dP/dv_t) dt
# where dP/dv_t = vega_BS / (2 * sigma_eff * T)
#
# Using the OU *response* interpretation (perturbation to X_{s_k} propagates):
#   dP/dX_{s_k} = integral_0^T R_OU(t,s_k) * (dP/dv_t) * v_t dt

dp_dv_const = vega_bs / (2.0 * sigma_eff * T)    # constant for flat v model approx

def sensitivity_func_ou(t_arr):
    """dP/dv at each t — weighted by v_t to convert dv/dX = v response."""
    v_interp = np.interp(t_arr, t_grid, v_t)
    return dp_dv_const * v_interp                 # dP/dX_t contribution

def kernel_ou(t_arr, s_k):
    return ou_response_kernel(t_arr, s_k, THETA)

print("--- Method B: response propagator (OU kernel — OU-field perturbation) ---")
vegas_rp_ou, t_rp_ou = compute_bucketed_vega_response(
    sensitivity_func_ou, kernel_ou, t_grid, bucket_times
)
print(f"  Runtime: {t_rp_ou*1e3:.2f} ms")
print(f"  Vegas:   {np.round(vegas_rp_ou, 6)}")
print(f"  Sum:     {vegas_rp_ou.sum():.6f}\n")

# ---------------------------------------------------------------------------
# Method B variant: local sensitivity (no propagation through OU memory)
# Matches bump-and-revalue for MULTIPLICATIVE vol bumps: the sensitivity
# dP/d(eps_k) [multiplicative] = ∫_{bucket_k} (dP/dv_t) · v_t dt
# So sensitivity_func must include the v_t factor.
# ---------------------------------------------------------------------------

def sensitivity_func_local(t_arr):
    """
    Sensitivity for multiplicative vol bump: (dP/dv_t) * v_t at each t.
    This is what bump-and-revalue computes per unit epsilon.
    """
    v_interp = np.interp(t_arr, t_grid, v_t)
    return dp_dv_const * v_interp

def kernel_local(t_arr, s_k):
    """Box kernel: 1 inside the bucket containing s_k, else 0."""
    k_idx = int(s_k / dt_bucket)
    k_idx = min(k_idx, N_BUCKETS - 1)
    lo = bucket_edges[k_idx]
    hi = bucket_edges[k_idx + 1]
    return np.where((t_arr >= lo) & (t_arr < hi), 1.0, 0.0)

print("--- Method B (local box kernel — should match BAV) ---")
vegas_rp_local, t_rp_local = compute_bucketed_vega_response(
    sensitivity_func_local, kernel_local, t_grid, bucket_times
)
print(f"  Runtime: {t_rp_local*1e3:.2f} ms")
print(f"  Vegas:   {np.round(vegas_rp_local, 6)}")
print(f"  Sum:     {vegas_rp_local.sum():.6f}\n")

# ---------------------------------------------------------------------------
# Rough kernel variant
# ---------------------------------------------------------------------------

def sensitivity_func_rough(t_arr):
    """dP/dv_t for rough model — v_t approximated as flat v_stat."""
    v_stat = np.exp(MU + SIGMA ** 2 / (2.0 * THETA))
    return dp_dv_const * v_stat * np.ones_like(t_arr)

def kernel_rough(t_arr, s_k):
    return rough_response_kernel(t_arr, s_k, ETA_ROUGH, H_ROUGH)

print(f"--- Rough response kernel (H={H_ROUGH}, eta={ETA_ROUGH}) ---")
vegas_rp_rough, t_rp_rough = compute_bucketed_vega_response(
    sensitivity_func_rough, kernel_rough, t_grid, bucket_times
)
print(f"  Runtime: {t_rp_rough*1e3:.2f} ms")
print(f"  Vegas:   {np.round(vegas_rp_rough, 6)}")
print(f"  Sum:     {vegas_rp_rough.sum():.6f}\n")

# ---------------------------------------------------------------------------
# Runtime comparison for increasing n_buckets
# ---------------------------------------------------------------------------

bucket_sizes = [5, 10, 20, 50, 100]
runtimes_bav = []
runtimes_rp  = []

print("--- Runtime scaling with n_buckets ---")
for nb in bucket_sizes:
    bt_loc = 0.5 * (np.linspace(0, T, nb + 1)[:-1] + np.linspace(0, T, nb + 1)[1:])
    be_loc = np.linspace(0, T, nb + 1)

    def _pf(mults, _be=be_loc):
        v_mod = v_t.copy()
        for k, m in enumerate(mults):
            mask = (t_grid >= _be[k]) & (t_grid < _be[k+1])
            v_mod[mask] *= m
        s_eff = effective_sigma(t_grid, v_mod)
        p, _ = black_call_and_vega(F0, K, s_eff, T)
        return p

    _, t_b = compute_bucketed_vega_bump(_pf, nb, T, eps=eps)
    _, t_r = compute_bucketed_vega_response(
        sensitivity_func_ou, kernel_ou, t_grid, bt_loc
    )
    runtimes_bav.append(t_b * 1e3)
    runtimes_rp.append(t_r * 1e3)
    print(f"  n_buckets={nb:4d}:  BAV={t_b*1e3:.2f} ms  RP={t_r*1e3:.2f} ms  "
          f"speedup={t_b/max(t_r, 1e-9):.1f}×")

# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------

records = []
for k in range(N_BUCKETS):
    records.append(dict(
        bucket=k,
        bucket_centre=bucket_times[k],
        vega_bav=vegas_bav[k],
        vega_rp_ou=vegas_rp_ou[k],
        vega_rp_local=vegas_rp_local[k],
        vega_rp_rough=vegas_rp_rough[k],
    ))

df_v = pd.DataFrame(records)
df_v.to_csv('/workspace/project/results/exp10_risk_engine.csv',
            index=False, float_format='%.8f')

df_rt = pd.DataFrame(dict(n_buckets=bucket_sizes,
                          runtime_bav_ms=runtimes_bav,
                          runtime_rp_ms=runtimes_rp))
df_rt.to_csv('/workspace/project/results/exp10_risk_engine_runtime.csv',
             index=False, float_format='%.4f')
print("\nSaved: results/exp10_risk_engine.csv")
print("Saved: results/exp10_risk_engine_runtime.csv")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle(
    'EXP 10: Bucketed-Vega Risk Engine\n'
    fr'exp-OU: θ={THETA}, μ={MU}, σ={SIGMA};  '
    fr'rough kernel: H={H_ROUGH}, η={ETA_ROUGH}',
    fontsize=11,
)

# Panel 1: bucketed vegas comparison (BAV vs RP-local — should match)
ax = axes[0]
x = np.arange(N_BUCKETS)
width = 0.35
ax.bar(x - width/2, vegas_bav,       width, label='Bump-and-revalue',  color='steelblue', alpha=0.85)
ax.bar(x + width/2, vegas_rp_local,  width, label='RP (local kernel)',  color='coral',     alpha=0.85)
ax.set_xlabel('Bucket index k')
ax.set_ylabel('dP/d(vol_k)')
ax.set_title('Local vega: BAV vs RP-local kernel\n(should match)')
ax.set_xticks(x)
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.35)

# Panel 2: RP-OU vs RP-rough (memory vega comparison)
ax2 = axes[1]
ax2.plot(bucket_times, vegas_rp_ou,    'o-', lw=2, ms=6, color='steelblue', label='RP OU kernel')
ax2.plot(bucket_times, vegas_rp_rough, 's-', lw=2, ms=6, color='coral',     label='RP rough kernel')
ax2.set_xlabel('Bucket centre time s_k')
ax2.set_ylabel('dP/dX_{s_k}  (memory vega)')
ax2.set_title('Memory vega: OU vs Rough kernel')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.35)

# Panel 3: runtime scaling
ax3 = axes[2]
ax3.plot(bucket_sizes, runtimes_bav, 'o-', lw=2, ms=6, color='steelblue', label='Bump-and-revalue')
ax3.plot(bucket_sizes, runtimes_rp,  's-', lw=2, ms=6, color='coral',     label='Response propagator')
ax3.set_xlabel('Number of buckets')
ax3.set_ylabel('Runtime (ms)')
ax3.set_title('Runtime scaling')
ax3.legend(fontsize=9)
ax3.grid(alpha=0.35)

plt.tight_layout()
plt.savefig('/workspace/project/results/exp10_risk_engine.png',
            dpi=120, bbox_inches='tight')
plt.close()
print("Saved: results/exp10_risk_engine.png")
