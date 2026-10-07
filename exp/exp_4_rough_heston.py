"""
Experiment 4: Rough Heston — Riccati discretisation error ablation.

Compares N=50 vs N=200 time-step fractional Riccati solutions across maturities.

Parameters:
  kappa=1, theta=0.04, lambda=0.5, H=0.1, v0=0.04, rho_hv=0.0
  T in {0.5, 1.0, 2.0}
  K/F in linspace(0.8, 1.2, 11)

Metric: absolute IV error (N=50 vs N=200 reference) in basis points.
Saves results/exp4_rough_heston.csv and results/exp4_rough_heston.png.
"""

import sys
import os

sys.path.insert(0, '/workspace/project')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.rough_heston import rough_heston_call_cos, rough_heston_log_cf, bs_impvol

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

kappa = 1.0
theta = 0.04
lam = 0.5
H = 0.1
v0 = 0.04
rho_hv = 0.0
F0 = 100.0

T_arr = [0.5, 1.0, 2.0]
K_arr = F0 * np.linspace(0.8, 1.2, 11)
moneyness = K_arr / F0

N_COARSE = 50
N_REF = 200
N_COS = 64   # keeps u_max ~42, stable for both N_riccati values

os.makedirs('/workspace/project/results', exist_ok=True)

# ---------------------------------------------------------------------------
# Compute IV for each maturity
# ---------------------------------------------------------------------------

records = []

print("=== EXP 4 — Rough Heston: Riccati discretisation error ===")
print(f"Parameters: kappa={kappa}, theta={theta}, lam={lam}, H={H}, v0={v0}, rho_hv={rho_hv}")
print(f"N_coarse={N_COARSE}, N_ref={N_REF}, N_cos={N_COS}\n")

all_results = {}

for T in T_arr:
    p_coarse = rough_heston_call_cos(
        K_arr, F0, T, kappa, theta, lam, H, v0,
        rho_hv=rho_hv, N_riccati=N_COARSE, N_cos=N_COS
    )
    p_ref = rough_heston_call_cos(
        K_arr, F0, T, kappa, theta, lam, H, v0,
        rho_hv=rho_hv, N_riccati=N_REF, N_cos=N_COS
    )

    iv_coarse = bs_impvol(p_coarse, F0, K_arr, T)
    iv_ref = bs_impvol(p_ref, F0, K_arr, T)
    err_bp = (iv_coarse - iv_ref) * 1e4

    rmse = float(np.sqrt(np.nanmean(err_bp ** 2)))
    mae = float(np.nanmean(np.abs(err_bp)))
    atm_err = float(err_bp[5]) if np.isfinite(err_bp[5]) else np.nan

    print(f"T={T:.1f}: RMSE={rmse:.1f}bp  MAE={mae:.1f}bp  ATM_err={atm_err:.1f}bp")
    print(f"  IV_ref (%): {np.round(iv_ref*100, 2)}")
    print(f"  err (bp):   {np.round(err_bp, 1)}\n")

    for i, K in enumerate(K_arr):
        records.append(dict(
            T=T,
            K=K,
            KF=K / F0,
            price_N50=float(p_coarse[i]),
            price_N200=float(p_ref[i]),
            iv_N50=float(iv_coarse[i]) if np.isfinite(iv_coarse[i]) else np.nan,
            iv_N200=float(iv_ref[i]) if np.isfinite(iv_ref[i]) else np.nan,
            err_bp=float(err_bp[i]) if np.isfinite(err_bp[i]) else np.nan,
        ))

    all_results[T] = (iv_coarse, iv_ref, err_bp)

# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------

df = pd.DataFrame(records)
df.to_csv('/workspace/project/results/exp4_rough_heston.csv', index=False, float_format='%.6f')
print("Saved: results/exp4_rough_heston.csv")

# ---------------------------------------------------------------------------
# Plot: IV smiles + error per maturity
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
fig.suptitle(
    'EXP 4: Rough Heston — Riccati Discretisation Error\n'
    fr'$\kappa={kappa}$, $\theta={theta}$, $\lambda={lam}$, '
    fr'$H={H}$, $v_0={v0}$, $\rho_{{hv}}={rho_hv}$',
    fontsize=11
)

for col, T in enumerate(T_arr):
    iv_coarse, iv_ref, err_bp = all_results[T]

    # Top row: IV smiles
    ax_iv = axes[0, col]
    ax_iv.plot(moneyness, iv_ref * 100, 'k-', lw=2, label=f'N={N_REF} (ref)')
    ax_iv.plot(moneyness, iv_coarse * 100, 'r--', lw=1.5, label=f'N={N_COARSE}')
    ax_iv.set_title(f'T = {T}', fontsize=10)
    ax_iv.set_xlabel('K/F', fontsize=9)
    ax_iv.set_ylabel('IV (%)', fontsize=9)
    ax_iv.legend(fontsize=8)
    ax_iv.grid(alpha=0.3)

    # Bottom row: error in bp
    ax_err = axes[1, col]
    ax_err.bar(moneyness, err_bp, width=0.035, color='steelblue', alpha=0.8)
    ax_err.axhline(0, color='k', lw=0.8)
    ax_err.set_xlabel('K/F', fontsize=9)
    ax_err.set_ylabel('IV error (bp)', fontsize=9)
    ax_err.set_title(f'T={T}: RMSE={np.sqrt(np.nanmean(err_bp**2)):.1f}bp', fontsize=10)
    ax_err.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/project/results/exp4_rough_heston.png', dpi=120, bbox_inches='tight')
plt.close()
print("Saved: results/exp4_rough_heston.png")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print("\n=== Summary ===")
print(f"{'T':>5}  {'RMSE(bp)':>10}  {'MAE(bp)':>9}  {'ATM err(bp)':>12}")
for T in T_arr:
    _, iv_ref, err_bp = all_results[T]
    rmse = np.sqrt(np.nanmean(err_bp ** 2))
    mae = np.nanmean(np.abs(err_bp))
    atm = err_bp[5] if np.isfinite(err_bp[5]) else np.nan
    print(f"{T:>5.1f}  {rmse:>10.1f}  {mae:>9.1f}  {atm:>12.1f}")
