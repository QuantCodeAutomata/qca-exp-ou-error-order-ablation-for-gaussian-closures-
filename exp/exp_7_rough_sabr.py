"""
Experiment 7: Rough SABR — RQMC vs Monte Carlo across a parameter grid.

Grid:
  beta  in {0.3, 0.5, 0.7}
  eta   in {0.5, 1.0}
  T     in {0.5, 1.0}
  rho   in {-0.5, 0.0}
  H     in {0.1, 0.3}

= 3 x 2 x 2 x 2 x 2 = 48 cells.

For each cell:
  - Benchmark MC   : rough_sabr_mc  (4096 paths, N=200)
  - RQMC method    : rough_sabr_rqmc (1024 paths, 4 scrambles, N=30)

Strikes: 11 equally spaced from 80 to 120,  F0=100, alpha0=0.2*F0^{1-beta}.

Reports per-cell RMSE in basis points (IV RMSE vs MC benchmark).
Saves results/exp7_rough_sabr.csv and results/exp7_rough_sabr.png.
"""

import sys
import os

sys.path.insert(0, '/workspace/project')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools

from src.rough_sabr import rough_sabr_rqmc, rough_sabr_mc
from src.absorbed_cev import bs_impvol
from src.metrics import rmse_bp

os.makedirs('/workspace/project/results', exist_ok=True)

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

BETAS = [0.3, 0.5, 0.7]
ETAS  = [0.5, 1.0]
TS    = [0.5, 1.0]
RHOS  = [-0.5, 0.0]
HS    = [0.1, 0.3]

F0 = 100.0
N_STRIKES = 11
K_arr = np.linspace(80.0, 120.0, N_STRIKES)

N_MC   = 200    # Euler steps for MC benchmark
N_RQMC = 30     # time steps for RQMC
N_PATHS_BENCH = 4096
N_PATHS_RQMC  = 1024
N_SCRAMBLES   = 4

print("=== EXP 7 — Rough SABR: RQMC vs MC benchmark ===\n")
print(f"Grid: {len(BETAS)} beta × {len(ETAS)} eta × {len(TS)} T × {len(RHOS)} rho × {len(HS)} H"
      f" = {len(BETAS)*len(ETAS)*len(TS)*len(RHOS)*len(HS)} cells\n")

records = []
cell_idx = 0

for beta, eta, T, rho, H in itertools.product(BETAS, ETAS, TS, RHOS, HS):
    cell_idx += 1
    alpha0 = 0.2 * F0 ** (1.0 - beta)   # normalise so ATM vol ≈ 20%

    # --- MC benchmark ---
    bench_prices = rough_sabr_mc(
        alpha0=alpha0, eta=eta, rho=rho, H=H, beta=beta, T=T,
        strikes=K_arr, F0=F0, N=N_MC, n_paths=N_PATHS_BENCH, seed=42
    )

    # --- RQMC ---
    rqmc_prices = rough_sabr_rqmc(
        alpha0=alpha0, eta=eta, rho=rho, H=H, beta=beta, T=T,
        strikes=K_arr, F0=F0, N=N_RQMC, n_paths=N_PATHS_RQMC, n_scrambles=N_SCRAMBLES
    )

    # --- Implied vols ---
    bench_ivs = bs_impvol(bench_prices, F0, K_arr, T)
    rqmc_ivs  = bs_impvol(rqmc_prices,  F0, K_arr, T)

    err = rmse_bp(rqmc_ivs, bench_ivs)

    if cell_idx % 8 == 0 or cell_idx <= 3:
        print(f"  [{cell_idx:3d}] beta={beta}, eta={eta}, T={T}, rho={rho}, H={H}: "
              f"RMSE={err:.1f}bp")

    records.append(dict(
        cell=cell_idx,
        beta=beta, eta=eta, T=T, rho=rho, H=H,
        alpha0=alpha0,
        rmse_rqmc_bp=err,
    ))

print(f"\nTotal cells processed: {cell_idx}")

# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------

df = pd.DataFrame(records)
df.to_csv('/workspace/project/results/exp7_rough_sabr.csv', index=False, float_format='%.4f')
print("Saved: results/exp7_rough_sabr.csv")

# Summary statistics
print("\n--- Summary statistics ---")
print(f"Mean RMSE : {df['rmse_rqmc_bp'].mean():.2f} bp")
print(f"Median    : {df['rmse_rqmc_bp'].median():.2f} bp")
print(f"Max       : {df['rmse_rqmc_bp'].max():.2f} bp")
print(f"Min       : {df['rmse_rqmc_bp'].min():.2f} bp")

# ---------------------------------------------------------------------------
# Plot: RMSE heatmap across (beta, H) for each (eta, T, rho) combination
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('EXP 7: Rough SABR RQMC vs MC — RMSE (bp) by parameter', fontsize=12)
axes = axes.ravel()

ax_idx = 0
for rho in RHOS:
    for eta in ETAS:
        sub = df[(df['rho'] == rho) & (df['eta'] == eta)]
        mat = np.zeros((len(BETAS), len(HS)))
        for bi, beta in enumerate(BETAS):
            for hi, H in enumerate(HS):
                cell = sub[(sub['beta'] == beta) & (sub['H'] == H)]['rmse_rqmc_bp']
                mat[bi, hi] = cell.mean() if len(cell) > 0 else np.nan

        ax = axes[ax_idx]
        im = ax.imshow(mat, aspect='auto', cmap='YlOrRd')
        ax.set_xticks(range(len(HS)))
        ax.set_xticklabels([f'H={h}' for h in HS])
        ax.set_yticks(range(len(BETAS)))
        ax.set_yticklabels([f'β={b}' for b in BETAS])
        ax.set_title(f'ρ={rho}, η={eta}', fontsize=9)
        for bi in range(len(BETAS)):
            for hi in range(len(HS)):
                ax.text(hi, bi, f'{mat[bi, hi]:.0f}', ha='center', va='center',
                        fontsize=8, color='black')
        plt.colorbar(im, ax=ax, label='RMSE (bp)')
        ax_idx += 1

# Fill unused axes
for ax in axes[ax_idx:]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig('/workspace/project/results/exp7_rough_sabr.png', dpi=120, bbox_inches='tight')
plt.close()
print("Saved: results/exp7_rough_sabr.png")
print("\nexp_7 done")
