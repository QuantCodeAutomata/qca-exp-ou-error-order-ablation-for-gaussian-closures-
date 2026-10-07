"""
Experiment 3: SABR vanilla pricing — Hagan formula vs routed stack vs MC benchmark.

Grid: beta in {0.3, 0.5, 0.7} x T in {2, 5, 10} x rho in {-0.5, 0.0, 0.5}
      = 27 training cells.
Held-out: beta in {0.2, 0.6} x rho in {-0.4, 0.3} x T in {3, 7} = 8 cells.

For each cell:
  - MC benchmark (n_paths=16384, 512 steps/yr)
  - Hagan (2002) asymptotic implied vol
  - Routed stack (32-point GH over alpha_T, absorbed-CEV inner pricer)

Reports RMSE in bp (vs MC) for Hagan and stack across 15 strikes [70, 130].
Saves results/exp3_sabr_results.csv and results/exp3_sabr_plot.png.
"""

import sys
import os

sys.path.insert(0, '/workspace/project')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.absorbed_cev import bs_impvol
from src.sabr_stack import hagan_impvol, sabr_stack_impvol, sabr_mc_impvol

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

F0 = 100.0
nu = 0.4
N_STRIKES = 15
K_arr = np.linspace(70.0, 130.0, N_STRIKES)

BETAS_TRAIN = [0.3, 0.5, 0.7]
TS_TRAIN = [2.0, 5.0, 10.0]
RHOS_TRAIN = [-0.5, 0.0, 0.5]

BETAS_HELD = [0.2, 0.6]
TS_HELD = [3.0, 7.0]
RHOS_HELD = [-0.4, 0.3]

os.makedirs('/workspace/project/results', exist_ok=True)


def run_cell(beta, T, rho, seed=42):
    """Compute MC / Hagan / Stack IVs for one (beta, T, rho) cell."""
    alpha0 = 0.2 * F0 ** (1.0 - beta)   # normalise so ATM IV ≈ 20%

    # MC benchmark
    mc_iv = sabr_mc_impvol(K_arr, F0, T, alpha0, beta, rho, nu,
                           n_paths=16384, n_steps_per_year=512, seed=seed)

    # Hagan asymptotic
    hagan_iv = hagan_impvol(K_arr, F0, T, alpha0, beta, rho, nu)

    # Routed stack
    stack_iv = sabr_stack_impvol(K_arr, F0, T, alpha0, beta, rho, nu, n_gh=32)

    # RMSE in bp (bp = 1e-4), ignoring NaN strikes
    def rmse_bp(iv_approx, iv_mc):
        diff = (iv_approx - iv_mc) * 1e4
        mask = np.isfinite(diff)
        return float(np.sqrt((diff[mask] ** 2).mean())) if mask.any() else np.nan

    hagan_rmse = rmse_bp(hagan_iv, mc_iv)
    stack_rmse = rmse_bp(stack_iv, mc_iv)
    return mc_iv, hagan_iv, stack_iv, hagan_rmse, stack_rmse


# ---------------------------------------------------------------------------
# Training grid
# ---------------------------------------------------------------------------

records = []
all_ivs_train = {}

print("=== EXP 3 — SABR: training grid (27 cells) ===")
for beta in BETAS_TRAIN:
    for T in TS_TRAIN:
        for rho in RHOS_TRAIN:
            mc_iv, hagan_iv, stack_iv, h_rmse, s_rmse = run_cell(beta, T, rho)
            rec = dict(split='train', beta=beta, T=T, rho=rho,
                       hagan_rmse_bp=h_rmse, stack_rmse_bp=s_rmse)
            records.append(rec)
            all_ivs_train[(beta, T, rho)] = (mc_iv, hagan_iv, stack_iv)
            print(f"  beta={beta:.1f} T={T:4.1f} rho={rho:+.1f}  "
                  f"Hagan RMSE={h_rmse:6.1f}bp  Stack RMSE={s_rmse:6.1f}bp")

# ---------------------------------------------------------------------------
# Held-out cells
# ---------------------------------------------------------------------------

print("\n=== Held-out (8 cells) ===")
all_ivs_held = {}
for beta in BETAS_HELD:
    for T in TS_HELD:
        for rho in RHOS_HELD:
            mc_iv, hagan_iv, stack_iv, h_rmse, s_rmse = run_cell(beta, T, rho, seed=99)
            rec = dict(split='held', beta=beta, T=T, rho=rho,
                       hagan_rmse_bp=h_rmse, stack_rmse_bp=s_rmse)
            records.append(rec)
            all_ivs_held[(beta, T, rho)] = (mc_iv, hagan_iv, stack_iv)
            print(f"  beta={beta:.1f} T={T:4.1f} rho={rho:+.1f}  "
                  f"Hagan RMSE={h_rmse:6.1f}bp  Stack RMSE={s_rmse:6.1f}bp")

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

df = pd.DataFrame(records)
train_df = df[df['split'] == 'train']
held_df = df[df['split'] == 'held']

print(f"\n=== Summary ===")
print(f"Train — Hagan: mean={train_df['hagan_rmse_bp'].mean():.1f}bp  "
      f"max={train_df['hagan_rmse_bp'].max():.1f}bp")
print(f"Train — Stack: mean={train_df['stack_rmse_bp'].mean():.1f}bp  "
      f"max={train_df['stack_rmse_bp'].max():.1f}bp")
print(f"Held  — Hagan: mean={held_df['hagan_rmse_bp'].mean():.1f}bp  "
      f"max={held_df['hagan_rmse_bp'].max():.1f}bp")
print(f"Held  — Stack: mean={held_df['stack_rmse_bp'].mean():.1f}bp  "
      f"max={held_df['stack_rmse_bp'].max():.1f}bp")

# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------

df.to_csv('/workspace/project/results/exp3_sabr_results.csv', index=False)
print("\nSaved: results/exp3_sabr_results.csv")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharey=False)
fig.suptitle('EXP 3: SABR IV Smiles — MC vs Hagan vs Routed Stack\n'
             r'$\nu=0.4$, $F_0=100$, $\alpha_0=0.2 \cdot F_0^{1-\beta}$',
             fontsize=12)

for row_idx, beta in enumerate(BETAS_TRAIN):
    for col_idx, T in enumerate(TS_TRAIN):
        ax = axes[row_idx, col_idx]
        rho = 0.0  # plot the rho=0 slice
        mc_iv, hagan_iv, stack_iv = all_ivs_train[(beta, T, rho)]
        moneyness = K_arr / F0
        ax.plot(moneyness, mc_iv * 100, 'k.', ms=6, label='MC', zorder=5)
        ax.plot(moneyness, hagan_iv * 100, 'b--', lw=1.5, label='Hagan')
        ax.plot(moneyness, stack_iv * 100, 'r-', lw=1.5, label='Stack')
        ax.set_title(fr'$\beta={beta}$, $T={int(T)}$, $\rho=0$', fontsize=9)
        ax.set_xlabel('K/F', fontsize=8)
        ax.set_ylabel('IV (%)', fontsize=8)
        ax.tick_params(labelsize=7)
        if row_idx == 0 and col_idx == 0:
            ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig('/workspace/project/results/exp3_sabr_plot.png', dpi=120, bbox_inches='tight')
plt.close()
print("Saved: results/exp3_sabr_plot.png")

# ---------------------------------------------------------------------------
# Second plot: RMSE bar chart by method
# ---------------------------------------------------------------------------

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
fig2.suptitle('EXP 3 — RMSE (bp) vs MC benchmark', fontsize=12)

for ax, split, df_s in zip(axes2, ['train', 'held'], [train_df, held_df]):
    x = np.arange(len(df_s))
    width = 0.35
    ax.bar(x - width / 2, df_s['hagan_rmse_bp'].values, width, label='Hagan', color='steelblue')
    ax.bar(x + width / 2, df_s['stack_rmse_bp'].values, width, label='Stack', color='coral')
    labels = [f"β={r['beta']},T={r['T']},ρ={r['rho']}" for _, r in df_s.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha='right', fontsize=6)
    ax.set_ylabel('RMSE (bp)')
    ax.set_title(f'{split.capitalize()} set ({len(df_s)} cells)')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.4)

plt.tight_layout()
fig2.savefig('/workspace/project/results/exp3_sabr_rmse.png', dpi=120, bbox_inches='tight')
plt.close()
print("Saved: results/exp3_sabr_rmse.png")
