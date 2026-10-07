"""
Experiment 5: Rough Bergomi — Gaussian closure vs RQMC across regimes.

Three volatility regimes are tested:
  Mild     : H=0.20, eta=1.0, rho=-0.40, v0=0.04, T=1.0
  Moderate : H=0.10, eta=1.5, rho=-0.70, v0=0.04, T=1.0
  Extreme  : H=0.07, eta=1.9, rho=-0.90, v0=0.04, T=2.0

For each regime:
  - Benchmark RQMC  (2048 paths × 8 scrambles)
  - Gaussian closure (log-normal on V, GH quadrature)
  - Light RQMC       (512 paths × 4 scrambles)

Reports RMSE in basis points and saves:
  results/exp5_rough_bergomi.csv
  results/exp5_rough_bergomi.png
"""

import sys
import os

sys.path.insert(0, '/workspace/project')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.metrics import implied_vol, rmse_bp
from src.rough_bergomi_rqmc import rough_bergomi_rqmc
from src.rough_bergomi_closure import rough_bergomi_gaussian_closure

os.makedirs('/workspace/project/results', exist_ok=True)

# ---------------------------------------------------------------------------
# Regimes
# ---------------------------------------------------------------------------

regimes = [
    dict(name='Mild',     H=0.20, eta=1.0, rho=-0.40, v0=0.04, T=1.0),
    dict(name='Moderate', H=0.10, eta=1.5, rho=-0.70, v0=0.04, T=1.0),
    dict(name='Extreme',  H=0.07, eta=1.9, rho=-0.90, v0=0.04, T=2.0),
]

F0 = 1.0
strikes_ratio = np.array([0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15])
strikes = F0 * strikes_ratio

print("=== EXP 5 — Rough Bergomi: Gaussian closure vs RQMC ===\n")

records = []
all_results = {}

for reg in regimes:
    name = reg['name']
    print(f"Processing {name}  (H={reg['H']}, eta={reg['eta']}, rho={reg['rho']}, T={reg['T']})...")
    p = {k: reg[k] for k in ['v0', 'eta', 'rho', 'H', 'T']}

    # Benchmark: high-fidelity RQMC
    bench_prices = rough_bergomi_rqmc(
        **p, strikes=strikes, F0=F0, N=50, n_paths=2048, n_scrambles=8
    )
    # Method 1: Gaussian closure
    closure_prices = rough_bergomi_gaussian_closure(
        **p, strikes=strikes, F0=F0, N=50
    )
    # Method 2: Light RQMC
    rqmc_prices = rough_bergomi_rqmc(
        **p, strikes=strikes, F0=F0, N=50, n_paths=512, n_scrambles=4
    )

    bench_ivs   = np.array([implied_vol(max(bp, 1e-10), F0, K, reg['T']) for bp, K in zip(bench_prices, strikes)])
    closure_ivs = np.array([implied_vol(max(cp, 1e-10), F0, K, reg['T']) for cp, K in zip(closure_prices, strikes)])
    rqmc_ivs    = np.array([implied_vol(max(rp, 1e-10), F0, K, reg['T']) for rp, K in zip(rqmc_prices, strikes)])

    rmse_closure = rmse_bp(closure_ivs, bench_ivs)
    rmse_rqmc    = rmse_bp(rqmc_ivs,   bench_ivs)

    print(f"  closure RMSE = {rmse_closure:.2f} bp")
    print(f"  rqmc RMSE   = {rmse_rqmc:.2f} bp")

    records.append(dict(
        regime=name, H=reg['H'], eta=reg['eta'], rho=reg['rho'], T=reg['T'],
        rmse_closure_bp=rmse_closure, rmse_rqmc_bp=rmse_rqmc,
    ))
    all_results[name] = dict(
        bench_ivs=bench_ivs, closure_ivs=closure_ivs, rqmc_ivs=rqmc_ivs,
        T=reg['T'],
    )

# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------

df = pd.DataFrame(records)
df.to_csv('/workspace/project/results/exp5_rough_bergomi.csv', index=False, float_format='%.4f')
print("\nSaved: results/exp5_rough_bergomi.csv")
print(df.to_string(index=False))

# ---------------------------------------------------------------------------
# Plot: IV smiles for each regime
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('EXP 5: Rough Bergomi — Gaussian Closure vs RQMC', fontsize=12)

for ax, reg in zip(axes, regimes):
    name = reg['name']
    res = all_results[name]
    T = res['T']
    bench_ivs   = res['bench_ivs'] * 100
    closure_ivs = res['closure_ivs'] * 100
    rqmc_ivs    = res['rqmc_ivs'] * 100

    valid = np.isfinite(bench_ivs) & np.isfinite(closure_ivs) & np.isfinite(rqmc_ivs)

    ax.plot(strikes_ratio[valid], bench_ivs[valid],   'k-',  lw=2,  label='Bench (RQMC hi-fi)')
    ax.plot(strikes_ratio[valid], closure_ivs[valid], 'b--', lw=1.5, label='Closure')
    ax.plot(strikes_ratio[valid], rqmc_ivs[valid],    'r:',  lw=1.5, label='RQMC light')

    rmse_cl = records[[r['regime'] for r in records].index(name)]['rmse_closure_bp']
    rmse_rq = records[[r['regime'] for r in records].index(name)]['rmse_rqmc_bp']
    ax.set_title(
        f'{name} (H={reg["H"]}, η={reg["eta"]}, ρ={reg["rho"]})\n'
        f'Closure {rmse_cl:.1f}bp  RQMC {rmse_rq:.1f}bp',
        fontsize=9
    )
    ax.set_xlabel('K/F')
    ax.set_ylabel('IV (%)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/project/results/exp5_rough_bergomi.png', dpi=120, bbox_inches='tight')
plt.close()
print("Saved: results/exp5_rough_bergomi.png")
print("\nexp_5 done")
