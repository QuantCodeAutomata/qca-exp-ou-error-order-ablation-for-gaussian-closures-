"""
Experiment 8: exp-OU exotic option pricing — correlation sweep.

Sweeps rho in {-0.5, -0.3, 0.0, 0.3, 0.5} for three exotic payoffs:
  1. One-touch barrier      (B/S0 = 1.06, pays 1 if max S_t >= B)
  2. Reverse knock-out call (K = 100, B/S0 = 1.10, alive while S_t < B)
  3. Forward-vol agreement  (FVA: E[sqrt(int_{0.5}^{1.0} v_t dt)])

Model: dX = -theta*(X-mu)*dt + sigma*dW1,  v = exp(X),  corr(dW1,dW2)=rho
Parameters: theta=1, mu=-3.76, sigma=1.8, T=1

Saves results/exp8_expou_exotics.csv and results/exp8_expou_exotics.png.
"""

import sys
import os

sys.path.insert(0, '/workspace/project')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.exotic_pricing import (
    price_one_touch_mc,
    price_reverse_knockout_mc,
    price_fva_mc,
)

os.makedirs('/workspace/project/results', exist_ok=True)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

THETA = 1.0
MU = -3.76
SIGMA = 1.8
T = 1.0
F0 = 100.0

BARRIER_RATIO_OT = 1.06      # one-touch barrier  B/S0
K_RKO = 100.0                # RKO strike (ATM)
BARRIER_RATIO_RKO = 1.10     # RKO barrier  B/S0

T_START_FVA = 0.5
DELTA_FVA = 0.5

N_PATHS = 20000
N_STEPS = 200

RHOS = [-0.5, -0.3, 0.0, 0.3, 0.5]

# ---------------------------------------------------------------------------
# Sweep rho
# ---------------------------------------------------------------------------

records = []
ot_prices, ot_ses = [], []
rko_prices, rko_ses = [], []
fva_prices, fva_ses = [], []

print("=== EXP 8 — exp-OU exotic option pricing ===")
print(f"theta={THETA}, mu={MU}, sigma={SIGMA}, T={T}")
print(f"One-touch B/S0={BARRIER_RATIO_OT}")
print(f"RKO call:  K={K_RKO}, B/S0={BARRIER_RATIO_RKO}")
print(f"FVA:       T_start={T_START_FVA}, Delta={DELTA_FVA}")
print(f"MC:        n_paths={N_PATHS}, n_steps={N_STEPS}\n")

for rho in RHOS:
    ot_p, ot_se = price_one_touch_mc(
        THETA, MU, SIGMA, rho, T, BARRIER_RATIO_OT,
        F0=F0, n_paths=N_PATHS, n_steps=N_STEPS, seed=42,
    )
    rko_p, rko_se = price_reverse_knockout_mc(
        THETA, MU, SIGMA, rho, T, K_RKO, BARRIER_RATIO_RKO,
        F0=F0, n_paths=N_PATHS, n_steps=N_STEPS, seed=42,
    )
    fva_p, fva_se = price_fva_mc(
        THETA, MU, SIGMA, rho, T_START_FVA, DELTA_FVA,
        F0=F0, n_paths=N_PATHS, n_steps=N_STEPS, seed=42,
    )

    ot_prices.append(ot_p);   ot_ses.append(ot_se)
    rko_prices.append(rko_p); rko_ses.append(rko_se)
    fva_prices.append(fva_p); fva_ses.append(fva_se)

    print(
        f"rho={rho:+.1f}  "
        f"OT={ot_p:.4f}±{ot_se:.4f}  "
        f"RKO={rko_p:.4f}±{rko_se:.4f}  "
        f"FVA={fva_p:.4f}±{fva_se:.4f}"
    )

    records.append(dict(
        rho=rho,
        ot_price=ot_p,  ot_se=ot_se,
        rko_price=rko_p, rko_se=rko_se,
        fva_price=fva_p, fva_se=fva_se,
    ))

# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------

df = pd.DataFrame(records)
df.to_csv('/workspace/project/results/exp8_expou_exotics.csv',
          index=False, float_format='%.6f')
print("\nSaved: results/exp8_expou_exotics.csv")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

rho_arr = np.array(RHOS)
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle(
    'EXP 8: exp-OU Exotic Pricing vs Spot-Vol Correlation ρ\n'
    fr'θ={THETA},  μ={MU},  σ={SIGMA},  T={T},  n_paths={N_PATHS}',
    fontsize=11,
)

plot_specs = [
    (ot_prices,  ot_ses,  f'One-Touch  (B/S₀={BARRIER_RATIO_OT})',   'Touch probability'),
    (rko_prices, rko_ses, f'Reverse KO Call  (K={K_RKO}, B/S₀={BARRIER_RATIO_RKO})',
     'Price'),
    (fva_prices, fva_ses,
     f'FVA  E[√V]  (window [{T_START_FVA},{T_START_FVA+DELTA_FVA}])',
     'E[√(realised vol)]'),
]

for ax, (prices, ses, title, ylabel) in zip(axes, plot_specs):
    ax.errorbar(rho_arr, prices, yerr=2.0 * np.array(ses),
                fmt='o-', capsize=5, color='steelblue', lw=2, ms=7)
    ax.set_xlabel('ρ  (spot-vol correlation)', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-0.62, 0.62)
    ax.grid(alpha=0.35)

plt.tight_layout()
plt.savefig('/workspace/project/results/exp8_expou_exotics.png',
            dpi=120, bbox_inches='tight')
plt.close()
print("Saved: results/exp8_expou_exotics.png")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n=== Summary ===")
print(df.to_string(index=False, float_format=lambda x: f'{x:.5f}'))
