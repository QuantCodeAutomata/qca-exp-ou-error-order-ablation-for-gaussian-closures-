"""
Experiment 6: Rough Bergomi (rho=0) — 2PI Hartree closure vs Gaussian cumulant.

For rho=0 the log-return characteristic function factorises:
  phi(u) = E[exp(-w*V)]   where  w = 0.5*(iu + u^2),
           V = int_0^T v_t dt,   v_t = v0*exp(eta*Y_t - 0.5*eta^2*t^{2H}).

Two analytical approximations are compared against an RQMC benchmark:

  GC  (Gaussian cumulant) : log phi(u) ≈ -w*EV - 0.5*w^2*VarV
  2PI Hartree              : self-consistent propagator G via gap equations
                             G^{-1} = K^{-1} + diag(eta^2 * w * dt * vtilde)
                             vtilde_i = v0*exp(0.5*eta^2*G_ii - 0.5*eta^2*t_i^{2H})
                             Free energy: W = -w*dt*sum(vtilde) + 0.5*logdet(G/K)

Call prices obtained via COS method.  N=20 time steps for the 2PI (fast).

Saves results/exp6_rbergomi_rho0.csv and results/exp6_rbergomi_rho0.png.
"""

import sys
import os

sys.path.insert(0, '/workspace/project')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.rough_field import build_rough_covariance
from src.rough_bergomi_rqmc import rough_bergomi_rqmc
from src.metrics import implied_vol, rmse_bp

os.makedirs('/workspace/project/results', exist_ok=True)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

CASES = [
    dict(name='H02_eta10', H=0.20, eta=1.0, v0=0.04, T=1.0),
    dict(name='H10_eta15', H=0.10, eta=1.5, v0=0.04, T=1.0),
    dict(name='H07_eta19', H=0.07, eta=1.9, v0=0.04, T=2.0),
]

F0 = 1.0
rho = 0.0   # rho=0 exactly
strikes_ratio = np.array([0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15])
strikes = F0 * strikes_ratio

N_2PI = 20      # time steps used in 2PI / GC
N_COS = 64      # COS Fourier terms
L_COS = 10.0    # truncation half-width in vol units
HARTREE_ITER = 50
HARTREE_DAMP = 0.35


# ---------------------------------------------------------------------------
# Analytic moment helpers (exact from K)
# ---------------------------------------------------------------------------

def compute_ev_varv(v0, eta, H, T, N=N_2PI):
    """Return (EV, VarV) from fBm covariance.

    v_t = v0*exp(eta*W_t - 0.5*eta^2*t^{2H}) is a martingale: E[v_t] = v0.
    E[v_t * v_s] = v0^2 * exp(eta^2 * K(t,s)).
    """
    K_cov, t_grid = build_rough_covariance(T, N, H)
    dt = T / N
    EV = v0 * T
    Ev_ts = v0 ** 2 * np.exp(eta ** 2 * K_cov)
    EV2 = np.sum(Ev_ts) * dt ** 2
    VarV = EV2 - EV ** 2
    return EV, VarV, K_cov, t_grid


# ---------------------------------------------------------------------------
# COS pricer from log-CF array
# ---------------------------------------------------------------------------

def cos_prices_from_logcf(log_phi_fn, F0, T, strikes, N_cos=N_COS, L=L_COS):
    """
    COS method call prices given a function u -> log phi(u) (complex).

    log_phi_fn(u_arr) must accept a 1-D real array and return a 1-D complex array.
    """
    # Approximate variance for truncation
    log_phi_1 = log_phi_fn(np.array([1.0]))[0]
    approx_var = max(abs(np.real(log_phi_1)) * 4.0, 0.04 * T)
    approx_std = np.sqrt(max(approx_var, 0.04 * T))
    a = -L * approx_std
    b = L * approx_std
    c = b - a

    k_arr = np.arange(N_cos, dtype=float)
    u_arr = k_arr * np.pi / c

    log_phi = log_phi_fn(u_arr)            # (N_cos,) complex
    phi = np.exp(log_phi - 1j * u_arr * a)
    phi[0] = 1.0
    phi = np.where(np.isfinite(phi), phi, 0.0)
    phi_real = np.real(phi)

    kpi_c = k_arr * np.pi / c
    safe_kpi_c = np.where(k_arr == 0, 1.0, kpi_c)

    prices = np.empty(len(strikes))
    for i, K in enumerate(strikes):
        x0 = np.log(K / F0)
        a_k = a - x0
        b_k = b - x0
        lo = max(a_k, 0.0)
        hi = b_k
        if lo >= hi:
            prices[i] = max(F0 - K, 0.0)
            continue

        chi = np.where(
            k_arr == 0,
            np.exp(hi) - np.exp(lo),
            (np.exp(hi) * (np.cos(kpi_c * (hi - a_k)) + kpi_c * np.sin(kpi_c * (hi - a_k)))
             - np.exp(lo) * (np.cos(kpi_c * (lo - a_k)) + kpi_c * np.sin(kpi_c * (lo - a_k))))
            / (1.0 + kpi_c ** 2),
        )
        psi = np.where(
            k_arr == 0,
            hi - lo,
            (np.sin(safe_kpi_c * (hi - a_k)) - np.sin(safe_kpi_c * (lo - a_k))) / safe_kpi_c,
        )
        V_k = (2.0 / c) * (chi - psi)
        terms = phi_real * V_k
        terms[0] *= 0.5
        prices[i] = max(K * terms.sum(), 0.0)

    return prices


# ---------------------------------------------------------------------------
# Method 1: Gaussian Cumulant (GC)
# ---------------------------------------------------------------------------

def gc_logcf(u_arr, EV, VarV):
    """Two-cumulant log-CF: log phi(u) = -w*EV - 0.5*w^2*VarV."""
    w = 0.5 * (1j * u_arr + u_arr ** 2)
    return -w * EV - 0.5 * w ** 2 * VarV


# ---------------------------------------------------------------------------
# Method 2: 2PI Hartree
# ---------------------------------------------------------------------------

def hartree_logcf_one(u, v0, eta, H, t_grid, K_cov, dt,
                      n_iter=HARTREE_ITER, damp=HARTREE_DAMP):
    """Hartree free energy W(u) for a single real Fourier variable u."""
    N = len(t_grid)
    w = 0.5 * (1j * u + u ** 2)

    K_inv = np.linalg.inv(K_cov)
    t2H = t_grid ** (2 * H)

    # Initialise G = K (real), then iterate with complex self-energy
    G_diag = np.diag(K_cov).astype(complex)
    G = K_cov.astype(complex)

    for _ in range(n_iter):
        # Effective variance with Gaussian averaging over G
        v_tilde = v0 * np.exp(0.5 * eta ** 2 * G_diag - 0.5 * eta ** 2 * t2H)

        # Diagonal self-energy: Sigma_ii = w * eta^2 * dt * v_tilde_i
        Sigma_diag = w * eta ** 2 * dt * v_tilde  # (N,) complex

        # Gap equation: G^{-1} = K^{-1} + diag(Sigma)
        G_inv_new = K_inv + np.diag(Sigma_diag)
        G_new = np.linalg.inv(G_inv_new)
        G_diag_new = np.diag(G_new)

        # Damped update of G_diag and G
        G_diag = (1.0 - damp) * G_diag + damp * G_diag_new
        G = G_new

    # Final effective variance
    v_tilde = v0 * np.exp(0.5 * eta ** 2 * G_diag - 0.5 * eta ** 2 * t2H)
    EV_eff = np.sum(v_tilde) * dt

    # Log-det ratio: 0.5 * (logdet G - logdet K)
    sign_G, logdet_G = np.linalg.slogdet(G)
    sign_K, logdet_K = np.linalg.slogdet(K_cov)
    logdet_ratio = 0.5 * (np.log(complex(sign_G)) + logdet_G
                          - np.log(complex(sign_K)) - logdet_K)

    W = -w * EV_eff + logdet_ratio
    return W


def hartree_logcf(u_arr, v0, eta, H, t_grid, K_cov, dt):
    """Vectorise hartree_logcf_one over u_arr."""
    return np.array([
        hartree_logcf_one(float(u), v0, eta, H, t_grid, K_cov, dt)
        for u in u_arr
    ])


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

print("=== EXP 6 — Rough Bergomi (rho=0): 2PI Hartree vs Gaussian Cumulant ===\n")

records = []
all_results = {}

for case in CASES:
    name = case['name']
    v0, eta, H, T = case['v0'], case['eta'], case['H'], case['T']
    print(f"Case {name}  (H={H}, eta={eta}, T={T})...")

    EV, VarV, K_cov, t_grid = compute_ev_varv(v0, eta, H, T, N=N_2PI)
    dt = T / N_2PI

    # --- Benchmark: RQMC ---
    bench_prices = rough_bergomi_rqmc(
        v0=v0, eta=eta, rho=rho, H=H, T=T, strikes=strikes,
        F0=F0, N=50, n_paths=2048, n_scrambles=8
    )

    # --- GC: Gaussian cumulant ---
    gc_prices = cos_prices_from_logcf(
        lambda u, EV=EV, VarV=VarV: gc_logcf(u, EV, VarV),
        F0, T, strikes
    )

    # --- 2PI Hartree ---
    h2pi_prices = cos_prices_from_logcf(
        lambda u, v0=v0, eta=eta, H=H, tg=t_grid, Kc=K_cov, dt=dt:
            hartree_logcf(u, v0, eta, H, tg, Kc, dt),
        F0, T, strikes
    )

    # --- Implied vols ---
    bench_ivs = np.array([implied_vol(max(p, 1e-10), F0, K, T) for p, K in zip(bench_prices, strikes)])
    gc_ivs    = np.array([implied_vol(max(p, 1e-10), F0, K, T) for p, K in zip(gc_prices, strikes)])
    h2pi_ivs  = np.array([implied_vol(max(p, 1e-10), F0, K, T) for p, K in zip(h2pi_prices, strikes)])

    rmse_gc   = rmse_bp(gc_ivs,   bench_ivs)
    rmse_h2pi = rmse_bp(h2pi_ivs, bench_ivs)

    print(f"  EV={EV:.4f}  VarV={VarV:.6f}")
    print(f"  GC   RMSE = {rmse_gc:.2f} bp")
    print(f"  2PI  RMSE = {rmse_h2pi:.2f} bp")

    records.append(dict(
        case=name, H=H, eta=eta, T=T,
        EV=EV, VarV=VarV,
        rmse_gc_bp=rmse_gc, rmse_2pi_bp=rmse_h2pi,
    ))
    all_results[name] = dict(
        bench_ivs=bench_ivs, gc_ivs=gc_ivs, h2pi_ivs=h2pi_ivs, T=T,
    )

# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------

df = pd.DataFrame(records)
df.to_csv('/workspace/project/results/exp6_rbergomi_rho0.csv', index=False, float_format='%.4f')
print("\nSaved: results/exp6_rbergomi_rho0.csv")
print(df.to_string(index=False))

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('EXP 6: Rough Bergomi (ρ=0) — 2PI Hartree vs Gaussian Cumulant', fontsize=12)

for ax, case in zip(axes, CASES):
    name = case['name']
    res = all_results[name]
    T = res['T']
    bench = res['bench_ivs'] * 100
    gc    = res['gc_ivs'] * 100
    h2pi  = res['h2pi_ivs'] * 100

    valid = np.isfinite(bench) & np.isfinite(gc) & np.isfinite(h2pi)
    ax.plot(strikes_ratio[valid], bench[valid], 'k-',  lw=2,  label='RQMC benchmark')
    ax.plot(strikes_ratio[valid], gc[valid],    'b--', lw=1.5, label='GC (2-cumulant)')
    ax.plot(strikes_ratio[valid], h2pi[valid],  'r:',  lw=1.5, label='2PI Hartree')

    row = records[[r['case'] for r in records].index(name)]
    ax.set_title(
        f'{name}\nGC {row["rmse_gc_bp"]:.1f}bp   2PI {row["rmse_2pi_bp"]:.1f}bp',
        fontsize=9
    )
    ax.set_xlabel('K/F')
    ax.set_ylabel('IV (%)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/project/results/exp6_rbergomi_rho0.png', dpi=120, bbox_inches='tight')
plt.close()
print("Saved: results/exp6_rbergomi_rho0.png")
print("\nexp_6 done")
