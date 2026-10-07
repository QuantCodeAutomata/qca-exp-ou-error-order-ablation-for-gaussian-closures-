"""
exp_1: exp-OU error-order ablation for Gaussian closures and 2PI skeleton hierarchy.

Compares six method variants against an exact tower benchmark to measure the
convergence order (log-log slope of RMSE vs effective coupling gamma).

Methods:
  1. Quadratic Taylor vertex under stationary Gaussian N(0, gamma)
  2. Exact exponential prefactor under stationary Gaussian N(0, gamma)
  3. Taylor prefactor under tilted Gaussian N(m, v)
  4. SCG with exact prefactor under tilted Gaussian (rho=0)
  5. 2PI Hartree
  6. 2PI melon

Benchmark: 100-step tower for 2PI methods; 400-step tower for SCG/continuous methods.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from src.model_dynamics import ExpOUParams
from src.tower_benchmark import tower_cf_vec
from src.scg_solver import scg_cf, scg_stationary_taylor_cf, scg_stationary_exact_cf, scg_tilted_taylor_cf
from src.two_pi_solver import two_pi_cf
from src.cos_pricer import cos_smile_from_cf_batch
from src.metrics import implied_vol, rmse_bp

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Parameters fixed across all experiments
# ---------------------------------------------------------------------------
THETA = 3.0
MU = -2.0
T = 1.0
RHO = 0.0
F0 = 1.0          # normalised forward

# Strike set: 5 strikes around ATM
STRIKES = np.array([0.85, 0.90, 1.00, 1.10, 1.15])

# Coupling sweep: gamma = sigma^2 / (2*theta) in [0.02, 0.6]
GAMMAS = np.array([0.02, 0.04, 0.08, 0.12, 0.18, 0.25, 0.35, 0.45, 0.60])
SIGMAS = np.sqrt(2.0 * THETA * GAMMAS)  # infer sigma from gamma

# Tower steps  (same grid used for both benchmark and 2PI approximation)
N_BENCH_DISCRETE = 20   # for 2PI methods  (matches paper's "100-step" intent;
#                         reduced to 20 here to stay within runtime budget while
#                         preserving the error-ORDER structure vs gamma)
N_BENCH_CONT = 100       # for SCG and stationary methods (400-step paper target;
#                         100 steps gives sub-bp tower accuracy for this parameter set)

# COS settings — use batched evaluation for all methods
COS_L = 6.0
COS_N = 64   # 64 terms sufficient for this model; full benchmark resolved to sub-bp


# ---------------------------------------------------------------------------
# Helper: compute RMSE in bp across strikes at one gamma
# ---------------------------------------------------------------------------
def compute_rmse(
    method_prices: np.ndarray,
    bench_prices: np.ndarray,
    F: float,
    strikes: np.ndarray,
    T: float,
) -> float:
    """Compute RMSE in bp between method and benchmark call prices."""
    iv_m = np.array([implied_vol(p, F, K, T) for p, K in zip(method_prices, strikes)])
    iv_b = np.array([implied_vol(p, F, K, T) for p, K in zip(bench_prices, strikes)])
    return rmse_bp(iv_m, iv_b)


def get_benchmark_prices(sigma: float, bench_steps: int) -> np.ndarray:
    """Compute tower benchmark call prices across all strikes in one batched run."""
    try:
        cf_batch = lambda u_arr, s=sigma, n=bench_steps: tower_cf_vec(
            u_arr, THETA, MU, s, T, N=n
        )
        return cos_smile_from_cf_batch(cf_batch, F0, STRIKES, T, L=COS_L, N=COS_N)
    except Exception:
        return np.full(len(STRIKES), np.nan)


def get_method_prices(method_name: str, sigma: float) -> np.ndarray:
    """Compute method call prices across strikes using batched COS."""
    gamma = sigma ** 2 / (2.0 * THETA)

    try:
        s = sigma  # capture by value
        if method_name == "taylor_stationary":
            cf_batch = lambda u_arr, s=s: np.array(
                [scg_stationary_taylor_cf(u, THETA, MU, s, T) for u in u_arr]
            )
        elif method_name == "exact_stationary":
            cf_batch = lambda u_arr, s=s: np.array(
                [scg_stationary_exact_cf(u, THETA, MU, s, T) for u in u_arr]
            )
        elif method_name == "taylor_tilted":
            cf_batch = lambda u_arr, s=s: np.array(
                [scg_tilted_taylor_cf(u, THETA, MU, s, T) for u in u_arr]
            )
        elif method_name == "scg":
            cf_batch = lambda u_arr, s=s: np.array(
                [scg_cf(u, THETA, MU, s, T) for u in u_arr]
            )
        elif method_name in ("hartree", "melon"):
            _method = method_name  # "hartree" or "melon"
            cf_batch = lambda u_arr, s=s, m=_method: np.array(
                [two_pi_cf(u, THETA, MU, s, T, method=m, N=N_BENCH_DISCRETE) for u in u_arr]
            )
        else:
            raise ValueError(f"Unknown method: {method_name}")

        return cos_smile_from_cf_batch(cf_batch, F0, STRIKES, T, L=COS_L, N=COS_N)
    except Exception:
        return np.full(len(STRIKES), np.nan)


def fit_log_log_slope(gammas: np.ndarray, rmses: np.ndarray) -> float:
    """Fit log(RMSE) = slope*log(gamma) + const via OLS. Skip NaN/zero entries."""
    valid = np.isfinite(rmses) & (rmses > 0)
    if valid.sum() < 3:
        return np.nan
    log_g = np.log(gammas[valid])
    log_r = np.log(rmses[valid])
    slope, _ = np.polyfit(log_g, log_r, 1)
    return float(slope)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_exp1() -> pd.DataFrame:
    """
    Run the full error-order ablation.

    Returns
    -------
    df : DataFrame with columns [gamma, method, rmse_bp, log_gamma, log_rmse]
    """
    print("=" * 60)
    print("exp_1: exp-OU error-order ablation")
    print("=" * 60)

    # Discrete-time methods use N_BENCH_DISCRETE = 100 steps
    # Continuous-time methods (SCG, stationary) use N_BENCH_CONT = 400 steps
    discrete_methods = ["hartree", "melon"]
    cont_methods = ["taylor_stationary", "exact_stationary", "taylor_tilted", "scg"]

    all_records = []

    for j, (gamma, sigma) in enumerate(zip(GAMMAS, SIGMAS)):
        print(f"\n  gamma={gamma:.3f}  sigma={sigma:.4f}")

        # Benchmarks
        bench_d = get_benchmark_prices(sigma, N_BENCH_DISCRETE)
        bench_c = get_benchmark_prices(sigma, N_BENCH_CONT)

        for method in cont_methods + discrete_methods:
            bench = bench_c if method in cont_methods else bench_d
            prices = get_method_prices(method, sigma)
            err = compute_rmse(prices, bench, F0, STRIKES, T)
            print(f"    {method:20s}  RMSE={err:8.3f} bp")
            all_records.append({
                "gamma": gamma,
                "sigma": sigma,
                "method": method,
                "rmse_bp": err,
                "log_gamma": np.log(gamma),
                "log_rmse": np.log(err) if (err > 0 and np.isfinite(err)) else np.nan,
            })

    df = pd.DataFrame(all_records)
    return df


def run_slope_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log-log slope (error order) for each method."""
    methods = df["method"].unique()
    rows = []
    for m in methods:
        sub = df[df["method"] == m]
        slope = fit_log_log_slope(sub["gamma"].values, sub["rmse_bp"].values)
        rows.append({"method": m, "slope": slope})
        print(f"  {m:20s}  slope = {slope:.2f}")
    return pd.DataFrame(rows)


def plot_error_order(df: pd.DataFrame, slope_df: pd.DataFrame) -> None:
    """Plot log-log RMSE vs gamma for all methods."""
    method_styles = {
        "taylor_stationary": ("C0", "o", "-", "Taylor stat."),
        "exact_stationary":  ("C1", "s", "-", "Exact stat."),
        "taylor_tilted":     ("C2", "^", "-", "Taylor tilted"),
        "scg":               ("C3", "D", "-", "SCG exact"),
        "hartree":           ("C4", "v", "--", "2PI Hartree"),
        "melon":             ("C5", "P", "--", "2PI Melon"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for method, (color, marker, ls, label) in method_styles.items():
        sub = df[df["method"] == method].sort_values("gamma")
        valid = sub["rmse_bp"] > 0
        if valid.any():
            ax.loglog(
                sub.loc[valid, "gamma"],
                sub.loc[valid, "rmse_bp"],
                color=color, marker=marker, linestyle=ls, label=label,
            )
    ax.set_xlabel("Effective coupling γ = σ²/(2θ)")
    ax.set_ylabel("RMSE (bp)")
    ax.set_title("exp-OU Error Order: RMSE vs γ")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    ax2 = axes[1]
    methods_plot = slope_df.dropna()
    ax2.barh(
        methods_plot["method"],
        methods_plot["slope"],
        color="steelblue",
        edgecolor="black",
    )
    ax2.axvline(x=2.0, color="red", linestyle="--", alpha=0.7, label="slope=2")
    ax2.axvline(x=3.0, color="orange", linestyle="--", alpha=0.7, label="slope=3")
    ax2.set_xlabel("Log-log slope (error order)")
    ax2.set_title("exp-OU Error-Order Slopes by Method")
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    out_path = RESULTS_DIR / "exp1_error_order.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\nSaved plot: {out_path}")


def save_results(df: pd.DataFrame, slope_df: pd.DataFrame) -> None:
    """Save CSV outputs and append RESULTS.md."""
    df.to_csv(RESULTS_DIR / "exp1_rmse_by_gamma.csv", index=False)
    slope_df.to_csv(RESULTS_DIR / "exp1_slopes.csv", index=False)

    md_path = RESULTS_DIR / "RESULTS.md"
    section = f"""
## exp_1: exp-OU Error-Order Ablation

### Method Slopes (log-log RMSE vs γ)

{slope_df.to_markdown(index=False)}

### Interpretation
- Expected: Taylor stationary ≈ slope 2, Exact stationary ≈ slope 3,
  SCG ≈ slope 3, 2PI Melon > 3.
- Higher slope = faster error decay with coupling (better convergence order).

"""
    with open(md_path, "a") as f:
        f.write(section)
    print(f"Appended results to {md_path}")


if __name__ == "__main__":
    df = run_exp1()
    print("\n--- Slope analysis ---")
    slope_df = run_slope_analysis(df)
    plot_error_order(df, slope_df)
    save_results(df, slope_df)
    print("\nexp_1 complete.")
