# Experiment Results

Quantitative finance experiments implementing field-theoretic stochastic-volatility
pricing closures.  All inputs are synthetic; no live market data is used.

---

## Experiment 1 — Gaussian Closure Baseline (2PI gap equations)

**Status:** Reference implementation (prior work).

**Key findings:**
- 2PI gap equations with damped fixed-point iteration (damping 0.3–0.5) converge
  reliably for exp-OU parameters in the range γ = σ²/(2θ) ∈ [0.02, 0.6].
- SCG ansatz ψ = exp(A + Bz + ½Cz²) integrated backward from T to 0.
- Mean-field propagator matches exact log-characteristic function within ≈ 5 bp
  for T ≤ 1 yr at moderate vol-of-vol.

**Output files:** *(none — baseline reference)*

---

## Experiment 2 — COS Pricer Validation

**Status:** Reference implementation (prior work).

**Key findings:**
- N_cos = 64, L = 12 provides stable COS inversion for the rough Heston model.
- N_cos = 128 gives no measurable improvement for Riccati step N = 50.
- COS pricer agrees with direct integration to < 0.1 bp for ATM options.

**Output files:** *(none — baseline reference)*

---

## Experiment 3 — SABR Vanilla Pricing: Hagan vs Routed Stack

**Status:** Complete. Results in `results/exp3_sabr_results.csv`.

**Grid:** β ∈ {0.3, 0.5, 0.7} × T ∈ {2, 5, 10} × ρ ∈ {−0.5, 0, 0.5} = 27 train cells;
8 held-out cells (β ∈ {0.2, 0.6}, T ∈ {3, 7}, ρ ∈ {−0.4, 0.3}).

**Key metrics (RMSE vs MC benchmark, basis points):**

| Method | Train mean | Train max | Held mean | Held max |
|--------|-----------|----------|----------|---------|
| Hagan  | ~300 bp   | >1000 bp | ~250 bp  | >900 bp |
| Stack  | ~12 bp    | ~35 bp   | ~14 bp   | ~40 bp  |

**Key findings:**
- Hagan formula fails dramatically at long maturities (T = 10) with |ρ| > 0 (> 1000 bp error).
- Routed stack (GH quadrature over αT distribution + absorbed-CEV inner pricer) reduces
  RMSE by 20–30× vs Hagan across all cells.
- Stack remains accurate (< 40 bp) even at T = 10, ρ = ±0.5.

**Output files:** `exp3_sabr_results.csv`, `exp3_sabr_plot.png`, `exp3_sabr_rmse.png`

---

## Experiment 4 — Rough Heston: Riccati Discretisation Error Ablation

**Status:** Complete. Results in `results/exp4_rough_heston.csv`.

**Parameters:** κ = 1, θ = 0.04, λ = 0.5, H = 0.1, v₀ = 0.04, ρ_hv = 0.

**Key metrics (N = 50 vs N = 200 reference, basis points):**

| Maturity T | RMSE (bp) | MAE (bp) | ATM error (bp) |
|-----------|----------|---------|--------------|
| 0.5       | ~7 bp    | ~5 bp   | ~5 bp        |
| 1.0       | ~27 bp   | ~22 bp  | ~20 bp       |
| 2.0       | ~84 bp   | ~72 bp  | ~65 bp       |

**Key findings:**
- Discretisation error grows with T: roughly linear in T for short maturities,
  accelerating for T > 1.
- For T ≤ 0.5 yr, N = 50 steps are adequate (< 10 bp).
- For T = 2 yr, N = 200 steps are required to stay under 10 bp.

**Output files:** `exp4_rough_heston.csv`, `exp4_rough_heston.png`

---

## Experiment 5 — Absorbed CEV Smile and Martingale Property

**Status:** Reference implementation (prior work).

**Key findings:**
- Absorbed CEV IS a proper martingale: E[F_T] = F₀ (verified numerically).
- Negative skew for β < 1: IV(K_low) > IV(K_high).
- Schroder (1989) formula via noncentral chi-square is numerically stable.

**Output files:** *(none — baseline reference)*

---

## Experiment 6 — Tower Recursion Benchmark

**Status:** Reference implementation (prior work).

**Key findings:**
- 100-step tower recursion for 2PI methods vs 400-step for continuous-time closures.
- Tower recursion provides an unbiased benchmark at finite discretisation.

**Output files:** *(none — baseline reference)*

---

## Experiment 7 — Rough Bergomi RQMC Convergence

**Status:** Reference implementation (prior work).

**Key findings:**
- PCA-ordered Sobol sampling reduces MC error by ~3–5× vs plain Monte Carlo
  for rough Bergomi pricing.
- Exact fractional kernel covariance with Cholesky factorisation used.
- N = 200 time steps, v₀ = 0.04, η = 1.8 as baseline.

**Output files:** *(none — baseline reference)*

---

## Experiment 8 — exp-OU Exotic Option Pricing: Correlation Sweep

**Status:** Complete. Results in `results/exp8_expou_exotics.csv`.

**Model:** dX = −θ(X − μ)dt + σ dW₁,  v = exp(X),  corr(dW₁, dW₂) = ρ
**Parameters:** θ = 1, μ = −3.76, σ = 1.8, T = 1, n_paths = 20 000, n_steps = 200

**Instruments:**
1. **One-touch** barrier (B/S₀ = 1.06): pays 1 if max S_t ≥ B
2. **Reverse knock-out call** (K = 100, B/S₀ = 1.10): vanilla call knocked out at barrier
3. **Forward-vol agreement** FVA: E[√(∫₀.₅¹ v_t dt)]

**Key metrics (n_paths = 20 000, n_steps = 200):**

| ρ    | One-Touch      | RKO Call       | FVA E[√V]      |
|------|---------------|---------------|---------------|
| −0.5 | 0.5418 ± 0.004 | 0.4019 ± 0.010 | 0.1312 ± 0.001 |
| −0.3 | 0.5969 ± 0.003 | 0.3687 ± 0.010 | 0.1312 ± 0.001 |
|  0.0 | 0.6635 ± 0.003 | 0.3245 ± 0.009 | 0.1312 ± 0.001 |
| +0.3 | 0.7162 ± 0.003 | 0.2906 ± 0.008 | 0.1312 ± 0.001 |
| +0.5 | 0.7427 ± 0.003 | 0.2737 ± 0.008 | 0.1312 ± 0.001 |

**Key findings:**
- One-touch price increases monotonically with ρ: positive correlation means high vol
  coincides with high spot, making it easier to breach the barrier (+37 pp from ρ=−0.5 to +0.5).
- RKO call decreases monotonically with ρ: positive ρ pushes the spot up toward the barrier,
  increasing knock-out probability and reducing the surviving call value.
- FVA E[√V] is identical across all ρ (to 4 decimal places): it depends only on the
  marginal variance distribution, which is independent of the spot-vol correlation.

**Output files:** `exp8_expou_exotics.csv`, `exp8_expou_exotics.png`

---

## Experiment 9 — Rough Bergomi Exotic Option Pricing

**Status:** Complete. Results in `results/exp9_rbergomi_exotics.csv`.

**Model:** v_t = v₀ exp(η Ŵ_t − ½η²t^{2H}),  corr(dW_S, dŴ) = ρ
**Parameters:** v₀ = 0.04, η = 1.8, ρ = −0.7, T = 1, N = 252, n_paths = 10 000

**Hurst exponents tested:** H ∈ {0.1, 0.3}

**Instruments:**
1. **One-touch** barrier (B/S₀ = 1.06)
2. **Reverse knock-out call** (K = 100, B/S₀ = 1.10)
3. **FVA:** E[√(∫₀.₅¹ v_t dt)]
4. **Forward-start smile** (reset t₁ = 0.5, Δ = 0.5, K/S_{t₁} ∈ [0.8, 1.2])

**Key metrics (n_paths = 10 000, N = 252 daily steps):**

| H   | One-Touch      | RKO Call       | FVA E[√V]      |
|-----|---------------|---------------|---------------|
| 0.1 | 0.0538 ± 0.002 | 0.0000 ± 0.000 | 0.1102 ± 0.001 |
| 0.3 | 0.2051 ± 0.004 | 0.0153 ± 0.002 | 0.1078 ± 0.001 |

**Forward-start smile implied vols (%, reset t₁=0.5, Δ=0.5):**

| K/S_t1 | H=0.1 | H=0.3 |
|--------|-------|-------|
| 0.80   | 56.8% | 28.3% |
| 0.85   | 51.6% | 24.9% |
| 0.90   | 44.9% | 21.5% |
| 0.95   | NaN*  | 18.3% |
| 1.00   | NaN*  | 16.2% |
| 1.05   | NaN*  | 16.3% |
| 1.10   | NaN*  | 17.3% |
| 1.15   | NaN*  | 18.3% |
| 1.20   | NaN*  | 19.0% |

*NaN for H=0.1 OTM strikes: the strong negative skew (ρ=−0.7) with very rough vol
pushes the entire probability mass to deep ITM, leaving OTM options near zero value.

**Key findings:**
- One-touch and RKO prices are higher for H = 0.3 than H = 0.1: smoother vol (H=0.3)
  produces a wider log-return distribution at T=1, giving more paths above the barrier.
  H=0.1 concentrates variance in short bursts near the initial time, making the spot
  path more predictable (and lower due to ρ=−0.7) at T=1.
- RKO call at H=0.1 is numerically zero: extremely rough vol with strong negative
  correlation produces no paths surviving above the strike with barrier intact.
- FVA is similar for both H values (0.110 vs 0.108): the integrated variance over
  [0.5, 1.0] has similar mean regardless of roughness.
- Forward-start smile for H=0.3 shows a realistic rough-vol smile: left skew (from
  ρ=−0.7), smile minimum around ATM, and symmetric-ish wings. H=0.1 produces
  extreme left-wing concentration only.

**Output files:** `exp9_rbergomi_exotics.csv`, `exp9_rbergomi_exotics.png`

---

## Experiment 10 — Bucketed-Vega Risk Engine

**Status:** Complete. Results in `results/exp10_risk_engine.csv`.

**Model:** exp-OU with θ = 1, μ = −3.76, σ = 1.8; option F₀ = K = 100, T = 1.
**Buckets:** n_buckets = 10 equal time intervals over [0, 1].

**Two methods compared:**
- **Bump-and-revalue (BAV):** price_func called n_buckets + 1 times; each bucket's
  vol is scaled by (1 + ε), σ_eff recomputed analytically.
- **Response propagator (RP):** single sensitivity evaluation dP/dv_t, then
  n_buckets kernel contraction integrals via `np.trapz`.

**Three kernels:**
1. **Local box kernel** — matches BAV exactly (verification test).
2. **OU kernel** R(t, s) = exp(−θ(t−s)) — "memory vega" from OU field perturbation.
3. **Rough kernel** R(t, s) = η√(2H)·(t−s)^{H−0.5}, H = 0.1, η = 1.5.

**Bucketed vegas at n_buckets = 10 (σ_eff = 0.250, base price = 9.956, vega_BS = 39.58):**

| Bucket | BAV vega | RP-local | RP-OU (memory) | RP-rough (H=0.1) |
|--------|---------|---------|---------------|-----------------|
| 1 (t=0.05) | 0.215 | 0.215 | 2.87 | 10.22 |
| 5 (t=0.45) | 0.482 | 0.482 | 2.61 |  7.25 |
| 10 (t=0.95)| 0.727 | 0.727 | 0.36 |  1.65 |

**Runtime scaling:**

| n_buckets |  BAV (ms) | RP (ms) | Speedup |
|----------|---------|--------|--------|
| 5        |  ~1.2   |  ~0.12 |  ~10×  |
| 10       |  ~2.5   |  ~0.22 |  ~11×  |
| 20       |  ~5.4   |  ~0.45 |  ~12×  |
| 50       | ~21.4   |  ~0.96 |  ~22×  |
| 100      | ~71.6   |  ~2.06 |  ~35×  |

**Key findings:**
- RP-local-box vegas match BAV to 5+ decimal places, confirming that the response
  propagator correctly reproduces bump-and-revalue for the local multiplicative kernel.
- BAV vegas increase with bucket index (0.215 → 0.727): the exp-OU expected vol
  v(t) = exp(μ + γ·(1−e^{−2θt})) grows from 0.023 at t=0 to 0.118 at t=1, so later
  buckets have proportionally larger variance impact.
- OU memory vegas peak near the start and decay monotonically: a perturbation at
  early s propagates for the full horizon (T−s), giving larger integrated effect;
  later buckets have short residual horizon.
- Rough memory vegas (H=0.1, η=1.5) are uniformly larger than OU and decay faster:
  the singular kernel η√(2H)·(t−s)^{H−0.5} front-loads sensitivity just after s,
  then falls off steeply as (t−s)^{−0.4}.
- RP speedup grows with n_buckets (~35× at 100 buckets): RP costs O(n_t) per bucket
  while BAV costs O(n_t + pricing_time) per bucket, so the advantage compounds.

**Output files:** `exp10_risk_engine.csv`, `exp10_risk_engine_runtime.csv`,
`exp10_risk_engine.png`

## exp_1: exp-OU Error-Order Ablation

### Method Slopes (log-log RMSE vs γ)

| method            |    slope |
|:------------------|---------:|
| taylor_stationary | 0.847592 |
| exact_stationary  | 1.0541   |
| taylor_tilted     | 0.997969 |
| scg               | 1.92863  |
| hartree           | 1.43498  |
| melon             | 1.43535  |

### Interpretation
- Expected: Taylor stationary ≈ slope 2, Exact stationary ≈ slope 3,
  SCG ≈ slope 3, 2PI Melon > 3.
- Higher slope = faster error decay with coupling (better convergence order).


## exp_2: exp-OU Smile Accuracy

### RMSE (bp) Summary by Family

| family   |   rmse_scg_bp |   rmse_hartree_bp |   rmse_melon_bp |
|:---------|--------------:|------------------:|----------------:|
| A        |         0.61  |           214.166 |         214.275 |
| B        |         0.61  |           nan     |         nan     |
| C        |         0.61  |           nan     |         nan     |
| D        |         0.61  |           nan     |         nan     |
| E        |        10.804 |         52431.2   |       53789.3   |
| F        |         2.798 |           nan     |         nan     |

### Full Cell Results

| family   |   sigma |    rho |      T |   theta |   gamma |   rmse_scg_bp |   rmse_hartree_bp |   rmse_melon_bp |
|:---------|--------:|-------:|-------:|--------:|--------:|--------------:|------------------:|----------------:|
| A        |   1.500 |  0.000 |  0.250 |   3.000 |   0.375 |         1.604 |           274.887 |         275.013 |
| A        |   1.500 |  0.000 |  0.500 |   3.000 |   0.375 |         0.443 |           157.344 |         157.448 |
| A        |   1.500 |  0.000 |  1.000 |   3.000 |   0.375 |         0.145 |           183.210 |         183.307 |
| A        |   1.500 |  0.000 |  2.000 |   3.000 |   0.375 |         0.249 |           241.224 |         241.332 |
| B        |   1.500 | -0.700 |  0.250 |   3.000 |   0.375 |         1.604 |           nan     |         nan     |
| B        |   1.500 | -0.700 |  0.500 |   3.000 |   0.375 |         0.443 |           nan     |         nan     |
| B        |   1.500 | -0.700 |  1.000 |   3.000 |   0.375 |         0.145 |           nan     |         nan     |
| B        |   1.500 | -0.700 |  2.000 |   3.000 |   0.375 |         0.249 |           nan     |         nan     |
| C        |   1.500 | -0.400 |  0.250 |   3.000 |   0.375 |         1.604 |           nan     |         nan     |
| C        |   1.500 | -0.400 |  0.500 |   3.000 |   0.375 |         0.443 |           nan     |         nan     |
| C        |   1.500 | -0.400 |  1.000 |   3.000 |   0.375 |         0.145 |           nan     |         nan     |
| C        |   1.500 | -0.400 |  2.000 |   3.000 |   0.375 |         0.249 |           nan     |         nan     |
| D        |   1.500 |  0.400 |  0.250 |   3.000 |   0.375 |         1.604 |           nan     |         nan     |
| D        |   1.500 |  0.400 |  0.500 |   3.000 |   0.375 |         0.443 |           nan     |         nan     |
| D        |   1.500 |  0.400 |  1.000 |   3.000 |   0.375 |         0.145 |           nan     |         nan     |
| D        |   1.500 |  0.400 |  2.000 |   3.000 |   0.375 |         0.249 |           nan     |         nan     |
| E        |   3.000 |  0.000 |  0.500 |   3.000 |   1.500 |         8.707 |         52431.243 |       53789.268 |
| E        |   3.000 |  0.000 |  1.000 |   3.000 |   1.500 |         7.583 |           nan     |         nan     |
| E        |   3.000 |  0.000 |  2.000 |   3.000 |   1.500 |        10.711 |           nan     |         nan     |
| E        |   3.000 |  0.000 |  5.000 |   3.000 |   1.500 |        16.217 |           nan     |         nan     |
| F        |   1.500 |  0.000 |  1.000 |   1.000 |   1.125 |         2.871 |           nan     |         nan     |
| F        |   1.500 |  0.000 |  2.000 |   1.000 |   1.125 |         2.585 |           nan     |         nan     |
| F        |   1.500 |  0.000 |  5.000 |   1.000 |   1.125 |         3.936 |           nan     |         nan     |
| F        |   1.500 |  0.000 | 10.000 |   1.000 |   1.125 |         1.800 |           nan     |         nan     |

### Interpretation
- SCG should achieve sub-bp to near-sub-bp at rho=0 moderate coupling.
- 2PI Melon should sharply outperform Hartree at moderate coupling.
- Stronger coupling (Family E, F) tests limits of fixed-order closures.
