# Repository: qca-exp-ou-error-order-ablation-for-gaussian-closures

## Project Overview
Quantitative finance experiments implementing field-theoretic stochastic-volatility
pricing closures. No market data needed — all experiments use synthetic inputs.

## Key Architecture Decisions
- All src/ modules use NumPy/SciPy for numerics (double precision throughout)
- COS method for Fourier inversion; tower recursion for benchmarks
- 2PI solver uses damped fixed-point iteration with continuation in Fourier variable u
- Rough-field covariance built from exact fractional kernel; PCA-ordered Sobol sampling
- pyfeng used for implied-vol conversion and Black formula utilities
- Results saved to results/ directory as .csv files and PNG plots

## Module Dependency Map
- cos_pricer.py: depends on numpy/scipy only
- tower_benchmark.py: depends on numpy/scipy
- scg_solver.py: depends on scipy.integrate.solve_ivp, cos_pricer.py
- two_pi_solver.py: depends on numpy/scipy, cos_pricer.py
- model_dynamics.py: base model parameter containers
- rough_field.py: rough Gaussian field covariance and KL sampling
- absorbed_cev.py: noncentral chi-square via scipy.stats
- sabr_stack.py: depends on absorbed_cev.py, gauss-hermite quadrature
- sabr_jump_closure.py: depends on absorbed_cev.py
- rough_bergomi_rqmc.py: depends on rough_field.py
- rough_bergomi_closure.py: depends on rough_field.py, cos_pricer.py
- exotic_pricing.py: image propagator, ADI PDE solver
- response_propagator.py: response kernel contraction

## Critical Implementation Notes
- exp-OU: theta=3, mu=-2, T=1, rho=0; gamma=sigma^2/(2*theta) in [0.02, 0.6]
- SCG ansatz: psi = exp(A + B*z + 0.5*C*z^2), integrated backward from T to 0
- Tower benchmark: 100-step for 2PI methods, 400-step for continuous-time closures
- 2PI gap equations: m = -K[D(1) - M^(1)], G^{-1} = K^{-1} + diag(D(2)-M^(2)) - P
- Damping parameter for 2PI iteration: typically 0.3-0.5
- Rough Bergomi: N=200 time points, v_t = v0*exp(eta*W_hat_t - 0.5*eta^2*t^(2H))

## Testing Notes
- All tests in tests/ use pytest
- No live API calls in tests (all synthetic data)
- Test files must be independently runnable

## Rough Heston Implementation (rough_heston.py)
- Model: El Euch & Rosenbaum (2019); alpha = H + 0.5; H=0.1 → alpha=0.6
- g(u) = -(u² + iu)/2  [SIGN: minus, not plus — preserves E[S_T]=S_0 martingale]
- h_tilde in the log-CF formula is h(u,T) — endpoint value, NOT fractional integral of h
- Corrector: quadratic formula on A*h² + B*h - S = 0; select root with Re(h)≤0
- lam=0 (A=0) case: linear corrector h = S/B_base (branch handled separately)
- psi integral: use safe_kpi_c = where(k==0, 1.0, kpi_c) to avoid 0/0 warning
- N_cos=64, L=12 gives stable results; N_cos=128 gives no improvement for N=50
- Riccati discretisation error grows with T: ~7bp (T=0.5), ~27bp (T=1), ~84bp (T=2) for N=50 vs N=200

## SABR Stack Implementation (sabr_stack.py)
- Hagan formula breaks down for large T·nu and rho≠0 (errors >1000bp for T=10)
- Stack outperforms Hagan at large T because it averages over alpha_T distribution
- alpha0 normalisation: alpha0 = 0.2 * F0^(1-beta) so ATM CEV vol ≈ 20%
  (e.g. beta=0.5, F0=100 → alpha0 = 2.0, not 0.04!)
- GH quadrature: roots_hermitenorm(32), weights divided by sqrt(2*pi) for N(0,1)
- rho=0 cells: both methods accurate (<30bp); rho≠0: large errors at long T

## CEV Pricer Notes (absorbed_cev.py)
- Absorbed CEV IS a proper martingale: E[F_T]=F0 (NOT a sub-martingale)
- Negative skew for beta<1: IV(K_low) > IV(K_high), confirming correct Schroder formula
- bs_call: use safe_sqrtT=where(T>0, sqrtT, 1.0) to avoid divide-by-zero at T=0
- cev_call_lamperti: bridge Z0_eff can be negative (absorbed state) → returns 0

## Exotic Pricing Notes (exotic_pricing.py)
- exp-OU model: X_t follows OU, v_t = exp(X_t), spot via logS SDE, X initialized at mu
- One-touch: np.any(S_hist >= B, axis=0); RKO: np.all(S_hist < B, axis=0) * payoff
- FVA: accumulates exp(X)*dt only for t > T_start using scalar time counter
- black_call: returns float; vol<=0 or T<=0 → intrinsic
- Martingale check for exp-OU: E[S_T/S_0] ≈ 1 within 5% at n_paths=8000, n_steps=100

## Response Propagator Notes (response_propagator.py)
- OU kernel: exp(-theta*(t-s)) for t>s, 0 otherwise (strict causality)
- Rough kernel: eta*sqrt(2H)*(t-s)^(H-0.5) for t>s; singular but integrable for H<0.5
- compute_bucketed_vega_bump: uses price_func(ones) as base; bump[k] = 1+eps
- compute_bucketed_vega_response: sensitivity_func(t_grid) then trapz(R*dp_dv, t_grid)
- IMPORTANT: BAV uses multiplicative bumps (v_t → v_t*(1+eps)); RP local kernel must
  include v_t factor in sensitivity_func to match: sens(t) = dP/dv_t * v_t
- RP OU memory vegas DECREASE with bucket index (less residual horizon for late buckets)
- RP rough memory vegas LARGER than OU and decrease faster (singular front-loading)

## Rough Bergomi Exotic Notes (exp_9)
- N=252 daily steps; Cholesky of 252×252 fBm covariance matrix (expensive but exact)
- Forward-start smile: use log_ratio = logS_hist[-1] - logS_hist[N1]; empirical forward
  F_empirical = mean(exp(log_ratio)); use Black(F_empirical, K_ratio, IV, Delta)
- For H=0.1, rho=-0.7: extreme left skew; OTM forward-start calls near zero (NaN IV)
- RKO at H=0.1, rho=-0.7: typically zero (all paths below strike or knocked out)
- FVA roughly constant across H (depends on expected variance level, not roughness)

## Risk Engine Notes (exp_10)
- sigma_eff = sqrt(trapz(E[v_t], t_grid) / T) where E[v_t] = exp(mu+gamma*(1-exp(-2*theta*t)))
- BAV price_func builds modified v_t by bucket multipliers, recomputes sigma_eff, calls BS
- RP local-box matches BAV when sensitivity includes v_t factor (multiplicative bump interpretation)
- Runtime: RP ~10-35x faster than BAV; advantage grows linearly with n_buckets
- Rough kernel produces singular, front-loaded vegas vs OU exponential decay

## Rough Bergomi Implementation Notes (rough_bergomi_rqmc.py, rough_bergomi_closure.py)

### Martingale property (CRITICAL)
- `v_t = v0*exp(eta*W_t - 0.5*eta^2*t^{2H})` is a martingale: E[v_t] = v0 for ALL t.
- Therefore EV = E[∫ v_t dt] = v0*T  (NOT v0*∫exp(0.5*eta^2*t^{2H})dt — that is WRONG).
- E[v_t*v_s] = v0^2 * exp(eta^2 * K_fBm(t,s))  where K_fBm is the fBm covariance.
- VarV = dt^2 * v0^2 * Σ_{i,j} (exp(eta^2 * K_ij) - 1).
- These two facts are essential for the Gaussian closure moment formulas in rough_bergomi_closure.py.

### fBm increment scaling (CRITICAL for rho != 0)
- fBm increments ΔY_k = Y_{t_{k+1}} - Y_{t_k} have variance (dt)^{2H}, NOT dt.
- NEVER use ΔY_k / sqrt(dt) as a proxy for a standard BM increment — for H=0.1 this
  inflates variance by factor (dt)^{-0.8} ≈ 250 for dt=0.0025, making prices nonsensical.

### Causal Cholesky scheme for rho != 0
The correct treatment uses the Cholesky L of K_fBm:
  1. Compute epsilon = L^{-1} @ Y  →  epsilon_k ~ i.i.d. N(0,1)
  2. Causal field:  Y^causal_k = Y_k - L_kk * epsilon_k  (past-only, predictable)
  3. v^causal_k = v0 * exp(eta*Y^causal_k - 0.5*eta^2*t_k^{2H})
  4. J = sqrt(dt) * Σ_k sqrt(v^causal_k) * epsilon_k
  5. mu_cond = -V/2 + rho*J,  s2_cond = (1-rho^2)*V
This ensures E[S_T] = F0 exactly (verified numerically for rho in {-0.4,-0.7,-0.9}).

### Gaussian closure (rho=0 only)
- Fits lognormal to V distribution using exact first two moments (see above).
- GH quadrature (32 nodes) over log(V) to compute E[Black(F0, K, sqrt(V/T), T)].
- Accurate to ~3 bp for mild regime (H=0.2, eta=1.0); RQMC benchmark for rho!=0.

### RQMC sampling
- PCA-ordered Sobol in rough_field.sample_rough_field_sobol; n_scrambles=4 standard.
- Cholesky L computed once per call for the causal scheme; solve with np.linalg.solve.

## Rough SABR Implementation Notes (rough_sabr.py)

### Same causal Cholesky scheme applies
- Both rough_sabr_rqmc and rough_sabr_mc must use the causal Cholesky (Y^causal for alpha).
- rough_sabr_mc additionally generates independent N(0,dt) RVs for the F-innovation.
- Euler for F: F_{k+1} = max(F_k + alpha^causal_k * F_k^beta * dW_F_k, 0).
- alpha0 normalization: alpha0 = 0.2 * F0^{1-beta} so ATM vol ≈ 20% for any beta.

### Absorbed CEV Lamperti connection
- For the RQMC, prices conditional on V_Z via Gaussian Lamperti bridge.
- Z_0 = F0^{betac}/betac where betac = 1 - beta (Lamperti transform).
- Z0_eff = Z0 + rho*J displaces the bridge mean; var_Z = (1-rho^2)*V_Z.

## Experiment Results
- exp_3: results/exp3_sabr_results.csv, exp3_sabr_plot.png, exp3_sabr_rmse.png
- exp_4: results/exp4_rough_heston.csv, exp4_rough_heston.png
- exp_5: results/exp5_rough_bergomi.csv, exp5_rough_bergomi.png
  - Gaussian closure ignores rho → 130-600bp error vs RQMC ~1bp (expected)
- exp_6: results/exp6_rbergomi_rho0.csv, exp6_rbergomi_rho0.png
  - rho=0 Gaussian cumulant outperforms 2PI Hartree at these params
- exp_7: results/exp7_rough_sabr.csv, exp7_rough_sabr.png
  - 48 cells: median RMSE ~107bp (RQMC 1024 paths vs MC benchmark 4096 paths)
- exp_8: results/exp8_expou_exotics.csv, exp8_expou_exotics.png
- exp_9: results/exp9_rbergomi_exotics.csv, exp9_rbergomi_exotics.png
- exp_10: results/exp10_risk_engine.csv, exp10_risk_engine_runtime.csv, exp10_risk_engine.png
