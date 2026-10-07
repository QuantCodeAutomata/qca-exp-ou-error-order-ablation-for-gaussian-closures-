# exp-OU Error-Order Ablation for Gaussian Closures and 2PI Skeleton Hierarchy

This repository implements numerical experiments from a paper on non-perturbative
stochastic-volatility pricing using field-theoretic closures. It covers:

- **exp_1**: exp-OU error-order ablation — Gaussian closures vs Taylor truncation
- **exp_2**: exp-OU smile accuracy across maturities, correlations, and couplings
- **exp_3**: SABR vanilla pricing — routed deterministic stack and jump closure
- **exp_4**: rough Heston control experiment for Volterra numerics validation
- **exp_5**: rough Bergomi vanilla pricing — conditional-field RQMC + composite closure
- **exp_6**: rough Bergomi rho=0 evaluated skeleton ladder on native Volterra field
- **exp_7**: rough SABR vanilla pricing — conditional-field RQMC + jump closure
- **exp_8**: exp-OU exotic pricing with Markovian correlation-response machinery
- **exp_9**: rough Bergomi exotic pricing and forward-start smiles
- **exp_10**: Response-propagator latent-factor risk engine validation

## Repository Structure

```
.
├── data/              # Data loading utilities (synthetic data only)
├── src/               # Core implementation modules
│   ├── model_dynamics.py        # exp-OU, SABR, rough Heston, rough Bergomi, rough SABR
│   ├── tower_benchmark.py       # Exact tower recursion/PDE benchmark
│   ├── cos_pricer.py            # COS method for Fourier inversion
│   ├── scg_solver.py            # SCG ODE solver (log-quadratic ansatz)
│   ├── two_pi_solver.py         # 2PI Hartree and melon gap-equation solver
│   ├── rough_field.py           # Rough Gaussian field covariance and sampling
│   ├── absorbed_cev.py          # Exact absorbed-CEV Green's function
│   ├── sabr_stack.py            # SABR routed deterministic stack
│   ├── sabr_jump_closure.py     # SABR jump closure (Chebyshev–Lobatto)
│   ├── rough_bergomi_rqmc.py    # rough Bergomi conditional-field RQMC
│   ├── rough_bergomi_closure.py # rough Bergomi composite closure
│   ├── rough_sabr.py            # rough SABR methods
│   ├── exotic_pricing.py        # Exotic option pricing (barriers, FVA)
│   ├── response_propagator.py   # Response-propagator risk engine
│   └── metrics.py               # Pricing metrics, RMSE, implied vol utilities
├── exp/               # Experiment scripts
│   ├── exp_1_error_order.py
│   ├── exp_2_smile_accuracy.py
│   ├── exp_3_sabr.py
│   ├── exp_4_rough_heston.py
│   ├── exp_5_rough_bergomi.py
│   ├── exp_6_rough_bergomi_rho0.py
│   ├── exp_7_rough_sabr.py
│   ├── exp_8_expou_exotics.py
│   ├── exp_9_rbergomi_exotics.py
│   └── exp_10_risk_engine.py
├── tests/             # pytest tests
│   ├── test_model_dynamics.py
│   ├── test_tower_benchmark.py
│   ├── test_cos_pricer.py
│   ├── test_scg_solver.py
│   ├── test_two_pi_solver.py
│   ├── test_rough_field.py
│   ├── test_absorbed_cev.py
│   ├── test_sabr.py
│   ├── test_metrics.py
│   └── test_integration.py
├── results/           # Output artifacts
│   └── RESULTS.md
├── AGENTS.md          # Persistent memory for OpenHands agent
├── README.md
└── .gitignore
```

## Setup

```bash
pip install numpy scipy pandas matplotlib seaborn quantstats pyfeng arch hmmlearn
```

## Running Experiments

```bash
# Individual experiment
python exp/exp_1_error_order.py

# All experiments
for f in exp/exp_*.py; do python "$f"; done

# Tests
pytest tests/ -v
```

## Results

See `results/RESULTS.md` for a summary of all experiment outcomes and figures.
