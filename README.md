# SABR and CEV-Heston Option Pricing

*Martingale-preserving Monte Carlo pricing of European options under SABR and CEV-Heston, built on Choi-Hu-Kwok (2024) and PyFENG. MATH 5030 final project.*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PXYE-1029/SABR-Option-Pricing-Conditional-MC/blob/main/notebooks/demo.ipynb)

## What this project does

The project develops two Monte Carlo pricers for European options under stochastic volatility:

- **Phase 1** — SABR (`beta = 1`) with plain Monte Carlo and conditional Monte Carlo, demonstrating textbook variance reduction by conditioning on the volatility path.
- **Phase 2** — **CEV-Heston** (a hybrid where the asset has CEV elasticity `beta` and the variance follows the Heston / CIR process), implementing the Choi-Hu-Kwok (2024) three-step framework adapted to CIR variance. The Phase 2 simulator preserves the martingale property `E[F_T] = F_0` exactly, with no empirical martingale correction.

CEV-Heston dynamics:

$$
dF_t / F_t^{\beta} = \sqrt{v_t}\,dW^F_t,\qquad dv_t = \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW^v_t,\qquad \mathrm{corr}(dW^F, dW^v) = \rho.
$$


## Method

The Phase 2 CEV-Heston simulator advances \((F_t, v_t)\) over each time step \([t,t+h]\) using a three-step framework.

### 1. Variance Step

First, we sample the next variance \(v_{t+h}\) and the integrated variance over the step,

```math
I_t^h = \int_t^{t+h} v_s\,ds.
```

This step uses PyFENG's Heston Poisson-conditioning implementation, `HestonMcChoiKwok2023PoisGe`, as the variance-side backend. When PyFENG is unavailable or when \(\xi = 0\), an Andersen QE-style fallback is used.

### 2. Conditional Mean Step

Given \(v_t\), \(v_{t+h}\), and \(I_t^h\), we compute the conditional mean adjustment for the forward-price step.

The key identity comes from the CIR variance process:

```math
\int_t^{t+h}\sqrt{v_s}\,dW_s^v
=
\frac{
v_{t+h}-v_t-\kappa\theta h+\kappa I_t^h
}{\xi}.
```

This identity allows the correlated variance Brownian integral to be expressed using quantities already sampled in the variance step. As a result, no additional random draw is needed for this correlated component.

This conditional-mean adjustment is the main mathematical modification needed when adapting the SABR/CEV simulation idea to the CEV-Heston setting.

### 3. Conditional CEV Sampling Step

After the conditional mean is computed, the next forward price \(F_{t+h}\) is sampled from a CEV-style conditional distribution.

The implementation uses an exact CEV sampler based on the shifted-Poisson-mixture-Gamma representation of Makarov-Glew (2010) and Kang (2014). This avoids inverse-CDF root finding and replaces a naive full-path Euler update with a structured conditional sampling step.

### Baseline Comparison

For comparison, we also implement the classical Islah (2009) approximation as a baseline. This allows us to evaluate the proposed martingale-preserving CEV-style simulation against a standard approximation method.

### Efficiency Goal

Overall, the method combines:

- Heston-based variance and integrated-variance simulation,
- a conditional mean adjustment using the CIR identity,
- and exact CEV sampling for the asset-price step.

The goal is to improve martingale preservation and the accuracy-efficiency tradeoff relative to simpler simulation baselines.

Implementation details and numerical routines are provided in `src/`.
## Repository structure

```text
src/
  utils.py                        # parameter dataclasses & helpers
  black_scholes.py / sabr_simulation.py / mc_pricer.py / conditional_mc.py / integration.py
                                  # Phase 1 SABR (beta = 1)
  cev_sampling.py                 # Phase 2: exact CEV sampler
  cir_simulation.py               # Phase 2: Andersen QE fallback
  pyfeng_adapter.py               # Phase 2: PyFENG variance stepper
  heston_cev_simulation.py        # Phase 2: main CEV-Heston simulator
  islah_approximation.py          # Phase 2: Islah baseline
  heston_cev_benchmark.py         # Phase 2: Fourier / self-reference benchmarks
experiments/                      # 9 runnable experiment scripts
results/{tables,figures}/         # CSV tables and PNG figures produced by experiments
tests/                            # 32 unit tests
report/references.md              # annotated bibliography
```

## Installation

```bash
git clone https://github.com/PXYE-1029/SABR-Option-Pricing-Conditional-MC.git
cd SABR-Option-Pricing-Conditional-MC
python3 -m pip install -e ".[dev]"
python3 -m pytest tests/ -v
```

Runtime dependencies: `numpy`, `scipy`, `matplotlib`, `pyfeng`. PyFENG is required for the Phase 2 production path; the Phase 1 SABR side and the offline test suite work without it.

## Quick start

### Phase 1: SABR

```python
from src.utils import SABRModelParameters, EuropeanOption
from src.conditional_mc import price_european_option_conditional_mc

params = SABRModelParameters(
    spot=100.0, initial_volatility=0.2, beta=1.0,
    vol_of_vol=0.4, correlation=-0.3, risk_free_rate=0.01,
)
option = EuropeanOption(strike=100.0, maturity=1.0, option_type="call")
result = price_european_option_conditional_mc(params, option, n_steps=50, n_paths=5000, seed=123)
print(f"{result.price:.4f} +/- {result.standard_error:.4f}")
```

### Phase 2: CEV-Heston

```python
from src.utils import CEVHestonModelParameters, EuropeanOption
from src.heston_cev_simulation import price_european_option_heston_cev

params = CEVHestonModelParameters(
    spot=100.0, initial_variance=0.04, kappa=1.5, theta=0.04,
    xi=0.4, correlation=-0.5, beta=0.5, risk_free_rate=0.0,
)
option = EuropeanOption(strike=100.0, maturity=1.0, option_type="call")
result = price_european_option_heston_cev(params, option, n_steps=20, n_paths=50_000, seed=2026)
print(f"{result.price:.4f} +/- {result.standard_error:.4f}")
print(f"Variance backend: {result.simulation.diagnostics['variance_backend']}")
```

## Running the experiments

```bash
# Phase 1
python3 experiments/experiment_variance.py
python3 experiments/experiment_runtime.py
python3 experiments/experiment_timestep.py
python3 experiments/experiment_validation_bs_limit.py
python3 experiments/experiment_parameter_sweep_nu.py

# Phase 2
python3 experiments/experiment_heston_cev_martingale.py        # martingale preservation
python3 experiments/experiment_heston_cev_option_price.py      # option price vs benchmark
python3 experiments/experiment_heston_cev_speed.py             # speed-vs-RMS trade-off
```

Each script saves a CSV under `results/tables/` and PNGs under `results/figures/`.

## Headline results

### Phase 1 (SABR `beta = 1`)

Conditional MC reduces variance by **42-46x** versus plain MC across all tested path counts, while running slightly faster at large `N`. At `N = 50,000` the plain-MC standard error is `0.059` and the conditional-MC standard error is `0.009`.

![Standard error: plain MC vs conditional MC across path counts](results/figures/variance_standard_errors_beta1_call.png)

### Phase 2 (CEV-Heston)

**Martingale preservation.** At `T = 10` the raw simulator (no EMC) gives `|E[F_T] - F_0| = 0.029`, about one Monte Carlo standard error. The forward-error curve stays inside the analytic `+/- 2 SE` band across all maturities `T = 1, ..., 10`, with no systematic drift.

![Forward-price error vs maturity, inside the +/- 2 SE Monte Carlo band](results/figures/heston_cev_martingale_error.png)

**Option price accuracy.** Across `T = 1, ..., 10`, the ATM call price tracks a high-resolution self-reference benchmark within Monte Carlo noise. The Islah-baseline simulator is shown for direct comparison.

![ATM call pricing error vs maturity: ours vs Islah baseline](results/figures/heston_cev_option_price_error.png)

**Speed-vs-RMS trade-off.** Sweeping `N` and `dt` jointly at `T = 4`, the project's CEV scheme dominates Islah at fixed work as the grid is refined: Islah's RMS plateaus where its frozen-coefficient bias does not vanish, while ours continues to track Monte Carlo noise downward.

![RMS error vs CPU time: ours vs Islah vs truncated Euler baseline](results/figures/heston_cev_speed_rms_tradeoff.png)

All 32 unit tests pass on Python 3.10 / 3.11 / 3.12. See `src/` codes for full numerical results and methodology.

## References

The most central works (full annotated list in `report/references.md`):

- Hagan et al. (2002) — original SABR specification
- **Choi, Hu, Kwok (2024)** — the SABR simulation paper this project ports to CEV-Heston
- **Choi, Kwok (2024)** — the Heston Poisson-conditioning scheme used via PyFENG
- Islah (2009) — the classical baseline our scheme corrects
- Makarov-Glew (2010), Kang (2014) — exact CEV sampling
- Andersen (2008) — QE scheme for the offline / `xi = 0` fallback
- Glasserman (2003), Black-Scholes (1973) — standard background
- MATH 5030 (Numerical Methods) course materials
