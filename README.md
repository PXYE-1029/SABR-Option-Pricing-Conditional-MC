# SABR Option Pricing with Conditional Monte Carlo

*A Python implementation of SABR option pricing with plain Monte Carlo, conditional Monte Carlo, integrated variance approximation, and benchmarking for a MATH 5030 final project.*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PXYE-1029/SABR-Option-Pricing-Conditional-MC/blob/main/notebooks/demo.ipynb)

## Project Overview

This repository studies numerical pricing of European call options under the SABR stochastic volatility model. The project focuses on the beta = 1 case and compares two estimators:

- plain Monte Carlo, which simulates the full asset path directly
- conditional Monte Carlo, which simulates only the volatility path and then prices conditionally with a Black-Scholes formula

The main goal is to measure how much variance reduction and computational efficiency can be gained from conditioning on the volatility path. The repository includes reusable pricing modules, experiment scripts, saved benchmark tables, and figures suitable for a course project report.

The repository now also includes a **Beta-Heston Extension** prototype, added after feedback from the course instructor. This new direction keeps the Heston variance process but replaces the usual geometric-BM asset base with a SABR/CEV-style `S_t^beta` asset process.

## SABR Model Setup

We work with the SABR model:

$$
\begin{aligned}
dS_t &= \sigma_t S_t^{\beta}\, dW_t \\
d\sigma_t &= \nu\, \sigma_t\, dZ_t \\
dW_t\, dZ_t &= \rho\, dt
\end{aligned}
$$

In the current implementation we restrict attention to **beta = 1**, so the asset dynamics reduce to a stochastic-volatility lognormal model. The parameters are:

- `S0`: initial asset price
- `sigma0`: initial volatility
- `beta`: SABR backbone parameter
- `nu`: volatility of volatility
- `rho`: correlation between the asset and volatility shocks
- `r`: risk-free rate

For the asset-path simulation, we use the beta = 1 log-Euler update:

$$
\log S_{t+dt} = \log S_t + \left(r - \tfrac{1}{2}\sigma_t^2\right) dt + \sigma_t \sqrt{dt}\, Z
$$

with the left-endpoint volatility on each step. This preserves positivity of the simulated asset price.

## Methodology

Our project studies a **SABR-style extension of the Heston model**.

The variance process follows the Heston framework, while the asset process is modified by introducing a **beta-power term**. In this way, the model preserves the stochastic-variance structure of Heston while incorporating a CEV/SABR-style elasticity into the asset dynamics.

### Model Setup

The model is given by

$$
dS_t = \mu S_t^\beta dt + \sqrt{v_t}\, S_t^\beta dW_t^S
$$

$$
dv_t = \kappa(\theta - v_t)dt + \xi \sqrt{v_t}\, dW_t^v
$$

$$
dW_t^S dW_t^v = \rho\, dt
$$

Here, $v_t$ is the stochastic variance process, and the parameter $\beta$ controls the nonlinear elasticity of the asset-price diffusion.

A key quantity in the simulation is the **integrated variance**

$$
I_{0,t} = \int_0^t v_s\, ds
$$

which summarizes the cumulative variance over the time interval $[0,t]$.

### Simulation Framework

To simulate this model efficiently, we separate the problem into two components:

$$
\text{Simulation} =
\underbrace{\text{Variance step}}_{\{v_t,\ I_{0,t}\}}
+
\underbrace{\text{Conditional asset-price step}}_{\text{simulate } S_t \mid v_t,\ I_{0,t}}
$$

This decomposition is the core methodological idea of the project.

### Variance Side

On the variance side, we simulate both the variance process and the integrated variance. For this part, we rely on existing Heston simulation methods, especially implementations already available in **PyFENG**. This allows us to build on established variance simulation schemes rather than reconstructing the full variance module from scratch.

### Conditional Asset-Price Side

The main methodological focus of the project is on the asset-price simulation.

Instead of using a naive step-by-step Euler simulation for the full asset path, we adopt a **conditional simulation** idea inspired by Professor Choi’s recent work. Once the variance-related quantities are available, we simulate the terminal asset price conditionally rather than propagating the full path through many small noisy steps.

In particular, we use a **CEV-style conditional approximation** for the asset-price side. The purpose of this approximation is to preserve the key structural features of the model while improving simulation efficiency and numerical stability.

### Overall Simulation Logic

The full method can be summarized as follows:

- First simulate the **variance process** and the **integrated variance**
- Then simulate the **asset price conditionally** on those variance quantities
- Replace naive full-path simulation with a more structured conditional approximation

This design is intended to improve the **accuracy-efficiency tradeoff** of Monte Carlo pricing.

### Integrated Variance Approximation

The project implements two numerical approximations for

$$
V_T = \int_0^T \sigma_t^2\, dt
$$

- trapezoidal rule
- Simpson's rule

Both operate on the same path convention used throughout the codebase:

- path arrays have shape `(n_paths, n_steps + 1)`
- shock arrays have shape `(n_paths, n_steps)`

## Repository Structure

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml
├── LICENSE
├── README.md
├── pyproject.toml
├── requirements.txt
├── notebooks/
│   └── demo.ipynb
├── src/
│   ├── __init__.py
│   ├── beta_heston_simulation.py
│   ├── black_scholes.py
│   ├── conditional_mc.py
│   ├── integration.py
│   ├── mc_pricer.py
│   ├── sabr_simulation.py
│   └── utils.py
├── experiments/
│   ├── experiment_hagan_comparison.py
│   ├── experiment_beta_heston_prototype.py
│   ├── experiment_parameter_sweep_nu.py
│   ├── experiment_runtime.py
│   ├── experiment_timestep.py
│   ├── experiment_validation_bs_limit.py
│   └── experiment_variance.py
├── results/
│   ├── figures/
│   └── tables/
├── report/
│   └── references.md
└── tests/
    ├── test_beta_heston_simulation.py
    ├── test_black_scholes.py
    ├── test_conditional_mc.py
    └── test_integration.py
```

Key saved outputs:

- Beta-Heston prototype table: [results/tables/beta_heston_prototype_comparison.csv](results/tables/beta_heston_prototype_comparison.csv)
- variance benchmark table: [results/tables/variance_comparison_beta1_call.csv](results/tables/variance_comparison_beta1_call.csv)
- timestep benchmark table: [results/tables/timestep_sensitivity_beta1_call.csv](results/tables/timestep_sensitivity_beta1_call.csv)
- runtime benchmark table: [results/tables/runtime_benchmark_beta1_call.csv](results/tables/runtime_benchmark_beta1_call.csv)

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/PXYE-1029/SABR-Option-Pricing-Conditional-MC.git
cd SABR-Option-Pricing-Conditional-MC
python3 -m pip install -r requirements.txt
```

Install the project as a package:

```bash
python3 -m pip install .
```

Install in editable mode for development and testing:

```bash
python3 -m pip install -e ".[dev]"
```

Run the test suite:

```bash
python3 -m pytest
```

The core runtime dependencies are:

- `numpy`
- `scipy`
- `matplotlib`

The project also includes a GitHub Actions workflow in `.github/workflows/ci.yml`
that runs the tests on Python 3.10, 3.11, and 3.12.

## Quick Start

The package currently exposes its modules under the `src` import namespace.

```python
from src.utils import SABRModelParameters, EuropeanOption
from src.mc_pricer import price_european_option_mc
from src.conditional_mc import price_european_option_conditional_mc

parameters = SABRModelParameters(
    spot=100.0,
    initial_volatility=0.2,
    beta=1.0,
    vol_of_vol=0.4,
    correlation=-0.3,
    risk_free_rate=0.01,
)
option = EuropeanOption(strike=100.0, maturity=1.0, option_type="call")

mc_result = price_european_option_mc(
    parameters=parameters,
    option=option,
    n_steps=50,
    n_paths=5000,
    seed=123,
)
cmc_result = price_european_option_conditional_mc(
    parameters=parameters,
    option=option,
    n_steps=50,
    n_paths=5000,
    seed=123,
    integration_method="trapezoidal",
)

print(mc_result.price, mc_result.standard_error)
print(cmc_result.price, cmc_result.standard_error)
```

For a notebook version of this workflow, see [notebooks/demo.ipynb](notebooks/demo.ipynb).

## How to Run

Run the runtime benchmark:

```bash
python3 experiments/experiment_runtime.py
```

Run the variance comparison:

```bash
python3 experiments/experiment_variance.py
```

Run the Beta-Heston prototype experiment:

```bash
python3 experiments/experiment_beta_heston_prototype.py
```

Run the timestep sensitivity experiment:

```bash
python3 experiments/experiment_timestep.py
```

Run the dedicated Black-Scholes-limit validation experiment:

```bash
python3 experiments/experiment_validation_bs_limit.py
```

Run the vol-of-vol parameter sweep:

```bash
python3 experiments/experiment_parameter_sweep_nu.py
```

Each script:

- prints a short summary to the terminal
- saves a CSV table under `results/tables/`
- saves one or more figures under `results/figures/`

## API Reference

Core public modules:

- `src.utils`
  Exposes `SABRModelParameters` and `EuropeanOption`, along with shared helpers
  such as `get_rng`, `standard_error`, and `time_call`.
- `src.black_scholes`
  Provides `black_scholes_call_price` and `black_scholes_price` for scalar or
  vectorized Black-Scholes pricing.
- `src.sabr_simulation`
  Provides `simulate_volatility_paths` and `simulate_sabr_paths` for the beta = 1
  SABR simulation layer.
- `src.mc_pricer`
  Provides `price_european_option_mc` and returns `MonteCarloPricingResult`.
- `src.conditional_mc`
  Provides `price_european_option_conditional_mc` and returns
  `ConditionalMonteCarloPricingResult`.
- `src.integration`
  Provides `trapezoidal_rule`, `simpson_rule`,
  `trapezoidal_integrated_variance`, and `simpson_integrated_variance`.
- `src.beta_heston_simulation`
  Provides the separate Beta-Heston prototype layer, including
  `BetaHestonParameters`, Heston variance-path simulation, baseline Euler asset
  simulation, and the corrected conditional approximation layer.

## Beta-Heston Extension

Based on the professor's suggestion, the repository now includes a new project direction that combines:

- a standard Heston variance process
- a SABR/CEV-style asset process with `dS_t = sqrt(v_t) S_t^beta dW_t`

The current implementation is **Option A only**. It uses:

- simple full-truncation Euler simulation for the Heston variance path
- a baseline Euler simulation for the beta-power asset process
- a corrected Heston-specific conditional mean term
- an exact conditional lognormal step when `beta = 1`
- a prototype conditional CEV approximation fallback when `0 < beta < 1`

For this prototype, we adapt the SABR-style conditional formula by identifying `sigma_t = sqrt(v_t)` and using the Heston-specific term

- `A_step = (v_{t+h} - v_t - kappa theta h + kappa I_step) / xi`

with `I_step ≈ ∫_t^{t+h} v_s ds` from a trapezoidal step rule.

In the current environment, **PyFENG is not available**, so the `beta < 1` branch uses a clearly labeled local fallback rather than an exact CEV sampler.

The dedicated prototype experiment is:

- `experiments/experiment_beta_heston_prototype.py`

It currently compares beta values `1.0`, `0.7`, and `0.5`, and saves:

- `results/tables/beta_heston_prototype_comparison.csv`
- `results/figures/beta_heston_prototype_comparison.png`

Current limitations of the Beta-Heston extension:

- it does **not** yet use a PyFENG / Poisson-conditioned Heston variance simulation
- the `beta < 1` CEV layer is a prototype approximation rather than a finished exact transition scheme
- the more ambitious follow-up direction ("Option B") remains future work

### Additional Experiment Coverage

- `experiment_beta_heston_prototype.py` is the first professor-suggested extension prototype, combining a simple Heston variance path with a SABR-style beta-power asset process and a corrected conditional approximation baseline.
- `experiment_validation_bs_limit.py` isolates the `nu = 0` deterministic-volatility case and compares plain MC, conditional MC, and the exact Black-Scholes benchmark across several path counts.
- `experiment_parameter_sweep_nu.py` varies the SABR vol-of-vol parameter `nu` over a fixed grid to test whether the conditional Monte Carlo variance-reduction advantage persists away from the baseline parameter set.

## Results Summary

All current benchmarks use the beta = 1 SABR setup

- `S0 = 100`
- `sigma0 = 0.2`
- `K = 100`
- `T = 1`
- `r = 0.01`
- `nu = 0.4`
- `rho = -0.3`

unless a validation case explicitly sets `nu = 0`.

### Main Findings

- Conditional Monte Carlo consistently reduces standard error relative to plain Monte Carlo.
- In the variance benchmark, the variance reduction ratio stays around **42 to 46** across all tested path counts.
- At the largest variance benchmark (`50,000` paths), plain MC has standard error **0.0591**, while conditional MC has standard error **0.00875**, with a variance reduction ratio of **45.53**.
- In the timestep benchmark, the conditional estimator is much more stable across grid sizes than plain Monte Carlo.
- The trapezoidal and Simpson integrated-variance approximations are already very close at moderate timestep counts and become nearly indistinguishable on fine grids.
- In the runtime benchmark, conditional MC is not only lower variance but also slightly faster in the current implementation at large path counts, and it is dramatically better on the variance-times-runtime efficiency metric.

### Variance Benchmark

The most important variance result is that conditioning turns a high-variance payoff estimator into a much smoother pathwise pricing estimator.

At `50,000` paths:

- plain MC standard error: `0.059073`
- conditional MC standard error: `0.008754`
- variance reduction ratio: `45.534`

![Standard error comparison](results/figures/variance_standard_errors_beta1_call.png)

### Timestep Sensitivity

The timestep experiment compares:

- plain MC
- conditional MC with trapezoidal integration
- conditional MC with Simpson integration

On the finest tested grid (`n_steps = 200`):

- plain MC price: `8.469764`
- conditional MC with trapezoidal integration: `8.488667`
- conditional MC with Simpson integration: `8.488663`

The trapezoidal versus Simpson price difference decreases from about `1.30e-3` at `10` steps to about `4.18e-6` at `200` steps, which suggests that the conditional estimator is not very sensitive to the integration rule once the grid is reasonably fine.

![Timestep price comparison](results/figures/timestep_prices_beta1_call.png)

### Runtime and Efficiency

The runtime benchmark compares plain MC and conditional MC over the path grid

- `2,000`
- `5,000`
- `10,000`
- `20,000`
- `50,000`

At `50,000` paths:

- plain MC runtime: `0.291913` seconds
- conditional MC runtime: `0.181884` seconds
- plain MC variance-times-runtime: `5.123278e+01`
- conditional MC variance-times-runtime: `6.868466e-01`

So in the current implementation, conditional MC is both slightly faster and far more statistically efficient.

![Runtime efficiency benchmark](results/figures/runtime_efficiency_beta1_call.png)

## Validation

The key validation case is the deterministic-volatility limit:

- when `vol_of_vol = 0`
- and `beta = 1`

the model reduces to Black-Scholes with constant volatility `sigma0`.

This is an important consistency check for both pricers:

- in the conditional Monte Carlo implementation, the `nu = 0` branch reduces directly to the Black-Scholes call price
- in the plain Monte Carlo implementation, the estimator should converge to the same Black-Scholes benchmark as the number of paths increases

This gives a clean analytical sanity check for both the direct simulation and the conditional representation.

The repository now includes a dedicated validation script for this case:

- `experiments/experiment_validation_bs_limit.py`

It is intended to produce:

- `results/tables/validation_bs_limit_beta1_call.csv`
- `results/figures/validation_bs_limit_prices_beta1_call.png`
- `results/figures/validation_bs_limit_abs_error_beta1_call.png`

In addition, the repository now includes a dedicated vol-of-vol sweep:

- `experiments/experiment_parameter_sweep_nu.py`

This script is intended to produce:

- `results/tables/parameter_sweep_nu_beta1_call.csv`
- `results/figures/parameter_sweep_nu_standard_errors_beta1_call.png`
- `results/figures/parameter_sweep_nu_variance_ratio_beta1_call.png`

## Current Limitations

- The main SABR implementation currently supports **beta = 1 only**.
- The pricers currently support **European call options only**.
- The Beta-Heston extension is currently a separate prototype rather than a full replacement for the SABR project.
- The Beta-Heston prototype does **not** yet use PyFENG / Poisson-conditioned Heston variance simulation.
- The Beta-Heston `beta < 1` CEV layer currently uses a **prototype local fallback** rather than exact CEV sampling.
- The repository does **not** yet implement the Hagan closed-form comparison experiment.
- The current results are based on fixed benchmark grids rather than a full parameter sweep.

## References

See the fuller project reference list in [report/references.md](report/references.md).

Core references used in this repository include:

- Hagan, P. S., Kumar, D., Lesniewski, A. S., and Woodward, D. E. (2002). *Managing Smile Risk*.
- Black, F., and Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities*.
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*.
- Course materials from **MATH 5030 (Numerical Methods)**.
