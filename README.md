# SABR Option Pricing with Conditional Monte Carlo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PXYE-1029/SABR-Option-Pricing-Conditional-MC/blob/main/notebooks/demo.ipynb)

Numerical option pricing in Python for a MATH 5030 final project, centered on SABR conditional Monte Carlo and a prototype Beta-Heston extension.

## Project Overview

This project started as a numerical study of European call pricing under the SABR stochastic volatility model. The main completed implementation is a `beta = 1` SABR pricing framework that compares plain Monte Carlo with conditional Monte Carlo and measures the gains in variance reduction, runtime, and numerical stability.

After professor feedback, we added a second direction: a Beta-Heston extension. This replaces the usual geometric-Brownian-motion asset base in Heston with a SABR/CEV-style `S_t^beta` diffusion. The current implementation of that extension uses an Euler variance backend, while this feature branch also contains a further experimental PyFENG-backed variance / integrated-variance backend.

### Instructor Feedback and Project Extension

The original project topic was SABR `beta = 1` option pricing with conditional Monte Carlo. After instructor feedback, we extended the project by combining a Heston variance process with a SABR/CEV-style beta-power asset process, which led to the current Beta-Heston prototype and the experimental PyFENG-backed branch work.

## Part I: SABR Conditional Monte Carlo

### Model Setup

The SABR model used in this project is

$$
\begin{aligned}
dS_t &= \sigma_t S_t^\beta\, dW_t,\\
d\sigma_t &= \nu \sigma_t\, dZ_t,\\
dW_t\, dZ_t &= \rho\, dt.
\end{aligned}
$$

The implemented SABR pricing layer is restricted to `beta = 1`, so the asset process becomes a lognormal stochastic-volatility model. The main inputs are:

- `S0`: initial asset price
- `sigma0`: initial volatility
- `nu`: volatility of volatility
- `rho`: correlation
- `r`: risk-free rate

### Numerical Methods

The SABR portion of the project implements:

- plain Monte Carlo, which simulates volatility paths and full asset paths directly
- conditional Monte Carlo, which conditions on the volatility path and prices each path with a conditional Black-Scholes formula
- trapezoidal and Simpson approximations for the integrated variance

For `beta = 1`, the conditional representation uses the integrated variance

$$
V_T = \int_0^T \sigma_t^2\,dt.
$$

and turns the terminal pricing problem into a pathwise Black-Scholes call evaluation with much lower variance than direct payoff simulation.

### Main SABR Result

The main conclusion is clear: conditional Monte Carlo reduces standard error substantially relative to plain Monte Carlo.

At `50,000` paths in the saved variance benchmark:

- plain MC standard error: `0.0591`
- conditional MC standard error: `0.00875`
- variance reduction ratio: `45.53`

In the runtime benchmark at the same path count, conditional Monte Carlo is also slightly faster in the current implementation.

## Part II: Beta-Heston Extension

Professor feedback suggested extending the Heston model by replacing the standard GBM asset base with a SABR-style `S_t^beta` process.

The model studied in this extension is

$$
\begin{aligned}
dv_t &= \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dZ_t,\\
dS_t &= rS_t\,dt + \sqrt{v_t}S_t^\beta\,dW_t,\\
dW_t\,dZ_t &= \rho\,dt.
\end{aligned}
$$

### Current Beta-Heston Prototype

The current Beta-Heston implementation in this repository is a lightweight prototype:

- Heston variance paths are simulated with full-truncation Euler
- the asset process uses `dS_t = r S_t dt + sqrt(v_t) S_t^beta dW_t`
- the conditional asset step uses a corrected Heston-specific conditional mean
- for `beta = 1`, the conditional step reduces to a lognormal form
- for `0 < beta < 1`, the asset step uses a prototype CEV-style approximation

The key Heston-specific correction is

$$
A_{\text{step}}
=
\frac{
v_{t+h} - v_t - \kappa\theta h + \kappa I_{\text{step}}
}{\xi},
\qquad
I_{\text{step}} \approx \int_t^{t+h} v_s\,ds.
$$

This replaces directly copying the SABR volatility correction and instead uses the Heston variance identity appropriate for this model.

This Beta-Heston layer is intentionally presented as a prototype rather than a finished pricing method.

### Experimental PyFENG Backend

This feature branch also includes an experimental PyFENG-backed extension.

- this work lives on `feature/option-b-pyfeng-backend`
- PyFENG is used only for the Heston variance / integrated-variance simulation
- the custom `S_t^beta` asset layer remains in this repository
- the default backend is still Euler
- PyFENG is optional and is not required for the main project

The PyFENG-backed backend currently uses `HestonMcChoiKwok2023PoisTd` to generate Heston variance and average variance step information, which is then fed into the project’s own Beta-Heston conditional asset layer.

This should be treated as experimental branch work, not as the default project implementation.

## Numerical Results

### SABR

The strongest completed result in the project is the SABR variance reduction result:

- conditional Monte Carlo reduces standard error by about a factor of `6.7`
- the pathwise variance reduction ratio is about `45.5`

### Current Beta-Heston Prototype

For the Beta-Heston prototype comparison, the corrected conditional approximation stays reasonably close to the Euler baseline across the tested beta values:

- `beta = 1.0`: absolute price difference `0.084`
- `beta = 0.7`: absolute price difference `0.022`
- `beta = 0.5`: absolute price difference `0.009`

This is useful as a first consistency check, but the `beta < 1` asset step is still approximate.

### Experimental PyFENG Backend

Because this branch contains the experimental PyFENG backend comparison, we also report the current backend comparison result.

The important interpretation is not that PyFENG is already “better,” but that changing the variance / integrated-variance backend can move prices even when the asset-side approximation is kept fixed.

In the current saved run:

- at `beta = 1.0`, the Euler and PyFENG prices differ by about `0.243`
- at `beta = 0.5`, the two prices are nearly identical, differing by about `0.00021`
- the comparison CSV also reports average integrated variance under each backend to help diagnose these differences

## Repository Structure

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── LICENSE
├── src/
│   ├── utils.py
│   ├── black_scholes.py
│   ├── integration.py
│   ├── sabr_simulation.py
│   ├── mc_pricer.py
│   ├── conditional_mc.py
│   └── beta_heston_simulation.py
├── experiments/
│   ├── experiment_runtime.py
│   ├── experiment_variance.py
│   ├── experiment_timestep.py
│   ├── experiment_validation_bs_limit.py
│   ├── experiment_parameter_sweep_nu.py
│   ├── experiment_beta_heston_prototype.py
│   └── experiment_beta_heston_backend_comparison.py
├── results/
│   ├── tables/
│   └── figures/
├── notebooks/
│   └── demo.ipynb
├── tests/
│   ├── test_beta_heston_simulation.py
│   ├── test_black_scholes.py
│   ├── test_conditional_mc.py
│   └── test_integration.py
└── report/
    └── references.md
```

## Installation and Quick Start

Install the standard project dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Install in editable mode for development and testing:

```bash
python3 -m pip install -e ".[dev]"
```

Because this branch includes the experimental PyFENG backend, the optional PyFENG extra can also be installed:

```bash
python3 -m pip install -e ".[pyfeng]"
```

### Quick Start Example

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

## How to Run Experiments

Run the core SABR experiments:

```bash
python3 experiments/experiment_variance.py
python3 experiments/experiment_runtime.py
python3 experiments/experiment_timestep.py
python3 experiments/experiment_validation_bs_limit.py
python3 experiments/experiment_parameter_sweep_nu.py
```

Run the Beta-Heston prototype experiment:

```bash
python3 experiments/experiment_beta_heston_prototype.py
```

Run the experimental PyFENG backend comparison in this branch:

```bash
python3 experiments/experiment_beta_heston_backend_comparison.py
```

## Tests and CI

The repository includes:

- local tests with `pytest`
- GitHub Actions CI
- tested Python versions: `3.10`, `3.11`, and `3.12`

Run the tests locally with:

```bash
python3 -m pytest
```

## Current Limitations

- The main SABR implementation currently supports `beta = 1` only.
- The main pricing layer currently supports European calls only.
- The Beta-Heston extension is still a prototype rather than a finished second project.
- The `beta < 1` CEV-style asset step is still approximate.
- In this branch, the PyFENG backend is optional and experimental, not the default.
- The PyFENG path changes only the variance / integrated-variance backend; the `S_t^beta` asset layer remains custom.
- PyFENG also introduces GPL licensing considerations, so this optional backend should be treated carefully before broader distribution.

## References

- Hagan, P. S., Kumar, D., Lesniewski, A. S., and Woodward, D. E. (2002). *Managing Smile Risk*.
- Black, F., and Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities*.
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*.
- Choi, J., and Kwok, Y. K. (2023). *Simulation Schemes for the Heston Model with Poisson Conditioning*.
- SABR-style `S_t^beta` conditional simulation ideas discussed in professor feedback and related SABR/CEV literature.
- Course materials for **MATH 5030 (Numerical Methods)**.
