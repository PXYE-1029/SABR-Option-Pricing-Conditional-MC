# SABR Option Pricing with Conditional Monte Carlo

*A Python implementation of SABR model option pricing with Euler simulation, conditional Monte Carlo, integrated variance approximation, and efficiency benchmarking.*

## Project Overview

This project studies numerical option pricing under the SABR stochastic volatility model. We implement both plain Monte Carlo and conditional Monte Carlo methods for European option pricing and compare them in terms of pricing accuracy, estimator variance, and runtime.

Our main goal is to improve computational efficiency while maintaining pricing accuracy. To achieve this, we simulate the stochastic volatility process, approximate the integrated variance numerically, and then use conditional Black-Scholes pricing whenever possible. We compare the methods in terms of pricing accuracy, standard error, runtime, and robustness across different parameter settings.

This repository is being developed as a final project for **MATH 5030 (Numerical Methods)**.

## Main Features

- SABR model simulation in Python
- Plain Monte Carlo pricer
- Conditional Monte Carlo pricer
- Numerical integration of integrated variance
- Runtime and variance benchmarking
- Comparison across parameter settings

## Model Setup

We consider the SABR stochastic volatility model:

$$
dS_t = \sigma_t S_t^\beta \, dW_t
$$

$$
d\sigma_t = \nu \sigma_t \, dZ_t
$$

with correlation

$$
dW_t dZ_t = \rho \, dt
$$

where:

- $S_t$ is the asset price
- $\sigma_t$ is the stochastic volatility
- $\beta \in [0,1]$ controls the backbone
- $\nu$ is the volatility of volatility
- $\rho$ is the correlation between asset and volatility shocks

## Numerical Methods

We implement the following numerical methods:

### 1. Plain Monte Carlo

We simulate the SABR paths directly using time discretization and estimate the option price by averaging discounted payoffs.

### 2. Conditional Monte Carlo

Instead of simulating the terminal asset price directly, we condition on the volatility path and integrated variance, and then evaluate the option price using an analytical pricing formula. This reduces Monte Carlo variance and improves computational efficiency.

### 3. Integrated Variance Approximation

We numerically approximate the integrated variance using:

- Trapezoidal rule
- Simpson's rule

### 4. Variance Reduction

We investigate variance reduction through conditional Monte Carlo and, if helpful, additional techniques such as antithetic sampling.

## Repository Structure

The following is the **planned repository structure** for the project:

```text
.
├── README.md
├── requirements.txt
├── src/
│   ├── sabr_simulation.py
│   ├── mc_pricer.py
│   ├── conditional_mc.py
│   ├── integration.py
│   ├── black_scholes.py
│   └── utils.py
├── experiments/
│   ├── experiment_runtime.py
│   ├── experiment_variance.py
│   ├── experiment_timestep.py
│   └── experiment_hagan_comparison.py
├── notebooks/
│   └── demo.ipynb
├── results/
│   ├── figures/
│   └── tables/
└── report/
    └── references.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/your-repo-name.git
cd your-repo-name
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

Example commands for the planned experiment scripts:

Run the plain Monte Carlo pricer:

```bash
python experiments/experiment_runtime.py
```

Run the conditional Monte Carlo experiment:

```bash
python experiments/experiment_variance.py
```

Run the timestep convergence experiment:

```bash
python experiments/experiment_timestep.py
```

These commands will be updated as the codebase is implemented.

## Results Summary

Our experiments compare plain Monte Carlo and conditional Monte Carlo under the SABR model.

Main findings we expect to investigate:

- Conditional Monte Carlo reduces estimator variance significantly relative to plain Monte Carlo.
- Numerical integration of the integrated variance using Simpson's rule is generally more accurate than the trapezoidal rule.
- Runtime increases with finer time discretization, but accuracy improves as expected.
- The efficiency gain from conditional Monte Carlo becomes more pronounced when a high level of precision is required.

## Validation and Benchmark

We validate our implementation by:

- comparing pricing outputs across multiple simulation methods
- checking convergence as the number of paths increases
- studying timestep sensitivity
- comparing results against Hagan's SABR approximation when applicable

## Team Contributions

This section will be updated once tasks are assigned among the three teammates. A possible split is:

- Member 1: SABR simulation and path generation
- Member 2: conditional Monte Carlo implementation
- Member 3: numerical integration, benchmarking, plotting, and README/report writing

## References

- Hagan, P. S., Kumar, D., Lesniewski, A. S., and Woodward, D. E. (2002). *Managing Smile Risk.*
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering.*
- Course materials from **MATH 5030 Numerical Methods**.
