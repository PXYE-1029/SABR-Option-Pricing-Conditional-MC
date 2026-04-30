"""Martingale sanity tests for the CEV-Heston simulator.

Verifies E[F_T] = F_0 across four scenarios that exercise different
code paths in heston_cev_simulation.simulate_heston_cev_terminal:

  (a) xi = 0   : deterministic-variance CEV limit (CIR collapses to
                 a deterministic ODE; the stochastic-integral term in
                 the conditional mean must vanish, not divide by zero).
  (b) rho = 0  : drift / convexity terms are zero by construction;
                 the conditional law is exact CEV.
  (c) beta = 1 : standard Heston with the lognormal closed-form asset
                 update branch.
  (d) full case: beta = 0.5, rho = -0.5, xi = 0.4 -- the realistic
                 stress-test parameters used by the project's main
                 martingale experiment.

For each scenario we run T = 1 and T = 10 and require the empirical
absolute error |E[F_T] - F_0| to fall within five Monte Carlo standard
errors of zero. We also verify that the clamp trigger rate stays
small (the safeguard should only fire on genuinely pathological
paths, not as a routine numerical crutch).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

from src.heston_cev_simulation import simulate_heston_cev_terminal
from src.utils import CEVHestonModelParameters

N_PATHS = 100_000
N_STEPS_PER_YEAR = 20
SE_TOLERANCE = 5.0
MAX_ACCEPTABLE_CLAMP_RATE = 0.01  # 1% of paths is the upper sanity bound


def _baseline(beta: float, xi: float, rho: float) -> CEVHestonModelParameters:
    return CEVHestonModelParameters(
        spot=100.0,
        initial_variance=0.04,
        kappa=1.5,
        theta=0.04,
        xi=xi,
        correlation=rho,
        beta=beta,
        risk_free_rate=0.0,
    )


def _martingale_check(
    parameters: CEVHestonModelParameters,
    maturity: float,
    seed: int,
) -> None:
    n_steps = max(int(round(maturity * N_STEPS_PER_YEAR)), 1)
    result = simulate_heston_cev_terminal(
        parameters=parameters,
        maturity=maturity,
        n_steps=n_steps,
        n_paths=N_PATHS,
        seed=seed,
    )
    samples = result.terminal_prices
    sample_mean = float(np.mean(samples))
    sample_se = float(np.std(samples, ddof=1) / np.sqrt(N_PATHS))
    error = abs(sample_mean - parameters.spot)

    assert error < SE_TOLERANCE * sample_se, (
        f"Martingale violated: |E[F_T] - F_0| = {error:.5f} "
        f"exceeds {SE_TOLERANCE} * SE = {SE_TOLERANCE * sample_se:.5f} "
        f"(F_T_mean={sample_mean:.5f}, T={maturity})"
    )
    assert result.clamp_trigger_rate < MAX_ACCEPTABLE_CLAMP_RATE, (
        f"Clamp triggered on {result.clamp_trigger_rate:.4%} of path-steps; "
        f"expected far below {MAX_ACCEPTABLE_CLAMP_RATE:.0%}"
    )


@pytest.mark.parametrize("maturity", [1.0, 10.0])
def test_martingale_xi_zero_deterministic_variance(maturity: float) -> None:
    """xi = 0: variance follows a deterministic ODE; no division by zero."""
    parameters = _baseline(beta=0.5, xi=0.0, rho=-0.5)
    _martingale_check(parameters, maturity=maturity, seed=int(2026 + maturity))


@pytest.mark.parametrize("maturity", [1.0, 10.0])
def test_martingale_rho_zero_exact_cev(maturity: float) -> None:
    """rho = 0: drift / convexity terms drop out; F is sampled exactly."""
    parameters = _baseline(beta=0.5, xi=0.4, rho=0.0)
    _martingale_check(parameters, maturity=maturity, seed=int(3026 + maturity))


@pytest.mark.parametrize("maturity", [1.0, 10.0])
def test_martingale_beta_one_standard_heston(maturity: float) -> None:
    """beta = 1: lognormal asset branch (no CEV sampler call)."""
    parameters = _baseline(beta=1.0, xi=0.4, rho=-0.5)
    _martingale_check(parameters, maturity=maturity, seed=int(4026 + maturity))


@pytest.mark.parametrize("maturity", [1.0, 10.0])
def test_martingale_full_case(maturity: float) -> None:
    """Realistic stress test: beta = 0.5, rho = -0.5, xi = 0.4."""
    parameters = _baseline(beta=0.5, xi=0.4, rho=-0.5)
    _martingale_check(parameters, maturity=maturity, seed=int(5026 + maturity))
