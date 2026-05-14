"""Sanity tests for the Andersen QE CIR variance simulator.

Verifies two structural properties of the bundled Andersen QE
stepper that any correct CIR sampler must satisfy:

  * Mean reversion: E[v_T | v_0] = theta + (v_0 - theta) exp(-kappa T),
    the closed-form mean of a CIR transition. We check this at
    T = 1 and at T = 5.
  * Non-negativity: under the QE scheme, v_T should never go below
    zero (the absorbing boundary is enforced inside the stepper).

These tests run without network access (no PyFENG required) and are
the only Heston-side tests that can validate the variance simulator
in isolation.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.cir_simulation import AndersenQEVarianceSimulator
from src.utils import CEVHestonModelParameters


N_PATHS = 200_000
SE_TOLERANCE = 5.0


def _build_parameters() -> CEVHestonModelParameters:
    return CEVHestonModelParameters(
        spot=100.0,
        initial_variance=0.04,
        kappa=1.5,
        theta=0.04,
        xi=0.4,
        correlation=-0.5,
        beta=0.5,
        risk_free_rate=0.0,
    )


def _walk_variance(
    parameters: CEVHestonModelParameters,
    maturity: float,
    n_steps: int,
    seed: int,
) -> np.ndarray:
    """Run the QE stepper for ``n_steps`` steps and return ``v_T``."""
    rng = np.random.default_rng(seed)
    stepper = AndersenQEVarianceSimulator(parameters=parameters, rng=rng)
    dt = maturity / n_steps
    v = np.full(N_PATHS, parameters.initial_variance, dtype=float)
    for _ in range(n_steps):
        v, _ = stepper.step(dt, v)
    return v


@pytest.mark.parametrize("maturity", [1.0, 5.0])
def test_qe_mean_reversion_matches_closed_form(maturity: float) -> None:
    """E[v_T] should match theta + (v_0 - theta) exp(-kappa T)."""
    parameters = _build_parameters()
    v_T = _walk_variance(parameters, maturity=maturity, n_steps=int(maturity * 50), seed=11)

    expected_mean = parameters.theta + (
        parameters.initial_variance - parameters.theta
    ) * np.exp(-parameters.kappa * maturity)
    sample_mean = float(np.mean(v_T))
    sample_se = float(np.std(v_T, ddof=1) / np.sqrt(N_PATHS))
    error = abs(sample_mean - expected_mean)

    assert error < SE_TOLERANCE * sample_se, (
        f"QE mean reversion violated: |E[v_T] - exact| = {error:.6f} "
        f"exceeds {SE_TOLERANCE} * SE = {SE_TOLERANCE * sample_se:.6f} "
        f"(sample_mean={sample_mean:.6f}, expected={expected_mean:.6f}, "
        f"T={maturity})"
    )


@pytest.mark.parametrize("maturity", [1.0, 5.0])
def test_qe_variance_paths_are_non_negative(maturity: float) -> None:
    """The QE scheme enforces v >= 0 at every step by construction."""
    parameters = _build_parameters()
    v_T = _walk_variance(parameters, maturity=maturity, n_steps=int(maturity * 50), seed=22)

    assert np.all(v_T >= 0.0), (
        f"QE produced negative variance: min(v_T) = {np.min(v_T):.6e}"
    )


def test_qe_xi_zero_collapses_to_deterministic_ode() -> None:
    """With xi = 0 the variance follows an exact ODE."""
    parameters = CEVHestonModelParameters(
        spot=100.0,
        initial_variance=0.10,
        kappa=2.0,
        theta=0.04,
        xi=0.0,
        correlation=-0.5,
        beta=0.5,
        risk_free_rate=0.0,
    )
    maturity = 2.0
    n_steps = 50
    v_T = _walk_variance(parameters, maturity=maturity, n_steps=n_steps, seed=33)

    # All paths should be identical (deterministic) and equal to the ODE solution.
    expected = parameters.theta + (parameters.initial_variance - parameters.theta) * np.exp(
        -parameters.kappa * maturity
    )
    assert np.std(v_T) < 1e-12, "Variance is not deterministic when xi = 0"
    assert abs(float(np.mean(v_T)) - expected) < 1e-10, (
        f"QE deterministic limit drifted: mean={np.mean(v_T):.10f}, expected={expected:.10f}"
    )
