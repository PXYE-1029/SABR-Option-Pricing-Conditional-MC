"""Sanity tests for the PyFENG adapter.

Verifies that the PyFENG variance stepper returns the correctly
*dimensional* integrated variance (i.e. ``int_t^{t+h} v_s ds``, not
the ``average variance`` or the dimensionless ``I_t^h`` of Choi
et al. 2024). This catches the most damaging failure mode we
encountered during development: silently confusing the
``cond_states_step`` return convention shrinks the conditional
mean's drift / convexity terms by a factor of ``dt`` and produces
a 50-percent martingale violation that is opaque to the standard
SABR-side debugging playbook.

These tests are skipped automatically when PyFENG is not
installed (e.g. in offline CI), so they impose no extra
dependency on environments that do not need them.
"""

from __future__ import annotations

import pytest

from src.utils import CEVHestonModelParameters


pyfeng = pytest.importorskip("pyfeng", reason="PyFENG is not installed")


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


def test_pyfeng_returns_dimensional_integrated_variance() -> None:
    """The adapter should return ``int v ds``, not the average variance."""
    from src.pyfeng_adapter import verify_pyfeng_integrated_variance_scale

    parameters = _build_parameters()
    diagnostic = verify_pyfeng_integrated_variance_scale(
        parameters=parameters,
        n_path=50_000,
        dt=0.1,
        rtol=0.05,
        seed=42,
    )

    assert diagnostic["looks_correctly_dimensional"], (
        f"PyFENG IV scale convention has changed. Diagnostic snapshot: {diagnostic}"
    )
    # The dimensional integral over a single dt should be roughly v_0 * dt.
    assert (
        abs(diagnostic["stepper_returns_dimensional_mean"]
            - diagnostic["expected_dimensional_integral"])
        < 0.05 * diagnostic["expected_dimensional_integral"]
    ), f"IV mean is far from v_0 * dt: {diagnostic}"


def test_pyfeng_v_next_mean_matches_cir_mean_reversion() -> None:
    """``v_{t+h}`` from PyFENG should match the closed-form CIR mean."""
    import numpy as np

    from src.pyfeng_adapter import PyFengVarianceStepper

    parameters = _build_parameters()
    n_path = 100_000
    dt = 0.1
    stepper = PyFengVarianceStepper(
        parameters=parameters, n_path=n_path, dt=dt, seed=123,
    )
    v0 = np.full(n_path, parameters.initial_variance, dtype=float)
    v_next, _ = stepper.step(dt, v0)
    expected = parameters.theta + (parameters.initial_variance - parameters.theta) * np.exp(
        -parameters.kappa * dt
    )
    sample_mean = float(np.mean(v_next))
    sample_se = float(np.std(v_next, ddof=1) / np.sqrt(n_path))
    error = abs(sample_mean - expected)
    assert error < 5.0 * sample_se, (
        f"PyFENG v_next mean drifts: {sample_mean:.6f} vs expected {expected:.6f} "
        f"(SE = {sample_se:.6f}, error = {error:.6f})"
    )
