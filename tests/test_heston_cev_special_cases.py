"""Special-case and regression tests for the CEV-Heston simulator."""

from __future__ import annotations

import numpy as np
import pytest

from src.heston_cev_simulation import (
    _conditional_mean_correlated,
    price_european_option_heston_cev,
    simulate_heston_cev_terminal,
)
from src.utils import CEVHestonModelParameters, EuropeanOption


class _DeterministicVarianceStepper:
    backend_name = "test_deterministic"

    def __init__(self, params: CEVHestonModelParameters) -> None:
        self.params = params

    def step(self, dt: float, v_current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        v_next = self.params.theta + (v_current - self.params.theta) * np.exp(
            -self.params.kappa * dt
        )
        integrated = 0.5 * dt * (v_current + v_next)
        return v_next, integrated


def _base_parameters(**overrides) -> CEVHestonModelParameters:
    values = {
        "spot": 1.0,
        "initial_variance": 0.09,
        "kappa": 1.5,
        "theta": 0.09,
        "xi": 0.6,
        "correlation": -0.8,
        "beta": 0.4,
        "risk_free_rate": 0.0,
    }
    values.update(overrides)
    return CEVHestonModelParameters(**values)


def test_xi_zero_distribution_is_independent_of_rho_with_same_seed() -> None:
    """Deterministic variance should not retain rho-dependent variance removal."""

    common = {
        "spot": 1.0,
        "initial_variance": 0.04,
        "kappa": 1.3,
        "theta": 0.06,
        "xi": 0.0,
        "beta": 0.5,
        "risk_free_rate": 0.0,
    }
    params_rho_zero = CEVHestonModelParameters(**common, correlation=0.0)
    params_rho_neg = CEVHestonModelParameters(**common, correlation=-0.85)

    result_zero = simulate_heston_cev_terminal(
        params_rho_zero, maturity=2.0, n_steps=20, n_paths=20_000, seed=12345,
    )
    result_neg = simulate_heston_cev_terminal(
        params_rho_neg, maturity=2.0, n_steps=20, n_paths=20_000, seed=12345,
    )

    np.testing.assert_allclose(result_zero.terminal_prices, result_neg.terminal_prices)


def test_rho_zero_ignores_correlated_anchor_scheme_with_same_seed() -> None:
    """When rho = 0, both scheme names must collapse to the same CEV step."""

    params = _base_parameters(correlation=0.0)
    old = simulate_heston_cev_terminal(
        params,
        maturity=1.5,
        n_steps=15,
        n_paths=20_000,
        seed=2468,
        variance_stepper=_DeterministicVarianceStepper(params),
        conditional_scheme="frozen_left",
    )
    new = simulate_heston_cev_terminal(
        params,
        maturity=1.5,
        n_steps=15,
        n_paths=20_000,
        seed=2468,
        variance_stepper=_DeterministicVarianceStepper(params),
        conditional_scheme="power_projected",
    )

    np.testing.assert_allclose(old.terminal_prices, new.terminal_prices)


def test_beta_one_ignores_power_projected_scheme_with_same_seed() -> None:
    """The beta = 1 branch is the Heston/lognormal conditional step."""

    params = _base_parameters(beta=1.0)
    old = simulate_heston_cev_terminal(
        params,
        maturity=1.0,
        n_steps=20,
        n_paths=20_000,
        seed=13579,
        variance_stepper=_DeterministicVarianceStepper(params),
        conditional_scheme="frozen_left",
    )
    new = simulate_heston_cev_terminal(
        params,
        maturity=1.0,
        n_steps=20,
        n_paths=20_000,
        seed=13579,
        variance_stepper=_DeterministicVarianceStepper(params),
        conditional_scheme="power_projected",
    )

    np.testing.assert_allclose(old.terminal_prices, new.terminal_prices)


def test_correlated_mean_uses_cir_integral_with_correct_sign_and_units() -> None:
    """Positive rho*A should lift the anchor after the Ito correction."""

    params = _base_parameters(spot=100.0, beta=0.5, correlation=0.5)
    F_t = np.array([100.0])
    v_t = np.array([0.04])
    dt = 0.25
    integrated_variance = np.array([0.01])
    target_A = np.array([0.20])
    v_next = (
        target_A * params.xi
        + v_t
        + params.kappa * params.theta * dt
        - params.kappa * integrated_variance
    )

    anchor, n_clamped = _conditional_mean_correlated(
        F_t=F_t,
        v_t=v_t,
        v_next=v_next,
        integrated_variance=integrated_variance,
        parameters=params,
        dt=dt,
        epsilon_clamp=1e-8,
    )

    beta_star = 1.0 - params.beta
    denom = F_t ** beta_star
    expected = F_t * np.exp(
        params.correlation * target_A / denom
        - 0.5 * params.correlation ** 2 * integrated_variance / (denom * denom)
    )
    assert n_clamped == 0
    np.testing.assert_allclose(anchor, expected)
    assert float(anchor[0]) > float(F_t[0])


def test_power_projected_reduces_fixed_seed_correlated_pricing_bias() -> None:
    """Regression check for the correlated coarse-step option bias."""

    pytest.importorskip("pyfeng", reason="PyFENG-backed benchmark required")

    params = _base_parameters()
    option = EuropeanOption(strike=1.0, maturity=2.0, option_type="call")
    reference = price_european_option_heston_cev(
        params,
        option,
        n_steps=80,
        n_paths=120_000,
        seed=777,
        conditional_scheme="frozen_left",
    ).price
    old = price_european_option_heston_cev(
        params,
        option,
        n_steps=2,
        n_paths=50_000,
        seed=778,
        conditional_scheme="frozen_left",
    ).price
    improved = price_european_option_heston_cev(
        params,
        option,
        n_steps=2,
        n_paths=50_000,
        seed=778,
        conditional_scheme="power_projected",
    ).price

    assert abs(improved - reference) < 0.8 * abs(old - reference)
