"""Basic tests for the Beta-Heston prototype extension."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.experiment_beta_heston_backend_comparison import (
    run_experiment as run_backend_experiment,
)
from experiments.experiment_beta_heston_prototype import run_experiment
from src.beta_heston_simulation import (
    BetaHestonParameters,
    _compute_heston_conditional_step_terms,
    _conditional_mean_anchor,
    is_pyfeng_available,
    price_beta_heston_european_call_conditional_approximation,
    simulate_beta_heston_asset_paths_euler,
    simulate_beta_heston_variance_paths,
)
from src.black_scholes import black_scholes_call_price
from src.utils import EuropeanOption


PYFENG_AVAILABLE = is_pyfeng_available()


def test_beta_heston_variance_paths_remain_nonnegative() -> None:
    """Full-truncation Euler should keep stored variance paths nonnegative."""

    parameters = BetaHestonParameters(
        spot=100.0,
        v0=0.04,
        kappa=1.5,
        theta=0.04,
        xi=0.7,
        beta=0.7,
        rho=-0.5,
        risk_free_rate=0.01,
    )
    result = simulate_beta_heston_variance_paths(
        parameters=parameters,
        maturity=1.0,
        n_steps=50,
        n_paths=128,
        seed=123,
    )

    assert result.variance_backend == "euler"
    assert np.all(result.variance_paths >= 0.0)
    assert np.all(result.step_integrated_variance >= 0.0)
    assert np.all(result.integrated_variance >= 0.0)
    assert result.independent_shocks.shape == (128, 50)


def test_pyfeng_backend_missing_dependency_raises_clear_error() -> None:
    """The optional backend should fail clearly when PyFENG is not installed."""

    if PYFENG_AVAILABLE:
        pytest.skip("PyFENG is installed in this environment.")

    parameters = BetaHestonParameters(
        spot=100.0,
        v0=0.04,
        kappa=1.5,
        theta=0.04,
        xi=0.4,
        beta=0.7,
        rho=-0.5,
        risk_free_rate=0.01,
    )

    with pytest.raises(ImportError, match="optional PyFENG dependency"):
        simulate_beta_heston_variance_paths(
            parameters=parameters,
            maturity=1.0,
            n_steps=10,
            n_paths=32,
            seed=123,
            variance_backend="pyfeng_choi_kwok_td",
        )


def test_beta_one_euler_matches_linear_heston_like_update() -> None:
    """When beta=1, the Euler asset step should be linear in the current spot."""

    parameters = BetaHestonParameters(
        spot=100.0,
        v0=0.04,
        kappa=0.0,
        theta=0.04,
        xi=0.0,
        beta=1.0,
        rho=0.0,
        risk_free_rate=0.01,
    )
    variance_result = simulate_beta_heston_variance_paths(
        parameters=parameters,
        maturity=1.0,
        n_steps=6,
        n_paths=16,
        seed=321,
    )
    asset_paths = simulate_beta_heston_asset_paths_euler(parameters, variance_result)

    dt = variance_result.dt
    sqrt_dt = np.sqrt(dt)
    manual_paths = np.empty_like(asset_paths)
    manual_paths[:, 0] = parameters.spot

    for step in range(variance_result.price_shocks.shape[1]):
        manual_paths[:, step + 1] = manual_paths[:, step] * (
            1.0
            + parameters.risk_free_rate * dt
            + np.sqrt(parameters.v0) * sqrt_dt * variance_result.price_shocks[:, step]
        )

    assert np.allclose(variance_result.variance_paths, parameters.v0)
    assert np.allclose(asset_paths, manual_paths)


def test_beta_one_conditional_scheme_matches_black_scholes_when_variance_is_constant() -> None:
    """The beta=1 conditional branch should reduce to a lognormal benchmark."""

    parameters = BetaHestonParameters(
        spot=100.0,
        v0=0.04,
        kappa=0.0,
        theta=0.04,
        xi=0.0,
        beta=1.0,
        rho=-0.6,
        risk_free_rate=0.01,
    )
    option = EuropeanOption(strike=100.0, maturity=1.0, option_type="call")

    result = price_beta_heston_european_call_conditional_approximation(
        parameters=parameters,
        option=option,
        n_steps=50,
        n_paths=20_000,
        seed=123,
    )
    benchmark = black_scholes_call_price(
        spot=parameters.spot,
        strike=option.strike,
        maturity=option.maturity,
        rate=parameters.risk_free_rate,
        volatility=np.sqrt(parameters.v0),
    )

    assert abs(result.price - benchmark) < 4.0 * result.standard_error


def test_rho_zero_removes_the_correlated_conditional_mean_adjustment() -> None:
    """When rho=0, the correlated conditional-mean correction should vanish."""

    parameters = BetaHestonParameters(
        spot=100.0,
        v0=0.04,
        kappa=1.5,
        theta=0.04,
        xi=0.4,
        beta=0.7,
        rho=0.0,
        risk_free_rate=0.01,
    )
    variance_result = simulate_beta_heston_variance_paths(
        parameters=parameters,
        maturity=1.0,
        n_steps=12,
        n_paths=32,
        seed=99,
    )
    a_steps, correlated_variance_steps, _, _ = _compute_heston_conditional_step_terms(
        parameters,
        variance_result,
    )

    spot_current = np.full(32, parameters.spot)
    anchor = _conditional_mean_anchor(
        parameters=parameters,
        spot_current=spot_current,
        dt=variance_result.dt,
        a_step=a_steps[:, 0],
        correlated_variance_step=correlated_variance_steps[:, 0],
    )
    forward_base = parameters.spot * np.exp(parameters.risk_free_rate * variance_result.dt)

    assert np.allclose(correlated_variance_steps, 0.0)
    assert np.allclose(anchor, forward_base)


def test_near_zero_xi_uses_the_clean_edge_case_branch() -> None:
    """Near-zero xi should avoid unstable division and keep full step variance."""

    parameters = BetaHestonParameters(
        spot=100.0,
        v0=0.04,
        kappa=1.5,
        theta=0.04,
        xi=1e-14,
        beta=0.7,
        rho=-0.5,
        risk_free_rate=0.01,
    )
    variance_result = simulate_beta_heston_variance_paths(
        parameters=parameters,
        maturity=1.0,
        n_steps=12,
        n_paths=32,
        seed=77,
    )
    a_steps, correlated_variance_steps, independent_variance_steps, _ = (
        _compute_heston_conditional_step_terms(parameters, variance_result)
    )

    assert np.allclose(a_steps, 0.0)
    assert np.allclose(correlated_variance_steps, 0.0)
    assert np.allclose(
        independent_variance_steps,
        variance_result.step_integrated_variance,
    )


def test_beta_heston_prototype_experiment_runs_on_small_configuration(tmp_path) -> None:
    """The prototype experiment should run and save outputs on a small grid."""

    table_path = tmp_path / "beta_heston_small.csv"
    figure_path = tmp_path / "beta_heston_small.png"
    rows = run_experiment(
        beta_values=[1.0, 0.7],
        n_steps=10,
        n_paths=200,
        table_path=table_path,
        figure_path=figure_path,
    )

    assert len(rows) == 2
    assert table_path.exists()
    assert figure_path.exists()
    assert all("euler_price" in row for row in rows)
    assert all("conditional_approximation_price" in row for row in rows)


@pytest.mark.skipif(not PYFENG_AVAILABLE, reason="PyFENG is not installed.")
def test_pyfeng_backend_shapes_match_euler_backend() -> None:
    """PyFENG and Euler variance backends should share the same output shapes."""

    parameters = BetaHestonParameters(
        spot=100.0,
        v0=0.04,
        kappa=1.5,
        theta=0.04,
        xi=0.4,
        beta=0.7,
        rho=-0.5,
        risk_free_rate=0.01,
    )

    euler_result = simulate_beta_heston_variance_paths(
        parameters=parameters,
        maturity=1.0,
        n_steps=12,
        n_paths=64,
        seed=123,
        variance_backend="euler",
    )
    pyfeng_result = simulate_beta_heston_variance_paths(
        parameters=parameters,
        maturity=1.0,
        n_steps=12,
        n_paths=64,
        seed=123,
        variance_backend="pyfeng_choi_kwok_td",
    )

    assert pyfeng_result.variance_backend == "pyfeng_choi_kwok_td"
    assert pyfeng_result.variance_paths.shape == euler_result.variance_paths.shape
    assert (
        pyfeng_result.step_integrated_variance.shape
        == euler_result.step_integrated_variance.shape
    )
    assert pyfeng_result.integrated_variance.shape == euler_result.integrated_variance.shape
    assert pyfeng_result.independent_shocks.shape == euler_result.independent_shocks.shape
    assert np.all(pyfeng_result.variance_paths >= 0.0)
    assert np.all(pyfeng_result.step_integrated_variance >= 0.0)


@pytest.mark.skipif(not PYFENG_AVAILABLE, reason="PyFENG is not installed.")
def test_pyfeng_backend_conditional_pricer_runs() -> None:
    """The PyFENG-backed conditional prototype should run on a small sample."""

    parameters = BetaHestonParameters(
        spot=100.0,
        v0=0.04,
        kappa=1.5,
        theta=0.04,
        xi=0.4,
        beta=1.0,
        rho=-0.5,
        risk_free_rate=0.01,
    )
    option = EuropeanOption(strike=100.0, maturity=1.0, option_type="call")

    result = price_beta_heston_european_call_conditional_approximation(
        parameters=parameters,
        option=option,
        n_steps=12,
        n_paths=128,
        seed=123,
        variance_backend="pyfeng_choi_kwok_td",
    )

    assert result.price > 0.0
    assert result.standard_error >= 0.0
    assert result.simulation.variance_backend == "pyfeng_choi_kwok_td"


@pytest.mark.skipif(not PYFENG_AVAILABLE, reason="PyFENG is not installed.")
def test_pyfeng_backend_handles_near_zero_xi_cleanly() -> None:
    """Near-zero xi should not force the PyFENG wrapper through a singular path."""

    parameters = BetaHestonParameters(
        spot=100.0,
        v0=0.04,
        kappa=1.5,
        theta=0.04,
        xi=1e-14,
        beta=1.0,
        rho=-0.5,
        risk_free_rate=0.01,
    )
    option = EuropeanOption(strike=100.0, maturity=1.0, option_type="call")

    result = price_beta_heston_european_call_conditional_approximation(
        parameters=parameters,
        option=option,
        n_steps=12,
        n_paths=128,
        seed=123,
        variance_backend="pyfeng_choi_kwok_td",
    )

    assert result.price > 0.0
    assert result.standard_error >= 0.0
    assert result.simulation.variance_backend == "pyfeng_choi_kwok_td"


@pytest.mark.skipif(not PYFENG_AVAILABLE, reason="PyFENG is not installed.")
def test_backend_comparison_experiment_runs_on_small_configuration(tmp_path) -> None:
    """The backend comparison experiment should run on a small grid."""

    table_path = tmp_path / "beta_heston_backend_small.csv"
    figure_path = tmp_path / "beta_heston_backend_small.png"
    rows = run_backend_experiment(
        beta_values=[1.0, 0.7],
        n_steps=8,
        n_paths=128,
        table_path=table_path,
        figure_path=figure_path,
    )

    assert len(rows) == 4
    assert table_path.exists()
    assert figure_path.exists()
    assert any(row["backend"] == "euler" for row in rows)
    assert any(row["backend"] == "pyfeng_choi_kwok_td" for row in rows)
    assert all("average_integrated_variance" in row for row in rows)
    assert all("price_difference_from_euler_backend" in row for row in rows)
