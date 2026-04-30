"""Prototype Beta-Heston simulation and pricing helpers.

This module implements the professor-suggested extension direction:

``dS_t = r S_t dt + sqrt(v_t) S_t^beta dW_t``
``dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dZ_t``

with ``dW_t dZ_t = rho dt``.

Current scope:
- full-truncation Euler simulation for the Heston variance path
- baseline Euler simulation for the beta-power asset process
- a corrected Heston-specific conditional step
- an exact conditional lognormal step when ``beta = 1``
- a prototype CEV approximation fallback when ``0 < beta < 1``

The ``beta < 1`` conditional branch is intentionally labeled as a prototype.
It uses the correct Heston-specific correlated-variance term together with a
local CEV fallback based on the Lamperti transform. No exact CEV sampler or
PyFENG backend is available in this environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

import numpy as np

from .integration import trapezoidal_rule
from .sabr_simulation import generate_correlated_standard_normal_shocks
from .utils import EuropeanOption, standard_error


_XI_TOLERANCE = 1e-12
_RHO_TOLERANCE = 1e-12
_SAFE_POSITIVE_FLOOR = 1e-12


@dataclass(frozen=True)
class BetaHestonParameters:
    """Container for the prototype Beta-Heston model parameters."""

    spot: float
    v0: float
    kappa: float
    theta: float
    xi: float
    beta: float
    rho: float
    risk_free_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.spot <= 0.0:
            raise ValueError("spot must be positive")
        if self.v0 < 0.0:
            raise ValueError("v0 must be non-negative")
        if self.kappa < 0.0:
            raise ValueError("kappa must be non-negative")
        if self.theta < 0.0:
            raise ValueError("theta must be non-negative")
        if self.xi < 0.0:
            raise ValueError("xi must be non-negative")
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("beta must lie in [0, 1]")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError("rho must lie in [-1, 1]")


@dataclass(frozen=True)
class BetaHestonVarianceSimulationResult:
    """Container for the variance-layer simulation output.

    Shape convention:
    - ``time_grid`` has shape ``(n_steps + 1,)``
    - ``variance_paths`` has shape ``(n_paths, n_steps + 1)``
    - shock arrays have shape ``(n_paths, n_steps)``
    - ``step_integrated_variance`` has shape ``(n_paths, n_steps)``
    - ``integrated_variance`` has shape ``(n_paths,)``
    """

    time_grid: np.ndarray
    variance_paths: np.ndarray
    price_shocks: np.ndarray
    variance_shocks: np.ndarray
    step_integrated_variance: np.ndarray
    integrated_variance: np.ndarray

    @property
    def dt(self) -> float:
        """Return the uniform time step size."""

        return float(self.time_grid[1] - self.time_grid[0])


@dataclass(frozen=True)
class BetaHestonSimulationResult(BetaHestonVarianceSimulationResult):
    """Container for a full Beta-Heston path simulation."""

    asset_paths: np.ndarray
    method: str


@dataclass(frozen=True)
class BetaHestonPricingResult:
    """Container for prototype pricing outputs."""

    price: float
    standard_error: float
    discounted_payoffs: np.ndarray
    simulation: BetaHestonSimulationResult
    method: str


def module_status() -> str:
    """Return a short status string for small smoke tests."""

    return "beta-heston prototype ready"


def beta_heston_conditional_backend() -> str:
    """Return the current conditional-CEV backend description."""

    return "lamperti_fallback_without_pyfeng"


def _validate_common_simulation_inputs(
    maturity: float,
    n_steps: int,
    n_paths: int,
) -> None:
    """Validate common simulation inputs."""

    if maturity <= 0.0:
        raise ValueError("maturity must be positive")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")


def _validate_european_call(option: EuropeanOption) -> None:
    """Restrict the prototype pricers to European calls for now."""

    if option.option_type != "call":
        raise NotImplementedError("The Beta-Heston prototype supports only calls.")


def simulate_beta_heston_variance_paths(
    parameters: BetaHestonParameters,
    maturity: float,
    n_steps: int,
    n_paths: int,
    seed: int | np.random.Generator | None = None,
) -> BetaHestonVarianceSimulationResult:
    """Simulate the Heston variance path with full-truncation Euler.

    We evolve

    ``dv_t = kappa (theta - v_t^+) dt + xi sqrt(v_t^+) dZ_t``

    and then store ``max(v_{t+dt}, 0)`` at each step for numerical stability.
    """

    _validate_common_simulation_inputs(maturity, n_steps, n_paths)

    time_grid = np.linspace(0.0, maturity, n_steps + 1, dtype=float)
    dt = maturity / n_steps
    sqrt_dt = np.sqrt(dt)

    price_shocks, variance_shocks = generate_correlated_standard_normal_shocks(
        correlation=parameters.rho,
        n_paths=n_paths,
        n_steps=n_steps,
        rng=seed,
    )

    variance_paths = np.empty((n_paths, n_steps + 1), dtype=float)
    variance_paths[:, 0] = parameters.v0

    for step in range(n_steps):
        variance_current = variance_paths[:, step]
        variance_plus = np.maximum(variance_current, 0.0)
        variance_next = (
            variance_current
            + parameters.kappa * (parameters.theta - variance_plus) * dt
            + parameters.xi
            * np.sqrt(variance_plus)
            * sqrt_dt
            * variance_shocks[:, step]
        )
        variance_paths[:, step + 1] = np.maximum(variance_next, 0.0)

    step_integrated_variance = (
        0.5 * (variance_paths[:, :-1] + variance_paths[:, 1:]) * dt
    )
    integrated_variance = np.asarray(
        trapezoidal_rule(variance_paths, dx=dt),
        dtype=float,
    )

    return BetaHestonVarianceSimulationResult(
        time_grid=time_grid,
        variance_paths=variance_paths,
        price_shocks=price_shocks,
        variance_shocks=variance_shocks,
        step_integrated_variance=step_integrated_variance,
        integrated_variance=integrated_variance,
    )


def simulate_beta_heston_asset_paths_euler(
    parameters: BetaHestonParameters,
    variance_result: BetaHestonVarianceSimulationResult,
) -> np.ndarray:
    """Simulate the beta-power asset process with an Euler step.

    The step uses the left-endpoint state:

    ``S_{t+dt} = S_t + r S_t dt + sqrt(v_t) (S_t^+)^beta sqrt(dt) Z``

    where ``S_t^+ = max(S_t, 0)`` is used inside the fractional power so the
    scheme remains well-defined when beta < 1. Positivity is therefore not
    guaranteed for the Euler asset path itself, which is an important
    limitation of this baseline scheme.
    """

    dt = variance_result.dt
    sqrt_dt = np.sqrt(dt)
    n_paths, n_steps = variance_result.price_shocks.shape

    asset_paths = np.empty((n_paths, n_steps + 1), dtype=float)
    asset_paths[:, 0] = parameters.spot

    for step in range(n_steps):
        spot_current = asset_paths[:, step]
        variance_left = np.maximum(variance_result.variance_paths[:, step], 0.0)
        diffusion_scale = np.sqrt(variance_left) * np.power(
            np.maximum(spot_current, 0.0),
            parameters.beta,
        )
        asset_paths[:, step + 1] = (
            spot_current
            + parameters.risk_free_rate * spot_current * dt
            + diffusion_scale * sqrt_dt * variance_result.price_shocks[:, step]
        )

    return asset_paths


def _compute_independent_price_shocks(
    variance_result: BetaHestonVarianceSimulationResult,
    rho: float,
) -> np.ndarray:
    """Recover the Brownian shocks orthogonal to the variance driver."""

    if abs(rho) >= 1.0 - _RHO_TOLERANCE:
        return np.zeros_like(variance_result.price_shocks)

    rho_complement = np.sqrt(max(0.0, 1.0 - rho * rho))
    return (
        variance_result.price_shocks - rho * variance_result.variance_shocks
    ) / rho_complement


def _compute_heston_conditional_step_terms(
    parameters: BetaHestonParameters,
    variance_result: BetaHestonVarianceSimulationResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the Heston-specific conditional step terms.

    Returns
    -------
    a_steps:
        The correlated Heston integral term
        ``A_step = integral sqrt(v_s) dZ_s`` inferred from the variance path.
    correlated_variance_steps:
        The part of the step quadratic variation already explained by the
        correlated ``rho dZ`` channel.
    independent_variance_steps:
        The remaining conditional variance for the independent ``dW_perp`` part.
    independent_shocks:
        Standard normal shocks for the independent conditional driver.
    """

    step_integrated_variance = np.maximum(variance_result.step_integrated_variance, 0.0)
    if parameters.xi <= _XI_TOLERANCE:
        # When xi is zero (or numerically negligible), the variance path does
        # not reveal the Z-integral. The conditional step therefore retains the
        # full variance as unexplained Brownian noise.
        a_steps = np.zeros_like(step_integrated_variance)
        correlated_variance_steps = np.zeros_like(step_integrated_variance)
        independent_variance_steps = step_integrated_variance
        independent_shocks = variance_result.price_shocks
        return (
            a_steps,
            correlated_variance_steps,
            independent_variance_steps,
            independent_shocks,
        )

    variance_left = variance_result.variance_paths[:, :-1]
    variance_right = variance_result.variance_paths[:, 1:]
    dt = variance_result.dt

    a_steps = (
        variance_right
        - variance_left
        - parameters.kappa * parameters.theta * dt
        + parameters.kappa * step_integrated_variance
    ) / parameters.xi

    correlated_variance_steps = parameters.rho * parameters.rho * step_integrated_variance
    independent_variance_steps = np.maximum(
        (1.0 - parameters.rho * parameters.rho) * step_integrated_variance,
        0.0,
    )
    independent_shocks = _compute_independent_price_shocks(
        variance_result,
        parameters.rho,
    )
    return (
        a_steps,
        correlated_variance_steps,
        independent_variance_steps,
        independent_shocks,
    )


def _conditional_mean_anchor(
    parameters: BetaHestonParameters,
    spot_current: np.ndarray,
    dt: float,
    a_step: np.ndarray,
    correlated_variance_step: np.ndarray,
) -> np.ndarray:
    """Return the pathwise conditional mean anchor for one step.

    For ``beta = 1``, this reduces to the exact Heston-conditional lognormal
    anchor ``S_t exp(rh + rho A_step - 0.5 rho^2 I_step)``.

    For ``0 < beta < 1``, we adapt Professor Choi's SABR-style ``S_t^beta``
    idea by scaling the correlated Heston term with ``S_t^(1-beta)``.
    """

    safe_spot = np.maximum(spot_current, _SAFE_POSITIVE_FLOOR)
    beta_star = 1.0 - parameters.beta
    if beta_star <= _RHO_TOLERANCE:
        denominator = np.ones_like(safe_spot)
    else:
        denominator = np.power(safe_spot, beta_star)

    return safe_spot * np.exp(
        parameters.risk_free_rate * dt
        + parameters.rho * a_step / denominator
        - 0.5 * correlated_variance_step / np.square(denominator)
    )


def _conditional_cev_lamperti_fallback_step(
    parameters: BetaHestonParameters,
    anchor_spot: np.ndarray,
    independent_variance_step: np.ndarray,
    independent_shock: np.ndarray,
) -> np.ndarray:
    """Approximate one conditional CEV step with a Lamperti-transform fallback.

    No exact CEV sampler is available in this environment. We therefore use a
    local approximation on the transformed variable

    ``X = S^(1-beta) / (1-beta)``

    whose diffusion becomes locally constant. This keeps the approximation
    clearly tied to the CEV structure while remaining lightweight and explicit
    about its limitations.
    """

    beta = parameters.beta
    beta_star = 1.0 - beta
    if beta_star <= _RHO_TOLERANCE:
        raise ValueError("The Lamperti CEV fallback requires beta < 1.")

    safe_anchor = np.maximum(anchor_spot, _SAFE_POSITIVE_FLOOR)
    safe_independent_variance = np.maximum(independent_variance_step, 0.0)
    transformed_anchor = np.power(safe_anchor, beta_star) / beta_star

    # Ito correction for the transformed CEV state, frozen over one step.
    singular_drift_correction = -0.5 * beta * safe_independent_variance / np.maximum(
        np.power(safe_anchor, beta_star),
        _SAFE_POSITIVE_FLOOR,
    )
    transformed_next = (
        transformed_anchor
        + singular_drift_correction
        + np.sqrt(safe_independent_variance) * independent_shock
    )

    return np.power(
        np.maximum(beta_star * transformed_next, 0.0),
        1.0 / beta_star,
    )


def simulate_beta_heston_asset_paths_conditional_approximation(
    parameters: BetaHestonParameters,
    variance_result: BetaHestonVarianceSimulationResult,
) -> np.ndarray:
    r"""Simulate the corrected conditional Beta-Heston approximation.

    The Heston-specific correlated term is computed from

    ``A_step = (v_{t+h} - v_t - kappa theta h + kappa I_step) / xi``

    where ``I_step ≈ \int_t^{t+h} v_s ds``.

    - If ``beta = 1``, we use the exact conditional lognormal step

      ``S_{t+h} = S_t exp(rh + rho A_step - 0.5 I_step + sqrt(Q_step) N)``

      with ``Q_step = (1-rho^2) I_step`` when ``xi > 0``.

    - If ``0 < beta < 1``, we use the corrected conditional mean anchor plus a
      local Lamperti-transform CEV fallback for the independent component.

    This is still a prototype approximation for ``beta < 1``. It is more
    faithful to the Heston/SABR decomposition than the earlier mean-only
    version, but it is not exact conditional CEV sampling.
    """

    dt = variance_result.dt
    n_paths, n_steps = variance_result.price_shocks.shape
    asset_paths = np.empty((n_paths, n_steps + 1), dtype=float)
    asset_paths[:, 0] = parameters.spot

    (
        a_steps,
        correlated_variance_steps,
        independent_variance_steps,
        independent_shocks,
    ) = _compute_heston_conditional_step_terms(parameters, variance_result)

    beta_star = 1.0 - parameters.beta
    for step in range(n_steps):
        spot_current = asset_paths[:, step]
        a_step = a_steps[:, step]
        correlated_variance_step = correlated_variance_steps[:, step]
        independent_variance_step = independent_variance_steps[:, step]
        independent_shock = independent_shocks[:, step]

        anchor = _conditional_mean_anchor(
            parameters=parameters,
            spot_current=spot_current,
            dt=dt,
            a_step=a_step,
            correlated_variance_step=correlated_variance_step,
        )

        if beta_star <= _RHO_TOLERANCE:
            asset_paths[:, step + 1] = anchor * np.exp(
                -0.5 * independent_variance_step
                + np.sqrt(independent_variance_step) * independent_shock
            )
        else:
            asset_paths[:, step + 1] = _conditional_cev_lamperti_fallback_step(
                parameters=parameters,
                anchor_spot=anchor,
                independent_variance_step=independent_variance_step,
                independent_shock=independent_shock,
            )

    return asset_paths


def simulate_beta_heston_asset_paths_cev_conditional_mean(
    parameters: BetaHestonParameters,
    variance_result: BetaHestonVarianceSimulationResult,
) -> np.ndarray:
    """Backward-compatible wrapper for the refined conditional approximation."""

    return simulate_beta_heston_asset_paths_conditional_approximation(
        parameters=parameters,
        variance_result=variance_result,
    )


def simulate_beta_heston_paths_euler(
    parameters: BetaHestonParameters,
    maturity: float,
    n_steps: int,
    n_paths: int,
    seed: int | np.random.Generator | None = None,
) -> BetaHestonSimulationResult:
    """Simulate full Beta-Heston paths with the baseline Euler asset step."""

    variance_result = simulate_beta_heston_variance_paths(
        parameters=parameters,
        maturity=maturity,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
    )
    asset_paths = simulate_beta_heston_asset_paths_euler(parameters, variance_result)

    return BetaHestonSimulationResult(
        time_grid=variance_result.time_grid,
        variance_paths=variance_result.variance_paths,
        price_shocks=variance_result.price_shocks,
        variance_shocks=variance_result.variance_shocks,
        step_integrated_variance=variance_result.step_integrated_variance,
        integrated_variance=variance_result.integrated_variance,
        asset_paths=asset_paths,
        method="euler",
    )


def simulate_beta_heston_paths_conditional_approximation(
    parameters: BetaHestonParameters,
    maturity: float,
    n_steps: int,
    n_paths: int,
    seed: int | np.random.Generator | None = None,
) -> BetaHestonSimulationResult:
    """Simulate full Beta-Heston paths with the refined conditional prototype."""

    variance_result = simulate_beta_heston_variance_paths(
        parameters=parameters,
        maturity=maturity,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
    )
    asset_paths = simulate_beta_heston_asset_paths_conditional_approximation(
        parameters,
        variance_result,
    )

    return BetaHestonSimulationResult(
        time_grid=variance_result.time_grid,
        variance_paths=variance_result.variance_paths,
        price_shocks=variance_result.price_shocks,
        variance_shocks=variance_result.variance_shocks,
        step_integrated_variance=variance_result.step_integrated_variance,
        integrated_variance=variance_result.integrated_variance,
        asset_paths=asset_paths,
        method="conditional_approximation",
    )


def simulate_beta_heston_paths_cev_conditional_mean(
    parameters: BetaHestonParameters,
    maturity: float,
    n_steps: int,
    n_paths: int,
    seed: int | np.random.Generator | None = None,
) -> BetaHestonSimulationResult:
    """Backward-compatible wrapper for the refined conditional prototype."""

    return simulate_beta_heston_paths_conditional_approximation(
        parameters=parameters,
        maturity=maturity,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
    )


def _discounted_call_payoffs(
    terminal_asset_prices: np.ndarray,
    strike: float,
    rate: float,
    maturity: float,
) -> np.ndarray:
    """Return discounted European call payoffs."""

    payoffs = np.maximum(terminal_asset_prices - strike, 0.0)
    return exp(-rate * maturity) * payoffs


def price_beta_heston_european_call_euler(
    parameters: BetaHestonParameters,
    option: EuropeanOption,
    n_steps: int,
    n_paths: int,
    seed: int | np.random.Generator | None = None,
) -> BetaHestonPricingResult:
    """Price a European call with the baseline Euler Beta-Heston scheme."""

    _validate_european_call(option)
    simulation = simulate_beta_heston_paths_euler(
        parameters=parameters,
        maturity=option.maturity,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
    )
    discounted_payoffs = _discounted_call_payoffs(
        simulation.asset_paths[:, -1],
        strike=option.strike,
        rate=parameters.risk_free_rate,
        maturity=option.maturity,
    )

    return BetaHestonPricingResult(
        price=float(np.mean(discounted_payoffs)),
        standard_error=float(standard_error(discounted_payoffs)),
        discounted_payoffs=discounted_payoffs,
        simulation=simulation,
        method=simulation.method,
    )


def price_beta_heston_european_call_conditional_approximation(
    parameters: BetaHestonParameters,
    option: EuropeanOption,
    n_steps: int,
    n_paths: int,
    seed: int | np.random.Generator | None = None,
) -> BetaHestonPricingResult:
    """Price a European call with the refined conditional Beta-Heston scheme."""

    _validate_european_call(option)
    simulation = simulate_beta_heston_paths_conditional_approximation(
        parameters=parameters,
        maturity=option.maturity,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
    )
    discounted_payoffs = _discounted_call_payoffs(
        simulation.asset_paths[:, -1],
        strike=option.strike,
        rate=parameters.risk_free_rate,
        maturity=option.maturity,
    )

    return BetaHestonPricingResult(
        price=float(np.mean(discounted_payoffs)),
        standard_error=float(standard_error(discounted_payoffs)),
        discounted_payoffs=discounted_payoffs,
        simulation=simulation,
        method=simulation.method,
    )


def price_beta_heston_european_call_cev_conditional_mean(
    parameters: BetaHestonParameters,
    option: EuropeanOption,
    n_steps: int,
    n_paths: int,
    seed: int | np.random.Generator | None = None,
) -> BetaHestonPricingResult:
    """Backward-compatible wrapper for the refined conditional prototype."""

    return price_beta_heston_european_call_conditional_approximation(
        parameters=parameters,
        option=option,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
    )
