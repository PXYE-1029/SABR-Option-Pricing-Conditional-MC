"""Conditional Monte Carlo pricing under the beta = 1 SABR model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .black_scholes import black_scholes_call_price
from .integration import (
    simpson_integrated_variance,
    trapezoidal_integrated_variance,
)
from .sabr_simulation import VolatilitySimulationResult, simulate_volatility_paths
from .utils import EuropeanOption, SABRModelParameters, standard_error


@dataclass(frozen=True)
class ConditionalMonteCarloPricingResult:
    """Container for conditional Monte Carlo pricing outputs."""

    price: float
    standard_error: float
    pathwise_conditional_prices: np.ndarray
    volatility_simulation: VolatilitySimulationResult
    integrated_variance: np.ndarray
    integration_method: str


def module_status() -> str:
    """Return a short status string for small smoke tests."""

    return "conditional Monte Carlo pricer ready"


def _compute_integrated_variance(
    sigma_paths: np.ndarray,
    time_grid: np.ndarray,
    integration_method: str,
) -> np.ndarray:
    """Compute pathwise integrated variance with the requested rule."""

    if integration_method == "trapezoidal":
        return np.asarray(
            trapezoidal_integrated_variance(sigma_paths, time_grid),
            dtype=float,
        )
    if integration_method == "simpson":
        return np.asarray(
            simpson_integrated_variance(sigma_paths, time_grid),
            dtype=float,
        )
    raise ValueError("integration_method must be 'trapezoidal' or 'simpson'")


def price_european_option_conditional_mc(
    parameters: SABRModelParameters,
    option: EuropeanOption,
    n_steps: int,
    n_paths: int,
    seed: int | np.random.Generator | None = None,
    integration_method: str = "trapezoidal",
) -> ConditionalMonteCarloPricingResult:
    r"""Price a European call with conditional Monte Carlo under beta = 1 SABR.

    For ``nu > 0``, we use the standard beta = 1 decomposition

    ``dW = rho dZ + sqrt(1-rho^2) dW_perp``

    together with ``d sigma_t = nu sigma_t dZ_t``. This gives

    ``log S_T = log S_0 + rT - 0.5 V_T + (rho/nu)(sigma_T-sigma_0)
                + sqrt((1-rho^2)V_T) N``

    conditional on the volatility path, where ``V_T = \int_0^T sigma_t^2 dt``.
    We therefore price each path by a Black-Scholes call with

    ``S_cond = S_0 exp((rho/nu)(sigma_T-sigma_0) - 0.5 rho^2 V_T)``

    and

    ``sigma_cond = sqrt((1-rho^2) V_T / T)``.
    """

    if parameters.beta != 1.0:
        raise NotImplementedError("Phase 4 conditional MC supports only beta = 1.")
    if option.option_type != "call":
        raise NotImplementedError(
            "Phase 4 conditional MC supports only European calls."
        )

    volatility_simulation = simulate_volatility_paths(
        parameters=parameters,
        maturity=option.maturity,
        n_steps=n_steps,
        n_paths=n_paths,
        rng=seed,
    )
    integrated_variance = _compute_integrated_variance(
        volatility_simulation.sigma_paths,
        volatility_simulation.time_grid,
        integration_method,
    )

    if parameters.vol_of_vol == 0.0:
        black_scholes_price = black_scholes_call_price(
            spot=parameters.spot,
            strike=option.strike,
            maturity=option.maturity,
            rate=parameters.risk_free_rate,
            volatility=parameters.initial_volatility,
        )
        pathwise_prices = np.full(n_paths, black_scholes_price, dtype=float)
    else:
        sigma_terminal = volatility_simulation.sigma_paths[:, -1]
        rho = parameters.correlation

        conditional_spots = parameters.spot * np.exp(
            (rho / parameters.vol_of_vol)
            * (sigma_terminal - parameters.initial_volatility)
            - 0.5 * rho * rho * integrated_variance
        )
        conditional_volatilities = np.sqrt(
            np.maximum(0.0, (1.0 - rho * rho) * integrated_variance / option.maturity)
        )
        pathwise_prices = np.asarray(
            black_scholes_call_price(
            spot=conditional_spots,
            strike=option.strike,
            maturity=option.maturity,
            rate=parameters.risk_free_rate,
            volatility=conditional_volatilities,
        ),
            dtype=float,
        )

    return ConditionalMonteCarloPricingResult(
        price=float(np.mean(pathwise_prices)),
        standard_error=float(standard_error(pathwise_prices)),
        pathwise_conditional_prices=pathwise_prices,
        volatility_simulation=volatility_simulation,
        integrated_variance=integrated_variance,
        integration_method=integration_method,
    )
