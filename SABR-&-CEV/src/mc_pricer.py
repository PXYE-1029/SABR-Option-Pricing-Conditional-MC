"""Plain Monte Carlo pricing under the SABR model."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

import numpy as np

from .sabr_simulation import SABRSimulationResult, simulate_sabr_paths
from .utils import EuropeanOption, SABRModelParameters, standard_error


@dataclass(frozen=True)
class MonteCarloPricingResult:
    """Container for plain Monte Carlo pricing outputs."""

    price: float
    standard_error: float
    discounted_payoffs: np.ndarray
    simulation: SABRSimulationResult


def module_status() -> str:
    """Return a short status string for small smoke tests."""

    return "plain Monte Carlo pricer ready"


def _call_payoff(terminal_asset_prices: np.ndarray, strike: float) -> np.ndarray:
    """Return European call payoffs."""

    return np.maximum(terminal_asset_prices - strike, 0.0)


def price_european_option_mc(
    parameters: SABRModelParameters,
    option: EuropeanOption,
    n_steps: int,
    n_paths: int,
    seed: int | np.random.Generator | None = None,
) -> MonteCarloPricingResult:
    """Price a European option with plain Monte Carlo under SABR.

    Phase 3 intentionally supports only European calls and the beta = 1 SABR
    case. Broader support can be added once the baseline path/pricing pipeline
    is validated.
    """

    if parameters.beta != 1.0:
        raise NotImplementedError("Phase 3 pricing supports only beta = 1.")
    if option.option_type != "call":
        raise NotImplementedError("Phase 3 pricing supports only European calls.")

    simulation = simulate_sabr_paths(
        parameters=parameters,
        maturity=option.maturity,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
    )

    terminal_asset_prices = simulation.asset_paths[:, -1]
    payoffs = _call_payoff(terminal_asset_prices, option.strike)
    discounted_payoffs = exp(-parameters.risk_free_rate * option.maturity) * payoffs

    return MonteCarloPricingResult(
        price=float(np.mean(discounted_payoffs)),
        standard_error=float(standard_error(discounted_payoffs)),
        discounted_payoffs=discounted_payoffs,
        simulation=simulation,
    )
