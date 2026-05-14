"""SABR simulation helpers for volatility paths and correlated shocks.

Shape convention used in this module:
- ``time_grid`` has shape ``(n_steps + 1,)``
- path arrays such as ``sigma_paths`` and ``asset_paths`` have shape
  ``(n_paths, n_steps + 1)``
- shock arrays such as ``price_shocks`` and ``volatility_shocks`` have shape
  ``(n_paths, n_steps)``

Phase 3 adds full asset-path simulation for the beta = 1 SABR case. The
volatility simulation layer remains reusable on its own so that later pricing
methods, including conditional Monte Carlo, can build on it directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import SABRModelParameters, get_rng


@dataclass(frozen=True)
class VolatilitySimulationResult:
    """Container for SABR volatility-path simulation output."""

    time_grid: np.ndarray
    sigma_paths: np.ndarray
    price_shocks: np.ndarray
    volatility_shocks: np.ndarray

    @property
    def dt(self) -> float:
        """Return the uniform time step size."""

        return float(self.time_grid[1] - self.time_grid[0])


@dataclass(frozen=True)
class SABRSimulationResult(VolatilitySimulationResult):
    """Container for full SABR simulation output in the beta = 1 case."""

    asset_paths: np.ndarray


def module_status() -> str:
    """Return a short status string for small smoke tests."""

    return "sabr asset and volatility simulation ready"


def generate_correlated_standard_normal_shocks(
    correlation: float,
    n_paths: int,
    n_steps: int,
    rng: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate correlated standard normal shocks for the SABR Brownian drivers.

    Returns a pair ``(price_shocks, volatility_shocks)`` with shape
    ``(n_paths, n_steps)`` for each array.
    """

    if not -1.0 <= correlation <= 1.0:
        raise ValueError("correlation must lie in [-1, 1]")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")

    generator = get_rng(rng)
    independent_1 = generator.standard_normal(size=(n_paths, n_steps))
    independent_2 = generator.standard_normal(size=(n_paths, n_steps))

    price_shocks = independent_1
    volatility_shocks = (
        correlation * independent_1
        + np.sqrt(1.0 - correlation * correlation) * independent_2
    )
    return price_shocks, volatility_shocks


def simulate_volatility_paths(
    parameters: SABRModelParameters,
    maturity: float,
    n_steps: int,
    n_paths: int,
    rng: int | np.random.Generator | None = None,
) -> VolatilitySimulationResult:
    """Simulate SABR volatility paths and correlated normal shocks.

    The volatility process

    ``d sigma_t = nu * sigma_t * dZ_t``

    is a geometric Brownian motion, so we use its exact lognormal update. This
    standard choice keeps simulated volatilities strictly positive.
    """

    if maturity <= 0.0:
        raise ValueError("maturity must be positive")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")

    time_grid = np.linspace(0.0, maturity, n_steps + 1, dtype=float)
    dt = maturity / n_steps
    sqrt_dt = np.sqrt(dt)

    price_shocks, volatility_shocks = generate_correlated_standard_normal_shocks(
        correlation=parameters.correlation,
        n_paths=n_paths,
        n_steps=n_steps,
        rng=rng,
    )

    nu = parameters.vol_of_vol
    log_increment_drift = -0.5 * nu * nu * dt
    log_increments = log_increment_drift + nu * sqrt_dt * volatility_shocks

    sigma_paths = np.empty((n_paths, n_steps + 1), dtype=float)
    sigma_paths[:, 0] = parameters.initial_volatility
    sigma_paths[:, 1:] = parameters.initial_volatility * np.exp(
        np.cumsum(log_increments, axis=1)
    )

    return VolatilitySimulationResult(
        time_grid=time_grid,
        sigma_paths=sigma_paths,
        price_shocks=price_shocks,
        volatility_shocks=volatility_shocks,
    )


def simulate_asset_paths_beta_one(
    parameters: SABRModelParameters,
    volatility_result: VolatilitySimulationResult,
) -> np.ndarray:
    """Simulate asset-price paths for the beta = 1 SABR case.

    The update uses the left-endpoint volatility on each step:

    ``log S_{t+dt} = log S_t + (r - 0.5 sigma_t^2) dt + sigma_t sqrt(dt) Z``

    which preserves positivity of the simulated asset paths.
    """

    if parameters.beta != 1.0:
        raise NotImplementedError("Phase 3 supports only the beta = 1 SABR case.")

    sigma_left = volatility_result.sigma_paths[:, :-1]
    dt = volatility_result.dt
    sqrt_dt = np.sqrt(dt)

    log_increments = (
        (parameters.risk_free_rate - 0.5 * np.square(sigma_left)) * dt
        + sigma_left * sqrt_dt * volatility_result.price_shocks
    )

    n_paths, n_steps = volatility_result.price_shocks.shape
    asset_paths = np.empty((n_paths, n_steps + 1), dtype=float)
    asset_paths[:, 0] = parameters.spot
    asset_paths[:, 1:] = parameters.spot * np.exp(np.cumsum(log_increments, axis=1))
    return asset_paths


def simulate_sabr_paths(
    parameters: SABRModelParameters,
    maturity: float,
    n_steps: int,
    n_paths: int,
    seed: int | np.random.Generator | None = None,
) -> SABRSimulationResult:
    """Simulate full SABR paths for the beta = 1 case.

    This function reuses the standalone volatility simulation layer so later
    pricing routines can still access and reuse sigma paths directly.
    """

    volatility_result = simulate_volatility_paths(
        parameters=parameters,
        maturity=maturity,
        n_steps=n_steps,
        n_paths=n_paths,
        rng=seed,
    )
    asset_paths = simulate_asset_paths_beta_one(parameters, volatility_result)

    return SABRSimulationResult(
        time_grid=volatility_result.time_grid,
        sigma_paths=volatility_result.sigma_paths,
        asset_paths=asset_paths,
        price_shocks=volatility_result.price_shocks,
        volatility_shocks=volatility_result.volatility_shocks,
    )
