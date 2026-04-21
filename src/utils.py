"""Shared data structures and small validation helpers.

Array shape convention used across the project:
- path arrays have shape ``(n_paths, n_steps + 1)``
- time runs along the last axis
- column ``0`` stores the initial value at time ``t = 0``
- increment/shock arrays have shape ``(n_paths, n_steps)``
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Literal, TypeVar

import numpy as np


T = TypeVar("T")
OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class SABRModelParameters:
    """Container for the core SABR model parameters.

    The parameter names follow the standard SABR notation:
    - spot: current asset price S_0
    - initial_volatility: initial volatility sigma_0
    - beta: backbone parameter in [0, 1]
    - vol_of_vol: volatility of volatility nu
    - correlation: correlation rho between price and volatility shocks
    - risk_free_rate: continuously compounded risk-free rate
    """

    spot: float
    initial_volatility: float
    beta: float
    vol_of_vol: float
    correlation: float
    risk_free_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.spot <= 0.0:
            raise ValueError("spot must be positive")
        if self.initial_volatility <= 0.0:
            raise ValueError("initial_volatility must be positive")
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("beta must lie in [0, 1]")
        if self.vol_of_vol < 0.0:
            raise ValueError("vol_of_vol must be non-negative")
        if not -1.0 <= self.correlation <= 1.0:
            raise ValueError("correlation must lie in [-1, 1]")


@dataclass(frozen=True)
class EuropeanOption:
    """Container for a plain-vanilla European option contract."""

    strike: float
    maturity: float
    option_type: OptionType = "call"

    def __post_init__(self) -> None:
        if self.strike <= 0.0:
            raise ValueError("strike must be positive")
        if self.maturity <= 0.0:
            raise ValueError("maturity must be positive")
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")


def get_rng(seed: int | np.random.Generator | None = None) -> np.random.Generator:
    """Return a NumPy random number generator.

    Parameters
    ----------
    seed:
        If ``None``, create a fresh generator.
        If an ``int``, seed a new generator.
        If already a ``np.random.Generator``, return it unchanged.
    """

    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def standard_error(samples: np.ndarray, axis: int | None = None) -> float | np.ndarray:
    """Compute the Monte Carlo standard error."""

    values = np.asarray(samples, dtype=float)
    if values.size == 0:
        raise ValueError("samples must contain at least one value")

    sample_size = values.shape[axis] if axis is not None else values.size
    if sample_size < 2:
        raise ValueError("at least two samples are required to compute a standard error")

    return np.std(values, axis=axis, ddof=1) / np.sqrt(sample_size)


def time_call(func: Callable[..., T], *args: Any, **kwargs: Any) -> tuple[T, float]:
    """Execute a callable and return its result together with elapsed seconds."""

    start = perf_counter()
    result = func(*args, **kwargs)
    elapsed = perf_counter() - start
    return result, elapsed
