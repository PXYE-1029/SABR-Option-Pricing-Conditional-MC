"""Shared data structures and small validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Literal, TypeVar

import numpy as np


T = TypeVar("T")
OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class SABRModelParameters:
    spot: float
    initial_volatility: float
    beta: float
    vol_of_vol: float
    correlation: float
    risk_free_rate: float = 0.0


@dataclass(frozen=True)
class EuropeanOption:
    strike: float
    maturity: float
    option_type: OptionType = "call"


@dataclass(frozen=True)
class CEVHestonModelParameters:
    """Parameters for the CEV-Heston (beta-Heston) hybrid model.

    The forward price ``F_t`` and instantaneous variance ``v_t`` follow

        dF_t / F_t^beta = sqrt(v_t) dW^F_t,
        dv_t            = kappa (theta - v_t) dt + xi sqrt(v_t) dW^v_t,

    with ``corr(dW^F, dW^v) = rho``. The boundary ``F = 0`` is absorbing
    (consistent with the CEV literature). When ``beta = 1`` the model
    reduces to the standard Heston model; when ``xi = 0`` it reduces to
    the deterministic-variance CEV model with mean-reverting variance;
    when ``rho = 0`` the conditional terminal price is exactly CEV.
    """

    spot: float
    initial_variance: float
    kappa: float
    theta: float
    xi: float
    correlation: float
    beta: float = 1.0
    risk_free_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.spot <= 0.0:
            raise ValueError("spot must be positive")
        if self.initial_variance <= 0.0:
            raise ValueError("initial_variance must be positive")
        if self.kappa <= 0.0:
            raise ValueError("kappa must be positive")
        if self.theta <= 0.0:
            raise ValueError("theta must be positive")
        if self.xi < 0.0:
            raise ValueError("xi (vol-of-vol) must be non-negative")
        if not -1.0 <= self.correlation <= 1.0:
            raise ValueError("correlation must lie in [-1, 1]")
        if not 0.0 < self.beta <= 1.0:
            raise ValueError("beta must lie in (0, 1]")

    @property
    def feller_ratio(self) -> float:
        """``2 kappa theta / xi^2``; the Feller condition holds iff this is >= 1."""

        if self.xi == 0.0:
            return float("inf")
        return 2.0 * self.kappa * self.theta / (self.xi * self.xi)


def get_rng(seed: int | np.random.Generator | None = None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def standard_error(samples: np.ndarray, axis: int | None = None) -> float | np.ndarray:
    values = np.asarray(samples, dtype=float)
    sample_size = values.shape[axis] if axis is not None else values.size
    return np.std(values, axis=axis, ddof=1) / np.sqrt(sample_size)


def time_call(func: Callable[..., T], *args: Any, **kwargs: Any) -> tuple[T, float]:
    start = perf_counter()
    result = func(*args, **kwargs)
    elapsed = perf_counter() - start
    return result, elapsed
