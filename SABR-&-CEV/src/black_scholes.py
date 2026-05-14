"""Black-Scholes helper functions used by conditional Monte Carlo."""

from __future__ import annotations

from math import exp

import numpy as np
from scipy.special import ndtr


def module_status() -> str:
    """Return a short status string for small smoke tests."""

    return "black_scholes helper ready"


def _maybe_scalar(value: np.ndarray) -> float | np.ndarray:
    """Return a Python float for scalar outputs and an array otherwise."""

    if value.ndim == 0:
        return float(value)
    return value


def black_scholes_call_price(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    maturity: float | np.ndarray,
    rate: float | np.ndarray,
    volatility: float | np.ndarray,
) -> float | np.ndarray:
    """Return the Black-Scholes price of a European call option.

    This helper accepts either scalars or NumPy-broadcastable arrays, which
    keeps the conditional Monte Carlo pathwise pricing step fully vectorized.
    For ``volatility == 0``, the terminal asset price is deterministic and we
    return the discounted intrinsic value exactly.
    """

    spot_array, strike_array, maturity_array, rate_array, volatility_array = (
        np.broadcast_arrays(
            np.asarray(spot, dtype=float),
            np.asarray(strike, dtype=float),
            np.asarray(maturity, dtype=float),
            np.asarray(rate, dtype=float),
            np.asarray(volatility, dtype=float),
        )
    )

    if np.any(spot_array <= 0.0):
        raise ValueError("spot must be positive")
    if np.any(strike_array <= 0.0):
        raise ValueError("strike must be positive")
    if np.any(maturity_array < 0.0):
        raise ValueError("maturity must be non-negative")
    if np.any(volatility_array < 0.0):
        raise ValueError("volatility must be non-negative")

    maturity_zero = maturity_array == 0.0
    regular_case = (maturity_array > 0.0) & (volatility_array > 0.0)

    discount_factor = np.exp(-rate_array * maturity_array)
    intrinsic_now = np.maximum(spot_array - strike_array, 0.0)
    deterministic_value = np.maximum(spot_array - strike_array * discount_factor, 0.0)

    safe_spot = np.where(regular_case, spot_array, 1.0)
    safe_strike = np.where(regular_case, strike_array, 1.0)
    safe_maturity = np.where(regular_case, maturity_array, 1.0)
    safe_volatility = np.where(regular_case, volatility_array, 1.0)

    sqrt_maturity = np.sqrt(safe_maturity)
    sigma_root_t = safe_volatility * sqrt_maturity
    d1 = (
        np.log(safe_spot / safe_strike)
        + (rate_array + 0.5 * safe_volatility * safe_volatility) * safe_maturity
    ) / sigma_root_t
    d2 = d1 - sigma_root_t

    regular_value = (
        spot_array * ndtr(d1)
        - strike_array * discount_factor * ndtr(d2)
    )
    call_value = np.where(
        regular_case,
        regular_value,
        np.where(maturity_zero, intrinsic_now, deterministic_value),
    )
    return _maybe_scalar(np.asarray(call_value, dtype=float))


def black_scholes_price(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    maturity: float | np.ndarray,
    rate: float | np.ndarray,
    volatility: float | np.ndarray,
    option_type: str = "call",
) -> float | np.ndarray:
    """Return a European Black-Scholes option price."""

    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    call_price = black_scholes_call_price(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=volatility,
    )
    if option_type == "call":
        return call_price

    maturity_array = np.asarray(maturity, dtype=float)
    discount_factor = np.exp(-np.asarray(rate, dtype=float) * maturity_array)
    put_price = np.asarray(call_price, dtype=float) - np.asarray(spot, dtype=float) + (
        np.asarray(strike, dtype=float) * discount_factor
    )
    return _maybe_scalar(np.asarray(put_price, dtype=float))
