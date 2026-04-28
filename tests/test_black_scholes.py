"""Basic sanity checks for the Black-Scholes helper."""

from __future__ import annotations

import numpy as np
import pytest

from src.black_scholes import black_scholes_call_price


def test_black_scholes_call_price_matches_known_value() -> None:
    """Check a standard at-the-money Black-Scholes call value."""

    price = black_scholes_call_price(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.0,
        volatility=0.2,
    )
    assert price == pytest.approx(7.9655674554, rel=1e-10)


def test_black_scholes_call_price_vectorizes() -> None:
    """Check the vectorized pricing path used by conditional MC."""

    prices = black_scholes_call_price(
        spot=np.array([100.0, 100.0]),
        strike=100.0,
        maturity=1.0,
        rate=0.0,
        volatility=np.array([0.2, 0.3]),
    )
    assert isinstance(prices, np.ndarray)
    assert prices.shape == (2,)
    assert prices[0] == pytest.approx(7.9655674554, rel=1e-10)
    assert prices[1] > prices[0]
