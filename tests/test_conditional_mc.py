"""Basic validation tests for the conditional Monte Carlo pricer."""

from __future__ import annotations

import numpy as np
import pytest

from src.black_scholes import black_scholes_call_price
from src.conditional_mc import price_european_option_conditional_mc
from src.utils import EuropeanOption, SABRModelParameters


def test_conditional_mc_matches_black_scholes_when_nu_is_zero() -> None:
    """The nu=0 branch should collapse exactly to Black-Scholes."""

    parameters = SABRModelParameters(
        spot=100.0,
        initial_volatility=0.2,
        beta=1.0,
        vol_of_vol=0.0,
        correlation=-0.3,
        risk_free_rate=0.01,
    )
    option = EuropeanOption(strike=100.0, maturity=1.0, option_type="call")

    result = price_european_option_conditional_mc(
        parameters=parameters,
        option=option,
        n_steps=20,
        n_paths=128,
        seed=123,
        integration_method="trapezoidal",
    )
    bs_price = black_scholes_call_price(
        spot=parameters.spot,
        strike=option.strike,
        maturity=option.maturity,
        rate=parameters.risk_free_rate,
        volatility=parameters.initial_volatility,
    )

    assert result.price == pytest.approx(bs_price, rel=1e-12)
    assert result.standard_error == pytest.approx(0.0, abs=1e-15)
    assert np.allclose(result.pathwise_conditional_prices, bs_price)
