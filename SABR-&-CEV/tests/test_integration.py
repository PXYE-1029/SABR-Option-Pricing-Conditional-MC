"""Sanity checks for the integration helpers."""

from __future__ import annotations

import numpy as np
import pytest

from src.integration import (
    simpson_integrated_variance,
    simpson_rule,
    trapezoidal_integrated_variance,
    trapezoidal_rule,
)


def test_basic_trapezoidal_and_simpson_rules() -> None:
    """Check simple one-dimensional integration examples."""

    values = np.array([0.0, 1.0, 4.0])
    assert trapezoidal_rule(values, 1.0) == pytest.approx(3.0)
    assert simpson_rule(values, 1.0) == pytest.approx(8.0 / 3.0)


def test_integrated_variance_for_constant_volatility_path() -> None:
    """Constant sigma should give an exact integrated variance."""

    time_grid = np.linspace(0.0, 1.0, 5)
    sigma_path = np.full_like(time_grid, 2.0)

    trap = trapezoidal_integrated_variance(sigma_path, time_grid)
    simp = simpson_integrated_variance(sigma_path, time_grid)

    assert trap == pytest.approx(4.0)
    assert simp == pytest.approx(4.0)
