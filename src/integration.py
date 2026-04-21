"""Numerical integration helpers for uniformly spaced grids.

Array shape convention:
- a single path or time series may be one-dimensional with shape ``(n_steps + 1,)``
- collections of paths use shape ``(n_paths, n_steps + 1)``
- integration is always carried out along the last axis
"""

from __future__ import annotations

import numpy as np


def module_status() -> str:
    """Return a short status string for small smoke tests."""

    return "integration helpers ready"


def _validate_uniform_time_grid(time_grid: np.ndarray) -> float:
    """Validate a one-dimensional uniform time grid and return its spacing."""

    grid = np.asarray(time_grid, dtype=float)
    if grid.ndim != 1:
        raise ValueError("time_grid must be one-dimensional")
    if grid.size < 2:
        raise ValueError("time_grid must contain at least two points")

    increments = np.diff(grid)
    if np.any(increments <= 0.0):
        raise ValueError("time_grid must be strictly increasing")

    dt = increments[0]
    if not np.allclose(increments, dt):
        raise ValueError("time_grid must be uniformly spaced")

    return float(dt)


def trapezoidal_rule(values: np.ndarray, dx: float) -> float | np.ndarray:
    """Integrate uniformly spaced samples with the trapezoidal rule."""

    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        raise ValueError("values must be at least one-dimensional")
    if array.shape[-1] < 2:
        raise ValueError("at least two grid values are required")
    if dx <= 0.0:
        raise ValueError("dx must be positive")

    return dx * (
        0.5 * array[..., 0]
        + np.sum(array[..., 1:-1], axis=-1)
        + 0.5 * array[..., -1]
    )


def simpson_rule(values: np.ndarray, dx: float) -> float | np.ndarray:
    """Integrate uniformly spaced samples with Simpson's rule.

    Simpson's rule requires an even number of intervals, so the number of grid
    values along the last axis must be odd.
    """

    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        raise ValueError("values must be at least one-dimensional")
    if array.shape[-1] < 3:
        raise ValueError("at least three grid values are required for Simpson's rule")
    if (array.shape[-1] - 1) % 2 != 0:
        raise ValueError("Simpson's rule requires an even number of intervals")
    if dx <= 0.0:
        raise ValueError("dx must be positive")

    return (dx / 3.0) * (
        array[..., 0]
        + array[..., -1]
        + 4.0 * np.sum(array[..., 1:-1:2], axis=-1)
        + 2.0 * np.sum(array[..., 2:-2:2], axis=-1)
    )


def trapezoidal_integrated_variance(
    volatility_path: np.ndarray,
    time_grid: np.ndarray,
) -> float | np.ndarray:
    r"""Approximate ``\int_0^T \sigma_t^2 \, dt`` with the trapezoidal rule.

    ``volatility_path`` can be a single path with shape ``(n_steps + 1,)`` or a
    collection of paths with shape ``(n_paths, n_steps + 1)``.
    """

    volatility_values = np.asarray(volatility_path, dtype=float)
    dt = _validate_uniform_time_grid(time_grid)
    if volatility_values.shape[-1] != len(time_grid):
        raise ValueError("volatility_path and time_grid must have matching lengths")

    return trapezoidal_rule(np.square(volatility_values), dt)


def simpson_integrated_variance(
    volatility_path: np.ndarray,
    time_grid: np.ndarray,
) -> float | np.ndarray:
    r"""Approximate ``\int_0^T \sigma_t^2 \, dt`` with Simpson's rule.

    ``volatility_path`` can be a single path with shape ``(n_steps + 1,)`` or a
    collection of paths with shape ``(n_paths, n_steps + 1)``.
    """

    volatility_values = np.asarray(volatility_path, dtype=float)
    dt = _validate_uniform_time_grid(time_grid)
    if volatility_values.shape[-1] != len(time_grid):
        raise ValueError("volatility_path and time_grid must have matching lengths")

    return simpson_rule(np.square(volatility_values), dt)
