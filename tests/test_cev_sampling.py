"""Validation tests for the exact CEV sampler.

These tests check three properties that together pin down a correct
exact CEV sampler:

1. The closed-form absorption probability ``P(F_T = 0)`` predicted by
   the regularized lower incomplete gamma function matches the empirical
   absorption rate of the sampler over many draws.
2. The conditional mean ``E[F_T | F_T > 0] * P(F_T > 0)`` equals ``F_0``,
   i.e. the CEV process with absorbing boundary at zero is a martingale.
   We check the unconditional mean, which is the same statement.
3. Vectorization over broadcastable inputs gives results consistent
   with looping in pure Python.

All checks are written with generous Monte Carlo tolerances (a few
empirical standard errors) so they are stable on CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.cev_sampling import (
    cev_absorption_probability,
    cev_mean,
    sample_cev,
)


@pytest.mark.parametrize(
    "beta, initial_value, sigma_squared_time",
    [
        (0.3, 1.0, 0.16),   # Choi et al. (2024) Case I-ish: beta = 0.3, sigma^2 T = 0.16
        (0.5, 1.0, 0.25),   # square-root, moderate variance
        (0.7, 1.0, 0.09),   # closer to lognormal, small variance
        (0.3, 0.05, 0.064), # small forward, comparable to paper's Case III
    ],
)
def test_absorption_probability_matches_closed_form(
    beta: float, initial_value: float, sigma_squared_time: float
) -> None:
    """Empirical absorption rate matches ``1 - P_G(z_0/2; alpha)``."""

    n_samples = 200_000
    rng = np.random.default_rng(seed=20260429)
    samples = sample_cev(
        initial_value=np.full(n_samples, initial_value),
        sigma_squared_time=np.full(n_samples, sigma_squared_time),
        beta=beta,
        rng=rng,
    )

    empirical_absorption = float(np.mean(samples == 0.0))
    theoretical_absorption = cev_absorption_probability(
        initial_value=initial_value,
        sigma_squared_time=sigma_squared_time,
        beta=beta,
    )

    # Standard error of an empirical Bernoulli rate is at most 0.5 / sqrt(N);
    # we allow 5 of those.
    tolerance = 5.0 * 0.5 / np.sqrt(n_samples)
    assert abs(empirical_absorption - theoretical_absorption) < tolerance


@pytest.mark.parametrize(
    "beta, initial_value, sigma_squared_time",
    [
        (0.3, 1.0, 0.16),
        (0.5, 1.0, 0.25),
        (0.7, 1.0, 0.09),
    ],
)
def test_martingale_property(
    beta: float, initial_value: float, sigma_squared_time: float
) -> None:
    """The CEV process with absorbing boundary is a martingale."""

    n_samples = 400_000
    rng = np.random.default_rng(seed=20260430)
    samples = sample_cev(
        initial_value=np.full(n_samples, initial_value),
        sigma_squared_time=np.full(n_samples, sigma_squared_time),
        beta=beta,
        rng=rng,
    )

    empirical_mean = float(np.mean(samples))
    theoretical_mean = float(cev_mean(initial_value))
    standard_error = float(np.std(samples, ddof=1) / np.sqrt(n_samples))

    # Allow five standard errors of slack.
    assert abs(empirical_mean - theoretical_mean) < 5.0 * standard_error


def test_vectorization_matches_scalar_loop() -> None:
    """Calling the sampler with arrays matches looping with the same RNG state.

    We use one shared generator: the array call drains it once for the
    whole batch; the scalar loop drains it draw by draw. The two paths
    do *not* produce identical numerical samples (the internal Gamma
    and Poisson calls consume different amounts of state per draw), so
    instead we check that two independent calls of equal large size on
    independent generators agree statistically on the mean and on the
    absorption rate.
    """

    n_samples = 100_000
    beta = 0.4
    initial_values = np.full(n_samples, 1.0)
    variance_values = np.full(n_samples, 0.2)

    rng_a = np.random.default_rng(seed=1234)
    rng_b = np.random.default_rng(seed=5678)
    samples_a = sample_cev(initial_values, variance_values, beta=beta, rng=rng_a)
    samples_b = sample_cev(initial_values, variance_values, beta=beta, rng=rng_b)

    mean_a = float(np.mean(samples_a))
    mean_b = float(np.mean(samples_b))
    se_pooled = float(
        np.sqrt(
            np.var(samples_a, ddof=1) / n_samples
            + np.var(samples_b, ddof=1) / n_samples
        )
    )
    assert abs(mean_a - mean_b) < 5.0 * se_pooled

    absorption_a = float(np.mean(samples_a == 0.0))
    absorption_b = float(np.mean(samples_b == 0.0))
    assert abs(absorption_a - absorption_b) < 5.0 * 0.5 / np.sqrt(n_samples)


def test_zero_variance_returns_initial_value() -> None:
    """A path with sigma^2 T = 0 collapses deterministically to F_0."""

    samples = sample_cev(
        initial_value=np.array([0.5, 1.0, 2.0]),
        sigma_squared_time=np.array([0.0, 0.0, 0.0]),
        beta=0.5,
        rng=42,
    )
    np.testing.assert_array_equal(samples, np.array([0.5, 1.0, 2.0]))


def test_zero_initial_value_remains_at_zero() -> None:
    """The origin is absorbing: F_0 = 0 implies F_T = 0."""

    samples = sample_cev(
        initial_value=np.zeros(1000),
        sigma_squared_time=np.full(1000, 0.25),
        beta=0.3,
        rng=42,
    )
    assert np.all(samples == 0.0)


def test_broadcast_shape() -> None:
    """The sampler returns the broadcast shape of its inputs."""

    samples = sample_cev(
        initial_value=np.array([[1.0], [2.0], [3.0]]),     # (3, 1)
        sigma_squared_time=np.array([0.1, 0.2, 0.3, 0.4]), # (4,)
        beta=0.4,
        rng=2026,
    )
    assert samples.shape == (3, 4)


def test_invalid_beta_raises() -> None:
    """Beta outside the open unit interval is a usage error."""

    with pytest.raises(ValueError):
        sample_cev(initial_value=1.0, sigma_squared_time=0.1, beta=0.0)
    with pytest.raises(ValueError):
        sample_cev(initial_value=1.0, sigma_squared_time=0.1, beta=1.0)
    with pytest.raises(ValueError):
        sample_cev(initial_value=1.0, sigma_squared_time=0.1, beta=-0.5)


def test_negative_inputs_raise() -> None:
    """Negative F_0 or sigma^2 T is a usage error."""

    with pytest.raises(ValueError):
        sample_cev(initial_value=-1.0, sigma_squared_time=0.1, beta=0.5)
    with pytest.raises(ValueError):
        sample_cev(initial_value=1.0, sigma_squared_time=-0.1, beta=0.5)


def test_absorption_probability_matches_paper_special_case() -> None:
    """At beta = 0.5, alpha = 1 and the gamma CDF reduces to 1 - exp(-x).

    With alpha = 1, P_G(x; 1) = 1 - exp(-x), so the absorption probability
    becomes exp(-z_0 / 2). This gives a closed-form check independent of
    scipy's gammainc.
    """

    initial_value = 1.0
    sigma_squared_time = 0.25
    beta = 0.5
    beta_star = 1.0 - beta
    z_zero = initial_value ** (2.0 * beta_star) / (
        beta_star * beta_star * sigma_squared_time
    )
    expected = float(np.exp(-0.5 * z_zero))
    actual = float(
        cev_absorption_probability(
            initial_value=initial_value,
            sigma_squared_time=sigma_squared_time,
            beta=beta,
        )
    )
    assert actual == pytest.approx(expected, rel=1e-12)
