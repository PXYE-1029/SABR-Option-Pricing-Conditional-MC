"""Exact sampling of the constant-elasticity-of-variance (CEV) distribution.

Implements the shifted-Poisson-mixture Gamma representation of
Makarov and Glew (2010, Section 3.3) together with Kang (2014,
'Algorithm 3') as summarized in Algorithm 3 of Choi, Hu, and Kwok (2024,
arXiv:2408.01898). The CEV process

    ``dF_t / F_t^beta = sigma_0 dW_t``,    0 < beta < 1,

is augmented with an absorbing boundary at the origin. Conditional on
``F_0`` and ``sigma_0^2 T``, the terminal price ``F_T`` is sampled
exactly using a constant number of elementary random variates per draw,
without any inverse-CDF root finding.

The sampling layer here is fully vectorized over broadcastable
``initial_value`` and ``sigma_squared_time`` inputs. It is meant to slot
in unchanged as the per-step asset sampler of a two-step
Choi-Hu-Kwok-style simulator (e.g. for SABR with 0 < beta < 1, or for the
Heston-CEV extension developed in this project).
"""

from __future__ import annotations

import numpy as np

from .utils import get_rng


__all__ = [
    "sample_cev",
    "cev_absorption_probability",
    "cev_mean",
]


def sample_cev(
    initial_value: float | np.ndarray,
    sigma_squared_time: float | np.ndarray,
    beta: float,
    rng: int | np.random.Generator | None = None,
) -> np.ndarray:
    r"""Sample ``F_T \sim CEV_beta(F_0, sigma_0^2 T)`` exactly.

    Parameters
    ----------
    initial_value:
        ``F_0``. Scalar or NumPy-broadcastable array. Must be
        non-negative; zero entries are returned as-is (origin is
        absorbing).
    sigma_squared_time:
        ``sigma_0^2 * T``. Scalar or NumPy-broadcastable array. Must be
        non-negative; zero entries collapse to deterministic ``F_0``.
    beta:
        CEV elasticity in the open interval ``(0, 1)``. The boundary
        cases ``beta = 0`` (normal) and ``beta = 1`` (lognormal) are not
        handled by this routine because they admit much simpler exact
        samplers and the transformation below is singular at the
        endpoints.
    rng:
        Either ``None``, an integer seed, or an existing
        ``numpy.random.Generator``. The generator is forwarded
        unchanged when one is supplied.

    Returns
    -------
    numpy.ndarray
        Samples with the broadcast shape of the two array inputs. Paths
        absorbed at the origin appear as exact zeros. A scalar pair of
        inputs therefore returns a *single* sample; to draw ``N``
        independent samples with the same parameters, broadcast against
        an array of length ``N`` (e.g. pass ``np.full(N, F_0)``).

    Notes
    -----
    The algorithm proceeds in three steps for each path. Let
    ``alpha = 1 / (2 * (1 - beta))`` and
    ``z_0 = F_0^{2(1 - beta)} / ((1 - beta)^2 * sigma_0^2 * T)``.

    1. Draw ``X ~ Gamma(alpha, 1)``.
    2. If ``X >= z_0 / 2``, the path is absorbed and ``F_T = 0``. The
       probability of this event is exactly the CEV mass at zero,
       ``1 - P_G(z_0/2; alpha)``, where ``P_G`` is the regularized lower
       incomplete gamma function.
    3. Otherwise, draw ``N ~ Poisson(z_0/2 - X)`` and
       ``z_T ~ 2 * Gamma(N + 1, 1)``. Then
       ``F_T = ((1 - beta)^2 * sigma_0^2 * T * z_T) ** (1 / (2(1-beta)))``.

    Step 2 uses the optimization noted in Choi et al. (2024, Remark 9):
    when sampling ``N`` is purely an intermediate step toward ``F_T``,
    the rejection inherent in the shifted-Poisson distribution coincides
    *exactly* with the CEV absorption event, so a single ``Gamma``
    draw suffices instead of repeated rejection sampling.
    """

    if not 0.0 < beta < 1.0:
        raise ValueError(
            "beta must lie strictly in (0, 1); for beta = 0 use a normal "
            "sampler and for beta = 1 use a lognormal sampler."
        )

    generator = get_rng(rng)

    initial_array = np.asarray(initial_value, dtype=float)
    variance_array = np.asarray(sigma_squared_time, dtype=float)
    if np.any(initial_array < 0.0):
        raise ValueError("initial_value must be non-negative")
    if np.any(variance_array < 0.0):
        raise ValueError("sigma_squared_time must be non-negative")

    broadcast_initial, broadcast_variance = np.broadcast_arrays(
        initial_array, variance_array
    )
    output_shape = broadcast_initial.shape
    flat_initial = np.ascontiguousarray(broadcast_initial, dtype=float).ravel()
    flat_variance = np.ascontiguousarray(broadcast_variance, dtype=float).ravel()
    n_samples = flat_initial.size

    samples = np.zeros(n_samples, dtype=float)

    # Degenerate paths handled exactly without invoking the CEV sampler.
    deterministic_mask = (flat_variance == 0.0) | (flat_initial == 0.0)
    samples[deterministic_mask] = flat_initial[deterministic_mask]
    active_mask = ~deterministic_mask
    if not np.any(active_mask):
        return samples.reshape(output_shape)

    active_initial = flat_initial[active_mask]
    active_variance = flat_variance[active_mask]
    n_active = active_initial.size

    beta_star = 1.0 - beta
    alpha = 1.0 / (2.0 * beta_star)

    z_zero = (active_initial ** (2.0 * beta_star)) / (
        beta_star * beta_star * active_variance
    )
    half_z_zero = 0.5 * z_zero

    # Step 1: outer Gamma draw shared by absorption and survival branches.
    gamma_draw = generator.standard_gamma(shape=alpha, size=n_active)

    # Step 2: paths with X >= z_0 / 2 hit the absorbing boundary exactly.
    survives = gamma_draw < half_z_zero
    if np.any(survives):
        survival_indices = np.flatnonzero(survives)
        poisson_intensity = half_z_zero[survives] - gamma_draw[survives]
        # NumPy's Poisson sampler rejects lam > ~9.2e18 (its internal
        # int64 limit). For such pathological intensities -- which
        # only arise in aggressive parameter regimes that push the
        # asset close to the absorbing boundary -- we fall back to a
        # Gaussian approximation of Poisson, exact to leading order
        # in 1/sqrt(lam). This degrades sampling far less than letting
        # the simulator crash.
        SAFE_POISSON_LAM = 1.0e18
        too_large = poisson_intensity > SAFE_POISSON_LAM
        poisson_counts = np.empty_like(poisson_intensity, dtype=float)
        if np.any(~too_large):
            poisson_counts[~too_large] = generator.poisson(
                lam=poisson_intensity[~too_large]
            )
        if np.any(too_large):
            large_lam = poisson_intensity[too_large]
            normal_draws = generator.standard_normal(size=large_lam.shape)
            poisson_counts[too_large] = np.maximum(
                np.round(large_lam + np.sqrt(large_lam) * normal_draws),
                0.0,
            )
        z_terminal = 2.0 * generator.standard_gamma(
            shape=poisson_counts + 1.0, size=poisson_counts.shape
        )
        terminal_values = (
            beta_star * beta_star * active_variance[survives] * z_terminal
        ) ** (1.0 / (2.0 * beta_star))

        active_indices = np.flatnonzero(active_mask)
        samples[active_indices[survival_indices]] = terminal_values

    return samples.reshape(output_shape)


def cev_absorption_probability(
    initial_value: float | np.ndarray,
    sigma_squared_time: float | np.ndarray,
    beta: float,
) -> float | np.ndarray:
    r"""Return the closed-form probability ``P(F_T = 0)`` under the CEV model.

    Uses the identity ``P(F_T = 0) = 1 - P_G(z_0 / 2; alpha)`` with
    ``alpha = 1 / (2(1 - beta))`` and
    ``z_0 = F_0^{2(1-beta)} / ((1-beta)^2 sigma_0^2 T)`` from
    Choi et al. (2024, Eq. (22b)). ``P_G`` is the regularized lower
    incomplete gamma function. This is the exact target distribution
    that ``sample_cev`` matches and is used as a closed-form benchmark
    in the unit tests.
    """

    if not 0.0 < beta < 1.0:
        raise ValueError("beta must lie strictly in (0, 1)")

    from scipy.special import gammainc

    initial_array = np.asarray(initial_value, dtype=float)
    variance_array = np.asarray(sigma_squared_time, dtype=float)
    beta_star = 1.0 - beta
    alpha = 1.0 / (2.0 * beta_star)

    z_zero = (initial_array ** (2.0 * beta_star)) / (
        beta_star * beta_star * variance_array
    )
    survival_probability = gammainc(alpha, 0.5 * z_zero)
    absorption_probability = 1.0 - survival_probability

    if absorption_probability.ndim == 0:
        return float(absorption_probability)
    return np.asarray(absorption_probability, dtype=float)


def cev_mean(initial_value: float | np.ndarray) -> float | np.ndarray:
    """Return ``E[F_T] = F_0`` for the absorbed CEV distribution.

    The CEV process is a martingale under absorption at the origin, so
    its mean is the initial forward price. This trivial helper exists so
    that downstream code and tests have a single canonical reference
    point for the martingale check.
    """

    return np.asarray(initial_value, dtype=float)
