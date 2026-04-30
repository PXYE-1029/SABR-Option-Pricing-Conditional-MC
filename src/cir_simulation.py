"""CIR variance simulation with Andersen's Quadratic-Exponential scheme.

This module provides a self-contained CIR simulator that returns, for
each time step, both the next variance ``v_{t+h}`` and an estimate of
the integrated variance ``int_t^{t+h} v_s ds`` over the step. It is
used as a fallback when PyFENG is unavailable and as a unit-test
reference for the main CEV-Heston simulator.

Reference: Andersen (2008), "Simple and efficient simulation of the
Heston stochastic volatility model", Journal of Computational Finance.
"""

from __future__ import annotations

import numpy as np

from .utils import CEVHestonModelParameters, get_rng


PSI_CRITICAL_DEFAULT = 1.5


def cir_qe_step(
    v_current: np.ndarray,
    dt: float,
    kappa: float,
    theta: float,
    xi: float,
    rng: np.random.Generator,
    psi_critical: float = PSI_CRITICAL_DEFAULT,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Advance a CIR variance by one step with the Andersen QE scheme.

    Given ``v_t`` (broadcastable array), returns the pair
    ``(v_{t+dt}, integrated_variance_dt)``, where the integrated
    variance is approximated by the trapezoidal rule
    ``0.5 * dt * (v_t + v_{t+dt})``. This is the same approximation used
    in Andersen (2008, Sec. 4.4) for computing the cross-term in the
    log-asset update; it is exact to leading order in ``dt`` and gives a
    consistent ``O(dt)`` estimator of the integrated variance suitable
    for the Choi-Hu-Kwok-style conditional pricing developed elsewhere
    in this package.

    The QE step itself (sampling ``v_{t+dt}`` given ``v_t``) is exact
    to first two moments of the CIR transition density and behaves
    correctly across both the Feller and non-Feller regimes.
    """

    if dt <= 0.0:
        raise ValueError("dt must be positive")

    v_current = np.asarray(v_current, dtype=float)
    n = v_current.size

    exp_kappa_dt = float(np.exp(-kappa * dt))
    one_minus_exp = 1.0 - exp_kappa_dt
    xi_sq = xi * xi

    conditional_mean = theta + (v_current - theta) * exp_kappa_dt
    if xi == 0.0:
        # Deterministic limit: variance follows the mean-reversion ODE.
        v_next = conditional_mean
    else:
        conditional_variance = (
            v_current * (xi_sq / kappa) * exp_kappa_dt * one_minus_exp
            + theta * (xi_sq / (2.0 * kappa)) * one_minus_exp * one_minus_exp
        )

        psi = np.where(
            conditional_mean > 0.0,
            conditional_variance / np.maximum(conditional_mean * conditional_mean, 1e-300),
            np.inf,
        )

        v_next = np.empty_like(conditional_mean)

        quadratic_branch = psi <= psi_critical
        if np.any(quadratic_branch):
            psi_q = psi[quadratic_branch]
            two_over_psi = 2.0 / psi_q
            b_squared = two_over_psi - 1.0 + np.sqrt(two_over_psi * (two_over_psi - 1.0))
            a = conditional_mean[quadratic_branch] / (1.0 + b_squared)
            z = rng.standard_normal(size=psi_q.shape)
            b = np.sqrt(b_squared)
            v_next[quadratic_branch] = a * (b + z) ** 2

        exponential_branch = ~quadratic_branch
        if np.any(exponential_branch):
            psi_e = psi[exponential_branch]
            mean_e = conditional_mean[exponential_branch]
            p = (psi_e - 1.0) / (psi_e + 1.0)
            beta_e = (1.0 - p) / np.maximum(mean_e, 1e-300)
            uniform = rng.uniform(size=psi_e.shape)
            below = uniform <= p
            sample = np.empty_like(uniform)
            sample[below] = 0.0
            mask_above = ~below
            sample[mask_above] = (
                np.log((1.0 - p[mask_above]) / np.maximum(1.0 - uniform[mask_above], 1e-300))
                / beta_e[mask_above]
            )
            v_next[exponential_branch] = sample

        v_next = np.maximum(v_next, 0.0)

    integrated_variance = 0.5 * dt * (v_current + v_next)
    return v_next, integrated_variance


class AndersenQEVarianceSimulator:
    """A simple variance-stepping interface compatible with the main
    CEV-Heston simulator. Each call to ``step`` advances the variance by
    ``dt`` and returns ``(v_next, integrated_variance_over_dt)``.

    This class is the fallback used in this repository when PyFENG is
    unavailable. The main simulator can also accept any object that
    exposes a ``step(dt, v_current)`` method with the same return
    contract; see ``HestonCEVSimulator`` for details.
    """

    def __init__(
        self,
        parameters: CEVHestonModelParameters,
        rng: int | np.random.Generator | None = None,
        psi_critical: float = PSI_CRITICAL_DEFAULT,
    ) -> None:
        self._parameters = parameters
        self._rng = get_rng(rng)
        self._psi_critical = psi_critical
        self.backend_name = "andersen_qe"

    def step(
        self, dt: float, v_current: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return cir_qe_step(
            v_current=v_current,
            dt=dt,
            kappa=self._parameters.kappa,
            theta=self._parameters.theta,
            xi=self._parameters.xi,
            rng=self._rng,
            psi_critical=self._psi_critical,
        )
