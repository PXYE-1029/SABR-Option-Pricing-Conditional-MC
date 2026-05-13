"""Islah-style approximation of CEV-Heston for direct comparison.

This module implements the conditional-distribution approximation of
Islah (2009) ported to the CEV-Heston setting. Islah's approximation
is the de facto baseline that almost every SABR simulator in the
literature has used since 2009 (Chen et al. 2012, Cai et al. 2017,
Leitao et al. 2017a/b, Grzelak et al. 2019, Cui et al. 2021,
Kyriakou et al. 2023). Choi, Hu, and Kwok (2024) showed that this
baseline systematically violates the martingale property of the
forward price, with errors that accumulate roughly linearly in T
(see their Figure 3a). Their headline result is that the alternative
CEV approximation (which we implement in ``heston_cev_simulation``)
preserves the martingale by construction.

This module exists only so that we can reproduce that comparison on
the Heston-CEV side: any experiment that quantifies the martingale
advantage of our scheme needs Islah as a reference baseline. We do
not recommend it for production use.

The Islah approximation in the SABR setting (Choi et al. 2024,
Eq. 25) is

    P(F_{t+h} >= y | sigma_{t+h}, I_t^h)
        = P_chi2( z'_t ; (1 - beta_* rho^2) / (beta_* rho_*^2), z(y) ),

where the modified degree of freedom uses ``(1 - beta_* rho^2) /
(beta_* rho_*^2)`` rather than the ``1/beta_*`` of the genuine CEV
distribution. The conditional non-centrality parameter ``z'_t`` is

    z'_t = ( F_t^{beta_*} + (beta_* rho / nu)(sigma_{t+h} - sigma_t) )^2
              / ( beta_*^2 rho_*^2 sigma_t^2 h I_t^h ).

We port this directly to Heston-CEV by replacing
``(sigma_{t+h} - sigma_t)/nu`` (the SABR Ito integral
``int dZ`` for the lognormal volatility process) by the corresponding
CIR Ito integral ``int sqrt(v) dW^v``, which is closed-form via

    int_t^{t+h} sqrt(v_s) dW^v_s
       = ( v_{t+h} - v_t - kappa theta h + kappa IV_t^h ) / xi.

The dimensional integrated variance ``sigma_t^2 h I_t^h`` becomes
``IV_t^h := int v ds``.

The exact CEV sampler is reused via the algebraic identity in
Choi et al. (2024, Appendix B): a power transformation of the
Islah-distributed ``F_{t+h}`` follows a true CEV distribution with a
modified elasticity ``beta'`` and modified scale; we sample that CEV
variate and invert the transform.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cev_sampling import sample_cev
from .heston_cev_simulation import (
    DEFAULT_EPSILON_CLAMP,
    HestonCEVSimulationResult,
    VarianceStepper,
    _build_default_variance_stepper,
)
from .utils import CEVHestonModelParameters, EuropeanOption, get_rng, standard_error


__all__ = [
    "simulate_heston_cev_islah",
    "price_european_option_heston_cev_islah",
]


def _islah_modified_elasticity(beta: float, rho: float) -> float:
    r"""Return Islah's modified elasticity ``beta'``.

    Defined by Choi et al. (2024, Appendix B):
    ``beta' = beta / (1 - beta_* rho^2)`` with ``beta_* = 1 - beta``.
    Always satisfies ``beta <= beta' <= 1``.
    """

    beta_star = 1.0 - beta
    return beta / (1.0 - beta_star * rho * rho)


def _conditional_mean_islah(
    F_t: np.ndarray,
    v_t: np.ndarray,
    v_next: np.ndarray,
    integrated_variance: np.ndarray,
    parameters: CEVHestonModelParameters,
    dt: float,
    epsilon_clamp: float,
) -> tuple[np.ndarray, int]:
    r"""Compute Islah's conditional mean.

    Islah's approximation puts ``F_{t+h}`` on a CEV distribution with
    elasticity ``beta'`` (different from the true ``beta``) and an
    initial value of

        F-bar'_t^h = ( F_t^{beta_*} + beta_* rho / xi * sqrt_v_integral )^{1/beta'_*}.

    The ``epsilon_clamp`` here protects the same numerical pathology
    as in the main module: ``F_t^{beta_*}`` near the absorbing
    boundary, exponentiated through ``1/beta'_*``, can blow up.
    """

    rho = parameters.correlation
    xi = parameters.xi
    beta = parameters.beta
    beta_star = 1.0 - beta
    beta_prime = _islah_modified_elasticity(beta, rho)
    beta_prime_star = 1.0 - beta_prime

    if xi == 0.0:
        sqrt_v_integral = np.zeros_like(F_t)
    else:
        sqrt_v_integral = (
            v_next - v_t - parameters.kappa * parameters.theta * dt
            + parameters.kappa * integrated_variance
        ) / xi

    F_for_denom = np.maximum(F_t, epsilon_clamp)
    n_clamped = int(np.sum(F_t < epsilon_clamp))

    # Choi et al. (2024) Appendix B: F_bar' is the location of the CEV
    # distribution at modified elasticity beta_prime, with the
    # additional pre-factor beta_*'/beta_*. The SABR-side coefficient
    # ``(sigma_{t+h} - sigma_t) / nu`` is the Heston CIR Ito integral
    # ``int sqrt(v) dW^v_s`` (already absorbed into sqrt_v_integral).
    inner = F_for_denom ** beta_star + beta_star * rho * sqrt_v_integral
    inner_safe = np.maximum(inner, 0.0)
    scale = beta_prime_star / beta_star
    F_bar_prime = (scale * inner_safe) ** (1.0 / beta_prime_star)
    return F_bar_prime, n_clamped


def simulate_heston_cev_islah(
    parameters: CEVHestonModelParameters,
    maturity: float,
    n_steps: int,
    n_paths: int,
    seed: int | np.random.Generator | None = None,
    variance_stepper: VarianceStepper | None = None,
    epsilon_clamp: float = DEFAULT_EPSILON_CLAMP,
) -> HestonCEVSimulationResult:
    """Simulate ``F_T`` under CEV-Heston using Islah's approximation.

    Returns the same ``HestonCEVSimulationResult`` shape as
    ``simulate_heston_cev_terminal`` so the two are drop-in
    interchangeable inside experiments.
    """

    if maturity <= 0.0:
        raise ValueError("maturity must be positive")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")

    rng = get_rng(seed)
    seed_for_pyfeng = seed if isinstance(seed, int) else None
    dt = maturity / n_steps

    if variance_stepper is None:
        variance_stepper, backend_name = _build_default_variance_stepper(
            parameters=parameters,
            rng=rng,
            n_path=n_paths,
            dt=dt,
            seed=seed_for_pyfeng,
        )
    else:
        backend_name = getattr(
            variance_stepper, "backend_name", type(variance_stepper).__name__
        )

    beta_prime = _islah_modified_elasticity(parameters.beta, parameters.correlation)

    F = np.full(n_paths, parameters.spot, dtype=float)
    v = np.full(n_paths, parameters.initial_variance, dtype=float)

    n_clamp_total = 0
    n_active_total = 0
    active_count_per_step: list[int] = []

    rho_perp_squared = 1.0 - parameters.correlation ** 2

    for _ in range(n_steps):
        v_next, integrated_variance = variance_stepper.step(dt, v)
        v_next = np.asarray(v_next, dtype=float)
        integrated_variance = np.maximum(np.asarray(integrated_variance, dtype=float), 0.0)

        active = F > 0.0
        n_active_step = int(np.sum(active))
        active_count_per_step.append(n_active_step)
        n_active_total += n_active_step
        if n_active_step == 0:
            v = v_next
            continue

        F_active = F[active]
        v_active = v[active]
        v_next_active = v_next[active]
        iv_active = integrated_variance[active]

        if parameters.xi == 0.0:
            # In the deterministic-variance limit the variance path
            # carries no information about the Brownian component W^v.
            # Correlation is therefore distributionally irrelevant:
            # use the original CEV elasticity and the full integrated
            # variance, instead of the Islah correlated transform.
            F_bar_active = F_active
            n_clamped_step = 0
            effective_variance = iv_active
            if parameters.beta == 1.0:
                z = rng.standard_normal(size=F_bar_active.shape)
                log_inc = (
                    np.sqrt(np.maximum(effective_variance, 0.0)) * z
                    - 0.5 * effective_variance
                )
                F_next_active = F_bar_active * np.exp(log_inc)
            else:
                F_next_active = sample_cev(
                    initial_value=F_bar_active,
                    sigma_squared_time=effective_variance,
                    beta=parameters.beta,
                    rng=rng,
                )
            F[active] = F_next_active
            v = v_next
            continue

        F_bar_active, n_clamped_step = _conditional_mean_islah(
            F_t=F_active,
            v_t=v_active,
            v_next=v_next_active,
            integrated_variance=iv_active,
            parameters=parameters,
            dt=dt,
            epsilon_clamp=epsilon_clamp,
        )
        n_clamp_total += n_clamped_step

        # Sample the CEV variate at the *modified* elasticity beta'.
        # Per Choi et al. (2024) Appendix B, F_{t+h} is then recovered
        # by the inverse power transform
        #     F_{t+h} = (beta_* / beta_*')^{1/beta_*} * Y^{beta_*' / beta_*}.
        beta = parameters.beta
        beta_star = 1.0 - beta
        beta_prime_star = 1.0 - beta_prime
        effective_variance = rho_perp_squared * iv_active
        if beta_prime == 1.0:
            z = rng.standard_normal(size=F_bar_active.shape)
            log_inc = (
                np.sqrt(np.maximum(effective_variance, 0.0)) * z
                - 0.5 * effective_variance
            )
            Y_sample = F_bar_active * np.exp(log_inc)
        else:
            Y_sample = sample_cev(
                initial_value=F_bar_active,
                sigma_squared_time=effective_variance,
                beta=beta_prime,
                rng=rng,
            )

        # Inverse power transform back to F_{t+h}.
        prefactor = (beta_star / beta_prime_star) ** (1.0 / beta_star)
        exponent = beta_prime_star / beta_star
        F_next_active = prefactor * (Y_sample ** exponent)

        F[active] = F_next_active
        v = v_next

    n_paths_absorbed = int(np.sum(F == 0.0))
    clamp_trigger_rate = (
        float(n_clamp_total) / float(n_active_total) if n_active_total > 0 else 0.0
    )

    diagnostics = {
        "active_count_per_step": active_count_per_step,
        "n_clamp_triggered": n_clamp_total,
        "n_active_total": n_active_total,
        "variance_backend": backend_name,
        "scheme": "islah",
        "modified_elasticity_beta_prime": beta_prime,
    }

    return HestonCEVSimulationResult(
        terminal_prices=F,
        n_paths_absorbed=n_paths_absorbed,
        clamp_trigger_rate=clamp_trigger_rate,
        parameters=parameters,
        diagnostics=diagnostics,
    )


@dataclass(frozen=True)
class HestonCEVIslahPricingResult:
    """Container for European-option pricing under Islah-style CEV-Heston."""

    price: float
    standard_error: float
    discounted_payoffs: np.ndarray
    simulation: HestonCEVSimulationResult


def price_european_option_heston_cev_islah(
    parameters: CEVHestonModelParameters,
    option: EuropeanOption,
    n_steps: int,
    n_paths: int,
    seed: int | np.random.Generator | None = None,
    variance_stepper: VarianceStepper | None = None,
    epsilon_clamp: float = DEFAULT_EPSILON_CLAMP,
) -> HestonCEVIslahPricingResult:
    """Price a European option under Islah-style CEV-Heston."""

    if option.option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    simulation = simulate_heston_cev_islah(
        parameters=parameters,
        maturity=option.maturity,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
        variance_stepper=variance_stepper,
        epsilon_clamp=epsilon_clamp,
    )
    terminal = simulation.terminal_prices
    if option.option_type == "call":
        payoffs = np.maximum(terminal - option.strike, 0.0)
    else:
        payoffs = np.maximum(option.strike - terminal, 0.0)

    discount_factor = float(np.exp(-parameters.risk_free_rate * option.maturity))
    discounted = discount_factor * payoffs

    return HestonCEVIslahPricingResult(
        price=float(np.mean(discounted)),
        standard_error=float(standard_error(discounted)),
        discounted_payoffs=discounted,
        simulation=simulation,
    )
