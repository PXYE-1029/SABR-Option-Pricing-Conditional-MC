"""Step-by-step conditional simulation of the CEV-Heston model.

This module ports the two-step Choi-Hu-Kwok (2024) simulation framework
for the SABR model to the CEV-Heston (a.k.a. beta-Heston) hybrid:

    dF_t / F_t^beta = sqrt(v_t) dW^F_t,
    dv_t            = kappa (theta - v_t) dt + xi sqrt(v_t) dW^v_t,
    corr(dW^F, dW^v) = rho.

The forward price ``F_t`` and the variance ``v_t`` are evolved jointly
over each step ``[t, t + h]`` in three sub-steps:

  Step 1.  Sample ``v_{t+h}`` and the integrated variance
           ``IV_t^h := int_t^{t+h} v_s ds`` from the CIR variance
           process. The recommended PyFENG-backed path is to delegate to
           PyFENG's peer-reviewed implementation of the Choi-Kwok
           (2024) Poisson-conditioning scheme via
           ``src.pyfeng_adapter.PyFengVarianceStepper``. When PyFENG
           is not installed (e.g. in offline CI) the simulator falls
           back to the bundled Andersen QE scheme in
           ``src.cir_simulation``; the asset-side mathematics is
           independent of which stepper is chosen.

  Step 2.  Compute the conditional mean ``F-bar_t^h`` of the asset over
           the step, using a CIR-aware adaptation of Choi et al. (2024,
           Eq. 16b). The key identity used is the CIR Ito formula

               int_t^{t+h} sqrt(v_s) dW^v_s
                  = ( v_{t+h} - v_t - kappa theta h + kappa IV_t^h ) / xi,

           which is closed-form once ``v_{t+h}`` and ``IV_t^h`` have
           been sampled. This gives the martingale-preserving local
           drift / convexity decomposition without any ad-hoc
           empirical-martingale correction.

  Step 3.  Sample ``F_{t+h}`` exactly from the CEV distribution
           ``CEV_beta(F-bar_t^h, (1 - rho^2) IV_t^h)`` using the
           shifted-Poisson-mixture-Gamma sampler in ``cev_sampling``.
           When ``beta = 1`` we use the closed-form lognormal update
           directly and skip the CEV sampler. When ``rho = 0`` the
           drift / convexity terms vanish and the conditional law is
           exactly CEV (no frozen-coefficient error).

A few engineering safeguards are layered on top of the math:

  * The frozen-coefficient denominator ``F_t^{1-beta}`` is clamped at
    a small positive ``epsilon_clamp`` to prevent a single near-zero
    path from producing a ``np.exp(huge)`` outlier that wrecks the
    sample mean. The default clamp is ``1e-8``, which is well below
    typical ``F`` values in our experiments and triggers on
    ``O(0.01%)`` of paths or fewer; the simulator records the trigger
    rate as a diagnostic so the user can verify this. Paths returned
    as exact zero by the CEV sampler are treated as fully absorbed and
    bypass the clamp branch entirely.

  * The discount factor ``exp(-rT)`` is *not* applied inside the
    pathwise simulator. The simulator returns the un-discounted
    ``F_T`` so that the martingale check ``E[F_T] = F_0`` is direct;
    discounting for option pricing is applied separately by the
    pricing helper.

  * When ``beta = 1`` and ``rho = 0`` and ``xi = 0`` we fall through
    to the closed-form Black-Scholes setting; this is used by the
    sanity tests in this repository.

References
----------
Choi, J., Hu, L., Kwok, Y. K. (2024). Efficient and accurate simulation
of the SABR model. arXiv:2408.01898v2. (See Sec. 4 for the original
SABR-side derivation that we adapt here.)

Andersen, L. (2008). Simple and efficient simulation of the Heston
stochastic volatility model. Journal of Computational Finance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from .cev_sampling import sample_cev
from .utils import CEVHestonModelParameters, EuropeanOption, get_rng, standard_error


__all__ = [
    "HestonCEVSimulationResult",
    "HestonCEVPricingResult",
    "VarianceStepper",
    "simulate_heston_cev_terminal",
    "price_european_option_heston_cev",
]


DEFAULT_EPSILON_CLAMP = 1.0e-8

ConditionalAnchorScheme = Literal["frozen_left", "power_projected"]


class VarianceStepper(Protocol):
    """Minimal protocol that any CIR variance simulator must satisfy.

    A stepper is a stateful object whose ``step(dt, v_current)`` method
    advances the variance by one time-step of size ``dt`` and returns
    the pair ``(v_next, integrated_variance_over_step)``. The
    integrated variance must be the *dimensional* quantity
    ``int_t^{t+dt} v_s ds`` (units of variance times time), not the
    dimensionless paper-style ``I_t^h`` and not the average variance
    ``int v ds / dt``.

    The bundled ``AndersenQEVarianceSimulator`` in ``src.cir_simulation``
    satisfies this protocol; a thin adapter for PyFENG is in
    ``adapt_pyfeng_stepper`` below.
    """

    def step(
        self, dt: float, v_current: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(frozen=True)
class HestonCEVSimulationResult:
    """Container for one full simulation run.

    Attributes
    ----------
    terminal_prices:
        Un-discounted ``F_T`` for each path. Absorbed paths appear as
        exact zeros.
    n_paths_absorbed:
        Number of paths that hit the absorbing boundary at some point
        during simulation.
    clamp_trigger_rate:
        Average fraction of active paths (across all steps) for which
        the frozen-coefficient denominator was clamped. A healthy run
        reports a value well below 0.01.
    parameters:
        The model parameters used for this run, retained for
        provenance.
    """

    terminal_prices: np.ndarray
    n_paths_absorbed: int
    clamp_trigger_rate: float
    parameters: CEVHestonModelParameters
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class HestonCEVPricingResult:
    """Container for European-option pricing under CEV-Heston."""

    price: float
    standard_error: float
    discounted_payoffs: np.ndarray
    simulation: HestonCEVSimulationResult


def _build_default_variance_stepper(
    parameters: CEVHestonModelParameters,
    rng: np.random.Generator,
    n_path: int,
    dt: float,
    seed: int | None,
) -> tuple[VarianceStepper, str]:
    """Return the default variance stepper.

    Tries PyFENG first (the recommended PyFENG-backed path that uses the
    peer-reviewed Choi-Kwok 2024 Poisson-conditioning implementation),
    and falls back to the bundled Andersen QE stepper if PyFENG is
    not importable, or if the requested parameter regime cannot be
    handled by PyFENG's class.

    The ``xi = 0`` (deterministic-variance) limit is a legitimate
    degenerate case discussed in Choi et al. (2024) Section 2.3, but
    PyFENG's ``HestonMcChoiKwok2023PoisGe`` divides by ``xi**2`` in
    its constructor while computing the NCX2 degree of freedom and so
    cannot be instantiated at ``xi = 0``. We detect this upfront and
    use the bundled Andersen QE stepper, which collapses to the exact
    mean-reversion ODE when ``xi = 0``.
    """

    if parameters.xi == 0.0:
        from .cir_simulation import AndersenQEVarianceSimulator

        stepper = AndersenQEVarianceSimulator(parameters=parameters, rng=rng)
        return stepper, stepper.backend_name + "_xi_zero"

    try:
        from .pyfeng_adapter import PyFengVarianceStepper

        stepper = PyFengVarianceStepper(
            parameters=parameters,
            n_path=n_path,
            dt=dt,
            seed=seed,
        )
        return stepper, stepper.backend_name
    except ImportError:
        from .cir_simulation import AndersenQEVarianceSimulator

        stepper = AndersenQEVarianceSimulator(parameters=parameters, rng=rng)
        return stepper, stepper.backend_name + "_fallback"


def _conditional_mean_correlated(
    F_t: np.ndarray,
    v_t: np.ndarray,
    v_next: np.ndarray,
    integrated_variance: np.ndarray,
    parameters: CEVHestonModelParameters,
    dt: float,
    epsilon_clamp: float,
) -> tuple[np.ndarray, int]:
    r"""Compute the conditional mean ``F-bar_t^h`` for the correlated case.

    Implements the CIR-adapted version of Choi et al. (2024, Eq. 16b):

        F-bar_t^h = F_t * exp( rho * sqrt(v) integral / F_t^{1-beta}
                               - 0.5 * rho^2 * IV_t^h / F_t^{2(1-beta)} ),

    where ``int_t^{t+h} sqrt(v_s) dW^v_s = ( v_{t+h} - v_t - kappa
    theta h + kappa IV_t^h ) / xi`` by CIR's Ito formula.

    Returns the conditional mean and the count of paths whose
    ``F_t^{1-beta}`` denominator hit the clamp.
    """

    rho = parameters.correlation
    xi = parameters.xi
    beta_star = 1.0 - parameters.beta

    # Stochastic integral, closed-form via CIR Ito.
    # In the degenerate xi = 0 limit, the variance is deterministic and
    # int sqrt(v) dW^v = 0 by definition; we set it to zero explicitly
    # to avoid a 0 / 0 in the formula.
    if xi == 0.0:
        sqrt_v_integral = np.zeros_like(F_t)
    else:
        sqrt_v_integral = (
            v_next - v_t - parameters.kappa * parameters.theta * dt
            + parameters.kappa * integrated_variance
        ) / xi

    # Frozen-coefficient denominator with numerical safeguard.
    F_for_denom = np.maximum(F_t, epsilon_clamp)
    n_clamped = int(np.sum(F_t < epsilon_clamp))
    F_beta_star = F_for_denom ** beta_star

    drift = (rho / F_beta_star) * sqrt_v_integral
    convexity = -0.5 * (rho * rho * integrated_variance) / (F_beta_star * F_beta_star)

    return F_t * np.exp(drift + convexity), n_clamped


def _conditional_mean_power_projected(
    F_t: np.ndarray,
    v_t: np.ndarray,
    v_next: np.ndarray,
    integrated_variance: np.ndarray,
    parameters: CEVHestonModelParameters,
    dt: float,
    epsilon_clamp: float,
) -> tuple[np.ndarray, int]:
    r"""Compute a CEV-transform predictor with martingale projection.

    The paper's Eq. (13) freezes ``F^{1-beta}``` in the log equation
    for the correlated operator. For ``beta < 1`` a natural companion
    approximation is to work in the CEV power coordinate
    ``Y = F^{1-beta}``. Ito's formula gives, for the correlated
    operator alone,

        dY = beta_* rho sqrt(v) dW^v
             - 0.5 beta_* beta rho^2 v / Y dt,

    where ``beta_* = 1 - beta``. Freezing only the denominator in the
    Ito correction over one step gives the predictor

        Y_{t+h} ~= Y_t + beta_* rho A
                  - 0.5 beta_* beta rho^2 I / Y_t,

    with ``A = int sqrt(v)dW^v`` from the CIR identity and
    ``I = int v ds``. The predictor is then mapped back to the price
    coordinate and projected so its cross-sectional mean equals the
    incoming active-path mean. This keeps the step martingale-centered
    while using a beta-aware correlated shift instead of the purely
    log-frozen shift.

    The projection is deliberately local to one time step and only used
    by the opt-in ``power_projected`` scheme. The original Eq. (13)
    anchor remains available as ``frozen_left``.
    """

    rho = parameters.correlation
    xi = parameters.xi
    beta = parameters.beta
    beta_star = 1.0 - beta

    if beta_star <= 0.0:
        return _conditional_mean_correlated(
            F_t=F_t,
            v_t=v_t,
            v_next=v_next,
            integrated_variance=integrated_variance,
            parameters=parameters,
            dt=dt,
            epsilon_clamp=epsilon_clamp,
        )

    if xi == 0.0:
        sqrt_v_integral = np.zeros_like(F_t)
    else:
        sqrt_v_integral = (
            v_next - v_t - parameters.kappa * parameters.theta * dt
            + parameters.kappa * integrated_variance
        ) / xi

    F_for_denom = np.maximum(F_t, epsilon_clamp)
    n_clamped = int(np.sum(F_t < epsilon_clamp))
    y_t = F_for_denom ** beta_star
    y_next = (
        y_t
        + beta_star * rho * sqrt_v_integral
        - 0.5 * beta_star * beta * rho * rho * integrated_variance / np.maximum(y_t, 1e-300)
    )
    raw_anchor = np.maximum(y_next, 0.0) ** (1.0 / beta_star)

    raw_mean = float(np.mean(raw_anchor)) if raw_anchor.size else 0.0
    target_mean = float(np.mean(F_t)) if F_t.size else 0.0
    if raw_mean <= 0.0 or not np.isfinite(raw_mean):
        # Fall back to the paper anchor in the pathological case where
        # every transformed predictor has hit the absorbing boundary.
        return _conditional_mean_correlated(
            F_t=F_t,
            v_t=v_t,
            v_next=v_next,
            integrated_variance=integrated_variance,
            parameters=parameters,
            dt=dt,
            epsilon_clamp=epsilon_clamp,
        )

    anchor = raw_anchor * (target_mean / raw_mean)
    return anchor, n_clamped


def _conditional_mean_for_scheme(
    F_t: np.ndarray,
    v_t: np.ndarray,
    v_next: np.ndarray,
    integrated_variance: np.ndarray,
    parameters: CEVHestonModelParameters,
    dt: float,
    epsilon_clamp: float,
    conditional_scheme: ConditionalAnchorScheme,
) -> tuple[np.ndarray, int]:
    if conditional_scheme == "frozen_left":
        return _conditional_mean_correlated(
            F_t=F_t,
            v_t=v_t,
            v_next=v_next,
            integrated_variance=integrated_variance,
            parameters=parameters,
            dt=dt,
            epsilon_clamp=epsilon_clamp,
        )
    if conditional_scheme == "power_projected":
        return _conditional_mean_power_projected(
            F_t=F_t,
            v_t=v_t,
            v_next=v_next,
            integrated_variance=integrated_variance,
            parameters=parameters,
            dt=dt,
            epsilon_clamp=epsilon_clamp,
        )
    raise ValueError(
        "conditional_scheme must be one of 'frozen_left' or 'power_projected'"
    )


def _sample_next_price(
    F_bar: np.ndarray,
    integrated_variance: np.ndarray,
    parameters: CEVHestonModelParameters,
    rng: np.random.Generator,
    residual_variance_scale: float | None = None,
) -> np.ndarray:
    r"""Sample ``F_{t+h}`` given ``F-bar_t^h`` and ``IV_t^h``.

    The conditional law in the paper's framework is
    ``CEV_beta(F-bar, (1 - rho^2) * IV)``. For ``beta = 1`` the CEV
    distribution degenerates to lognormal and we use the closed-form
    update directly. The ``xi = 0`` and ``rho = pm 1`` degeneracies
    are handled implicitly by the standard CEV / lognormal samplers.
    """

    if residual_variance_scale is None:
        # If xi = 0, variance is deterministic and conditioning on the
        # variance path reveals no component of the asset Brownian
        # motion. Correlation is then distributionally irrelevant and
        # the full integrated variance must remain in the CEV step.
        residual_variance_scale = (
            1.0 if parameters.xi == 0.0 else 1.0 - parameters.correlation ** 2
        )
    rho_perp_squared = float(residual_variance_scale)
    effective_variance = rho_perp_squared * integrated_variance

    if parameters.beta == 1.0:
        # Lognormal closed form: F_{t+h} = F-bar * exp( sqrt(eff) Z - 0.5 eff ).
        # For paths with effective_variance == 0 (e.g. rho == +/- 1) this
        # reduces to F_{t+h} = F-bar exactly.
        z = rng.standard_normal(size=F_bar.shape)
        log_increment = np.sqrt(np.maximum(effective_variance, 0.0)) * z - 0.5 * effective_variance
        return F_bar * np.exp(log_increment)

    # General 0 < beta < 1 case: exact CEV sampler.
    return sample_cev(
        initial_value=F_bar,
        sigma_squared_time=effective_variance,
        beta=parameters.beta,
        rng=rng,
    )


def simulate_heston_cev_terminal(
    parameters: CEVHestonModelParameters,
    maturity: float,
    n_steps: int,
    n_paths: int,
    seed: int | np.random.Generator | None = None,
    variance_stepper: VarianceStepper | None = None,
    epsilon_clamp: float = DEFAULT_EPSILON_CLAMP,
    conditional_scheme: ConditionalAnchorScheme = "frozen_left",
) -> HestonCEVSimulationResult:
    """Simulate ``F_T`` under the CEV-Heston model.

    Parameters
    ----------
    parameters:
        Model parameters; see ``CEVHestonModelParameters``.
    maturity:
        Time to terminal in years.
    n_steps:
        Number of equal-size sub-steps used for the variance and the
        conditional asset sampling.
    n_paths:
        Number of Monte Carlo paths.
    seed:
        Either an integer seed, an existing ``numpy.random.Generator``,
        or ``None`` for a fresh generator. The same generator is used
        for both the price-side sampling here and the default CIR
        stepper (when no external stepper is supplied), so a fixed seed
        gives a fully reproducible simulation.
    variance_stepper:
        Optional CIR stepper. If ``None``, the bundled Andersen QE
        stepper is used. To plug in PyFENG (recommended for production)
        wrap its ``cond_states_step`` interface; an example adapter is
        provided in the package documentation.
    epsilon_clamp:
        Lower bound on ``F_t`` used inside the frozen-coefficient
        denominator. Default ``1e-8``. The simulator records the
        per-step clamp trigger rate as a diagnostic.
    conditional_scheme:
        Correlated-asset anchor used when ``rho != 0`` and ``xi > 0``.
        ``"frozen_left"`` is the direct Choi-Hu-Kwok Eq. (13)/(16b)
        adaptation. ``"power_projected"`` uses a CEV-power-coordinate
        predictor with a local martingale projection.

    Returns
    -------
    HestonCEVSimulationResult
        ``terminal_prices`` are *un-discounted* ``F_T``. Discounting
        is applied separately by the pricing helper so that martingale
        diagnostics can be computed cleanly on the raw output.
    """

    if maturity <= 0.0:
        raise ValueError("maturity must be positive")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    if epsilon_clamp <= 0.0:
        raise ValueError("epsilon_clamp must be positive")
    if conditional_scheme not in ("frozen_left", "power_projected"):
        raise ValueError(
            "conditional_scheme must be one of 'frozen_left' or 'power_projected'"
        )

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
        # Allow stepper instances to self-identify via a `backend_name`
        # attribute; otherwise fall back to the class name.
        backend_name = getattr(
            variance_stepper, "backend_name", type(variance_stepper).__name__
        )

    F = np.full(n_paths, parameters.spot, dtype=float)
    v = np.full(n_paths, parameters.initial_variance, dtype=float)

    n_clamp_total = 0
    n_active_total = 0
    active_count_per_step: list[int] = []

    for _ in range(n_steps):
        # Step 1: advance the variance and obtain integrated variance.
        v_next, integrated_variance = variance_stepper.step(dt, v)
        v_next = np.asarray(v_next, dtype=float)
        integrated_variance = np.asarray(integrated_variance, dtype=float)
        # Numerical safety: integrated variance must be non-negative.
        integrated_variance = np.maximum(integrated_variance, 0.0)

        # Identify the active subset of paths (not yet absorbed).
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

        # Step 2: compute the conditional mean.
        if parameters.correlation == 0.0 or parameters.xi == 0.0:
            # Uncorrelated, or deterministic variance: the conditioned
            # variance path reveals no correlated Brownian integral.
            F_bar_active = F_active
            n_clamped_step = 0
        else:
            F_bar_active, n_clamped_step = _conditional_mean_for_scheme(
                F_t=F_active,
                v_t=v_active,
                v_next=v_next_active,
                integrated_variance=iv_active,
                parameters=parameters,
                dt=dt,
                epsilon_clamp=epsilon_clamp,
                conditional_scheme=conditional_scheme,
            )
        n_clamp_total += n_clamped_step

        # Step 3: sample F_{t+h} from the (effective) CEV distribution.
        residual_variance_scale = (
            1.0 if parameters.xi == 0.0 else 1.0 - parameters.correlation ** 2
        )
        F_next_active = _sample_next_price(
            F_bar=F_bar_active,
            integrated_variance=iv_active,
            parameters=parameters,
            rng=rng,
            residual_variance_scale=residual_variance_scale,
        )

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
        "conditional_scheme": conditional_scheme,
    }

    return HestonCEVSimulationResult(
        terminal_prices=F,
        n_paths_absorbed=n_paths_absorbed,
        clamp_trigger_rate=clamp_trigger_rate,
        parameters=parameters,
        diagnostics=diagnostics,
    )


def price_european_option_heston_cev(
    parameters: CEVHestonModelParameters,
    option: EuropeanOption,
    n_steps: int,
    n_paths: int,
    seed: int | np.random.Generator | None = None,
    variance_stepper: VarianceStepper | None = None,
    epsilon_clamp: float = DEFAULT_EPSILON_CLAMP,
    conditional_scheme: ConditionalAnchorScheme = "frozen_left",
) -> HestonCEVPricingResult:
    """Price a European option under CEV-Heston by Monte Carlo.

    Wraps ``simulate_heston_cev_terminal`` and applies the deterministic
    discount factor ``exp(-r T)`` to the option payoff. Both calls and
    puts are supported; for puts we use the absorbing boundary directly
    (``max(K - F_T, 0)`` includes the absorbed paths with ``F_T = 0``).
    """

    if option.option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    simulation = simulate_heston_cev_terminal(
        parameters=parameters,
        maturity=option.maturity,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
        variance_stepper=variance_stepper,
        epsilon_clamp=epsilon_clamp,
        conditional_scheme=conditional_scheme,
    )

    terminal = simulation.terminal_prices
    if option.option_type == "call":
        payoffs = np.maximum(terminal - option.strike, 0.0)
    else:
        payoffs = np.maximum(option.strike - terminal, 0.0)

    discount_factor = float(np.exp(-parameters.risk_free_rate * option.maturity))
    discounted = discount_factor * payoffs

    return HestonCEVPricingResult(
        price=float(np.mean(discounted)),
        standard_error=float(standard_error(discounted)),
        discounted_payoffs=discounted,
        simulation=simulation,
    )
