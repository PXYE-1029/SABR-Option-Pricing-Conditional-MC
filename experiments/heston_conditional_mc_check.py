"""Minimal sanity check: does a one-shot Heston conditional MC
(per the course-slide M8 page 10) match PyFENG FFT?

This script implements the conditional-MC formula from the course slides:
    log(S_T/S_0) | (v_T, V_T) ~ N(mu, rho_*^2 V_T)
where
    mu = (rho/xi) * (v_T - v_0 + kappa*(V_T - theta*T)) - 0.5 V_T

Conditional on (v_T, V_T), S_T is lognormal, and the call price has a
Black-Scholes closed form (no further asset-side simulation). We then
average across paths over (v_T, V_T) drawn from the variance process.

We compare three things at the SAME parameters:
  (1) the one-shot conditional-MC price (this script),
  (2) PyFENG HestonFft -- the independent FFT benchmark,
  (3) the current src step-by-step simulator.

If (1) ~ (2) within a couple of SEs, that confirms the closed-form
formula is correct AND that the variance backend (PyFENG/QE) is
producing reasonable (v_T, V_T) marginals. Any large gap between (3)
and (1)-(2) is then attributable to the step-by-step architecture's
accumulated error, not to anything wrong with the conditional formula.

Parameters chosen are conservative ('textbook Heston'), avoiding the
xi -> 0 limit where PyFENG FFT becomes unstable.
"""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.heston_cev_benchmark import heston_fourier_benchmark_beta_one
from src.heston_cev_simulation import price_european_option_heston_cev
from src.pyfeng_adapter import PyFengVarianceStepper
from src.utils import CEVHestonModelParameters, EuropeanOption


# ============================================================================
# Parameters: standard textbook Heston, far from any pathological limit.
# ============================================================================
PARAMS = CEVHestonModelParameters(
    spot=100.0,
    initial_variance=0.04,    # sigma_0 = 0.2
    kappa=1.0,
    theta=0.04,
    xi=0.3,                   # moderate vol-of-vol，FFT stable
    correlation=-0.7,
    beta=1.0,
    risk_free_rate=0.0,
)
OPTION = EuropeanOption(strike=100.0, maturity=1.0, option_type="call")
N_PATHS = 200_000
N_STEPS = 50         # for the variance path: enough for V_T accuracy
SEED = 12345


# ============================================================================
# (1) One-shot conditional MC per the course slide formula.
# ============================================================================
def bs_call_vectorized(spot, strike, sigma, T, r=0.0):
    """Black-Scholes call, fully vectorized over the spot/sigma arrays.

    Handles sigma -> 0 cleanly by returning the intrinsic value; this
    avoids 0/0 in d1/d2 when occasional paths have very small V_T.
    """
    spot = np.asarray(spot, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sqrtT = np.sqrt(T)
    # Safe sigma: anything below this falls back to intrinsic.
    safe = sigma > 1e-12
    out = np.maximum(spot - strike * np.exp(-r * T), 0.0)
    if np.any(safe):
        s_safe = sigma[safe]
        sp_safe = spot[safe]
        d1 = (np.log(sp_safe / strike) + (r + 0.5 * s_safe ** 2) * T) / (s_safe * sqrtT)
        d2 = d1 - s_safe * sqrtT
        call_safe = sp_safe * norm.cdf(d1) - strike * np.exp(-r * T) * norm.cdf(d2)
        out[safe] = call_safe
    return out


def conditional_mc_price(parameters, option, n_paths, n_steps, seed):
    """One-shot Heston conditional MC."""
    rng = np.random.default_rng(seed)
    dt = option.maturity / n_steps

    # Use the SAME PyFENG variance backend the rest of the codebase uses,
    # so that any difference vs ours-step-by-step isolates the asset-side
    # architecture, not the variance backend.
    stepper = PyFengVarianceStepper(
        parameters=parameters,
        n_path=n_paths,
        dt=dt,
        seed=seed,
    )

    v = np.full(n_paths, parameters.initial_variance, dtype=float)
    V_T = np.zeros(n_paths, dtype=float)  # accumulated integrated variance

    for _ in range(n_steps):
        v_next, iv_step = stepper.step(dt, v)
        v_next = np.asarray(v_next, dtype=float)
        iv_step = np.maximum(np.asarray(iv_step, dtype=float), 0.0)
        V_T = V_T + iv_step
        v = v_next

    v_T_paths = v  # terminal variance per path

    # Course-slide formula (M8 page 10):
    # E[S_T | v_T, V_T] = S_0 * exp(rho/xi * (v_T - v_0 + kappa*(V_T - theta*T))
    #                               - 0.5 rho^2 V_T)
    rho = parameters.correlation
    xi = parameters.xi
    kappa = parameters.kappa
    theta = parameters.theta
    v_0 = parameters.initial_variance
    T = option.maturity

    sqrt_v_integral_total = (
        v_T_paths - v_0 + kappa * (V_T - theta * T)
    ) / xi
    log_S_eff_drift = rho * sqrt_v_integral_total - 0.5 * (rho ** 2) * V_T
    S_eff = parameters.spot * np.exp(log_S_eff_drift)

    # sigma_BS = rho_* * sqrt(V_T / T)
    sigma_eff = np.sqrt((1.0 - rho ** 2) * V_T / T)

    # Black-Scholes call per path, then average.
    call_prices = bs_call_vectorized(
        spot=S_eff, strike=option.strike, sigma=sigma_eff,
        T=T, r=parameters.risk_free_rate,
    )
    discount_factor = float(np.exp(-parameters.risk_free_rate * T))
    discounted = discount_factor * call_prices
    return float(np.mean(discounted)), float(np.std(discounted, ddof=1) / np.sqrt(n_paths))


# ============================================================================
# (2) PyFENG FFT independent benchmark (already wrapped in src).
# ============================================================================
def fft_benchmark(parameters, option):
    bp = heston_fourier_benchmark_beta_one(parameters, option)
    return bp.price


# ============================================================================
# (3) Current src step-by-step simulator (for comparison).
# ============================================================================
def step_by_step_price(parameters, option, n_paths, n_steps, seed):
    res = price_european_option_heston_cev(
        parameters=parameters, option=option,
        n_steps=n_steps, n_paths=n_paths, seed=seed,
    )
    return res.price, res.standard_error


# ============================================================================
# Main: run all three and report.
# ============================================================================
def main():
    print("=" * 70)
    print("Minimal sanity check: one-shot conditional MC vs PyFENG FFT vs ours")
    print("=" * 70)
    print(
        f"Parameters: F_0={PARAMS.spot}, v_0=theta={PARAMS.initial_variance}, "
        f"kappa={PARAMS.kappa}, xi={PARAMS.xi}, rho={PARAMS.correlation}, "
        f"beta={PARAMS.beta}, T={OPTION.maturity}, K={OPTION.strike}"
    )
    print(f"N_PATHS={N_PATHS:,}, N_STEPS={N_STEPS}, SEED={SEED}")
    print()

    # (2) FFT benchmark first --- it's nearly instant.
    t0 = perf_counter()
    fft_price = fft_benchmark(PARAMS, OPTION)
    t_fft = perf_counter() - t0
    print(f"(2) PyFENG FFT (independent benchmark)")
    print(f"      price = {fft_price:.6f}  ({t_fft*1000:.1f} ms)")
    print()

    # (1) Conditional MC.
    t0 = perf_counter()
    cmc_price, cmc_se = conditional_mc_price(PARAMS, OPTION, N_PATHS, N_STEPS, SEED)
    t_cmc = perf_counter() - t0
    cmc_diff = cmc_price - fft_price
    print(f"(1) One-shot conditional MC (course-slide formula)")
    print(f"      price = {cmc_price:.6f} +/- {cmc_se:.6f}  ({t_cmc:.1f} s)")
    print(f"      diff vs FFT = {cmc_diff:+.6f} ({cmc_diff/cmc_se:+.2f} SEs)")
    print()

    # (3) Current src step-by-step.
    t0 = perf_counter()
    sbs_price, sbs_se = step_by_step_price(PARAMS, OPTION, N_PATHS, N_STEPS, SEED)
    t_sbs = perf_counter() - t0
    sbs_diff = sbs_price - fft_price
    print(f"(3) Current src step-by-step simulator")
    print(f"      price = {sbs_price:.6f} +/- {sbs_se:.6f}  ({t_sbs:.1f} s)")
    print(f"      diff vs FFT = {sbs_diff:+.6f} ({sbs_diff/sbs_se:+.2f} SEs)")
    print()

    # Verdict.
    print("=" * 70)
    print("Verdict:")
    if abs(cmc_diff) < 3.0 * cmc_se:
        print("  (1) tracks (2) within MC noise -> conditional MC formula is correct.")
        if abs(sbs_diff) > 5.0 * sbs_se:
            print(
                "  (3) deviates from (2) by many SEs -> step-by-step is the source"
                " of the bias seen in earlier experiments."
            )
        else:
            print("  (3) also tracks (2). Then earlier bias was parameter-regime specific.")
    else:
        print(
            "  (1) deviates from (2) by many SEs. Either the variance backend"
            " has a sampling bias, or the FFT benchmark is unreliable here."
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
