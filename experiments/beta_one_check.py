"""Long-T regression: does the step-by-step ours bias relative to a
one-shot conditional MC and PyFENG FFT?

option_price experiment shows ours systematically below benchmark by
-0.011 at T=10 (50 SEMs significant), which is incompatible with the
martingale experiment showing E[F_T] = F_0. The reconciliation is that
ours has correct mean but wrong variance/distribution.

This script tests whether the step-by-step architecture is responsible.
We compare three at IDENTICAL parameters and SAME PyFENG variance backend:

  (A) PyFENG HestonFft -- closed-form benchmark.
  (B) One-shot conditional MC (course-slide M8 page 10 formula): use
      stepper to accumulate (v_T, V_T) over [0, T], then plug into
      closed-form lognormal under conditional measure.
  (C) The current src step-by-step simulator.

If (B) tracks (A) within MC noise but (C) is offset by ~-0.011 at T=10,
the step-by-step accumulation has a real bug; if both (B) and (C) are
offset, the variance backend's V_T accumulation is biased.
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


# Same as option_price experiment, but with beta=1 so we have a closed-form benchmark.
PARAMS = CEVHestonModelParameters(
    spot=1.0,
    initial_variance=0.09,
    kappa=1.5,
    theta=0.09,
    xi=0.6,
    correlation=-0.8,
    beta=1.0,            # KEY: beta=1 so HestonFft is exact
    risk_free_rate=0.0,
)
N_PATHS = 200_000
N_REPLICATIONS = 10
SEED_BASE = 12345
MATURITIES = [1.0, 4.0, 10.0]


def bs_call_vec(S, K, sigma, T, r=0.0):
    S = np.asarray(S, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sqrtT = np.sqrt(T)
    safe = sigma > 1e-12
    out = np.maximum(S - K * np.exp(-r * T), 0.0)
    if np.any(safe):
        s = sigma[safe]
        sp = S[safe]
        d1 = (np.log(sp / K) + (r + 0.5 * s ** 2) * T) / (s * sqrtT)
        d2 = d1 - s * sqrtT
        out[safe] = sp * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return out


def conditional_mc_one_shot(parameters, option, n_paths, n_steps, seed):
    """One-shot conditional MC per course slide M8 page 10.

    Drives the SAME PyFENG variance backend as the step-by-step ours
    so that any difference vs (C) is attributable to asset-side
    architecture, not the variance simulation.
    """
    dt = option.maturity / n_steps
    stepper = PyFengVarianceStepper(
        parameters=parameters, n_path=n_paths, dt=dt, seed=seed,
    )
    v = np.full(n_paths, parameters.initial_variance, dtype=float)
    V_T = np.zeros(n_paths, dtype=float)
    for _ in range(n_steps):
        v_next, iv_step = stepper.step(dt, v)
        v_next = np.asarray(v_next, dtype=float)
        iv_step = np.maximum(np.asarray(iv_step, dtype=float), 0.0)
        V_T = V_T + iv_step
        v = v_next
    v_T = v

    rho = parameters.correlation
    xi = parameters.xi
    kappa = parameters.kappa
    theta = parameters.theta
    v_0 = parameters.initial_variance
    T = option.maturity

    # Course-slide M8 page 10:
    # E[S_T | v_T, V_T] = S_0 * exp(rho/xi (v_T - v_0 + kappa(V_T - theta T))
    #                               - 0.5 rho^2 V_T)
    # sigma_BS = rho_* sqrt(V_T / T)
    sqrt_v_int = (v_T - v_0 + kappa * (V_T - theta * T)) / xi
    log_drift = rho * sqrt_v_int - 0.5 * rho ** 2 * V_T
    S_eff = parameters.spot * np.exp(log_drift)
    sigma_eff = np.sqrt((1.0 - rho ** 2) * V_T / T)

    call_prices = bs_call_vec(
        S=S_eff, K=option.strike, sigma=sigma_eff,
        T=T, r=parameters.risk_free_rate,
    )
    df = float(np.exp(-parameters.risk_free_rate * T))
    discounted = df * call_prices
    return float(np.mean(discounted)), float(np.std(discounted, ddof=1) / np.sqrt(n_paths)), float(np.mean(V_T))


def step_by_step(parameters, option, n_paths, n_steps, seed):
    res = price_european_option_heston_cev(
        parameters=parameters, option=option,
        n_steps=n_steps, n_paths=n_paths, seed=seed,
    )
    return res.price, res.standard_error


def theoretical_E_VT(parameters, T):
    """Closed-form E[V_T] = theta T + (v_0 - theta)(1 - exp(-kappa T))/kappa."""
    return (
        parameters.theta * T
        + (parameters.initial_variance - parameters.theta)
        * (1.0 - np.exp(-parameters.kappa * T)) / parameters.kappa
    )


def run_at_T(T):
    print(f"\n{'='*72}")
    print(f"T = {T}")
    print(f"{'='*72}")

    option = EuropeanOption(strike=1.0, maturity=T, option_type="call")
    n_steps = max(int(round(T)), 1)  # match option_price experiment: h=1

    # (A) FFT
    fft_price = float(heston_fourier_benchmark_beta_one(PARAMS, option).price)
    print(f"(A) PyFENG FFT          = {fft_price:.6f}")

    # (B) One-shot conditional MC: m=10 replications
    cmc_prices = np.empty(N_REPLICATIONS)
    V_T_means = np.empty(N_REPLICATIONS)
    for r in range(N_REPLICATIONS):
        p, _, vt = conditional_mc_one_shot(
            PARAMS, option, N_PATHS, n_steps, seed=SEED_BASE + r,
        )
        cmc_prices[r] = p
        V_T_means[r] = vt
    cmc_mean = float(np.mean(cmc_prices))
    cmc_sem = float(np.std(cmc_prices, ddof=1) / np.sqrt(N_REPLICATIONS))
    cmc_diff = cmc_mean - fft_price
    print(f"(B) One-shot CMC        = {cmc_mean:.6f} +/- {cmc_sem:.6f}  "
          f"(diff vs FFT = {cmc_diff:+.6f}, {cmc_diff/cmc_sem:+.2f} SEMs)")

    # IV diagnostic: theoretical E[V_T] vs PyFENG accumulated mean.
    theo_VT = theoretical_E_VT(PARAMS, T)
    obs_VT = float(np.mean(V_T_means))
    iv_rel_diff = (obs_VT - theo_VT) / theo_VT
    print(f"    PyFENG accumulated E[V_T] = {obs_VT:.6f}")
    print(f"    Theoretical    E[V_T] = {theo_VT:.6f}")
    print(f"    Relative difference   = {iv_rel_diff*100:+.3f}%")

    # (C) src step-by-step: m=10 replications
    sbs_prices = np.empty(N_REPLICATIONS)
    for r in range(N_REPLICATIONS):
        p, _ = step_by_step(PARAMS, option, N_PATHS, n_steps, seed=SEED_BASE + r)
        sbs_prices[r] = p
    sbs_mean = float(np.mean(sbs_prices))
    sbs_sem = float(np.std(sbs_prices, ddof=1) / np.sqrt(N_REPLICATIONS))
    sbs_diff = sbs_mean - fft_price
    print(f"(C) src step-by-step    = {sbs_mean:.6f} +/- {sbs_sem:.6f}  "
          f"(diff vs FFT = {sbs_diff:+.6f}, {sbs_diff/sbs_sem:+.2f} SEMs)")


def main():
    print("Long-T regression: ours architecture vs one-shot CMC vs FFT")
    print(f"Parameters: F_0={PARAMS.spot}, v_0=theta={PARAMS.initial_variance}, "
          f"kappa={PARAMS.kappa}, xi={PARAMS.xi}, rho={PARAMS.correlation}, "
          f"beta={PARAMS.beta}")
    print(f"N_PATHS={N_PATHS:,}, m={N_REPLICATIONS}, h=1 (n_steps = T)")
    for T in MATURITIES:
        run_at_T(T)


if __name__ == "__main__":
    main()
