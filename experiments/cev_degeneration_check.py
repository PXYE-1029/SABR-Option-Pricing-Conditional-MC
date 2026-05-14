"""CEV-degeneration test for ours simulator under Heston-CEV.

Setup proposed by the professor: pick parameters that pin v_t at
near-constant v_0, so that the Heston-CEV model degenerates into the
pure CEV model. The CEV model has a closed-form option price
(noncentral chi-square; PyFENG provides it), which gives an
INDEPENDENT ground truth -- no self-reference, no Fourier benchmark,
no other Monte Carlo.

Degeneration parameters:
  - kappa large (e.g. 50)  -> very fast mean reversion
  - theta = v_0            -> mean-reversion target is the initial value
  - xi very small (0.01)   -> almost no random shock
  Net effect: v_t ~ v_0 + tiny noise, essentially constant.

Under v_t = constant = v_0, the asset SDE
    dF / F^beta = sqrt(v_0) dW^F
is exactly the CEV model with sigma = sqrt(v_0). The European call
under this model has a known noncentral chi-square closed form,
which PyFENG exposes via the Cev (or CevAnalytic / Cev) class.

This test isolates the asset-side machinery of ours -- the CEV
sampler and the frozen-coefficient handling -- without touching
the variance backend's sophistication. Combined with the earlier
xi -> 0 + beta = 1 BS test (which exercised the asset-side
lognormal branch instead), this completes the asset-side validation.

If ours matches the CEV closed form across rho values:
  -> the asset-side implementation is correct, and the rho^2-scaled
     option-price bias observed in the full Heston-CEV setting is
     genuinely an interaction between the frozen coefficient and the
     stochastic variance backend (rather than a CEV-sampler bug).

If ours deviates from the CEV closed form:
  -> there is an asset-side implementation issue that must be fixed
     before any other claim about the simulator can be trusted.
"""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pyfeng as pf

from src.heston_cev_simulation import price_european_option_heston_cev
from src.utils import CEVHestonModelParameters, EuropeanOption


# ===========================================================================
# Step 1: Verify the PyFENG CEV class parameter convention.
# ===========================================================================
def cev_closed_form_price(spot, strike, sigma, T, beta):
    """Closed-form CEV European call price via PyFENG.

    PyFENG exposes the CEV model under pf.Cev. The constructor takes
    (sigma, beta) -- sigma is the LEVEL volatility, not variance. We
    verify this convention against the BS limit before trusting it.
    """
    model = pf.Cev(sigma=sigma, beta=beta, intr=0.0, divr=0.0)
    return float(model.price(strike=strike, spot=spot, texp=T, cp=1))


def verify_pyfeng_cev_convention():
    """Sanity-check pf.Cev against BS by using beta near 1.

    Note: pf.Cev internally uses 1/(1-beta), so beta = 1 raises
    ZeroDivisionError. We instead use beta = 0.99 -- close enough to
    BS that the CEV price matches BS to ~1e-3 relative error, which
    is sufficient to confirm sigma is volatility (NOT variance). If
    sigma were misinterpreted as variance, the discrepancy would be
    on the order of tens of percent.
    """
    print("=" * 70)
    print("Step 0: verify PyFENG Cev parameter convention (near-BS sanity)")
    print("=" * 70)

    sigma = 0.3
    T = 4.0
    S = K = 1.0
    beta_near_one = 0.99   # cannot use exactly 1 due to PyFENG's 1/(1-beta)

    cev_near_bs = cev_closed_form_price(
        spot=S, strike=K, sigma=sigma, T=T, beta=beta_near_one,
    )

    # Direct BS for ATM: S * [N(d1) - N(d2)] with d1 = -d2 = sigma*sqrt(T)/2.
    from scipy.stats import norm
    d_half = 0.5 * sigma * np.sqrt(T)
    bs_direct = S * (norm.cdf(d_half) - norm.cdf(-d_half))

    print(f"  pf.Cev(sigma=0.3, beta=0.99).price = {cev_near_bs:.6f}")
    print(f"  Direct BS closed form               = {bs_direct:.6f}")
    print(f"  Relative difference                  = "
          f"{abs(cev_near_bs - bs_direct)/bs_direct*100:.2f}%")
    # If sigma were treated as variance instead of volatility, the
    # mismatch would be ~50% or more (recall the HestonFft bug). A
    # 1-2% gap here would be consistent with beta=0.99 vs beta=1 only.
    if abs(cev_near_bs - bs_direct) / bs_direct < 0.05:
        print("  --> PyFENG Cev sigma convention = volatility level. OK.")
        return True
    else:
        print("  --> Mismatch! Need to inspect PyFENG Cev convention.")
        return False


# ===========================================================================
# Step 2: Run ours under CEV-degeneration parameters and compare.
# ===========================================================================
def run_ours_in_cev_limit(
    rho: float,
    T: float,
    n_paths: int = 100_000,
    n_steps: int = 32,
    seed: int = 12345,
    kappa: float = 50.0,
    xi: float = 0.01,
) -> tuple[float, float]:
    """Run ours simulator in the CEV-degeneration limit.

    Parameters tuned so v_t stays effectively at v_0 = theta:
      kappa = 50, xi = 0.01, theta = v_0.

    Returns (price, standard_error).
    """
    parameters = CEVHestonModelParameters(
        spot=1.0,
        initial_variance=0.09,
        kappa=kappa,
        theta=0.09,
        xi=xi,
        correlation=rho,
        beta=0.4,
        risk_free_rate=0.0,
    )
    option = EuropeanOption(strike=1.0, maturity=T, option_type="call")
    result = price_european_option_heston_cev(
        parameters=parameters, option=option,
        n_steps=n_steps, n_paths=n_paths, seed=seed,
    )
    return float(result.price), float(result.standard_error)


def main():
    if not verify_pyfeng_cev_convention():
        print("Aborting -- PyFENG Cev convention check failed.")
        return

    print()
    print("=" * 70)
    print("Step 1: ours in CEV-degeneration limit vs PyFENG Cev closed form")
    print("=" * 70)
    print("Degeneration parameters: kappa=50, xi=0.01, theta=v_0=0.09")
    print("(makes v_t effectively constant at v_0)")
    print()
    print("Asset parameters: F_0 = K = 1, beta = 0.4, r = q = 0")
    print("CEV reference: sigma = sqrt(v_0) = 0.3")
    print()

    sigma_cev = float(np.sqrt(0.09))  # 0.3
    BETA = 0.4

    # Sweep T and rho. The CEV closed form doesn't depend on rho
    # (since v is constant, there is no correlation to express).
    # ours simulator still takes rho as an input; if the implementation
    # is correct, the result should be rho-independent in this limit.
    MATURITIES = [1.0, 4.0, 10.0]
    RHOS = [0.0, -0.4, -0.8]

    print(f"{'T':>5s} {'rho':>6s}  {'CEV closed':>12s}  {'ours':>14s}  "
          f"{'diff':>10s}  {'SE units':>10s}")
    print("-" * 70)

    rows = []
    for T in MATURITIES:
        cev_price = cev_closed_form_price(
            spot=1.0, strike=1.0, sigma=sigma_cev, T=T, beta=BETA,
        )
        for rho in RHOS:
            t0 = perf_counter()
            ours_price, ours_se = run_ours_in_cev_limit(
                rho=rho, T=T, n_paths=200_000, n_steps=int(8 * T),
                seed=int(12345 + 1000 * T + 100 * abs(rho)),
            )
            elapsed = perf_counter() - t0
            diff = ours_price - cev_price
            se_units = diff / max(ours_se, 1e-300)
            print(f"{T:>5.1f} {rho:>+6.2f}  {cev_price:>12.6f}  "
                  f"{ours_price:>8.6f} +/- {ours_se:.4f}  "
                  f"{diff:>+10.6f}  {se_units:>+10.2f} "
                  f"({elapsed:.1f}s)")
            rows.append({
                "T": T, "rho": rho,
                "cev_closed": cev_price,
                "ours": ours_price, "ours_se": ours_se,
                "diff": diff, "se_units": se_units,
            })

    print()
    print("=" * 70)
    print("Interpretation:")
    print("  If all |diff / SE| < 3:  ours's asset-side implementation is OK.")
    print("  If diff is uniformly negative and scales with |rho|:")
    print("    -> there is a real bug in frozen-coefficient handling that is")
    print("       independent of variance dynamics.")
    print("  If diff is zero at rho=0 but grows with |rho| in this near-CEV")
    print("    limit: that would be unexpected -- frozen-coefficient bias")
    print("    should depend on the dynamic interplay between F and v, which")
    print("    is suppressed when v is constant.")
    print("=" * 70)


if __name__ == "__main__":
    main()
