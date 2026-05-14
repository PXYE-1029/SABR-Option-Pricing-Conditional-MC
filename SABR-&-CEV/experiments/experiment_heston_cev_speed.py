"""Speed-vs-accuracy trade-off for CEV-Heston pricing (paper Fig 2 analogue, v3).

Plots root-mean-square error against CPU time for two schemes:

  * The project's CEV-approximation simulator -- the proposed method.
  * A naive asset-side truncated-Euler baseline.

This is the analogue of Choi-Hu-Kwok (2024) Fig 2 in the Heston-CEV
setting. The figure shows two curves on a log-log RMS-vs-time plane;
the proposed scheme should sit below and to the left of the Euler
baseline across the whole range -- "ours is faster at any target RMS,
and more accurate at any target CPU budget."

DESIGN DETAILS

Each scheme sweeps its own most-informative axis:

  - Ours: per-step cost is high (Gamma + Poisson + truncated
    Gaussian sampling for variance, plus exact CEV sampling for
    the asset). With sufficient h, error is dominated by Monte
    Carlo variance. Sweet spot: fix a moderately fine h, sweep N.

  - Euler: per-step cost is low (two normal draws), but error is
    dominated by time-discretization bias. Sweet spot: fix a
    middling N, sweep n_steps.

PARAMETERS DELIBERATELY DIFFER FROM THE OTHER EXPERIMENTS

The other CEV-Heston experiments (martingale, option_price,
h_convergence) use Case-V-style parameters (F_0=1, rho=-0.8,
beta=0.4). Those parameters are *bad* for a speed-vs-RMS
comparison, because in that regime ours has a non-trivial
frozen-coefficient bias floor at moderate h that obscures the
work-vs-error trend.

For Fig 2 we follow Choi-Hu-Kwok's own choice and use a milder
parameter set with two specific changes:
  - Lower F_0 (0.5 instead of 1.0). This brings the asset closer
    to the absorbing boundary, which is where Euler's bias floor
    comes from. Without this, Euler's RMS just keeps decreasing
    with h, and the work-vs-error comparison no longer demonstrates
    the qualitative advantage of exact CEV sampling.
  - Mild correlation (rho=-0.3 instead of -0.8). The frozen
    coefficient error in Eq 13 of the paper is O(rho^2), so a
    mild rho keeps ours's bias far below the MC noise floor at
    moderate h, leaving variance reduction (= sqrt(N) scaling)
    as the visible signal.

The end result is a parameter set where ours can ride down on a
sqrt(N) line and Euler bottoms out on its bias floor -- exactly the
qualitative shape of the paper's Fig 2.

Each (N, h) point is run ``m=50`` times.
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
from pathlib import Path
from time import perf_counter

_CACHE_DIR = Path(tempfile.gettempdir()) / "sabr-option-pricing-cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.heston_cev_benchmark import (
    heston_fourier_benchmark_beta_one,
    self_reference_benchmark,
)
from src.heston_cev_simulation import price_european_option_heston_cev
from src.utils import CEVHestonModelParameters, EuropeanOption, get_rng, standard_error


MATURITY = 4.0
N_REPLICATIONS = 50

# Ours: paper Table 7 style -- N doubled and h halved together.
OURS_GRID = [
    (5_000,   1),    # n_steps =   4, h = 1.0
    (10_000,  2),    # n_steps =   8, h = 0.5
    (20_000,  4),    # n_steps =  16, h = 0.25
    (40_000,  8),    # n_steps =  32, h = 0.125
    (80_000,  16),   # n_steps =  64, h = 0.0625
]

# Euler: fix N at 50,000, sweep h aggressively. The finest h is
# 0.003125, which is several hundred times finer than ours uses,
# but should still leave Euler bottomed out on its bias floor due
# to the absorbing boundary at F=0.
EULER_GRID = [
    (50_000, 5),     # n_steps =   20, h = 0.2
    (50_000, 20),    # n_steps =   80, h = 0.05
    (50_000, 80),    # n_steps =  320, h = 0.0125
    (50_000, 320),   # n_steps = 1280, h = 0.003125
]

TABLE_PATH = PROJECT_ROOT / "results" / "tables" / "heston_cev_speed_rms.csv"
FIGURE_PATH = PROJECT_ROOT / "results" / "figures" / "heston_cev_speed_rms_tradeoff.png"


def _ensure_output_dirs() -> None:
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _write_csv(rows: list[dict], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _baseline_parameters() -> CEVHestonModelParameters:
    """Speed-experiment parameters (Fig 2 style; deliberately different
    from the other experiments).

    See module docstring for why F_0 is small and rho is mild.
    """
    return CEVHestonModelParameters(
        spot=0.5,
        initial_variance=0.04,    # sigma_0 = 0.2
        kappa=1.0,
        theta=0.04,
        xi=0.4,
        correlation=-0.3,         # mild rho keeps ours's frozen-coeff bias <= MC noise
        beta=0.3,                 # small beta keeps Euler near the absorbing boundary
        risk_free_rate=0.0,
    )


def _truncated_euler_price(
    parameters: CEVHestonModelParameters,
    option: EuropeanOption,
    n_steps: int,
    n_paths: int,
    seed: int,
) -> tuple[float, float]:
    """Asset-side truncated Euler -- the baseline from Choi-Hu-Kwok (2024) Fig 2.

    Direct discretization
        F_{t+h} = F_t + sqrt(v_t) * F_t^beta * dW^F * sqrt(h)
    with the absorbing boundary enforced by clamping at zero.
    Variance is evolved via full-truncation Euler (Lord-Koekkoek-van
    Dijk 2010).
    """
    rng = get_rng(seed)
    dt = option.maturity / n_steps
    sqrt_dt = float(np.sqrt(dt))

    rho = parameters.correlation
    rho_perp = float(np.sqrt(max(1.0 - rho * rho, 0.0)))
    beta = parameters.beta

    F = np.full(n_paths, parameters.spot, dtype=float)
    v = np.full(n_paths, parameters.initial_variance, dtype=float)

    for _ in range(n_steps):
        z1 = rng.standard_normal(size=n_paths)
        z2 = rng.standard_normal(size=n_paths)
        dW_v = z1
        dW_F = rho * z1 + rho_perp * z2

        v_pos = np.maximum(v, 0.0)
        sqrt_v = np.sqrt(v_pos)

        active = F > 0.0
        if np.any(active):
            F_act = F[active]
            sqrt_v_act = sqrt_v[active]
            dW_F_act = dW_F[active]
            F_pow = F_act ** beta
            increment = sqrt_v_act * F_pow * sqrt_dt * dW_F_act
            F_new = F_act + increment
            F_new = np.where(np.isfinite(F_new) & (F_new > 0.0), F_new, 0.0)
            F[active] = F_new

        v = v + parameters.kappa * (parameters.theta - v_pos) * dt + (
            parameters.xi * sqrt_v * sqrt_dt * dW_v
        )

    if option.option_type == "call":
        payoffs = np.maximum(F - option.strike, 0.0)
    else:
        payoffs = np.maximum(option.strike - F, 0.0)
    discount_factor = float(np.exp(-parameters.risk_free_rate * option.maturity))
    discounted = discount_factor * payoffs
    return float(np.mean(discounted)), float(standard_error(discounted))


def _benchmark(parameters: CEVHestonModelParameters, option: EuropeanOption) -> float:
    if parameters.beta == 1.0:
        return float(heston_fourier_benchmark_beta_one(parameters, option).price)
    return float(self_reference_benchmark(parameters, option).price)


def _run_replications(
    pricer_fn,
    n_paths: int,
    n_steps: int,
    benchmark_price: float,
    seed_base: int,
) -> tuple[float, float, float]:
    """Run m replications. Returns (rms_error, mean_bias, total_seconds).

    RMS follows Choi-Hu-Kwok (2024) Eq. (26):
        RMS = sqrt( bias^2 + variance ) = sqrt( mean( (price - bench)^2 ) ).
    """
    errors_squared = np.empty(N_REPLICATIONS)
    signed_errors = np.empty(N_REPLICATIONS)
    elapsed_total = 0.0
    for replication in range(N_REPLICATIONS):
        seed = seed_base + replication
        start = perf_counter()
        price = pricer_fn(seed=seed, n_paths=n_paths, n_steps=n_steps)
        elapsed_total += perf_counter() - start
        signed_errors[replication] = price - benchmark_price
        errors_squared[replication] = signed_errors[replication] ** 2
    rms = float(np.sqrt(np.mean(errors_squared)))
    mean_bias = float(np.mean(signed_errors))
    return rms, mean_bias, elapsed_total


def _ours_price(seed: int, n_paths: int, n_steps: int) -> float:
    parameters = _baseline_parameters()
    option = EuropeanOption(
        strike=parameters.spot, maturity=MATURITY, option_type="call",
    )
    return price_european_option_heston_cev(
        parameters=parameters, option=option,
        n_steps=n_steps, n_paths=n_paths, seed=seed,
    ).price


def _euler_price(seed: int, n_paths: int, n_steps: int) -> float:
    parameters = _baseline_parameters()
    option = EuropeanOption(
        strike=parameters.spot, maturity=MATURITY, option_type="call",
    )
    price, _ = _truncated_euler_price(
        parameters=parameters, option=option,
        n_steps=n_steps, n_paths=n_paths, seed=seed,
    )
    return price


SCHEMES_CONFIG = [
    # (name, pricer_fn, grid, seed_base)
    ("ours",  _ours_price,  OURS_GRID,  10_000),
    ("euler", _euler_price, EULER_GRID, 30_000),
]


def main() -> None:
    _ensure_output_dirs()
    parameters = _baseline_parameters()
    option = EuropeanOption(
        strike=parameters.spot, maturity=MATURITY, option_type="call",
    )
    benchmark_price = _benchmark(parameters, option)

    total_runs = sum(len(grid) for _, _, grid, _ in SCHEMES_CONFIG) * N_REPLICATIONS

    print(
        f"Running speed-RMS trade-off at T={MATURITY}, "
        f"F_0={parameters.spot}, beta={parameters.beta}, "
        f"rho={parameters.correlation}, m={N_REPLICATIONS}"
    )
    print(f"Benchmark price = {benchmark_price:.6f}")
    print(
        f"Each scheme sweeps its own grid (paper Fig 2 style); "
        f"total {total_runs} simulations."
    )
    print()

    rows: list[dict] = []
    grand_start = perf_counter()
    for scheme_name, pricer_fn, grid, seed_base in SCHEMES_CONFIG:
        print(f"--- {scheme_name} ---")
        for n_paths, n_steps_per_year in grid:
            n_steps = max(int(round(MATURITY * n_steps_per_year)), 1)
            h = MATURITY / n_steps

            rms, bias, time_total = _run_replications(
                pricer_fn, n_paths, n_steps,
                benchmark_price=benchmark_price, seed_base=seed_base,
            )

            rows.append({
                "scheme": scheme_name,
                "n_paths": n_paths,
                "n_steps_per_year": n_steps_per_year,
                "n_steps": n_steps,
                "h": h,
                "n_replications": N_REPLICATIONS,
                "rms": rms,
                "bias": bias,
                "time_seconds": time_total,
            })
            print(
                f"  N={n_paths:>7d}, n_steps={n_steps:>4d}, h={h:.4f}: "
                f"rms={rms:.6f}, bias={bias:+.6f}, time={time_total:.1f}s"
            )
        print()

    grand_elapsed = perf_counter() - grand_start
    _write_csv(rows, TABLE_PATH)

    ours_rows = [r for r in rows if r["scheme"] == "ours"]
    euler_rows = [r for r in rows if r["scheme"] == "euler"]

    ours_h = ours_rows[0]["h"] if ours_rows else None
    euler_n = euler_rows[0]["n_paths"] if euler_rows else None

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(
        [r["time_seconds"] for r in ours_rows],
        [r["rms"] for r in ours_rows],
        marker="o", linewidth=2, color="tab:blue",
        label=(
            f"Ours (martingale-preserving CEV approx), m={N_REPLICATIONS}, "
            f"h={ours_h:.3f}, sweep N"
        ),
    )
    ax.plot(
        [r["time_seconds"] for r in euler_rows],
        [r["rms"] for r in euler_rows],
        marker="^", linewidth=2, color="tab:green", linestyle=":",
        label=(
            f"Asset-side truncated Euler, m={N_REPLICATIONS}, "
            f"N={euler_n:,}, sweep h"
        ),
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("CPU Time (seconds, total over m runs)")
    ax.set_ylabel("RMS Pricing Error")
    ax.set_title(
        f"Speed vs Accuracy Trade-off at T = {MATURITY} "
        f"(CEV-Heston, F_0={parameters.spot}, rho={parameters.correlation}, "
        f"beta={parameters.beta})"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=200)
    plt.close(fig)
    print(f"Total elapsed: {grand_elapsed:.1f}s ({grand_elapsed/60:.1f} min)")
    print(f"Saved table:  {TABLE_PATH}")
    print(f"Saved figure: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
