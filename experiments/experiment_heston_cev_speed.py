"""Preliminary speed-vs-accuracy trade-off for CEV-Heston pricing.

Plots root-mean-square error against CPU time for four schemes:

  * The original frozen-left CEV anchor.
  * The beta-aware power-projected CEV anchor.
  * Islah's approximation.
  * A naive asset-side truncated-Euler baseline.

This is a diagnostic, not a strict ranking. All methods are measured
against the same high-resolution self-reference benchmark.

DESIGN DETAILS

Each method uses the same grid of ``(N, h)`` points. Each point is run
``m=50`` times.
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
from src.islah_approximation import price_european_option_heston_cev_islah
from src.utils import CEVHestonModelParameters, EuropeanOption, get_rng, standard_error


MATURITY = 4.0
N_REPLICATIONS = 50

COMMON_GRID = [
    (5_000,   1),    # n_steps =   4, h = 1.0
    (10_000,  2),    # n_steps =   8, h = 0.5
    (20_000,  4),    # n_steps =  16, h = 0.25
    (40_000,  8),    # n_steps =  32, h = 0.125
    (80_000,  16),   # n_steps =  64, h = 0.0625
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
        correlation=-0.3,
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


def _frozen_left_price(seed: int, n_paths: int, n_steps: int) -> float:
    parameters = _baseline_parameters()
    option = EuropeanOption(
        strike=parameters.spot, maturity=MATURITY, option_type="call",
    )
    return price_european_option_heston_cev(
        parameters=parameters, option=option,
        n_steps=n_steps, n_paths=n_paths, seed=seed,
        conditional_scheme="frozen_left",
    ).price


def _power_projected_price(seed: int, n_paths: int, n_steps: int) -> float:
    parameters = _baseline_parameters()
    option = EuropeanOption(
        strike=parameters.spot, maturity=MATURITY, option_type="call",
    )
    return price_european_option_heston_cev(
        parameters=parameters, option=option,
        n_steps=n_steps, n_paths=n_paths, seed=seed,
        conditional_scheme="power_projected",
    ).price


def _islah_price(seed: int, n_paths: int, n_steps: int) -> float:
    parameters = _baseline_parameters()
    option = EuropeanOption(
        strike=parameters.spot, maturity=MATURITY, option_type="call",
    )
    return price_european_option_heston_cev_islah(
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
    ("frozen_left", _frozen_left_price, COMMON_GRID, 10_000),
    ("power_projected", _power_projected_price, COMMON_GRID, 20_000),
    ("islah", _islah_price, COMMON_GRID, 30_000),
    ("euler", _euler_price, COMMON_GRID, 40_000),
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
        f"Each scheme uses the same (N, h) grid; "
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

    frozen_rows = [r for r in rows if r["scheme"] == "frozen_left"]
    power_rows = [r for r in rows if r["scheme"] == "power_projected"]
    islah_rows = [r for r in rows if r["scheme"] == "islah"]
    euler_rows = [r for r in rows if r["scheme"] == "euler"]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(
        [r["time_seconds"] for r in frozen_rows],
        [r["rms"] for r in frozen_rows],
        marker="o", linewidth=2, color="tab:blue",
        label=f"Frozen-left anchor, m={N_REPLICATIONS}",
    )
    ax.plot(
        [r["time_seconds"] for r in power_rows],
        [r["rms"] for r in power_rows],
        marker="D", linewidth=2, color="tab:purple", linestyle="-.",
        label=f"Power-projected anchor, m={N_REPLICATIONS}",
    )
    ax.plot(
        [r["time_seconds"] for r in islah_rows],
        [r["rms"] for r in islah_rows],
        marker="s", linewidth=2, color="tab:red", linestyle="--",
        label=f"Islah-style, m={N_REPLICATIONS}",
    )
    ax.plot(
        [r["time_seconds"] for r in euler_rows],
        [r["rms"] for r in euler_rows],
        marker="^", linewidth=2, color="tab:green", linestyle=":",
        label=f"Asset-side truncated Euler, m={N_REPLICATIONS}",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("CPU Time (seconds, total over m runs)")
    ax.set_ylabel("RMS Pricing Error")
    ax.set_title(
        f"Preliminary Speed vs Accuracy at T = {MATURITY} "
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
