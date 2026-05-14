"""Correlated CEV-Heston timestep-convergence diagnostic.

This experiment asks whether the remaining correlated option-price
bias is primarily a coarse-step/time-discretization effect. It fixes
the Case-V-style correlated parameter set used by the main option
experiment, prices one ATM call at T = 4, and compares:

  * frozen_left: the direct Eq. 13/16b frozen-log anchor.
  * power_projected: the beta-aware power-coordinate projected anchor.

Both schemes are measured against one high-resolution self-reference
benchmark. For beta < 1 this is not a closed-form truth price; the
figure is a convergence diagnostic, not an exact-transition claim.
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

from src.heston_cev_benchmark import self_reference_benchmark
from src.heston_cev_simulation import price_european_option_heston_cev
from src.utils import CEVHestonModelParameters, EuropeanOption


MATURITY = 4.0
N_STEPS_GRID = [25, 50, 100, 200, 400]
N_PATHS = 25_000
N_REPLICATIONS = 12

# One benchmark shared by all timestep rows. At T=4 this gives 800
# steps, finer than the largest diagnostic grid point.
N_PATHS_REFERENCE = 120_000
N_STEPS_PER_YEAR_REFERENCE = 200
BENCHMARK_SEED = 999_999

TABLE_PATH = PROJECT_ROOT / "results" / "tables" / "heston_cev_timestep_convergence.csv"
FIGURE_PATH = PROJECT_ROOT / "results" / "figures" / "heston_cev_timestep_convergence.png"


def _ensure_output_dirs() -> None:
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _write_csv(rows: list[dict], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _baseline_parameters() -> CEVHestonModelParameters:
    return CEVHestonModelParameters(
        spot=1.0,
        initial_variance=0.09,
        kappa=1.5,
        theta=0.09,
        xi=0.6,
        correlation=-0.8,
        beta=0.4,
        risk_free_rate=0.0,
    )


def _run_scheme_replications(
    parameters: CEVHestonModelParameters,
    option: EuropeanOption,
    n_steps: int,
    scheme: str,
    seed_base: int,
) -> tuple[float, float, float, float]:
    prices = np.empty(N_REPLICATIONS)
    elapsed_total = 0.0
    for replication in range(N_REPLICATIONS):
        seed = seed_base + replication
        start = perf_counter()
        result = price_european_option_heston_cev(
            parameters=parameters,
            option=option,
            n_steps=n_steps,
            n_paths=N_PATHS,
            seed=seed,
            conditional_scheme=scheme,  # type: ignore[arg-type]
        )
        elapsed_total += perf_counter() - start
        prices[replication] = result.price

    mean_price = float(np.mean(prices))
    stdev = float(np.std(prices, ddof=1))
    sem = stdev / float(np.sqrt(N_REPLICATIONS))
    return mean_price, stdev, sem, elapsed_total


def _plot(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1, alpha=0.65)

    styles = {
        "frozen_left": {
            "marker": "o",
            "linestyle": "-",
            "color": "tab:blue",
            "label": "Frozen-left anchor",
        },
        "power_projected": {
            "marker": "D",
            "linestyle": "-.",
            "color": "tab:purple",
            "label": "Power-projected anchor",
        },
    }
    for scheme, style in styles.items():
        scheme_rows = [row for row in rows if row["scheme"] == scheme]
        ax.errorbar(
            [row["n_steps"] for row in scheme_rows],
            [row["bias_signed"] for row in scheme_rows],
            yerr=[2.0 * row["combined_sem"] for row in scheme_rows],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2,
            color=style["color"],
            capsize=3,
            capthick=1.2,
            label=f"{style['label']}, m={N_REPLICATIONS}",
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of Timesteps")
    ax.set_ylabel("ATM Call Bias vs Self-Reference")
    ax.set_title(
        f"Correlated CEV-Heston Timestep Convergence "
        f"(T={MATURITY:g}, N={N_PATHS:,})"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=200)
    plt.close(fig)


def main() -> None:
    _ensure_output_dirs()
    parameters = _baseline_parameters()
    option = EuropeanOption(strike=parameters.spot, maturity=MATURITY, option_type="call")

    print("Running correlated CEV-Heston timestep-convergence diagnostic...")
    print(
        f"Configuration: T={MATURITY}, beta={parameters.beta}, rho={parameters.correlation}, "
        f"N={N_PATHS:,}, m={N_REPLICATIONS}, grids={N_STEPS_GRID}"
    )
    print(
        f"Benchmark: n_paths={N_PATHS_REFERENCE:,}, "
        f"n_steps_per_year={N_STEPS_PER_YEAR_REFERENCE}"
    )

    benchmark = self_reference_benchmark(
        parameters=parameters,
        option=option,
        n_paths_reference=N_PATHS_REFERENCE,
        n_steps_per_year_reference=N_STEPS_PER_YEAR_REFERENCE,
        seed=BENCHMARK_SEED,
        conditional_scheme="frozen_left",
    )
    print(
        f"Benchmark price={benchmark.price:.6f}, "
        f"SE={benchmark.standard_error:.6f}, method={benchmark.method}"
    )
    print()

    rows: list[dict] = []
    grand_start = perf_counter()
    for n_steps in N_STEPS_GRID:
        h = MATURITY / n_steps
        for scheme, seed_offset in [
            ("frozen_left", 100_000),
            ("power_projected", 200_000),
        ]:
            mean_price, stdev, sem, elapsed = _run_scheme_replications(
                parameters=parameters,
                option=option,
                n_steps=n_steps,
                scheme=scheme,
                seed_base=seed_offset + n_steps * 1_000,
            )
            bias = mean_price - benchmark.price
            combined_sem = float(np.sqrt(sem * sem + benchmark.standard_error ** 2))
            row = {
                "maturity_T": MATURITY,
                "scheme": scheme,
                "n_steps": n_steps,
                "h": h,
                "n_paths": N_PATHS,
                "n_replications": N_REPLICATIONS,
                "benchmark_price": benchmark.price,
                "benchmark_se": benchmark.standard_error,
                "benchmark_method": benchmark.method,
                "mean_price": mean_price,
                "bias_signed": bias,
                "abs_bias": abs(bias),
                "stdev_across_replications": stdev,
                "sem_of_bias": sem,
                "combined_sem": combined_sem,
                "abs_bias_in_combined_sems": abs(bias) / max(combined_sem, 1e-300),
                "elapsed_seconds": elapsed,
            }
            rows.append(row)
            print(
                f"  n_steps={n_steps:>4d}, {scheme:<15s}: "
                f"bias={bias:+.6f}, abs={abs(bias):.6f}, "
                f"cSEM={row['abs_bias_in_combined_sems']:.2f}, "
                f"time={elapsed:.1f}s"
            )

    grand_elapsed = perf_counter() - grand_start
    _write_csv(rows, TABLE_PATH)
    _plot(rows)

    print()
    print("Summary by timestep grid:")
    for n_steps in N_STEPS_GRID:
        grid_rows = [row for row in rows if row["n_steps"] == n_steps]
        max_abs_bias = max(row["abs_bias"] for row in grid_rows)
        max_combined = max(row["abs_bias_in_combined_sems"] for row in grid_rows)
        print(
            f"  n_steps={n_steps:>4d}: "
            f"max_abs_bias={max_abs_bias:.6f}, "
            f"max_combined_SE_error={max_combined:.2f}"
        )

    print()
    print(f"Total elapsed: {grand_elapsed:.1f}s ({grand_elapsed/60:.1f} min)")
    print(f"Saved table:  {TABLE_PATH}")
    print(f"Saved figure: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
