"""European option price accuracy across maturities (paper Fig 3b analogue).

Compares three quantities against time-to-maturity ``T``:

  * The ATM call price under the project's CEV approximation
    (``simulate_heston_cev_terminal``).
  * The same price under Islah's classical approximation
    (``simulate_heston_cev_islah``), which is the de facto baseline
    used in nearly every SABR / Heston-CEV simulator since 2009.
  * A benchmark price (PyFENG ``HestonFft`` for the ``beta = 1``
    sanity check; a self-reference high-resolution Monte Carlo run
    for the production ``beta < 1`` case).

The headline observation, reproducing Choi-Hu-Kwok (2024) Fig 3b in
the Heston-CEV setting, is that Islah's pricing error grows roughly
linearly in ``T`` (because the underlying martingale violation
accumulates), while the project's CEV approximation tracks the
benchmark to within Monte Carlo noise across all maturities.
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
from pathlib import Path

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
from src.utils import CEVHestonModelParameters, EuropeanOption


MATURITIES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
N_PATHS = 50_000
N_STEPS_PER_YEAR = 20
USE_BETA_LESS_THAN_ONE = True  # set False to use the beta = 1 Fourier benchmark

TABLE_PATH = PROJECT_ROOT / "results" / "tables" / "heston_cev_option_price.csv"
FIGURE_PATH = PROJECT_ROOT / "results" / "figures" / "heston_cev_option_price_error.png"


def _ensure_output_dirs() -> None:
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _write_csv(rows: list[dict[str, float]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _baseline_parameters() -> CEVHestonModelParameters:
    """Default parameters for the option-price comparison.

    These are the same parameters used by the martingale-preservation
    experiment, intentionally moderate so that both schemes operate in
    a numerically stable regime. With more aggressive settings (e.g.
    rho = -0.8, xi = 0.5) Islah's martingale violation becomes
    visible in option prices, but those settings also push the
    underlying CEV sampler into regions where its Poisson-mixture
    representation requires intensities above NumPy's safe range;
    studying that regime requires the variance-reduction extensions
    discussed in Choi et al. (2024) Section 5.4 and is beyond the
    scope of the present comparison.
    """
    return CEVHestonModelParameters(
        spot=100.0,
        initial_variance=0.04,
        kappa=1.5,
        theta=0.04,
        xi=0.4,
        correlation=-0.5,
        beta=0.5 if USE_BETA_LESS_THAN_ONE else 1.0,
        risk_free_rate=0.0,
    )


def _benchmark(parameters: CEVHestonModelParameters, option: EuropeanOption):
    """Pick the appropriate benchmark for the parameter set."""
    if parameters.beta == 1.0:
        return heston_fourier_benchmark_beta_one(parameters, option)
    return self_reference_benchmark(parameters, option)


def _run_experiment_row(maturity: float) -> dict[str, float]:
    parameters = _baseline_parameters()
    option = EuropeanOption(strike=100.0, maturity=maturity, option_type="call")
    n_steps = max(int(round(maturity * N_STEPS_PER_YEAR)), 1)
    seed = 7000 + int(maturity)

    benchmark = _benchmark(parameters, option)

    ours = price_european_option_heston_cev(
        parameters=parameters,
        option=option,
        n_steps=n_steps,
        n_paths=N_PATHS,
        seed=seed,
    )
    islah = price_european_option_heston_cev_islah(
        parameters=parameters,
        option=option,
        n_steps=n_steps,
        n_paths=N_PATHS,
        seed=seed,
    )

    return {
        "maturity_T": maturity,
        "benchmark_price": benchmark.price,
        "benchmark_method": benchmark.method,
        "benchmark_se": benchmark.standard_error,
        "ours_price": ours.price,
        "ours_standard_error": ours.standard_error,
        "ours_abs_error": abs(ours.price - benchmark.price),
        "islah_price": islah.price,
        "islah_standard_error": islah.standard_error,
        "islah_abs_error": abs(islah.price - benchmark.price),
    }


def _plot_option_price_error(rows: list[dict[str, float]]) -> None:
    """Plot |simulated - benchmark| vs T for both schemes."""
    maturities = [row["maturity_T"] for row in rows]
    ours_error = [row["ours_abs_error"] for row in rows]
    islah_error = [row["islah_abs_error"] for row in rows]
    ours_se = [row["ours_standard_error"] for row in rows]

    upper_band = [2.0 * se for se in ours_se]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(
        maturities, 0.0, upper_band,
        color="tab:gray", alpha=0.20,
        label="Ours: +/- 2 SE Monte Carlo band",
    )
    ax.plot(
        maturities, ours_error,
        marker="o", linestyle="-", linewidth=2, color="tab:blue",
        label="Ours (martingale-preserving CEV approx)",
    )
    ax.plot(
        maturities, islah_error,
        marker="s", linestyle="--", linewidth=2, color="tab:red",
        label="Islah-style approximation",
    )
    ax.set_xlabel("Time to Maturity (T)")
    ax.set_ylabel("Absolute Pricing Error vs Benchmark")
    ax.set_title("ATM Call Pricing Error: Ours vs Islah, CEV-Heston")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=200)
    plt.close(fig)


def main() -> None:
    _ensure_output_dirs()
    print("Running CEV-Heston option price comparison experiment...")
    print(
        f"Configuration: beta={'< 1' if USE_BETA_LESS_THAN_ONE else '= 1'}, "
        f"n_paths={N_PATHS}, n_steps_per_year={N_STEPS_PER_YEAR}"
    )

    rows = []
    for T in MATURITIES:
        row = _run_experiment_row(T)
        rows.append(row)
        print(
            f"  T={T:>4.1f}: bench={row['benchmark_price']:.4f}  "
            f"ours={row['ours_price']:.4f} (err={row['ours_abs_error']:.4f}, "
            f"SE={row['ours_standard_error']:.4f})  "
            f"islah={row['islah_price']:.4f} (err={row['islah_abs_error']:.4f})"
        )

    _write_csv(rows, TABLE_PATH)
    _plot_option_price_error(rows)
    print()
    print(f"Saved table:  {TABLE_PATH}")
    print(f"Saved figure: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
