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

Following Choi-Hu-Kwok (2024, Sec. 5), each maturity is run
``m`` times independently. We report:

  * Per-scheme bias = mean(price across m runs) - benchmark.
  * Per-scheme SEM  = stdev across runs / sqrt(m).
  * Error bars in the figure are +/- 2 SEMs.

This zero-correlation variant is a diagnostic: when rho = 0, the
correlated frozen-coefficient term disappears and the conditional CEV
step should be much cleaner. For beta < 1 the benchmark is still a
self-reference Monte Carlo benchmark rather than a closed-form truth.
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
from src.utils import CEVHestonModelParameters, EuropeanOption


MATURITIES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
N_PATHS = 100_000           # paper uses 1e5
N_REPLICATIONS = 50         # paper uses 50
N_STEPS_PER_YEAR = 1        # paper Fig 3 uses h = 1
USE_BETA_LESS_THAN_ONE = True  # set False to use the beta = 1 Fourier benchmark

TABLE_PATH = PROJECT_ROOT / "results" / "tables" / "heston_cev_option_zerocorr_price.csv"
FIGURE_PATH = PROJECT_ROOT / "results" / "figures" / "heston_cev_option_zerocorr_price_error.png"


def _ensure_output_dirs() -> None:
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _write_csv(rows: list[dict[str, float]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _baseline_parameters() -> CEVHestonModelParameters:
    """Parameters chosen to match Choi-Hu-Kwok (2024) Case V scaling.

    KEEP IN SYNC with experiment_heston_cev_martingale.py and
    experiment_heston_cev_speed.py. The key dimensionless quantity is
    sigma_eff = sqrt(v_0) / F_0^{1-beta}, which controls the size of
    Islah's per-step martingale violation. The paper's Case V has
    sigma_eff approximately 0.28; we match this with F_0 = 1,
    v_0 = theta = 0.09 (so sqrt(v_0) = 0.3) and beta = 0.4. With
    these settings Islah's pricing error at T = 10 is expected to
    be a few percent of F_0, comfortably above the Monte Carlo
    noise at N = 1e5, m = 50.
    """
    return CEVHestonModelParameters(
        spot=1.0,
        initial_variance=0.09,
        kappa=1.5,
        theta=0.09,
        xi=0.6,
        correlation=0,
        beta=0.4 if USE_BETA_LESS_THAN_ONE else 1.0,
        risk_free_rate=0.0,
    )


def _benchmark(parameters: CEVHestonModelParameters, option: EuropeanOption):
    """Pick the appropriate benchmark for the parameter set.

    For beta = 1 we use PyFENG's Fourier pricer (essentially exact;
    SE = 0). For beta < 1 we use a high-resolution self-reference
    MC run (default 500k paths, 50 steps/year). The benchmark is run
    ONCE per maturity, not replicated, because it is already a
    fine-grid expensive run; its own SE is reported in the CSV so
    that downstream consumers can factor it into accuracy claims.
    """
    if parameters.beta == 1.0:
        return heston_fourier_benchmark_beta_one(parameters, option)
    return self_reference_benchmark(parameters, option)


def _single_run(
    parameters: CEVHestonModelParameters,
    option: EuropeanOption,
    n_steps: int,
    seed: int,
    scheme: str,
) -> float:
    """Run one replication of the requested scheme; return the price."""
    if scheme == "ours":
        result = price_european_option_heston_cev(
            parameters=parameters, option=option,
            n_steps=n_steps, n_paths=N_PATHS, seed=seed,
        )
    elif scheme == "islah":
        result = price_european_option_heston_cev_islah(
            parameters=parameters, option=option,
            n_steps=n_steps, n_paths=N_PATHS, seed=seed,
        )
    else:
        raise ValueError(f"unknown scheme: {scheme}")
    return float(result.price)


def _run_experiment_row(maturity: float) -> dict[str, float]:
    """Run m replications at one maturity and return the aggregated row.

    Both schemes use the SAME per-replication seed so that their MC
    noise is paired across the comparison; this is variance reduction
    for the scheme-vs-scheme comparison and matches the design of
    the martingale experiment.
    """
    parameters = _baseline_parameters()
    option = EuropeanOption(strike=1.0, maturity=maturity, option_type="call")
    n_steps = max(int(round(maturity * N_STEPS_PER_YEAR)), 1)

    benchmark = _benchmark(parameters, option)

    prices_ours = np.empty(N_REPLICATIONS)
    prices_islah = np.empty(N_REPLICATIONS)

    start = perf_counter()
    for replication in range(N_REPLICATIONS):
        seed = 7_000_000 + int(maturity) * 10_000 + replication
        prices_ours[replication] = _single_run(
            parameters, option, n_steps, seed, scheme="ours",
        )
        prices_islah[replication] = _single_run(
            parameters, option, n_steps, seed, scheme="islah",
        )
    elapsed = perf_counter() - start

    bias_ours = float(np.mean(prices_ours)) - benchmark.price
    bias_islah = float(np.mean(prices_islah)) - benchmark.price
    std_ours = float(np.std(prices_ours, ddof=1))
    std_islah = float(np.std(prices_islah, ddof=1))
    sem_ours = std_ours / float(np.sqrt(N_REPLICATIONS))
    sem_islah = std_islah / float(np.sqrt(N_REPLICATIONS))

    return {
        "maturity_T": maturity,
        "n_paths": N_PATHS,
        "n_replications": N_REPLICATIONS,
        "elapsed_seconds": elapsed,
        "benchmark_price": benchmark.price,
        "benchmark_se": benchmark.standard_error,
        "benchmark_method": benchmark.method,
        # Ours
        "ours_mean_price": float(np.mean(prices_ours)),
        "ours_bias_signed": bias_ours,
        "ours_abs_bias": abs(bias_ours),
        "ours_stdev_across_replications": std_ours,
        "ours_sem_of_bias": sem_ours,
        "ours_abs_bias_in_sems": abs(bias_ours) / max(sem_ours, 1e-300),
        # Islah
        "islah_mean_price": float(np.mean(prices_islah)),
        "islah_bias_signed": bias_islah,
        "islah_abs_bias": abs(bias_islah),
        "islah_stdev_across_replications": std_islah,
        "islah_sem_of_bias": sem_islah,
        "islah_abs_bias_in_sems": abs(bias_islah) / max(sem_islah, 1e-300),
    }


def _plot_option_price_error(rows: list[dict[str, float]]) -> None:
    """Plot signed pricing bias vs T, with +/- 2 SEM error bars.

    Mirrors the visual style of the martingale experiment for
    consistency. Islah's bias is expected to grow roughly linearly
    in T (the option-price error tracks the underlying martingale
    violation, scaled by the option's delta).
    """
    maturities = [row["maturity_T"] for row in rows]
    bias_ours = [row["ours_bias_signed"] for row in rows]
    bias_islah = [row["islah_bias_signed"] for row in rows]
    err_ours = [2.0 * row["ours_sem_of_bias"] for row in rows]
    err_islah = [2.0 * row["islah_sem_of_bias"] for row in rows]

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.axhline(0, color="black", linestyle=":", linewidth=1, alpha=0.6)
    ax.errorbar(
        maturities, bias_ours, yerr=err_ours,
        marker="o", linestyle="-", linewidth=2, color="tab:blue",
        capsize=3, capthick=1.2,
        label=f"Ours (CEV approx), m={N_REPLICATIONS}",
    )
    ax.errorbar(
        maturities, bias_islah, yerr=err_islah,
        marker="s", linestyle="--", linewidth=2, color="tab:red",
        capsize=3, capthick=1.2,
        label=f"Islah-style, m={N_REPLICATIONS}",
    )
    ax.set_xlabel("Time to Maturity (T)")
    ax.set_ylabel("ATM Call Pricing Error (price - benchmark)")
    ax.set_title(
        f"ATM Call Pricing Error vs T: Ours vs Islah, CEV-Heston "
        f"(N={N_PATHS:,}, m={N_REPLICATIONS}, h={1.0/N_STEPS_PER_YEAR:.2f})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=200)
    plt.close(fig)


def main() -> None:
    _ensure_output_dirs()
    print("Running CEV-Heston option price comparison experiment...")
    print(
        f"Configuration: beta={'< 1' if USE_BETA_LESS_THAN_ONE else '= 1'}, "
        f"N={N_PATHS:,}, m={N_REPLICATIONS}, "
        f"n_steps_per_year={N_STEPS_PER_YEAR}"
    )
    print(
        f"Total simulations to run: "
        f"{len(MATURITIES) * N_REPLICATIONS * 2} "
        f"({len(MATURITIES)} T x {N_REPLICATIONS} m x 2 schemes), "
        f"plus {len(MATURITIES)} benchmark runs"
    )
    print()

    rows = []
    grand_start = perf_counter()
    for T in MATURITIES:
        row = _run_experiment_row(T)
        rows.append(row)
        print(
            f"  T={T:>4.1f} ({row['elapsed_seconds']:.1f}s): "
            f"bench={row['benchmark_price']:.5f} (SE={row['benchmark_se']:.5f}) | "
            f"ours bias={row['ours_bias_signed']:+.5f} +/- {2.0*row['ours_sem_of_bias']:.5f} "
            f"({row['ours_abs_bias_in_sems']:.2f} SEMs) | "
            f"islah bias={row['islah_bias_signed']:+.5f} +/- {2.0*row['islah_sem_of_bias']:.5f} "
            f"({row['islah_abs_bias_in_sems']:.2f} SEMs)"
        )

    grand_elapsed = perf_counter() - grand_start
    _write_csv(rows, TABLE_PATH)
    _plot_option_price_error(rows)
    print()
    print(f"Total elapsed: {grand_elapsed:.1f}s ({grand_elapsed/60:.1f} min)")
    print(f"Saved table:  {TABLE_PATH}")
    print(f"Saved figure: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
