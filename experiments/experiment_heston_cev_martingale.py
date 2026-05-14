"""Martingale preservation validation for the step-by-step CEV-Heston model.

This experiment validates that the step-by-step CEV exact sampling, combined
with the theoretically correct conditional mean for the CIR process over dt,
natively preserves the martingale property E[F_T] = F_0 across all maturities,
and contrasts it with Islah's classical approximation, which is expected to
violate the martingale property monotonically in T.

We follow Choi-Hu-Kwok (2024, Sec. 5) by running ``m`` independent
replications at each maturity and reporting:

  * The bias E[F_T] - F_0, averaged across replications.
  * The cross-replication standard deviation of the per-run mean
    (this is the m=50 quantity reported in the paper's tables).
  * The standard error of the across-replication mean, ``stdev / sqrt(m)``,
    which is the appropriate uncertainty band for the bias estimate
    itself; the figure uses +/- 2 of these SEMs as error bars.

A correct CEV-approximation simulator (ours) produces a curve that
fluctuates around zero within its error bars, with no systematic
drift in T. Islah's curve, in contrast, is expected to drift upward
monotonically as the per-step martingale violation accumulates
(Choi-Hu-Kwok 2024, Fig 3a).

CPU note: at N_PATHS=100,000, N_REPLICATIONS=50, and 10 maturities,
this script runs ~1000 single simulations across ours+Islah. On a
laptop this is roughly 15--40 minutes total depending on the variance
backend and N_STEPS_PER_YEAR. For a quick smoke test, drop
N_REPLICATIONS to 5 or 10 first.
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

from src.heston_cev_simulation import simulate_heston_cev_terminal
from src.islah_approximation import simulate_heston_cev_islah
from src.pyfeng_adapter import PyFengVarianceStepper
from src.utils import CEVHestonModelParameters


MATURITIES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
N_PATHS = 100_000           # paper uses 1e5
N_REPLICATIONS = 50         # paper uses 50
N_STEPS_PER_YEAR = 1        # paper Fig 3 uses h = 1

TABLE_PATH = PROJECT_ROOT / "results" / "tables" / "heston_cev_martingale.csv"
FIGURE_PATH = PROJECT_ROOT / "results" / "figures" / "heston_cev_martingale_error.png"


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

    The key dimensionless quantity is sigma_eff = sqrt(v_0) / F_0^{1-beta},
    which controls the size of Islah's per-step martingale violation.
    The paper's Case V has sigma_eff approximately 0.28; we match this
    with F_0 = 1, v_0 = theta = 0.09 (so sqrt(v_0) = 0.3) and beta = 0.4.
    With these settings Islah's bias at T = 10 is expected to be a few
    percent of F_0 (i.e. order 0.03 in absolute terms), comfortably
    above the Monte Carlo noise at N = 1e5, m = 50.
    """
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


def _single_run(
    parameters: CEVHestonModelParameters,
    maturity: float,
    n_steps: int,
    seed: int,
    scheme: str,
) -> tuple[float, float, int]:
    """Run a single N-path simulation and return (mean, clamp_rate, n_absorbed).

    A fresh PyFengVarianceStepper is constructed for every run because
    the PyFENG MC class instance carries internal random state across
    `cond_states_step` calls; reusing it would silently couple
    consecutive runs through that state.
    """
    dt = maturity / n_steps
    stepper = PyFengVarianceStepper(
        parameters=parameters,
        n_path=N_PATHS,
        dt=dt,
        seed=seed,
    )
    if scheme == "ours":
        result = simulate_heston_cev_terminal(
            parameters=parameters,
            maturity=maturity,
            n_steps=n_steps,
            n_paths=N_PATHS,
            seed=seed,
            variance_stepper=stepper,
        )
    elif scheme == "islah":
        result = simulate_heston_cev_islah(
            parameters=parameters,
            maturity=maturity,
            n_steps=n_steps,
            n_paths=N_PATHS,
            seed=seed,
            variance_stepper=stepper,
        )
    else:
        raise ValueError(f"unknown scheme: {scheme}")
    return (
        float(np.mean(result.terminal_prices)),
        float(result.clamp_trigger_rate),
        int(result.n_paths_absorbed),
    )


def _run_experiment_row(maturity: float) -> dict[str, float]:
    """Run m replications at one maturity and return the aggregated row.

    Both schemes use the SAME per-replication seed so that their MC
    noise is paired (variance reduction across the comparison). Across
    different replications we increment the seed deterministically.
    """
    parameters = _baseline_parameters()
    n_steps = max(int(round(maturity * N_STEPS_PER_YEAR)), 1)

    means_ours = np.empty(N_REPLICATIONS)
    means_islah = np.empty(N_REPLICATIONS)
    clamp_ours = np.empty(N_REPLICATIONS)
    clamp_islah = np.empty(N_REPLICATIONS)
    absorbed_ours = np.empty(N_REPLICATIONS, dtype=int)
    absorbed_islah = np.empty(N_REPLICATIONS, dtype=int)

    start = perf_counter()
    for replication in range(N_REPLICATIONS):
        # Multiplicative offset on T avoids seed overlap between
        # different maturities while keeping the replication index
        # the rapidly varying digit.
        seed = 1_000_000 + int(maturity) * 10_000 + replication

        means_ours[replication], clamp_ours[replication], absorbed_ours[replication] = (
            _single_run(parameters, maturity, n_steps, seed, scheme="ours")
        )
        means_islah[replication], clamp_islah[replication], absorbed_islah[replication] = (
            _single_run(parameters, maturity, n_steps, seed, scheme="islah")
        )
    elapsed = perf_counter() - start

    # Across-replication aggregates. Bias is signed: E[F_T] - F_0.
    # Islah is expected to give a positive bias (supermartingale).
    bias_ours = float(np.mean(means_ours)) - parameters.spot
    bias_islah = float(np.mean(means_islah)) - parameters.spot

    # Standard deviation of the per-run mean across the m replications
    # (this is the column the paper reports in its tables) and the
    # standard error of the *across-replication mean*, which is the
    # uncertainty on the bias estimate itself. The latter is what the
    # error bars in the figure use.
    std_ours = float(np.std(means_ours, ddof=1))
    std_islah = float(np.std(means_islah, ddof=1))
    sem_ours = std_ours / float(np.sqrt(N_REPLICATIONS))
    sem_islah = std_islah / float(np.sqrt(N_REPLICATIONS))

    return {
        "maturity_T": maturity,
        "exact_mean": parameters.spot,
        "n_paths": N_PATHS,
        "n_replications": N_REPLICATIONS,
        "elapsed_seconds": elapsed,
        # Ours
        "ours_mean_of_means": float(np.mean(means_ours)),
        "ours_bias_signed": bias_ours,
        "ours_abs_bias": abs(bias_ours),
        "ours_stdev_across_replications": std_ours,
        "ours_sem_of_bias": sem_ours,
        "ours_abs_bias_in_sems": abs(bias_ours) / max(sem_ours, 1e-300),
        "ours_avg_clamp_trigger_rate": float(np.mean(clamp_ours)),
        "ours_avg_n_paths_absorbed": float(np.mean(absorbed_ours)),
        # Islah
        "islah_mean_of_means": float(np.mean(means_islah)),
        "islah_bias_signed": bias_islah,
        "islah_abs_bias": abs(bias_islah),
        "islah_stdev_across_replications": std_islah,
        "islah_sem_of_bias": sem_islah,
        "islah_abs_bias_in_sems": abs(bias_islah) / max(sem_islah, 1e-300),
        "islah_avg_clamp_trigger_rate": float(np.mean(clamp_islah)),
        "islah_avg_n_paths_absorbed": float(np.mean(absorbed_islah)),
    }


def _plot_martingale_error(rows: list[dict[str, float]]) -> None:
    """Plot signed bias E[F_T] - F_0 vs T, with +/- 2 SEM error bars.

    The signed bias is preferred to |bias| here because Islah is a
    supermartingale (bias > 0 systematically); plotting the sign
    makes the directionality of the violation visible. Error bars
    are 2 * (stdev_across_replications / sqrt(m)), which is the
    correct uncertainty on the bias point estimate.
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
    ax.set_ylabel("Forward Price Error E[F_T] - F_0")
    ax.set_title(
        f"Martingale Preservation: Ours vs Islah, CEV-Heston "
        f"(N={N_PATHS:,}, m={N_REPLICATIONS}, h={1.0/N_STEPS_PER_YEAR:.2f})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=200)
    plt.close(fig)


def main() -> None:
    _ensure_output_dirs()
    print("Running step-by-step CEV-Heston martingale preservation experiment...")
    print(
        f"Configuration: N={N_PATHS:,}, m={N_REPLICATIONS}, "
        f"n_steps_per_year={N_STEPS_PER_YEAR}, total maturities={len(MATURITIES)}"
    )
    print(
        f"Total simulations to run: "
        f"{len(MATURITIES) * N_REPLICATIONS * 2} "
        f"({len(MATURITIES)} T x {N_REPLICATIONS} m x 2 schemes)"
    )
    print()

    rows = []
    grand_start = perf_counter()
    for T in MATURITIES:
        row = _run_experiment_row(T)
        rows.append(row)
        print(
            f"  T={T:>4.1f} ({row['elapsed_seconds']:.1f}s): "
            f"ours bias={row['ours_bias_signed']:+.5f} +/- {2.0*row['ours_sem_of_bias']:.5f} "
            f"({row['ours_abs_bias_in_sems']:.2f} SEMs) | "
            f"islah bias={row['islah_bias_signed']:+.5f} +/- {2.0*row['islah_sem_of_bias']:.5f} "
            f"({row['islah_abs_bias_in_sems']:.2f} SEMs)"
        )

    grand_elapsed = perf_counter() - grand_start
    _write_csv(rows, TABLE_PATH)
    _plot_martingale_error(rows)

    print()
    print(f"Total elapsed: {grand_elapsed:.1f}s ({grand_elapsed/60:.1f} min)")
    print(f"Saved table:  {TABLE_PATH}")
    print(f"Saved figure: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
