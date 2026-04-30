"""Martingale preservation validation for the step-by-step CEV-Heston model.

This experiment validates that the step-by-step CEV exact sampling, combined
with the theoretically correct conditional mean for the CIR process over dt,
natively preserves the martingale property E[F_T] = F_0 across all maturities.

The expected appearance of the resulting figure is a curve that fluctuates
around zero within the analytic +/- 2 SE Monte Carlo confidence band, with
no monotonic drift in T. A curve that grows linearly in T (or any other
systematic deviation outside the band) indicates a bug, not statistical noise.
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

from src.heston_cev_simulation import simulate_heston_cev_terminal
from src.pyfeng_adapter import PyFengVarianceStepper
from src.utils import CEVHestonModelParameters


MATURITIES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
N_PATHS = 50_000
N_STEPS_PER_YEAR = 20

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
    return CEVHestonModelParameters(
        spot=100.0,
        initial_variance=0.04,
        kappa=1.5,
        theta=0.04,
        xi=0.4,
        correlation=-0.5,
        beta=0.5,
        risk_free_rate=0.0,
    )


def _run_experiment_row(maturity: float) -> dict[str, float]:
    """Run one maturity and return the diagnostic row.

    The variance side delegates explicitly to PyFENG's
    ``HestonMcChoiKwok2023PoisGe`` (the peer-reviewed Choi-Kwok 2024
    Poisson-conditioning scheme), as required by the project brief.
    The asset side is the project's own CEV-on-Heston extension.

    The simulator returns un-discounted terminal forward prices, so the
    martingale check ``E[F_T] = F_0`` is direct -- there is no discount
    factor to undo. Discounting is applied separately by the pricing
    helper when option prices are needed.
    """
    parameters = _baseline_parameters()
    n_steps = max(int(round(maturity * N_STEPS_PER_YEAR)), 1)
    seed = 1000 + int(maturity)
    dt = maturity / n_steps

    pyfeng_stepper = PyFengVarianceStepper(
        parameters=parameters,
        n_path=N_PATHS,
        dt=dt,
        seed=seed,
    )

    result = simulate_heston_cev_terminal(
        parameters=parameters,
        maturity=maturity,
        n_steps=n_steps,
        n_paths=N_PATHS,
        seed=seed,
        variance_stepper=pyfeng_stepper,
    )
    terminal_prices = result.terminal_prices

    simulated_mean = float(np.mean(terminal_prices))
    simulated_se = float(np.std(terminal_prices, ddof=1) / np.sqrt(N_PATHS))
    abs_error = abs(simulated_mean - parameters.spot)

    return {
        "maturity_T": maturity,
        "exact_mean": parameters.spot,
        "simulated_mean": simulated_mean,
        "absolute_error": abs_error,
        "monte_carlo_standard_error": simulated_se,
        "error_in_standard_errors": abs_error / max(simulated_se, 1e-300),
        "clamp_trigger_rate": result.clamp_trigger_rate,
        "n_paths_absorbed": result.n_paths_absorbed,
    }


def _plot_martingale_error(rows: list[dict[str, float]]) -> None:
    """Plot the martingale error against the analytic +/- 2 SE band.

    A correct simulator produces a curve that wanders inside the band
    without any systematic drift in T. The band, being computed from
    the per-maturity Monte Carlo standard error of F_T, is the only
    honest yardstick for what counts as ``small''.
    """
    maturities = [row["maturity_T"] for row in rows]
    errors = [row["absolute_error"] for row in rows]
    se_values = [row["monte_carlo_standard_error"] for row in rows]

    upper_band = [2.0 * se for se in se_values]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(
        maturities, 0.0, upper_band,
        color="tab:gray", alpha=0.20,
        label="+/- 2 SE Monte Carlo band (one-sided)",
    )
    ax.plot(
        maturities, errors,
        marker="o", linestyle="-", linewidth=2, color="tab:blue",
        label="|E[F_T] - F_0| (raw, no EMC)",
    )
    ax.axhline(0, color="black", linestyle=":", linewidth=1, alpha=0.5)

    ax.set_xlabel("Time to Maturity (T)")
    ax.set_ylabel("Forward Price Error |E[F_T] - F_0|")
    ax.set_title("Martingale Preservation: Step-by-Step Exact Simulation")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    head_room = max(max(errors), max(upper_band)) * 1.5
    ax.set_ylim(0.0, head_room if head_room > 0 else 0.01)

    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=200)
    plt.close(fig)


def run_diagnostic_check() -> None:
    """End-to-end sanity print for T = 10.

    Compares the simulated mean to both ``F_0`` and ``F_0 * exp(-rT)``.
    With the simulator returning un-discounted ``F_T``, the two
    should agree exactly when ``r = 0`` and the difference vs ``F_0``
    should be a small multiple of the Monte Carlo standard error.
    """
    print("\n" + "=" * 50)
    print("End-to-End Diagnostic Check for T = 10")
    print("=" * 50)

    parameters = _baseline_parameters()
    T = 10.0
    n_steps = int(T * N_STEPS_PER_YEAR)
    dt = T / n_steps

    pyfeng_stepper = PyFengVarianceStepper(
        parameters=parameters,
        n_path=N_PATHS,
        dt=dt,
        seed=2026,
    )

    result = simulate_heston_cev_terminal(
        parameters=parameters,
        maturity=T,
        n_steps=n_steps,
        n_paths=N_PATHS,
        seed=2026,
        variance_stepper=pyfeng_stepper,
    )
    samples = result.terminal_prices
    sample_mean = float(np.mean(samples))
    sample_se = float(np.std(samples, ddof=1) / np.sqrt(N_PATHS))
    discounted_F0 = parameters.spot * np.exp(-parameters.risk_free_rate * T)

    print(f"1. simulated mean(F_T)             = {sample_mean:.6f}")
    print(f"2. F_0 * exp(-rT) (discounted F_0) = {discounted_F0:.6f}")
    print(f"3. F_0 (raw spot)                  = {parameters.spot:.6f}")
    print("-" * 50)
    print(f"Diff vs F_0                        = {abs(sample_mean - parameters.spot):.6f}")
    print(f"Monte Carlo standard error of mean = {sample_se:.6f}")
    print(f"Diff in units of SE                = {abs(sample_mean - parameters.spot) / sample_se:.3f}")
    print("-" * 50)
    print(f"Clamp trigger rate (per active step) = {result.clamp_trigger_rate:.2e}")
    print(f"Paths absorbed                       = {result.n_paths_absorbed} / {N_PATHS}")
    print(f"Variance backend                     = {result.diagnostics['variance_backend']}")
    print("=" * 50 + "\n")


def main() -> None:
    _ensure_output_dirs()
    print("Running step-by-step CEV-Heston martingale preservation experiment...")

    rows = [_run_experiment_row(T) for T in MATURITIES]
    _write_csv(rows, TABLE_PATH)
    _plot_martingale_error(rows)

    print("Experiment complete.")
    print(f"Saved table:  {TABLE_PATH}")
    print(f"Saved figure: {FIGURE_PATH}")

    run_diagnostic_check()


if __name__ == "__main__":
    main()
