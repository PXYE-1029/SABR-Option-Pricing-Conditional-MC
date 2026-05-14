"""Timestep sensitivity experiment for plain MC versus conditional MC."""

from __future__ import annotations

import csv
import os
import sys
import tempfile
from pathlib import Path

_CACHE_DIR = Path(tempfile.gettempdir()) / "sabr-option-pricing-cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(_CACHE_DIR / "matplotlib"),
)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow `python experiments/experiment_timestep.py` from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import EuropeanOption, SABRModelParameters
from src.conditional_mc import price_european_option_conditional_mc
from src.mc_pricer import price_european_option_mc


TIMESTEP_GRID = [10, 20, 50, 100, 200]
N_PATHS = 20_000
TABLE_PATH = PROJECT_ROOT / "results" / "tables" / "timestep_sensitivity_beta1_call.csv"
PRICE_FIGURE_PATH = PROJECT_ROOT / "results" / "figures" / "timestep_prices_beta1_call.png"
SE_FIGURE_PATH = PROJECT_ROOT / "results" / "figures" / "timestep_standard_errors_beta1_call.png"


def _ensure_output_dirs() -> None:
    """Create output folders if they do not already exist."""

    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PRICE_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _write_csv(rows: list[dict[str, float | int]], output_path: Path) -> None:
    """Write rows to a CSV file."""

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _baseline_parameters() -> SABRModelParameters:
    """Return the common beta = 1 SABR parameter set used in Phase 5."""

    return SABRModelParameters(
        spot=100.0,
        initial_volatility=0.2,
        beta=1.0,
        vol_of_vol=0.4,
        correlation=-0.3,
        risk_free_rate=0.01,
    )


def _run_experiment_row(n_steps: int) -> dict[str, float | int]:
    """Run all timestep-sensitive pricing variants for one grid size."""

    parameters = _baseline_parameters()
    option = EuropeanOption(strike=100.0, maturity=1.0, option_type="call")
    seed = 2_000 + n_steps

    mc_result = price_european_option_mc(
        parameters=parameters,
        option=option,
        n_steps=n_steps,
        n_paths=N_PATHS,
        seed=seed,
    )
    cmc_trapezoidal = price_european_option_conditional_mc(
        parameters=parameters,
        option=option,
        n_steps=n_steps,
        n_paths=N_PATHS,
        seed=seed,
        integration_method="trapezoidal",
    )

    if n_steps % 2 == 0:
        cmc_simpson = price_european_option_conditional_mc(
            parameters=parameters,
            option=option,
            n_steps=n_steps,
            n_paths=N_PATHS,
            seed=seed,
            integration_method="simpson",
        )
        simpson_price = cmc_simpson.price
        simpson_standard_error = cmc_simpson.standard_error
        trap_simpson_difference = abs(cmc_trapezoidal.price - cmc_simpson.price)
    else:
        simpson_price = float("nan")
        simpson_standard_error = float("nan")
        trap_simpson_difference = float("nan")

    return {
        "n_steps": n_steps,
        "mc_price": mc_result.price,
        "mc_standard_error": mc_result.standard_error,
        "cmc_trapezoidal_price": cmc_trapezoidal.price,
        "cmc_trapezoidal_standard_error": cmc_trapezoidal.standard_error,
        "cmc_simpson_price": simpson_price,
        "cmc_simpson_standard_error": simpson_standard_error,
        "mc_cmc_trapezoidal_abs_diff": abs(mc_result.price - cmc_trapezoidal.price),
        "trap_simpson_abs_diff": trap_simpson_difference,
    }


def _plot_prices(rows: list[dict[str, float | int]]) -> None:
    """Plot price estimates against the timestep grid."""

    n_steps = [int(row["n_steps"]) for row in rows]
    mc_price = [float(row["mc_price"]) for row in rows]
    cmc_trap_price = [float(row["cmc_trapezoidal_price"]) for row in rows]
    cmc_simpson_price = [float(row["cmc_simpson_price"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(n_steps, mc_price, marker="o", linewidth=2, label="Plain MC")
    ax.plot(n_steps, cmc_trap_price, marker="s", linewidth=2, label="Conditional MC (Trap)")
    ax.plot(n_steps, cmc_simpson_price, marker="^", linewidth=2, label="Conditional MC (Simpson)")
    ax.set_xlabel("Number of timesteps")
    ax.set_ylabel("Price")
    ax.set_title("Price Sensitivity to Time Discretization")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PRICE_FIGURE_PATH, dpi=200)
    plt.close(fig)


def _plot_standard_errors(rows: list[dict[str, float | int]]) -> None:
    """Plot standard errors against the timestep grid."""

    n_steps = [int(row["n_steps"]) for row in rows]
    mc_se = [float(row["mc_standard_error"]) for row in rows]
    cmc_trap_se = [float(row["cmc_trapezoidal_standard_error"]) for row in rows]
    cmc_simpson_se = [float(row["cmc_simpson_standard_error"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(n_steps, mc_se, marker="o", linewidth=2, label="Plain MC")
    ax.plot(n_steps, cmc_trap_se, marker="s", linewidth=2, label="Conditional MC (Trap)")
    ax.plot(n_steps, cmc_simpson_se, marker="^", linewidth=2, label="Conditional MC (Simpson)")
    ax.set_xlabel("Number of timesteps")
    ax.set_ylabel("Standard error")
    ax.set_title("Standard Error Sensitivity to Time Discretization")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(SE_FIGURE_PATH, dpi=200)
    plt.close(fig)


def main() -> None:
    """Run the timestep sensitivity experiment and save outputs."""

    _ensure_output_dirs()
    rows = [_run_experiment_row(n_steps) for n_steps in TIMESTEP_GRID]
    _write_csv(rows, TABLE_PATH)
    _plot_prices(rows)
    _plot_standard_errors(rows)

    last_row = rows[-1]
    print("Timestep experiment complete")
    print(f"Saved table: {TABLE_PATH}")
    print(f"Saved figure: {PRICE_FIGURE_PATH}")
    print(f"Saved figure: {SE_FIGURE_PATH}")
    print(
        "Summary at finest timestep grid: "
        f"n_steps={int(last_row['n_steps'])}, "
        f"mc_price={float(last_row['mc_price']):.6f}, "
        f"cmc_trapezoidal_price={float(last_row['cmc_trapezoidal_price']):.6f}, "
        f"cmc_simpson_price={float(last_row['cmc_simpson_price']):.6f}"
    )


if __name__ == "__main__":
    main()
