"""Validation experiment for the deterministic-volatility Black-Scholes limit.

This experiment checks the ``nu = 0`` consistency case described in the README:
when the volatility of volatility is zero and beta = 1, the model reduces to a
constant-volatility Black-Scholes setting. We therefore compare:

- plain Monte Carlo under the SABR path simulator
- conditional Monte Carlo under the SABR conditional representation
- the exact Black-Scholes call price
"""

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

# Allow `python experiments/experiment_validation_bs_limit.py` from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import EuropeanOption, SABRModelParameters
from src.black_scholes import black_scholes_call_price
from src.conditional_mc import price_european_option_conditional_mc
from src.mc_pricer import price_european_option_mc


PATH_COUNTS = [2_000, 5_000, 10_000, 20_000, 50_000]
N_STEPS = 100
TABLE_PATH = PROJECT_ROOT / "results" / "tables" / "validation_bs_limit_beta1_call.csv"
PRICE_FIGURE_PATH = PROJECT_ROOT / "results" / "figures" / "validation_bs_limit_prices_beta1_call.png"
ERROR_FIGURE_PATH = PROJECT_ROOT / "results" / "figures" / "validation_bs_limit_abs_error_beta1_call.png"


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


def _validation_parameters() -> SABRModelParameters:
    """Return the deterministic-volatility beta = 1 validation setup."""

    return SABRModelParameters(
        spot=100.0,
        initial_volatility=0.2,
        beta=1.0,
        vol_of_vol=0.0,
        correlation=-0.3,
        risk_free_rate=0.01,
    )


def _black_scholes_benchmark(parameters: SABRModelParameters, option: EuropeanOption) -> float:
    """Return the exact Black-Scholes benchmark price for the validation setup."""

    return float(
        black_scholes_call_price(
            spot=parameters.spot,
            strike=option.strike,
            maturity=option.maturity,
            rate=parameters.risk_free_rate,
            volatility=parameters.initial_volatility,
        )
    )


def _run_experiment_row(n_paths: int) -> dict[str, float | int]:
    """Run both pricers for one path count in the Black-Scholes limit."""

    parameters = _validation_parameters()
    option = EuropeanOption(strike=100.0, maturity=1.0, option_type="call")
    bs_price = _black_scholes_benchmark(parameters, option)
    seed = 4_000 + n_paths

    mc_result = price_european_option_mc(
        parameters=parameters,
        option=option,
        n_steps=N_STEPS,
        n_paths=n_paths,
        seed=seed,
    )
    cmc_result = price_european_option_conditional_mc(
        parameters=parameters,
        option=option,
        n_steps=N_STEPS,
        n_paths=n_paths,
        seed=seed,
        integration_method="trapezoidal",
    )

    return {
        "n_paths": n_paths,
        "black_scholes_price": bs_price,
        "mc_price": mc_result.price,
        "mc_standard_error": mc_result.standard_error,
        "mc_abs_error_vs_bs": abs(mc_result.price - bs_price),
        "cmc_price": cmc_result.price,
        "cmc_standard_error": cmc_result.standard_error,
        "cmc_abs_error_vs_bs": abs(cmc_result.price - bs_price),
        "mc_cmc_abs_diff": abs(mc_result.price - cmc_result.price),
    }


def _plot_prices(rows: list[dict[str, float | int]]) -> None:
    """Plot price estimates against path count with the BS benchmark."""

    n_paths = [int(row["n_paths"]) for row in rows]
    bs_price = [float(row["black_scholes_price"]) for row in rows]
    mc_price = [float(row["mc_price"]) for row in rows]
    cmc_price = [float(row["cmc_price"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(n_paths, bs_price, linestyle="--", linewidth=2, label="Exact Black-Scholes")
    ax.plot(n_paths, mc_price, marker="o", linewidth=2, label="Plain MC")
    ax.plot(n_paths, cmc_price, marker="s", linewidth=2, label="Conditional MC")
    ax.set_xscale("log")
    ax.set_xlabel("Number of paths")
    ax.set_ylabel("Price")
    ax.set_title("Validation in the nu = 0 Black-Scholes Limit")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PRICE_FIGURE_PATH, dpi=200)
    plt.close(fig)


def _plot_absolute_errors(rows: list[dict[str, float | int]]) -> None:
    """Plot absolute pricing errors against the exact Black-Scholes benchmark."""

    n_paths = [int(row["n_paths"]) for row in rows]
    mc_error = [float(row["mc_abs_error_vs_bs"]) for row in rows]
    cmc_error = [float(row["cmc_abs_error_vs_bs"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(n_paths, mc_error, marker="o", linewidth=2, label="Plain MC abs error")
    ax.plot(n_paths, cmc_error, marker="s", linewidth=2, label="Conditional MC abs error")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of paths")
    ax.set_ylabel("Absolute error vs exact BS price")
    ax.set_title("Validation Error in the nu = 0 Black-Scholes Limit")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ERROR_FIGURE_PATH, dpi=200)
    plt.close(fig)


def main() -> None:
    """Run the Black-Scholes-limit validation experiment and save outputs."""

    _ensure_output_dirs()
    rows = [_run_experiment_row(n_paths) for n_paths in PATH_COUNTS]
    _write_csv(rows, TABLE_PATH)
    _plot_prices(rows)
    _plot_absolute_errors(rows)

    last_row = rows[-1]
    print("Validation experiment complete")
    print(f"Saved table: {TABLE_PATH}")
    print(f"Saved figure: {PRICE_FIGURE_PATH}")
    print(f"Saved figure: {ERROR_FIGURE_PATH}")
    print(
        "Summary at largest path count: "
        f"n_paths={int(last_row['n_paths'])}, "
        f"bs_price={float(last_row['black_scholes_price']):.6f}, "
        f"mc_price={float(last_row['mc_price']):.6f}, "
        f"cmc_price={float(last_row['cmc_price']):.6f}"
    )


if __name__ == "__main__":
    main()
