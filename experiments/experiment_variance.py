"""Variance comparison experiment for plain MC versus conditional MC."""

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
import numpy as np

# Allow `python experiments/experiment_variance.py` from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import EuropeanOption, SABRModelParameters
from src.conditional_mc import price_european_option_conditional_mc
from src.mc_pricer import price_european_option_mc


PATH_COUNTS = [2_000, 5_000, 10_000, 20_000, 50_000]
N_STEPS = 100
TABLE_PATH = PROJECT_ROOT / "results" / "tables" / "variance_comparison_beta1_call.csv"
SE_FIGURE_PATH = PROJECT_ROOT / "results" / "figures" / "variance_standard_errors_beta1_call.png"
RATIO_FIGURE_PATH = PROJECT_ROOT / "results" / "figures" / "variance_reduction_ratio_beta1_call.png"


def _ensure_output_dirs() -> None:
    """Create output folders if they do not already exist."""

    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SE_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _write_csv(rows: list[dict[str, float | int]], output_path: Path) -> None:
    """Write rows to a CSV file."""

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _sample_variance(samples: np.ndarray) -> float:
    """Return the sample variance of pathwise estimator values."""

    return float(np.var(samples, ddof=1))


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


def _run_experiment_row(n_paths: int) -> dict[str, float | int]:
    """Run both pricing methods for one path count."""

    parameters = _baseline_parameters()
    option = EuropeanOption(strike=100.0, maturity=1.0, option_type="call")
    seed = 1_000 + n_paths

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

    mc_variance = _sample_variance(mc_result.discounted_payoffs)
    cmc_variance = _sample_variance(cmc_result.pathwise_conditional_prices)

    return {
        "n_paths": n_paths,
        "mc_price": mc_result.price,
        "mc_standard_error": mc_result.standard_error,
        "mc_path_variance": mc_variance,
        "cmc_price": cmc_result.price,
        "cmc_standard_error": cmc_result.standard_error,
        "cmc_path_variance": cmc_variance,
        "price_absolute_difference": abs(mc_result.price - cmc_result.price),
        "variance_reduction_ratio": mc_variance / cmc_variance,
    }


def _plot_standard_errors(rows: list[dict[str, float | int]]) -> None:
    """Plot standard error against path count for both methods."""

    n_paths = [int(row["n_paths"]) for row in rows]
    mc_se = [float(row["mc_standard_error"]) for row in rows]
    cmc_se = [float(row["cmc_standard_error"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(n_paths, mc_se, marker="o", linewidth=2, label="Plain MC")
    ax.plot(n_paths, cmc_se, marker="s", linewidth=2, label="Conditional MC")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of paths")
    ax.set_ylabel("Standard error")
    ax.set_title("Standard Error vs Path Count")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(SE_FIGURE_PATH, dpi=200)
    plt.close(fig)


def _plot_variance_reduction(rows: list[dict[str, float | int]]) -> None:
    """Plot the variance reduction ratio against path count."""

    n_paths = [int(row["n_paths"]) for row in rows]
    ratios = [float(row["variance_reduction_ratio"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(n_paths, ratios, marker="o", linewidth=2, color="tab:green")
    ax.set_xscale("log")
    ax.set_xlabel("Number of paths")
    ax.set_ylabel("Variance reduction ratio")
    ax.set_title("Plain-MC Variance / Conditional-MC Variance")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RATIO_FIGURE_PATH, dpi=200)
    plt.close(fig)


def main() -> None:
    """Run the variance comparison experiment and save outputs."""

    _ensure_output_dirs()
    rows = [_run_experiment_row(n_paths) for n_paths in PATH_COUNTS]
    _write_csv(rows, TABLE_PATH)
    _plot_standard_errors(rows)
    _plot_variance_reduction(rows)

    last_row = rows[-1]
    print("Variance experiment complete")
    print(f"Saved table: {TABLE_PATH}")
    print(f"Saved figure: {SE_FIGURE_PATH}")
    print(f"Saved figure: {RATIO_FIGURE_PATH}")
    print(
        "Summary at largest path count: "
        f"n_paths={int(last_row['n_paths'])}, "
        f"mc_se={float(last_row['mc_standard_error']):.6f}, "
        f"cmc_se={float(last_row['cmc_standard_error']):.6f}, "
        f"variance_reduction_ratio={float(last_row['variance_reduction_ratio']):.3f}"
    )


if __name__ == "__main__":
    main()
