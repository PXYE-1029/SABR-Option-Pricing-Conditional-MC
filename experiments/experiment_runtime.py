"""Runtime benchmark for plain MC versus conditional MC."""

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

# Allow `python experiments/experiment_runtime.py` from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import EuropeanOption, SABRModelParameters
from src.conditional_mc import price_european_option_conditional_mc
from src.mc_pricer import price_european_option_mc
from src.utils import time_call


PATH_COUNTS = [2_000, 5_000, 10_000, 20_000, 50_000]
N_STEPS = 100
TABLE_PATH = PROJECT_ROOT / "results" / "tables" / "runtime_benchmark_beta1_call.csv"
RUNTIME_FIGURE_PATH = PROJECT_ROOT / "results" / "figures" / "runtime_scaling_beta1_call.png"
EFFICIENCY_FIGURE_PATH = PROJECT_ROOT / "results" / "figures" / "runtime_efficiency_beta1_call.png"


def _ensure_output_dirs() -> None:
    """Create output folders if they do not already exist."""

    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUNTIME_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)


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
    """Run the runtime benchmark for one path count."""

    parameters = _baseline_parameters()
    option = EuropeanOption(strike=100.0, maturity=1.0, option_type="call")
    seed = 3_000 + n_paths

    mc_result, mc_runtime = time_call(
        price_european_option_mc,
        parameters=parameters,
        option=option,
        n_steps=N_STEPS,
        n_paths=n_paths,
        seed=seed,
    )
    cmc_result, cmc_runtime = time_call(
        price_european_option_conditional_mc,
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
        "mc_runtime_seconds": mc_runtime,
        "mc_price": mc_result.price,
        "mc_standard_error": mc_result.standard_error,
        "mc_path_variance": mc_variance,
        "mc_variance_time_product": mc_variance * mc_runtime,
        "cmc_runtime_seconds": cmc_runtime,
        "cmc_price": cmc_result.price,
        "cmc_standard_error": cmc_result.standard_error,
        "cmc_path_variance": cmc_variance,
        "cmc_variance_time_product": cmc_variance * cmc_runtime,
        "price_absolute_difference": abs(mc_result.price - cmc_result.price),
    }


def _plot_runtime(rows: list[dict[str, float | int]]) -> None:
    """Plot runtime scaling against path count."""

    n_paths = [int(row["n_paths"]) for row in rows]
    mc_runtime = [float(row["mc_runtime_seconds"]) for row in rows]
    cmc_runtime = [float(row["cmc_runtime_seconds"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(n_paths, mc_runtime, marker="o", linewidth=2, label="Plain MC")
    ax.plot(n_paths, cmc_runtime, marker="s", linewidth=2, label="Conditional MC")
    ax.set_xscale("log")
    ax.set_xlabel("Number of paths")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Runtime Scaling")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RUNTIME_FIGURE_PATH, dpi=200)
    plt.close(fig)


def _plot_efficiency(rows: list[dict[str, float | int]]) -> None:
    """Plot variance-times-runtime as a simple efficiency metric."""

    n_paths = [int(row["n_paths"]) for row in rows]
    mc_efficiency = [float(row["mc_variance_time_product"]) for row in rows]
    cmc_efficiency = [float(row["cmc_variance_time_product"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(n_paths, mc_efficiency, marker="o", linewidth=2, label="Plain MC")
    ax.plot(n_paths, cmc_efficiency, marker="s", linewidth=2, label="Conditional MC")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of paths")
    ax.set_ylabel("Variance × runtime")
    ax.set_title("Efficiency Benchmark")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(EFFICIENCY_FIGURE_PATH, dpi=200)
    plt.close(fig)


def main() -> None:
    """Run the runtime benchmark and save outputs."""

    _ensure_output_dirs()
    rows = [_run_experiment_row(n_paths) for n_paths in PATH_COUNTS]
    _write_csv(rows, TABLE_PATH)
    _plot_runtime(rows)
    _plot_efficiency(rows)

    last_row = rows[-1]
    print("Runtime benchmark complete")
    print(f"Saved table: {TABLE_PATH}")
    print(f"Saved figure: {RUNTIME_FIGURE_PATH}")
    print(f"Saved figure: {EFFICIENCY_FIGURE_PATH}")
    print(
        "Summary at largest path count: "
        f"n_paths={int(last_row['n_paths'])}, "
        f"mc_runtime={float(last_row['mc_runtime_seconds']):.6f}s, "
        f"cmc_runtime={float(last_row['cmc_runtime_seconds']):.6f}s, "
        f"mc_efficiency={float(last_row['mc_variance_time_product']):.6e}, "
        f"cmc_efficiency={float(last_row['cmc_variance_time_product']):.6e}"
    )


if __name__ == "__main__":
    main()
