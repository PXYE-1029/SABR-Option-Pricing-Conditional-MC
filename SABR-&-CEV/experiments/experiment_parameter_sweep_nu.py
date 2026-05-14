"""Parameter sweep over the SABR vol-of-vol parameter nu.

This experiment keeps the beta = 1 European-call setup fixed and varies only
``nu`` to test whether the conditional Monte Carlo variance-reduction advantage
persists beyond the single baseline benchmark used elsewhere in the repository.
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
import numpy as np

# Allow `python experiments/experiment_parameter_sweep_nu.py` from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import EuropeanOption, SABRModelParameters
from src.conditional_mc import price_european_option_conditional_mc
from src.mc_pricer import price_european_option_mc


NU_GRID = [0.0, 0.1, 0.2, 0.4, 0.8]
N_STEPS = 100
N_PATHS = 20_000
TABLE_PATH = PROJECT_ROOT / "results" / "tables" / "parameter_sweep_nu_beta1_call.csv"
SE_FIGURE_PATH = PROJECT_ROOT / "results" / "figures" / "parameter_sweep_nu_standard_errors_beta1_call.png"
RATIO_FIGURE_PATH = PROJECT_ROOT / "results" / "figures" / "parameter_sweep_nu_variance_ratio_beta1_call.png"


def _ensure_output_dirs() -> None:
    """Create output folders if they do not already exist."""

    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SE_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _write_csv(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    """Write rows to a CSV file."""

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _sample_variance(samples: np.ndarray) -> float:
    """Return the sample variance of pathwise estimator values."""

    return float(np.var(samples, ddof=1))


def _parameters_for_nu(vol_of_vol: float) -> SABRModelParameters:
    """Return the common beta = 1 parameter set with a modified nu."""

    return SABRModelParameters(
        spot=100.0,
        initial_volatility=0.2,
        beta=1.0,
        vol_of_vol=vol_of_vol,
        correlation=-0.3,
        risk_free_rate=0.01,
    )


def _run_experiment_row(vol_of_vol: float, seed_offset: int) -> dict[str, float | int | str]:
    """Run both pricing methods for one nu value."""

    parameters = _parameters_for_nu(vol_of_vol)
    option = EuropeanOption(strike=100.0, maturity=1.0, option_type="call")
    seed = 5_000 + seed_offset

    mc_result = price_european_option_mc(
        parameters=parameters,
        option=option,
        n_steps=N_STEPS,
        n_paths=N_PATHS,
        seed=seed,
    )
    cmc_result = price_european_option_conditional_mc(
        parameters=parameters,
        option=option,
        n_steps=N_STEPS,
        n_paths=N_PATHS,
        seed=seed,
        integration_method="trapezoidal",
    )

    mc_variance = _sample_variance(mc_result.discounted_payoffs)
    cmc_variance = _sample_variance(cmc_result.pathwise_conditional_prices)
    variance_reduction_ratio = (
        mc_variance / cmc_variance if cmc_variance > 0.0 else float("nan")
    )
    ratio_note = "undefined because CMC variance is zero" if cmc_variance == 0.0 else ""

    return {
        "nu": vol_of_vol,
        "n_paths": N_PATHS,
        "n_steps": N_STEPS,
        "mc_price": mc_result.price,
        "mc_standard_error": mc_result.standard_error,
        "mc_path_variance": mc_variance,
        "cmc_price": cmc_result.price,
        "cmc_standard_error": cmc_result.standard_error,
        "cmc_path_variance": cmc_variance,
        "price_absolute_difference": abs(mc_result.price - cmc_result.price),
        "variance_reduction_ratio": variance_reduction_ratio,
        "note": ratio_note,
    }


def _plot_standard_errors(rows: list[dict[str, float | int | str]]) -> None:
    """Plot standard error against nu for both methods."""

    nu_values = [float(row["nu"]) for row in rows]
    mc_se = [float(row["mc_standard_error"]) for row in rows]
    cmc_se = [float(row["cmc_standard_error"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(nu_values, mc_se, marker="o", linewidth=2, label="Plain MC")
    ax.plot(nu_values, cmc_se, marker="s", linewidth=2, label="Conditional MC")
    ax.set_xlabel("Vol-of-vol nu")
    ax.set_ylabel("Standard error")
    ax.set_title("Standard Error Across a nu Parameter Sweep")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(SE_FIGURE_PATH, dpi=200)
    plt.close(fig)


def _plot_variance_reduction(rows: list[dict[str, float | int | str]]) -> None:
    """Plot variance reduction ratio for strictly positive nu values."""

    filtered_rows = [row for row in rows if float(row["nu"]) > 0.0]
    nu_values = [float(row["nu"]) for row in filtered_rows]
    ratios = [float(row["variance_reduction_ratio"]) for row in filtered_rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(nu_values, ratios, marker="o", linewidth=2)
    ax.set_xlabel("Vol-of-vol nu")
    ax.set_ylabel("Variance reduction ratio")
    ax.set_title("Plain-MC Variance / Conditional-MC Variance Across nu")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RATIO_FIGURE_PATH, dpi=200)
    plt.close(fig)


def main() -> None:
    """Run the nu parameter sweep and save outputs."""

    _ensure_output_dirs()
    rows = [
        _run_experiment_row(vol_of_vol=nu_value, seed_offset=index)
        for index, nu_value in enumerate(NU_GRID)
    ]
    _write_csv(rows, TABLE_PATH)
    _plot_standard_errors(rows)
    _plot_variance_reduction(rows)

    best_row = rows[-1]
    print("Parameter sweep experiment complete")
    print(f"Saved table: {TABLE_PATH}")
    print(f"Saved figure: {SE_FIGURE_PATH}")
    print(f"Saved figure: {RATIO_FIGURE_PATH}")
    print(
        "Summary at largest nu value: "
        f"nu={float(best_row['nu']):.2f}, "
        f"mc_se={float(best_row['mc_standard_error']):.6f}, "
        f"cmc_se={float(best_row['cmc_standard_error']):.6f}, "
        f"variance_reduction_ratio={float(best_row['variance_reduction_ratio']):.3f}"
    )


if __name__ == "__main__":
    main()
