"""Prototype experiment for the professor-suggested Beta-Heston extension."""

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

# Allow `python experiments/experiment_beta_heston_prototype.py` from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.beta_heston_simulation import (
    BetaHestonParameters,
    beta_heston_conditional_backend,
    price_beta_heston_european_call_conditional_approximation,
    price_beta_heston_european_call_euler,
)
from src.utils import EuropeanOption, time_call


BETA_VALUES = [1.0, 0.7, 0.5]
N_PATHS = 10_000
N_STEPS = 100
TABLE_PATH = PROJECT_ROOT / "results" / "tables" / "beta_heston_prototype_comparison.csv"
FIGURE_PATH = (
    PROJECT_ROOT / "results" / "figures" / "beta_heston_prototype_comparison.png"
)


def _write_csv(rows: list[dict[str, float]], output_path: Path) -> None:
    """Write rows to a CSV file."""

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _make_parameters(beta: float) -> BetaHestonParameters:
    """Return the common Beta-Heston parameter set for this prototype."""

    return BetaHestonParameters(
        spot=100.0,
        v0=0.04,
        kappa=1.5,
        theta=0.04,
        xi=0.4,
        beta=beta,
        rho=-0.6,
        risk_free_rate=0.01,
    )


def _run_row(beta: float, n_steps: int, n_paths: int) -> dict[str, float]:
    """Run both prototype methods for one beta value."""

    parameters = _make_parameters(beta)
    option = EuropeanOption(strike=100.0, maturity=1.0, option_type="call")
    seed = 11_000 + int(round(beta * 100))

    euler_result, euler_runtime = time_call(
        price_beta_heston_european_call_euler,
        parameters=parameters,
        option=option,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
    )
    conditional_result, conditional_runtime = time_call(
        price_beta_heston_european_call_conditional_approximation,
        parameters=parameters,
        option=option,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
    )

    return {
        "beta": beta,
        "euler_price": euler_result.price,
        "euler_standard_error": euler_result.standard_error,
        "euler_runtime_seconds": euler_runtime,
        "conditional_approximation_price": conditional_result.price,
        "conditional_approximation_standard_error": conditional_result.standard_error,
        "conditional_approximation_runtime_seconds": conditional_runtime,
        "price_absolute_difference": abs(euler_result.price - conditional_result.price),
        "standard_error_ratio": (
            euler_result.standard_error / conditional_result.standard_error
            if conditional_result.standard_error > 0.0
            else float("inf")
        ),
    }


def _plot(rows: list[dict[str, float]], output_path: Path) -> None:
    """Save one simple two-panel prototype comparison figure."""

    betas = [row["beta"] for row in rows]
    euler_prices = [row["euler_price"] for row in rows]
    conditional_prices = [row["conditional_approximation_price"] for row in rows]
    euler_se = [row["euler_standard_error"] for row in rows]
    conditional_se = [
        row["conditional_approximation_standard_error"] for row in rows
    ]

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.0), sharex=True)

    axes[0].plot(betas, euler_prices, marker="o", linewidth=2, label="Euler asset")
    axes[0].plot(
        betas,
        conditional_prices,
        marker="s",
        linewidth=2,
        label="Conditional approximation",
    )
    axes[0].set_ylabel("Call price")
    axes[0].set_title("Beta-Heston Prototype Comparison")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        betas,
        euler_se,
        marker="o",
        linewidth=2,
        label="Euler asset",
    )
    axes[1].plot(
        betas,
        conditional_se,
        marker="s",
        linewidth=2,
        label="Conditional approximation",
    )
    axes[1].set_xlabel("Beta")
    axes[1].set_ylabel("Standard error")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_experiment(
    beta_values: list[float] | None = None,
    n_steps: int = N_STEPS,
    n_paths: int = N_PATHS,
    table_path: Path = TABLE_PATH,
    figure_path: Path = FIGURE_PATH,
) -> list[dict[str, float]]:
    """Run the Beta-Heston prototype experiment and save outputs."""

    betas = beta_values if beta_values is not None else list(BETA_VALUES)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [_run_row(beta, n_steps=n_steps, n_paths=n_paths) for beta in betas]
    _write_csv(rows, table_path)
    _plot(rows, figure_path)
    return rows


def main() -> None:
    """Run the saved-output Beta-Heston prototype experiment."""

    rows = run_experiment()
    if any(beta < 1.0 for beta in BETA_VALUES):
        print(
            "Warning: beta < 1 currently uses a prototype conditional CEV "
            f"approximation backend ({beta_heston_conditional_backend()})."
        )
    print("Beta-Heston prototype experiment complete")
    print(f"Saved table: {TABLE_PATH}")
    print(f"Saved figure: {FIGURE_PATH}")
    for row in rows:
        print(
            f"beta={row['beta']:.1f}: "
            f"euler_price={row['euler_price']:.6f}, "
            f"conditional_price={row['conditional_approximation_price']:.6f}, "
            f"euler_se={row['euler_standard_error']:.6f}, "
            f"conditional_se={row['conditional_approximation_standard_error']:.6f}"
        )


if __name__ == "__main__":
    main()
