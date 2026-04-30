"""Compare Euler and PyFENG variance backends for the Beta-Heston prototype."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.beta_heston_simulation import (
    BetaHestonParameters,
    is_pyfeng_available,
    price_beta_heston_european_call_conditional_approximation,
)
from src.utils import EuropeanOption, time_call


BETA_VALUES = [1.0, 0.7, 0.5]
VARIANCE_BACKENDS = ["euler", "pyfeng_choi_kwok_td"]
N_PATHS = 10_000
N_STEPS = 100
TABLE_PATH = (
    PROJECT_ROOT / "results" / "tables" / "beta_heston_backend_comparison.csv"
)
FIGURE_PATH = (
    PROJECT_ROOT / "results" / "figures" / "beta_heston_backend_comparison.png"
)


def _make_parameters(beta: float) -> BetaHestonParameters:
    """Return the shared Beta-Heston parameter set."""

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


def _write_csv(rows: list[dict[str, float | str]], output_path: Path) -> None:
    """Write the comparison rows to a CSV file."""

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, float | str]], output_path: Path) -> None:
    """Save a compact two-panel backend comparison figure."""

    betas = sorted({float(row["beta"]) for row in rows})
    backend_labels = {
        "euler": "Euler variance backend",
        "pyfeng_choi_kwok_td": "PyFENG POIS-TD variance backend",
    }
    style_map = {
        "euler": ("o", 2.0),
        "pyfeng_choi_kwok_td": ("s", 2.0),
    }

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.2), sharex=True)
    for backend in VARIANCE_BACKENDS:
        backend_rows = [row for row in rows if row["backend"] == backend]
        backend_rows.sort(key=lambda row: float(row["beta"]))
        marker, linewidth = style_map[backend]
        axes[0].plot(
            [float(row["beta"]) for row in backend_rows],
            [float(row["price"]) for row in backend_rows],
            marker=marker,
            linewidth=linewidth,
            label=backend_labels[backend],
        )
        axes[1].plot(
            [float(row["beta"]) for row in backend_rows],
            [float(row["runtime_seconds"]) for row in backend_rows],
            marker=marker,
            linewidth=linewidth,
            label=backend_labels[backend],
        )

    axes[0].set_ylabel("Call price")
    axes[0].set_title("Beta-Heston Backend Comparison")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Beta")
    axes[1].set_ylabel("Runtime (s)")
    axes[1].set_xticks(betas)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _run_one_backend(
    beta: float,
    variance_backend: str,
    n_steps: int,
    n_paths: int,
) -> dict[str, float | str]:
    """Run one conditional-approximation pricing configuration."""

    parameters = _make_parameters(beta)
    option = EuropeanOption(strike=100.0, maturity=1.0, option_type="call")
    seed = 21_000 + int(round(beta * 100))

    result, runtime = time_call(
        price_beta_heston_european_call_conditional_approximation,
        parameters=parameters,
        option=option,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
        variance_backend=variance_backend,
    )

    return {
        "beta": beta,
        "backend": variance_backend,
        "price": result.price,
        "standard_error": result.standard_error,
        "runtime_seconds": runtime,
        "average_integrated_variance": float(
            result.simulation.integrated_variance.mean()
        ),
    }


def run_experiment(
    beta_values: list[float] | None = None,
    n_steps: int = N_STEPS,
    n_paths: int = N_PATHS,
    table_path: Path = TABLE_PATH,
    figure_path: Path = FIGURE_PATH,
) -> list[dict[str, float | str]]:
    """Run and save the Euler-vs-PyFENG backend comparison."""

    if not is_pyfeng_available():
        raise ImportError(
            "The backend comparison experiment requires PyFENG. Install it with "
            'python3 -m pip install -e ".[pyfeng]" and rerun the script.'
        )

    betas = beta_values if beta_values is not None else list(BETA_VALUES)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str]] = []
    for beta in betas:
        euler_row = _run_one_backend(
            beta=beta,
            variance_backend="euler",
            n_steps=n_steps,
            n_paths=n_paths,
        )
        pyfeng_row = _run_one_backend(
            beta=beta,
            variance_backend="pyfeng_choi_kwok_td",
            n_steps=n_steps,
            n_paths=n_paths,
        )

        euler_row["price_difference_from_euler_backend"] = 0.0
        euler_row["integrated_variance_difference_from_euler_backend"] = 0.0
        pyfeng_row["price_difference_from_euler_backend"] = abs(
            float(pyfeng_row["price"]) - float(euler_row["price"])
        )
        pyfeng_row["integrated_variance_difference_from_euler_backend"] = abs(
            float(pyfeng_row["average_integrated_variance"])
            - float(euler_row["average_integrated_variance"])
        )
        rows.extend([euler_row, pyfeng_row])

    _write_csv(rows, table_path)
    _plot(rows, figure_path)
    return rows


def main() -> None:
    """Run the saved-output backend comparison experiment."""

    rows = run_experiment()
    print("Beta-Heston backend comparison complete")
    print(f"Saved table: {TABLE_PATH}")
    print(f"Saved figure: {FIGURE_PATH}")
    print("This experiment changes only the Heston variance/integrated-variance backend.")
    print("PyFENG is used only for the Heston variance/integrated-variance layer.")
    print("The beta-power asset layer remains a project-specific prototype.")
    print(
        "The Euler and PyFENG prices therefore do not have to match exactly, "
        "and the current results should be treated as an experimental comparison."
    )
    for beta in BETA_VALUES:
        euler_row = next(
            row for row in rows if row["beta"] == beta and row["backend"] == "euler"
        )
        pyfeng_row = next(
            row
            for row in rows
            if row["beta"] == beta and row["backend"] == "pyfeng_choi_kwok_td"
        )
        print(
            f"beta={beta:.1f}: "
            f"euler_price={float(euler_row['price']):.6f}, "
            f"pyfeng_price={float(pyfeng_row['price']):.6f}, "
            f"euler_se={float(euler_row['standard_error']):.6f}, "
            f"pyfeng_se={float(pyfeng_row['standard_error']):.6f}, "
            f"euler_avg_int_var={float(euler_row['average_integrated_variance']):.6f}, "
            f"pyfeng_avg_int_var={float(pyfeng_row['average_integrated_variance']):.6f}, "
            f"abs_price_diff={float(pyfeng_row['price_difference_from_euler_backend']):.6f}, "
            "abs_int_var_diff="
            f"{float(pyfeng_row['integrated_variance_difference_from_euler_backend']):.6f}"
        )


if __name__ == "__main__":
    main()
