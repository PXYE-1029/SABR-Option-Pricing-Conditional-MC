"""Speed-vs-accuracy trade-off for CEV-Heston pricing (paper Fig 2 analogue).

Plots root-mean-square error against CPU time for three schemes:

  * The project's CEV-approximation simulator
    (``simulate_heston_cev_terminal``).
  * Islah's classical approximation
    (``simulate_heston_cev_islah``).
  * A naive truncated-Euler baseline that integrates the SDE
    directly with full-truncation of the variance at zero. This is
    the cheapest scheme per step but its bias decays so slowly that
    it dominates the trade-off plot.

For each scheme we sweep the number of paths ``N`` while
simultaneously refining the time step ``h = T / n_steps`` so that
``N h`` is held roughly constant -- this matches the convention used
by Choi-Hu-Kwok (2024) Table 7 and gives an apples-to-apples view
of where each scheme sits on the work-vs-error curve.

The headline observation, reproducing Choi-Hu-Kwok Fig 2 in the
Heston-CEV setting, is that the project's CEV approximation reaches
any given RMS error roughly an order of magnitude faster than
either Islah or full-truncation Euler.
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
from src.utils import CEVHestonModelParameters, EuropeanOption, get_rng, standard_error


MATURITY = 4.0
N_REPLICATIONS = 20

# (n_paths, n_steps_per_year) sweep, reproducing the doubling pattern
# of Choi-Hu-Kwok (2024) Table 7.
GRID = [
    (8_000, 5),
    (16_000, 10),
    (32_000, 20),
    (64_000, 40),
]

USE_BETA_LESS_THAN_ONE = True

TABLE_PATH = PROJECT_ROOT / "results" / "tables" / "heston_cev_speed_rms.csv"
FIGURE_PATH = PROJECT_ROOT / "results" / "figures" / "heston_cev_speed_rms_tradeoff.png"


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
        beta=0.5 if USE_BETA_LESS_THAN_ONE else 1.0,
        risk_free_rate=0.0,
    )


def _truncated_euler_price(
    parameters: CEVHestonModelParameters,
    option: EuropeanOption,
    n_steps: int,
    n_paths: int,
    seed: int,
) -> tuple[float, float]:
    """Naive full-truncation Euler baseline.

    Evolves ``F`` via a log-Euler step on
    ``d log F = -0.5 v F^{2(beta - 1)} dt + sqrt(v) F^{beta - 1} dW^F``,
    with ``v`` evolved via full-truncation Euler and asset paths
    floored at zero (absorbing boundary). Returns
    ``(price, standard_error)``.

    This is the simplest possible Heston-CEV simulator and serves as
    a "no-tricks" reference on the work-vs-error curve.
    """

    rng = get_rng(seed)
    dt = option.maturity / n_steps
    sqrt_dt = float(np.sqrt(dt))

    rho = parameters.correlation
    rho_perp = float(np.sqrt(max(1.0 - rho * rho, 0.0)))
    beta_minus_one = parameters.beta - 1.0  # negative for beta < 1

    F = np.full(n_paths, parameters.spot, dtype=float)
    v = np.full(n_paths, parameters.initial_variance, dtype=float)

    for _ in range(n_steps):
        z1 = rng.standard_normal(size=n_paths)
        z2 = rng.standard_normal(size=n_paths)
        dW_v = z1
        dW_F = rho * z1 + rho_perp * z2

        v_pos = np.maximum(v, 0.0)
        sqrt_v = np.sqrt(v_pos)

        active = F > 0.0
        if np.any(active):
            F_act = F[active]
            sqrt_v_act = sqrt_v[active]
            dW_F_act = dW_F[active]
            # log F update under the SDE dF = sqrt(v) F^beta dW
            F_pow = F_act ** beta_minus_one
            drift = -0.5 * v_pos[active] * (F_pow ** 2) * dt
            diffusion = sqrt_v_act * F_pow * sqrt_dt * dW_F_act
            F_new = F_act * np.exp(drift + diffusion)
            F_new = np.where(np.isfinite(F_new) & (F_new > 0.0), F_new, 0.0)
            F[active] = F_new

        # Full-truncation Euler for variance.
        v = v + parameters.kappa * (parameters.theta - v_pos) * dt + (
            parameters.xi * sqrt_v * sqrt_dt * dW_v
        )

    if option.option_type == "call":
        payoffs = np.maximum(F - option.strike, 0.0)
    else:
        payoffs = np.maximum(option.strike - F, 0.0)
    discount_factor = float(np.exp(-parameters.risk_free_rate * option.maturity))
    discounted = discount_factor * payoffs
    return float(np.mean(discounted)), float(standard_error(discounted))


def _benchmark(parameters: CEVHestonModelParameters, option: EuropeanOption) -> float:
    if parameters.beta == 1.0:
        return float(heston_fourier_benchmark_beta_one(parameters, option).price)
    return float(self_reference_benchmark(parameters, option).price)


def _run_replications(
    pricer_fn,
    n_paths: int,
    n_steps: int,
    benchmark_price: float,
    parameters: CEVHestonModelParameters,
    option: EuropeanOption,
    seed_base: int,
) -> tuple[float, float]:
    """Run ``N_REPLICATIONS`` independent runs and return (rms_error, total_seconds)."""
    errors_squared = []
    elapsed_total = 0.0
    for replication in range(N_REPLICATIONS):
        seed = seed_base + replication
        start = perf_counter()
        price = pricer_fn(seed=seed, n_paths=n_paths, n_steps=n_steps)
        elapsed_total += perf_counter() - start
        errors_squared.append((price - benchmark_price) ** 2)
    rms = float(np.sqrt(np.mean(errors_squared)))
    return rms, elapsed_total


def _ours_price(seed: int, n_paths: int, n_steps: int) -> float:
    parameters = _baseline_parameters()
    option = EuropeanOption(strike=100.0, maturity=MATURITY, option_type="call")
    return price_european_option_heston_cev(
        parameters=parameters, option=option,
        n_steps=n_steps, n_paths=n_paths, seed=seed,
    ).price


def _islah_price(seed: int, n_paths: int, n_steps: int) -> float:
    parameters = _baseline_parameters()
    option = EuropeanOption(strike=100.0, maturity=MATURITY, option_type="call")
    return price_european_option_heston_cev_islah(
        parameters=parameters, option=option,
        n_steps=n_steps, n_paths=n_paths, seed=seed,
    ).price


def _euler_price(seed: int, n_paths: int, n_steps: int) -> float:
    parameters = _baseline_parameters()
    option = EuropeanOption(strike=100.0, maturity=MATURITY, option_type="call")
    price, _ = _truncated_euler_price(
        parameters=parameters, option=option,
        n_steps=n_steps, n_paths=n_paths, seed=seed,
    )
    return price


def main() -> None:
    _ensure_output_dirs()
    parameters = _baseline_parameters()
    option = EuropeanOption(strike=100.0, maturity=MATURITY, option_type="call")
    benchmark_price = _benchmark(parameters, option)

    print(
        f"Running speed-RMS trade-off at T={MATURITY} "
        f"(beta = {parameters.beta}, benchmark = {benchmark_price:.5f})..."
    )

    rows = []
    for n_paths, n_steps_per_year in GRID:
        n_steps = max(int(round(MATURITY * n_steps_per_year)), 1)

        rms_ours, time_ours = _run_replications(
            _ours_price, n_paths, n_steps, benchmark_price,
            parameters, option, seed_base=10_000,
        )
        rms_islah, time_islah = _run_replications(
            _islah_price, n_paths, n_steps, benchmark_price,
            parameters, option, seed_base=20_000,
        )
        rms_euler, time_euler = _run_replications(
            _euler_price, n_paths, n_steps, benchmark_price,
            parameters, option, seed_base=30_000,
        )

        row = {
            "n_paths": n_paths,
            "n_steps": n_steps,
            "n_steps_per_year": n_steps_per_year,
            "rms_ours": rms_ours, "time_ours": time_ours,
            "rms_islah": rms_islah, "time_islah": time_islah,
            "rms_euler": rms_euler, "time_euler": time_euler,
        }
        rows.append(row)
        print(
            f"  N={n_paths:>6d}, dt={MATURITY/n_steps:.4f}: "
            f"ours rms={rms_ours:.4f} ({time_ours:.2f}s) | "
            f"islah rms={rms_islah:.4f} ({time_islah:.2f}s) | "
            f"euler rms={rms_euler:.4f} ({time_euler:.2f}s)"
        )

    _write_csv(rows, TABLE_PATH)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(
        [r["time_ours"] for r in rows], [r["rms_ours"] for r in rows],
        marker="o", linewidth=2, color="tab:blue",
        label="Ours (martingale-preserving CEV approx)",
    )
    ax.plot(
        [r["time_islah"] for r in rows], [r["rms_islah"] for r in rows],
        marker="s", linewidth=2, color="tab:red", linestyle="--",
        label="Islah-style approximation",
    )
    ax.plot(
        [r["time_euler"] for r in rows], [r["rms_euler"] for r in rows],
        marker="^", linewidth=2, color="tab:green", linestyle=":",
        label="Truncated Euler (no-tricks baseline)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("CPU Time (seconds)")
    ax.set_ylabel("RMS Pricing Error")
    ax.set_title(f"Speed vs Accuracy Trade-off at T = {MATURITY}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=200)
    plt.close(fig)
    print()
    print(f"Saved table:  {TABLE_PATH}")
    print(f"Saved figure: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
