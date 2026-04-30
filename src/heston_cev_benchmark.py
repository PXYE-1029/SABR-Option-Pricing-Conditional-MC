"""European option price benchmarks for CEV-Heston validation.

The CEV-Heston model with ``0 < beta < 1`` does not admit a tractable
closed-form European option price, so any quantitative validation of
a simulator needs an independent reference. This module provides two
benchmarks:

  * For ``beta = 1`` (the standard Heston special case) we wrap
    PyFENG's Fourier-based ``HestonFft`` pricer. This is essentially
    closed-form to several decimal places and runs in milliseconds;
    it is the gold standard for the ``beta = 1`` row of any
    validation table.

  * For ``0 < beta < 1`` we provide a "self-reference" benchmark:
    the simulator from ``heston_cev_simulation`` is run with a very
    large path count and a very fine time grid, and the resulting
    price is treated as the reference for cheaper runs at coarser
    settings. This is the standard way to benchmark a stochastic
    simulator against itself in the absence of an independent
    closed form (Choi et al. 2024 use the FDM for the same purpose).

Each benchmark returns a ``BenchmarkPrice`` containing both the
point estimate and a confidence interval; for the closed-form
PyFENG path the CI is degenerate (zero width).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import CEVHestonModelParameters, EuropeanOption


__all__ = [
    "BenchmarkPrice",
    "heston_fourier_benchmark_beta_one",
    "self_reference_benchmark",
]


@dataclass(frozen=True)
class BenchmarkPrice:
    """Container for a benchmark European option price.

    Attributes
    ----------
    price:
        The benchmark price.
    standard_error:
        Standard error of the benchmark. Zero for closed-form
        Fourier benchmarks; non-zero (but typically very small) for
        self-reference Monte Carlo benchmarks.
    method:
        Short string identifying the benchmark, suitable for
        labelling table rows and chart legends.
    """

    price: float
    standard_error: float
    method: str


def heston_fourier_benchmark_beta_one(
    parameters: CEVHestonModelParameters,
    option: EuropeanOption,
) -> BenchmarkPrice:
    """Closed-form European option price for ``beta = 1`` Heston via PyFENG.

    PyFENG's ``HestonFft`` implements the standard COS / Fourier
    pricer for Heston European calls and puts; we wrap it here to
    match the project's parameter conventions. The returned price
    has effectively zero numerical error at the precision relevant
    for validating Monte Carlo runs.
    """

    if parameters.beta != 1.0:
        raise ValueError(
            "heston_fourier_benchmark_beta_one is only valid for beta = 1; "
            f"got beta = {parameters.beta}. Use self_reference_benchmark for "
            "beta < 1."
        )

    try:
        import pyfeng as pf  # noqa: F401
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "PyFENG is required for the Heston Fourier benchmark. "
            "Install it with `pip install pyfeng`."
        ) from err

    fft_model = pf.HestonFft(
        sigma=float(np.sqrt(parameters.initial_variance)),
        vov=float(parameters.xi),
        mr=float(parameters.kappa),
        theta=float(parameters.theta),
        rho=float(parameters.correlation),
        intr=float(parameters.risk_free_rate),
    )
    is_call = option.option_type == "call"
    price = float(
        fft_model.price(
            strike=float(option.strike),
            spot=float(parameters.spot),
            texp=float(option.maturity),
            cp=1 if is_call else -1,
        )
    )
    return BenchmarkPrice(price=price, standard_error=0.0, method="pyfeng_heston_fft")


def self_reference_benchmark(
    parameters: CEVHestonModelParameters,
    option: EuropeanOption,
    n_paths_reference: int = 500_000,
    n_steps_per_year_reference: int = 50,
    seed: int = 999_999,
) -> BenchmarkPrice:
    """High-resolution self-reference benchmark for ``0 < beta <= 1``.

    Runs the project's own CEV-Heston simulator with a deliberately
    expensive setting (default 500k paths, 50 steps per year) and
    returns the resulting price. The Monte Carlo standard error is
    reported as ``standard_error`` so that downstream consumers can
    factor it into accuracy tolerances. The seed is fixed by default
    so that the benchmark is reproducible from a single run.

    For ``beta = 1``, prefer ``heston_fourier_benchmark_beta_one``;
    this function is intended for the ``0 < beta < 1`` regime where
    no closed form is available.
    """

    # Imported lazily to avoid a hard cycle: heston_cev_simulation
    # imports from utils, and this module is a peer of it under src/.
    from .heston_cev_simulation import price_european_option_heston_cev

    n_steps = max(int(round(option.maturity * n_steps_per_year_reference)), 1)
    pricing = price_european_option_heston_cev(
        parameters=parameters,
        option=option,
        n_steps=n_steps,
        n_paths=n_paths_reference,
        seed=seed,
    )
    return BenchmarkPrice(
        price=pricing.price,
        standard_error=pricing.standard_error,
        method=f"self_ref_n{n_paths_reference}_dt{1.0/n_steps_per_year_reference:.4f}",
    )
