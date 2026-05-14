"""PyFENG adapter for the CEV-Heston simulator's variance step.

This module wraps a PyFENG Heston Monte Carlo model so it satisfies
the ``VarianceStepper`` protocol of ``heston_cev_simulation``. It is
the recommended production path: PyFENG ships a peer-reviewed
implementation of the Choi-Kwok (2024) Poisson-conditioning scheme for
CIR which we should not re-implement.

The adapter intentionally lives in its own module so that:

  * The main simulator does not import PyFENG at module load time and
    can still be used (with the bundled QE fallback) when PyFENG is
    not installed.
  * Sandbox / CI environments without network access can still run the
    unit tests using the bundled fallback.
  * Any future API drift in PyFENG can be absorbed here without
    touching the main algorithmic code.

Reference: Choi, J. and Kwok, Y. K. (2024). Simulation schemes for
the Heston model with Poisson conditioning. European Journal of
Operational Research 314(1), 363-376.
"""

from __future__ import annotations

import numpy as np

from .utils import CEVHestonModelParameters


__all__ = ["PyFengVarianceStepper", "verify_pyfeng_integrated_variance_scale"]


# Default PyFENG model class to instantiate. We use the Choi-Kwok 2023
# Poisson conditioning scheme by default because that is the variant
# the project's supervisor pointed at; users who prefer a different
# scheme can pass any compatible class via ``model_cls``.
_DEFAULT_PYFENG_MODEL = "HestonMcChoiKwok2023PoisGe"


class PyFengVarianceStepper:
    """Variance stepper backed by a PyFENG Heston MC model.

    Each call to ``step(dt, v_current)`` advances the variance by one
    step and returns the pair ``(v_next, integrated_variance)``, where
    the integrated variance is *dimensional* -- i.e.
    ``int_t^{t+dt} v_s ds`` in units of variance times time. This is
    the contract required by ``simulate_heston_cev_terminal``.

    PyFENG's ``cond_states_step`` returns the *average* variance over
    the step, ``avg_v = (1/dt) * int v ds`` (this is the convention
    used internally by the Choi-Kwok 2024 Heston paper). We therefore
    multiply by ``dt`` here, once, in a single well-documented place,
    so that the rest of the codebase never has to think about which
    convention is in flight. If you suspect that a future PyFENG
    release has changed the scaling, run
    ``verify_pyfeng_integrated_variance_scale`` from this module to
    detect it before you do anything expensive.

    Parameters
    ----------
    parameters:
        Project-side model parameters.
    n_path:
        Number of paths the underlying PyFENG model should be
        configured for. Must equal the ``n_paths`` argument later
        passed to ``simulate_heston_cev_terminal``.
    dt:
        Time-step size. Must equal ``maturity / n_steps`` from the
        outer simulator.
    seed:
        Optional integer seed for ``numpy.random.seed``. PyFENG's
        Monte Carlo classes use the global NumPy RNG, so this is the
        only seeding hook available.
    model_cls:
        Optional alternative PyFENG class name; defaults to the
        Choi-Kwok 2023 Poisson conditioning scheme.
    """

    def __init__(
        self,
        parameters: CEVHestonModelParameters,
        n_path: int,
        dt: float,
        seed: int | None = None,
        model_cls: str = _DEFAULT_PYFENG_MODEL,
    ) -> None:
        try:
            import pyfeng as pf  # noqa: F401  (intentional optional dep)
        except ImportError as err:  # pragma: no cover - import-time guard
            raise ImportError(
                "PyFENG is required for PyFengVarianceStepper. Install it "
                "with `pip install pyfeng`, or fall back to the bundled "
                "Andersen QE stepper in src.cir_simulation by passing "
                "variance_stepper=None to simulate_heston_cev_terminal."
            ) from err

        try:
            ModelClass = getattr(pf, model_cls)
        except AttributeError as err:  # pragma: no cover - version drift guard
            raise AttributeError(
                f"PyFENG does not expose a class named {model_cls!r}. "
                f"Check the installed PyFENG version and adjust the "
                f"`model_cls` argument."
            ) from err

        self._parameters = parameters
        self._dt_expected = float(dt)

        # PyFENG's Heston classes parameterize on sigma = sqrt(initial
        # variance) and vov = xi (the volatility of variance). The
        # mean-reversion speed and long-run variance map directly.
        self._model = ModelClass(
            sigma=float(parameters.initial_variance),
            vov=float(parameters.xi),
            mr=float(parameters.kappa),
            theta=float(parameters.theta),
            rho=float(parameters.correlation),
        )

        if seed is not None:
            np.random.seed(int(seed))
        self._model.set_num_params(n_path=int(n_path), dt=float(dt))

        # Self-identification used by the main simulator's diagnostics.
        self.backend_name = f"pyfeng:{model_cls}"

    def step(
        self, dt: float, v_current: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if abs(dt - self._dt_expected) > 1e-12 * max(1.0, abs(self._dt_expected)):
            # PyFENG's cond_states_step is configured at construction
            # time with a fixed dt; calling it with a different dt
            # silently produces wrong results. We refuse loudly.
            raise ValueError(
                f"PyFengVarianceStepper was constructed with dt={self._dt_expected} "
                f"but step() was called with dt={dt}. PyFENG's Heston classes do "
                f"not support variable dt; rebuild the stepper if you need a "
                f"different step size."
            )

        result = self._model.cond_states_step(dt, v_current)
        v_next, second = _unpack_pyfeng_step_result(result)
        # PyFENG returns the AVERAGE variance over the step. We need
        # the dimensional integral, so multiply by dt once, here.
        integrated_variance = np.asarray(second, dtype=float) * dt
        return np.asarray(v_next, dtype=float), integrated_variance


def _unpack_pyfeng_step_result(result):
    """Robustly unpack the variance and average-variance from PyFENG.

    PyFENG returns either a 2-tuple or a longer tuple depending on
    version and class; the first two entries are always
    ``(v_next, average_variance)``.
    """
    if isinstance(result, tuple):
        if len(result) < 2:
            raise RuntimeError(
                f"PyFENG cond_states_step returned a tuple of length "
                f"{len(result)}; expected at least 2."
            )
        return result[0], result[1]
    raise TypeError(
        f"PyFENG cond_states_step returned {type(result).__name__}; expected tuple."
    )


def verify_pyfeng_integrated_variance_scale(
    parameters: CEVHestonModelParameters,
    n_path: int = 50_000,
    dt: float = 0.1,
    rtol: float = 0.05,
    seed: int = 42,
    model_cls: str = _DEFAULT_PYFENG_MODEL,
) -> dict:
    """Sanity-check the scale of PyFENG's integrated-variance return.

    Runs one variance step starting from ``v_0 = parameters.initial_variance``
    and reports several candidate quantities so you can see at a glance
    whether PyFENG returned the dimensional integral, the average
    variance, the dimensionless ``I_t^h`` of Choi et al. (2024), or
    something else. Specifically, with ``v_0 = 0.04``, ``dt = 0.1``:

      * dimensional integral ``int v ds``  should be near 0.004.
      * average variance ``int v ds / dt`` should be near 0.04.
      * dimensionless ``I_t^h``           should be near 1.0.

    This function returns a diagnostic dictionary; it does not raise
    on a mismatch (different PyFENG versions or model classes can
    legitimately differ). It is meant to be run interactively before
    the first production run on a new PyFENG version.
    """
    stepper = PyFengVarianceStepper(
        parameters=parameters,
        n_path=n_path,
        dt=dt,
        seed=seed,
        model_cls=model_cls,
    )
    v0 = np.full(n_path, parameters.initial_variance, dtype=float)
    v_next, integrated_variance = stepper.step(dt, v0)

    mean_iv = float(np.mean(integrated_variance))
    expected_dimensional = float(parameters.initial_variance * dt)
    relative_difference = abs(mean_iv - expected_dimensional) / max(expected_dimensional, 1e-300)

    return {
        "dt": dt,
        "v0": parameters.initial_variance,
        "expected_dimensional_integral": expected_dimensional,
        "expected_average_variance": parameters.initial_variance,
        "expected_dimensionless_I": 1.0,
        "stepper_returns_dimensional_mean": mean_iv,
        "relative_difference": relative_difference,
        "looks_correctly_dimensional": bool(relative_difference < rtol),
        "v_next_mean": float(np.mean(v_next)),
    }
