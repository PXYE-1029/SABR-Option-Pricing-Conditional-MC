"""ξ=0.001 / β=1 sanity check — does ours match PyFENG FFT in the BS limit?

This deliberately does NOT pass variance_stepper, so the simulator builds
its own internally and shares its RNG with the stepper, avoiding the
same-seed-on-two-Generators trap.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import CEVHestonModelParameters, EuropeanOption
from src.heston_cev_simulation import price_european_option_heston_cev
from src.heston_cev_benchmark import heston_fourier_benchmark_beta_one


params = CEVHestonModelParameters(
    spot=1.0,
    initial_variance=0.09,
    kappa=1.5,
    theta=0.09,
    xi=0.001,
    correlation=-0.8,
    beta=1.0,            # β=1 → closed form Fourier benchmark
)
opt = EuropeanOption(strike=1.0, maturity=4.0, option_type="call")

bench = heston_fourier_benchmark_beta_one(params, opt)
ours = price_european_option_heston_cev(
    parameters=params,
    option=opt,
    n_steps=4,
    n_paths=200_000,
    seed=42,
    # no variance_stepper
)

print(f"benchmark    = {bench.price:.6f}")
print(f"ours         = {ours.price:.6f} +/- {ours.standard_error:.6f}")
print(f"bias in SE   = {(ours.price - bench.price) / ours.standard_error:.2f}")
print(f"backend used = {ours.simulation.diagnostics['variance_backend']}")
