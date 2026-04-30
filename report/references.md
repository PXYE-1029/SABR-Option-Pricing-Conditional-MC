# Project References

The references below are organized to mirror the structure of the codebase: SABR-side Phase 1 work first, CEV-Heston Phase 2 extension second, and shared numerical foundations third. Each entry is followed by a one-line note on how it is used in this repository.

## Phase 1: SABR Model and Conditional Monte Carlo

1. **Hagan, P. S., Kumar, D., Lesniewski, A. S., and Woodward, D. E. (2002).** *Managing Smile Risk*. Wilmott Magazine, September 2002, 84-108.
   - Original SABR specification. The asymptotic implied-volatility formula derived here is the de facto industry standard for SABR option pricing and sets up the smile-modelling motivation that runs through the entire repository.

2. **Black, F., and Scholes, M. (1973).** *The Pricing of Options and Corporate Liabilities*. Journal of Political Economy, 81(3), 637-654.
   - Used both as the closed-form benchmark in the deterministic-volatility validation case (`vol_of_vol = 0`) and as the per-path conditional pricer inside the SABR conditional Monte Carlo estimator.

## Phase 2: CEV-Heston Extension

3. **Choi, J., Hu, L., and Kwok, Y. K. (2024).** *Efficient and accurate simulation of the SABR model*. arXiv:2408.01898v2.
   - The central methodological reference for Phase 2. Provides the martingale-preserving CEV approximation of the conditional terminal price (their Section 4 and Eq. 16), the exact shifted-Poisson-mixture-Gamma sampler for the CEV distribution (their Section 4.2 and Algorithm 3), and the comparative analysis against Islah's classical baseline (their Section 4.3 and Figure 3). The Heston-CEV simulator in this repository is a direct adaptation of this framework, with the SABR lognormal volatility replaced by Heston's CIR variance.

4. **Choi, J., and Kwok, Y. K. (2024).** *Simulation schemes for the Heston model with Poisson conditioning*. European Journal of Operational Research, 314(1), 363-376.
   - Provides the Poisson-conditioning scheme for the CIR variance step. PyFENG's `HestonMcChoiKwok2023PoisGe` is the published reference implementation; the project's `pyfeng_adapter.py` wraps this class to satisfy the CEV-Heston simulator's variance-stepper protocol, fulfilling the brief's "use existing code in PyFENG" requirement.

5. **Andersen, L. (2008).** *Simple and efficient simulation of the Heston stochastic volatility model*. Journal of Computational Finance, 11, 1-42.
   - The Quadratic-Exponential (QE) scheme implemented in `cir_simulation.py`. Used as an offline fallback for the variance step when PyFENG is not installed, and as the primary path when `xi = 0` (PyFENG's class divides by `xi^2` and cannot be instantiated in that limit).

6. **Islah, O. (2009).** *Solving SABR in Exact Form and Unifying it with LIBOR Market Model*. SSRN Electronic Journal.
   - The classical conditional-distribution approximation that almost every SABR / CEV-Heston simulator since 2009 has adopted, and the baseline against which we benchmark. Implemented in `islah_approximation.py` for direct head-to-head comparison; its known martingale-violation pathology is the motivating contrast for our scheme's headline result.

7. **Makarov, R. N., and Glew, D. (2010).** *Exact simulation of Bessel diffusions*. Monte Carlo Methods and Applications, 16, 283-306.
   - Provides the shifted-Poisson-mixture-Gamma representation of the CEV distribution that underlies the exact CEV sampler in `cev_sampling.py` (Step 3 of the simulator).

8. **Kang, C. (2014).** *Simulation of the shifted Poisson distribution with an application to the CEV model*. Management Science and Financial Engineering, 20, 27-32.
   - The companion algorithm that replaces the rejection sampler for the shifted-Poisson auxiliary variable with a single Gamma draw, optimised for the CEV-simulation use case (their "Algorithm 3"). Used in our exact CEV sampler.

9. **Cox, J. C. (1996).** *The Constant Elasticity of Variance Option Pricing Model*. Journal of Portfolio Management, 23, 15-17.
   - Introduces the CEV model and its connection to the noncentral chi-squared distribution. Used implicitly throughout Phase 2 wherever CEV distributional facts (absorption probability, conditional density) appear.

## Shared Numerical Foundations

10. **Glasserman, P. (2003).** *Monte Carlo Methods in Financial Engineering*. Springer.
    - The standard reference for Monte Carlo pricing in finance. Used throughout the project for variance-reduction theory, standard-error estimation, and the convergence rate / efficiency framing of the runtime experiments.

11. **Boyle, P. P. (1977).** *Options: A Monte Carlo Approach*. Journal of Financial Economics, 4(3), 323-338.
    - The earliest application of Monte Carlo methods to option pricing; cited for historical context in the report's introduction.

12. **Course materials for MATH 5030 (Numerical Methods).**
    - Lecture notes on time-discretization schemes, integration rules (trapezoidal, Simpson), and stochastic-differential-equation simulation. The repository's implementation choices, timestep experiments, and report framing follow the course conventions.

## How These References Connect to This Project

- The SABR model specification, the smile-modelling motivation, and the Phase 1 conditional Monte Carlo decomposition come from **Hagan et al. (2002)** and standard textbook material (**Glasserman 2003**), with **Black-Scholes (1973)** providing the per-path conditional pricer.
- The Phase 2 simulator is a direct adaptation of **Choi, Hu, and Kwok (2024)** to the CEV-Heston model. The variance-side Step 1 delegates to **Choi-Kwok (2024)** via PyFENG, the asset-side Step 2 reuses their operator-splitting and frozen-coefficient argument with the SABR Ito integral replaced by the closed-form CIR Ito integral, and the asset-side Step 3 invokes the exact CEV sampler of **Makarov-Glew (2010)** combined with **Kang (2014)**.
- The **Andersen (2008)** QE scheme is the offline / `xi = 0` fallback for the variance step.
- The **Islah (2009)** approximation is the comparison baseline used to demonstrate the martingale-preservation advantage of our scheme.
- The benchmarking and estimator-comparison framework follows standard Monte Carlo and variance-reduction ideas from **Glasserman (2003)**, with **Boyle (1977)** for historical grounding.
- The report framing, the numerical-methods toolkit, and the timestep / integration experiments align with the **MATH 5030** course materials.
