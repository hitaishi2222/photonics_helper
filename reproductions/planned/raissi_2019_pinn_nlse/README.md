# Planned — Physics-informed neural networks (Raissi et al. 2019/2026 reprint) — Schrödinger example, powered by photonics_helper

**Reference.** M. Raissi, P. Perdikaris, G. E. Karniadakis, "Physics-informed
neural networks: A deep learning framework for solving forward and inverse
problems involving nonlinear PDEs", *J. Comput. Phys.* **378**, 686–707 (2019),
doi:10.1016/j.jcp.2018.10.045. PDF: `raissi_arxiv1711.10561.pdf` (arXiv Part I);
pages under `pages/`.

**Status:** `[~]` planning; PDF present. `reproduce.py` not yet written.
**Target module:** this one produces a **new satellite capability** — a small
`photonics_helper.pinns` (or reproduction-local) PINN Trainer on top of the
existing torch-based `[pinns]` extra (see `inverse_design.fit_shg_autodiff`)
and the GNLSE engine as data generator.

## Paper's §I Schrödinger example — exact setup

- **PDE.** `i h_t = −0.5 h_xx − |h|² h` (their Eq. 5: standard continuous-time
  PINN target equation; λ = 1 is implicitly learned in some variants; the
  2.97e-3 relative-L2 case below is *solution-only*), 1-D, x ∈ [−5, 5] periodic.
- **Ground-truth data** — the paper generates it with Chebfun (256 Fourier
  modes, RK4, Δt = π/2·10⁻⁶), initial `h(0, x) = 2 sech(x)`, up to `t = π/2`.
- **Training data:** `N0 = 50` points of `h(0, x)` sampled at `t = 0`
  (Latin Hypercube of the initial line), plus `Nb = 50` boundary collocation
  points {t(s), x}, plus `Nf = 20 000` interior collocation points.
- **Network:** 5 hidden layers × 100 neurons, tanh activation; joint
  representation `h(t, x) = [u + i v]` with the residual
  `r = −h_t − 0.5 h_xx − |h|² h` (or its complex-PDE-safe equivalent).
- **Validation:** relative L2 error target in the paper: **1.97e−3**.

## What to reproduce — photonics-helper flavored

1. **The paper's verification (Fig. 2):** reproduce the PINN solving the
   periodic Schrödinger equation to relative L2 ≈ 2e−3 (comparable, given the
   stochastic training; accept ≤ 5e−3 with random seeding and ≤ 10⁻² worst-case).
2. **Data generation from our own engine:** the twist that validates this
   library — generate `h(t, x)` NOT from Chebfun but with
   `photonics_helper.gnlse.SplitStepEngine` (specifically an NLSE-only run with
   matching 1-D GVD/nl-scaled coefficients). Exact test:
   the SplitStepEngine output against the paper's Eq. 5 at machine precision up
   to RK/close-form benchmark (i.e. `test_h()` in the reproduction), before
   PINN training starts.
3. **(Stretch goal 2) λ (nonlinearity coefficient) discovery:** add a learnable
   λ in the network's residual and recover λ = 1 to ~1 % from the same data.
4. **Stretch goal 3 — fiber-physics version:** repeat the same PINN with data
   from the actual GNLSE (Raman + self-steepening) — showing that a PINN can
   learn a GNLSE model surrogate on a slice of a supercontinuum evolution.

## Implementation notes

- The network and loss are small; do it fully in `torch` 64-bit (same
  convention as `inverse_design.fit_shg_autodiff`).
- Destruction of alternates: enforce the periodic boundary condition by
  periodic padding of u/v on a periodic domain (x ∈ [−5, 5], Fourier basis or
  modulo-bc loss at both ends).
- Relative L2 evaluation points: same as paper's t = 0.59, 0.79, 0.98 profiles.
- With the paper's architecture, train on GPU/ASI torch for ~20–30 k epochs; use
  Adaptive L-BFGS via `torch.optim.LBFGS` at the end (paper used Adam+LBFGS);
  budget CPU fallback runtime in the README.

## Execution checklist

- [ ] `parameters.json`: every number above (N0, Nb, Nf, depth/width, act fun).
- [ ] Engine-data splitter: our GNLSE StepEngine produces a 256-mode spectral
      solution of the same equations fragment — hard-coded agreement ≤1e-10 to
      an analytic reference (the sech soliton sector).
- [ ] `reproduce.py` — PINN training: torch-only, 64-bit double, deterministic
      seeds; ends with (i) top-panel comparison figure `pinns/pinns.png`,
      (ii) the relative L2 metric vs the paper's 1.97e−3 (target ≤ 5e−3),
      (iii) cut profiles at t = 0.59, 0.79, 0.98.
- [ ] λ-discovery variant behind a flag (default off, budget permitting).
- [ ] README: results vs paper, compute cost, deviations.
