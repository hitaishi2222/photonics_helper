# Planned — Solving the NLSE with an unsupervised neural network (Monterola & Saloma 2001)

**Reference.** C. Monterola, C. Saloma, *Opt. Express* **9**, 72 (2001),
doi:10.1364/OE.9.000072. PDF: `OE.9.000072.pdf` (author-supplied); pages under
`pages/` (gitignored).

**Status:** `[~]` implemented, full-budget training running in a background
pane (30k Adam + 3k closure-count-corrected L-BFGS per β–γ case, torch
float64; `--device auto` with the ROCm iGPU guardrails — memory cap 0.6,
event-cache flush, OOM→cpu fallback, per ISSUES.md #7).

## What is implemented

The paper's Eq. (4) NLSE, `−jΨz + (β/2)Ψtt − γ|Ψ|²Ψ = 0`, re-trained as a
modern unsupervised PINN (torch float64):

- 4×42 tanh MLP, two outputs merged into (Ψ_R, Ψ_I);
- trial ansatz `Ψ = α(t) + s(z)·N(z,t)` with `α(t) = exp(−t²/2)` (the
  paper's `Ψ(0,t) = Ψ₀` clause absorbed exactly, i.e. the C1 energy term),
  and `s(z) = 1 − exp(−3z)` (vanishes at z = 0);
- energy = `|F(z,t)|²` (Eq. 7 residual) + `|Ψ|² + |Ψ_t|²` at `t = ±27`
  (paper Eq. 6 / C2 far-field decay); collocation on the paper's domain —
  core band `t ∈ [−3, 3]` + wings `[−27, −10] ∪ [10, 27]`, `z ∈ [0, 5]`;
- NMSE defined exactly as paper Sec. 4.2 (`Σ|Ψref − Ψq|²/Σ|Ψref|²`, 2e4
  test datapoints, eval band `t ∈ [−6, 6]`).

### Engine mapping (asserted)

With `T₀ = 1 ps`, `L = 1 m`, `β₂ = β·1e−24 s²/m`, `γ = 1 W⁻¹ m⁻¹`
(`A_eff = 2πn₂/(λγ)`), the engine's convention
`A_z = −i(β₂/2)A_tt + iγ|A|²A` IS paper Eq. (4). All three (β, gamma)
cases close this cycle before any training:

| Oracle | rel-L2 |
|---|---|
| script split-step vs analytic case A `(1−jz)^{−1/2}e^{−t²/(2(1−jz))}` | 2.6e-7 |
| script split-step vs analytic case B `e^{−t²/2}e^{jz e^{−t²}}` | 4.4e-12 |
| engine vs script (A / B / C) | 1e-11 / 2e-11 / 4.9e-5 |

## Paper's quoted benchmarks (targets)

- E(200) (adaptive-η, 200 iterations): 4.3e-5 (β=1,γ=0) / 8.3e-4
  (β=0,γ=1) / 1.3e-5 (β=1,γ=1).
- NMSE ξ: 7.07e-5 (A) / 2.87e-6 (B) — Sec. 4.2/4.3, PDF p-05/p-06.
- Assert bands here (150× larger optimizer budget): NMSE ≤ 1e-4 (A, B);
  rel-L2 vs the engine-oracle snapshots ≤ 5e-3 (C).

## Progress notes

- The unsupervised optimization is slow to leave its 4e-2 residual
  plateau under plain Adam; the closure-count-corrected L-BFGS refinement
  (ISSUES.md #6's known torch-LBFGS pitfall) is in place.
- The case-C engine oracle is evaluated on the engine's snapshot stack
  (Δz_snap = 8.3 mm; nonlinear phase error ≤ 8e-3 rad), no z interpolation.
- The sech-soliton variant (paper mentions it as the soliton-producing
  input) is NOT one of the paper's own three NLSE cases — documented skip.
