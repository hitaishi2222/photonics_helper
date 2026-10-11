# `formal/` — the specification in Lean 4

This directory holds the **mathematical specification** that the Python
library in `photonics_helper/` implements.

```
   Lean 4 + Mathlib          the specification
        │
        ▼
   photonics_helper          the implementation, which follows the spec
        │
        ▼
   findings → literature     where they disagree, the literature decides
```

Roadmap: [`PLAN.md`](PLAN.md). Theorem ↔ Python register:
[`CONFORMANCE.md`](CONFORMANCE.md) — written concurrently with each stage.
Findings and corrections: [`CHANGELOG.md`](CHANGELOG.md).

## Status

| Stage | Content | Status |
|---|---|---|
| 0 | Toolchain, Mathlib pinned | **done** — builds green |
| A | Conventions, constants, dimensional units | **done** |
| B | Grid + Fourier convention | **done** — incl. `dft_pair`, `conj_dft`, `hermitian` |
| C | Dispersion and soliton relations | **done** — 19 theorems |
| D | Shock validity, Raman causality | **done** — Raman gain bound, the shock sign convention, and the operator-symbol exactness (`code_symbol_is_exact`) proved |
| E | Thin films (DBR), SHG | **done** — `R + T = 1` (`DBREnergy.lean`), admittance branches and Snell/obliquity (E5, E6) proved |
| F | FWM and modulation instability | **done** — MI gain (band, peak, height), extended-Δ form, DW sign, sinc² efficiency, phase-truncation remainder (F19) proved |
| G | Structured light | **done** — measure-theoretic overlap (G10) deliberately out of scope |
| H | RK4 integrators | **done** — interaction picture, operator ordering, and the analytic `O(h⁵)` bound (`exp_taylor_abs_error_le`) proved |
| I | Strang splitting, Kerr sub-step | **done** |
| J | Dispersion operator, split-step conservation | **done** |

240 theorems proved. **No `sorry` remains anywhere in `formal/`.**
Build: `LAKE_JOBS=2 LEAN_NUM_THREADS=2 lake build`.

**2026-10-27, side-quest §17.5 closure:** six theorems added to
`Nonlinear/Shock.lean` — the solver's shock symbol `1 + Ωτ_shock` is exactly
`ω/ω₀` bin-by-bin (`code_symbol_is_exact`), ending the "first-order truncation"
reading, and the guard's provenance is restated as Blow & Wood's `< ω₀/3`
derivation regime (`window_of_guard`, `bw_regime_needs_sharper`). The
reciprocal-ratio theorems are kept, relabelled as counterfactuals. 246 theorems
in total.

**2026-10, side-quest §17.6 Stage S3:** ETDRK4 shipped — `scheme="etdrk4"` in
`gnlse.py`. Cox & Matthews Eqs. (26)–(29) on the diagonal GNLSE operator.
Two theorems in `Photonics/Solvers/Etdrk4.lean` (the half-argument φ₁ carry
`Qcoef_mul_id` and the `N₀` φ-combination `f1coef_phi_combo`). 247 theorems.
*Gate measured* — `etdrk4` observes 3.97 on all three §17.6 decks (finest
pair 3.99; `yoshida4` 4.02/4.14/4.15 there; table in `PLAN.md` §17.6); on
the Raman+shock deck the (A5) photon number is conserved to 2.2e-14. The
same session also repaired `Nonlinear/Shock.lean`, which did not compile in
the working tree (duplicated block comment; `accuracy_implies_existence`
was false as stated — non-negativity now a hypothesis, conclusion `x < 1`),
and added the missing `GNLSESolver.photon_number_vs_z` passthrough.

### Open statements

**None.** Every theorem in `formal/` is proved, and `lake build` is green. The
open statements E5/E6 (admittance branches and the Snell/obliquity derivation
with the `np.clip` guard), H6's analytic bound, and F19 (the phase-truncation
remainder behind `_taylor_phase_max_error`) were closed on 2026-10-26.

The three statements that were open when this file was last written are now
closed. `conj_dft` and `hermitian` (Stage B) turned out not to need new ideas
but a repair of the draft proofs — three Mathlib-revision mismatches and one
genuinely ill-typed statement, all recorded in `CHANGELOG.md` (2026-10-24).
`ramanDrive_le_max` (Stage D) closed with a termwise bound that avoids the
`Fin`-reindexing the previous session had been attempting.

**Closed since:** `conj_dft`, `hermitian` (Stage B) and `ramanDrive_le_max`
(Stage D) are proved, and the **shock sign convention** is now a theorem
(`Nonlinear.shock_factor_neg`, `convention_difference`, `shock_skews_blue`) rather
than a numerical argument — see `CHANGELOG.md` and `CONFORMANCE.md` S17–S21.

**Closed on 2026-10-26** (18 new theorems, one example — see `CHANGELOG.md`):

- **Stage E's remaining admittance branch** — `_admittance`'s TE/TM forms (E5:
  the branches agree at normal incidence, `η_TE·η_TM = n²`, both nonnegative
  with the principal root) and the Snell/obliquity derivation (E6: the Snell
  invariant, the `np.clip` guard characterised exactly as the identity on the
  admissible set, and the phase thickness `δ = 2π n d cos θ_l/λ ≥ 0`).
- **H6's analytic bound** — `exp_taylor_abs_error_le`: for every `z ≠ 0`,
  `|exp z − Σ_{k≤4} zᵏ/k!| ≤ exp|z|·|z|⁵/5!`, proved from
  `taylor_mean_remainder_lagrange_iteratedDeriv`; no hypothesis carries the
  expansion any more.
- **F19** — `phaseRemainder_lagrange` / `phaseRemainder_bound` in
  `Dispersion/Beta.lean`: the statement that `_taylor_phase_max_error`
  estimates numerically.

**Still open, and deliberately not claimed:**

- **The magnitude of the β₂/β₃ fast estimate.** Lean pins the cubic-truncation
  root at `−3β₂/β₃`; the fallback uses `−2β₂/β₃`. Same sign — and the
  blue-shift invariant is proved for both — but the `3/2` magnitude is routed
  to the soliton-RDW source (F22).
- **The shock guard constant vs. Blow & Wood's bandwidth assumption.**
  `_SHOCK_TAYLOR_LIMIT = 0.2` bounds the simulated window by `2ω₀/5`, slightly
  *outside* the `ω₀/3` bandwidth under which Blow & Wood derive the GNLS;
  strict adherence needs `x < 1/6`. The constant is unchanged (changing it
  would newly warn on grids the decks already use) and the difference is
  recorded in `CONFORMANCE.md` F11 as a judgement call for the library owner.
  **The `−14 % shock energy drift` is resolved** (2026-10-27): it was the
  unweighted energy changing while the conserved quantity — Blow & Wood's
  App. Eq. (A5) photon number `Σ|Â(ω)|²/(ω₀+ω)dω` — stayed constant to
  ≤5×10⁻⁵ relative. See `CHANGELOG.md` and the new
  `photon_number_vs_z` accessor in `gnlse.py`.
- **Stage G's measure-theoretic cell (G10)** — the LG modal overlap and
  normalisation, which the plan scoped out (§8.5).
- **Solver coverage** in `gnlse.py` beyond its shock terms, `raman_transfer.py`,
  `structured.py`, `pulse.py`, `breathers.py`, `noise.py`.

No remaining open *statement*. See `PLAN.md` §7–§9 and §15.

## Building

```bash
cd formal
LAKE_JOBS=2 LEAN_NUM_THREADS=2 lake build
```

The toolchain is pinned by `lean-toolchain` and the Mathlib revision by
`lake-manifest.json`. Both are committed, so a build here is reproducible.
`.lake/build` is not committed.

**Note on machine resources.** This box reports 24 cores but has ~15 GB RAM
with ~9 GB free. Lean spawns one worker thread per core *per process*, so an
unbounded `lake build` exhausts memory and fails with
`resource exhausted (error code: 12)`.

First-time setup also needs the Mathlib olean cache (~5 GB):

```bash
LAKE_JOBS=2 LEAN_NUM_THREADS=2 lake exe cache get
```

**Do not run `lake update` casually.** It rewrites `lean-toolchain` to match
whatever Mathlib revision it resolves — moving to `master` silently replaced the
pinned `v4.34.1` with `v4.35.0-rc4` and invalidated the cache. The pin in
`lakefile.toml` (`git#v4.34.1`) exists precisely to prevent that.

## What this does not verify

Stated plainly, because a formal development that overclaims is worse than no
formal development:

- **Floating-point accuracy.** Rounding, FFT-backend parity, and error
  estimates are not verified here.
- **The Python control flow.** A theorem about an operator does not check that
  the shipped code composes its sub-operators in that order.
- **Experimental parameters** of the reproduction decks, only the closed-form
  laws they compare against.
- **The material database.** Provenance and licensing are data-governance
  concerns.
- **Measure-theory statements** — notably the modal-overlap integrals in
  `structured.py` (`CONFORMANCE.md` G10).
- **DBR energy conservation (`R + T = 1`)** for absorbing stacks — the
  identity is proved (`DBREnergy.lean`, F21) **only** with the lossless shape
  hypothesis; the absorbing-layer counterexample in `CONFORMANCE.md` F13
  stands.
- **Conformance is machine-checked on the Lean side only.** The register is a
  reviewed artefact, not a compiler guarantee.

## Conventions of the development

- Every statement is one you would defend in a paper. If a statement cannot be
  made cleanly, that is reported as a finding rather than forced.
- Where a physical law holds only in a regime (Taylor validity of the shock
  term, RMS effects in a linear theory, normalisation over a bounded domain),
  the regime is a hypothesis in the theorem, not a caveat in a docstring.
- Physical constants are opaque reals with proved relations. No numeric literals.
- Grids are finite (`Fin n → ℂ`), matching what the code actually computes.
- **A `sorry`-free build is not the same as a complete development.** Zero
  `sorry` means nothing is silently unproved; it says nothing about which
  theorems *exist*. The status table above is the coverage claim, and it is
  maintained by hand.
- Edit Lean files with the editor tools, never through shell one-liners: two
  sessions have been lost to unicode corruption (`ℤ` → `Ζ`) and over-eager
  `sed` ranges.