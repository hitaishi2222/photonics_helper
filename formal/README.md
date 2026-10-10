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
| B | Grid + Fourier convention | **done** — including `dft_pair` |
| C | Dispersion and soliton relations | **done** — 19 theorems |
| D | Shock validity, Raman causality | **partly done** |
| E | Thin films (DBR), SHG | **partly done** |
| F | FWM and modulation instability | **started** |
| G | Structured light | **done** |
| H | RK4 integrators | **partly done** |
| I | Strang splitting, Kerr sub-step | **done** |
| J | Dispersion operator, split-step conservation | **done** |

156 theorems proved. Build: `LAKE_JOBS=2 LEAN_NUM_THREADS=2 lake build`.

### Open statements

Three theorems remain open, all listed here. None is hidden.

| Theorem | Location | What it still needs |
|---|---|---|
| `conj_dft` | `Conventions.lean` | the origin-bin identity `(ω_0 + ω_0)·t_j = 2π·(n/2 − j)`, verified numerically but not proved. The reindexing it rests on (`sum_neg_eq`, `neg_eq_sub`, `omega_reflect`) is proved. |
| `hermitian` | `Conventions.lean` | the same; it is `conj_dft` restricted to real input, not a separate theorem. |
| `ramanDrive_le_max` | `Nonlinear/Raman.lean` | the gain bound `Ω ≤ 1`; the `Fin`-indexed bookkeeping was not finished. |

`Constants.lean`, `Grid/Basic.lean`, `Dispersion/Beta.lean`, `Optics/DBR.lean`,
`Optics/Chi2.lean`, `Optics/FWM.lean`, `Optics/Structured.lean`,
`Solvers/Strang.lean` and `Solvers/Dispersion.lean` have **no** `sorry`.

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
- Edit Lean files with the editor tools, never through shell one-liners: two
  sessions have been lost to unicode corruption (`ℤ` → `Ζ`) and over-eager
  `sed` ranges.