# Xing 2022 — intermodal dispersive waves in a multimode chalcogenide fibre

**Reference:** D.-K. Xing, J.-Y. He, P. Wang, H.-Y. Guo, K. Chang, C.-C. Liu,
Y.-G. Liu, Z. Wang, "Intermodal dispersive waves and soliton collision during
multimode supercontinuum generation in chalcogenide glass fiber", *Laser Phys.*
**32**, 115401 (2022) · [10.1088/1555-6611/ac979a](https://doi.org/10.1088/1555-6611/ac979a).

**What this paper is.** A step-index **As₂Se₃-core / As₂S₃-clad** multimode
chalcogenide fibre (10 µm core, 246 µm cladding), pumped at the **HE₁₁
zero-dispersion wavelength (6.104 µm)** with 200 fs / 1 kW pulses over 50 mm.
It solves the MM-GN LSE (its Eq. 1) with Raman + Kerr coupling tensors and the
full non-degenerate FWM sum, and reports an **intermodal dispersive wave at
10.8 µm** (TE₀₁ ↔ HE₂₁ᵦ) from the phase-matching condition of its **Eq. (7)**.

**Why a reproduction.** In the originating project this is the *closest
conceptual competitor* to a "per-mode-dispersion multimode chalcogenide SC"
claim. A reproduction is only useful if it is honest about what it can and
cannot recover from the paper's *printed* inputs — which is what this deck does.

**Deck:** `parameters.json` (all inputs, quoted with their paper locations) ·
`xing2022.py` (Sellmeier + scalar-LP + exact-vector mode solvers + Eq. 7) ·
`reproduce.py` (`validate()`, ~30 s) → `xing_2022_dispersion.png`.

```bash
python reproductions/xing_2022_multimode_chg/reproduce.py
```

## Verdict — material ✅, dispersion ❌

| layer | paper | this deck | reproducible? |
|---|---|---|---|
| As₂S₃ index (Eq. 5) | 5-term Sellmeier | matches the repository's validated As₂S₃ row | ✅ |
| As₂Se₃ index (Eq. 6) | 3-term Sellmeier | **as printed it is broken** — the middle `B` lost its square (`19`, a pole at √19 = 4.359 µm); restored to `19² = 361` it matches the repository's As₂Se₃ row exactly (n(4 µm) = 2.783, material ZDW 7.22 µm) | ✅ (with the typo fixed) |
| HE₁₁ ZDW | **6.104 µm** | exact-vector solver → **≈4.85 µm** (scalar LP ≈4.96 µm) | ❌ (≈1.1 µm short) |
| intermodal DW | **10.8 µm** (TE₀₁↔HE₂₁ᵦ) | **no pair** has an Eq. (7) root in the 8–13 µm red window, so the 10.8 µm DW does not exist; the paper's pair instead phase-matches a **blue** DW at ≈1.94 µm (HE₁₁-soliton partners ≈2.24 µm); the repository's own `phase_matching.dispersive_wave_roots` agrees on the blue root and the absent red one | ❌ |
| higher-order modes (TE₀₁/TM₀₁/HE₂₁) normal at the pump | yes | **no** — anomalous, with ZDWs ≈3.8–3.9 µm (= the scalar-LP LP₁₁ ZDW, so it is the guide, not the solver) | ❌ |

So the paper's **material model is fully reproducible** (after fixing Eq. 6),
but **none of its dispersion claims** survive: the HE₁₁ ZDW is ~1.3 µm short, and
the higher-order modes are anomalous at the pump rather than normal — i.e. the
**staggered-ZDW picture** the multimode SC scenario rests on is not recoverable.
The residual is plausibly the idealised analytic-mode treatment versus the
paper's COMSOL full-vector FEM on a high-Δ guide, but the practical consequence
stands: the *numbers* are not independently checkable as printed.

## The Eq. (6) typo

```
printed:  n² = 1 + 4.9948719 λ²/(λ² − 0.0584) + 0.1207152 λ²/(λ² − 19)      + 1.7123685 λ²/(λ² − 0.2336)
correct:  n² = 1 + 4.9948719 λ²/(λ² − 0.0584) + 0.1207152 λ²/(λ² − 19² = 361) + 1.7123685 λ²/(λ² − 0.2336)
```

As printed the fit has a pole at λ = √19 = 4.359 µm and a wrong material ZDW;
the Thompson-2008 As₂Se₃ fit it is copied from carries `B = 361`.

## Method

* **Scalar LP modes** are obtained from the sibling deck's `lp_roots`
  (`reproductions/poletti_2008_multimode/overlap.py`).
* **Exact vector modes** (HE₁₁, TE₀₁, TM₀₁, HE₂₁) come from the closed-form
  eigenvalue equations for a circular step-index fibre (Okamoto, *Fundamentals
  of Optical Waveguides*, Eqs. 3.97-3.99), seeded from the scalar LP roots and
  refined with `brentq`, with the Bessel-zero poles masked. The vector solver is
  needed because the scalar model cannot split TE₀₁/TM₀₁/HE₂₁.
* **β₂** is a 5-point ω-stencil of β(ω); **Δβ** in Eq. (7) uses the same
  β₁ from a 5-point ω-stencil of the pump mode.
* **Upstream API gap (found here).** `photonics_helper.phase_matching.
  dispersive_wave_roots` solves the **intramodal** Cherenkov condition: it reads
  β_sol, β(ω) and β₁(ω_s) all from a single `beta_fn`. An **intermodal** DW needs
  β(ω) from the radiated partner (HE₂₁) but β_sol and β₁ from the soliton (HE₁₁),
  which that API cannot express — fed the partner it returns the soliton
  wavelength (6.104 µm) as a **spurious root** (`β_sol := β_partner(ω_s)` makes
  `f(ω_s) = 0` identically). The intermodal Eq. (7) scan is therefore done from
  scratch, and re-checked against a **different derivative scheme** (2-point vs
  5-point β₁) to confirm the root location is scheme-independent.

## Scope / caveats (recorded, not hidden)

* **No 6-mode MM-GN LSE propagation.** The paper's Fig.-level supercontinuum and
  soliton collisions are not re-simulated: that needs a 6-mode deck with vector
  overlap tensors and a 50 mm / 1 kW run. This deck reproduces the **dispersion
  layer** that the paper's headline numbers rest on, which is where the
  discrepancy is. A propagation run could not rescue a phase-matching condition
  that has no root.
* **`f_R` material mismatch.** The paper's Raman parameters (f_R = 0.148,
  τ₁ = 23 fs, τ₂ = 164.5 fs) are classic **As₂S₃** values while the core is
  **As₂Se₃**. Treated here as "as published" and flagged.
* **No PDF archived** (not open access; paywall/copyright). Inputs are quoted in
  `parameters.json` from the IOP full text.

## Provenance

Ported from the originating research project's Gate-0.3 working scripts
(`prototypes/xing2022_zdw.py`, `scripts/xing2022_vector_modes.py`) and write-up
(`logs/phase0_xing2022_repro.md`, `docs/xing_2022_params.md`) with no change to
the physics; this folder is the self-contained, repo-native version.
