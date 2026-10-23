# GAP: What blocks chalcogenide multimode SCG in `photonics_helper`

**Status:** CLOSED 2026-10-16, all four gaps bridged (step7 `mode_analysis`, step8 multimode Raman, step9 shock/TPA/FCA, step10 full-β + full-tensor FWM; each N=1 == scalar engine, selection rules preserved). Was: open · **Created:** 2026-10-16 · **Owner:** Hitaishi V
**Companion to:** `PLAN.md` (multimode SCG design) · **Repo:** `~/Hiti/projects/photonics_helper` (own project, upstream PRs are just normal work)

This document covers the two engineering gaps from the design review (findings #2 and #3),
plus two adjacent gaps that will bite the moment we try to produce an octave-spanning spectrum:

| Gap | Short name | Blocks |
|---|---|---|
| **A** | No mode-solver → multimode-engine bridge | per-mode β/group-delay/overlap/γ for a *real* geometry |
| **B** | No multimode **Raman** | long-wavelength extension (ChG `f_R` ≈ 0.5!) |
| **C** | No multimode **self-steepening / TPA / free carriers** | blue edge, shock, coherence |
| **D** | Taylor β only; FWM restricted to the degenerate `n==p` slice | octave-spanning faithfulness; intermodal pairs |

Each gap below states: *what exists*, *what is missing*, *the physics*, and *the papers with
verified DOIs* we build from. A consolidated bibliography is in §5.

> **Not in scope here:** re-implementing a Maxwell mode solver. We use **tidy3d** (local mode
> solver) and **femwell** (FEM) for eigenmodes, exactly as instructed; everything below is the
> *glue* between those exports and the propagation engine.

---

## 1. Gap A: mode-solver → multimode-engine bridge

### A.1 What exists today

| Piece | Location | Limit |
|---|---|---|
| femwell/tidy3d solve → `n_eff(λ)`, `n_g`, fields | `examples/generate_waveguide_mode_data.py` | hard-wired `num_modes=1`, single λ-sweep, one geometry |
| `WaveguideMode` (import a single mode) | `photonics_helper/fiber.py` | one mode: `neff`, `ng`, `wl`; `from_csv/from_npz`, `neff_at`, `to_propagation_constant()`, `to_dispersion()` |
| `Dispersion.get_betas(polyOrder, λ)` | `photonics_helper/fiber.py` | Taylor coefficients for **one** β(ω) |
| `ZDependentDispersion` | `photonics_helper/fiber.py` | one β(ω, z), scalar taper |
| LP overlap tensors `Q⁽¹⁾, Q⁽²⁾`, `engine_weights()` | `reproductions/poletti_2008_multimode/overlap.py` | **step-index fibre only**, analytic LP basis, not arbitrary (rib/ridge) fields |
| `MultimodeSplitStepEngine` | `photonics_helper/multimode_gnlse.py` | needs `betas` (list), `group_delays`, `xpm_weights`, `fwm_weights`, one shared scalar γ, all supplied by hand |
| Overlap / A_eff / per-mode γ from fields | n/a | **does not exist anywhere** |

So `examples/38` and every reproduction *hard-code* the tensors and one γ. There is no path
from "a mode solve of a 1.5 × 0.8 µm GeAsSe rib" to a runnable engine.

### A.2 What is missing (precise)

1. **N-mode export + continuation.** Solve ≥3 modes over a wide λ grid (1.2–6 µm), export
   `n_eff(λ)`, `n_g(λ)` and the transverse vector fields per mode, and **label modes
   consistently across λ by field overlap** (eigen-index ordering is not stable, modes swap).
2. **Per-mode dispersion → engine arguments.** From `β_m(ω)` build
   - Taylor `betas` per mode (or full β(ω): Gap D), and
   - `group_delays[m] = β₁⁽ᵐ⁾ − β₁⁽⁰⁾` (s/m): the engine rejects `group_delays[0] ≠ 0`;
   - `phase_offsets[m]` from `β₀⁽ᵐ⁾` (only meaningful for GRIN-like index profiles).
3. **Overlap tensors from arbitrary vector fields.** With normalized transverse profiles
   `F_m(x,y)` (so `∬|F_m|² = 1`, units W^-1/2):
   ```
   A_eff,m   = (∬|F_m|²)² / ∬|F_m|⁴
   Γ_m       = ∬_core |F_m|²           (confinement; multiple-cladding aware)
   γ_m       = n₂ ω₀ Γ_m / (c A_eff,m)
   S⁽¹⁾_mn   = ∬ |F_m|² |F_n|²                    → xpm_weights (N×N), SPM = S⁽¹⁾_mm
   S⁽²⁾_mnpq = ∬ F_m* F_n F_p F_q*                → fwm_weights (N×N×N×N)
   ```
   These are exactly Poletti–Horak Eq. (7) `Q⁽¹⁾/Q⁽²⁾`; in the LP basis `Q⁽¹⁾ = Q⁽²⁾`
   (real modes), which we already reproduce to 0 in the fibre case.
   The integrator must **not** assume circular symmetry, grid quadrature over the tidy3d/femwell
   field arrays, with the azimuthal index recovered numerically for the `oam_l` gate.
4. **Per-mode γ routing.** The engine takes **one** γ (`_fiber_gamma`). For high-index-contrast
   rib modes `γ_m` differ by ≳2×, so we must either (i) push mode-resolved `γ_mn` in absolute
   units through the existing `coupling_tensor` (N×N complex, W⁻¹m⁻¹) / `fwm_coupling_tensor`
   slots, or (ii) add a `gamma` per mode to the engine. (i) works today; (ii) is cleaner.
5. **Cross-validation.** femwell vs tidy3d per-mode `n_eff` (<2 %, the bar already set in
   `tests/test_waveguide_import.py`); tensors vs `reproductions/poletti_2008_multimode`
   (≲1e-3); reduction tests (1 mode → scalar engine, 2 modes → `vector_gnlse`) at machine precision.

### A.3 Proposed shape (new module, ~your repo's conventions)

`photonics_helper/mode_analysis.py`
```python
load_modes(export_dir)                      -> list[WaveguideMode]           # multi-mode
mode_dispersion(modes, order, lambda0)      -> DispersionSet(betas, group_delays, phase_offsets)
overlap_tensors_from_fields(modes, grid, n, lambda0)
    -> ModePairParameters(gamma, A_eff, xpm_weights, fwm_weights, oam_l, coupling_tensor)
build_multimode_deck(geom, pump)            -> kwargs for MultimodeSplitStepEngine
```
+ `tests/test_mode_analysis.py`, `docs/api/mode-analysis.md`, and a *shape-agnostic* extraction of the
Poletti `overlap.py` integrals (keep the fibre-specific solver where it is, in `reproductions/`).

### A.4 References to build from (DOI verified)

- Poletti & Horak, *JOSA B* **25**, 1645 (2008): **the multimode GNLSE + Eq. (7) overlaps
  `Q⁽¹⁾/Q⁽²⁾`**; already the spec of our engine. `10.1364/JOSAB.25.001645`
- Wright, Ziegler, Lushnikov, Zhu, Eftekhar, Christodoulides & Wise, *IEEE JSTQE* **24**,
  5100516 (2018), **GMMNLSE: the full nonlinear mode-coupling coefficient tensors `S⁽¹⁾/S⁽²⁾`
  from mode overlaps, plus the reference GPU solver**. This is the blueprint for A.2(3).
  `10.1109/JSTQE.2017.2779749` · open solver: `github.com/WiseLabAEP/GMMNLSE-Solver-FINAL`
- Afshar V., Monro et al., *Opt. Express* **17**, 2298 (2009): **full-vectorial waveguide
  nonlinear coefficients for high-index-contrast (subwavelength) guides**, the correct
  SPM/XPM/FWM overlaps when the scalar LP treatment fails (our rib case). `10.1364/OE.17.002298`
- Mumtaz, Essiambre & Agrawal, *JLT* **31**, 398 (2013): our engine's Manakov/overlap
  weight spec (SPM 8/9, XPM 4/3). `10.1109/JLT.2012.2235414`

---

## 2. Gap B: no multimode Raman

### B.1 What exists today

- Scalar GNLSE has full Raman: `RamanSpec.from_database(...)`, `RamanResponse` (Agrawal
  single-Lorentzian, with `τ₁ = 1/(2πν_R)`, `τ₂ = 1/(πΔν)`), and the scalar engine convolves
  `h_R(t)` correctly (`gordon_1986_ssfs` validates SSFS).
- `MultimodeSplitStepEngine` has **no Raman code path at all** (`__init__` has no
  `include_raman`/`raman_response`). The only hook is the documented
  `coupling_tensor` slot.
- `docs/multimode-gnlse.md` nonetheless claims *"Raman is the scalar per-channel response"*;
  **docs ≠ code**. (Fix whichever way we implement.) `examples/18` sets
  `fiber.raman_response` and the *scalar* solver uses it; the multimode engine ignores it.
- ChG data available (single-Lorentzian): `As2S3` (345 cm⁻¹, f_R 0.15), `As2Se3` (310, 0.55),
  `GeAsSe` (250, 0.50). No multi-peak ChG phonon sets yet.

### B.2 Why this is not optional for our device

`f_R ≈ 0.5` for GeAsSe/As₂Se₃, half the nonlinear response is delayed. Raman sets the
soliton self-frequency shift (`raman_shift_rate`), the red edge, and the coherence. Also, in a
few-mode guide **intermodal Raman** transfers power between modes with *different group
velocities*, a documented mechanism for near-100 % mode power transfer and strong
fundamental-mode depletion. Ignoring it both over-predicts bandwidth and mis-predicts the modal
energy budget.

### B.3 Physics to implement

Two layers, cheap → correct:

1. **Per-channel (intramodal) Raman**: minimal, matches the current doc claim:
   ```
   ∂A_m/∂z ⊃ i γ (1−f_R)|A_m|²A_m + i γ f_R A_m ∫ h_R(t−t') |A_m(t')|² dt'
   ```
   Reduces exactly to the scalar engine at N=1.
2. **Intermodal Raman** (the real prize): delayed nonlinear coupling with its own overlap
   tensor:
   ```
   ∂A_m/∂z ⊃ i γ f_R Σ_{n,p,q} T_R^{(m,n,p,q)} A_n(z,t) ∫ h_R(t−t') A_p(z,t') A_q*(z,t') dt'
   T_R^{(m,n,p,q)} = ∬ F_m* F_n F_p F_q*  with the Raman (not electronic) weight
   ```
   Implemented as a memory kernel along the RK4 path, i.e., generalise the existing
   `coupling_tensor` slot from instantaneous to *delayed*. Cost: one FFT-based convolution per
   step per tensor term; keep the tensor rank sparse (Poletti-type selection rules prune it
   hard, we already reproduce those).

### B.4 References to build from (DOI verified)

- Turner, Monro, Afshar V. et al., *Opt. Express* **17**, 11565 (2009): **"part II:
  Stimulated Raman Scattering" of the full-vectorial subwavelength model**; the correct Raman
  response and its mode-overlap weighting for high-contrast waveguides. **Primary reference
  for B.3.** `10.1364/OE.17.011565`
- Wright et al., *IEEE JSTQE* **24**, 5100516 (2018): GMMNLSE with Raman included in the
  multimode expansion; the reference implementation. `10.1109/JSTQE.2017.2779749`
- Christensen, Koefoed, Friis et al., *Sci. Rep.* **6**, 34693 (2016): **measured intermodal
  Raman intensity overlaps** between mode groups; gives realistic magnitudes for `T_R`.
  `10.1038/srep34693`
- Antikainen, Tai, Rishøj, Ramachandran & Agrawal, CLEO 2018, FTh4E.3: **intermodal Raman
  scattering of ultrashort pulses**: power transfer needs *group-velocity matching + non-zero
  overlap*; symmetry can forbid it. `10.1364/CLEO_QELS.2018.FTh4E.3`
- Pourbeyram, Agrawal & Mafi, *Appl. Phys. Lett.* **102**, 201107 (2013): Raman cascade in a
  GRIN MMF; peaks appear in specific modes (beam cleanup). `10.1063/1.4807620`
- Wu, Ding, Peng et al., *Appl. Sci.* **13**, 12923 (2023): **multi-peaked Raman response of
  chalcogenide fibre**; the reference for upgrading our single-Lorentzian ChG model to
  multi-peak (matters for MIR broadening). `10.3390/app132312923`
- Dudley, Genty & Coen, *Rev. Mod. Phys.* **78**, 1135 (2006): `R(t) = (1−f_R)δ(t) + f_R h_R(t)`
  and the shock term; our scalar engine's spec. `10.1103/RevModPhys.78.1135`

---

## 3. Gap C: no multimode self-steepening / TPA / free carriers

### C.1 What exists

Scalar engine has `iτ_shock ∂/∂t` (self-steepening, `examples/39_conserving_shock_self_steepening.py`),
time-resolved TPA and free-carrier steps. The multimode engine has **none** of these
(`eftekhar_2019` names exactly this as why its DW-comb cascade was out of scope).

### C.2 Physics

- **Self-steepening** is the frequency dependence of the *nonlinear* response; in the mode
  expansion it appears as (i) a `∂/∂t` acting on the nonlinear polarization and (ii) the mode
  fields/coupling coefficients evaluated at the *local* frequency. Practical form per mode:
  ```
  ∂A_m/∂z ⊃ i γ_m [ |A_m|²A_m + (i/ω₀) ∂/∂t(|A_m|²A_m) ]        # shock, τ_shock ≈ 1/ω₀
  ```
  In the rigorous GMMNLSE it is handled by working with the spectral field and the full
  coupling tensor (Wright JSTQE §III), that is the route we should mirror, not a per-term hack.
- **TPA / FCA** scale with `Im(χ⁽³⁾)`; needed if any part of the broadened spectrum crosses
  `E_g/2` (GeAsSe `E_g = 1.6 eV` → ≈1.55 µm). At a 2 µm pump the *pump* is safe, but the
  blue edge is not.

### C.3 References (DOI verified)

- Kolesik & Moloney, *Phys. Rev. E* **70**, 036604 (2004): unidirectional pulse propagation
  (UPPE), the rigorous basis for shock/self-steepening in a mode expansion.
  `10.1103/PhysRevE.70.036604`
- Wright et al., *IEEE JSTQE* **24**, 5100516 (2018): how the GMMNLSE retains self-steepening.
  `10.1109/JSTQE.2017.2779749`
- Hesketh, Poletti & Horak, *Nonlinear Photonics* NP.2012.NTu2D.6: spatio-temporal collapse
  of ultrashort pulses in multimode fibre (the regime where shock + self-focusing matter).
  `10.1364/NP.2012.NTu2D.6`

---

## 4. Gap D: Taylor β only; FWM on the degenerate `n==p` slice

- **D.1 Full β(ω).** The engine takes Taylor `betas` about ω₀. An octave (or two) of MIR
  broadening is *exactly* where a truncated Taylor series fails, our own `dudley_2006_scg`
  reproduction already documents grid-order sensitivity. Two options: (i) raise the order and
  prove convergence (order 8–10, report the order), or (ii) add a **full-β multimode** path
  (spectral operator per mode, mirroring the scalar `TaperedGNLSESolver`/`ZDependentDispersion`).
- **D.2 FWM slice.** `_fwm_rhs` only closes the **degenerate `n == p`** geometry (gated by
  `oam_l`). Intermodal FWM in our design may live in the *non-degenerate* Eq.-(6) arms, which
  are outside the engine's channel description. **Verify this in Phase 3/4 before trusting any
  intermodal-FWM prediction.** If needed, widen the FWM index set to the full `S⁽²⁾_mnpq`
  contraction (same tensor we build in Gap A).

References: Poletti & Horak (Eq. 6/18/19 selection rules) `10.1364/JOSAB.25.001645`;
Wright et al. JSTQE (full tensor contraction) `10.1109/JSTQE.2017.2779749`.

---

## 5. Bibliography (DOI-verified)

### Multimode propagation & coupling tensors
| Ref | DOI | Use |
|---|---|---|
| Poletti & Horak, *JOSA B* **25**, 1645 (2008) | `10.1364/JOSAB.25.001645` | MM-GNLSE; Eq. 7 overlaps `Q⁽¹⁾/Q⁽²⁾`; selection rules |
| Wright et al., *IEEE JSTQE* **24**, 5100516 (2018) | `10.1109/JSTQE.2017.2779749` | GMMNLSE; full coupling tensors; Raman; shock; reference solver |
| Mumtaz, Essiambre & Agrawal, *JLT* **31**, 398 (2013) | `10.1109/JLT.2012.2235414` | Manakov weights (SPM 8/9, XPM 4/3), our engine's spec |
| Afshar V. & Monro, *Opt. Express* **17**, 2298 (2009) | `10.1364/OE.17.002298` | Full-vectorial Kerr mode overlaps (high-contrast), Gap A |
| Turner, Monro & Afshar V. et al., *Opt. Express* **17**, 11565 (2009) | `10.1364/OE.17.011565` | Full-vectorial **Raman** in subwavelength guides, Gap B |
| Kolesik & Moloney, *Phys. Rev. E* **70**, 036604 (2004) | `10.1103/PhysRevE.70.036604` | UPPE basis for shock/self-steepening, Gap C |
| Wright, Wu, Christodoulides & Wise, *Nat. Phys.* **18**, 1018 (2022) | `10.1038/s41567-022-01691-z` | Physics of highly multimode nonlinear systems (review/context) |

### Multimode Raman
| Ref | DOI | Use |
|---|---|---|
| Christensen et al., *Sci. Rep.* **6**, 34693 (2016) | `10.1038/srep34693` | Measured intermodal Raman overlaps |
| Antikainen et al., CLEO 2018, FTh4E.3 | `10.1364/CLEO_QELS.2018.FTh4E.3` | Intermodal Raman scattering of ultrashort pulses |
| Pourbeyram, Agrawal & Mafi, *APL* **102**, 201107 (2013) | `10.1063/1.4807620` | Raman cascade in GRIN MMF |
| Wu, Ding & Peng et al., *Appl. Sci.* **13**, 12923 (2023) | `10.3390/app132312923` | Multi-peaked chalcogenide Raman response |

### Multimode solitons / dispersive waves / STMI (physics anchors we already reproduce)
| Ref | DOI |
|---|---|
| Wright, Renninger, Christodoulides & Wise, *Opt. Express* **23**, 3492 (2015) | `10.1364/OE.23.003492` |
| Wright, Wabnitz, Christodoulides & Wise, *Phys. Rev. Lett.* **115**, 223902 (2015) | `10.1103/PhysRevLett.115.223902` |
| Wright, Christodoulides & Wise, *Nat. Photon.* **9**, 306 (2015) | `10.1038/nphoton.2015.61` |
| Eftekhar et al., *Nat. Commun.* **10**, 1638 (2019) | `10.1038/s41467-019-09687-9` |
| Krupa et al., *APL Photonics* **4**, 110901 (2019) | `10.1063/1.5119434` |
| Renninger & Wise, *Nat. Commun.* **4**, 1719 (2013) | `10.1038/ncomms2739` |
| Hesketh, Poletti & Horak, NP.2012.NTu2D.6 | `10.1364/NP.2012.NTu2D.6` |

### Chalcogenide material + SCG (design targets / material source)
| Ref | DOI | Use |
|---|---|---|
| Karim, Rahman & Agrawal, *Opt. Express* **23**, 6903 (2015) | `10.1364/OE.23.006903` | **Ge₁₁.₅As₂₄Se₆₄.₅ channel-waveguide SCG**; corrected GeAsSe dispersion |
| Karim & Rahman, *Opt. Quantum Electron.* **48** (2016) | `10.1007/s11082-016-0458-5` | GeAsSe rib waveguide, ultra-broadband MIR SCG |
| Kubat & Bang, *Opt. Express* **24**, 2513 (2016) | `10.1364/OE.24.002513` | **Multimode SCG in chalcogenide fibres** (nearest competitor) |
| Xing et al., *Laser Phys.* **32**, 115401 (2022) | `10.1088/1555-6611/ac979a` | **Intermodal DW + soliton collision in multimode ChG fibre** (Phase-0 anchor) |
| Khalifa, Salem & Cherif, *Appl. Opt.* **56**, 4319 (2017) | `10.1364/AO.56.004319` | MIR SCG in **multimode** As₂Se₃ PCF |
| Xia, Huang & Zhang et al., *Front. Phys.* **9**, 598091 (2021) | `10.3389/fphy.2021.598091` | **On-chip** ChG waveguide MIR SCG |
| Eslami et al., *JOSA B* **36**, A72 (2019) | `10.1364/JOSAB.36.000A72` | High-power SC in multimode fluoride fibre |
| Naskar, Adhikary & Basu, *Phys. Scr.* **101**, 385502 (2026) | `10.1088/1402-4896/aea389` | Taper-assisted ChG rib, 2.90 octaves (scalar competitor to our taper) |
| Ghosh et al., *J. Phys. Photonics* **1**, 044003 (2019) | `10.1088/2515-7647/ab3b1e` | ChG PM-PCF MIR SCG |
| Zhang et al., *Infrared Phys. Technol.* (2025) | `10.1016/j.infrared.2025.105882` | Coherent MIR SCG in ChG fibre |
| Dudley, Genty & Coen, *Rev. Mod. Phys.* **78**, 1135 (2006) | `10.1103/RevModPhys.78.1135` | SCG physics + `R(t)`/shock reference |

---

## 6. Implementation order & test plan

Proposed order (each = one PR to `photonics_helper`, with tests + `docs/api/*.md` + `CHANGELOG.md`):

1. **PR-A** `mode_analysis.py`: N-mode export loader, `mode_dispersion`, `overlap_tensors_from_fields`, `build_multimode_deck`.
   *Tests:* femwell↔tidy3d `<2 %`; tensors vs `reproductions/poletti_2008_multimode` `≲1e-3`;
   reduction to scalar/vector engines at machine precision; mode-continuation invariance.
2. **PR-B** multimode Raman: `include_raman`, `raman_response` (per-channel) then intermodal
   `raman_tensors`. *Tests:* N=1 == scalar engine with Raman; SSFS vs `gordon_1986_ssfs`;
   energy/photon-number conservation; a documented intermodal-Raman transfer case.
3. **PR-C** multimode self-steepening (+ optional TPA/FCA). *Tests:* N=1 == scalar shock deck
   (`examples/39`); convergence/energy-drift guards as in the scalar engine.
4. **PR-D** full-β multimode dispersion operator; widen FWM to the full `S⁽²⁾_mnpq` contraction.
   *Tests:* β(ω) input vs high-order Taylor agreement in the narrowband limit; Poletti Eq. 18/19
   selection rules preserved; octave-spanning deck stable at `dt ≤ 1 fs`.

Until PR-B/C/D land, every MM-SCG run must **state** which of these are switched off
(the plan's §5 "honesty" criterion).

> **2026-10-16: LANDED.** PR-A (step7 `photonics_helper/mode_analysis` + `tests/test_mode_analysis.py`), PR-B (step8 `include_raman`/`raman_tensors` + `tests/test_multimode_raman.py`), PR-C (step9 `include_shock`/`include_tpa`/`include_free_carriers` + `tests/test_multimode_shock_tpa.py`), PR-D (step10 `full_beta`/`fwm_full_tensor` + `tests/test_multimode_full_beta_fwm.py`). The honesty disclaimer is retired (`docs/multimode-gnlse.md` scope line).
>
> **2026-10-16: VERIFIED (full pass, then corrected).** Every claim above was checked against the actual code and re-run. Five defects found and fixed in the same uncommitted batch:
> 1. **step10, shock applied twice on the FWM arm**: `_fwm_rhs` pre-applied `(1 + iτΩ)` AND `_nonlinear_rhs_tensor` applied it again to the assembled sum → `(1 + iτΩ)²` on the FWM polarization (verified numerically: the error was the size of the whole shock term). Shock is now applied exactly once by `_nonlinear_rhs_tensor`; a regression test asserts `_nonlinear_rhs_tensor` == shock(diag + raw FWM) to 0.0 and that the doubled operator differs.
> 2. **step7 → engine contract, complex `S⁽²⁾` silently clipped**: `overlap_tensors_from_fields` always emitted complex128 `fwm_weights` and the engine coerced them `dtype=float` (ComplexWarning, imaginary part dropped, physically meaningful for propagating-mode fields). Now: real-if-close on the step7 side, complex-capable `fwm_weights`/`_fwm_factor`/`_fwm_rate_max` on the engine side; a test proves a complex-field export keeps its imaginary `S⁽²⁾`.
> 3. **step7, `cladding_index` was a silent no-op**: `Γ_m` was hardcoded to 1 while the docstring/API claimed a multi-cladding-aware confinement mask (which the fields-only export contract cannot provide). Parameter removed; `Γ_m = 1` is documented as honest scope (deck authors pre-scale `n2·Γ_m`), in code, `docs/api/mode-analysis.md`, and the CHANGELOG.
> 4. **step7, dead/mismatched code**: `ensure_continuation` claimed overlap-based relabelling but sorted by mid-grid `n_eff` (permutation-invariant, fundamental first, which is what the engine's `group_delays[0] == 0` contract actually needs); the unused `_overlap_matrix` and a never-consumed numerical OAM-winding computation (`oam_l` correctly returns the export labels) are removed and the docstrings/docs/CHANGELOG now state the real behaviour: cross-λ continuation is the solver-export contract (one physical mode per file).
> 5. **step8/step9, two foot-guns now fail loudly at init**: `raman_tensors` without `include_raman` used to crash later on an `assert` (they share the `h_R` kernel); `conserving_shock` was silently ignored on the coupled tensor/FWM paths. Both are rejected with `ValueError` at construction, with tests.
>
> Post-fix suite: `tests/` green (1243 → 1247 tests, +4 new guards), all four step test files pass.

> **2026-10-20: POST-CLOSURE LITERATURE AUDIT (found and fixed).** Re-reading
> Poletti–Horak Eq. 6/7, Mumtaz Eq. 6/8 and Wright Eq. 1 against the shipped
> code found four coefficient-scale defects that the step7–10 tests did not
> exercise (they were shape/reduction tests, not absolute-scale tests):
> 1. `coupling_tensor` was `½(γ_i+γ_j)·xpm`; the literature value is
>    `γ₀·xpm = (n₂ω₀/c)·Q⁽¹⁾_iijj`. The diagonal must be `γ_i`, the old form
>    was low by `(A_eff,0/A_eff,i)²` (0.75² on the rib deck).
> 2. `lp_degenerate` (XPM 2/3) is the two-polarization vector strip-down, not
>    the co-polarized spatial-mode XPM. Added `coef_model="co_polarized"`
>    (SPM 1, XPM 2, FWM 1, Mumtaz Eq. 6 scalar limit) and made the mode bridge
>    default to co-polarized XPM (`xpm_model="co_polarized"`).
> 3. `S_0000`-normalised FWM/Raman tensors inherited `Fiber.A_eff`; added
>    `gamma_reference = γ₀` (engine + deck) so their absolute scale is fixed.
> 4. `_fwm_rate_max` ignored absolute `fwm_coupling_tensor`; stability could
>    be under-estimated. Now uses `_fwm_coupling` on both arms.
> 5. `_photon_number` omitted the `dω` spectral measure the scalar engine
>    carries; absolute (A5) photon number was low by `dω`. Fixed.
>
> Regression suite: `tests/test_multimode_audit.py`. Physics: Agrawal's
> co-polarized Kerr limit (SPM 1, XPM 2) and Mumtaz Eq. 6; see
> `CHANGELOG.md`.

## 7. Doc/impl mismatches to fix regardless (both FIXED 2026-10-16, step8)

- `docs/multimode-gnlse.md` claims Raman is implemented per channel → either implement (PR-B)
  or correct the doc immediately.
- `docs/multimode-gnlse.md` "Scope (v1)" still lists *pump depletion* as future work; it is
  implemented (`fwm_pump_depletion`), stale.

---

## 8. External project GAP audit (2026-10-20): `~/Research/sim/multimode_scg/GAP.md`

The project-side GAP listed gaps E–H. Each was checked against the library
code and re-derived; only the genuine library-side items were implemented.

| External gap | Verdict | Landing |
|---|---|---|
| **E** scalar static full-β(ω) | **genuine, blocking** | `SplitStepEngine`/`GNLSESolver(full_beta=...)`, ETDRK4 support, adaptive β₂, N=1 ≡ multimode. `tests/test_gnlse_full_beta.py` |
| **F1** energy-based launch helper | **genuine** (the draft's per-shape energy factors were partly wrong) | `Envelope.from_energy` + `SHAPE_ENERGY_FACTORS`; corrected lorentzian `π/2`, triangular `2/3`, exponential `1`, super-gaussian `2^(−1/(2N))Γ(1/(2N))/N`. Airy rejected (not in L²). `tests/test_pulse_energy.py` |
| **F2** amplitude-convention marker + guard | **genuine** | `Envelope.amplitude_convention`, `Envelope.from_peak_power`/`from_fwhm_peak_power` marked `"field"`, `Wave/Envelope.to_engine_units`, engine-side `ValueError`. |
| **F3** scalar `raman_response=` kwarg | **genuine** | `SplitStepEngine`/`GNLSESolver` accept the override with `fiber.raman_response` fallback; parity test in `tests/test_pulse_energy.py`. |
| **G** Airy FWHM metadata ≠ measurement | **genuine, narrower than stated** | `Envelope.measured_width()` is shape-aware; Airy is documented as the **main-lobe** FWHM (the tail is not in L²). `Envelope/Wave.calc_width` now agree with metadata for canonical shapes. Also fixed the parabolic FWHM (`2√(1−1/√2)·T0`), which was falling through to `T0`. |
| **H** project launch √2 narrower | **project-side, not a library gap** | Not implemented here. The library's `from_fwhm` already takes the intensity FWHM for every shape; the project must route its launch through it (or `from_energy`). |
| TaperedGNLSESolver full-β | **not implemented deliberately** | `TaperedGNLSESolver` is the z-dependent `β(ω,z)` API. A static full-β curve belongs on `SplitStepEngine`/`GNLSESolver`; adding a second, contradictory dispersion argument to a taper solver would be an API trap. |

False / overstated claims in the external draft:

- **Airy energy factor** (`PR-F1`): `Ai(−x)` oscillates as `|x|^(−1/4)` for
  `x→+∞`, so `∫|Ai|²dx` **diverges**. There is no finite whole-line Airy
  energy; `from_energy("airy", ...)` correctly raises.
- **Per-shape energy factors**: lorentzian is `π/2` (not `π`), triangular
  `2/3` (not `1`), exponential `1` (not `ln2`), super-gaussian
  `2^(−1/(2N))Γ(1/(2N))/N` (not `√(π/N)`). All are now derived and
  numerically round-tripped.
- **`TaperedGNLSESolver` full-β**: not a blocker for the scalar path; the
  scalar `SplitStepEngine` is the engine the project needs.

---

## 9. External gap audit, 2026-10-23 (`~/Research/sim/oam_scg/GAPS.md`)

The OAM project's gap ledger (`GAPS.md`, IDs A to D) was audited against the
library. This section records what was closed in this repo, what was already
closed, and what is not a library task. Openspec change:
`openspec/changes/close-oam-library-gaps/`.

### Library gaps (A)

| Gap | Verdict | Landing |
|---|---|---|
| A1, exact vector solver unreachable | Genuine, P1 | `photonics_helper/vector_modes.py` solves the Okamoto hybrid, TE, and TM equations for any order. It uses a pole-aware full scan and LP-root seeding, with no materials or radius defaults, and returns `(kind, ℓ, ν, m, u, n_eff, β)`. `tests/test_vector_modes.py` checks the weakly-guiding reduction below 1e-4, agreement with the published solver to 1e-9, and finite `HE21`/`TM01` where the ±0.5 bracket returns NaN. |
| A2, only "Bessel" entry point is scalar LP, no `n_eff` | Genuine, P2 | `reproductions/poletti_2008_multimode/overlap.py::lp_neff` returns `n_eff` from `u/V` and optionally `β`. The docstring states weakly-guiding only and points at `vector_modes`. |
| A3, no spin/azimuthal decomposition | Genuine, P1, closed | `mode_analysis.azimuthal_spectrum`, `spin_azimuthal_spectrum`, `azimuthal_purity`, and `azimuthal_charge`. `tests/test_mode_azimuthal.py` pins purity 1.000 and 0.500, C₄ leakage below 1e-3, and the axis-swap sign flip. This landed during the OAM audit and is now covered by the openspec delta. |
| A4, ℓ-gate never measured | Genuine, P2 | `MultimodeSplitStepEngine.power_by_l()` and `power_by_l_vs_z`. `tests/test_oam_power.py` checks a flat budget with FWM off, a class-changing exchange with it on, and a loud failure without labels. |
| A5, no beat-length helper (1000× error) | Genuine, P3 | `phase_matching.beat_length` returns a `Length`. Tests cover the 1.55 mm regression, zero Δn, and sign invariance. |
| A6, no photoelastic or CTE data or stress model | Genuine but data-blocked | Not implemented. `materials.db` has no `p₁₁`, `p₁₂`, or CTE values, so a `stress_birefringence` model would be an unpopulated API. Add the data first, then the column and model. |
| A7, structured light cannot be projected onto solved modes | Genuine, P2 | `mode_analysis.project_field_onto_modes` and `ModalProjection` compute `a_m = ∬ conj(F_m) E dA`, the coupling efficiency, and the residual power. `tests/test_mode_projection.py`. |

### Sibling project, toolchain, and project gaps (B, C, D)

- B1, `export_rib_modes.py` axis renaming: external. It lives in
  `multimode_scg`. The sign risk is now measurable with `azimuthal_charge`,
  and the axis-swap test pins `ℓ → -ℓ`.
- B2, prototype drift: external. Promote or delete in `multimode_scg`; the
  promoted function now exists upstream.
- B3, `f_R` caveat on reused numbers: a rule, not code.
- C1, femwell fork API delta: external (installed package). The project's
  `env.md` records the delta.
- C2, femwell `complex64` splitting floor: external. Patch the fork.
- C3, tidy3d local subpixel is licensed: external. Use the analytic solver
  (`vector_modes`) or femwell for symmetry-sensitive numbers.
- C4, no polar/radial mesh recipe: external. Project docs and script.
- D1: closed by A1. `vector_modes` is standalone; nothing imports the
  reproduction's equations.
- D2: closed by A5. There is one unit layer for `λ/Δn`.
- D3: project-side. Audit the C₆ᵥ ring geometry in `oam_scg`.
- D4: project-side. `p ≈ 2` remains a hypothesis to falsify, not a library
  claim.

### Regression suite

`tests/test_vector_modes.py`, `tests/test_mode_projection.py`,
`tests/test_oam_power.py`, `tests/test_mode_azimuthal.py`, and the new
beat-length cases in `tests/test_phase_matching.py`.
