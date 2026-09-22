# Analysis — Dudley, Genty & Coen (2006), *Rev. Mod. Phys.* **78**, 1135

**DOI:** [10.1103/RevModPhys.78.1135](https://doi.org/10.1103/RevModPhys.78.1135)

This note records what the paper actually contains, which parts are
reproducible with the current library, and the order in which they should be
reproduced. The PDF is checked in at
`../dudley_2006_cherenkov_dw/revmodphys.78.1135.pdf` and rendered page images
(with a contact sheet) are in `../dudley_2006_cherenkov_dw/paper_pages/`.

## What the paper is

A 50-page topical review of supercontinuum generation in photonic crystal
fiber. It is built around a single numerical model (the GNLSE, Eq. 2) and a
single reference PCF (Table I; d = 1.4 µm, Λ = 1.6 µm, ZDW ≈ 780 nm), and then
walks through the parameter space:

| Section | Regime | Physical processes |
|---|---|---|
| II–III | review + fiber design | GVD engineering, ZDW |
| IV | model | GNLSE, numerical issues, spectrogram (Eq. 4), noise (Eq. 5) |
| V | 50 fs, 10 kW, anomalous | soliton fission, Raman SSFS, dispersive waves |
| VI | 20–500 fs, wavelength/chirp/noise | SC scaling, coherence |
| VII | 10 ps–CW | MI/FWM, Raman cascade, pico/nanosecond SC |
| VIII | special cases | two ZDWs, multi-pump, polarization |

The paper's *central* result is Fig. 3: the full femtosecond supercontinuum,
which is then "deconstructed" into the individual processes (Figs. 5–9).

## Model inputs (Table I / Sec. V.A)

```
λ₀ = 835 nm,  γ = 0.11 W⁻¹m⁻¹,  L = 15 cm
T₀ = 28.4 fs  (FWHM = 1.763 T₀ = 50 fs),  P₀ = 10 kW
β₂ = −11.830e-3 ps²/m     β₃ = 8.1038e-5 ps³/m    β₄ = −9.5205e-8 ps⁴/m
β₅ = 2.0737e-10           β₆ = −5.3943e-13        β₇ = 1.3486e-15
β₈ = −2.5495e-18          β₉ = 3.0524e-21         β₁₀ = −1.7140e-24
fR = 0.18,  h_R: τ₁ = 12.2 fs, τ₂ = 32 fs,  τ_shock = 0.56 fs
derived: L_D = 6.8 cm, L_NL = 0.91 mm, N = 8.5, z_sol = 10.6 cm
```

The whole analysis below uses exactly these numbers (see
[`parameters.json`](parameters.json)).

## Figure-by-figure reproducibility

Legend: ✅ reproduced here · 🟡 reproducible with caveats · ⛔ needs a library
feature that does not exist yet.

### Femtosecond regime (Sec. V–VI)

| Fig | Content | Method | Status | Where |
|---|---|---|---|---|
| 1 | PCF electron micrograph | image | ⛔ (image only) | — |
| 2 | β₂(λ), D(λ) GVD curves | mode solver / fit | 🟡 Taylor reconstruction gives ZDW 779.9 nm | `fig23_mi_gain.zDW_nm()` |
| **3** | **basic SCG, spectral/temporal evolution** | GNLSE + Raman + shock | ✅ | `fig03_basic_scg.py` |
| 4 | output detail, A/B/C features | spectral filtering | ✅ | `fig04_output_features.py` |
| 5 | ideal N=3 soliton period | NLSE (β₂ only) | ✅ | `fig05_ideal_soliton_period.py` |
| 6 | Raman-induced fission, N=3 | β₂ + Raman | ✅ | `fig06_raman_fission.py` |
| 7 | fission detail vs Kodama–Hasegawa | β₂ + Raman | ✅ | `fig07_fission_detail.py` |
| 8 | DW from j=1 soliton, ±Raman | full β + DW phase matching | ✅ | `fig08_dispersive_wave.py` |
| 9 | soliton ω(z) and DW energy | post-processing | ✅ | `fig09_dw_energy.py` |
| 10 | SC spectrogram (165 THz beat) | Eq. 4 | 🟡 | `fig10_spectrogram.py` (beat differs; method validated) |
| 11 | exp. vs sim. (Corwin, 22 fs) | experiment digitisation | ⛔ | needs digitised data |
| 12 | measured spectrogram | experiment | ⛔ | needs digitised data |
| 13 | SC vs pump wavelength | parameter sweep | 🟡 | script template exists |
| 14 | density plots, pump wavelengths | parameter sweep | 🟡 | extends Fig. 3 |
| 15 | spectral density vs pump λ | parameter sweep | 🟡 | extends Fig. 3 |
| 16 | SC vs pulse duration 20–500 fs | parameter sweep | 🟡 | extends Fig. 3 |
| 17 | evolution for 100/500 fs | parameter sweep | 🟡 | extends Fig. 3 |
| 18 | noise sensitivity (5 shots) | stochastic noise | ⛔ | **no noise source** |
| 19 | mean spectrum + coherence g⁽¹⁾ | noise ensemble | ⛔ | **no noise source** |
| 20 | spectrum & coherence evolution | noise ensemble | ⛔ | **no noise source** |
| 21 | coherence vs wavelength | noise ensemble | ⛔ | **no noise source** |
| 22 | average coherence vs N | noise ensemble | ⛔ | **no noise source** |

### Picosecond / CW regime (Sec. VII–VIII)

| Fig | Content | Method | Status |
|---|---|---|---|
| **23** | **parametric/MI gain vs pump λ** | linear stability | ✅ `fig23_mi_gain.py` (reconstructed β(ω)) |
| 24 | picosecond SC, 2 m, 500 W, 20 ps | GNLSE (long) | 🟡 heavy but direct |
| 25 | spectral evolution, 700/750 nm | GNLSE (long) | 🟡 heavy |
| 26 | spectral evolution, 780/900 nm | GNLSE (long) | 🟡 heavy |
| 27 | SC for 10–500 ps, 2 m | GNLSE (very long) | 🟡 heavy |
| 28 | spectrum + coherence (Raman noise) | stochastic | ⛔ no noise |
| 2-ZDW / multi-pump / polarization (Sec. VIII) | qualitative | GNLSE + dichroic β | ⛔ needs β(ω) with two ZDWs |

## What to reproduce first (priority order)

1. **Fig. 3 — the basic SCG.** It is the paper's headline result and exercises
   the whole stack. *Done* (`fig03_basic_scg.py`); output is octave-spanning
   (500–1257 nm at −20 dB) with the paper's `N` and `z_sol`.
2. **Fig. 5 — ideal N=3 soliton periodicity.** Cheapest and strongest
   validation of the solver (`z_sol` recurrence overlap 0.998). *Done.*
3. **Figs. 6–7 — Raman fission and the Kodama–Hasegawa ejected soliton.**
   Gives an analytic prediction for the *constituent solitons* (power/width),
   which is a much sharper test than an integrated spectrum. *Done* (4 % / 8 %).
4. **Figs. 8–9 — dispersive wave / Cherenkov radiation.** Ties the GNLSE to the
   Akhmediev–Karlsson phase-matching formula with the full β. *Done* (2.6 %).
5. **Fig. 10 — spectrogram.** The natural way to attribute temporal and
   spectral features; *method* validated.
6. **Fig. 23 — MI gain.** Cheap analytic figure that already exposed the
   Ω_c² convention; *done*.
7. **Figs. 13–17 — parameter sweeps.** Straightforward extensions of Fig. 3;
   worth doing once the shock-time issue is fixed.
8. **Figs. 18–22, 28 — coherence/noise.** Require a stochastic Raman source
   that the library does not have; this is the main missing physics feature.
9. **Figs. 24–27 — picosecond/CW.** Direct but computationally expensive
   (metre-scale fibers, long windows, many steps).

## What is missing in the library to go further

> Status audit 2026-09-21: items 1 and 2 below have since **landed**
> (`step2-raman-noise-source`; `tau_shock` override + RK4IP integrator in
> 0.1.9). The reproduction scripts have not yet been re-wired to use them —
> that, plus a full β(ω) PCF model, is what bounds Figs. 13–28 today.

1. **Stochastic Raman noise source** (`Γ_R` in the paper's Eq. 5) and input
   shot noise. Without it Figs. 18–22 and 28 cannot be reproduced, and the
   coherence analysis is impossible. — *LANDED* (`noise.py`: ASE background,
   one-photon-per-mode spontaneous Raman seed with thermal factor), figures
   still to be re-run on top of it.
2. **A user-settable shock time** (`τ_shock`). The paper uses the
   effective-area-corrected 0.56 fs; the library hard-codes `1/ω₀ = 0.443 fs`.
   This is the single largest systematic difference in the full SCG.
   — *LANDED*: `tau_shock=` override (0.1.9).
3. **A full β(ω) model for the PCF** (or a bundled dispersion table). Only the
   835 nm Taylor coefficients are available here, so multi-wavelength figures
   (13–17, 23, 24–27) are either reconstructed (Fig. 23) or not attempted.
4. **A norm-preserving shock integrator.** The existing RK4 shock step leaks a
   few percent of energy at high bandwidth, which limits quantitative
   agreement in the full SCG. — *LANDED as the interaction-picture RK4IP
   integrator*; the residual ≈5–6 % drift is model-intrinsic (first-order
   shock term), not integrator error — see `ISSUES.md` #1.
5. **Digitised experimental data** for Figs. 11–12 (Corwin 22 fs, Dudley 25 fs).

## Verification harness

Every script follows the same pattern: `validate(fast=..., make_plot=...)`
asserts the figure's physics against an analytic reference and returns the
numbers; `main()` writes the PNG. The `fast=True` paths are what
`tests/test_reproductions.py` runs; the `fast=False` paths generate the
paper-quality figures and are intended to be run (or cached) separately.
