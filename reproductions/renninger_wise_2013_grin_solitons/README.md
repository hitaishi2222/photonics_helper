# Reproduction — Optical solitons in graded-index multimode fibres (Renninger & Wise 2013)

**Reference.** W. H. Renninger & F. W. Wise, *Nature Communications* **4**, 1719
(2013), doi:10.1038/ncomms2739. PDF: `ncomms2739.pdf`; rendered pages under
`pages/`.

**Module stack:** `photonics_helper.multimode_gnlse.MultimodeSplitStepEngine`
(3 symmetric GRIN modes, isotropic XPM/FWM overlap tensors).

**Status:** ✅ reproduced and validated. Runtime ≈ 10 s; added to
`tests/test_reproductions.py`.

## What was reproduced

A ~300 fs, 0.5 nJ sech pulse at 1550 nm is launched into a GRIN MMF through a
10 µm-diameter Gaussian seed, which distributes 92.27 / 7.18 / 0.56 % of the
energy over the first three radially symmetric modes (p = 0, 1, 2). Over 52 m
the coupled-mode equations (paper Eq. 3) form a **multicomponent soliton**.

The reproduction runs the engine twice (linear and nonlinear), building the
3-mode system from the paper's paraxial model:

- per-mode dispersion: one shared material `β₂ = −281 fs²/cm` (paper Eq. 3),
- inter-modal walk-off `β₁⁽ᵖ⁾ − β₁⁽⁰⁾` from Eq. (2): 33.7 / 101.6 fs/m,
- SPM/XPM and FWM overlap tensors from the Eq. (2) Laguerre–Gauss modes,

and checks the results below.

## Ground truth and outcome

| Check | Target | Measured | Agreement |
|---|---|---|---|
| Fundamental mode size `w₀ = (2R²/k₀²Δ)^{1/4}` | 6.66 µm (Eq. 2) | 6.658 µm | exact |
| Self-imaging period `πR/√(2Δ)` | 408 µm | 407.6 µm | exact |
| Inter-modal walk-off p = 1, 2 | paper 33 / 99 fs/m | 33.7 / 101.6 fs/m | ≤ 2.6 % |
| Seed energy fractions | 92.2 / 7.17 / 0.56 % | 92.27 / 7.18 / 0.56 % | < 0.1 pp |
| Linear mode separation over 52 m | `β₁⁽ᵖ⁾·L` = 1.754 / 5.282 ps | 1.754 / 5.282 ps | exact |
| Nonlinear mode separation (locked) | « linear walk-off | 0.052 / −0.317 ps | lock ratio 0.06 |
| Output FWHM vs Eq. (6) fixed point | 212.8 fs | 214.8 fs | 1.0 % |
| Compression vs linear GVD output | « 1 | 45× | — |
| Total-energy drift (nonlinear) | ≈ 0 | 1.0 % | Strang-split error |
| MFD self-imaging period (linear, post-processing) | 408 µm | 400.5 µm | 1.7 % |
| MFD self-imaging period (linear, direct `phase_offsets` run) | 408 µm | 400.5 µm | 1.7 % |
| Required locking shift p=1, p=2 (Δω = −Δβ₁/β₂) | −1.53 / −4.61 nm (blue) | +1.01 / +4.30 nm measured | sign pending ISSUES #0 |

The decisive result — **temporal locking** — is reproduced: the two higher
modes, which walk off by up to 5.3 ps linearly, collapse to |Δt| ≤ 0.32 ps
relative to the fundamental and co-propagate with a common group velocity.
The output duration sits on the analytic Eq. (6) soliton fixed point to 1 %.
Figure `renninger_wise_2013.png` mirrors the paper layout: temporal profiles
(Fig. 2b), per-mode output spectra (Fig. 2c), separation and FWHM vs z, energy
conservation, and the linear MFD self-imaging.

## Caveats / findings

1. **Higher-order-mode spectral shift ordering is not cleanly recovered.**
   The paper reports the higher-order modes blue-shifted (Fig. 2c).  The
   *magnitudes* of the measured centroid shifts (+1.01 / +4.30 nm for
   p = 1, 2) match the kinematically required shifts for group-velocity
   locking (Δω = −Δβ₁/β₂ → −1.53 / −4.61 nm, blue) within 7–34 %, but the
   sign comes out opposite under the library's wavelength map
   `λ = c/(ω₀ + grid.w)`.  The measured shifts + walk-off + XPM trap are
   mutually consistent inside the engine; the sign discrepancy traces to an
   **engine-level Fourier-convention inconsistency** (the dispersion
   operator's group-delay direction is time-reversed relative to the
   group-delay and Raman operators — see **ISSUES.md #0**, established with
   an independent dense-DFT reference at 1e-13).  Until that audit lands,
   the reproduction records `higher_modes_blue_shifted = false` and treats
   the temporal locking (the physical content of "same group velocity") as
   the decisive test.
2. **Spatial self-imaging (paper Fig. 3d) — now reproduced directly.**
   The engine's retarded-frame Taylor expansion starts at β₂ and originally
   omitted the absolute Δβ₀ phase.  Status 2026-09-21: `phase_offsets`
   (added for the Krupa GPI reproduction) carries the `exp(iΔβ₀z)` phase,
   and this reproduction now runs a **direct** short fine-step (20 µm)
   linear propagation with `phase_offsets`: the MFD oscillates at 400.5 µm
   (1.7 % of the analytic πR/√(2Δ) = 407.6 µm) straight from the propagated
   fields, with the old analytic / post-processing reconstruction kept as a
   cross-check (both curves in Fig. f).  The `phase_offsets` run must stay
   short and fine-stepped: at the production 0.1 m step the Δβ₀ phase
   rotates by ~1540 rad/step and would alias.
3. The nonlinear run shows a ~1 % total-energy drift from the Strang-split FWM
   substep (step 0.1 m). The engine's own monitor warns only above 5 %.
4. The engine needs an explicit `nsaves`: the original draft called
   `propagate()` without it, snapshotting every one of ~5000 steps
   (≈ 4 GB per run) and repeatedly rebuilding the full history in
   `fields_vs_z()`, which exhausted system RAM. All diagnostics now use a
   fixed 261 snapshots.

## Usage

```bash
python reproductions/renninger_wise_2013_grin_solitons/reproduce.py
python -m pytest tests/test_reproductions.py -k renninger
```
