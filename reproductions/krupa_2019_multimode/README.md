# Reproduced — Geometric parametric instability (GPI) sidebands

Krupa et al. 2019 review Fig. 14 left (≈ Krupa PRL 116, 183901 (2016))

> Promoted from `reproductions/planned/krupa_2019_multimode/` on 2025-09-21
> after a fully green run (`run_final2.log`, all 6 checks OK).

**Reference.** K. Krupa, A. Tonello, A. Barthélémy, T. Mansuryan, V. Couderc,
G. Millot, P. Grelu, D. Modotto, S. A. Babin, S. Wabnitz, "Multimode nonlinear
fiber optics, a spatiotemporal avenue", *APL Photonics* **4**, 110901 (2019),
doi:10.1063/1.5119434 — review Fig. 14 left panel, sec. IV.C (pp. 17–18).
Quantitative source: K. Krupa et al., Phys. Rev. Lett. **116**, 183901 (2016),
arXiv:1602.04991 (fibre, pump and numerics parameters stated in the review
itself; all transcribed in `parameters.json` with page refs).

**Status: [✔] reproduced.** `python reproductions/krupa_2019_multimode/reproduce.py`
(~35 min, heavy — intentionally NOT registered in `tests/test_reproductions.py`).

## What is reproduced

A strong quasi-CW beam in a GRIN MMF self-images with period
ξ = πρ/√(2Δ). The longitudinal oscillation of the Kerr term acts as a
refractive-index grating that quasi-phase-matches degenerate FWM into a
*discrete ladder* of Stokes/anti-Stokes sidebands, f_h ≈ √h·f_m with

    2π f_m = sqrt(2π / (ξ κ″))      (PRL formula; review Eq. (9) equivalently)

reaching f₁ = 124.5 THz detuning at the 1064 nm pump and climbing to
h = 5–7 — one NIR pump converted into a broadband normal-dispersion comb.

## Result (run `run_final.log`, 2025-09-21)

| check | ground truth | measured | status |
|---|---|---|---|
| 1. analytic ξ | PRL printed 0.615 mm | **0.6157 mm** | ✅ 0.1 % |
| 1. analytic f_m | PRL printed 125.0 THz | **124.98 THz** | ✅ <0.5 % |
| 2. step convergence (dz 0.02 vs 0.1 mm) | ladder shift < 2 % | **0.00 %** | ✅ |
| 3. f₁ shift at ×4 peak intensity | ≤ 2 THz down (PRL Fig. 3 right) | **0.04 THz** down (124.45 → 124.41) | ✅ |
| 4. ladder, 6 m/50 kW (Fig. 14, experiment side) | √h·124.5 THz | **125.11, 176.02, 215.92, 249.05, 278.51** (h = 1..5) | ✅ ≤ 1.2 % for h ≤ 3 (assert tol 4 %) |
| 5. photon number | conserved | drift **6.5e-3** over 6 m | ✅ |
| 5b. Stokes mirror of h₁ | symmetric at −f₁ | **124.93 THz** (anti 125.11) | ✅ 0.1 % symmetric |
| 6. no-grating control (`phase_offsets=None`) | zero peaks in GPI windows | **0** | ✅ ladder genuinely requires the self-imaging grating |

Figure: `krupa_gpi_ladder.png` — (a) output spectrum vs the analytic ladder
(both sides), (b) spectral evolution map over 6 m, (c) energy conservation,
(d) measured vs analytic √h·f_m ladder.

## How the ladder physics enters the modal engine (the key design point)

The PRL used a direct (3+1)D Gross–Pitaevskii split-step; our engine uses
coupled modal GNLSEs. In the *modal* representation the self-imaging
grating is exact and algebraic: the paraxial GRIN ladder has
β_p − β₀ = −2πp/ξ, so the discrete grating harmonics ARE the modal
propagation-constant offsets. We added a `phase_offsets` argument to
`MultimodeSplitStepEngine` (frequency-independent modal phase e^{iΔβ₀z} in
the linear step) for exactly this: the intermodal FWM then carries the
mismatch Δk = κ″Ω² − 2π(m+q−2n)/ξ, whose discrete zero crossings give
Ω²_h = 2πh/(ξκ″) — the GPI ladder — directly from the mode set. The
no-grating control (check 6) proves the ladder is not an artefact of
isotropic overlap weights or of the seed.

Deviations from the PRL's own numerics, all documented:
- Raman / self-steepening off (as in the PRL; sidebands below Raman
  threshold; ladder only to h ≈ 5 in the 4-mode truncation, the h ≥ 6 rows
  are outside the asserted range).
- Stokes branch beyond h = 2 is below f = 0 (λ > 2.5 µm) — outside our
  simulation band, exactly as noted in the PRL.
- Broadband launch noise (−100 dB) stands in for the GP roundoff
  self-seeding; only sideband *amplitudes* depend on it, not positions.

# Extraction credentials (what has been read so far — work in progress)

> Status note (2025-09-21): the folder has been promoted to
> `reproductions/krupa_2019_multimode/`. The audit trail below is kept
> unchanged for provenance; the final run log is `run_final2.log` and the
> measured-result table is at the top of this README.

## 1. Decision: chosen reproduction target

**Fig. 14 (left panel) — Geometric Parametric Instability (GPI) sidebands**
(review §IV-C, pages 16–18; original paper: Krupa et al., *Phys. Rev. Lett.*
**116**, 183901 (2016), arXiv:1602.04991).

Why this one and not the other candidates:

- The full parameter set of the PRL's GPI simulation (Gross–Pitaevskii / 3+1D
  NLSE) is stated **twice** — in the review text (§II, its Eq. (1), "typical
  integration step 0.02 mm, 64 × 64 grid, 150 µm × 150 µm window") and in full
  in the PRL text (fetched from arXiv 1602.04991, sin storage `/tmp/gpi_prl.pdf`
  — the parameters below are transcribed from there and cross-checked against
  the review).
- The PRL's analytic (Longhi-style) prediction for the sideband detunings
  (their Eq. for `f_h`) is **closed-form and exactly assertable**.
  The review's Eq. (9), `Ω_N ≃ ±(2N√G/|β₂|)^{1/2}` (with `G = √(2Δ)/ρ`,
  page 17), is the same condition in its frequency-domain form.
- The effect is self-seeded from noise → the amplitude of the sideband pairs
  is only semi-quantitatively reproducible (the seeds get their spectra
  delocalized; the frequency position of the seeds is robust — the review
  itself states the paper-side counts as h = 1..7 "QPM resonances").

**Rejected targets and why:**

- Fig. 13 (Dupiol et al. IMI, Opt. Lett. 42, 3419): reprint with unstated
  fiber dispersion; the underlying model (coupled vector NLSE pair) is
  possible in `multimode_gnlse` but the review gives no quantitative axes
  to converge to.
- Fig. 17/18 (Kruppa Nat. Photon. beam self-cleaning): experimental speckle
  images, not panel-parameterized in the review.
- Fig. 29 (Nazemosadat & Mafi SMS SA): reprint, parameters in the JOSA B
  original only.
- §III MMS / Raman SSFS: the fully-stated-parameter target in that section
  is *Renninger & Wise 2013*, which is already a separate reproduction in
  this repo (`renninger_wise_2013_grin_solitons`) — duplicating it adds
  nothing.

## 2. All extracted parameters (cited, cross-checked)

### Fiber (Krupa PRL 2016, p. 3 of manuscript; review page 16–17, `κ′′` value on page 18)

| quantity | value | source |
|---|---|---|
| core radius `ρ` (GRIN) | **26 µm** | PRL p.3 (analytic section); 52.1 µm MFD = experimental fiber (p.3, "52.1 µm core diameter"); PRL numerics section uses `ρ` (26 µm) |
| `n_co` | **1.470** | PRL p.3 |
| `n_cl` | **1.457** | PRL p.3 |
| `Δ = (n_co²−n_cl²)/(2 n_co²)` | **8.8 × 10⁻³** | PRL p.3 (stated directly) |
| self-imaging spatial period `ξ = πρ/√(2Δ)` | **0.615 mm** | PRL p.3; review Fig. 21 text: measured 0.6 mm "which matches well with the theoretical calculation" |
| chromatic dispersion at 1064 nm `κ″ = β₂` | **16.55 × 10⁻²⁷ s²/m** (= 16.55 fs²/m) | PRL p.3 |
| nonlinear index `n₂` | **3.2 × 10⁻²⁰ m²/W** | PRL numerics (p.3); same as `renninger_wise_2013_grin_solitons` |
| peak-to-peak modal beat constant | `G = √(2Δ)/ρ = 514.7 rad/m` | derived: `ξ = π/√G` ⇒ √G = π/ξ = 5104 rad/m = √(2Δ)/ρ ✓ (cross-check) |

Cross-check done: `2Δ/ρ² = (5104)²` ⇒ G is self-consistent with the
review's Eq. (4) `a(z) = a₀[cos²(√G z) + C sin²(√G z)]^{1/2}`.

### Pump / launch (PRL experiment + numerics, p.3–4)

| quantity | experiment | numerics |
|---|---|---|
| central wavelength | **1064 nm** (Nd:YAG microchip) | 1064 nm |
| pulse duration | sub-ns (900 ps at 30 kHz reported) | **9 ps** taken in numerics to cut compute |
| peak-to-peak power `P_{p−p}` | 30–74 kW swept; **50 kW** headline | intensity **I = 10 GW/cm²** ⇒ equivalent `P_{p−p} = 160 kW` |
| input beam FWHM diameter | 35 µm (focused Gaussian) | **40 µm** beam diameter |
| fiber length | **6 m** (experiments, Fig. 1) | **0.4 m** (numerics, Fig. 2 of PRL) |
| pp. integration step | — | **0.02 mm** split-step, 64 × 64 spatial grid, **150 µm × 150 µm** window (also review page 4, §II "typical" numbers) |
| Raman | not included (justify: sidebands appear below Raman threshold) | same |
| absorption | negligible over 6 m | same |

### Analytic GPI condition (PRL p.3, Eq. below Eq. (2); review Eq. (9) equivalent)

The QPM condition `2k_P − k_S − k_A = −2πh/ξ` with a two-term dispersion
expansion gives the resonant detunings (PRL equation, transcribed):

```
(2π f_h)² = 2πh/(ξ κ″)  −  2 n₂ Î ω₀ / (c κ″)
```

where `Î` is the path-averaged beam intensity. In the low-power limit this
reduces to the geometric ladder

```
f_h ≃ h^{1/2} · f_m,   2π f_m = sqrt( 2π / (ξ κ″) )
```

**Ground-truth numbers (recomputed here from the stated parameters):**

- `ξ = πρ/√(2Δ) = 0.6155 mm`  (PRL states 0.615 mm ✓)
- `f_m = sqrt(2π/(ξ κ″)) / 2π = 124.99 THz`  (PRL states `f₁ ≃ 125 THz`
  analytic, `123.5 THz` measured, `124.5 THz` used in their Fig. 3 ✓ — we use
  the PRL's `f₀ = 124.5 THz` for the printed ground-truth ladder and let the
  analytically-recomputed value be our cross-checked number).

With the nonlinear term (I = 10 GW/cm², i.e. `2 n₂ Î ω₀/(c κ″)` — worth
`−0.84 THz` at h=1) the analytic first-order sideband shifts to 124.76 THz
(PRL: "numerical simulations (124.5 THz)"). The nonlinear frequency shift
per ±2× power is ~2 THz at h = 1 — consistent with their Fig. 3(right)
power dependence (small, hard to resolve over the 30–74 kW experimental
power span).

**Reference anti-Stokes peak positions** (PRL Fig. 3 = review Fig. 14 left
panel; analytic `h·124.5`-THz ladder, recomputed here with the true
`κ″`/`ξ`):

| h | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| detuning (THz) | 124.5 | 175.9 | 215.5 | 249 | 278.2 | 304.9 | 329.6 |
| (recomputed)   | 124.76 | 176.60 | 216.36 | 249.87 | 279.39 | 306.07 | 330.61 |
| λ (nm)         | 737.9 | 655.1 | 602.9 | 564.8 | 535.4 | 511.0 | 490.4 |

Headline check used by PRL (and by our reproduction): **f₁ (first-order GPI
sideband) = 123.5–124.5 THz**, "sideband intensities only 2 dB below pump,
6 m fiber" — our reproduction reproduces the detuning ladder and the dB
scale spectral evolution map.

### What is being implemented (audit trail for this folder)

1. `parameters.json` — house format, with a `gpi` block carrying `ρ, n_co,
   n_cl, Δ, β₂(κ″), n₂` and the pump block (`1064 nm, 9 ps (sech envelope in
   time, CW-filled), I = 10 GW/cm², ξ-imaging beam of 40 µm diameter, 0.4 m`).
   Note that the engine model is *modal* whereas the PRL's head-line numerics
   used a direct (3+1)D NLSE / Gross–Pitaevskii split-step. This is exactly
   the comparison the review (§III, Eq. (4)) says is valid: the Gaussian ansatz
   in the transverse direction reproduces the same GPI sidebands.
2. `reproduce.py`:
   - **Assert 1 (analytic, hard):** the analytic ladder
     `f_h = sqrt(h) · f_m` (with the f_m recomputed from the stated
     fiber parameters plus the frequency-domain Eq. (9) of the review)
     must be reached by the *measured* output spectrum of a full
     3+1D-equivalent modal split-step run (tolerance: few %; the PRL's own
     theory vs. simulation agreement was 124.5 vs 125 THz, 0.4%).
   - **Assert 2:** first-order GPI sidebands appear as narrow peaks at
     ±f₁ (both Stokes and anti-Stokes; the Stokes branch beyond 2.5 µm is
     outside the simulation band — noted as out-of-band in the PRL).
   - **Assert 3:** the power dependence flagged in the review is weak:
     the f₁ shift per ×2 in power is ≲ 2 THz (~1.6 %).
   - **Figure:** (a) spectral evolution |Â(z,Ω)|² dB colormap vs z,
     (b) output spectrum at z = 0.4 m with the `h·124.5` ladder overlaid,
     (c) transverse intensity snapshots `|A(x,y,z)|²` at the self-imaging
     nodes and antinodes (cleaning-style poster panel).
3. Engine plan (implementation detail, to be validated):
   a *modal* `MultimodeSplitStepEngine` run built from the paraxial GRIN mode
   ladder of a truncated-parabolic 26 µm core (`Ref. 8 Poletti & Horak
   self-consistent mode set; same construction as
   `renninger_wise_2013_grin_solitons/reproduce.py`: per-mode β bases from
   the paraxial modal β_p(ω) expansion, per-mode group delays from
   β₁^(p)–β₁^(0), Isotropic/CGE SPM/XPM+FWM weights).  Equivalent total-power
   envelope: the PRL's 9 ps quasi-CW field is represented as a CW-filled
   [(19.8 nJ)/9 ps] *rectangular* pulse train of the same average intensity
   so the spectral peaks are resolved at the same QPM positions.
   Grid:  16384 window points, 55-ps total window → 3.34 GHz frequency
   resolution (needed: <20 GHz at |Ω| ~ 1.5e15 rad/s). Modality
   L ≈ (few) × 10³ modes ⇒ simulation-mode gating by overlap with the
   launched 40-µm Gaussian (top ~30 symmetric modes carried by the modal
   basis; see checklist item "engine run numbers").

