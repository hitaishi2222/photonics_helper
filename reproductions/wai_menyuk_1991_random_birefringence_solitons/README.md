# Reproduced — Stability of solitons in randomly varying birefringent fibers
# (Wai, Menyuk & Chen 1991)

**Reference.** P. K. A. Wai, C. R. Menyuk, H. H. Chen, *Opt. Lett.* **16**, 1231
(1991), **doi:10.1364/OL.16.001231**. PDF + rendered pages: local-only.

**Status** `[x]` — full reproduction, all measured metrics in the outcome
table below. Runtime ~30 s including figures.

## About the article — the physics in a nutshell

This letter extends Menyuk's birefringent-soliton theory (the repo's
[`menyuk_1987_birefringent_pulses`](../menyuk_1987_birefringent_pulses/)
reproduction) from *constant* linear birefringence to **randomly varying**
birefringence: the principal axes of a real fiber perform sudden random
rotations every section of length z_h, and the question is whether the
soliton survives.

The model (paper Eq. (1)) is the coupled NLSE in the mean group frame with
deterministic walk-off ±δ (each axis drifts δ·t₀ per unit of the normalized
distance ξ = |β₂|z/t₀²; relative walk-off 2δ) and the coherent FWM terms,
which average out because Rδ ≫ 1. In each section the axes rotate by the
SU(2) law of paper Eq. (2),

```text
U' =  cos θ · U + sin θ e^{+iφ} V
V' = −sin θ e^{−iφ} U + cos θ · V          (θ, φ uniform on [0, 2π))
```

— a rotation by 2θ at azimuth φ on the Poincaré sphere. Averaging over the
rotations reduces the model to the **Manakov equation with the 8/9
coefficient** (paper Eqs. (3)/(4)): the NLSE description survives and the
soliton does not split, whatever δ, but it *spreads* and *depolarizes*.

First-order perturbation theory (paper Eqs. (5)/(6)) gives the soliton's
random trajectory and the **shadow** — a small fractional pulse in the
orthogonal polarization:

```text
U⁰ = √(9/8)·A·sech(A(t − δ·I(z)))·e^{±iA²ξ/2},   I(z) = ∫ cos 2θ dξ
V  = δ·I₂(z)·∂U⁰/∂t − (i/12)·I₄(z)·|U⁰|²U⁰,
     I₂(z) = ∫ sin 2θ e^{−iφ} dξ,  I₄(z) = ∫ sin 4θ e^{−iφ} dξ
```

The delay is a random walk: the paper states (p. 1233) that these random
coefficients are Gaussian with variance π²z_hZ/(8z₀²) — which our discrete
per-section increments dξ = (π/2)·z_h/z₀ reproduce *exactly* (this fixed the
ξ-normalization of the walk-off integrals; see Conventions note).

**Paper numbers** (Sec. III): 50-ps (intensity FWHM) solitons, soliton period
z₀ = 55 km, z_h = 100 m = z₀/550, δ ∈ {1.25, 2.5, 4, 5, 7.5}, span 40 z₀,
one fixed θ/φ sequence for all figures (five sequences tried, "qualitatively
similar"). Derived: T₀ = 28.36 ps, β₂ = −22.97 ps²/km, γ = 2.027×10⁻³ W⁻¹m⁻¹,
P₁ = 14.09 mW (scalar soliton peak), launch √(9/8·P₁)·sech (Manakov soliton,
χ = 0), engine walk-off 1.620 fs/m per unit δ.

## What we reproduce, exactly

All checks run in `reproduce.py::validate()` on
`photonics_helper.vector_gnlse.RandomBirefringenceEngine` (subclass
`Wai1991Engine` adding the paper's Eq.-(2) rotation law and the
Eq.-(1) local walk-off per 100-m section, FWM dropped = `incoherent`
coupling, GVD applied inside the rotated frame — exact, since both axes
share β₂). Per section: rotate (Eq. 2) → symmetric Strang walk-off
halves ±walkoff/2 → incoherent Kerr step → rotate back.

### Checks and measured outcomes

| Check | Ground truth | Measured |
|---|---|---|
| Fig. 1 — shadow after one z₀, δ = 2.5 | doublet ∝ ∂(sech)/∂t, amplitude ~δ·I₂ | numeric peak **0.0582** vs analytic **0.0532** paper units (ratio 1.095); shape L2 (soliton-centred ±3t₀ window) **0.316** — the paper's own Fig. 1 solid-vs-dotted difference (see note below) |
| Fig. 2 — delay tracks the rotating axes, δ = 2.5, 40 z₀ | delay/δ = ∫cos2θ dξ (Eq. (6)) | corr **0.978**, L2 **0.281** — the paper's "agreed quite well"; amplitude ±0.28 (paper ±0.3). Exact drift mechanics cross-check: delay vs δ·∫(E_lx−E_ly)/E dξ = δ·J, L2 **0.219**; residual = dispersive-wave momentum drift (physical) |
| Fig. 3 — bounded width, no splitting | width oscillates wildly but bounded; "below 2.5 even at δ = 7.5" (their sequence) | max FWHM: δ=1.25 **1.06**, 2.5 **1.24**, 4 **1.57**, 5 **1.73**, 7.5 **3.10**; δ ≤ 5 matches the paper's curves to ~5 %; δ=7.5 excursions exceed the paper's ≤2.3 (see note below); single broadened hump at the maximum — **no splitting**; energy conserved |
| Fig. 4 — polarization budget, 40 z₀ | locked for δ ≤ 1.3; depolarizing toward the ½ asymptote | power in U direction at 40 z₀: δ=1.25 **0.966** (paper "almost constant" ✓), 2.5 **0.919**, 4 **0.800**, 5 **0.845**, 7.5 **0.630**; never below **0.548** in 40 z₀ |
| Energy conservation | engine contract | drift ≤ **6×10⁻¹²** over 22 000 steps, every δ |

### Findings surfaced while reproducing (see `../../REPORT.md`)

1. **The ξ-normalization of the perturbation integrals.** The paper's Eq. (6)
   integrals run over ξ = |β₂|z/t₀² (not over soliton periods): with section
   increment dξ = (π/2)·z_h/z₀ the discrete random walk reproduces the
   paper's own variance formula π²z_hZ/(8z₀²) exactly (p. 1233). Using
   z_h/z₀ instead is off by π/2 and misplaces every quantitative claim.
2. **Walk-off magnitude.** Paper Eq. (1) prints ±δ on the two axes →
   relative walk-off 2δ, engine value 2δ|β₂|/t₀ = 1.620 fs/m per unit δ.
   This convention is pinned by Fig. 2 (delay/δ = ∫cos2θ, not ½∫cos2θ) and
   by the δ-scaling of the Fig.-4 depolarization (0.966/0.919/0.800/0.845/
   0.630 across δ = 1.25…7.5 tracks the paper's curves).
3. **δ = 7.5 width excursions.** Our pinned sequence (and five others tested)
   gives max FWHM 3.1–5.5 at δ = 7.5, vs the paper's ≤ 2.3. δ ≤ 5 matches the
   paper to ~5 %, and the second-moment random-walk estimate for the printed
   model at z_h = 100 m (per-section relative drift 2δ·dξ·t₀ = 1.22 ps)
   predicts ≈ 3.2 — so our excursions are what the paper's own parameters
   imply; their smaller value is presumably a milder realization of their
   single sequence (they showed one of five) or a metric detail. The physics
   claim — wild oscillation, bounded, **no splitting** — is fully confirmed
   (the profile at the width maximum is one broadened hump holding ~90 % of
   the energy).
4. **The shadow's propagation.** The printed Eq. (5) freezes each generated
   shadow contribution at its generation point. The engine evolves it under
   the full Manakov dynamics, where the soliton's XPM potential partially
   traps it and broadens the doublet by ~20 % (peak ratio 1.10, shape L2
   0.32) — the same solid-vs-dotted difference visible in the paper's own
   Fig. 1. Free linear propagation of the shadow (a Green-function variant we
   tested) grossly *over*-broadens it (L2 1.4), confirming the trapping
   picture.

## Conventions note (paper frame ↔ engine frame)

- **Carrier phase.** The paper prints e^{−iA²z/2} in Eq. (6), but under its
  own Eq.-(3) sign convention the soliton phase is e^{+iA²ξ/2} — the same
  convention validated against the engine in the Menyuk-1987 reproduction
  (printed Eqs. (9)/(10) phases are positive there). We use the
  engine-consistent sign and pin the remaining φ-sign of the Eq.-(2) phase
  factor as printed (e^{−iφ}); the shadow shape is only weakly sensitive to
  these signs over one z₀ (phase advance π/4), L2 varies 0.316–0.359 across
  the four sign combinations.
- **Delay observable.** The soliton delay is measured as the intensity
  centroid in a ±8-FWHM window around the peak. The full-grid COM would
  additionally contain the dispersive wave's own momentum drift
  β₂·∫⟨Ω⟩dz, which is physical and correctly absent from the windowed
  soliton delay.
- **Timing jitter scale.** Var[delay/δ] at 40 z₀ = π²·z_h·Z/(8z₀²) = 0.0898,
  std 0.30 — the paper's Fig. 2 traces span ±0.25–0.3 ✓.

## Files

- `parameters.json` — inputs, conventions and derived quantities, traced to
  the paper page.
- `reproduce.py` — Checks 1–4 (`validate()` returns all metrics) and figures
  `fig1_shadow.png`, `fig2_delay.png`, `fig3_width.png`, `fig4_power.png`.
- `OL.16.001231.pdf`, `pages/` — source (local only, do not commit).
