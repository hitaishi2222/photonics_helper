# Planned — Self-organized instability in graded-index multimode fibres (Wright et al. 2015)

**Reference.** L. G. Wright, Z. Liu, D. A. Nolan, M.-J. Li, D. N. Christodoulides
& F. W. Wise, "Self-organized instability in graded-index multimode fibres",
*Nature Photonics* **10**, 471–476 (2016 online 2015), doi:10.1038/nphoton.2015.60/PDF:arXiv:1603.07414.
PDF: `wright2015_arxiv1603.07414.pdf`; pages under `pages/` (29 pp).

**Status:** `[~]` planning; reprint PDF available (arXiv author version, 2-column, full parameter text).
**Target modules:** `multimode_gnlse` (N-mode engine, inter-modal FWM with
`oam_l` gating) + `gnlse.mi_gain_spectrum` analog for multimode gain eigenvalues.

## What is expected

The paper observes (96 m GRIN, pulse λ 1064 nm, 500 fs–2 ps pulses) a
*spontaneous* spatiotemporal instability: noise-seeded pulses break up into
train-of-pulses with regularized spacing governed by GRIN-modulated
phase-matching, co-propagating patterned "instability", even far into the
normal-dispersion regime. Good reproduction targets:

1. **Linear stability / MI eigenvalue spectrum at each z** (paper Fig. 2): the
   GRIN sideband resonances with characteristic spacing; compare against the
   multimode eigenvalue problem (equivalent to our `include_fwm` Jacobian).
2. **Spontaneous formation from a noise seed** of the regularized pattern
   (paper Fig. 3): time-domain intensity vs z waterfall, and the spectral
   fringes with characteristic spacing.
3. **Pulse-rate scaling with core size / Δ** (paper Fig. 4): the characteristic
   periodicity from the GRIN meridian-modal walk-off scales like
   `1/√Δ` (or GRIN self-imaging length) — check our model reproduces this.

## Key physical parameters (extract from `pages/p-*.png` while working)

- GRIN 50 µm core, NA ≈ 0.2, Δ ≈ 0.015 (extract precisely)
- Pump 1064 nm, pulses from 200–800 fs (range reproduced)
- L = 96 m span
- Noise seed white in time and uniform over OAM modes.

## What is different from the Renninger&Wise 2013 case

- Nonlinear regime well past soliton/collapsed regime — modulational-like.
- Spatial profile must be resolved (OAM ℓ gating of FWM triplets), i.e. the
  ℓ_m = 2ℓ_n − ℓ_q selection rule matters more than 2D-beam overlap integrals.
- Requires the intermodal FWM Jacobian (`multimode_gnlse.include_fwm` with
  `fwm_pump_depletion=True`, per-position `fwm_weights` from overlap integrals).

## Implementation notes

- If the paper's own theory holds, a **good linear test** is to take the
  Jacobian of `_fwm_rhs` linearized about the flat-top CW state and extract the
  eigenvalues → gain spectrum; the peak/gain-band edges should match:
  `Omega_p ≈ v_g_z·m·Ω_G = sqrt(2·Δ·...)/(p·R)`-type closed form (paper Eq. 1 is
  the meridian phase-matched spacing: `Δτ_m = (L_GVIN / c)·2Δm² / R² cosθ` —
  re-derive from the paper text).
- Large mode number: paper's own PDE/work uses ~40 OAM modes per radial family —
  use the ℓ-gated triplet formalism (`oam_l=` for the engine) to keep costs sane:
  our engine supports per-mode `betas` differing in `group_delays` and OAM-ℓ.
- For the linear MI prediction our existing single-mode `mi_gain_spectrum` may
  need a *multimode* partner: implement `mm_mi_gain_spectrum(...)` inside the
  reproduction script (small dense linear stability study around the seeded
  multi-mode amplitudes; compare with growth from the split-step run).

## Execution checklist

- [ ] Read `pages/p-02..p-06.png` carefully to extract exact fibre & pulse
  parameters, and the exact gain-curve target values (Table / Fig. 2 insets).
- [ ] Write `parameters.json` + `reproduce.py` skeleton in this folder.
- [ ] Derive + implement the multimode Jacobian stability check (dense eigs).
- [ ] Run split-step with noise seed; show selected pattern metrics:
      `N_pulses`, characteristic periodicity, spectral spacing.
- [ ] README: statement of what is validated + measured RMS criterion (≤5–10 %).
