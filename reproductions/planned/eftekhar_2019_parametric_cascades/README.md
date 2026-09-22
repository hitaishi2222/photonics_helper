# Planned — Accelerated nonlinear interactions in graded-index multimode fibers (Eftekhar et al. 2019)

**Reference.** M. A. Eftekhar, R. A. Correa, L. G. Wright, Z. Liu, D. A. Nolan,
M.-J. Li, F. W. Wise, D. N. Christodoulides, "Accelerated nonlinear interactions
in graded-index multimode fibers", *Nature Communications* **10**, 1638 (2019),
doi:10.1038/s41467-019-09687-9. PDF: `s41467-019.pdf` (OA version); pages under `pages/`.

**Status:** `[~]` planning; PDF present (OA). Parameter extraction partly done,
parameters.json + reproduce.py not written.
**Target module:** `photonics_helper.multimode_gnlse.MultimodeSplitStepEngine`
(with per-mode dispersion + Raman) + optional **femwell** for real GRIN
taper-mode dispersion.

## What to reproduce

The physics: in a *tapering* GRIN MMF (z-dependently decreasing core radius a(z)),
the self-imaging oscillation rate accelerates along the taper. This produces

1. **Accelerated intermodal oscillation of dispersive waves / solitons**
   (Fig. 2): dispersive waves blue-shift by > 45 nm through the taper because
   the GRIN "transitional" mode spacing `Δτ ∝ a²` shrinks along z.
   With our mode-engine this is encoded by a **z-dependent group-delay
   gradient** — not supported natively; reproduce by *chunked propagation*:
   propagate the engine in successive segments with updated `group_delays`
   per segment (a(z) → β1_p(z)). Verify this reproduces the paper's blue
   drift without changing any physics constant.
2. **Blue-drifting multimode solitons in the anomalous region** (Fig. 3)
   — the paper's own Eq. (1) for the effective L_gI accelerates as a(z)².
   Compare a script-simulated soliton temporal trajectory with the analytically
   predicted accelerated phase from paper Eq. (1).
3. **Cherenkov DW blue-shift positions** (Fig. 4): phase matching of dispersive
   waves in the taper — reuse `phase_matching.dispersive_wave_roots` with the
   z-dependent a(z) dispersion.

## Parameters to extract from pages (Methods are on late pages)

- Pump wavelength / pulse duration / energy (look for "fs", "nJ" on pages ~6–8).
- Taper geometry: initial core radius, adiabatic or not, tapering rate γ_t
  (they use "tapering rate" — page ~ 180 in text), final core radius 10 µm.
- Input spot size = 1.5× fundamental mode.
- Total span before taper, through taper.

## Implementation notes

- Per-mode paraxial GRIN dispersion as in `renninger_wise_2013_grin_solitons`
  (same √(2Δ)(2p+1)/kR expansion) but with a(z) in the β_p(ω) formula each segment.
- Raman term important for soliton #1 (paper Fig. 4: intrapulse Raman
  red-shift) — enable the GNLSE Raman response (multi-phonon dispatch already
  validated in gnlse) — needs materials-db silica Raman response.
- Cherenkov DW phase-matching: convert `phase_matching.dispersive_wave_roots`
  to the taper (per-segment solver) and validate the final DW wavelength vs the
  measured >45 nm blue-shift (≤ ~5 % target).

## Execution checklist

- [ ] Extract per-taper geometry + pulse parameters from `pages/` (PNG snapshots).
- [ ] Write `parameters.json` with every number + provenance (page refs).
- [ ] Chunked `MultimodeSplitStepEngine` driver: iterate segments with updated
      per-mode betas + group_delays; assert small energy drift.
- [ ] Fig. 2 equivalent: spectral waterfall, measure >45 nm blue drift.
- [ ] Fig. 3 equivalent: soliton trajectory vs paper Eq. (1) prediction.
- [ ] Fig. 4 equivalent: Cherenkov root vs measured DW peak.
- [ ] README: outcome + deviations + (optional) femwell-vs-paraxial mode check.
