# Planned — Generalized MI in multimode fibers: wideband parametric amplification (Guasoni 2015)

**Reference.** M. Guasoni, *Phys. Rev. A* **92**, 033849 (2015),
**doi:10.1103/PhysRevA.92.033849**. PDF: `PhysRevA.92.033849.pdf`; 14 pages
under `pages/`.

**Status:** `[~]` parameters fully extracted (below), `reproduce.py` to be written.
**Target:** `multimode_gnlse` (engine SSFM) + a linear-stability eigen solver
implemented in the script (the paper's 4N×4N matrix M, Eq. 8/9) — this gives
the analytic gain curves to compare against.

## Paper's setup (fully stated — Fig. 1 + Tables I/II)

- **Fibre:** 4 modes (LP01, LP02, LP11, LP21, x-polarized, single degeneracy
  of each; isotropic fiber, no cross-degenerate coupling). Core radius R = 12 µm.
- **Pump:** 1550 nm, nsec-class, total 4000 W split **uniformly** over the 4
  modes (`p_nx(0) = √1000 W` each, cw); weak white Gaussian noise (per-freq
  complex noise) as MIRMI seed.
- **Table I** (per-mode, at 1550 nm):

| mode | β [µm⁻¹] | GVM [ps/m] | β₂ [ps²/km] | β₃ [fs³/µm] |
|---|---|---|---|---|
| LP01 | 6.0995 | 0 | 21.7 | 89.5 |
| LP02 | 6.0836 | 10.8 | −147.7 | 36.3 |
| LP11 | 6.0891 | 7.1 | −3.5 | −169.9 |
| LP21 | 6.0848 | 13.6 | −2128.7 | −7361.1 |

  (read again from `pages/p-04.png` when writing parameters.json — OCR above
  may shift digits; the paper adjusts these to limit IM-MI bandwidth to ~30 THz.)
- **Table II** — nonlinear overlap coefficients C_kn normalized to
  C₁₁ = 10 W⁻¹km⁻¹ (=γ fundamental):

| k\n | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 1 | 1.00 | 0.73 | 0.66 | 0.45 |
| 2 | 0.73 | 0.96 | 0.37 | 0.33 |
| 3 | 0.66 | 0.37 | 1.04 | 0.61 |
| 4 | 0.45 | 0.33 | 0.61 | 0.92 |

  (verify digits from pages; used as `xpm_weights`/`fwm_weights` after
  conversion γ_mn = C_kn; note our engine's `xpm_weights[i][i]=1` convention.)
- Noise seed: ASE-like white Gaussian noise, independent per frequency sample,
  added to each mode's cw pump (paper Eq. for r_nx(0,t)).

## What to reproduce

1. **Eigenvalue gain curves vs pump-sideband detuning Ω** (Fig. 3/4, Appendix):
   build the paper's 4N×4N matrix M at each Ω from our Table I/II numbers,
   eigen-decompose, take g_k = −Im(λ_k), plot, and overlay paper's values where
   numerically readable. Assert g_max at the Ω of the paper's figure with ≤ 5 %.
2. **Noise-seeded split-step verification (Fig. 5/10 equivalent):** run the
   multimode engine with the 4000 W CW pump + white-noise seed; measure the
   average noise power spectra R̂_nx(L,Ω) and the amplification factor Â_nx(L,Ω)
   (paper Eq. 12, log-averaged ratio) at Ω coverage across the amplified
   band; compare against the dominant-gain analytic estimate `Â_est = g_dom + L⁻¹ln|w̃|`
   (Appendix A). Target: peak-Ω match ≤ 5 %, gain slope ≤ 10 %.
3. **Dominant-gain switching** (the paper's novel "dominant gain" concept):
   identify Ω regimes where different eigenmodes dominate — qualitative figure.

## Implementation notes

- GVM 10–14 ps/m is enormous relative to ps-scale Intramodal? → the paper
  uses µs-scale (nsec pump): engine time window must span the full GVM × L
  (10.8 ps/m × L km) — the paper propagates L ∼ km (check Fig. 5 caption for
  exact L, likely 30 m–?; extract from pages) with window 200 ns? — settled
  when reading Figs. 5–10 from PNG; note engine dt budget: 30 THz band →
  ℱ ∼ 2¹⁴–2¹⁶ grid; run heavy test marked.
- Our engine's `group_delays` currently handles a single deterministic delay per
  mode: fine (GVM is exactly that). β₂/β₃ from Table I via `betas` per mode.
- Gain eigen solver: 16×16 dense matrix per Ω over the noise band (few hundred
  points) — trivial cost; implement in the script (paper Eq. 8/9 structure with
  C_mn matrix). Cross-check the *linear-stability* growth directly from the
  engine's split-step by measuring per-Ω growth when only that pair is seeded
  (second, independent confirmation).

## Execution checklist

- [x] PDF rendered (`pages/p-01..14.png`).
- [ ] Re-read Tables I/II digits from `pages/p-04.png` (+ propagation length
  L used for Fig. 5–10 from captions).
- [ ] `parameters.json` with the fiber/tables + page refs.
- [ ] `reproduce.py` — eigen-gain curves + Eq. (12) estimate; SSFM-vs-eigen
  comparison; figures matching Fig. 3/5 panel styles.
- [ ] Move to `reproductions/guasoni_2015_generalized_mi_multimode/` when green
  (rename the folder!) and update `planned/README.md` + main index.
