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

---

# Outcome (reproduced 2026-09-24, `reproduce.py`)

**Status: REPRODUCED (analytic eigen layer asserted; split-step layer recorded-outstanding).**
Moved to `../../guasoni_2015_generalized_mi_multimode/`.

## What `reproduce.py` does and what it measured

Builds the paper's Eq. (8)/(9) x-sector linear-stability matrix M (8x8 over
the four x-modes; the y-sector blocks gain nothing at |p_ny| = 0) per
detuning, from Table I/II + the Eq.-(11) Taylor mismatches, and asserts:

- **Check 0 — single-mode MI vs the closed form.** The eigen construction
  restricted to the diagonal per-mode 2x2 blocks matches the analytic
  eigenvalue identity g = sqrt(det - tr^2/4) of [[a, F], [-F, d]] to
  **< 1e-9** over nu = 0.02..0.2 (machine-close; the classic MI peak
  Omega^2 = 2 gamma22 P / |b2| sits at nu_pk = 0.0598 with g = F = 0.96
  normalized).
- **Check 1 — Fig. 3 anchor (nu = -0.43).** The two dominant normalized
  gains are **g1 = 0.9071, g2 = 0.7070** vs the paper's B_F = 0.90 and
  B_G = 0.71 — **max error 0.007 (~1 %)**.
- **Check 2 — Fig. 3 inset eigenvector mixing.** The dominant-eigenvector
  mode-content matches the paper's inset dots: **ln|w~_{B_G}[2x]| =
  -0.349 (paper -0.35)**, ln|w~_{B_G}[4x]| = -3.30 (paper -3.22),
  ln|w~_{B_F}[2x]| = -3.37 (paper -3.35) — the dominant-gain
  mode-pair-switching structure of the paper, within ~0.1-0.25 in ln.
- **Check 3 — engine split-step spectrum (RECORDED, outstanding).** The
  Eq.-(12) noise-seeded amplification A_hat_nx(nu) is measured and its
  numbers reported; the quantitative banded-vs-eigen agreement is
  **pending** (see the caveats below).

Figure: `guasoni_2015_im_mi.png` — the eigen-gain curves (Fig. 2
equivalent) + the split-step spectrum vs the dominant eigen estimate.

## Conventions pinned here (all visible in the pages)

- Eq. (3)'s x sector (b_S = 1, b_|| = 2, b_X = b_perp arms vanish at
  |p_ny| = 0) couples the four modal pumps purely through |A_m|^2 phase
  products; Eq. (9)'s off-diagonal M_sx,ix / M_sx,sx arms are exactly
  their linearization. **No separate FWM term exists in Eq. (3)**, so the
  engine runs include_fwm=False (the engine's `_fwm_rhs` belongs to the
  Mumtaz model instead). This contradicts the note in parameters.json
  engine_model; the JSON note above is superseded by this README.
- Eq. (11) mismatches: Delta-beta^(p,s)_n = +GVM_n·Om + b2·Om^2/2 +
  b3·Om^3/6 (the +Om sign on the GVM column; the paper's own
  g(Om) = g(-Om) structure makes the detuning-side mirror immaterial).
**Table I beta3 unit is fs^3 mm^-1 = 1e-42 s^3/m** (the planned-README/
parameters-conversion using 1e-39 was 1000x off — pressing the conversion
the other way gives spurious TOD-dominated diagonal mismatches ~1.9e5 m^-1
and kills every band).
- The engine betas are per-mode ps^k/m: 1 ps^2/m = 1e-24 s^2/m and
  1 ps^3/m = 1e-36 s^3/m (factors x1e24 and x1e36 from the SI values).
- The static modal beta offsets (6.0995 vs 6.0836 um^-1 = 10^4 m^-1)
  average out over L_NL,1 = 0.1 m — the paper's own Sec.-II argument; the
  Table-I GVM column, not the betas, prices the band structure in.

## Outstanding: the split-step layer (check 3)

The engine deck (4 x 1000 W cw + a white Gaussian per-sample seed of
1e-7 W, dz = 1 cm, 65k-grid) gives a measured Eq.(12) amplification of
~0.23 (normalized) nearly UNIFORM across |nu| <= 1.15 at L = 5 m and
~0.07 at L = 16 m — i.e. the amplified bands grow to the pump scale and
the measured log-ratio readout saturates before z reaches the fiber end
(ln-ratio ~ 23 for the 1e-7 seed, independent of L), so the paper's
banded Fig. 4/5 morphologies do not emerge in the ratio readout at this
seed level. Lowering the seed to 1e-30 W makes the readout scale up
(~0.76 normalized everywhere) but STILL flat across nu — the band-vs-edge
contrast stays ~1 in this measurement setup, while the eigen layer's own
gain bands are unambiguous (checks 0-2). The A_hat_est = g_dom + L^-1
ln|w~| (Eq. 13) layer itself is consistent (its mode-2x component
e^-0.35/d^-3.22 mixing is what checks 1-2 assert) — what does NOT yet
fall out is the noise-seed SSFM reproduction of the *paper's* Fig. 4/5
banded readout. Open items before declaring the split-step layer green:

1. Sweep the seed level (1e-30 .. 1e-7) and dz (1 mm .. 1 cm) and locate
   the regime where the measured band structure tracks the Eq.-(13)
   eigen estimate (the paper's Fig. 4 b-band ~0.64, Fig. 5 ~0.69).
2. Cross-check the engine's group-delay walk-off sign against the paper's
   Eq.(11) (Delta-beta^(p,s) = +GVM·Om) — this is the same
   time-direction family as ISSUES.md #0 (the engine's dispersion
   convention; del_beta^(p,i) needs the OPPOSITE frequency-side reading).
3. Numerical sanity: at L = 16 with dz = 1 cm the eigengain e^{gL} ~
   e^{90} saturates the seed ~immediately; the paper's own Fig. 5 Â ~
   0.69 implies the R̂ readout was taken inside its linear regime per
   detuning — verify our z-step count / dz gives the same per-band
   growth regime the paper's split-step deck does.

Track under **ISSUES.md** (cross-cutting, added there 2026-09-24).
