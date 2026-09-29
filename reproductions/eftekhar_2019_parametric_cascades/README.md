# Reproduced — Accelerated nonlinear interactions in graded-index multimode fibres (Eftekhar et al. 2019)

**Reference.** M. A. Eftekhar, Z. Sanjabi-Eznaveh, H. E. Lopez-Aviles,
S. Benis, J. E. Antonio-Lopez, M. Kolesik, F. Wise, R. Amezcuas-Correa &
D. N. Christodoulides, "Accelerated nonlinear interactions in graded-index
multimode fibers", *Nat. Commun.* **10**, 1638 (2019),
doi:10.1038/s41467-019-09687-9. PDF (OA, gitignored): `s41467-019.pdf`; pages
under `pages/` (gitignored).

**Status: REPRODUCED** (taper-acceleration backbone, 2026-09-27, ~90 s).
**Not in the (local-only) test suite does not exist on this rig; register locally** — script runtime ~90 s, register
as a normal (non-slow) test.

## What is reproduced

The paper's title mechanism on the Methods constants (PDF p-08:
Delta = 1.6e-3, n2 = 2.9e-20 m^2/W, core radius 40 -> 10 um, 1550 nm runs;
taper law a(z) = a0 exp(-gamma z/2), PDF p-02): the GRIN modal spacing and
self-imaging period scale as 1/a(z), so *intermodal dynamics accelerate
along the taper* (Fig. 1b). The z-dependence is propagated with the
chunked-taper pattern (40 uniform 1-mm chunks; per-chunk alliance of
`group_delays`, ladder `phase_offsets` and modal overlap tensors at the
mid-chunk radius).

## Ground truth and outcome

| Check | Ground truth | Measured | Tol | Status |
|---|---|---|---|---|
| MFD-oscillation period, first 15 % of taper | window-avg L_si = 2005.7 um | 2008.0 um | 5 % | ✅ 0.1 % |
| MFD-oscillation period, last 15 % | window-avg L_si = 617.3 um | 602.8 um | 5 % | ✅ 2.4 % |
| acceleration ratio (start/end periods) | 3.249 (window model; pointwise x4) | 3.331 | 5 % | ✅ 2.5 % |
| linear energy conservation | 1 | 1.000000000 | 1e-6 | ✅ |
| accumulated walk-off (linear), p1/p2 | 0.244 / 0.735 fs | 0.056 % / 0.041 % rel | 5 % | ✅ |
| accumulated walk-off (N≈1 nonlinear SPM/XPM) | same model | 0.053 % / 0.039 % rel | 8 % | ✅ |
| nonlinear energy conservation | — | 0.0 % drift | 0.5 % | ✅ |
| analytic walk-off scaling with a | 1/a² (x16.00; 16.31 measured, sqrt-curvature residue ≤2 %) | | 3 % | ✅ |

The *quadratic* 1/a² walk-off scaling is a model-level finding (the paper
quotes the GRIN-ladder picture qualitatively): db1_p ∝ x(a) ∝ 1/(k a)².

## Out of scope (documented)

- The full 55-mode self-organized dynamics, the DW comb cascade, and the
  Fig. 2c ">45 nm blue drift" readout need 55 modes + Raman + self-steepening
  (the paper's own gUPPE stack). The 3-mode engine has no Raman (and the
  multimode engine has no self-steepening); those figures are NOT asserted.
- The N>>1 fission cascade of the paper's 400-fs/500-kW pulses is out of
  scope here; the nonlinear check uses the N≈1 fundamental soliton scale
  (E τ = 2|β2|cA/(ω0 n2) = 1.05 nJ·170 fs for this fibre).
- Engine-side walk-off operator correctness in the Δ = 0.029 regime is
  covered by the Renninger-Wise / Menyuk-1987 / Wai-1991 reproductions; here
  the accumulated walk-off at Δ = 1.6e-3 (~0.24 fs over 4 cm) is asserted
  directly against the engine read-out (it is resolvable: 0.05 %).

## Caveats

- ISSUES.md #0 (dispersion time direction) does not affect these checks:
  only `group_delays` (audited) and `phase_offsets` drive them; the shared
  β2 is even-order and symmetric.
- Figure: `eftekhar_2019_taper.png` — (a) MFD oscillation through the taper
  (visual acceleration), (b) output spectra, (c) N≈1 soliton channels.
