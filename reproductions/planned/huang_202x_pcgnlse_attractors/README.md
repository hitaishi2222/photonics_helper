# Huang et al. (2026+) — Photon-conserving Raman soliton attractors

> Weiye Huang, Junhong Yang, Tao Sun, Qian Gao, Peilong Yang, Jintao Fan,
> Günter Steinmeyer, Jinhui Yuan, Chao Mei: "Photon-conserving Raman soliton
> attractors in focusing and defocusing Kerr media"
> (arXiv:2607.05244v1, 6 Jul 2026). PDF: `huang_pcgnlse.pdf` (LOCAL-ONLY,
> gitignored — do not commit). Page images in `pages/p-01..21.png`.

## Why this paper (what solving it decides)

This is the source paper of the opt-in photon-conserving shock operator we
shipped in `fix-audit-issues-batch` §5 (`SplitStepEngine(conserving_shock=True)`,
derived from *this* line of literature). Reproducing it decides:

1. **Independent bench of `conserving_shock=True`** — the operator we
   implemented must reproduce the paper's dimensionless simulations: photon
   number conserved to integrator error, monotonic redshift, monotonic
   energy decay, attractor fixed points for BOTH focusing and defocusing γ.
2. **A second, clean #0-style discrimination.** The soliton self-frequency
   shift direction is dispersion-time-direction sensitive: under the
   standard GNLSE the defocusing case predicts an *unphysical blue shift*
   — the paper's headline pathology. If our engine reproduces the red/blue
   split with the flag on/off, the post-#0 Raman direction passes another
   independent test in the negative-nonlinearity regime (untested so far;
   our Gordon SSFS tests are γ > 0 only).
3. **Attractor analytics**: the method-of-moments five-parameter closed
   forms and attractor conditions (Eqs. 25 bright / 37 dark) are closed
   forms we can assert against directly — analytically decisive evidence
   per the house rule.

## Module stack

- `SplitStepEngine(..., conserving_shock=True)` (the 2026-09 operator).
- Standard delayed-Raman response + self-steepening; **signed γ** —
  engine must accept γ < 0 and negative dγ/dω (audit this call site;
  if the engine clamps |γ| anywhere, that is a found bug).
- No multimode, no taper. Dimensionless scaling keeps runs cheap.

## Paper parameters (to extract in the working session, with page refs)

- The paper's dimensionless scaling: `s_D`, `δ_Ω`, chirp parameters and
  the two attractor conditions (Eqs. 25/37 + the supplementary S-numbers).
- The specific simulation panels (probably Figs. 2–5): initial chirped
  pulse + Raman fraction + propagation distance in z-units; grid + Raman
  response used (likely the single-decay-exponential r(t) = f_R·…/τ form).
-Extract exact values from `pages/` PNGs before writing `parameters.json`.

## Engine calls (planned)

- Standard case: `SplitStepEngine(include_self_steepening=True,
  include_raman=True)` — shows the paper's documented GNLSE pathology
  (blueshift + energy growth for γ < 0) — *recorded*, not "fixed".
- pcGNLSE case: same + `conserving_shock=True` — must match the paper.
- Attractor check: run the paper's moment integrator analytically
  (5-parameter ODEs from Eqs. 25/37) alongside the engine and compare the
  fixed point (peak power, width, chirp, detuning) to ≤ stated tolerance.
- Dark-soliton case: cw-hole initial condition, same asserts.

## Asserts (draft)

- Photon number monotone non-increasing to integrator error (γ < 0,
  flag on); energy decay monotone.
- Frequency-centroid trajectory: monotone redshift under pcGNLSE for
  BOTH signs of γ; blue-shift only in the standard-GNLSE negative-γ path.
- Attractor fixed point from the moment ODEs vs the engine's late-time
  pulse parameters (rel error stated per figure, target ≤ few %).
- Standard-GNLSE positive-γ path byte-identical to the pre-flag engine
  (flag-gated branch, already tested; re-assert here).

## Execution checklist

- [ ] Read `pages/p-02..p-14.png` + supplementary pages; extract every
      simulation parameter with page refs.
- [ ] `parameters.json` (strict house format) + `reproduce.py` skeleton.
- [ ] Audit engine γ-sign handling (incl. dγ/dω in the SS/Raman terms);
      fix anything that clamps signs silently.
- [ ] Dimensionless → SI mapping check (assert on one known ξ-soliton).
- [ ] Reproduce: standard pathology figures + pcGNLSE corrected figures
      + attractor fixed-point table.
- [ ] pytest wrapper in the local-only `tests/test_reproductions.py`
      (fast-config only; heavy configs marked slow).
- [ ] Move folder out of `planned/`; add the main README row; update
      `ISSUES.md` #1 (independent validation landed).

## v1 repro status (2026-09-29, first working session)

Implemented: dimensionless RK4IP solver for the paper's normalized
Eq. (10) (pcGNLSE) / Eq. (11) (standard GNLSE) + the paper's moment
diagnostics (Eqs. 12-16 bright / 26-30 dark) + the bright moment-ODE
overlay (Eqs. 18-22 with the field chirp; the printed chirp Eq. (23) is
inconsistent with the paper's own C = 0 attractor and is recorded, not
asserted) + an SI ENGINE sign bench.

Green (hard asserts):
- Engine bench, the paper's headline claim realized through
  `SplitStepEngine(conserving_shock=True)`: gamma>0 -> both redshift
  (identical magnitudes); gamma<0 -> standard GNLSE BLUESHIFTS (the
  paper's unphysical pathology) while pcGNLSE keeps the redshift.
- Bright Case I (paper Table I): delay 0 -> +1.61 (paper 1.6), frequency
  shift 0 -> -0.287 (paper -0.3), energy conserved to 8e-4, models
  identical at sigma = 0.
- Bright Case II: delay sign split (pc < 0 < GNLSE) and shift-sign split
  (pc redshift ~ -0.3, GNLSE blueshift ~ +0.3), energies conserved.
- Bright moment-ODE overlay (E, eta, Omega, rho) vs direct simulation
  within 15 % (Cases I/II).

Recorded caveats (open):
1. FIRST-ORDER RAMAN ARTIFACT: direct integration of the printed
   Eq. (10) at the paper's own tauR = 1 / rho0 = 2 has a high-w MI-like
   artifact with gain ~ w/35 (measured, dxi-independent) — the
   dimensionless analogue of the engine's `Omega_max < omega_0` shock
   guard. Reproduction uses a super-Gaussian low-pass (corner 12,
   outside the pulse bandwidth ~1.5); recorded, not hidden.
2. Case III panel anchors (E 1->0.5/1->1.3, Omega -> -2.3/+2.3) are not
   reached at the paper's Table-I parameters from the printed equations;
   the paper's own adaptive SSFM setup details (or a missing
   supplementary parameter) are evidently needed. Structural
   sign/trend asserts kept hard.
3. Dark layer RECORDED-OUTSTANDING: Figs. 4-6's delay conventions do not
   map onto the transcribed renormalized-moment definitions without the
   supplement's S1-S22 auxiliaries verbatim; needs a dedicated session.
4. Width/chirp flatness of Fig. 1(f)/(g): our rho drifts to ~2.96 and C
   to ~-0.56 by xi=10 (consistent with the moment ODEs), the paper
   shows flat; same-open-family as caveat 2.

Runtime: full validation ~30 s on this box (12 dimensionless runs +
engine bench, all CPU). Not yet registered in the local-only test
suite; folder stays in `planned/` until caveats 2-3 are resolved.
