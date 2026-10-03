# Huang et al. (2026+) — Photon-conserving Raman soliton attractors

> Weiye Huang, Junhong Yang, Tao Sun, Qian Gao, Peilong Yang, Jintao Fan,
> Günter Steinmeyer, Jinhui Yuan, Chao Mei: "Photon-conserving Raman soliton
> attractors in focusing and defocusing Kerr media"
> (arXiv:2607.05244v1, 6 Jul 2026). PDF: `huang_pcgnlse.pdf` (LOCAL-ONLY,
> gitignored — do not commit). Page images in `pages/p-01..21.png`.

## Runnable example

`examples/39_conserving_shock_self_steepening.py` isolates the same
discrimination at the SI-engine level: on a positive-γ fiber
`SplitStepEngine(conserving_shock=True)` is a no-op (|γ| = γ, output fields
agree to ~1e-14), and on the γ-sign bench below it cuts the accumulated
self-frequency shift by 3.2× without reversing it. The same file drives the
`_SHOCK_TAYLOR_LIMIT` grid guard that this reproduction's SI bench runs under.

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
   **FOLLOW-UP UPDATE (2026-09-30):** full-length re-scan puts the pc
   energy anchor ON at xi ≈ 105 (E 1 -> 0.517 vs paper 0.5 — the cited
   anchors belong to a much longer xi axis than the v1 xi=10 deck), while
   the pc Omega magnitude at that anchor is −0.41 vs the paper's −2.3:
   the E law reproduces, the SS-Raman shift magnitude does not
   (growth rate ~4.1e-3/ξ vs the paper's implied ~2.3e-2/ξ, factor ≈ 5).
   Recorded; same missing-detail family (their adaptive SSFM setup).
   Anchor row in parameters.json annotated accordingly.
3. Dark layer — **STRUCTURAL PART CLOSED 2026-09-30** (`diagnostics/check_dark_layer.py`):
   the supplement IS in the local PDF (`pages/p-15..21.png`); the dark
   renormalized-moment machinery now satisfies the paper's own identities
   as HARD asserts: (a) S47/S48 ansatz constraint E = 2 P0 Bd^2 rho
   (<1e-8 rel), (b) S104/S105 M = M_core + Omega*E with
   M_core = 2 P0(arcsin Bd - Bd sqrt(1-Bd^2)) (<2e-14), (c) S102
   Omega_tilde = -M/E (exact to f.p.), incl. Omega_tilde(0) =
   -M_core/E < 0 at zero Omega seeds (S107 third consequence).
   A REAL BUG was found and fixed in `dark_moments`: the momentum integral
   multiplied the phase gradient by the renormalization factor Gamma
   TWICE (S44 gives M = int (P-P0) phi' dtau = int P(1-P0/P) phi' dtau);
   the extra Gamma factor inflated |M| by ~2.9x for the Bd=0.9 deck and
   broke the S105 identity. The dark ODE overlay (S63/S73/S81/S97) and the
   S55 energy-decay sign assert stay RECORDED-OUTSTANDING: the dark pc-PDE
   deck shows the caveat-1 background-instability artifact family (E grows
   ~16x over xi=2 at the paper's parameters — the first-order-Raman
   high-w artifact, the dark analogue of the bright-side superseded note),
   so the S55 monotone-decay assertion is not safely assertable until the
   artifact family is tamed on the dark deck.
   **FOLLOW-UP UPDATE (2026-09-30, machine-ordinary session):** the dark
   ODE MACHINERY (S55/S63/S73/S81/S97 + all auxiliaries) is now fully
   transcribed and implemented in `diagnostics/dark_ode.py` — the clean
   pdftotext supplement export + 200-dpi page renders resolved the eta2/
   C-coupling ambiguities (see `diagnostics/DARK_ODE_TRANSCRIPTION.md`);
   visually-verified: eta_2 = Omega Q^2/Bd − rho*Omega, C4 =
   −(12P0/π²)[rho + (2−rho/3)/(Bd²−1)], coupling signs of S97 both
   subtract, S79's rho1 structure, and the printed-vs-derived M-pairing
   discrepancy (printed box S73 pairs sigma M1 + |s_gamma|tauR M2
   while its own derivation S69–S72 pairs |s_gamma|tauR→M1, sigma→M2).
   The machine implements BOTH variants behind a `variant` switch for
   numeric arbitration. The LONG-TIME dark-PDE overlay itself remains
   blocked by a now-diagnosed artifact family on a periodic grid: the
   dark ansatz is NOT periodic (u(±∞) = sqrt(P0)(±Bd + iQ) differs by
   2*Bd — an O(1) step discontinuity at the FFT seam), so any np.fft
   propagation smears the seam and corrupts the defect moments (measured
   dE/dxi ≈ +79 at dxi=1e-4 vs EXACTLY 0 analytic for sigma=0), on top
   of the physical MI of the cw background; single-application of the
   caveat-1 lowpass also demonstrably corrupts the field through the
   seam ringing. This is a simulation-methodology gap (the paper's own
   finite-region/trimming setup, cf. the printed "I_tau = cutoff
   integral over a finite region" in S76), NOT an equation error. The
   machinery is validated to the source by construction (Ω from S105, Bd
   from S48, no unknown inputs); remaining recorded-outstanding: a
   seam-clean dark-PDE deck (background-subtracted defect propagation
   or a non-periodic scheme) to run the S63/S73/S81/S97 overlay and the
   S55 decay assert.
4. Width/chirp flatness of Fig. 1(f)/(g): our rho drifts to ~2.96 and C
   to ~-0.56 by xi=10 (consistent with the moment ODEs), the paper
   shows flat; same-open-family as caveat 2.

## Supplement transcription (2026-09-30 session)

The full supplement IS in the local PDF (`pages/p-15..21.png`), so the
"S1-S22 auxiliaries missing" premise of caveat 3 is STALE: the paper's
supplement carries the dark-soliton derivation itself. Transcribed here,
valid target list for the dark layer (all Eq. numbers are the paper's):

- Dark ansatz Eq. (S47) (= main Eq. 29): u = sqrt(P0)[ Bd tanh((tau-eta)/rho)
  + i sqrt(1-Bd^2) ] exp[i phi - i Omega(tau-eta) - i C(tau-eta)^2/(2 rho^2)].
- Constraint E = 2 P0 Bd^2 rho (exact, all xi) — S48 window; renormalized
  moments Eqs. (S42)-(S46).
- dE/dxi = -(16/15)|sigma| tau_R P0^2 Bd^4/rho^3 (S55) — energy decay,
  Bd^4 blackness dependence.
- deta/dxi = s_D*eta1 - delta*eta2 - sigma*A1 - |sigma| tau_R eta3 (S63),
  with eta1 (S62), eta2, eta3, A1 (S60/S64). (RESOLVED 2026-09-30 by the
  pdftotext export + 200-dpi renders: eta_2 = Omega Q^2/Bd − rho*Omega,
  eta_1's glyph is eta; see diagnostics/DARK_ODE_TRANSCRIPTION.md. The
  S63-vs-S57 GVD-term inconsistency is exposed in dark_ode.deta_dxi's
  gvd_variant switch.)
- dM/dxi = sigma*M2 + |s_gamma| tau_R M1... — boxed Eq. (S73)
  `dM/dxi = sigma M1 + |s_gamma| tauR M2 + |sigma| tauR M3` (S70-S73),
  M1 = 16 P0^2 Bd^4/(15 rho), M2 = 4 C P0^2 Bd^2 rho/3 (1 - Bd^2/3) (S71),
  M3 = 32 P0^2 Bd^2 Omega/(15 rho) - 32 P0^2 Bd^2 sqrt(1-Bd^2)/15 (S72).
  (2026-09-30: the printed S73 box pairs sigma<->M1 while its own derivation
  S69-S72 pairs |s_gamma|tauR<->M1 and sigma<->M2; dark_ode.dM_dxi exposes
  both pairings for numeric arbitration.)
- d rho/dxi = -pi^2 s_D C/(12 rho) - delta A2 - |sigma| tau_R rho1 (S81),
  auxiliaries A2 (S76), A3 (S80), rho1 (S79).
- dC/dxi: boxed Eq. (S97) with C1..C5, A4..A10 (S87-S99); largest
  transcription risk. NOTE Eq. (S97) in the image shows
  `dC/dxi = -(C/E) dE/dxi - (M/E) deta/dxi + sD C1 + delta C2 + sigma C3
  + |s_gamma| C4 + |sigma| C5 + A8` — the leading minus signs of the two
  coupling terms must be re-verified; Eq. (S82) has `+ C dE/E + M deta/E`
  signs (both terms subtract); cross-checks:
  - Omega relation (S101)-(S106): Omega-tilde = Omega - M_core/E with
    M_core = 2 P0 (arcsin Bd - Bd sqrt(1-Bd^2)); Omega-tilde(0) = -M_core/E
    < 0 for a zero-Omega seed (S107 discussion).
- Omega-tilde vs Omega relation Eq. (S106) == main-text Eq. (36).

Cheap hard analytic checks implementable NOW (no PDE needed):
(a) S47 ansatz satisfies E = 2 P0 Bd^2 rho (algebraic identity).
(b) S103-S104: M(ansatz) = M_core + Omega*E by direct numerical quadrature
    of the ansatz (Darboux-identity check, machine precision).
(c) S102/S105: Omega-tilde = -M/E identity on the ansatz field.
(d) S55 energy-decay SIGN on a PDE integration (pc model, dark)
    + Bd^4 - scaling sensitivity (perturb Bd, compare decays).

The (a)-(c) tests close the *dark structural* layer; the dark ODE overlay
(S63/S73/S81/S97 machinery) stays recorded-outstanding until a clean PDF
text export is available or a vision double-check resolves the eta2/C-coupling
signs.

Runtime: full validation ~30 s on this box (12 dimensionless runs +
engine bench, all CPU). The folder is registered in the local-only test
suite (structural dark-layer test) and lives in `reproductions/`.

## Dark-ODE machinery status (2026-09-30, follow-up session) — CORRECTED

`diagnostics/dark_ode.py` implements the full printed dark-
moment ODE set (S55/S63/S73/S81/S97 with S60/S62/S64/S76/S79/S80/S87-S100
auxiliaries), state (E, eta, M, rho, C) all measured from the field via
`dark_moments`, Omega = (M − M_core)/E (S105) and Bd = sqrt(E/2P0rho) (S48).
Two printed-vs-interpretation ambiguities are exposed as alternatives
(`variant='printed'|'derived'`) for numeric arbitration against a clean
dark deck. The arbitration source remains recorded-outstanding: see the
seam-discontinuity diagnosis in `diagnostics/DARK_ODE_TRANSCRIPTION.md`
and caveat 3's follow-up above.
