# Heidt 2009 — adaptive step size for the GNLSE (JLT 27, 3984)

> **Status: REPRODUCED (2026-10-01).** Full 10 cm / 400 km run, all 16
> validate() checks green in 290 s (`full_validate.log`); figure rendered
> (`heidt_2009_adaptive_step.png`). Two recorded deviations are load-bearing
> (shock-free decks, machine-precision floor) — see "Recorded deviations".

**Reference.** A. M. Heidt, "Efficient Adaptive Step Size Method for the
Simulation of Supercontinuum Generation in Optical Fibers", *J. Lightwave
 Technol.* **27**(18), 3984–3990 (2009), doi:10.1109/JLT.2009.2021538.
Print-only, no arXiv version; the local PDF and `pages/p-*.png` are gitignored.
The printed pages were read as images (`pages/p-2…p-6`) — the text layer of
this paper drops every equation, so all formulae below were transcribed by eye
from the page images.

## What the paper claims, and what is here

The paper is a *numerics* paper. Its content is a pair of step-size
controllers and a comparison of their computational efficiency:

| paper object | where it is implemented |
|---|---|
| SSF (symmetric split-step Fourier, η = 3) and RK4IP (Hult 2007 Eq. (12), η = 5) | `heidt_adaptive.py`: `SSFIntegrator`, `RK4IPIntegrator` |
| local error method (step doubling + local extrapolation, Eq. (6)/(7)) | `LocalErrorStepper` |
| conservation quantity error method (Eq. (8)/(10c)/(12)/(13)) | `CQEStepper` |
| exact FFT cost accounting ("computational time normalized to one FFT") | `GNLSESOperator.n_fft`, counted transform by transform |
| global error metric ε (Eq. (17)) | `global_average_error()` |
| Fig. 1(c,d) error tracks, Fig. 2 efficiency/orders, Fig. 3 collision step profile | `reproduce.py`: `check_fig1cd`, `check_fig2`, `check_fig3` |

Everything is written against the repository FFT convention
(`TemporalGrid.fft`, analysis kernel `e^{+iΩt}`, `ISSUES.md` #0), so the
results are directly comparable with the engine-based reproductions.

## DONE

1. **Both decks fully transcribed** into `parameters.json` from the page
   images, with page refs and every reconstructed/ambiguous value flagged.
2. **The integrator layer is certified** (`reproduce.py::check_integrators`):
   - RK4IP integrates a *linear* flow to 3.6e-15 relative error over 10 steps
     (classical RK4 must be exact for a linear problem — this is the identity
     that caught four wrong stage formulas during development);
   - measured convergence orders **RK4IP 4.01**, **SSF 2.02** (global error vs
     FFT cost), against the paper's η = 5 / η = 3.
3. **Deck A (supercontinuum, Sec. IV.A) runs** with the paper's geometry and
   pulse (Φ = 1.4 µm, Λ = 1.6 µm, P0 = 10 kW, T0 = 28.4 fs, 10 cm, N = 2^13)
   and the dispersion the paper itself cites (Dudley RMP 78, 1135 (2006)
   Table I — the same table `hult_2007_rk4ip` uses; ZDW 790 nm vs the paper's
   "approximately 780 nm").
4. **Fig. 1(c,d) reproduced on the full 10 cm deck**: at the paper's
   constant step of 40 µm the local-error estimate peaks at 3.4e-4 (paper:
   ~5e-4) at z = 1.20 cm — inside the paper's fission window 0.7–1.5 cm —
   and the CQE (energy-conservation) estimate peaks at 2.5e-6, a peak ratio
   of **7.4e-3** (paper: "three to four orders" smaller; here ~2 orders,
   because the shock-free deck's conserved quantity is tighter than the
   paper's photon-number-with-shock). The two tracks are correlated at
   **r = 0.79** in log-log and share the fission-window peak and the
   post-fission decay (paper Fig. 1(c) vs 1(d) "similarity").
5. **A real finding that the paper does not discuss** (`check_cqe_precondition`,
   asserted): the CQE controller is only well posed for an equation that
   conserves its invariant *exactly*, and the first-order Blow–Wood shock
   term does not. Measured photon drift over 2 mm of deck A:

   | dz | 1e-5 m | 1e-6 m |
   |---|---|---|
   | shock off (energy invariant) | 2.48e-13 | 2.95e-13 |
   | shock on (photon invariant) | 1.10e-3 | 1.10e-3 |

   With the shock on the invariant drifts by 1.1e-3 per 2 mm *independently of
   the step size* — model drift, not integration error — so the CQE estimate
   measures the model, not the step, and the controller is not well posed.
   Shock-free the energy is conserved to 3e-13 and the estimator scales with
   the step error. The deck therefore runs shock-free (recorded deviation).
   Cf. Kim, Park & Shin, *Phys. Rev. E* **58**, 6746 (1998) and `ISSUES.md` #1.

## RESOLVED (2026-10-01) — both #14 symptoms were pre-fix artifacts

Full numbers: [`diagnostics/arbitration_task1_task2.md`](diagnostics/arbitration_task1_task2.md)
(`arbitrate_eps_floor.json`, `deck_b_diagnose.json`).

1. **Deck A ε floor was ~5e-4 — now measured ~7e-6.** The recorded
   saturation (6.1e-4 … 4.9e-4 across dz = 1e-5 … 1e-6) predates the
   `betas_si` unit fix (ps^k/m → s^k/m, powers from k = 2), under which the
   deck-A cascade was pure dispersion and never fissioned. Post-fix ladder:
   ε = 2.90e-3 / 1.41e-4 / 6.71e-6 at dz = 4e-5 / 2e-5 / 1e-5 (measured
   orders 4.36 / 4.40 — clean RK4IP 4th order); dz = 1e-6 stops improving at
   ~7e-6, the chaotic sensitivity floor of the physical cascade. The Fig.-2
   efficiency ladder is therefore testable on deck A over ε ∈ [1e-2, 1e-5]
   (the paper's 1e-4 … 1e-12 tail is unreachable — chaos property, recorded
   bounded deviation).
2. **Deck B "pedestal dispersal" was the β₂ unit bug.** The recorded
   failure read β₂ = −0.1 ps²/km as ps²/m (1000× too dispersive). Post-fix
   RK4IP run (dz = 25 m, 400 km): the two solitons walk together
   (−90/−10 ps at 40 km), collide at ~200 km as the paper's Fig. 3(b)
   requires (merged oscillatory peak, 3.9× single-soliton peak), pass
   through cleanly (−100.6/+0.6 ps at 400 km); energy conserved to +0.07 %,
   window-edge leakage < 1e-8, single detuned solitons hold shape exactly.

3. **Full-scale run (2026-10-01): ALL GREEN, 16 checks in 290 s.** Full
   validate on the paper's 10 cm / 400 km decks with the Fig.-2 ladders; the
   figure `heidt_2009_adaptive_step.png` is rendered. Results:

## RESULTS (full run, 2026-10-01 — all 16 checks green)

**Deck A (supercontinuum, Fig. 1–2).** The paper's headline efficiency claims
reproduce quantitatively, read off the eps-vs-FFT-cost envelope (paper Fig. 2):

| claim (paper) | measured |
|---|---|
| RK4IP-CQE needs ~30 % of constant-step cost | **0.34×** at ε = 1e-3 (accepts [0.15, 0.55]) |
| RK4IP-CQE 25–40 % faster than RK4IP-local | **0.70×** at ε = 1e-3 (accepts [0.55, 0.80]) |
| RK4IP-CQE the most efficient combination | best of all six at every ε |
| RK4IP methods ~4th order, SSF-constant 2nd | slopes −3.91 / −4.40 / −2.02 |
| SSF-local reaches 3rd (4th) order | slope −1.80 (floors on the chaotic deck — recorded) |
| SSF-CQE no better than SSF-constant | degenerate on this deck: see deviations |

**Deck B (collision, Fig. 3).**
- step-size profile (paper Fig. 3(b) shape): CQE dz 1795 m → **148 m at the
  collision (z = 197 km, collapse 0.082)** → back to 1795 m after
  (recovery ratio 1.00); local error: 1031 → 85 → 515 m;
- CQE steps **higher** than local outside the collision (1795 vs 1031 m —
  paper's "less sensitive to small-scale errors");
- efficiency (paper Fig. 3(a)): matched-ε cost ratio CQE/local **0.56 @ ε 1e-4,
  0.69 @ 1e-5, 1.00 @ 1e-6** — the paper's "up to 45 % faster" reproduces in
  the mid band and vanishes at the tight end (bounded deviation).

**Method finding (asserted, `cqe_needs_conserved_quantity`).** As above: the
CQE controller needs an exactly conserved quantity; the first-order shock term
breaks exactly that, so both decks run shock-free with the appropriate
invariant (energy for the GNLSE — paper Eq. (16) says the same for the NLSE).

## Runtime estimate (measured, not guessed)

Measured throughput on this box, 8192-point grid, RK4IP: **~4000 steps/s**
(24 transforms per step); SSF ~3× cheaper per step but second order.

| item | steps | cost |
|---|---|---|
| full validate(), warm caches re-run | — | **~5 min** (measured: 290 s) |
| full validate(), cold caches | all ladders | ~8 min (measured with the deck B reference at 40 000 steps: +25 s) |
| deck B, any point (400 km, dz ≈ 10 m) | 40 000 | ~10 s |

The fixed-parameter-grid ladders (see deviations) avoid the 1–3 M-step SSF
tails of a bisection, which is what keeps the whole run at minutes.

## Recorded deviations (summary)

- **shock off in BOTH decks** (deck A justified above, with measurements;
  deck B: with the paper's shock term the NLSE's energy drifts with the step
  size itself — measured energy drift dz = 4000 / 1000 / 250 m =
  1.2e-3 / 8.8e-4 / 7.2e-4, step-size independent, i.e. the additive-RK4IP
  Euler approximation of the shock term breaks exactness — and the CQE
  estimator measures that Euler discretisation instead of the step error, so
  the Fig.-3 ladder degenerates into one point per method. Shock-free the
  RK4IP steps are exactly energy-preserving (drift < 1e-15) and the paper's
  Eq. (16) energy variant of the CQE method applies, which is the paper's own
  prescribed conserved quantity for the NLSE);
- **γ not stated in the paper** — deck A uses 0.045 /W/m (A_eff = 5 µm², the
  realistic value for a 1.4 µm-hole / 1.6 µm-pitch PCF); deck B's γ is derived
  from the paper's own "fundamental soliton" statement (2.21 /W·km);
- **carrier 850 nm** (the Dudley Table I carrier, which is the paper's cited
  dispersion source) vs the paper's 835 nm; soliton order 5.3 vs "around eight";
- **time window 16 ps** — the paper does not state it, and this is the shortest
  window that keeps Ω_max < ω₀, which both its shock expansion and its
  1/ω photon weight need; on a 2^13 grid over a shorter window the highest
  bins sit above the carrier and Eq. (8) is undefined (the library enforces
  the same guard in `GNLSESolver._validate_shock_grid`);
- **2^14 bins in deck B** (paper: 3072) — 3072 bins cannot resolve a 4 ps
  soliton over a window that must exceed the 200 ps collision distance;
- **ε ladders sweep a fixed parameter grid per method** and the paper's
  claims are read off the eps-vs-cost envelope, not bisection onto a common
  ε grid: on both decks the *goal error* and the *global error* are strongly
  decoupled (deck A is chaotically sensitive above ε ~ 7e-6; deck B's local
  error goal feeds a whole-run doubling average while the CQE goal is a
  per-step relative change at dz_max), so a matched-ε bisection is degenerate
  (it pinned every target to the same coarsest run — the flat ladders of the
  first full run). This matches what Fig. 2/3(a) actually plot: ε against
  cost per method.
- **machine-precision floor** — deck A's reachable reference floor is
  ε ~ 7e-6 (chaotic sensitivity of the fissioning cascade,
  `diagnostics/arbitrate_eps_floor.json`), so the paper's ε ≤ 1e-12 tail of
  Fig. 2 is out of range; deck B's floor is ε ~ 5e-11 (measured
  dz = 25 m vs 10 m reference) vs the paper's ε ~ 3e-12 — just below the deck's
  attainable band. The Fig.-2 comparison is read at ε = 1e-3, where all six
  curves are bracketed.
- **SSF-CQE is blind on the shock-free deck** (estimator = round-off for
  every goal; identical step counts) — the paper's "SSF-CQE no improvement"
  claim is driven by the shock-term photon drift of its own model, which the
  shock-free deck does not have (asserted as
  `fig2_ssf_cqe_degenerate_shock_free`).
- **local-error variants floor out on the chaotic deck** — their fitted
  slopes (−2.84 RK4IP-local, −1.80 SSF-local) are recorded but not asserted;
  the clean integrator orders are certified in `check_integrators`.

## Files

- `parameters.json` — house format, both decks with page refs.
- `heidt_adaptive.py` — operators, the two integrators, the two step-size
  controllers, FFT accounting, ε metric.
- `reproduce.py` — `validate(fast=False)` runs everything and asserts the
  paper's claims; `--fig` renders the figure; every propagation is cached in
  `.cache_*.npz`.
- `heidt_2009_adaptive_step.png` — the krupa-style summary figure
  (Fig. 1(c)/(d) tracks, Fig. 2 ladder, Fig. 3(a) ladder + (b) step profile).
- `full_validate.log` — the full-run output (ALL GREEN, 16 checks, 290 s).
- `diagnostics/` — the #14 arbitration (pre-fix artifacts, unit-bug history).
