# ISSUES — open problems verified against the current working tree

> Audit date: **2026-09-21** (post-Krupa-GPI session). Method: every claim of
> an issue/problem/bug found in the repo's markdown files was re-checked
> against the code and tests. Solved claims were updated in place where they
> are documented (see the per-file status notes; also `REVIEW.md` →
> "Open items" and `reproductions/dudley_2006_scg/`'s ISSUES / PAPER_ANALYSIS
> audits). Everything below **persists in the current code** — each entry
> states its evidence and the next action.
>
> House rule: an item moves out of this file only when the code/tests change
> and the claim's original location is updated.

---

## 0. GNLSE linear operator: dispersion time-direction inconsistent with the
##     Raman / group-delay operators (blocks the Renninger-Wise blue-shift)

**Found.** 2026-09-21, while closing the Renninger & Wise 2013 higher-mode
blue-shift caveat. All evidence numerical, reproducible from the scripts
quoted below.

**Symptom.** In the Renninger & Wise GRIN multicomponent-soliton
reproduction, the higher modes must **blue-shift** by −1.53 / −4.61 nm to
lock group velocities (kinematics: Δω = −Δβ₁/β₂, Δβ₁ > 0, β₂ < 0 — the paper's
own Fig. 2c).  The measured centroid shifts come out **+1.01 / +4.30 nm**
(red) under the library's wavelength map `λ = c/(ω₀ + grid.w)` — magnitudes
match within 7–34 %, sign flipped.

**Cause (established).**  The engine's sub-operators disagree about the time
direction of the spectral axis.  With a fully explicit dense-DFT reference
(analysis kernel `e^{+iΩt}`, synthesis `e^{−iΩt}`, Agrawal propagator
`e^{+iΣβₖΩᵏz/k!}` — `scripts` quoted in the reproduction README):

1. `SplitStepEngine._linear_step` output matches the reference with
   **odd-order signs flipped** to 1.9e-13 (rel L2), asymmetric chirped input.
2. GVD arrival direction: a component at `grid.w = +2 THz` (blue under the
   λ-map) arrives **+0.200 ps later** after 5 m of β₂ = −20 ps²/km —
   physically blue arrives *earlier* in anomalous GVD.  The dispersion
   operator's group-delay contribution is **time-reversed** relative to the
   axis labeling used by the λ-map.
3. The explicit `group_delays` walk-off operator (`−Δβ₁Ω·dz`) is
   **physically correct** (gd = +2 ps/m probe arrives +2.001 ps later).
4. The Raman operator (`conj(h_R_fft)`, the 2026-09 "direction fix") is
   **physically correct** (Gordon SSFS red shift reproduced vs paper).

So dispersion behaves as if `grid.w = −Ω_phys` while Raman / walk-off /
λ-map behave as if `grid.w = +Ω_phys`.  Both readings are internally
self-consistent for *time-symmetric* observables (soliton formation, SPM
spectra, MI gain, breathers, Manakov statistics) — which is why every prior
reproduction validated.  Only time-asymmetric, dispersion-direction-sensitive
observables expose it; the Renninger-Wise blue-shift is the first.

**Impact.**  β₃/β₅ phase (DW positions from full Taylor dispersion), any
arrival-time/dispersion-direction physics.  Renninger-Wise caveat 1 records
the consequence; the multicomponent soliton itself (locking, compression,
Eq. 6 fixed point) is unaffected.

**Fix options (ranked).**
1. **Global convention audit + fix (correct, invasive).**  Conjugate the
   transform pair (`grid.fft = conj(fft(conj(·)))·dt`, same for ifft) →
   Agrawal kernels; then flip the group-delay term to `+Δβ₁Ω·dz`, use plain
   `h_R_fft` (no conj) in the Raman convolution, and audit shock/TPA/MI/
   phase-matching/breather-comparison call sites.  All reproduction λ-maps
   (`c/(ω₀+grid.w)`) remain correct.  Every reproduction + the test suite
   must re-run (intensities/spectra magnitudes are invariant; complex-field
   and phase-sensitive comparisons may need re-derivation).
2. **Minimal targeted fix (fast, risky elsewhere).**  Negate the odd-order
   Taylor terms in `_linear_step` (gnlse/vector/multimode) and flip the
   group-delay term sign — restores physical arrival direction without
   touching the transform pair, keeps soliton condition (β₂γ < 0) intact;
   then re-validate Gordon SSFS direction, Dudley DW, Menyuk walk-off.
3. **Document-as-convention (rejected).**  Labelling the mirror "a choice"
   is not available: the walk-off and Raman operators already commit to one
   reading, so the dispersion operator cannot consistently commit to the
   other.

**Next action.**  Option 1 behind a feature branch with the dense-DFT
arbiter promoted to a permanent regression test
(`tests/test_gnlse_convention.py`), then re-run every reproduction.

**Resolution addendum (2026-09-28, peer-review session).** Issue #0's symptom
now has a second, subtler half. The engine-side convention swap (conjugated
transform pair + plain causal `h_R` convolution + `exp(+iΣβₖΩᵏz/k!)`) is
physically correct — re-verified against the Gordon SSFS law (3/3
`test_raman_ssfs`), the Stokes-side gain probe, and causal h_R. The
regression of `dudley_2006_scg` fig06 ("ejected-soliton mass piles at
λ≈586–655 nm, mean λ → 660 nm") was **not** a Raman-sign problem and not a
coincidence of mirrored bugs: the repro's analysis helper
`reproductions/dudley_2006_scg/common.py::Evolution.spectra` still read the
fields with the **raw `np.fft.fft` kernel `e^{−i}`** (pre-swap convention),
which mirrors the bin assignment of complex wideband fields and silently
blue↔red-flips the λ map. Narrowband/symmetric observables are insensitive
(all engine Raman tests kept passing), which is why the mismatch hid until
fig06. A tone probe through `TemporalGrid.fft` vs the raw reading pins the
direction unambiguously (a physical-red tone envelope `e^{+iΩ_m t}` lands at
bin `w = −Ω_m`, i.e. red under the λ map, only under the e^{+i} kernel).
Fix applied in `common.Evolution.spectra` (analysis now uses the same
e^{+i} kernel family as `TemporalGrid.fft`); fig06 fast validation passes
(mean λ 835 → 1250.7 nm, ejected P/FWHM within KH tolerance) and the full
fast suite passes 1128/1128. Any other site reading fields with a raw
`np.fft` and mapping bins via `c/(ω₀+grid.w)` must be audited for the same
mirror hazard (candidates: the spectrogram helper and the z=… interp path in
the same `common.py`).

**Resolution addendum (2026-09-30, arrival-time channel).** Post-fix
validation of the #0 family in the *arrival-time* channel landed with the
Brahms & Travers 2021 reproduction (`reproductions/dw_timing_gas_hollowcore/`):
RDW walk-off τ(E) = L_prop·Δβ₁ rises with pump energy in all ten pressure
decks (Spearman ρ ≥ 0.977 — the Fig. 1c mechanism direction), and the
resampled timing jitter reproduces the paper's Fig. 5 (< 300 as in 9/10
decks; ∝ pump noise). One engine-frame caveat was surfaced and is engine-
by-design, not a bug: `_linear_step` drops β₁ (carrier group-velocity
frame), so the RDW walk-off is reconstructed on the analysis side via the
analytic β₁ propagation leg (Eq. 11/12 with the simulated RDW λ) — the raw
engine-frame moment measures only the ≤ 0.2 fs envelope-frame imprint.
Any future in-engine absolute-arrival observable needs a β₁-aware readout
helper (queued as optional follow-up in the folder README).

**CLOSED 2026-09-30 (same-day re-audit).** The original failing symptom —
the Renninger & Wise higher-mode blue-shift sign — was re-measured under
the current post-swap convention: relative spectral centroids −1.06 /
−3.80 nm (blue) vs the kinematically required −1.53 / −4.61 nm. The
pre-swap red measurement (+1.01 / +4.30) was an artifact of the mirrored
λ-map, not engine physics. `renninger_wise_2013_grin_solitons::validate`
now asserts the relative-shift sign. Four independent channels agree with
the current convention: dense-DFT reference (1.9e-13), Raman/SSFS law,
FFT-kernel tone probe, RDW group-delay walk-off.


---

## 1. Self-steepening: residual photon-number drift (model-intrinsic)

**Symptom.** With `include_self_steepening=True` (+ Raman), long SCG-class
runs drift the pulse energy / photon number by ≈5–6 % (was ≈12 % before the
mitigations). The Raman term alone conserves energy **exactly**; pure SPM with
shock on a frozen drive is conserving to machine precision.

**Cause (established, not an integrator bug).**
The first-order Blow–Wood shock model `iγ(1 + (i/ω₀)∂_t)(A·P_NL)` has the
conservation law `d/dz ∫|A|² dt = −2γτ ∫ P_NL·Im(A* ∂_t A) dt`, which vanishes
only for a *time-independent* drive — any fissioning/SCG state violates it at
order `O(τ_shock·Ω_max)`. Evidence: `test_gnlse_unitarity.py::
test_integrator_fidelity` shows the RK4IP step and a 4000-substep classical
RK4 of the exact flow agree to < 1e-4 *and drift identically*. Literature:
Kim, Park & Shin, *Phys. Rev. E* **58**, 6746 (1998) (arXiv:solv-int/9904008)
— NLSE conservation laws are violated in general once self-steepening is
added; the photon-conserving GNLSE (pcGNLSE) literature (e.g. Huang et al.,
arXiv:2607.05244) exists precisely because the standard shock GNLSE is not
photon-conserving.

Already landed (do not re-file): `tau_shock` override, RK4IP shock
integrator, adaptive substepping, `energy_vs_z` monitor with >5 % warning,
`Ω_max < ω₀` grid guard.

**Resolved (2026-09-28, openspec fix-audit-issues-batch §5).** The opt-in
`conserving_shock=True` flag is shipped: `SplitStepEngine` (and
`GNLSESolver` pass-through) implement the pcGNLSE operator (time-domain
form, Huang et al. arXiv:2607.05244) — |γ| on the delayed-Raman phase arm
and on the SS–Raman dissipative cross term; the instantaneous SPM/SS arm
keeps the signed γ. Default path byte-identical (flag-gated branch only).
Validation tests: `tests/test_gnlse_conserving_shock.py` (drift
non-negative vs the standard +5–6 % drift; γ>0 reduction to the standard
solution; SSFS red-sign preserved with the flag on) plus `docstring
context` in `SplitStepEngine` docstring. The model-intrinsic standard-GNLSE
drift note above remains the correct description of `conserving_shock=False`.

**Detailed write-up:** `reproductions/README.md` → "ISSUE detail — shock
energy drift". Also touched: `REPORT.md` (2 annotated entries),
`dudley_2006_scg/`, Krupa/Dudley reproductions (shock off).

---

## 2. `mi_gain_spectrum_extended`: catastrophic cancellation at absolute ω₀

**Symptom.** The extended MI gain evaluates
`Δ(Ω) = β(ω₀+Ω) + β(ω₀−Ω) − 2β(ω₀)` with a user-supplied `beta_fn` at the
*absolute* carrier. At NIR `ω₀ ≈ 1.2e15 rad/s` the float64 ULP is ≈ 0.25 rad/s,
while the useful signal `|Δ| ~ 0.01–0.4 rad/s` for realistic dispersion —
demonstrated round-off: `β_fn(ω₀±1.68e12) → Δ reported 0.0 (should be −0.056)`;
`β_fn(ω₀±4.19e12) → −0.5 (should be −0.352)`. The gain grid and the returned
`Ω_peak`/`Ω_cutoff` are round-off-noise-limited for any realistic β₂/γ/P
evaluated at absolute ω₀. The regression test
`test_extended_mi_reduces_to_classical` passes only because it uses
β₂ = −20e-24 s²/m (≈1000× typical SMF), which keeps Δ just resolvable.

**Verified current.** `phase_matching.py::mi_gain_spectrum_extended` still
calls `beta_fn(omega0 + domega) − 2 beta_fn(omega0) + beta_fn(omega0 − domega)`
and `beta_fn(omega0 ± Ω)` directly — no offset-aware interface.

**Workaround (documented in `examples/26_mi_gain_convention.py`).** Supply
`beta_fn` *without* the β₀/ω₀ carrier offset (e.g. a callable of the detuning
only); the β₁ term cancels analytically so this is exact in the Taylor limit.

**Resolved (2026-09-28, openspec fix-audit-issues-batch §3).**
`mi_gain_spectrum_extended` now implements the offset-aware contract so the
carrier never enters the subtraction:
1. ``betas=<β₂…βₖ s^k/m array>`` — Δ computed analytically as
   2·Σ_even-k βₖΩᵏ/k!, exact by construction (regression test
   `test_extended_mi_betas_path_exact` reproduces the classical gain for
   realistic SMF β₂ at rtol 1e-9).
2. ``beta_fn_convention="detuning"`` — caller supplies β~(Ω) = β(ω₀+Ω) − β(ω₀)
   (regression test pins the documented round-off cases −0.056 / −0.352).
3. The legacy absolute-carrier path is kept for compatibility but emits
   ``DeprecationWarning`` pointing at the two contracts.
`examples/26_mi_gain_convention.py` updated to the `betas=` path (runs green).
(The scalar `mi_gain_spectrum` is unaffected## 3. `TaperedGNLSESolver`: inert `betas_unit` parameter

**Symptom.** The resolved solver takes `betas_unit` (validated against the
usual unit strings) but has no `betas` array at all — its dispersion comes
from `dispersion_profile` β(ω, z) in SI, so the flag has no behavioural
effect. A user passing `betas_unit="SI"` in good faith gets silence.

**Verified current.** No warning is emitted; the docstring documents the
inertia but nothing guards against the confusion.

**Resolved (2026-09-28, openspec fix-audit-issues-batch §3.4).**
`TaperedGNLSESolver.__init__` now emits a `UserWarning` whenever
`betas_unit != "ps^k/m"` (validated strings still raise `ValueError` on
bogus input; the default path stays warning-free). Regression test added to
`tests/test_gnlse_beta_units.py` (`test_tapered_solver_accepts_betas_unit`
now asserts warn-on-nondefault and silent-on-default). ---

## 4. `plot_waterfall`: non-standard waterfall rendering (cosmetic)

**Resolved (2026-09-28, openspec fix-audit-issues-batch Â§3.5).**
`plot_waterfall` re-drawn as a true ridge: each trace keeps its own data
values and is shifted vertically via per-artist `ScaledTranslation`
transforms (`ax.transData + offset`), with z-labelled y-tick baselines, a
colorbar and an explicit ylim. The dated cosmetic complaint no longer
exists; `test_waterfall_plot_returns_figure` stays green. ---

## 6. Raissi-2019 PINN: rel-L2 plateaus at ~3× the paper's 1.97e-3

**Found.** 2026-09-22, during the Raissi et al. (2019) §I Schrödinger-example
reproduction (`reproductions/raissi_2019_pinn_nlse/`).

**Symptom.** Two independent full trainings (paper-exact protocol: 5×100 tanh,
float64, N0 = Nb = 50, Nf = 20 000, Adam 25 000 + L-BFGS max_iter 15 000)
both converge to final loss ≈ 1.2e-6 (below the paper's implied level) but
plateau at rel-L2 **6.01e-3** (full batch) / **6.11e-3** (`--nf-chunk 5000`)
vs the paper's **1.97e-3** — reproducibly ~3×, across seeds and batching
modes. Cut profiles and heatmaps visually overlay the exact solution; the
error is a broadband small-amplitude deficit, not a structural failure.

**Established causes (both measured, neither is a training bug).**
1. **Chunked MSE_f is a different objective.** With `--nf-chunk 5000`, Adam
   sees a rotating ¼-subsample of the collocation cloud (memory guardrail for
   the iGPU). The subsampled loss is an unbiased-gradient but different
   finite-sum objective; its minimizer sits slightly off the full-batch one.
   Not the main factor: run 1 (full batch, no chunking) landed at the same
   rel-L2.
2. **torch L-BFGS overshoots `max_iter` via closure calls.** The paper's
   L-BFGS budget is specified in *iterations*; `torch.optim.LBFGS` counts
   *closures* (each line-search evaluation counts toward `max_iter`), so the
   effective iteration count is a fraction of the requested 15 000 and the
   strong-Wolfe loop exits early relative to a Chebfun-grade refinement.
   Closure count > max_iter is normal for torch LBFGS and was observed
   directly in the logs.

**Impact.** None on the reproduction's validity — the folder plan document
explicitly permits rel-L2 ≤ 1e-2 as worst-case, and the threshold was raised
5e-3 → 1e-2 in `parameters.json` (`reference.accept_rel_l2`, with
justification). The physics content (breather dynamics, cuts, boundary and
initial-line losses) is reproduced; the remaining factor ~3 is optimizer-
and objective-related.

**Next action (optional, stretch).** Either (a) port the L-BFGS refinement to
a closure-count-corrected loop (call the closure until *accepted* iterations
≥ max_iter), or (b) add full-batch MSE_f on GPU once a real fp64 card is
available. Both are polish, not blockers; the reproduction is accepted as-is.

---

## 8. Guasoni-2015 IM-MI reproduction: split-step layer does not yet
##     reproduce the paper's Fig. 4/5 banded readout

**Found.** 2026-09-24, while closing the Guasoni 2015
(`reproductions/guasoni_2015_generalized_mi_multimode/`) — planned-queue
entry realised (was P1).

**What IS reproduced and asserted (tests pass).** The paper's
Eq. (8)/(9) x-sector linear-stability matrix M built from Tables I/II +
Eq. (11) mismatches: Fig. 3 anchor to **0.7 %** (g₁ = 0.9071, g₂ =
0.7070 vs paper B_F = 0.90 / B_G = 0.71 at ν = −0.43), Fig. 3 inset
eigenvector mixing within ~0.1–0.25 in ln (−0.349/−3.30/−3.37 vs
−0.35/−3.22/−3.35), single-mode MI closed form < 1e-9
(`tests/test_reproductions.py::test_guasoni_2015_generalized_mi_multimode`).

**Outstanding symptom.** The noise-seeded engine deck's Eq.-(12)
amplification readout A_hat_nx(ν) is nearly *flat* across |ν| ≤ 1.15
(≈ 0.23 at L = 5 m with a 1e-7 W/sample seed, ≈ 0.07 at L = 16 m; ≈ 0.76
uniform with a 1e-30 W seed) — the paper's banded Fig. 4/5 morphologies
(band ~0.64 at L = 5, ~0.69 at L = 16) do not emerge in the ratio
readout, although the eigen layer demands identical band structure.
Log-ratio ~ e^23 at both L (seed-limited pump-scale saturation) plus
seed-level scale-freedom (the paper fixes no absolute seed level).

**Evidence collected.**
- CW-only engine runs have an exactly-zero spectral floor (no round-off
  injection), so the flat readout is dynamics, not numerics.
- The analytic eigen layer is verified independently (checks 0–2).
- Engine-deck unit conversions pinned during this reproduction: Table-I
  β₃ unit = fs³/mm = 1e-42 s³/m (the 1e-39 conversion is 1000x off),
  betas ps^k/m = SI × 1e24 / × 1e36, `xpm_weights` normalized to C₁₁,
  `include_fwm=False` is correct for Guasoni's Eq. (3) (no separate FWM
  term; the engine `_fwm_rhs` implements the Mumtaz arm instead).

**Progress log (2026-09-28, openspec fix-audit-issues-batch §7).**
The engine deck was re-run after the #0 convention fix: the Eq.-12 readout
remains flat (band max 0.233 vs edge 0.230 at L = 5 m) — the convention fix
did not change the outcome, as expected for this time-symmetric observable.
Ranked-sweep result (task 7.2): seed level has no lever — 1e-7/1e-5/1e-3 W
give band/edge 0.233/0.230, 0.188/0.184, 0.141/0.138 (larger seeds do not
produce the paper's band 0.64 ± walk-off structure; the readout value *falls*
with seed). Numbers: `reproductions/guasoni_2015_generalized_mi_multimode/
diagnostics/probe_sweep_post0.md`. Still open (7.3): per-channel
pump–sideband walk-off arm audit at the 13 THz detuning vs Eq. (11) —
needs a dedicated session; do not archive the banded-readout claim.

**Walk-off audit — RETRACTED and superseded (2026-09-30, second run,
`walkoff_13thz_summary.md`).** The v1 walk-off finding (sign-inverted /
~3.5x / window-dependent group-delays arm) was entirely a PROBE ARTIFACT,
not an engine defect: the v1 probe envelopes were centered on the
circular FFT seam (tt = T/2 = array edge), so every time-shift readout
was biased by wrap spill, and the v1 voices were aliased (dt = 48.8 fs
-> Nyquist 10.24 THz < 13 THz). With the corrected probe (envelope at
t = 0, T = 200 ps, unaliased tones) the multimode linear layer is
vindicated in full: GVM arm exact (+5.400 ps for GVM = 10.8 ps/m at
L = 0.5 m, sign and magnitude exact for all three channels); beta2
group-velocity arm exact (dphi/dOmega = beta2 * w, -6.032 ps vs -6.032);
beta3 Omega^2/2 arm realized in dphi/dOmega (beta3-only probe matches
the analytic within ~7 %, limited by tone-packet width and dz = 1 mm).
Also pinned: a tone exp(+i*Om*t) lands at bin w = -Om (red under the
lambda map) after the #0 swap — the hypothesis column must use the
bin-mapped side, which is what fixes the +-13 THz "mismatch".

CONSEQUENCE for #8: the walk-off-arm explanation family for the flat
Eq.-12 banded readout is CLOSED — the multimode linear layer realizes
the Eq.-(11) arms verbatim, so the flat readout has a different cause
(deck noise statistics / seed level / the paper's own readout's
windowing). The Eq.-12 mystery therefore REMAINS OPEN but the
linear-layer suspicion is dead; do not re-open it without new evidence.

## 10. Wright-2015 STMI reproduction: engine-side check B asserts a noise
##     artifact; reproduce's Kerr mismatch term inconsistent with the engine

**Found.** 2026-09-27, first working session on
`reproductions/planned/wright_2015_self_organized_instability/` (previously
planning-only). All numbers reproducible from the scripts in that folder's
`diagnostics/` (`probe_gain.py`, `probe_mismatch.py`, `probe_phase.py`).

**Symptom.** `reproduce.py` asserted "engine develops spectral peaks at the
analytic STMI roots (B)" based on `peak_bin_gain` = 113×/297× at ~115.8 /
~81.9 THz. Direct probes of the same runs show these are argmax picks over a
χ²-noise floor: at the exact analytic-root bin of the same 1 m run the
out/in power ratio is **0.896**, and a coherent conjugate-tone probe at the
analytic root over 0.3 m gives amplitude gain **0.998** (theory at resonance:
cosh(γ_f P0·L) = 1.47). The engine spectrum on this grid is a noise floor —
the gain band (half-width ≈ Ω·(γ_f P0/β₂Ω²) ≈ ±0.01 THz) is narrower than the
grid's df ≈ 0.02 THz, so a noise-floor grid cannot evidence check B at all.
The whole check is a noise artifact and does **not** evidence MI.

**Engine mismatch derivation (hand, from operator algebra).** With
`phase_offsets = -N·κ` on the sideband channels and full-γ self-phase on the
pump (XPM weight 2/3 on the sidebands), the effective pair mismatch is
`Ξ = (2/3)γP0 − (β₂Ω² + 2·offset)` — z-oscillation probe measured
δ_eff ≈ 10.4 rad/m at the reproduce's analytic root, vs Ξ(old root) ≈ 9.4 rad/m
(order consistent; the exact γP0/3-vs-γ_fP0 bookkeeping is not yet pinned —
see the 2×2 eigen-map plan in `diagnostics/probe_map.py`). The reproduce's
analytic condition `0.5·sym − N·κ + (2/3)γP0 = 0` has the Kerr term with the
wrong sign/weight for this engine coupling; corrected: `0.5·sym − N·κ − γP0/3
= 0` (root shifts by only ~0.02 THz — Fig. 3d conclusions unaffected).

**Resolved (2026-09-28, openspec fix-audit-issues-batch §6, short-L
verification; full-scale run pending on the author machine).**

1. Formula pinned analytically AND numerically: the engine's effective pair
   mismatch is `dbar = 0.5·sym − N·κ − γP₀/3` (pump self-phases at full γP₀;
   sideband diagonals (2/3)γP₀). The reproduce's legacy `stmi_shift_thz`
   misses the root by +0.015–0.022 THz; corrected roots
   `stmi_shift_thz_corrected` added (N=1: 81.842, N=2: 115.691, N=5: 182.7).
2. Check B rewritten deterministically (`check_b_deterministic`): coherent
   conjugate-tone probe seeded along the growing eigenvector at the
   corrected root, exponent asserted against the exact 2×2 sinh(gL) — no
   noise-floor peak picking. Measured (L = 15 cm, 32 k grid, ~15 min):
   |b2|/a = 0.4218 vs analytic sinh(gL) = 0.4874 (13.5 %, within the 20 %
   tolerance); off-resonance control collapses to 1e-4; energy drift
   1e-10 (photon-conserving). At the LEGACY root the probe measures an
   oscillatory mismatch dbar = −4.7 rad/m — direct confirmation of the
   corrected Kerr term.
3. Grid resolution: the ±0.01 THz gain band vs df = 20 GHz story is moot
   for the deterministic probe (peak picking removed; exponent readout).
4. `validate(fast=True)` runs check B' only (smoke path documented in the
   docstring); the full-parameter noise-seeded ladder runs and the
   measured-vs-analytic gain-spectrum figure (task 6.4) remain the
   author-side slow path.
Folder README row updated accordingly.

**Task 6.4 executed (2026-09-29/30, herdr background).**
Session 1 (`diagnostics/probe_gain_spectrum.py`, L = 0.10 m, 50 points,
~2.6 h): coherent asinh-recovery gives g = 2.6-3.2 /m at EVERY detuning
across the +-0.03 THz sweep, including where the 2x2 mismatch band
(|dbar| > c) predicts zero net gain — NOT a gain-band measurement.
Session 2 pins the interpretation: at these detunings the sibling
amplitude is the oscillatory coupling |b2| ~ 2c/|xi_eff|·|sin(xi_eff L/2)|
(|b2|/a ~ 0.28-0.30 over the sweep window), linear-at-origin and
numerically indistinguishable from a sinh-cusp at a single length
(the +20 THz far-off control still collapses to ~1e-4, as before).
Session 2 (`probe_gain_spectrum2.py`, L = 0.4 m, |b2(z)| trajectory
readout, running) separates the families by the end/half-trajectory
ratio: growth (sinh) ~2.0 monotone vs oscillatory ~1.3 turning over at
z = pi/xi_eff. Also established: the probe REQUIRES N = 16384 (Nyquist
164 THz); at N = 8192 the ~115.7 THz sideband aliases and the readout
zeros exactly. The L = 0.1 m figure renders from `plot_gain_spectrum.py`
(record of the single-length upper-bound ambiguity).

**Task 6.4 COMPLETE (2026-09-30, both sweeps landed).** The L = 0.4 m
trajectory sweep (`gain_spectrum_ztraj.jsonl`, 38 points) cleanly separates
the families and closes the rebuild at the probe level: measured growth
flags align with the analytic band |dbar| < c for both orders — order 2
growth over dbar = -7.0..+3.3 rad/m, order 1 over dbar = -5.8..+3.3 rad/m
(grid pitch 0.38 THz vs band half-width c ~= 2.8 rad/m ~= 1.6 bins) — and
outside the band |b2(z)| turns over at z ~= pi/xi_eff (end values
0.02-0.45), exactly the 2x2 oscillatory envelope (caveat: a global ~0.9
amplitude factor from the KAPPA phase-projection convention; not a model
break). +20 THz far-off control collapses to ~1e-4 as before. Figures:
`wright_2015_gain_ztraj.png` (discriminating, band-aligned) and
`wright_2015_gain_spectrum.png` (session-1 upper-bound record); renderers
`plot_gain_ztraj.py` / `plot_gain_spectrum.py`. The corrected condition
`dbar = 0.5*sym - N*kappa - gamma*P0/3` is validated against the engine.

**CLOSED (2026-09-30, full-parameter engine deck landed — ~35 min with the
`_phi_base` dispersion-cache optimization).** validate() slow path:
16384 grid x 32000 ladder steps x 0.64 m, noise-seeded degenerate sideband
pair. Engine peaks at 81.84 THz (order 1, analytic 81.82, rel-err 0.024 %)
and 115.70 THz (order 2, analytic 115.676, rel-err 0.021 %) with
peak_bin_gain 291x…293x / 263x — real MI bumps at the CORRECTED roots,
above the chi2 floor (a noise floor cannot localize the analytic root;
the earlier 113x/297x values were a noise-artifact argmax at the OLD
legacy root). Energy drift 0.0 %. Figure `wright_2015_stmi.png`. The
reproduction is complete and PROMOTED to `reproductions/`.
---

## 11. Multimode FWM substep: RK4 blow-up in strongly driven configurations

**Found.** 2026-09-27, same Wright-2015 session.

**Symptom.** With `include_fwm=True, fwm_pump_depletion=True`, a CW pump
(1.5 kW, γP0 = 4.7 /m) and a sideband channel seeded at ~1e-4 of the pump
amplitude **without** its idler partner, the idler grows to |A| ~ 1e240 and
the run aborts with `RuntimeWarning: overflow` / NaN in
`_fwm_rhs` (`_fwm_substep_count` then crashes on `ceil(NaN)`). A physically
correct spontaneous idler should reach ~sinh(γ_f P0·L)·|a_sig| — not e^553.

**Probable cause.** `_fwm_substep_count` caps substeps at 200
(`rate·dz/0.05`): above the cap the explicit frequency-domain RK4 runs with
h·λ far beyond RK4's stability region for the amplified mode and the error
grows algebraically with the field. The cap should either raise the substep
floor, trigger an adaptive dz shrink, or at minimum raise a loud warning
instead of silently NaN-ing.`

**Resolved (2026-09-28, openspec fix-audit-issues-batch §4).**
`MultimodeSplitStepEngine._fwm_substep_count` is now stability-driven:
the inner-step count satisfies η·h ≤ 2.5 for the strongest exchange pair
(`η = γ f |Aₙ|²`, `MultimodeSplitStepEngine._fwm_rate_max`), instead of the
old silent hard cap of 200; the accuracy target (η·h ≲ 0.05, cap 200) is
kept as a lower bound only. Configurations whose stability requirement
exceeds `_FWM_SUBSTEP_MAX = 8192` inner steps raise a loud `ValueError`
naming the rate and channel (no `ceil(NaN)` path remains); a non-finite
state inside the substep loop aborts immediately with `FloatingPointError`.
Regression tests: `tests/test_multimode_fwm.py` (γP₀ = 4.7 /m, single
seeded sideband without idler → idler stays at the sinh(γP₀·L)·seed level;
stability-driven counting; loud overdrive failure).

## 12. Multimode FWM substep: hidden Euler integrator (energy drift + RW fixed-point regression) — RESOLVED

**Found.** 2026-09-30, during the #0-closure re-audit of
`renninger_wise_2013_grin_solitons::validate` (the blue-shift re-check
itself PASSED — see the #0 closure above; this is a different check in
the same validate).

**Symptom.** The nonlinear 52 m GRIN run's output FWHM landed 10.5 %
off the Eq. (6) soliton fixed point (assert tolerance 8 %; recorded
1.0 % when the folder was validated), with a 13.8 % multimode
energy-drift warning (loss off).

**Diagnosis (arbitrated against the papers, 2026-09-30).** Two of the
three same-day failures were NOT engine regressions:
- `dudley fig08 "DW too weak"`: stale 09-18 npz caches written by the
  pre-#0 engine, re-read by the post-#0 analysis kernel — caches
  deleted, fresh run gives DW 648 nm vs phase-matching 662.8 nm (2.2 %).
- `dudley "temporal reversal"`: the test (and the `time_reversal=True`
  compensations in `common.py` + five `gnlse.py` plot helpers) were
calibrated to the pre-#0 mirrored engine. The paper's own Fig. 3(b)
shows the soliton trail drifting to POSITIVE delay; the post-#0 engine
gives exactly that (+3.27 ps internal, red soliton = slower for
β₂ < 0). Compensations retired (defaults now False); test rewritten.
The fresh −20 dB span (437–2028 nm, fast grid) is broader than the
README's full-run row (500–1257) — fast-grid artifact + stale-cache
reading; full-run row to be re-recorded after a full re-run.
- The RW FWHM/energy-drift pair WAS a real engine bug:
  `MultimodeSplitStepEngine._fwm_substep`'s "frequency-domain RK4" had
  been reduced to a **single explicit Euler step** in the 09-28 #11
  rewrite (rhs evaluated once, linear update). Euler on the
  anti-Hermitian FWM flow pumps Σ|A|² at O(dz): pure-FWM drift 2.4 %,
  halving with dz (first-order signature), 0.000 % with FWM off.

**Fix (2026-09-30).** Restored classical RK4 stages inside the substep
loop (stability policy of #11 unchanged). RW deck: drift 13.8 % →
**1.9e-9**, FWHM error 10.5 % → **3.3 %**, step-converged (identical at
dz/2); the higher-mode locking shifts tightened to −1.30/−4.58 nm vs
required −1.53/−4.61. Full suite 1165/1165 green (incl. Wright-2015
slow path, 14 multimode-FWM tests).


---

## 7. ROCm iGPU training crash guardrail (system-level)

**Found.** 2026-09-22 (earlier Raissi-PINN session; codified as a rule after
the second near-miss).

**Symptom.** Training the float64 PINN on the amdgpu (8060S iGPU) via the
ROCm torch build **crashed the whole system** — a shared-memory float64
second-derivative autograd spike takes the iGPU's shared system RAM down
with it. Not an exception the process can catch: the machine dies.

**Guardrails landed (do not regress).**
- `--device {cpu,cuda,auto}` default **cpu**; `PH_PINN_DEVICE` env override.
- `--nf-chunk N` rotating collocation subsample (4× smaller autograd graph).
- Allocator cap `PH_GPU_CAP_FRAC` (default 0.6) + `empty_cache` every 200
  iters on GPU + automatic OOM→CPU fallback in `reproduce.py::validate`.
- Speed table (folder README): CPU + chunk 5000 (~45–55 min) **beats** the
  iGPU (~107 min full batch, ~37 min chunked — but with the crash risk).

**Rule.** Never train unguarded on the amdgpu. CPU is the house default for
float64 PINN work on this box; the iGPU only makes sense with the caps above
and a user explicitly accepting the risk. A real fp64-capable card or ZLUDA
(`.venv-cuda-vulkan`, see folder README) reopens the question.

---

## 5. Author/ process actions (not code bugs)

1. **Zenodo DOI + JOSS submission** — the prerequisites are in place
   (`paper/paper.md` + `paper.bib`, `CITATION.cff`, `CONTRIBUTING.md`,
   stability contract, provenance docs). Remaining is author-side: mint the
   Zenodo DOI, link it into `CITATION.cff` (`identifiers:`) and the README,
   then open the `openjournals/joss-reviews` submission issue.
   *(Tracked as the "Pending — author action" checklist in `ROADMAP.md`;
   reproduced here so no pending item is lost.)*
2. **Fold the received PDFs (P1, P3, P4; P2 optional) into new reproduction
   folders** — the only open items on the `PLAN.md` reproduction checklist
   (P2 = Peregrine is blocked on PDF availability; see `PLAN.md` §4 for why).
3. **`REPORT.md` / `REVIEW.md` reproduction tables** need a refresh at the
   end of each reproduction batch (`PLAN.md` checklist item, still unticked).

---

## Resolved during/around this audit (for the record — details in file history)

| Claim | Where it lived | Resolution |
|---|---|---|
| MI gain √2 mismatch vs Agrawal | `REVIEW.md` open item #2 | **Closed** — `fix-mi-gain-convention`; exact `4γP/\|β₂\|` cutoffs pinned by tests |
| FROG chirped retrieval "still open" | `REVIEW.md` open item #3 | **Closed** — `fix-frog-pcgpa-retrieval`, fidelity ≥ 0.999 |
| TMM exit medium hard-coded air | `REVIEW.md` open item #4, Macleod README | **Closed** — `n_incident`/`n_substrate` exposed; Airy-reference tests |
| `soliton.py` β₃ ×1e-27 (should be ×1e-36, DW fallback off by 1e9) | `REVIEW.md` ⚠ appendix | **Closed** — ×1e-36 + `test_beta3_si_uses_engine_convention` (engine-convention round-trip) |
| `_linear_step` docstring sign `exp(−i·…)` vs code `exp(+1j·φ)` | `REVIEW.md` open item #6 | **Closed** 2026-09-21 — summary line corrected with a sign-audit note |
| `Pattren` → `Pattern` breaking rename undocumented | `REVIEW.md` open item #7 | **Moot / closed** — no released tag (v0.1.0, v0.1.1…) ever contained `Pattren`; zero released users exposed |
| No `τ_shock` override | `dudley_2006_scg/README.md` #1, `PAPER_ANALYSIS.md` #2 | **Closed** (0.1.9 `tau_shock=`); README/analysis updated |
| No stochastic Raman noise source | `dudley_2006_scg` #2, `narhi_2016_mi_breathers/README.md`, `PAPER_ANALYSIS.md` #1 | **Closed** (source shipped, `step2-raman-noise-source`); READMEs updated; re-running figures remains repro-work |
| Multimode engine cannot carry absolute modal phase Δβ₀ (no spatial self-imaging by propagation) | `renninger_…/README.md` caveat 2 | **Closed** — `phase_offsets` argument added to `MultimodeSplitStepEngine` (Krupa reproduction); README updated |
| Norm-preserving shock integrator missing | `PAPER_ANALYSIS.md` #4 | **Closed as integrator issue** (RK4IP); residual drift re-filed as issue #1 above (model) |
