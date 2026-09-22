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

**Next action.** Optionally implement a photon-conserving shock term as an
opt-in `conserving_shock=True` flag (pcGNLSE form; two sign/|γ|
modifications vs the standard GNLSE), with conservation + red-shift +
blue-skew validation tests. Until then, quote the drift as a stated
tolerance per reproduction and respect the monitor warning.

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

**Next action.** Either document/validate the offset-free `beta_fn` contract,
or accept a relative-detuning callable / evaluate with an offset-aware
interface so the carrier never enters the subtraction. (The scalar
`mi_gain_spectrum` is unaffected — it takes β₂ directly and matches Agrawal
Eq. (5.1.9) exactly since `fix-mi-gain-convention`.)

---

## 3. `TaperedGNLSESolver`: inert `betas_unit` parameter

**Symptom.** The resolved solver takes `betas_unit` (validated against the
usual unit strings) but has no `betas` array at all — its dispersion comes
from `dispersion_profile` β(ω, z) in SI, so the flag has no behavioural
effect. A user passing `betas_unit="SI"` in good faith gets silence.

**Verified current.** No warning is emitted; the docstring documents the
inertia but nothing guards against the confusion.

**Next action.** Emit a `UserWarning` when `betas_unit != "ps^k/m"` is passed
to `TaperedGNLSESolver` (or drop the flag there; it is on the stable surface
now, so prefer warn-over-remove).
Source: `REVIEW.md` §change 1 note and the re-review of
`add-gnlse-beta-units-validation`.

---

## 4. `plot_waterfall`: non-standard waterfall rendering (cosmetic)

**Symptom.** The waterfall plot mixes time on the x-axis with a z offset
added directly to the y values (`envelope + z_steps[i]*1e3`), producing a
profile-of-offsets figure that is hard to interpret; plotting is
qualitative-only and the axis is not a proper lagged-perspective (no shared
axes, no z tick mapping).

**Verified current.** Still present in `gnlse.py::plot_waterfall` (confirmed
by `REPORT.md` §caveat 7 and the code shape; the readout is unchanged).
Cosmetic/UX only — no physics impact; all quantitative diagnostic surfaces
are separate.

**Next action.** Either re-draw as a true ridge/waterfall (per-trace y offset
handled by the artist transform, z-axis secondary ticks) or deprecate it in
favour of the 2-D evolution maps the newer reproductions use. Low priority.

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
