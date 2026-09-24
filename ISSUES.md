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

**Suspects / next actions (ranked).**
1. Noise-seed × step-size (dz 1 mm–1 cm) × seed-level sweep to find the
   regime where the Eq.-(13) estimate emerges in the measured readout.
2. Engine group-delay walk-off sign audit vs Eq. (11) — same
   family as **ISSUES.md #0** (the dispersion time-direction issue;
   Δβ^(p,i) requires the opposite frequency-side reading).
3. Evaluate whether the engine's `group_delays` channel-0-frame
   retarded-field convention (pumps never drift) needs a dedicated
   per-channel-delay arm for pump–sideband walk-off at 13 THz detuning.

**Status:** OPEN. Do not archive the Guasoni split-step layer's
"REPRODUCED" claim until this closes.

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
