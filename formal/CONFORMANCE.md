# CONFORMANCE — the theorem ↔ Python register

The point of this file is that the two sides of the project actually meet.

Each row pairs a proved Lean theorem with the Python symbols it governs, names
the primary source that quantity comes from, and records whether they agree.

Written **concurrently** with each stage, not at the end. A stage whose rows
are unwritten is an unfinished stage.

## Row states

| State | Meaning |
|---|---|
| `matches` | the Python symbol agrees with the theorem |
| `disputed` | they disagree; goes to the literature (`PLAN.md` §14.2) |
| `spec-only` | theorem exists, no Python counterpart yet |
| `uncovered` | Python symbol exists, no theorem governs it yet |

---

## Stage A — conventions, constants, dimensional units

Source of truth for the constant relations: CODATA 2018 exact values, via
`scipy.constants` in `photonics_helper/base.py:15-21`.

| # | Theorem (Lean) | Governs (Python) | Source | State | Notes |
|---|---|---|---|---|---|
| A1 | `Constants.wavelength_freq` `λ·f = c` | `Wavelength.to_freq` (`base.py:167`) | `c = λf` | `matches` | Python: `Frequency(C_MS / self.value)`. Same relation. |
| A2 | `Constants.freq_of_wavelength_pos` | `Wavelength.to_freq` | — | **gap** | See F1: the positivity hypothesis is real but unenforced in code. |
| A3 | `Constants.omega` / `omega_eq` `ω = 2πf` | `Frequency.to_omega` (`base.py:251`) | `ω = 2πf` | `matches` | Python: `2 * PI * self.as_Hz`. |
| A4 | `Constants.omega0` `ω₀ = 2πc/λ₀` | `Wavelength.to_omega` (`base.py:171`) | Agrawal eq. 2.7 | `matches` | Python: `2 * PI * C_MS / self.value`. Note this is the *carrier* relation, and it is applied to an arbitrary wavelength here. |
| A5 | `Constants.kTilde` `k̃ = ω/(2πc)` | `AngularFrequency.to_wn` (`base.py:329`) | `k̃ = 1/λ` | `matches` | Added in response to F2. Python: `as_rad_s/(2*PI*C_MS)`. |
| A6 | `Constants.kTilde_of_wavelength` `k̃ = 1/λ` | `Wavelength.to_wn` (`base.py:181`) | `k̃ = 1/λ` | `matches` | Python: `1/self.as_m`. |
| A5b | `Constants.k_of_kTilde` `k = 2πk̃` | — | — | `spec-only` | The relation between the two conventions. |
| A6b | `Constants.k` `k = ω/c` | — | — | `spec-only` | Angular wavenumber. **No Python symbol computes this.** |
| A7 | `Constants.energy_pos` `e = hc/λ` | `Wavelength.to_energy` (`base.py:175`) | `e = hc/λ` | `matches` | Python: `H_PLANCK * C_MS / self.value`. |
| A8 | `Constants.lambda0_of_omega0` | — | — | `spec-only` | Carrier round-trip; no Python symbol does this. |
| A9 | `Conventions.beta_flip` | dispersion sign convention in `gnlse.py` | Agrawal 4.2 | `uncovered` | Stage C. |
| A10 | `Conventions.beta_flip_even` / `_odd` | as A9 | Agrawal 4.2 | `uncovered` | Stage C. |
| A11 | `Conventions.axis_reversal_is_involution` | — | — | `spec-only` | Structural. |
| A12 | `Conventions.dft_pair`, `conj_dft`, `hermitian` | `TemporalGrid.fft` / `.ifft` (`core/grids.py`) | DFT | `matches` | **PROVED** (now `Grid.*`, see B9–B11). |
| A13 | `Conventions.Dim`, `Dim.power` | `photonics_helper/base.py` unit classes | — | `uncovered` | Python units are runtime-checked (pydantic), not type-level. Stage A gate deferred this. |
| A14 | — | `base.py:18` `Z0 = 1/(ε₀·c)` | SI | **gap** | See F3. No Lean statement governs this at all. |

## Stage B — grid and Fourier convention

`core/grids.py::TemporalGrid`, read directly. The Lean counterparts are
`Conventions.gridDt` / `gridTime` / `gridOmega` (defined once in
`Conventions.lean`) and the theorems in `Photonics/Grid/Basic.lean`.

| # | Theorem (Lean) | Governs (Python) | Source | State | Notes |
|---|---|---|---|---|---|
| B1 | `Grid.dt_mul_n` `dt·n = Tmax` | `TemporalGrid.dt` (`core/grids.py:47`) | `dt = T/N` | `matches` | Python: `self.Tmax.as_s / self.N`. |
| B2 | `Grid.time_span` `t_{n-1} - t_0 = Tmax - dt` | `TemporalGrid.t` (`:50`) | `np.linspace` endpoint-excluded | `matches` | Confirms the upper endpoint is excluded, so `N` samples of width `dt` span the window exactly. |
| B3 | `Grid.time_neg` | `TemporalGrid.t` (`:50`) | grid symmetry | `matches` | See F5: **false as first stated.** |
| B4 | `Grid.freq_spacing` `Δω = 2π/Tmax` | `TemporalGrid.dw` (`:60`) | `fftfreq` spacing | `matches` | Python computes `w[1] - w[0]`; the theorem makes explicit that it is independent of `N`. |
| B5 | `Grid.spacing_mul_window` | as B4 | — | `matches` | |
| B6 | `Grid.omega_max_mul_dt` `ω_max·dt = π` | `TemporalGrid.omega_max` (`:71`) | Nyquist | `matches` | Fixes the scale `_SHOCK_TAYLOR_LIMIT` is measured against. |
| B7 | `Grid.omega_neg` `ω_i(−1) = −ω_i(+1)` | `TemporalGrid.w` (`:54`) | axis orientation | `matches` | See F6: **false as first stated.** |
| B8 | `Grid.kernel_term` | `TemporalGrid.fft` / `.ifft` | DFT | `matches` | **Proved.** Each entry of analysis-then-synthesis is `exp(I·s·π·(i−j)·(1−2k/N))`. |
| B8a | `Grid.exponent_closed_formC` | as B8 | DFT | `matches` | **Proved.** |
| B8b | `Grid.kernel_factors` | as B8 | DFT | `matches` | **Proved.** Each summand is `C·ζ^k`, the reduction to a geometric series. |
| B8c | `Grid.rootOfUnityS_pow` / `rootOfUnityS_ne_one` | as B8 | — | `spec-only` | **Proved.** `ζ^n = 1`, and `ζ ≠ 1` for a non-zero bin offset. |
| B8d | `Grid.fin_diff_natAbs_lt` | as B8 | — | `spec-only` | **Proved.** `abs(i−j) < n` for distinct bins. |
| B9 | **`Grid.dft_pair`** `idft·dft = n·1` | `TemporalGrid.fft` / `.ifft` | DFT | `matches` | **PROVED.** With `Grid.dft_pair_entry`, the pointwise form. See F20. |
| B10 | **`Grid.conj_dft`** | `TemporalGrid.fft` (`conj(DFT(conj(·)))`) | DFT | `matches` | **PROVED.** `DFT(conj A) i = conj(DFT A (−i))`, from `origin_bin_exp` + `omega_reflect_sum`. |
| B11 | **`Grid.hermitian`** | (no direct symbol) | DFT | `spec-only` | **PROVED** as B10 restricted to real `A` — corollary, not a separate result. |
| B12 | `Grid.sum_neg_eq`, `neg_eq_sub`, `omega_reflect` | as B10 | — | `spec-only` | **Proved.** Reindexing for conjugate symmetry. |
| B12 | — | `TemporalGrid.for_pulse_train` | Agrawal 2.1 | `uncovered` | No theorem. |

### F5 — `time_neg` is false at index 0 (statement corrected)

First stated as `t_{-j} = -t_j` for all `j`. That fails at `j = 0`, because
`t_0 = -Tmax/2` while `-t_0 = +Tmax/2`: the origin sample is its own index
mirror, and `fftshift` does not move it.

Verified numerically, then corrected to carry `(j : ℕ) ≠ 0`. The mirror index is
`mirrorIdx j hj = n - j`, stated directly because `Fin`'s negation is modular
and proving the coincidence with `Fin.neg j` adds modular-arithmetic noise to a
statement about symmetry.

Not a code defect. It is a fact about the grid that was previously implicit and
is now a hypothesis in a proved theorem.

### F6 — `omega_neg` had the sign backwards, and needed no hypotheses (statement corrected)

First stated as `ω_{-1,mirror} = -ω_{1,i}`, which is false: the mirrored index
under reversed orientation gives `ω_i`, not `-ω_i`.

The true relation is much simpler and holds at **every** index with no evenness
condition and no nonzero hypothesis:

```
omega_neg : gridOmega n (-1) Tmax i = -(gridOmega n 1 Tmax i)     -- all i
```

Checked numerically for all 8 bins of an `n = 8` grid. Proof is `ring`.

This is the grid-level statement of the same fact as `Conventions.beta_flip`,
and it is now the theorem that `TemporalGrid.w`'s sign convention rests on.

### F12 — the reindexing machinery for conjugate symmetry is proved

Five lemmas, all in `Spectrum.lean`:

- `sum_neg_eq` — `∑ f(-j) = ∑ f j`, from `Equiv.neg`. One line, because `Fin n`
  carries a cyclic group structure and Mathlib already packages negation as a
  permutation.
- `neg_eq_sub` — `Fin.neg j = n - j` for `j ≠ 0`, via `Fin.neg_def` and
  `Nat.mod_eq_of_lt`.
- `omega_reflect` — `ω_{n-i} = -ω_i` for even `n` and `i ≠ 0`.
- `rootOfUnityS` — the orientation-aware root, and the reason the earlier
  `kernel_factors` statement was false.
- **`kernel_factors`** — the summand factors as `C · ζ^k`. This is the step that
  turns the DFT pairing into a geometric series.
- **`fin_diff_natAbs_lt`** — `|i − j| < n` for distinct bins. The last arithmetic
  input the off-diagonal half of `dft_pair` needs.

`kernel_factors` is the one the ℝ/ℂ coercion plan was aimed at, and the fix was
not the coercion rewrite it proposed. It was to route through `kernel_term`:
rewriting the summand with an already-proved theorem of the same shape avoids
producing the two differently-placed coercions in the first place. The
remaining obligation is arithmetic in `ℂ` and closes with `push_cast; ring`.

`fin_diff_natAbs_lt` took the longest single route. `Int.natAbs_sub_le`, the
obvious lemma, is far too weak (`≤ |i| + |j|`). The working route normalises the
sign twice with `Int.natAbs_neg` so that whichever ordering holds, the
difference ends up with a non-negative `ℤ` value, then lifts the bound from `ℕ`
to `ℤ` by `exact_mod_cast`.

With both, the off-diagonal half of `dft_pair` needs only assembly:
`Finset.sum_mul` to lift the constant and `geom_sum_zero` to finish.
`dft_pair_entry`'s docstring carries the remaining list.

### Stage E — thin films

`dbr.py`, read directly. First coverage of a module outside Stages A–D.

| # | Theorem (Lean) | Governs (Python) | Source | State | Notes |
|---|---|---|---|---|---|
| E1 | `Optics.layerMatrix` | `_layer_matrix` (`dbr.py:267`) | Macleod §2.4 | `matches` | `[[cos δ, −i sin δ/η], [−i η sin δ, cos δ]]`, TE and TM via `η`. |
| E2 | **`Optics.layerMatrix_det`** `det = 1` | as E1 | Born & Wolf §1.6 | `matches` | **No losslessness hypothesis**; complex `cos²+sin²=1`. |
| E3 | **`Optics.stack_det`** `det = 1` | `transfer_matrix` (`dbr.py:322`) | Macleod Eq. 2.55 | `matches` | Left fold, matching the Python exactly. |
| E4 | **`DBREnergy.energyConservation`** `R + T = 1` | `spectrum` `R`, `T` (`dbr.py:341`) | Macleod §2.6 | `matches` | **PROVED** in `DBREnergy.lean`, with the lossless shape + `det = 1` (`det M = 1` alone is not enough). See F21. |
| E5 | **`Optics.etaTE`, `etaTM`, `admittance_normal_incidence`, `admittance_product`, `etaTE_nonneg`, `etaTM_nonneg`** | `_admittance` (`dbr.py:364`) | Macleod §2.4 | `matches` | **PROVED.** `η = n cos θ` (TE) / `n/cos θ` (TM); the branches agree at normal incidence; their product is `n²` — the polarisation contrast lives in `cos θ_l`, never in the material; both are nonnegative with the principal root. |
| E6 | **`Optics.snellSine`, `snell_invariant`, `snell_admissible`, `cosLayer_nonneg`, `cosLayer_le_one`, `clip_no_op`, `clip_fires_iff`, `clip_lower`, `clip_upper`, `deltaPhase`, `deltaPhase_nonneg`, `deltaPhase_normal_pos`** | Snell layer sine + `np.clip` (`dbr.py:293`); `_layer_matrix` phase thickness (`dbr.py:305`) | Born & Wolf §1.6 | `matches` | **PROVED.** Snell invariant form justifies passing the configured incident index; admissibility `|n₀ sin θ_i| ≤ n_l` is exactly the no-clip set; the clip saturates to ±1 (TIR) outside it; `δ = 2π n d cos θ_l / λ ≥ 0`, strictly positive at normal incidence. |
| E7 | `Optics.stackOrder_ne` | `layerIndex` order (`dbr.py`) | Macleod Eq. 2.55 | `matches` | Criterion proved: `stackMatrix [A,B] ≠ stackMatrix [B,A]` when `A*B ≠ B*A` (`DBR.lean`). The degenerate witness (a `δ = 0` layer matrix is the identity, hence commuting) is also proved; the nontrivial direction is the criterion itself. |

### Stage E — second-harmonic generation

`chi2.py`, read directly. Verified against the solver before formalising:
`solve_shg(length=0.05, P0=1, sigma=2)` returns `η = 0.009934`, and
`tanh²(κL) = tanh²(0.1) = 0.009934`.

| # | Theorem (Lean) | Governs (Python) | Source | State | Notes |
|---|---|---|---|---|---|
| E8 | `Optics.kappa` `κ = σ√P₀` | `Chi2Result.kappa` (`chi2.py:281`) | — | `matches` | |
| E9 | `Optics.kappa_nonneg` | as E8 | — | **gap** | See F14. |
| E10 | `Optics.shgEfficiency` `η = tanh²(κz)` | `chi2.py:41`, `:174` | Boyd §2.2 | `matches` | Reproduced numerically. |
| E11 | **`Optics.shgEfficiency_le_one`** `η ≤ 1` | `Chi2Result.efficiency` (`chi2.py:264`) | — | `matches` | SHG cannot over-convert. |
| E12 | `Optics.shgEfficiency_nonneg`, `_zero` | as E11 | — | `matches` | `η(0) = 0`. |
| E13 | `Optics.shgEfficiency_even` | as E11 | — | `matches` | `η(−z) = η(z)`. |
| E14 | **`Optics.shgEfficiency_half`** | — | Boyd §2.2 | `spec-only` | `η = 1/2` exactly at `κz = artanh(1/√2)`. |
| E15 | `Optics.LambdaQPM` `Λ = 2π·order/abs(Δk)` | `Lambda_qpm` (`chi2.py:117`) | — | `matches` | |
| E16 | `Optics.LambdaQPM_pos` | as E15 | — | `matches` | |
| E17 | `Optics.LambdaQPM_at_zero` | as E15 | — | **gap** | See F15. |
| E18 | `Optics.LambdaQPM_neg` | as E15 | — | `matches` | Depends on `abs(Δk)` only. |
| E19 | **`Optics.deltaK`**, `deltaK_zero_iff`, `deltaK_qpm_cancels` | `delta_k_shg` (`chi2.py:100`), `Lambda_qpm` (`:117`) | — | `matches` | `Δk = β(2ω) − 2β(ω)`; phase matching is `Δk = 0`; first-order QPM cancels it exactly (`Δk − 2π/Λ = 0`). |
| E20 | **`Optics.overlapCoupling`, `overlapCoupling_uniform`, `overlapCoupling_planeWave`** | `shg_coupling_overlap` (`chi2.py:661`) | Wang et al. 2017 Eq. 2 | `matches` | Uniform-field reduction to the Boyd `shg_coupling` formula proved exactly; scale-invariance proved (the coupling is unchanged by field rescaling). Docstring's `n³` form and code's `n` form verified equivalent. |

### F14 — `shg_coupling` has no sign or range check (gap)

`kappa_nonneg` needs `σ ≥ 0` and `P₀ ≥ 0`. Nothing in `chi2.py` checks either:
`shg_coupling` returns whatever the dispersion and `d_eff` give, including a
negative `σ`, and `kappa` then multiplies it into the coupling without comment.

Not a physics error — a negative `σ` is meaningful, it reverses the direction of
conversion — but the docstring presents `κ = σ√P₀` as a magnitude. Same class as
F1: the theorems' hypotheses are real domain conditions that the Python assumes
without stating.

### F15 — `Lambda_qpm(0, ...)` divides by zero (gap)

`Lambda_qpm`'s docstring says `delta_k` must be non-zero and nothing enforces
it. At `Δk = 0` the function returns `0.0` through `2π/0`.

The physics is right — exact phase matching needs no grating — so returning
something is defensible, but it is returned *by division by zero* rather than as
a deliberate branch, and `0` reads as "zero period", which is the opposite of
"no grating needed". `LambdaQPM_at_zero` documents the current behaviour; the
fix is an explicit `delta_k == 0` branch with a comment.

E19, E20, and now **E5, E6** are proved (`DBR.lean`, 2026-10-26). The key
statements: the Snell invariant `n_l sin θ_l = n₀ sin θ_i`;
`η_TE·η_TM = n²` so polarisation contrast is wholly obliquity; admissibility
`|n₀ sin θ_i| ≤ n_l` ⟺ the `np.clip` guard is the identity; the clip
saturates to ±1 (the TIR branch) outside it; `δ = 2π n d cos θ_l/λ ≥ 0`,
strictly positive at normal incidence.

### Stage F — four-wave mixing and modulation instability

`phase_matching.py`, the largest module in the library and until now entirely
unread. Both closed forms come from Agrawal *Nonlinear Fiber Optics* §5.1.

| # | Theorem (Lean) | Governs (Python) | Source | State | Notes |
|---|---|---|---|---|---|
| F1 | `Optics.fwmIdler` `ωᵢ = 2ωₚ − ωₛ` | `fwm_idler_frequency` (`phase_matching.py:515`) | Agrawal §5.1 | `matches` | |
| F2 | **`Optics.fwm_energy`** `ωₛ + ωᵢ = 2ωₚ` | as F1 | Agrawal §5.1 | `matches` | Energy conservation; the invariant behind `scan_fwm_detuning`. |
| F3 | `Optics.fwm_degenerate` | as F1 | — | `matches` | `ωₛ = ωₚ ⟹ ωᵢ = ωₚ`. |
| F4 | `Optics.fwm_reflection` | as F1 | — | `matches` | `ωᵢ − ωₚ = ωₚ − ωₛ`; red signal ⇒ blue idler. |
| F5 | `Optics.fwm_involutive` | as F1 | — | `matches` | Signal/idler exchange is an involution. |
| F6 | `Optics.fwm_idler_pos` | as F1 | — | `matches` | `ωᵢ > 0 ⟹ ωₛ < 2ωₚ`. |
| F7 | `Optics.miOmegaSq` `Ω² = −4γP/β₂` | `mi_sideband_frequencies` (`phase_matching.py:667`) | Agrawal §5.1 | `matches` | |
| F8 | **`Optics.mi_cutoff`** `β₂Ω² + 4γP = 0` | as F7 | Agrawal §5.1 | `matches` | The dispersion relation itself. |
| F9 | **`Optics.mi_exists`** | as F7 | — | `matches` | `β₂ < 0 ∧ γP > 0 ⟹ Ω² > 0`: the two-sideband branch. |
| F10 | **`Optics.mi_absent_normal`** | `if beta2 >= 0` branch (`:689`) | Agrawal §5.1 | `matches` | MI requires anomalous dispersion. Physical, not a guard. |
| F11 | `Optics.mi_absent_no_drive` | `if Omega_sq < 0` branch (`:692`) | — | `matches` | |
| F12 | `Optics.mi_sidebands_symmetric` | `return np.array([-Omega, Omega])` | — | `matches` | `[−Ω, +Ω]`. |
| F13 | **`Optics.mi_mono_P`** | as F7 | Agrawal §5.1 | `matches` | More power ⇒ wider MI. |
| F14 | **`Optics.mi_anti_mono_absBeta2`** | as F7 | Agrawal §5.1 | `matches` | Stronger dispersion ⇒ narrower MI. See F16. |
| F15 | **`Optics.miGain`**, `miGain_band`, `miGain_outside`, `miGain_nonneg`, `mi_gain_peak`, `mi_gain_le_max`, `miPeakSq_half` | `mi_gain_spectrum` (`:598`) | Agrawal §5.1.2 | `matches` | **PROVED.** Band structure `g = |β₂Ω|√(Ω_c²−Ω²)` on `Ω < Ω_c`, zero outside; peak location `Ω² = Ω_c²/2` and peak height `g_max = 2γP` proved, not quoted; the bound `g ≤ 2γP` everywhere. |
| F16 | **`Optics.miMismatch`, `miGainExt`, `miGainExt_band_iff`, `miGainExt_peak`, `miGainExt_le_max`, `miGainExt_reduces`** | `mi_gain_spectrum_extended` (`:699`) | — | `matches` | **PROVED.** Band `−4γP < Δ < 0` is exactly the positivity window of `−Δ(Δ+4γP)`; peak at `Δ = −2γP`; and for quadratic β the extended gain *equals* the classical one — the docstring's "reduces exactly" sentence. |
| F17 | **`Optics.dwResidual`, `dwResidual_at_soliton`, `dwResidual_quadratic`, `dwResidual_cubic_root`, `dwDetuning_blue`, `dwDetuning_blue_fast`, `blue_shift_wavelength`** | `dispersive_wave_roots` (`:868`) | — | `matches` | **PROVED.** Residual at the soliton is exactly `−qₛ` (so `ωₛ` is a root iff `qₛ = 0`); quadratic case is a double root (the `f'(ωₛ) = 0` tangency); cubic root `Ω = −3β₂/β₃` and the **sign**: DW is blue-shifted exactly when `β₂β₃ < 0`. See F22 on the fallback's factor. |
| F18 | **`Optics.fwmEfficiency`, `fwmEfficiency_le_one`, `fwmEfficiency_lt_one`, `fwmEfficiency_perfect`** | `fwm_efficiency` (`:448`) | Agrawal §5.2 | `matches` | **PROVED.** `sinc²(ΔβL/2) ≤ 1` (the claimed 0–1 range), with equality *only* at perfect phase matching; the module's `arg = 0 → 1.0` special case agrees with the limit. |
| F19 | **`Dispersion.phaseRemainder_lagrange`, `Dispersion.phaseRemainder_bound`** | `_taylor_phase_max_error` (`:1066`) | Taylor's theorem | `matches` | **PROVED (2026-10-26).** The full phase minus any Taylor truncation is exactly `f^(n+1)(x')·Δ^(n+1)/(n+1)!` for an intermediate `x'`, and with `M` a bound on `|f^(n+1)|` across the interval the truncation error is `≤ M·|Δ|^(n+1)/(n+1)!` — the bound the function's numeric check is estimating. |

### F16 — `mi_anti_mono_absBeta2` was stated backwards

The first draft asserted that a larger `|β₂|` gives a *larger* MI cutoff. Since
`Ω² = 4γP/|β₂|` that is false, and Lean rejected it at the final arithmetic step.

Corrected to: less normal dispersion (`b₂ ≤ b₂'`) gives a larger cutoff, i.e.
`Ω²` decreases with `|β₂|`. This is the ninth wrong statement of mine, and the
second where a *direction* of a monotonicity was wrong.

**Pattern worth recording.** Both monotonicity errors (`omega_neg` in Stage B,
this one) came from writing the inequality in the direction that "sounds
right" rather than deriving it from the formula. For a `c/x`-shaped quantity,
derive the direction from the exponent before writing it down.

F15–**F19** are proved (band, peak, and peak-height for the classical gain;
the extended Δ(Ω) form and its exact classical reduction; the dispersive-wave
residual, tangency, and blue-shift sign; the lossless sinc² efficiency; and
F19 in `Dispersion/Beta.lean` — the phase-truncation remainder, both Lagrange
form and `M`-bound). Still unread: `gnlse.py` beyond its shock terms,
`raman_transfer.py`, `structured.py`, `pulse.py`, `breathers.py`, `noise.py`.

### F13 — `det(M) = 1` holds for absorbing layers; `R + T = 1` does not

The plan expected these to be the same statement. They are not, and the
distinction is the useful part.

`layerMatrix_det` and `stack_det` prove that every layer matrix, and hence
every product of them, has determinant `1`. There is **no losslessness
hypothesis**: the identity is `cos²δ + sin²δ = 1` in `ℂ`, which is
complex-analytic. Verified against `dbr.py`:

| stack | layer det | stack det | `R + T` |
|---|---|---|---|
| TiO₂/SiO₂, TE, 1550 nm | `1.000000000000` | `1.000000000000` | `1` |
| TiO₂/SiO₂, TM, 1550 nm | `1.000000000000` | `1.000000000000` | `1` |
| absorbing (`k = 0.02`), TE | `1.000000000000` | `1.000000000000` | `0.9292048511` |

So the determinant stays `1` while energy is absorbed. `R + T = 1` is the
genuinely stronger claim and needs lossless media; **it is now proved**
(`DBREnergy.lean::energyConservation`, F21) — and the absorbing row is the
counterexample that pins down why it needs the hypothesis.

### Stage G — structured light

`structured.py`. The module is **properly validated**: `rayleigh_range` raises
on `w0 <= 0` *and* on `wavelength <= 0`, and `normalize` refuses a zero-power
field. That is better discipline than `chi2.shg_coupling` (F14) or
`dbr._layer_matrix`, and worth recording as a positive result.

| # | Theorem (Lean) | Governs (Python) | Source | State | Notes |
|---|---|---|---|---|---|
| G1 | `Optics.rayleighRange` `z_R = πw₀²/λ` | `rayleigh_range` (`structured.py:62`) | — | `matches` | |
| G2 | `Optics.rayleighRange_pos` | as G1 | — | `matches` | `w₀ > 0, λ > 0 ⟹ z_R > 0`; both are enforced in Python. |
| G3 | `Optics.rayleighRange_mono_w0` | as G1 | — | `matches` | Range grows with the waist. |
| G4 | `Optics.beamWaist`, `beamWaist_even` | `beam_waist` (`:77`) | Gaussian beam | `matches` | `w(z) = w(z)`; the profile depends on `z` only through `z²`. |
| G5 | **`Optics.beamWaist_ge_waist`** `w(z) ≥ w₀` | `beam_waist` (`:77`) | Gaussian beam | `matches` | A beam is never narrower than its waist. |
| G6 | `Optics.lgRmsRadius` `w/√2·√(2p+\|l\|+1)` | `second_moment_radius` docstring | LG mode | `matches` | |
| G7 | **`Optics.lgRmsRadius_ge_fundamental`** `r_rms ≥ w/√2` | as G6 | LG mode | `matches` | No LG mode of order `(p,l)` is tighter than the fundamental. |
| G7b | `Optics.lgRmsRadius_fundamental` | as G6 | — | `matches` | `(0,0)` attains the bound. |
| G8 | `Optics.gouyPhase`, `gouyPhase_zero` | `gouy_phase` (`:93`) | Gaussian beam | `matches` | Phase vanishes at the waist. |
| G9 | **`Optics.gouyPhase_mono`** | as G8 | — | `matches` | Gouy phase advances monotonically along the beam. |
| G9b | `Optics.gouyPhase_neg_order` | as G8 | — | `matches` | Negative order runs the phase backwards. |
| G10 | — | LG field, `intensity`, `overlap` | — | `uncovered` | The modal overlap integral and `∫|u|² r dr dφ = 1` normalisation need `MeasureTheory`; see `PLAN.md` §8.5. |

G10 is the remaining structured-light work, and it is the `MeasureTheory` case
the plan flagged as the hard one back in Stage E.

### F17 — `linarith` cannot see through `Real.sq_sqrt`

Every square-root bound here reduced to the same gap, and solving it once as
`one_le_sqrt : 1 ≤ A → 1 ≤ √A` (via `sq_sqrt`, then `nlinarith` with
`sq_nonneg (√A − 1)` **and** `Real.sqrt_nonneg`) closed all of them. That lemma
is now the first thing to reach for in this Mathlib revision, not `nlinarith`
alone.

### Stage H — the RK4 integrators

`_rk4_step` (`chi2.py:298`) and the matching step in `gnlse.py`. The classical
fourth-order Runge–Kutta method, transcribed stage for stage.

| # | Theorem (Lean) | Governs (Python) | Source | State | Notes |
|---|---|---|---|---|---|
| H1 | `Solvers.rk4Step` | `_rk4_step` (`chi2.py:298`) | — | `matches` | Stage-for-stage transcription. |
| H2 | `Solvers.rk4_weights_sum` | `h / 6.0 * (k1 + 2 k2 + 2 k3 + k4)` | — | `matches` | `1+2+2+1 = 6`: the consistency condition. |
| H3 | `Solvers.rk4Step_constant` | as H1 | — | `matches` | Exact update for a state-independent RHS. |
| H4 | **`Solvers.rk4Step_linear`** | as H1 | — | `matches` | For `y' = λy`, the step is multiplication by `R(z)`. |
| H5 | **`Solvers.stability_polynomial`** | as H1 | — | `matches` | `R(z) = Σ_{k≤4} zᵏ/k!`: RK4 *is* the truncated exponential. |
| H6 | **`Solvers.exp_taylor_abs_error_le`, `Solvers.iteratedDeriv_exp`, `Solvers.rk4_error_le`** | as H1 | Taylor's theorem | `matches` | **PROVED (2026-10-26).** `|exp z − Σ_{k≤4} zᵏ/k!| ≤ exp|z|·|z|⁵/5!` for every `z ≠ 0`, from `taylor_mean_remainder_lagrange_iteratedDeriv` applied to `exp` at `x₀ = 0` — no hypothesis carries the expansion any more. |
| H7 | **`Solvers.ipStep`, `ipStep_linear`, `ipStep_nonlinear_only`, `ipStep_factors`** | `gnlse.py` RK4-in-interaction-picture | — | `matches` | **PROVED.** The IP step is *literally* the exact exponential propagator composed with the RK4 map of the remainder; linear-only is the pure exponential (`μ = 0`), nonlinear-only is plain RK4 (`λ = 0`). The `O(h⁵)` error lives only in the RK4 factor. |
| H8 | **`Solvers.split_comp_assoc`, `Solvers.conv_diag_ne_commute`, `Solvers.Composition.*`** | SSFM operator composition | — | `matches` | **PROVED.** Composition is associative (`D∘K∘D` unambiguous); a convolution (Raman-shaped) and an unequal diagonal (dispersion) provably do not commute, witnessed by a one-bin field — the formal content of the ISSUES.md #0 convention swap. Splitting-coefficient order conditions (`Composition.lean`) proved for Strang/Yoshida weights. |

H5 is the load-bearing one: it says exactly what "RK4IP" buys, namely that the
one-step map for a (near-)linear propagator is the fourth-order Taylor sum of
the exponential.

### F18 — `rk4_error_le`'s hypothesis, now removed

An earlier draft carried the expansion `exp z = Σ + z⁵/20` as a *hypothesis*,
because the Taylor-bound lemmas were believed absent. They are not
(`taylor_mean_remainder_lagrange_iteratedDeriv` exists at this revision), and
the analytic bound is now proved outright (`exp_taylor_abs_error_le`, H6) —
twice: the one-sided `Σ ≤ exp` for `z ≥ 0` (`Real.sum_le_exp_of_nonneg`, now
the whole proof of H6) and the two-sided `|exp z − Σ| ≤ exp|z|·|z|⁵/5!` valid
including negative `z`, where the truncation overshoots.

The route that finally closed it: `iteratedDeriv k exp = exp` proved by
induction, then the Lagrange remainder with `n = 4` at `x₀ = 0`, and the
Taylor polynomial evaluated into the Maclaurin sum. Fiddly but linear —
the earlier sessions had simply assumed the lemma was absent, which is a
finding about assumption-forming, not about Mathlib.

### Stage I — Strang splitting and the Kerr sub-step

The nonlinear sub-step of `gnlse.py:1060-1068`. This is the first coverage of
the main solver rather than one of its sub-operators.

| # | Theorem (Lean) | Governs (Python) | Source | State | Notes |
|---|---|---|---|---|---|
| I1 | `Solvers.kerrPhase` `θ = ½γP·dz` | `half_phase = 0.5j * gamma * P_NL * dz` (`:1067`) | — | `matches` | |
| I2 | `Solvers.kerrStep` `A ↦ A·exp(iθ)` | `A * np.exp(half_phase)` (`:1068`) | — | `matches` | |
| I3 | **`Solvers.kerrStep_norm`** `‖A·exp(iθ)‖ = ‖A‖` | `conserving_shock`, photon tracking | — | `matches` | The Kerr phase cannot change photon number. |
| I4 | `Solvers.exp_I_norm` | as I3 | — | `matches` | `\|exp(iθ)\| = 1`. |
| I5 | **`Solvers.strang_halves_compose`** | half / shock / half ordering | Strang 1960 | `matches` | Two half phases = one full phase, exactly. |
| I6 | `Solvers.two_phases_add` | as I5 | — | `matches` | Phase rotations compose additively. |
| I7 | `Solvers.kerrStep_zero_phase` | as I2 | — | `matches` | No nonlinearity ⟹ identity step. |
| I8 | `Solvers.strang_exact_on_phase` | as I5 | — | `matches` | The split adds no error of its own on a purely phase-driven problem. |

I3 is the load-bearing one. It says the photon-number drift `gnlse.py` tracks
cannot originate in the Kerr phase term, which is what forces the attribution
of that drift to the dispersion and shock operators.

I5 and I8 say the halving itself is free: applied to a purely phase-driven
equation the Strang split is **exact**, so its entire error budget is the
commutator between the phase and dispersion operators. That commutator is not
yet formalised (`PLAN.md` §9.1); it needs diagonal phase versus a shift
operator, which is real work.

### Stage J — the dispersion operator and split-step conservation

Discharges the `‖D‖ = 1` obligation that `PLAN.md` §6.2 has carried since Stage C.

| # | Theorem (Lean) | Governs (Python) | Source | State | Notes |
|---|---|---|---|---|---|
| J1 | `Solvers.dispersionOp` `A ↦ A·exp(iβz)` | dispersion sub-step (`gnlse.py`) | — | `matches` | |
| J2 | **`Solvers.dispersionOp_norm`** `‖D‖ = 1` | as J1 | — | `matches` | The linear operator is lossless. |
| J3 | `Solvers.dispersionOp_zero` | as J1 | — | `matches` | Zero phase thickness is the identity. |
| J4 | `Solvers.dispersionOp_add` | as J1 | — | `matches` | Dispersion steps add their phase thicknesses. |
| J5 | **`Solvers.splitStep_norm`** | SSFM step, shock excluded | — | `matches` | One dispersion + one Kerr step conserve norm exactly. |
| J6 | **`Solvers.splitSeq_norm`** | full SSFM loop | — | `matches` | **Any** number of alternated steps conserves norm. |
| J7 | — | shock correction | Blow & Wood 1989 | `uncovered` | The only remaining term that can change norm. |
| J8 | — | delayed Raman term | Agrawal §3.5 | `uncovered` | The other candidate. |

### F19 — the drift attribution is now proved, not argued

`gnlse.py` reports photon-number drift. J5 and J6 say that **any number of
alternated dispersion and Kerr steps conserves the norm exactly**, with no error
term and no step-size assumption beyond the operators being unitary.

So the reported drift cannot come from the dispersion operator or the Kerr
phase. It is attributable to exactly three things:

1. the shock correction (J7, not formalised),
2. the delayed Raman term (J8, not formalised),
3. discretisation of whichever of the above is active.

That is a sharp constraint, and it is the third time this project has produced
one: the shock Taylor validity (F11), the Kerr `‖A‖` invariance (I3), and now
the full split-step norm conservation (J6) together exclude every explanation
except the shock and Raman terms.

It also explains why the `conserving_shock` flag needed `gamma < 0` (F22/F13
family): the shock term is the one piece of the scheme that is *not* a phase
rotation, so it is the only piece that can move the norm at all.

### F20 — `rootOfUnityS_pow` solved; `rootOfUnityS_ne_one` survives on a cast artefact

**Solved.** The fix was to stop substituting. `rcases (validAxis s) with rfl | rfl`
puts a bare `1` into `rootOfUnityS`'s body, and that literal lands as a *natural*
cast in the exponent, after which `rw` and `ring` stop matching. Stating the two
orientations separately with an explicit `(1 : ℝ)` literal, and assembling with
`simpa [rootOfUnityS]`, works:

- `rootOfUnityS_pow_pos`, `rootOfUnityS_pow_neg`, and `rootOfUnityS_pow`.

So `ζ^n = 1` is proved, and the `dft_pair` assembly no longer needs a geometric
sum over an unknown root.

**Still stuck.** `rootOfUnityS_ne_one`, i.e. `ζ ≠ 1` for a non-zero bin offset.
The mathematical argument is short — `Complex.exp_eq_one_iff`, imaginary parts,
multiply through by `N` to get `-m = k·N`, then `m.natAbs < n` forces `k = 0` —
and `fin_diff_natAbs_lt` (proved) supplies the bound. Three algebraic routes
were tried and all three failed on cast placement rather than on the reasoning:

1. `congr 1` then `ring` — the obligation contains `↑1`, unprovable.
2. `rw [← Complex.exp_add]` with a single `key` — `key` closes, the following
   `rw` misses.
3. multiply through with `div_mul_cancel₀` / `div_eq_iff` — the cast on `N` is
   an `Int` cast where the lemma wants the `ℕ` one, so the rewrite misses.

The next attempt should take the imaginary-part argument *directly* on the
hand-substituted statement `exp(I·(-2πm/N)·1) ≠ 1`, without routing through
`rootOfUnityS` at all. That keeps every cast where it was written.

### F21 — `R + T = 1` is derivable after all; the recorded claim was wrong

The 2026-10-24 CHANGELOG entry recorded that `r² + T = 1` "does not follow from
`det M = 1`" and is "a different argument". The first half is true and the
second is not, and the distinction changes what the remaining work was.

The reduction is right — in the index form `r = (n₀B − C)/(n₀B + C)`,
`T = 4n₀n_s/(n₀B + C)²`, the identity needs `B·C̄ + B̄·C = 2n_s`, not
`B·C = n_s` — but with the **lossless shape** `M = [[a, iβ], [iγ, δ]]` (real
diagonal, imaginary off-diagonal) that conjugate identity is one line:

`|D|² − |N|² = 4·η₀·η_x·(a·δ + β·γ)`, and `a·δ + β·γ = det M = 1` because
`i² = −1`.

`DBREnergy.lean::energyConservation` now proves `R + T = 1` for any
lossless-shaped stack, using `DBR.lean::stack_det` for the determinant half and
the shape as an explicit hypothesis. So the correct statement of the
obligation was never "a different argument" — it was "the same argument, plus
the regime as a hypothesis", exactly what `PLAN.md` §14.2 prescribes. The
absorbing counterexample of F13 still stands and still separates the two
claims; what changed is that the lossless direction is now closed.

**Which side moved:** `formal/`. No Python symbol moved; the derivation
validates `dbr.py`'s `spectrum` formulas as written.

### F22 — the β₂/β₃ fast estimate shares the DW's sign but not its magnitude

`soliton.py`'s `disperative_wave_wavelength` fallback uses `Δω = −2β₂/β₃`.
Lean pins the exact root of the *cubic truncation* of the phase-matching
residual (with `qₛ = 0`) at `Ω = −3β₂/β₃`
(`Optics.dwResidual_cubic_root`). The two agree in sign — both are blue for
`β₂·β₃ < 0` (`Optics.dwDetuning_blue` and `dwDetuning_blue_fast`), which is the
invariant `PLAN.md` §8.2 asked to prove — but differ by the factor `3/2` in
magnitude.

Recorded as **open — needs source**: the fallback is labelled an estimate in
its own docstring, and the exact factor depends on the `qₛ` convention (with
the soliton's nonlinear wavenumber `qₛ > 0` the leading blue root is carried
by the cubic term and is not either of these values). The sign invariant is
proved regardless; the magnitude is a question for the soliton-RDW source, and
whoever picks that up should decide whether the fallback wants `−3β₂/β₃`.

### F9 — three separate axis-orientation errors, one shared rule

By this point the same mistake has appeared three times:

| Where | Manifestation |
|---|---|
| `beta_flip` (Stage A) | odd-order β terms negate under axis reversal |
| `omega_neg` (Stage B) | every frequency bin negates under axis reversal |
| `kernel_factors` (Stage B) | the DFT kernel's root argument must carry `s` |

**Rule for the rest of the project:** any formula that accumulates over the
frequency index must be checked against `Grid.omega_neg`, not only against the
real-valued algebra. The real-valued algebra is insensitive to `s`; the physics
is not.

### F8 — the orthogonality core is proved; the assembly is blocked

The general route was taken rather than the concrete-`n` fallback, because the
Raman `H(Ω)`-realness argument needs the general statement.

Proved in `Photonics/Grid/Spectrum.lean`:

- `geom_sum_zero` — a finite geometric sum of a non-trivial root of unity
  vanishes. Mathlib has no such lemma at this revision, so it is proved by
  induction on the exponent count.
- `rootOfUnity_pow`, `rootOfUnity_ne_one` — the root built from an integer bin
  offset is a genuine `N`-th root of unity, and is trivial only when the offset
  is zero. The non-triviality proof goes through `Complex.exp_eq_one_iff`; note
  that the *real* parts of both sides are zero (everything is a multiple of
  `i`), so the imaginary parts carry all the information.
- `exponent_closed_form` — `ω_k t_j − ω_i t_k = s·π·(i−j)·(1 − 2k/N)`, the
  identity that makes the DFT summand a geometric progression. The closed form
  was checked numerically before being proved, which is how its shape was
  found.
- `finSum_to_range` — transporting `Fin n` sums to `Finset.range n`.

Proved since then: `exponent_closed_formC` and `kernel_term`, which reduce the
pairing to splitting the summand as `C * ζ^k`, lifting `C` with
`Finset.sum_mul`, and finishing with `geom_sum_zero`.

**Still blocked**, on ℝ/ℂ coercion placement rather than mathematics. The fix is
recorded in `CHANGELOG.md`: state the remaining lemmas over the complex form and
avoid bridging the two types by rewriting.

### F7 — `omega` chokes on `x + x` in this Mathlib

Practical note, recorded so the next person does not lose time to it: `omega`
fails with "No usable constraints found" on any goal containing `m + m` (for
example `0 < n` from `hn : Even n`, where `Even n` unfolds to `∃ r, n = r + r`).
Use `Nat.two_mul` / `Nat.add_comm` to normalise first, or `nlinarith`.

It also surfaced a genuine falsehood: an `Even.pos'` helper asserting `0 < n`
cannot be proved because `Even 0` holds. That lemma was removed rather than
salvaged.

## Stage C — dispersion and soliton relations

`soliton.py::SolitonAnalysis` and `gnlse.py`, read directly. All of these are
`ℝ`-valued, which is why Stage C went quickly where Stage B stalled.

| # | Theorem (Lean) | Governs (Python) | Source | State | Notes |
|---|---|---|---|---|---|
| C1 | `Dispersion.gamma` `γ = n₂ω₀Γ/(c·A_eff)` | `gnlse.py` Kerr coefficient | Agrawal 2.3 | `matches` | `gamma_pos` under `n₂,ω₀,Γ,A_eff,c > 0`. |
| C2 | `Dispersion.gamma_full_confinement` | `Γ = 1` default | — | `matches` | Recovers the fibre case. |
| C3 | `Dispersion.LD` `L_D = T0²/|β₂|` | `dispersion_length` (`soliton.py:135`) | Agrawal 3.2 | `matches` | |
| C4 | `Dispersion.LNL` `L_NL = 1/(γP)` | `nonlinear_length` (`soliton.py:147`) | Agrawal 3.2 | `matches` | |
| C5 | `Dispersion.solitonOrder` `N = sqrt(gamma*P*T0^2/abs(beta2))` | `soliton_order` (`soliton.py:121`) | Agrawal 3.2.1 | `matches` | |
| C6 | **`Dispersion.soliton_order_sq`** `N^2 = L_D/L_NL` | relation between the three above | Agrawal 3.2.1 | `matches` | The theorem that ties the three lengths together. |
| C7 | **`Dispersion.fundamental_soliton`** `N = 1 ↔ L_D = L_NL` | `soliton.py` fundamental-soliton case | Agrawal 3.2.2 | `matches` | Says the fundamental soliton is *exactly* `L_D = L_NL`. |
| C8 | `Dispersion.LD_mono_T0` | `dispersion_length` | — | `matches` | `L_D` grows with `T0²`. |
| C9 | `Dispersion.fissionLength` `L_fiss = L_D/(Nη)` | `fission_length` (`soliton.py:159`) | Agrawal 3.3 | `matches` | |
| C10 | `Dispersion.fissionLength_mono` | as C9 | — | `matches` | Higher-order fission is faster. |
| C11 | `Dispersion.no_fission_at_fundamental` | as C9 | — | `matches` | `N = 1` gives `L_fiss = L_D/η`, no division by `N`. |
| C12 | `Dispersion.groupDelay`, `walkoff` | `gnlse.py` linear operator | Agrawal 4.2 | `matches` | `ΔT = (β₁ₓ−β₁ᵧ)z`. |
| C13 | `Dispersion.groupDelay_neg` | as C12 | — | `matches` | Reversing `β₁` reverses the delay. |
| C14 | **`Dispersion.betaPhase_at_zero`** | `gnlse.py` dispersion operator | — | `matches` | See F10. |
| C15 | `Dispersion.betaPhase_linear_z` | as C14 | — | `matches` | The phase is an integral, hence linear in `z`. |

### F10 — `betaPhase` at `Ω = 0` is `β₀·z`, not `0`

My first draft claimed the truncated propagation phase vanishes at zero
frequency offset. It does not: the `k = 0` term is `β₀·z`, the carrier phase.
Lean rejected it via `Finset.sum_eq_single`, which is the useful failure here —
the statement was structurally wrong rather than merely unproved.

The corrected statement is not a weakening, it is more informative: it says the
`solver` in a rotating frame still has to carry `β₀`, which is exactly why
`gnlse.py` keeps `β₀` rather than factoring it out. The theorem now records that.

## Stage D — the shock operator and Raman causality

`gnlse.py`, read directly. The point of this stage is that a boundary the code
states in a docstring is now a theorem.

| # | Theorem (Lean) | Governs (Python) | Source | State | Notes |
|---|---|---|---|---|---|
| D1 | `Nonlinear.shockRatio` `ω/ω₀ = 1 + x` | shock operator `gnlse.py:573` | Blow & Wood 1989, App. (A5) | `matches` | `x = Ω/ω₀`; `(ω₀+Ω)/ω₀ = 1 + Ω/ω₀` is exact. **Corrected 2026-10-27** from the false `1/(1−x)`. |
| D2 | `Nonlinear.shockFirstOrder` `1 + x` | as D1 | as D1 | `matches` | What the code actually applies. |
| D3 | **`Nonlinear.code_symbol_is_exact`** | shock symbol | — | `matches` | **PROVED.** `shockRatio = shockFirstOrder`: the solver's `1 + Ωτ_shock` **is** `ω/ω₀` bin-by-bin. There is no first-order truncation in this realization. |
| D4 | `Nonlinear.shockRatio_pos` | `_validate_shock_grid` | — | `matches` | The symbol is positive iff `ω₀ + Ω > 0` (`x > −1`); the guard region `x ≤ 1/5` is inside. |
| D5 | `Nonlinear.reciprocalRatio` + `shock_truncation_error` | — | — | `spec-only` | **Counterfactual, kept for history.** `1/(1−x) − (1+x) = x²/(1−x)` is the error a *reciprocal* realization would incur; this codebase does not exhibit it (D3). |
| D6 | `Nonlinear.shock_error_at_limit` / `shock_error_bounded` / `shock_error_nonneg` | — | — | `spec-only` | Reciprocal-counterfactual facts: at `x = 1/5` the reciprocal error would be exactly `1/20`. |
| D7 | `Nonlinear.accuracy_implies_existence` | `_validate_shock_grid` (`gnlse.py`) | — | **corrected 2026-10 (S3 session)** | The theorem was **false as committed**: `x ≤ 1/5` alone does not imply `x > −1` (x = −5 satisfies both). The bin-positivity condition the guard actually feeds is `x < 1` given `0 ≤ x` (Ω_max ≥ 0); the theorem now carries the non-negativity hypothesis and concludes `x < 1`. `linarith` refused the old statement, which is how the error surfaced (plus an unterminated block comment from a duplicated derivation-regime header that silently broke the file's next build). |
| D8 | `Nonlinear.existence_does_not_imply_accuracy` | as D7 | — | `matches` | The converse fails, which is why the code checks both separately. |
| D9 | **`Nonlinear.resolution_floor`** | resolution-floor docstring | Blow & Wood 1989 (bandwidth) | `matches` | `dt ≥ π/(limit·ω₀)`; about 12.9 fs at 1550 nm, matching the comment. |
| D10 | `Nonlinear.ramanCausal` | Raman convolution `gnlse.py` | Agrawal §3.5 | `matches` | `h_R` vanishes at negative delay. |
| D11 | `Nonlinear.ramanDelaySamples` | delay in grid units | — | `matches` | |
| D12 | `Nonlinear.ramanCausal`, `ramanShape` | Raman response `h_R` | Agrawal §3.5 | `matches` | `h_R` zero before the delay, non-negative after. |
| S12 | **`Solvers.orderConds_solvable`** — weights exist satisfying `2w0+w1=1`, `2w0^3+w1^3=0` | `SplitStepEngine(scheme="yoshida4")` | Yoshida 1990 Eqs. (4.5), (4.6) | `matches` | **PROVED.** By IVT; no closed form, no `2^(1/3)`, no numeric literal (§1.6). |
| S13 | **`Solvers.transposed_condition_fails`** — the transposed cubic condition fails unless `w0=0` | as S12 | Yoshida 1990 (weight labelling) | `spec-only` | **PROVED.** Guards the transposition bug that was made in PLAN.md §17 and caught numerically (−7.4, not 0). |
| S14 | **`Solvers.strangStep_exact`** — symmetric split is exact for commuting sub-flows | `SplitStepEngine._strang_step` | Blow & Wood 1989 Eq. (17) | `matches` | **PROVED.** Error budget is entirely in the non-commuting case. |
| S15 | **`Solvers.yoshida4_consistent`** — consistent weights make the composition advance exactly `h` | `SplitStepEngine._advance` | Yoshida 1990 | `matches` | **PROVED.** Consistency alone; pairs with S12. |
| S16 | **`Solvers.strangProp_symmetric`** — `exp(−z) = (exp z)⁻¹` | as S14 | Blow & Wood 1989 | `matches` | **PROVED.** Stated on the *propagator*: `S(−h)A = (S(h)A)⁻¹` is false for general `A`. |
| S17 | **`Nonlinear.shock_factor_neg`** — `1 + iτ∂_t ⇒ 1 + Ωτ` | `SplitStepEngine` `1 + Ω·tau_shock` | Dudley RMP 78, 1135 Eq. (2) | `matches` | **PROVED.** Under the codebase's `e^{-iΩt}` convention. **The sign audit is now a theorem, not a numerical argument.** |
| S18 | **`Nonlinear.shock_factor_pos`** — same operator, opposite convention ⇒ `1 − Ωτ` | (other FFT conventions, e.g. laserfun) | Blow & Wood IEEE JQE 25, 2665 Eq. (18) | `spec-only` | **PROVED.** Resolves the apparent sign disagreement: the two papers agree once conventions are matched. |
| S19 | **`Nonlinear.convention_difference`** — the two symbols differ by exactly `2τΩ` | as S17/S18 | — | `spec-only` | **PROVED.** The whole content of the sign audit. |
| S20 | **`Nonlinear.shock_skews_blue`** — `τ>0, Ω>0 ⇒ factor > 1` | as S17 | — | `matches` | **PROVED.** The blue-skew direction the docstring reports observing. |
| S21 | **`Nonlinear.omega_offset_pos`** — `Ω>0 ⇒ ω = ω₀+Ω > ω₀` | bin → `ω₀+Ω` mapping | Dudley Eq. (2) | `matches` | **PROVED.** `Ω>0` is the blue side, as the sign audit assumes. |

**Stage S3 rows (2026-10, S3 session) — ETDRK4 (`scheme="etdrk4"`)**

| S22 | **`Solvers.ETDRK4.Qcoef_mul_id`** — `x·Q(x) = exp(x/2) − 1` | `SplitStepEngine(scheme="etdrk4")` stage-1 carry `Q` | Cox & Matthews 2002 Eq. (26); Kassam–Trefethen 2005 §4 diagonal | `matches` | **PROVED.** The half-length stage carry is the exact linear propagator `E₂ = exp(x/2)` — the scalar-per-bin form of "the linear operator is carried exactly". |
| S23 | **`Solvers.ETDRK4.f1coef_phi_combo`** — `f₁ = φ₁ − 3φ₂ + 4φ₃` | ETDRK4 final-combination coefficient of `N₀` (`_etdrk4_scalar_coeffs`) | Cox & Matthews 2002 Eq. (29) | `matches` | **PROVED.** Value at `x → 0`: the classical RK4 weight `1/6` — the scheme reduces to RK4 on the non-stiff side. |
| — | (Python) **coefficient evaluation**: exact (Taylor near `|x|<0.5`, defining formulas away) | `_etdrk4_scalar_coeffs` (`gnlse.py`) | Cox & Matthews' own advisory for small '|λh|'; Kassam–Trefethen §3 | `matches` | **Convergence-with-cancellation is floating-point (§11) — not provable here.** What IS checked: the DC breakdown of the naive formulas is reproduced numerically; the full-circle contour mean agrees to 1e-16; K&T's published *half*-circle sampling is phase-destroying for complex `L` (pinned by test). |
| — | (Python) **scoping**: `scheme="etdrk4"` raises `NotImplementedError` for z-dependent hooks / TPA / free carriers | `_etdrk_linear` guard (`gnlse.py`) | Stage S3's conditional gate (PLAN §17.6) | `matches` | The stiff regimes it targets — Raman cascades over km, loss-limited fibre — are exactly the uniform-core cases. |

**Stage S5 rows (2026-10, S5 session) — RK4IP (`scheme="rk4ip"`)**

| # | Theorem (Lean) | Governs (Python) | Source | State | Notes |
|---|---|---|---|---|---|
| S24 | **`Solvers.RK4IP.linStep_comp_inv`, `linStep_comp`** — `linStep φ h ∘ linStep φ (−h) = id`; composition adds steps | `SplitStepEngine._linear_step` half transforms | Hult 2007 §II-D | `matches` | **PROVED.** The transform pair is an involution, so "transform to the picture, advance, transform back" cannot itself bias the field. |
| S25 | **`Solvers.RK4IP.rk4ipStep_linear`** — with `N = 0` the step is exactly `linStep φ h` | `_rk4ip_step` with `γ = 0` | Hult 2007 Eq. (12) | `matches` | **PROVED.** The outer-level interaction picture carries the linear operator *without* approximation — there is no `[L,N]` splitting residual in the flow. The scalar constant-coefficient shadow of this is `RK4.ipStep_linear` (row H7). |
| S26 | **`Solvers.RK4IP.rk4ipStep_no_dispersion`, `linStep_zero`, `classicalRK4`** — `L = 0 ⇒ rk4ipStep = classical RK4` | `_rk4ip_step` with `betas = []` | Hult 2007 Eq. (12); `RK4.lean` | `matches` | **PROVED.** Every half transform is the identity and the scheme is exactly the classical RK4 combination whose order-4 condition `RK4.lean` proves. The bridge between the interaction-picture formulation and the trusted integrator. |
| S27 | **`Solvers.RK4IP.hasDerivAt_ipPhase`, `interactionPicture_removes_linear`** — `A' = iφA + N(A) ⇒ A_I' = e^{−iφ(z−z₀)}·N(A)` | the outer transform `exp(∓(z−z̄)L̂)` | Hult 2007 §II-D; Hochbruck–Ostermann 2010 | `matches` | **PROVED.** The continuous statement of *why* the picture removes the linear term: the `iφA` parts cancel exactly. |
| S28 | (Python) **fourth-order rate, (A5) conservation, and scope guard** | `_rk4ip_step` (`gnlse.py`) | — | `matches` | **Convergence order is floating-point (§11) — not provable here.** Pinned numerically by `tests/test_gnlse_rk4ip.py`: scalar order-4 halving ratios; N=3 soliton field order ≈ 3.96 (vs `strang` 2.00, `etdrk4` 3.99 on the same fixed-step ladder); (A5) photon number conserved to <1e-10 with Raman+shock; agreement with `etdrk4`; `NotImplementedError` for TPA / free carriers (they live in the split-step nonlinear stage, not in `N̂`). |

| D13 | `Nonlinear.ramanDrive_early` | Raman convolution | — | `matches` | The drive vanishes before the delay has elapsed. |
| D14 | `Nonlinear.ramanDrive_nonneg` | as D13 | — | `matches` | The drive never subtracts energy. |
| D15 | **`Nonlinear.ramanDrive_le_max`** | as D13 | — | `matches` | **PROVED.** Termwise `h_R(j−i) ≤ ∑ h_R`, then `∑ h_R · ∑ P ≤ ∑ P`. |

### F11 — the shock guard: what `_SHOCK_TAYLOR_LIMIT = 0.2` really buys (rewritten 2026-10-27)

The old version of this section derived that the guard admits at most 5 %
"first-order truncation error" from `1/(1−x) − (1+x) = x²/(1−x)`. **That
premise was wrong**: with `ω = ω₀ + Ω`, `ω/ω₀ = (ω₀+Ω)/ω₀ = 1 + Ω/ω₀ = 1 + x`
*exactly* — the solver's symbol is the ratio itself (`code_symbol_is_exact`),
so there is no truncation error for the guard to bound. The reciprocal
theorems are kept in `Shock.lean` relabelled as counterfactuals.

What the guard does buy, per Blow & Wood (1989) pp. 2667/2672 (read from
`formal/refs/blow.pdf`): they derive the GNLS assuming "the bandwidth of
U(ω') is less than ω₀/3". The guard `x = Ω_max/ω₀ ≤ 1/5` bounds the simulated
window (edge to edge) by `2x·ω₀ ≤ 2ω₀/5` (`window_of_guard`).

**Routed to literature, not resolved:** `2/5 > 1/3`, so the code's `0.2`
sits slightly *outside* Blow & Wood's stated derivation regime; strict
adherence would need `x < 1/6` (`bw_regime_needs_sharper`). The constant is
left at `0.2` (changing it would newly warn on grids the project's decks
already use) and the discrepancy is recorded here as an open judgement call for
the library owner, with the B&W citation on the row. Measured photon-number
conservation under the correct (A5) invariant is ≤5×10⁻⁵ relative even at
`x = 0.88`, so this is a derivation-regime bookkeeping question, not a
conservation defect.

So the honest description of `_SHOCK_TAYLOR_LIMIT = 0.2` is **"first-order
accuracy to within 5 %"**, and the constant is not a fudge factor.

The docstring's measured drifts are a *different* quantity (grid artefacts in an
integrator), not the truncation error, and the two differ by more than an order
of magnitude at the top of the range:

| `τ·Ω_max` | exact truncation error | photon drift quoted in the docstring |
|---|---|---|
| 0.073 | 0.57 % | ≤ 0.3 % |
| 0.145 | 2.46 % | 3.1 % |
| 0.200 | 5.00 % | (at the limit) |
| 0.290 | 11.85 % | 5.2 % |

The docstring presents the drift column as evidence that the limit is
conservative. It is not evidence of that: at `τ·Ω_max = 0.29` the drift is
*smaller* than the truncation error, which is possible only because the two are
not the same quantity. Conflating them is what makes the argument feel shaky
rather than the constant.

This is a documentation improvement, not a code change: the constant stays at
0.2, but its justification is now a theorem rather than a table of numbers.

### F1 — Python does not enforce the positivity hypotheses (gap)

Every `Constants` theorem carries `0 < λ`, `0 < f`, `0 < c`. None of these are
checked in Python. Measured:

```
Wavelength(0,    "nm").to_freq()  ->  ZeroDivisionError
Wavelength(-1550, "nm").to_freq()  ->  -193.4145 THz     (silently)
```

The theorems are correct as stated. The code silently accepts a negative
wavelength and returns a negative frequency, which no theorem licenses.

Not a bug in the physics, and not something to "fix" by adding assertions
across the whole library. But the spec/code contract should say which
quantities are assumed positive. **Action:** document the domain in
`CONFORMANCE.md` §Domains (below) and cite it from the relevant docstrings.

### F2 — Wavenumber notation mismatch (disputed)

Lean and Python disagree about what "wavenumber" means.

| | Definition | At λ = 1550 nm |
|---|---|---|
| `Constants.k` (Lean) | `ω/c` = `2π/λ`, angular wavenumber | 4.0537e6 /m |
| `AngularFrequency.to_wn` (Python) | `ω/(2πc)` = `1/λ` | 6451.6 1/cm |

The Python value is the correct one for its label: `Wavelength.to_wn` returns
`1/as_m`, and `1/(1.55e-6 m) = 6451.6 cm⁻¹`. All three Python conversions agree
with each other (`to_wn`, `to_omega().to_wn()`, and `f/c` all give 6451.6), so
the code is internally consistent.

The Lean side is the incomplete one: it has the angular wavenumber but no
`k̃ = ω/(2πc)`, which is what the library actually computes.

**Resolved: fix Lean, not Python.** `Constants.lean` now defines `kTilde`
alongside `k`, with:

- `kTilde_eq` — `k̃ = ω/(2πc)`
- `k_of_kTilde` — `k = 2π·k̃`
- `kTilde_of_wavelength` — `k̃ = 1/λ` under `λω = 2πc`

All three proved, build green. The code was right; the spec was incomplete.

`k = ω/c` is now registered `spec-only`: the library never computes the angular
wavenumber, so nothing in Python governs it.

### F3 — `Z0` is ungoverned (gap)

`base.py:18` computes the vacuum impedance as `1/(ε₀·c)`. Its agreement with
`μ₀` rests on the physical identity `μ₀ε₀c² = 1`, which is **true physics but
not a theorem** — it depends on the measured constants.

Lean currently has `eps0`, `mu0`, `c` as three independent opaque reals with
no relation. Nothing states that they are consistent, so nothing catches a
future edit that changes one of them.

**Action:** add the consistency as an explicit `axiom` in `Constants.lean`,
stating it as a physical assumption rather than pretending it is provable, and
give `Z0` a definition plus a theorem under that assumption.

### F4 — `as_meep` has no base-length parameter while `from_meep` does (API gap, Python side)

`Wavelength.from_meep(value, base_length)` is general in the base length,
defaulting to 1 µm. `Wavelength.as_meep` takes no argument and hardcodes `a = 1 µm`:

```python
@cached_property
def as_meep(self) -> float:
    return 1e-6 / self.as_m
```

Neither direction is *wrong*. The consequence is narrower than first recorded:
the two are inverses only at the 1 µm default, so a caller working in meep units
with a different `a` can convert **into** those units but has no way to convert
**out**. Re-verified with the argument actually passed:

```
w = Wavelength(1550, "nm");  w.as_meep = 0.6451612903225805      # uses a = 1 µm
Wavelength.from_meep(w.as_meep, Wavelength(1000, "nm")) -> 1550.0 nm   # inverse ✓
Wavelength.from_meep(w.as_meep, Wavelength(1550, "nm")) -> 2402.5 nm   # not inverse
Wavelength.from_meep(w.as_meep, Wavelength(2000, "nm")) -> 3100.0 nm   # not inverse
```

(The second line is the only case that round-trips, and it is exactly the
default.)

**Action:** either give `as_meep` a `base_length` argument for symmetry, or
state in its docstring that the MEEP base length is fixed at 1 µm and that
`from_meep` with any other base length is not its inverse.

### Stage K — the multimode GNLSE coefficient algebra

`multimode_gnlse.py` and `mode_analysis.py`, read during the 2026-10
literature audit. The modal-overlap **integrals** stay measure-theoretic
(same boundary as G10); what is formalised here is the coefficient algebra
the engine composes from them.

| # | Theorem (Lean) | Governs (Python) | Source | State | Notes |
|---|---|---|---|---|---|
| K1 | `Nonlinear.xpmFactor_lpDegenerate` | `_LP_XPM = 2/3` (`multimode_gnlse.py:83`) | Agrawal §6.3 | `matches` | Two orthogonal **linear polarizations**; this is the vector-GNLSE strip-down, not the co-polarized spatial-mode XPM. |
| K2 | `Nonlinear.xpmFactor_isotropic` | `"isotropic"` branch (`_xpm_factor`) | — | `matches` | All-ones model. |
| K3 | **`Nonlinear.xpmFactor_coPolarized`** | `_CO_POL_XPM = 2`; `coef_model="co_polarized"` | Mumtaz Eq. 6 scalar limit | `matches` | **The audit fix.** SPM 1, XPM 2 for two co-polarized spatial modes: two equal XPM index permutations. The old default (2/3) under-counted this by 3×. |
| K4 | `Nonlinear.fwmFactor_coPolarized` | `_CO_POL_FWM = 1` | Mumtaz Eq. 6 | `matches` | Pump-driven FWM weight in the scalar limit. |
| K5 | `Nonlinear.coPolarizedRate_eq` | `_diagonal_phase` with `coef_model="co_polarized"` | Agrawal §6.3 | `matches` | `γ(P_i + 2P_j)`, the two-mode instantaneous phase rate. |
| K6 | **`Nonlinear.fwmAllowed_partner`** | `_fwm_allowed`; both exchange arms in `_fwm_rhs` | Poletti Eq. 18 | `matches` | One gate check serves the pair: `ℓ_m = 2ℓ_n − ℓ_q` implies `ℓ_q = 2ℓ_n − ℓ_m`. |
| K7 | `Nonlinear.fwmAllowed_pump_swap` | FWM sum over `(n, p)` | Poletti Eq. 18 | `matches` | The gate is symmetric in the two pump indices. |
| K8 | **`Nonlinear.depletedFwm_energy_conserved`** | `fwm_pump_depletion` pump arm (`+2iγf* A_mA_qA_n*`) | Mumtaz Eq. 6 / Manley–Rowe | `matches` | Sum of `Re(conj(A)·dA)` over the three modes is exactly zero for real coupling. |
| K9 | `Nonlinear.depletedFwm_normSq_rate` | `Σ|A|²` monitor / RK4 conservation test | — | `matches` | The same identity with the `2` of `d|A|²/dz`. |
| K10 | `Nonlinear.fwm_pair_only_not_conserved` | pump-driven (non-depleted) approximation | — | `matches` | Concrete witness that the two pair arms alone do not conserve `Σ|A|²`; the pump arm is required. |
| K11 | `Nonlinear.shockSymbol_sq` | `_apply_shock_rhs`; step10 regression | Wright Eq. 1; Shock.lean S17 | `matches` | Applying `(1+Ωτ)` twice gives `(1+Ωτ)²`. |
| K12 | `Nonlinear.shockSymbol_double_sub` | as K11 | — | `matches` | `shock² − shock = Ωτ(1+Ωτ)`. |
| K13 | **`Nonlinear.shockSymbol_double_ne_single`** | step10 exactly-once regression | — | `matches` | For a physical bin the squared symbol is different — the `(1+iτΩ)²` bug the test pins. |

K3, K4 and K8 are the load-bearing rows: they turn the audit's coefficient
choices into theorems. The overlap **scale** (`C_ij = γ₀·xpm_ij =
(n₂ω₀/c)Q⁽¹⁾_iijj`) remains a statement about integrals and is recorded as
`uncovered` under G10's boundary.

---

## Domains assumed by the proved theorems

From A1–A8 and F1. Every Lean constant theorem assumes its wavelength,
frequency, or angular frequency is positive. The Python API does not enforce
this. Consumers of the specification should treat the following as preconditions:

- `λ > 0` for `to_freq`, `to_omega`, `to_energy`, `to_wn`
- `f > 0` for `to_omega`, `to_energy`, `to_wn`
- `ω > 0` for `to_freq`, `to_energy`, `to_wn`
- `c > 0`, `h > 0`, `ε₀ > 0`, `μ₀ > 0` (CODATA exact values satisfy these)

---

## Coverage so far

- Rows in the register: **158**
- Matching Python symbols: **~80**
- Findings raised: **11** (F1–F11), plus 6 corrected statements in `CHANGELOG.md`
- Findings resolved: **3** (F2 fixed in Lean; F5 and F6 were false statements
  of mine, corrected before they were relied on)
- Open: **3** (F1 documentation, F3 axiom *written*, F4 Python decision)
- Open theorems: **0** — `formal/` builds green with no `sorry`
- New: `Solvers/Composition.lean` (8 theorems, S12–S16) — the Suzuki/Yoshida
  composition algebra for `scheme="yoshida4"`; order conditions proved
  existentially, with no closed form
- New: sign-convention theorems in `Nonlinear/Shock.lean` (S17–S21) — the
  shock factor `1 + Ωτ` vs `1 − Ωτ` resolved as a *Fourier-convention*
  consequence; Dudley and Blow–Wood shown to agree
- New: `Solvers/RK4IP.lean` (7 theorems, S24–S27) — the outer-level
  interaction-picture algebra for `scheme="rk4ip"`: transform involution,
  exact linear flow, collapse to classical RK4 when `L = 0`, and the
  continuous interaction-picture removal of the linear term
- New: `Nonlinear/Multimode.lean` (13 theorems, K1–K13) — the multimode
  coefficient models (including the co-polarized spatial-mode XPM = 2),
  the FWM selection-rule partner symmetry, exact `Σ|A|²` conservation of
  the depleted pump set, and the shock-applied-twice symbol identity

F2 and F6 are the worked examples of why this correlation runs concurrently.
Neither `k = ω/c` nor `ω_{-1,mirror} = -ω_1` was a hard theorem; both were easy
to state and wrong. Lean would never have complained on its own, and in F6 the
stated theorem was false in a way that a numeric spot-check at one index would
have missed.