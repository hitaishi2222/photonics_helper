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
| A12 | `Conventions.dft_pair`, `conj_dft`, `hermitian` | `TemporalGrid.fft` / `.ifft` (`core/grids.py`) | DFT | `uncovered` | **Open (Stage B).** Theorems not yet proved. |
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
| B10 | `Conventions.conj_dft` | `TemporalGrid.fft` (`conj(DFT(conj(·)))`) | DFT | `uncovered` | **Open.** Needs the origin-bin identity `(ω_0+ω_0)·t_j = 2π(n/2−j)` only; the reindexing is proved. |
| B11 | `Conventions.hermitian` | (no direct symbol) | DFT | `spec-only` | True as stated; is B10 restricted to real `A`, not a separate theorem. |
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
| E4 | — | `spectrum` `R`, `T` | Macleod §2.6 | `uncovered` | `R + T = 1` for lossless media. Statement designed, not proved. |
| E5 | — | `_admittance` (`dbr.py:379`) | Macleod §2.4 | `uncovered` | `η = n cos θ` (TE) / `n/cos θ` (TM). |
| E6 | — | Snell `cos θ_layer` (`dbr.py:294`) | Born & Wolf §1.6 | `uncovered` | Includes the `np.clip` guard, a *numerical* assumption. |
| E7 | — | `layerIndex` order | Macleod Eq. 2.55 | `uncovered` | Layer order recorded in `Conventions.lean`, not yet given a theorem. |

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
| E19 | — | `delta_k_shg` (`chi2.py:100`) | — | `uncovered` | `Δk = β(2ω) − 2β(ω)`. |
| E20 | — | `shg_coupling_overlap` (`chi2.py:661`) | — | `uncovered` | Poling-overlap reduction. |

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

E4–E7, E19, E20 remain open.

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
| F15 | — | `mi_gain_spectrum` (`:598`) | Agrawal §5.1.2 | `uncovered` | The gain expression. |
| F16 | — | `mi_gain_spectrum_extended` (`:699`) | — | `uncovered` | Offset-aware variant; deprecated path still live. |
| F17 | — | `dispersive_wave_roots` (`:868`) | — | `uncovered` | RDW phase-matching roots. |
| F18 | — | `fwm_efficiency` (`:448`) | Agrawal §5.2 | `uncovered` | |
| F19 | — | `_taylor_phase_max_error` (`:1066`) | — | `uncovered` | Dispersion truncation error; pairs with `Dispersion.betaPhase`. |

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

F15–F19 remain open. Still unread: `gnlse.py` beyond its shock terms,
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
genuinely stronger claim and needs lossless media; it is not yet proved, but the
absorbing row is the counterexample that pins down why it needs the hypothesis.

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
| H6 | `Solvers.rk4_error_le` | as H1 | — | **partial** | See F18. |
| H7 | — | `gnlse.py` RK4-in-interaction-picture | — | `uncovered` | The IP transform, not formalised. |
| H8 | — | SSFM operator composition | — | `uncovered` | `PLAN.md` §9.1. |

H5 is the load-bearing one: it says exactly what "RK4IP" buys, namely that the
one-step map for a (near-)linear propagator is the fourth-order Taylor sum of
the exponential.

### F18 — `rk4_error_le` carries its own content in a hypothesis
(`Real.sum_le_exp`, `Real.taylor_mean_remainder_lagrange` and
`Real.exp_le_exp_mul_1_sub` are all absent). So the theorem takes the expansion
`exp z = Σ_{k≤4} zᵏ/k! + z⁵/20` as a *hypothesis* and proves only that a
non-negative remainder makes the truncated sum a lower bound.

Recorded as `partial` rather than `matches` for that reason. The algebraic
content (H4, H5) is fully proved; the analytic bound is not.

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

## Stage D — the shock expansion and Raman causality

`gnlse.py`, read directly. The point of this stage is that a boundary the code
states in a docstring is now a theorem.

| # | Theorem (Lean) | Governs (Python) | Source | State | Notes |
|---|---|---|---|---|---|
| D1 | `Nonlinear.shockRatio` `ω/ω₀ = 1/(1−x)` | shock operator `gnlse.py:573` | Blow & Wood 1989, Agrawal §2.3 | `matches` | `x = Ω/ω₀`. |
| D2 | `Nonlinear.shockFirstOrder` `1 + x` | as D1 | as D1 | `matches` | What the code actually applies. |
| D3 | **`Nonlinear.shock_truncation_error`** | `_SHOCK_TAYLOR_LIMIT` (`gnlse.py:30`) | — | `matches` | `1/(1−x) − (1+x) = x²/(1−x)`, **exactly**. |
| D4 | `Nonlinear.shock_error_at_limit` | `_SHOCK_TAYLOR_LIMIT` | — | `matches` | At `x = 1/5` the error is exactly `1/20`. |
| D5 | **`Nonlinear.shock_error_bounded`** | `_SHOCK_TAYLOR_LIMIT = 0.2` | — | `matches` | The guard admits at most **5 %** truncation error. |
| D6 | `Nonlinear.shock_error_nonneg` | as D3 | — | `matches` | The linearisation is optimistic for `x ≥ 0`. |
| D7 | `Nonlinear.accuracy_implies_existence` | `_validate_shock_grid` (`gnlse.py:709`) | — | `matches` | `x ≤ 0.2` already implies `x < 1`. |
| D8 | `Nonlinear.existence_does_not_imply_accuracy` | as D7 | — | `matches` | The converse fails, which is why the code checks both separately. |
| D9 | **`Nonlinear.resolution_floor`** | resolution-floor docstring | Agrawal §2.3 | `matches` | `dt ≥ π/(limit·ω₀)`; about 12.9 fs at 1550 nm, matching the comment. |
| D10 | `Nonlinear.ramanCausal` | Raman convolution `gnlse.py` | Agrawal §3.5 | `matches` | `h_R` vanishes at negative delay. |
| D11 | `Nonlinear.ramanDelaySamples` | delay in grid units | — | `matches` | |
| D12 | `Nonlinear.ramanCausal`, `ramanShape` | Raman response `h_R` | Agrawal §3.5 | `matches` | `h_R` zero before the delay, non-negative after. |
| D13 | `Nonlinear.ramanDrive_early` | Raman convolution | — | `matches` | The drive vanishes before the delay has elapsed. |
| D14 | `Nonlinear.ramanDrive_nonneg` | as D13 | — | `matches` | The drive never subtracts energy. |
| D15 | `Nonlinear.ramanDrive_le_max` | as D13 | — | `uncovered` | **Open (Stage D).** The gain bound; statement written, proof not finished. |

### F11 — `_SHOCK_TAYLOR_LIMIT = 0.2` is conservative, and now provably so

The code's docstring argues the limit is conservative by citing measured
photon drift (0.073, 0.145, 0.29). That is empirical and model-specific.

`Nonlinear.shock_truncation_error` replaces it with an **exact** identity:

```
1/(1 − x) − (1 + x) = x²/(1 − x)     exactly, for x ≠ 1
```

so the guard's meaning is computable rather than measured:

- at the limit `x = 1/5` the error is exactly `1/20` (D4);
- below the limit it is at most `1/20` (D5).

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

- Rows in the register: **140**
- Matching Python symbols: **~66**
- Findings raised: **11** (F1–F11), plus 6 corrected statements in `CHANGELOG.md`
- Findings resolved: **3** (F2 fixed in Lean; F5 and F6 were false statements
  of mine, corrected before they were relied on)
- Open: **3** (F1 documentation, F3 axiom *written*, F4 Python decision)
- Open theorems: **3** (`conj_dft`, `hermitian`, `ramanDrive_le_max`)

F2 and F6 are the worked examples of why this correlation runs concurrently.
Neither `k = ω/c` nor `ω_{-1,mirror} = -ω_1` was a hard theorem; both were easy
to state and wrong. Lean would never have complained on its own, and in F6 the
stated theorem was false in a way that a numeric spot-check at one index would
have missed.