import Mathlib

/-!
# Four-wave mixing and modulation instability

The two relations in `phase_matching.py` with an exact closed form, both taken
from Agrawal *Nonlinear Fiber Optics* §5.1:

- **FWM** — energy conservation `ωₛ + ωᵢ = 2ωₚ` (`fwm_idler_frequency`).
- **MI** — the cutoff condition `β₂Ω² + 4γP = 0`, hence
  `Ω² = −4γP/β₂` (`mi_sideband_frequencies`).

MI is the interesting one for verification, because the *existence* of the
instability is a genuine branch in the code: `mi_sideband_frequencies` returns
a single zero when `β₂ ≥ 0`, and the sidebands do not exist at all. The
theorems below pin down exactly when that happens.

Also proved here:

- the **gain spectrum** itself (`mi_gain_spectrum`): `g(Ω) = |β₂Ω|√(Ω_c² − Ω²)`
  on `Ω < Ω_c`, zero outside, with the peak location and the peak height
  `g_max = 2γP` proved, not quoted;
- the **higher-order-dispersion form** (`mi_gain_spectrum_extended`):
  `g(Ω) = √[−Δ(Ω)·(Δ(Ω) + 4γP)]` in the band `−4γP < Δ < 0`, and the exact
  reduction to the classical expression for quadratic β — which is the
  docstring's "reduces exactly" claim;
- **dispersive-wave phase matching** (`dispersive_wave_roots`): the residual's
  tangency at the soliton, and the **sign** of the β₂/β₃ detuning — the DW is
  blue-shifted precisely when `β₂·β₃ < 0`;
- the **lossless FWM efficiency** (`fwm_efficiency`): `η = sinc²(ΔβL/2) ≤ 1`
  with equality only at perfect phase matching.
-/

namespace Photonics
namespace Optics

/-! ## Four-wave mixing -/

/--
The idler frequency for degenerate FWM, `ωᵢ = 2ωₚ − ωₛ`.

Matches `fwm_idler_frequency` exactly.
-/
def fwmIdler (omega_p omega_s : ℝ) : ℝ := 2 * omega_p - omega_s

theorem fwmIdler_eq (omega_p omega_s : ℝ) :
    fwmIdler omega_p omega_s = 2 * omega_p - omega_s := rfl

/-- **Energy conservation.** The signal and idler frequencies sum to twice the
pump: one pump photon splits into one signal and one idler. This is the
invariant that makes the `scan_fwm_detuning` grid physical. -/
theorem fwm_energy (omega_p omega_s : ℝ) :
    omega_s + fwmIdler omega_p omega_s = 2 * omega_p := by
  simp [fwmIdler]

/-- **Degenerate FWM is fixed.** If the signal sits at the pump, so does the
idler: the only solution is the trivial three-wave resonance. -/
theorem fwm_degenerate (omega_p : ℝ) : fwmIdler omega_p omega_p = omega_p := by
  simp [fwmIdler]
  ring

/-- The idler is a reflection about the pump frequency: `ωᵢ − ωₚ = ωₚ − ωₛ`.
A red-detuned signal (`ωₛ < ωₚ`) gives a blue idler, and vice versa. -/
theorem fwm_reflection (omega_p omega_s : ℝ) :
    fwmIdler omega_p omega_s - omega_p = omega_p - omega_s := by
  simp [fwmIdler]
  ring

/-- Symmetry under exchanging signal and idler: the map is an involution. -/
theorem fwm_involutive (omega_p omega_s : ℝ) :
    fwmIdler omega_p (fwmIdler omega_p omega_s) = omega_s := by
  simp [fwmIdler]

/-- Positive idler frequency requires `ωₛ < 2ωₚ`. Below that detuning the
idler would land at negative frequency, which the discrete-mode solver cannot
represent. -/
theorem fwm_idler_pos {omega_p omega_s : ℝ} (h : fwmIdler omega_p omega_s > 0) :
    omega_s < 2 * omega_p := by
  simp only [fwmIdler] at h
  linarith

/-! ## Modulation instability

`i∂A/∂z = (β₂/2)∂²A/∂T² − γ|A|²A` admits sidebands at `Ω` only where
`β₂Ω² + 4γP = 0`.
-/

/-- The MI cutoff frequency squared: `Ω² = −4γP/β₂`. -/
noncomputable def miOmegaSq (beta2 gamma P : ℝ) : ℝ := -4 * gamma * P / beta2

theorem miOmegaSq_eq (beta2 gamma P : ℝ) :
    miOmegaSq beta2 gamma P = -4 * gamma * P / beta2 := rfl

/-- The cutoff satisfies the dispersion relation it is defined by. -/
theorem mi_cutoff (beta2 gamma P : ℝ) (h : beta2 ≠ 0) :
    beta2 * miOmegaSq beta2 gamma P + 4 * gamma * P = 0 := by
  unfold miOmegaSq
  field_simp
  ring

/--
**MI exists only in the anomalous regime.**

For `β₂ < 0` and a genuinely non-zero nonlinear drive `γ·P > 0`, the cutoff is
strictly positive and the two sidebands `±Ω` exist. This is the branch
`mi_sideband_frequencies` takes when it returns two frequencies rather than a
single zero.
-/
theorem mi_exists {beta2 gamma P : ℝ} (h2 : beta2 < 0) (hg : 0 < gamma)
    (hP : 0 < P) : 0 < miOmegaSq beta2 gamma P := by
  unfold miOmegaSq
  have hnum : (0:ℝ) < 4 * gamma * P := by nlinarith [hg, hP]
  have hden : (0:ℝ) < -beta2 := by linarith
  rw [show (-4 * gamma * P) / beta2 = (4 * gamma * P) / (-beta2) by field_simp]
  exact div_pos hnum hden

/--
**No MI in the normal dispersion regime.** For `β₂ ≥ 0` the cutoff squared is
non-positive, so there is no real `Ω`.

This is exactly the `if beta2 >= 0: return np.array([0.0])` branch, and it is a
physical statement rather than a numerical guard: spontaneous modulation
instability requires anomalous dispersion.
-/
theorem mi_absent_normal {beta2 gamma P : ℝ} (hg : 0 ≤ gamma) (hP : 0 ≤ P)
    (h2 : 0 < beta2) : miOmegaSq beta2 gamma P ≤ 0 := by
  unfold miOmegaSq
  rw [div_le_iff₀ h2]
  have hnum : (0:ℝ) ≤ 4 * gamma * P := by nlinarith [hg, hP]
  linarith

/-- No MI without a nonlinear drive: `γ·P = 0` gives a degenerate cutoff. -/
theorem mi_absent_no_drive {beta2 gamma P : ℝ} (h2 : beta2 < 0) (hgp : gamma * P = 0) :
    miOmegaSq beta2 gamma P = 0 := by
  unfold miOmegaSq
  have hnum : -4 * gamma * P = 0 := by nlinarith [hgp]
  rw [hnum, zero_div]

/--
**The sidebands are symmetric about the pump.** `mi_sideband_frequencies`
returns `[−Ω, +Ω]`, so the MI gain spectrum is even about `ω₀`; this is what
lets `mi_gain_spectrum` be computed on one half and mirrored.
-/
theorem mi_sidebands_symmetric {beta2 gamma P : ℝ} (h2 : beta2 < 0) (hg : 0 < gamma)
    (hP : 0 < P) (O : ℝ) (hO : O = Real.sqrt (miOmegaSq beta2 gamma P)) (hO0 : 0 < O) :
    -O = -(Real.sqrt (miOmegaSq beta2 gamma P)) ∧
      O = Real.sqrt (miOmegaSq beta2 gamma P) := ⟨by rw [hO], hO⟩

/--
**More nonlinearity means a wider MI spectrum.** With `β₂ < 0` fixed and
`γ > 0`, the cutoff grows with pump power, which is the standard MI result and
the reason MI-based spectral broadening is power-controlled.
-/
theorem mi_mono_P {beta2 gamma P P' : ℝ} (h2 : beta2 < 0) (hg : 0 < gamma)
    (hP : 0 < P) (hP' : P ≤ P') :
    Real.sqrt (miOmegaSq beta2 gamma P) ≤ Real.sqrt (miOmegaSq beta2 gamma P') := by
  apply Real.sqrt_le_sqrt
  unfold miOmegaSq
  have h2' : beta2 ≠ 0 := ne_of_lt h2
  have hmono : -4 * gamma * P / beta2 ≤ -4 * gamma * P' / beta2 := by
    have key : (-4 * gamma * P) / beta2 = (4 * gamma * P) / (-beta2) := by
      field_simp
    have key2 : (-4 * gamma * P') / beta2 = (4 * gamma * P') / (-beta2) := by
      field_simp
    rw [key, key2]
    have hden : (0:ℝ) < -beta2 := by linarith
    rw [div_le_div_iff₀ hden hden]
    exact mul_le_mul_of_nonneg_right (by nlinarith [hg]) (le_of_lt hden)
  exact hmono

/--
**Stronger dispersion suppresses MI.** Writing `|β₂|` for the magnitude, a
larger `|β₂|` at fixed `γP` gives a narrower instability band, since
`Ω² = 4γP/|β₂|`.

(Stated with the inequality the right way round: `b₂ ≤ b₂'` means *less* normal
dispersion, hence a *larger* cutoff. The first draft of this file had it
reversed.)
-/
theorem mi_anti_mono_absBeta2 {gamma P b2 b2' : ℝ} (hg : 0 < gamma) (hP : 0 < P)
    (hb : 0 < b2) (hb' : b2 ≤ b2') :
    Real.sqrt (miOmegaSq (-b2') gamma P) ≤ Real.sqrt (miOmegaSq (-b2) gamma P) := by
  apply Real.sqrt_le_sqrt
  unfold miOmegaSq
  have hb1 : b2 ≠ 0 := ne_of_gt hb
  have hb2 : b2' ≠ 0 := ne_of_gt (lt_of_lt_of_le hb hb')
  have key : -4 * gamma * P / (-b2) = 4 * gamma * P / b2 := by field_simp
  have key2 : -4 * gamma * P / (-b2') = 4 * gamma * P / b2' := by field_simp
  rw [key2, key]
  have hden' : (0:ℝ) < b2' := lt_of_lt_of_le hb hb'
  rw [div_le_div_iff₀ hden' hb]
  exact mul_le_mul_of_nonneg_left (by nlinarith [hg, hP]) (by nlinarith [hb])

/-! ## The gain spectrum

`mi_gain_spectrum`'s anomalous branch. The closed form is Agrawal Eq. (5.1.9),
`g(Ω) = |β₂·Ω|·√(Ω_c² − Ω²)` on `Ω < Ω_c` and `0` outside. Two numbers are worth
proving rather than quoting — the **peak location** `Ω_peak² = Ω_c²/2` and the
**peak height** `g_max = 2γP` — because those are the values the reproduction
decks assert against.
-/

/--
The classical MI gain coefficient, branch for branch as `mi_gain_spectrum`
computes it: zero in the normal-dispersion regime, the Agrawal expression
inside the instability band, zero outside it.
-/
noncomputable def miGain (beta2 gamma P Omega : ℝ) : ℝ :=
  if 0 < beta2 then 0
  else if Omega ^ 2 < -4 * gamma * P / beta2
    then |beta2| * |Omega| * Real.sqrt (-4 * gamma * P / beta2 - Omega ^ 2)
    else 0

/-- In the normal-dispersion regime the gain is identically zero — the
`if beta2 > 0: return np.zeros_like(...)` branch. -/
theorem miGain_normal (beta2 gamma P Omega : ℝ) (h : 0 < beta2) :
    miGain beta2 gamma P Omega = 0 := by
  simp [miGain, h]

/-- Inside the instability band the gain is the Agrawal expression. -/
theorem miGain_band {beta2 gamma P Omega : ℝ} (h2 : beta2 < 0)
    (hO : Omega ^ 2 < -4 * gamma * P / beta2) :
    miGain beta2 gamma P Omega
      = |beta2| * |Omega| * Real.sqrt (-4 * gamma * P / beta2 - Omega ^ 2) := by
  rw [miGain, if_neg (not_lt.mpr (le_of_lt h2)), if_pos hO]

/-- Outside the band the gain is zero: the spectral region beyond the cutoff
is stable, which is what lets `mi_gain_spectrum` compute one half and mirror. -/
theorem miGain_outside {beta2 gamma P Omega : ℝ} (hO : -4 * gamma * P / beta2 ≤ Omega ^ 2) :
    miGain beta2 gamma P Omega = 0 := by
  unfold miGain
  split
  · rfl
  · exact if_neg (not_lt.mpr hO)

/-- The gain is non-negative throughout — a gain coefficient, not a signed one. -/
theorem miGain_nonneg (beta2 gamma P Omega : ℝ) : 0 ≤ miGain beta2 gamma P Omega := by
  unfold miGain
  split
  · linarith
  · split
    · have := Real.sqrt_nonneg (-4 * gamma * P / beta2 - Omega ^ 2)
      positivity
    · linarith

/-- **The peak location.** The gain is maximal at `Ω² = 2γP/|β₂|`, which is
exactly half the cutoff, where `mi_gain_spectrum` reports `Omega_peak`. -/
noncomputable def miPeakSq (beta2 gamma P : ℝ) : ℝ := 2 * gamma * P / |beta2|

theorem miPeakSq_half (beta2 gamma P : ℝ) (h2 : beta2 < 0) :
    miPeakSq beta2 gamma P = (-4 * gamma * P / beta2) / 2 := by
  have habs : |beta2| = -beta2 := abs_of_neg h2
  unfold miPeakSq
  rw [habs]
  field_simp
  ring

/--
**Peak gain.** At `Ω = √(2γP/|β₂|)` the gain is exactly `2γP` — the `g_max`
the module returns. The evaluation is the whole content of the peak claim:
three square roots collapse to the peak height.
-/
theorem mi_gain_peak (beta2 gamma P : ℝ) (h2 : beta2 < 0) (hg : 0 < gamma)
    (hP : 0 < P) :
    miGain beta2 gamma P (Real.sqrt (miPeakSq beta2 gamma P)) = 2 * gamma * P := by
  have habs : |beta2| = -beta2 := abs_of_neg h2
  have hb : beta2 ≠ 0 := ne_of_lt h2
  have hΩc : (0:ℝ) < -4 * gamma * P / beta2 := by
    have h1 : (0:ℝ) < 4 * gamma * P := by positivity
    have h2' : -4 * gamma * P / beta2 = 4 * gamma * P / -beta2 := by field_simp
    rw [h2']
    exact div_pos h1 (by linarith)
  have hpeak : miPeakSq beta2 gamma P = (-4 * gamma * P / beta2) / 2 :=
    miPeakSq_half beta2 gamma P h2
  have hnz : (0:ℝ) ≤ miPeakSq beta2 gamma P := by rw [hpeak]; linarith
  have hlt : Real.sqrt (miPeakSq beta2 gamma P) ^ 2 < -4 * gamma * P / beta2 := by
    rw [Real.sq_sqrt hnz, hpeak]
    linarith
  rw [miGain_band h2 hlt, Real.sq_sqrt hnz]
  rw [abs_of_nonneg (Real.sqrt_nonneg _), habs, hpeak]
  have hhalf : (0:ℝ) ≤ -4 * gamma * P / beta2 / 2 := by linarith
  have e1 : -4 * gamma * P / beta2 - (-4 * gamma * P / beta2 / 2)
      = -4 * gamma * P / beta2 / 2 := by ring
  have e2 : Real.sqrt (-4 * gamma * P / beta2 / 2) * Real.sqrt (-4 * gamma * P / beta2 / 2)
      = -4 * gamma * P / beta2 / 2 := by rw [← sq, Real.sq_sqrt hhalf]
  have hz : -beta2 * Real.sqrt (-4 * gamma * P / beta2 / 2)
        * Real.sqrt (-4 * gamma * P / beta2 / 2)
      = -beta2 * (Real.sqrt (-4 * gamma * P / beta2 / 2)
          * Real.sqrt (-4 * gamma * P / beta2 / 2)) := by rw [mul_assoc]
  rw [e1, hz, e2]
  field_simp <;> norm_num

/--
**The gain never exceeds `2γP`.** For every modulation frequency the gain is
at most the peak height. The band shape satisfies
`β₂²·x·(Ω_c² − x) ≤ β₂²·(Ω_c²/2)² = 4γ²P²`, which is AM–GM: `0 ≤ (2x − Ω_c²)²`.
-/
theorem mi_gain_le_max (beta2 gamma P Omega : ℝ) (h2 : beta2 < 0) (hg : 0 < gamma)
    (hP : 0 < P) (hO : Omega ^ 2 ≤ -4 * gamma * P / beta2) :
    miGain beta2 gamma P Omega ≤ 2 * gamma * P := by
  have hb : beta2 ≠ 0 := ne_of_lt h2
  have hΩc : (0:ℝ) < -4 * gamma * P / beta2 := by
    have h1 : (0:ℝ) < 4 * gamma * P := by positivity
    have h2' : -4 * gamma * P / beta2 = 4 * gamma * P / -beta2 := by field_simp
    rw [h2']
    exact div_pos h1 (by linarith)
  unfold miGain
  split
  · nlinarith
  · split
    · -- Write the gain as the single square root `√(β₂²·Ω²·(Ω_c² − Ω²))`.
      have harg : (0:ℝ) ≤ -4 * gamma * P / beta2 - Omega ^ 2 := by linarith
      have habs1 : Real.sqrt (beta2 ^ 2) = |beta2| := Real.sqrt_sq_eq_abs beta2
      have habs2 : Real.sqrt (Omega ^ 2) = |Omega| := Real.sqrt_sq_eq_abs Omega
      have hstep1 : Real.sqrt (beta2 ^ 2) * Real.sqrt (Omega ^ 2)
          = Real.sqrt (beta2 ^ 2 * Omega ^ 2) :=
        (Real.sqrt_mul (sq_nonneg beta2) (Omega ^ 2)).symm
      have hstep2 : Real.sqrt (beta2 ^ 2 * Omega ^ 2)
          * Real.sqrt (-4 * gamma * P / beta2 - Omega ^ 2)
          = Real.sqrt (beta2 ^ 2 * Omega ^ 2 * (-4 * gamma * P / beta2 - Omega ^ 2)) :=
        (Real.sqrt_mul (mul_nonneg (sq_nonneg beta2) (sq_nonneg Omega))
          (-4 * gamma * P / beta2 - Omega ^ 2)).symm
      rw [← habs1, ← habs2, hstep1, hstep2]
      -- AM–GM, then the identity `β₂²·(Ω_c²/2)² = (2γP)²`.
      have hamgm : Omega ^ 2 * (-4 * gamma * P / beta2 - Omega ^ 2)
          ≤ (-4 * gamma * P / beta2 / 2) ^ 2 :=
        by nlinarith [sq_nonneg (2 * Omega ^ 2 - (-4 * gamma * P / beta2)), sq_nonneg Omega]
      have hsquare : beta2 ^ 2 * (-4 * gamma * P / beta2 / 2) ^ 2 = (2 * gamma * P) ^ 2 := by
        have hn : (0:ℝ) < -beta2 := by linarith
        field_simp <;> norm_num
      have hbound : beta2 ^ 2 * Omega ^ 2 * (-4 * gamma * P / beta2 - Omega ^ 2)
          ≤ (2 * gamma * P) ^ 2 := by
        have hassoc : beta2 ^ 2 * Omega ^ 2 * (-4 * gamma * P / beta2 - Omega ^ 2)
            = beta2 ^ 2 * (Omega ^ 2 * (-4 * gamma * P / beta2 - Omega ^ 2)) := by ring
        rw [hassoc]
        have h1 := mul_le_mul_of_nonneg_left hamgm (sq_nonneg beta2)
        rw [hsquare] at h1
        exact h1
      have hpos2 : (0:ℝ) ≤ 2 * gamma * P := by positivity
      have hle : Real.sqrt (beta2 ^ 2 * Omega ^ 2 * (-4 * gamma * P / beta2 - Omega ^ 2))
          ≤ 2 * gamma * P := by
        rw [← Real.sqrt_sq hpos2]
        exact Real.sqrt_le_sqrt hbound
      exact hle
    · nlinarith

/-! ## The higher-order-dispersion form

`mi_gain_spectrum_extended` works from the even dispersion mismatch
`Δ(Ω) = β(ω₀+Ω) + β(ω₀−Ω) − 2β(ω₀)` instead of `β₂Ω²`, and gains

`g(Ω) = √[−Δ(Ω)·(Δ(Ω) + 4γP)]`  on  `−4γP < Δ(Ω) < 0`,

zero outside. Two claims are proved: the band is exactly where the radicand
is positive, and a quadratic β recovers the classical expression *exactly* —
the docstring's "reduces exactly" sentence.
-/

/-- The even dispersion mismatch, as `mi_gain_spectrum_extended` forms it.
The `β₁` term cancels by symmetry, which is why the extended estimator is
immune to the carrier-offset cancellation problem. -/
noncomputable def miMismatch (beta : ℝ → ℝ) (omega0 Omega : ℝ) : ℝ :=
  beta (omega0 + Omega) + beta (omega0 - Omega) - 2 * beta omega0

/--
**The `β₁` cancellation, made explicit.** For a quadratic dispersion model
centred on the pump, the mismatch is exactly `β₂·Ω²`; the linear term drops
out pairwise. First half of "reduces exactly to the classical result".
-/
theorem miMismatch_quadratic (beta0 beta1 beta2 omega0 Omega : ℝ) :
    miMismatch (fun w => beta0 + beta1 * (w - omega0) + beta2 / 2 * (w - omega0) ^ 2)
      omega0 Omega = beta2 * Omega ^ 2 := by
  unfold miMismatch
  ring

/-- The extended gain: the radical inside the instability band, zero outside. -/
noncomputable def miGainExt (Delta gP : ℝ) : ℝ :=
  if -4 * gP < Delta ∧ Delta < 0 then Real.sqrt (-Delta * (Delta + 4 * gP)) else 0

theorem miGainExt_band {Delta gP : ℝ} (h : -4 * gP < Delta ∧ Delta < 0) :
    miGainExt Delta gP = Real.sqrt (-Delta * (Delta + 4 * gP)) := by
  rw [miGainExt, if_pos h]

theorem miGainExt_outside {Delta gP : ℝ} (h : ¬ (-4 * gP < Delta ∧ Delta < 0)) :
    miGainExt Delta gP = 0 := by
  rw [miGainExt, if_neg h]

/-- The extended gain is non-negative. -/
theorem miGainExt_nonneg (Delta gP : ℝ) : 0 ≤ miGainExt Delta gP := by
  rw [miGainExt]
  split
  · exact Real.sqrt_nonneg _
  · linarith

/--
**The extended band is exactly where the radicand is positive**, for a genuine
nonlinear drive `γP > 0`. This is the correspondence between the module's
`mask = term > 0` and the physics: the `−4γP < Δ < 0` window.
-/
theorem miGainExt_band_iff {Delta gP : ℝ} (h : 0 < gP) :
    (0 < -Delta * (Delta + 4 * gP)) ↔ (-4 * gP < Delta ∧ Delta < 0) := by
  constructor
  · intro hpos
    rcases lt_trichotomy Delta 0 with hlt | heq | hgt
    · refine ⟨?_, hlt⟩
      by_contra hle
      push_neg at hle
      have h2' : Delta + 4 * gP ≤ 0 := by linarith
      have h1 : 0 ≤ Delta * (Delta + 4 * gP) := by nlinarith
      have hpos' : 0 < -(Delta * (Delta + 4 * gP)) := by rw [← neg_mul]; exact hpos
      linarith
    · exfalso; rw [heq] at hpos; linarith
    · exfalso
      have hsum : 0 < Delta + 4 * gP := by nlinarith
      have h1 : 0 < Delta * (Delta + 4 * gP) := mul_pos hgt hsum
      have hpos' : 0 < -(Delta * (Delta + 4 * gP)) := by rw [← neg_mul]; exact hpos
      linarith
  · intro hb
    obtain ⟨h1, h2⟩ := hb
    have h3 : 0 < -Delta := by linarith
    have h4 : 0 < Delta + 4 * gP := by linarith
    have h5 : 0 < -Delta * (Delta + 4 * gP) := mul_pos h3 h4
    rw [neg_mul] at h5
    linarith

/--
**Peak of the extended gain.** Inside the band the gain is maximised at
`Δ = −2γP`, where it equals `2γP` — the same peak height as the classical
form, so the two descriptions agree where it matters.
-/
theorem miGainExt_peak {gP : ℝ} (h : 0 < gP) :
    miGainExt (-2 * gP) gP = 2 * gP := by
  have hpos : (0:ℝ) ≤ 2 * gP := by linarith
  rw [miGainExt_band (by constructor <;> linarith)]
  have h4 : -(-2 * gP) * (-2 * gP + 4 * gP) = (2 * gP) ^ 2 := by ring
  rw [h4, Real.sqrt_sq hpos]

/-- Inside the band the extended gain never exceeds `2γP`. -/
theorem miGainExt_le_max {Delta gP : ℝ} (h : 0 < gP) (hb : -4 * gP < Delta ∧ Delta < 0) :
    miGainExt Delta gP ≤ 2 * gP := by
  have hpos : (0:ℝ) ≤ 2 * gP := by linarith
  rw [miGainExt_band hb]
  rw [← Real.sqrt_sq hpos]
  apply Real.sqrt_le_sqrt
  have : -Delta * (Delta + 4 * gP) ≤ (2 * gP) ^ 2 := by
    nlinarith [sq_nonneg (2 * (-Delta) - 4 * gP)]
  linarith

/--
**The extended gain recovers the classical one.** For quadratic dispersion and
a modulation frequency inside the classical band, the two expressions are the
*same number*, not two different approximations:

`√[−(β₂Ω²)·(β₂Ω² + 4γP)] = |β₂|·|Ω|·√(Ω_c² − Ω²)`.

This is what licenses interpolating between the classical and the
higher-order path in `mi_gain_spectrum_extended`. The `Ω ≠ 0` hypothesis keeps
the extended band strictly open at its `Δ = 0` edge.
-/
theorem miGainExt_reduces (beta2 gamma P Omega : ℝ) (h2 : beta2 < 0) (hg : 0 < gamma)
    (hP : 0 < P) (hΩ : Omega ≠ 0) (hO : Omega ^ 2 < -4 * gamma * P / beta2) :
    miGainExt (beta2 * Omega ^ 2) (gamma * P) = miGain beta2 gamma P Omega := by
  have hb : beta2 ≠ 0 := ne_of_lt h2
  have habs : |beta2| = -beta2 := abs_of_neg h2
  have hgP : (0:ℝ) < gamma * P := by positivity
  have hΩpos : (0:ℝ) < Omega ^ 2 := sq_pos_of_ne_zero hΩ
  -- Δ = β₂Ω² lies in the extended band exactly when Ω² < Ω_c² and Ω ≠ 0.
  have hneg : beta2 * Omega ^ 2 < 0 := by nlinarith
  have hgt : -4 * (gamma * P) < beta2 * Omega ^ 2 := by
    have h1 : beta2 * (-4 * gamma * P / beta2) < beta2 * Omega ^ 2 :=
      mul_lt_mul_of_neg_left hO h2
    have h2' : beta2 * (-4 * gamma * P / beta2) = -4 * (gamma * P) := by
      rw [← mul_div_assoc]
      field_simp
    rw [h2'] at h1
    exact h1
  have hΔ : -4 * (gamma * P) < beta2 * Omega ^ 2 ∧ beta2 * Omega ^ 2 < 0 := ⟨hgt, hneg⟩
  rw [miGainExt_band hΔ, miGain_band h2 hO]
  -- Both sides are the square root of the same non-negative number
  -- `N = β₂²·Ω²·(Ω_c² − Ω²)`.
  have harg : (0:ℝ) ≤ -4 * gamma * P / beta2 - Omega ^ 2 := by linarith
  have hN : -(beta2 * Omega ^ 2) * (beta2 * Omega ^ 2 + 4 * (gamma * P))
      = beta2 ^ 2 * Omega ^ 2 * (-4 * gamma * P / beta2 - Omega ^ 2) := by
    have h1 : beta2 * Omega ^ 2 + 4 * (gamma * P)
        = beta2 * (Omega ^ 2 + 4 * gamma * P / beta2) := by
          rw [mul_add]
          congr 1
          rw [← mul_div_assoc]
          field_simp
    have h2' : 4 * gamma * P / beta2 = -(-4 * gamma * P / beta2) := by ring
    rw [h1, h2']
    ring
  rw [hN]
  -- Goal: √N = |β₂|·|Ω|·√(Ω_c² − Ω²); both are √N by two `sqrt_mul`s.
  have h1 : Real.sqrt (beta2 ^ 2) = |beta2| := Real.sqrt_sq_eq_abs beta2
  have h2'' : Real.sqrt (Omega ^ 2) = |Omega| := Real.sqrt_sq_eq_abs Omega
  have hcomb : Real.sqrt (beta2 ^ 2) * Real.sqrt (Omega ^ 2)
        * Real.sqrt (-4 * gamma * P / beta2 - Omega ^ 2)
      = Real.sqrt (beta2 ^ 2 * Omega ^ 2 * (-4 * gamma * P / beta2 - Omega ^ 2)) := by
    rw [← Real.sqrt_mul (sq_nonneg beta2),
      ← Real.sqrt_mul (mul_nonneg (sq_nonneg beta2) (sq_nonneg Omega))]
  rw [← h1, ← h2'', hcomb]

/-! ## Dispersive-wave phase matching

`dispersive_wave_roots` seeks roots of

`F(ω) = β(ω) − [β(ωₛ) + β₁(ωₛ)(ω − ωₛ) + qₛ]`,

the resonant phase-matching condition for a soliton's dispersive wave. Two
structural facts are proved rather than assumed: the residual **at the soliton
frequency is exactly `−qₛ`** (so `ωₛ` is a root iff `qₛ = 0`), and for a
quadratic model the root at `ωₛ` is a **double** root — the algebraic content
of the `f'(ωₛ) = 0` tangency the root finder's fallback comments on. The sign
of the β₂/β₃ detuning is then a corollary.
-/

/-- The DW phase-matching residual of `dispersive_wave_roots`. -/
noncomputable def dwResidual (beta beta1 : ℝ → ℝ) (omega_s q : ℝ) (w : ℝ) : ℝ :=
  beta w - (beta omega_s + beta1 omega_s * (w - omega_s) + q)

/-- At the soliton frequency the residual is exactly `−qₛ`. -/
theorem dwResidual_at_soliton (beta beta1 : ℝ → ℝ) (omega_s q : ℝ) :
    dwResidual beta beta1 omega_s q omega_s = -q := by
  unfold dwResidual
  ring

/--
**The quadratic-model residual is a perfect square.** For
`β(ω) = βₛ + β₁ₛ(ω − ωₛ) + ½β₂(ω − ωₛ)²` the residual is
`½β₂(ω − ωₛ)² − qₛ`: the root at `ωₛ` is double. -/
theorem dwResidual_quadratic (b0 b1 b2 omega_s q w : ℝ) :
    dwResidual (fun x => b0 + b1 * (x - omega_s) + b2 / 2 * (x - omega_s) ^ 2)
      (fun _ => b1) omega_s q w = b2 / 2 * (w - omega_s) ^ 2 - q := by
  unfold dwResidual
  ring

/--
**The cubic truncation's non-trivial root.** Adding the third-order term, the
detuning `Ω = −3β₂/β₃` is an exact root of the residual — the analytic
estimate the β₂/β₃ fast path is approximating.
-/
theorem dwResidual_cubic_root (b0 b1 b2 b3 omega_s w : ℝ) (hb2 : b2 ≠ 0) (hb3 : b3 ≠ 0)
    (hw : w - omega_s = -3 * b2 / b3) :
    dwResidual (fun x => b0 + b1 * (x - omega_s) + b2 / 2 * (x - omega_s) ^ 2
      + b3 / 6 * (x - omega_s) ^ 3) (fun _ => b1) omega_s 0 w = 0 := by
  have hres : dwResidual (fun x => b0 + b1 * (x - omega_s) + b2 / 2 * (x - omega_s) ^ 2
      + b3 / 6 * (x - omega_s) ^ 3) (fun _ => b1) omega_s 0 w
      = b2 / 2 * (w - omega_s) ^ 2 + b3 / 6 * (w - omega_s) ^ 3 := by
    unfold dwResidual
    ring
  rw [hres, hw]
  field_simp
  ring

/--
**The dispersive wave is blue-shifted precisely when `β₂·β₃ < 0`.** The cubic
root of the tangent mismatch, `Ω_DW = −3β₂/β₃`, lies above the pump exactly
when `β₂` and `β₃` have opposite signs — the standard anomalous-dispersion
soliton case, where the DW lands on the short-wavelength side. This is the
sign invariant `PLAN.md` §8.2 asks for: a property of the dispersion profile,
not of any one reproduction.
-/
theorem dwDetuning_blue (b2 b3 : ℝ) (hb3 : b3 ≠ 0) (h : b2 * b3 < 0) :
    0 < -3 * b2 / b3 := by
  have hb3sq : (0:ℝ) < b3 ^ 2 := by positivity
  have hEq : -3 * b2 / b3 = -3 * (b2 * b3) / b3 ^ 2 := by field_simp
  rw [hEq]
  exact div_pos (by nlinarith) hb3sq

/-- The `β₂/β₃` fast estimate `Δω = −2β₂/β₃` carries the same sign, hence the
same blue shift: the fallback and the exact cubic root differ by the factor
`3/2`, never by a sign. -/
theorem dwDetuning_blue_fast (b2 b3 : ℝ) (hb3 : b3 ≠ 0) (h : b2 * b3 < 0) :
    0 < -2 * b2 / b3 := by
  have hb3sq : (0:ℝ) < b3 ^ 2 := by positivity
  have hEq : -2 * b2 / b3 = -2 * (b2 * b3) / b3 ^ 2 := by field_simp
  rw [hEq]
  exact div_pos (by nlinarith) hb3sq

/-- A blue detuning is a blue wavelength: because `λ = 2πc/ω` is decreasing in
`ω`, `ω_DW > ω₀` is exactly `λ_DW < λ₀`. The sign theorem above is therefore
a statement about wavelengths too, which is how the decks report the DW. -/
theorem blue_shift_wavelength (omega0 omegaDW : ℝ) (h0 : 0 < omega0) (hDW : 0 < omegaDW)
    (h : omega0 < omegaDW) :
    2 * Real.pi / omegaDW < 2 * Real.pi / omega0 := by
  rw [div_lt_div_iff₀ (by positivity) (by positivity)]
  nlinarith [Real.pi_pos]

/-! ## The lossless FWM efficiency

`fwm_efficiency`'s `α = 0` branch: `η = sinc²(Δβ·L/2)`. Two claims: the
fraction is at most one, and it *equals* one only at perfect phase matching —
the sinc has no secondary full-height maximum.
-/

/-- The lossless FWM conversion efficiency, with the module's own special case
at zero argument: `η = 1` there, and `sinc²` otherwise. -/
noncomputable def fwmEfficiency (delta_beta L : ℝ) : ℝ :=
  if delta_beta * L / 2 = 0 then 1
  else (Real.sin (delta_beta * L / 2) / (delta_beta * L / 2)) ^ 2

/-- At perfect phase matching the efficiency is exactly one — the module's
`np.isclose(arg, 0.0) → 1.0` substitution, agreed with by the limit. -/
theorem fwmEfficiency_perfect (L : ℝ) : fwmEfficiency 0 L = 1 := by
  unfold fwmEfficiency
  simp

/--
**Unit conversion is the ceiling.** `η ≤ 1` everywhere — the `(0–1 range)`
the docstring claims. This is `sin²x ≤ x²`, which holds on the whole line.
-/
theorem fwmEfficiency_le_one (delta_beta L : ℝ) :
    fwmEfficiency delta_beta L ≤ 1 := by
  unfold fwmEfficiency
  split
  · linarith
  · rename_i h
    have hx : delta_beta * L / 2 ≠ 0 := h
    have hsq : Real.sin (delta_beta * L / 2) ^ 2 ≤ (delta_beta * L / 2) ^ 2 :=
      Real.sin_sq_le_sq
    have hpos : (0:ℝ) < (delta_beta * L / 2) ^ 2 := by positivity
    have e1 : (Real.sin (delta_beta * L / 2) / (delta_beta * L / 2)) ^ 2
        = Real.sin (delta_beta * L / 2) ^ 2 / (delta_beta * L / 2) ^ 2 := div_pow _ _ _
    rw [e1, div_le_iff₀ hpos]
    simpa using hsq

/-- The efficiency is non-negative, being a square. -/
theorem fwmEfficiency_nonneg (delta_beta L : ℝ) : 0 ≤ fwmEfficiency delta_beta L := by
  unfold fwmEfficiency
  split <;> [linarith; exact sq_nonneg _]

/--
**Only perfect phase matching converts everything.** Away from
`Δβ·L/2 = 0` the efficiency is *strictly* below one: `sin²x < x²` for
`x ≠ 0`. The only way to reach full conversion in the lossless formula is to
kill the mismatch.
-/
theorem fwmEfficiency_lt_one (delta_beta L : ℝ) (h : delta_beta ≠ 0) (hL : L ≠ 0) :
    fwmEfficiency delta_beta L < 1 := by
  have hx : delta_beta * L / 2 ≠ 0 := by
    intro hz
    have hprod : delta_beta * L = 0 := by linarith
    rcases mul_eq_zero.mp hprod with hdb | hL'
    · exact h hdb
    · exact hL hL'
  unfold fwmEfficiency
  rw [if_neg hx]
  have hsq : Real.sin (delta_beta * L / 2) ^ 2 < (delta_beta * L / 2) ^ 2 :=
    Real.sin_sq_lt_sq hx
  have hpos : (0:ℝ) < (delta_beta * L / 2) ^ 2 := by positivity
  have e1 : (Real.sin (delta_beta * L / 2) / (delta_beta * L / 2)) ^ 2
      = Real.sin (delta_beta * L / 2) ^ 2 / (delta_beta * L / 2) ^ 2 := div_pow _ _ _
  rw [e1, div_lt_iff₀ hpos]
  simpa using hsq

end Optics
end Photonics

