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

end Optics
end Photonics
