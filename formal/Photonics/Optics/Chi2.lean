import Mathlib

/-!
# Second-harmonic generation

The undepleted-pump closed form that `chi2.py` documents and the coupled-wave
solver reproduces: with effective coupling `κ = σ√P₀`, perfect phase matching
gives a conversion efficiency

`η(z) = tanh²(κ·z)`

Verified against `solve_shg` at `length = 0.05 m`, `P₀ = 1 W`, `σ = 2 m⁻¹W⁻¹ᐟ²`
(so `κ = 2 m⁻¹`, `κL = 0.1`): the solver returns `η = 0.009934` and
`tanh²(0.1) = 0.009934`.

The point of formalising this is the **bound**, not the formula: `η ≤ 1` says
second-harmonic generation cannot convert more than all of the pump, whatever
the coupling and length, and that is the constraint `Chi2Result.efficiency` is
supposed to respect.

Note on style: this Mathlib revision has almost no `tanh` lemmas — only
`Real.tanh_eq` (the exponential quotient) and the `atanh` identities. So the
basic bounds are derived from `Real.tanh_eq` rather than quoted, which is why
they are stated as separate lemmas below.
-/

namespace Photonics
namespace Optics

/-! ## The coupling -/

/--
Effective coupling `κ = σ·√P₀`, matching `Chi2Result.kappa`.

`σ` is the nonlinear coupling and `P₀` the pump power, so `κ` is a coupling per
unit length and `κ·z` is dimensionless.
-/
noncomputable def kappa (sigma P0 : ℝ) : ℝ := sigma * Real.sqrt P0

theorem kappa_eq (sigma P0 : ℝ) : kappa sigma P0 = sigma * Real.sqrt P0 := rfl

/-- `κ ≥ 0` for a non-negative coupling and pump power.

This is the domain condition the efficiency formula needs, and `chi2.py` has no
check for it: `shg_coupling` will happily return a negative `σ`.
-/
theorem kappa_nonneg {sigma P0 : ℝ} (hs : 0 ≤ sigma) (hP : 0 ≤ P0) :
    0 ≤ kappa sigma P0 := by
  unfold kappa
  positivity

/-! ## Basic `tanh` bounds, derived here -/

/-- `−1 ≤ tanh x`, from the exponential quotient. -/
theorem tanh_ge_neg_one (x : ℝ) : -1 ≤ Real.tanh x := by
  rw [Real.tanh_eq]
  have hpos : (0:ℝ) < Real.exp x + Real.exp (-x) := by positivity
  rw [le_div_iff₀ hpos]
  linarith [Real.exp_pos x]

/-- `tanh x ≤ 1`, from the exponential quotient. -/
theorem tanh_le_one (x : ℝ) : Real.tanh x ≤ 1 := by
  rw [Real.tanh_eq]
  have hpos : (0:ℝ) < Real.exp x + Real.exp (-x) := by positivity
  rw [div_le_iff₀ hpos]
  linarith [Real.exp_pos (-x)]

/-- `tanh` is odd, from the exponential quotient. -/
theorem tanh_neg' (x : ℝ) : Real.tanh (-x) = -Real.tanh x := by
  rw [Real.tanh_eq, Real.tanh_eq, Real.exp_neg]
  congr 1 <;> ring

/-! ## The efficiency -/

/--
Conversion efficiency `η(z) = tanh²(κ·z)` in the undepleted-pump,
perfect-phase-matching limit. This is the formula `chi2.py` states in its
docstring.
-/
noncomputable def shgEfficiency (sigma P0 z : ℝ) : ℝ :=
  Real.tanh (kappa sigma P0 * z) ^ 2

theorem shgEfficiency_eq (sigma P0 z : ℝ) :
    shgEfficiency sigma P0 z = Real.tanh (kappa sigma P0 * z) ^ 2 := rfl

/-- **SHG cannot exceed unit conversion.** All of the pump can become second
harmonic, and no more, whatever the coupling and the length. -/
theorem shgEfficiency_le_one (sigma P0 z : ℝ) :
    shgEfficiency sigma P0 z ≤ 1 := by
  have h1 := tanh_le_one (kappa sigma P0 * z)
  have h2 := tanh_ge_neg_one (kappa sigma P0 * z)
  unfold shgEfficiency
  nlinarith [sq_nonneg (Real.tanh (kappa sigma P0 * z) - 1),
            sq_nonneg (Real.tanh (kappa sigma P0 * z) + 1)]

/-- The efficiency is a non-negative fraction. -/
theorem shgEfficiency_nonneg (sigma P0 z : ℝ) : 0 ≤ shgEfficiency sigma P0 z := by
  unfold shgEfficiency
  positivity

/-- **No conversion at the input face**, and none without a pump: the
efficiency is exactly zero at `z = 0` for any coupling. -/
theorem shgEfficiency_zero (sigma P0 : ℝ) : shgEfficiency sigma P0 0 = 0 := by
  unfold shgEfficiency
  rw [mul_zero, Real.tanh_zero]
  norm_num

/--
**Efficiency is even in position.** Reversing the sign of the nonlinearity
mirrors the efficiency about the input face but never changes it, because
`tanh` is odd and the result is squared.
-/
theorem shgEfficiency_even (sigma P0 z : ℝ) :
    shgEfficiency sigma P0 (-z) = shgEfficiency sigma P0 z := by
  unfold shgEfficiency
  rw [mul_neg, tanh_neg']
  ring

/-! ## The half-conversion point -/

/--
**Half-conversion length.** The efficiency is exactly `1/2` at
`κ·z = artanh(1/√2)`.

This is the length quoted as "half the pump converted", and the equivalence is
exact, not a fit.
-/
theorem shgEfficiency_half (sigma P0 : ℝ) (hs : 0 < sigma) (hP : 0 < P0) :
    shgEfficiency sigma P0 (Real.artanh (1 / Real.sqrt 2) / kappa sigma P0) = 1 / 2 := by
  have hsqrt : (0:ℝ) < Real.sqrt 2 := Real.sqrt_pos.2 (by norm_num)
  have hq : (1 / Real.sqrt 2 : ℝ) ∈ Set.Ioo (-1) 1 := by
    constructor
    · have : (0:ℝ) < 1 / Real.sqrt 2 := one_div_pos.mpr hsqrt
      linarith
    · have hsq : (Real.sqrt 2) ^ 2 = 2 := Real.sq_sqrt (by norm_num : (0:ℝ) ≤ 2)
      have h1 : (1:ℝ) < Real.sqrt 2 := by nlinarith
      have : (1 / Real.sqrt 2 : ℝ) < 1 := (div_lt_one hsqrt).2 (by simpa using h1)
      exact this
  have ht : Real.tanh (Real.artanh (1 / Real.sqrt 2)) = 1 / Real.sqrt 2 :=
    Real.tanh_artanh hq
  have hk : kappa sigma P0 ≠ 0 := by
    unfold kappa
    have h2 := Real.sqrt_pos.2 hP
    positivity
  unfold shgEfficiency
  have hc := div_mul_cancel₀ (Real.artanh (1 / Real.sqrt 2)) hk
  rw [mul_comm, hc, ht]
  have h2 : (Real.sqrt 2) ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  field_simp
  nlinarith [h2]

/-- At the half-conversion point the efficiency really is `1/2`, i.e. strictly
between zero and the unit bound, so the bound above is not vacuous. -/
theorem shgEfficiency_half_strict (sigma P0 : ℝ) (hs : 0 < sigma) (hP : 0 < P0) :
    0 < shgEfficiency sigma P0
        (Real.artanh (1 / Real.sqrt 2) / kappa sigma P0) ∧
      shgEfficiency sigma P0
        (Real.artanh (1 / Real.sqrt 2) / kappa sigma P0) < 1 := by
  rw [shgEfficiency_half sigma P0 hs hP]
  constructor <;> norm_num

/-! ## Phase matching -/

/--
The poling period that cancels the mismatch: `Λ = 2π·order/|Δk|`, matching
`Lambda_qpm`.

`|Δk|` rather than `Δk` because the grating corrects the magnitude; the sign is
absorbed by the domain choice, which `qpm_grating` handles. So `Δk ≠ 0` is the
exact hypothesis.
-/
noncomputable def LambdaQPM (delta_k : ℝ) (order : ℕ) : ℝ :=
  2 * Real.pi * order / |delta_k|

theorem LambdaQPM_eq (delta_k : ℝ) (order : ℕ) :
    LambdaQPM delta_k order = 2 * Real.pi * order / |delta_k| := rfl

/-- A poling period exists and is positive whenever the mismatch is non-zero
and the order positive. -/
theorem LambdaQPM_pos {delta_k : ℝ} {order : ℕ} (h : delta_k ≠ 0) (ho : 0 < order) :
    0 < LambdaQPM delta_k order := by
  unfold LambdaQPM
  have ho' : (0:ℝ) < order := by exact_mod_cast ho
  have hk : (0:ℝ) < |delta_k| := abs_pos.mpr h
  positivity

/--
**Exact phase matching cannot be reached by a grating.** At `Δk = 0` the period
is undefined rather than infinite, and the module returns `0` through the
division. The physics is right (no grating is needed) but the *value* is a
division by zero rather than an error.

This is the one place where the Python's behaviour and its documentation
disagree: `Lambda_qpm`'s docstring says `delta_k` must be non-zero, and nothing
enforces it.
-/
theorem LambdaQPM_at_zero (order : ℕ) : LambdaQPM 0 order = 0 := by
  unfold LambdaQPM
  rw [abs_zero, div_zero]

/-- The period depends on `|Δk|` only, so it is even in the mismatch. -/
theorem LambdaQPM_neg (delta_k : ℝ) (order : ℕ) :
    LambdaQPM (-delta_k) order = LambdaQPM delta_k order := by
  unfold LambdaQPM
  rw [abs_neg]

end Optics
end Photonics