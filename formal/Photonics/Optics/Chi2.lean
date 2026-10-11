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

Also proved here: the SHG **phase mismatch** `Δk = β(2ω) − 2β(ω)` and the exact
first-order QPM cancellation that turns it into a poling period; and the
**uniform-field reduction** of the modal-overlap coupling — the sentence in
`shg_coupling_overlap`'s docstring that a uniform field recovers the Boyd
plane-wave coupling exactly.
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

/-! ## The phase mismatch

`delta_k_shg` in `chi2.py`: `Δk = β(2ω) − 2β(ω)`. The value is trivial; what
matters is the two things attached to it — that phase matching is exactly its
zero, and that the first-order poling period cancels it exactly.
-/

/-- The SHG phase mismatch, exactly `delta_k_shg`. -/
noncomputable def deltaK (beta : ℝ → ℝ) (omega : ℝ) : ℝ :=
  beta (2 * omega) - 2 * beta omega

theorem deltaK_eq (beta : ℝ → ℝ) (omega : ℝ) :
    deltaK beta omega = beta (2 * omega) - 2 * beta omega := rfl

/--
**Phase matching is the zero of the mismatch.** `Δk = 0` exactly when
`β(2ω) = 2β(ω)`: the fundamental and its second harmonic carry the same
propagation constant per unit of frequency doubling. This is the manifold the
solver's `delta_k == 0.0` branches act on, and the condition `Lambda_qpm`'s
docstring makes a hypothesis.
-/
theorem deltaK_zero_iff (beta : ℝ → ℝ) (omega : ℝ) :
    deltaK beta omega = 0 ↔ beta (2 * omega) = 2 * beta omega := by
  unfold deltaK
  constructor <;> intro h <;> linarith

/--
**First-order QPM cancels the mismatch exactly.** With the order-1 poling
period `Λ = 2π/|Δk|`, the compensated mismatch is zero:

`Δk − 2π/Λ = 0`  for `Δk > 0`.

So the grating is an exact cancellation, not an approximation: the period
absorbs the magnitude and the domain choice absorbs the sign, which is why
`LambdaQPM` is stated through `|Δk|`.
-/
theorem deltaK_qpm_cancels (beta : ℝ → ℝ) (omega : ℝ) (h : 0 < deltaK beta omega) :
    deltaK beta omega - 2 * Real.pi / LambdaQPM (deltaK beta omega) 1 = 0 := by
  have habs : |deltaK beta omega| = deltaK beta omega := abs_of_pos h
  unfold LambdaQPM
  rw [habs, Nat.cast_one, mul_one, div_div_eq_mul_div]
  field_simp [Real.pi_ne_zero]
  simp

/-! ## The modal-overlap coupling

`shg_coupling_overlap`'s docstring makes one structural claim: the Wang et al.
(2017) overlap integral, in its scale-invariant form, reduces *exactly* for a
uniform field to the Boyd plane-wave coupling of `shg_coupling`. The reduction
is pure arithmetic, so it can be proved rather than asserted.
-/

/--
The modal-overlap coupling in the scale-invariant form of
`shg_coupling_overlap`'s docstring:

`g = (ω·d₀/c)·√(2Z₀/(n_p²·n_s))·O / (I_p·√I_s)`

with `O` the overlap integral, `I_p`, `I_s` the two intensities.
-/
noncomputable def overlapCoupling (omega d0 c Z0 np ns O Ip Is : ℝ) : ℝ :=
  (omega * d0 / c) * Real.sqrt (2 * Z0 / (np ^ 2 * ns)) * O / (Ip * Real.sqrt Is)

/--
**A uniform field recovers the plane-wave residue.** Take the pump and
second-harmonic fields uniform with amplitudes `E` and `Es` over a common area
`A`: `I_p = E²A`, `I_s = Es²A`, `O = (d/d₀)·E·Es·A`. The overlap formula
then collapses to

`(ω·d/c)·√(2Z₀/(n_p²·n_s)) / √A`,

with every `E`-, `Es`- and `d₀`-dependence cancelled. That is the docstring's
"all ratios are scale-invariant" sentence, proved: changing the field scaling
or the `d`-normalisation moves `O`, `I_p` and `I_s` together and leaves `g`
fixed.
-/
theorem overlapCoupling_uniform (omega d0 c Z0 np ns d E Es A : ℝ)
    (hd0 : d0 ≠ 0) (hc : c ≠ 0) (hE : E ≠ 0) (hEsp : 0 < Es) (hA : 0 < A) :
    overlapCoupling omega d0 c Z0 np ns (d / d0 * E ^ 2 * Es * A) (E ^ 2 * A) (Es ^ 2 * A)
      = (omega * d / c) * Real.sqrt (2 * Z0 / (np ^ 2 * ns)) / Real.sqrt A := by
  have hsqrt : Real.sqrt (Es ^ 2 * A) = Es * Real.sqrt A := by
    rw [Real.sqrt_mul (sq_nonneg Es), Real.sqrt_sq (le_of_lt hEsp)]
  have hinner : d / d0 * E ^ 2 * Es * A / (E ^ 2 * A * (Es * Real.sqrt A))
      = d / d0 / Real.sqrt A := by
    rw [div_eq_div_iff (by positivity) (by positivity)]
    field_simp
  have hratio : (omega * d0 / c) * Real.sqrt (2 * Z0 / (np ^ 2 * ns))
      * (d / d0 * E ^ 2 * Es * A) / (E ^ 2 * A * (Es * Real.sqrt A))
      = (omega * d0 / c) * Real.sqrt (2 * Z0 / (np ^ 2 * ns)) * (d / d0 / Real.sqrt A) := by
    rw [mul_div_assoc, mul_div_assoc, hinner]
  unfold overlapCoupling
  rw [hsqrt, hratio]
  field_simp [hd0, hc]

/--
**The plane-wave residue is the Boyd coupling.** With equal indices
`n_p = n_s = n`, the residue of the theorem above equals

`σ = (ω·d/(n·c))·√(2Z₀/(n·A))`,

which is exactly `shg_coupling` (`prefactor = ω·d_eff/(n·C)`, times
`√(2·Z0/(n·A_eff))`). The bridge is `√(2Z₀/(n³A)) = √(2Z₀/(nA))/n`.
-/
theorem overlapCoupling_planeWave (omega d0 c Z0 np ns d E Es A : ℝ)
    (hd0 : d0 ≠ 0) (hc : c ≠ 0) (hE : E ≠ 0) (hEsp : 0 < Es) (hA : 0 < A) (hn : 0 < np)
    (hZ0 : 0 < Z0) (hns : ns = np) :
    overlapCoupling omega d0 c Z0 np ns (d / d0 * E ^ 2 * Es * A) (E ^ 2 * A) (Es ^ 2 * A)
      = (omega * d / (np * c)) * Real.sqrt (2 * Z0 / (np * A)) := by
  have h2np : (0:ℝ) < np := hn
  have key : Real.sqrt (2 * Z0 / (np ^ 2 * np)) / Real.sqrt A
      = Real.sqrt (2 * Z0 / (np * A)) / np := by
    have h1 : (Real.sqrt (2 * Z0 / (np ^ 2 * np)) / Real.sqrt A) ^ 2
        = 2 * Z0 / (np ^ 2 * np * A) := by
      rw [div_pow, Real.sq_sqrt (by positivity), Real.sq_sqrt hA.le]
      field_simp
    have h2 : (Real.sqrt (2 * Z0 / (np * A)) / np) ^ 2
        = 2 * Z0 / (np * A * np ^ 2) := by
      rw [div_pow, Real.sq_sqrt (by positivity)]
      field_simp
    have h3 : 2 * Z0 / (np ^ 2 * np * A) = 2 * Z0 / (np * A * np ^ 2) := by ring
    have hcases : Real.sqrt (2 * Z0 / (np ^ 2 * np)) / Real.sqrt A
        = Real.sqrt (2 * Z0 / (np * A)) / np ∨
      Real.sqrt (2 * Z0 / (np ^ 2 * np)) / Real.sqrt A
        = -(Real.sqrt (2 * Z0 / (np * A)) / np) :=
      (mul_self_eq_mul_self_iff.mp (by rw [← pow_two, ← pow_two, h1, h2, h3]))
    rcases hcases with hc' | hc'
    · exact hc'
    · exfalso
      have hX : (0:ℝ) < Real.sqrt (2 * Z0 / (np ^ 2 * np)) / Real.sqrt A := by
        apply div_pos <;> exact Real.sqrt_pos.2 (by positivity)
      have hY : (0:ℝ) < Real.sqrt (2 * Z0 / (np * A)) / np := by
        apply div_pos <;> [exact Real.sqrt_pos.2 (by positivity); linarith]
      linarith
  rw [overlapCoupling_uniform omega d0 c Z0 np ns d E Es A hd0 hc hE hEsp hA, hns,
    mul_div_assoc, key]
  field_simp [hn.ne', hc]

end Optics
end Photonics