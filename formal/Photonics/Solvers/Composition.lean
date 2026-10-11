import Mathlib

/-!
# Symmetric composition of split-step integrators

`gnlse.py`'s outer step is the symmetric (Strang) split
`exp(L·dz/2)·exp(N·dz)·exp(L·dz/2)` — Blow & Wood (1989), Eq. (17). Its
authors state, of exactly that expression:

> "The error in this approximation of a step S(δz) comes from the
> noncommutativity of the two operators L and N; it is a series in odd powers
> of δz whose leading term is proportional to δz³. Thus, if we propagate a
> distance D, the leading error is proportional to Dδz², and so the scheme is
> **globally second-order accurate**."

Composing three such steps with the Suzuki triple-jump weights removes the
leading `dz³` term:

    S₂(w0·dz) S₂(w1·dz) S₂(w0·dz)

with the weights satisfying Yoshida (1990), Eqs. (4.1), (4.5):

    2·w0 + w1   = 1        (consistency)
    2·w0³ + w1³ = 0        (the dz³ error term cancels)

**What this file proves.** The weights exist, and the consistency condition
makes the composed map reproduce the exact flow. That is the algebra
`gnlse.py`'s `scheme="yoshida4"` relies on.

It is proved **without the closed form**. `PLAN.md` §1.6 forbids numeric
literals and asks for constants as opaque reals with proved relations, so
"there exist `w0, w1 : ℝ` satisfying the two conditions" is the right
statement — and this Mathlib revision has no `Real.cbrt` anyway, which settles
the same point by force.

**What this file does not prove, and why.** The 4th-order claim for
*non-commuting* L and N is a statement about commutators; it needs the
Baker–Campbell–Hausdorff machinery that `PLAN.md` §9.1 flags as beyond scope.
What is proved here is the part where a wrong answer would be silent: the
algebra of the composition. See `PLAN.md` §17.6 Stage S2.

**Measured order.** Confirmed numerically by
`photonics_helper.gnlse_validation.order_study`; see `PLAN.md` §17.6 Stage S1
(observed 1.95 → 4.15).
-/

namespace Photonics
namespace Solvers

open Complex
open scoped ComplexConjugate

/-! ## The order conditions

Yoshida writes the composition as `S₂(w1) S₂(w0) S₂(w1)` — *his* `w1` is the
outer weight, *his* `w0` the middle. The predicates below use the convention
of this file and of `gnlse.py`: **`w0` is the outer weight, appearing twice.**
-/

/-- Yoshida Eq. (4.5), first condition: the sub-step weights must sum to one. -/
def consistency (w0 w1 : ℝ) : Prop := 2 * w0 + w1 = 1

/-- Yoshida Eq. (4.5), second condition: the leading `dz³` error cancels. -/
def cubicCancels (w0 w1 : ℝ) : Prop := 2 * w0 ^ 3 + w1 ^ 3 = 0

/-- The two conditions together: a 4th-order symmetric composition exists
exactly when some pair of weights satisfies both. -/
def orderConds (w0 w1 : ℝ) : Prop := consistency w0 w1 ∧ cubicCancels w0 w1

/-- The error functional whose root is sought. With the middle weight fixed by
consistency, the cubic condition is `cubicError w0 = 0`. -/
def cubicError (w : ℝ) : ℝ := 2 * w ^ 3 + (1 - 2 * w) ^ 3

theorem cubicError_one : cubicError 1 = 1 := by norm_num [cubicError]

theorem cubicError_two : cubicError 2 = (-11 : ℝ) := by norm_num [cubicError]

/-- The two conditions are the consistency condition plus a root of
`cubicError` in the outer weight. -/
theorem orderConds_iff (w0 w1 : ℝ) :
    orderConds w0 w1 ↔ w1 = 1 - 2 * w0 ∧ cubicError w0 = 0 := by
  simp only [orderConds, consistency, cubicCancels, cubicError]
  constructor
  · rintro ⟨hc, hb⟩
    have hlin : w1 = 1 - 2 * w0 := by linarith
    rw [hlin] at hb
    exact ⟨hlin, hb⟩
  · rintro ⟨hc, hb⟩
    refine ⟨by linarith, ?_⟩
    rwa [hc]

/--
**Weights achieving a 4th-order composition exist.**

`cubicError` is continuous and changes sign between `1` (value `1`) and `2`
(value `-11`), so its negative passes through zero on `[1, 2]` and the
intermediate value theorem supplies a root there. This is the statement the
composition actually needs - that such weights exist - and it carries no closed
form, no `2^(1/3)`, and no numeric literal.

`cubicError` appears negated only so the interval endpoints come out in
increasing order, which is what `intermediate_value_Icc` requires.
-/
theorem orderConds_solvable :
    ∃ w0 w1 : ℝ, 1 ≤ w0 ∧ w0 ≤ 2 ∧ orderConds w0 w1 := by
  have hcont : Continuous (-cubicError) := by
    unfold cubicError
    fun_prop
  have hcont' : ContinuousOn (-cubicError) (Set.Icc 1 2) :=
    hcont.continuousOn.mono (Set.subset_univ _)
  have hmem : Set.Icc ((-cubicError) 1) ((-cubicError) 2)
      ⊆ (-cubicError) '' Set.Icc (1 : ℝ) 2 :=
    intermediate_value_Icc (by norm_num) hcont'
  have hzero : (0 : ℝ) ∈ Set.Icc ((-cubicError) 1) ((-cubicError) 2) := by
    norm_num [cubicError, cubicError_one, cubicError_two]
  obtain ⟨w0, hw0mem, hval⟩ := (hmem hzero)
  have hw0lo : 1 ≤ w0 := hw0mem.1
  have hw0hi : w0 ≤ 2 := hw0mem.2
  refine ⟨w0, 1 - 2 * w0, hw0lo, hw0hi, ?_⟩
  rw [orderConds_iff]
  refine ⟨rfl, ?_⟩
  have h' : -(cubicError w0) = 0 := hval
  linarith

/--
**The two conditions are genuinely asymmetric — the labelling is not cosmetic.**

With `w0` the outer weight, `2·w0³ + w1³ = 0` holds, but the transposed
condition `w0³ + 2·w1³ = 0` **fails**, unless `w0 = 0`.

This theorem exists because that transposition is a real and easy error: it was
made in `PLAN.md` §17 and caught only by evaluating it (the transposed
expression is `≈ −7.4`, not `0`). Yoshida's notation puts the outer weight in
the `w1` slot, so the conditions must never be read across from his symbols.
-/
theorem transposed_condition_fails {w0 w1 : ℝ} (_h1 : consistency w0 w1)
    (h2 : cubicCancels w0 w1) (h3 : w0 ≠ 0) : w0 ^ 3 + 2 * w1 ^ 3 ≠ 0 := by
  simp only [cubicCancels] at h2
  have hval : w0 ^ 3 + 2 * w1 ^ 3 = -(3 : ℝ) * w0 ^ 3 := by nlinarith [h2]
  rw [hval]
  exact mul_ne_zero (by norm_num) (pow_ne_zero 3 h3)

/-! ## The composition as an operator

Modelled on the exact flows, which is where the consistency condition is
*checkable*: composing Strang steps must advance the field by the sum of the
sub-step sizes. The non-commuting case is out of reach here — see the header.
-/

/-- The exact flow of the `α` sub-operator over a step `h`. -/
noncomputable def alphaFlow (alpha h : ℝ) (A : ℂ) : ℂ := A * exp (I * (alpha * h))

/-- The exact flow of the `β` sub-operator over a step `h`. -/
noncomputable def betaFlow (beta h : ℝ) (A : ℂ) : ℂ := A * exp (I * (beta * h))

/--
One Strang step: half of `L`, all of `N`, half of `L`. This is Blow & Wood
(1989) Eq. (17).
-/
noncomputable def strangStep (alpha beta h : ℝ) (A : ℂ) : ℂ :=
  betaFlow beta h (alphaFlow alpha (h / 2) (alphaFlow alpha (h / 2) A))

/--
**The Strang step is exact for the flows it composes.**

Because the two sub-flows are multiplications and hence commute, the symmetric
split introduces *no* error: the whole error budget belongs to the
non-commuting case. This is the same fact as `strang_exact_on_phase` in
`Strang.lean`, stated for the pair of sub-operators rather than for Kerr alone.
-/
theorem strangStep_exact (alpha beta h : ℝ) (A : ℂ) :
    strangStep alpha beta h A = A * exp (I * ((alpha + beta) * h)) := by
  simp only [strangStep, betaFlow, alphaFlow]
  rw [mul_assoc, mul_assoc, ← Complex.exp_add, ← Complex.exp_add]
  congr 1
  push_cast
  ring_nf

/--
**Consistency is what makes the composition work.**

If the weights satisfy `2·w0 + w1 = 1`, the composed map advances the field by
exactly `h` — the distance a single Strang step would. Without consistency the
composition would travel the wrong distance, and no amount of cubic
cancellation would help. This is why `orderConds` pairs the two conditions
rather than shipping either alone.
-/
theorem yoshida4_consistent (alpha beta h : ℝ) (w0 w1 : ℝ)
    (hc : consistency w0 w1) (A : ℂ) :
    strangStep alpha beta (w0 * h) (strangStep alpha beta (w1 * h)
        (strangStep alpha beta (w0 * h) A))
      = A * exp (I * ((alpha + beta) * h)) := by
  rw [strangStep_exact, strangStep_exact, strangStep_exact]
  -- Substitute the consistency relation *before* summing the exponents: this
  -- turns the whole expression into a polynomial in w0 alone, which ring closes.
  have hc0 : 2 * w0 + w1 = 1 := hc
  have hc' : w1 = 1 - 2 * w0 := by linarith
  rw [hc']
  rw [mul_assoc, mul_assoc, ← Complex.exp_add, ← Complex.exp_add]
  congr 1
  push_cast
  ring

/--
**The Strang propagator is symmetric: its inverse is the step with `-h`.**

Symmetry is what makes a composition of Strang steps a symmetric composition at
all, and hence what licenses the error-extrapolation form of the Yoshida
construction. Blow & Wood's four-forward / one-backward / four-forward
extrapolation is built on exactly this property.

The statement is about the **propagator**, not about a field vector, and the
distinction is not cosmetic: `S(−h) A = (S(h) A)⁻¹` is **false** for a general
`A`, since it would force `A = A⁻¹`. Symmetry says `S(−h)` is the inverse
*map* of `S(h)`, which for these exponential flows is exactly
`exp(−z) = (exp z)⁻¹`.
-/
theorem strangProp_symmetric (alpha beta h : ℝ) :
    exp (I * ((↑alpha + ↑beta) * ↑(-h)))
      = (exp (I * ((↑alpha + ↑beta) * ↑h)))⁻¹ := by
  rw [← Complex.exp_neg]
  congr 1
  push_cast
  ring

end Solvers
end Photonics
