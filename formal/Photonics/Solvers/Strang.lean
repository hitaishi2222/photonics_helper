import Mathlib

/-!
# Strang splitting and the Kerr sub-step

The nonlinear sub-step of `gnlse.py`, and the property that makes it safe.

`gnlse.py:1060-1068` applies a symmetric Strang split: half the exact Kerr/Raman
phase, the shock correction, then the trailing half phase. The Kerr sub-step
itself is

`A ↦ A·exp(i·θ)`,  `θ = ½·γ·P·dz`

**The theorem that matters is `kerrStep_norm`: this step preserves the field
magnitude exactly.** A pure phase rotation cannot create or destroy photons,
which is the reason the photon-number drift the module tracks has to be
attributed to the dispersion and shock terms rather than to Kerr.

`strang_halves_compose` says the two half-phase steps are exactly one full phase
step, so the splitting introduces no error *of its own* in the phase; what it
introduces is the operator-commutator error, recorded in `PLAN.md` §9.1.
-/

namespace Photonics
namespace Solvers

open Complex
open scoped ComplexConjugate

/-- The Kerr phase increment over a half step: `θ = ½·γ·P·dz`, the scalar behind
`half_phase = 0.5j * gamma * P_NL * dz`. -/
noncomputable def kerrPhase (gamma P dz : ℝ) : ℝ := gamma * P * dz / 2

theorem kerrPhase_eq (gamma P dz : ℝ) :
    kerrPhase gamma P dz = gamma * P * dz / 2 := rfl

/-- One Kerr sub-step: `A ↦ A·exp(i·θ)`. -/
noncomputable def kerrStep (A : ℂ) (gamma P dz : ℝ) : ℂ :=
  A * exp (I * kerrPhase gamma P dz)

theorem kerrStep_eq (A : ℂ) (gamma P dz : ℝ) :
    kerrStep A gamma P dz = A * exp (I * kerrPhase gamma P dz) := rfl

/-- `exp(iθ)` is a unit-modulus rotation, for any real phase. -/
theorem exp_I_norm (r : ℝ) : ‖exp (I * r)‖ = 1 := by
  rw [Complex.norm_exp]
  norm_num [Complex.mul_re, Complex.I_re, Complex.I_im]

/--
**The Kerr sub-step conserves the field magnitude exactly.**

`|exp(iθ)| = 1` for every real `θ`, so `|A·exp(iθ)| = |A|`. A Kerr phase rotation
cannot change the photon number.

This is the formal counterpart of `conserving_shock` and of the photon-number
tracking in `gnlse.py`: whatever drift the module reports, it is not coming from
the phase term.
-/
theorem kerrStep_norm (A : ℂ) (gamma P dz : ℝ) :
    ‖kerrStep A gamma P dz‖ = ‖A‖ := by
  rw [kerrStep_eq, Complex.norm_mul, exp_I_norm]
  ring

/--
**The two half-phase steps compose to exactly one full-phase step.**

`exp(iθ/2)·exp(iθ/2) = exp(iθ)`. So the Strang half-half structure costs nothing
in accuracy *of the phase itself*: applied to a purely phase-driven equation the
symmetric split is exact, and all of its error is the commutator between the
phase operator and the dispersion operator, not the halving.
-/
theorem strang_halves_compose (A : ℂ) (theta : ℝ) :
    A * exp (I * (theta / 2)) * exp (I * (theta / 2)) = A * exp (I * theta) := by
  rw [mul_assoc, ← Complex.exp_add]
  congr 1
  field_simp
  ring_nf

/-- Two phase rotations compose additively. -/
theorem two_phases_add (A : ℂ) (θ₁ θ₂ : ℝ) :
    (A * exp (I * θ₁)) * exp (I * θ₂) = A * exp (I * (θ₁ + θ₂)) := by
  rw [mul_assoc, ← Complex.exp_add]
  congr 1
  push_cast
  ring

/--
**The Kerr step is the identity when there is no nonlinearity.**

`γ·P·dz = 0` (no Kerr coefficient, no power, or a zero step) leaves the field
untouched. This is the base case of "the split is exact for a linear problem".
-/
theorem kerrStep_zero_phase (A : ℂ) (gamma P dz : ℝ) (h : kerrPhase gamma P dz = 0) :
    kerrStep A gamma P dz = A := by
  rw [kerrStep_eq, h]
  simp

/--
**The split is exact for the pure phase, so the Kerr error budget is zero on
its own.**

Composing the two halves reproduces the single full step, and a full step with
zero phase is the identity. Together these say the Strang split introduces no
error when only the Kerr operator is active.
-/
theorem strang_exact_on_phase (A : ℂ) (h : kerrPhase 0 0 1 = 0) :
    kerrStep (kerrStep A 0 0 1) 0 0 1 = A := by
  rw [kerrStep_zero_phase A 0 0 1 h, kerrStep_zero_phase A 0 0 1 h]

/-! ## Operator ordering: the commutator the split carries

`PLAN.md` §9.1 asks for the ordering half of the operator story, which is the
formal counterpart of ISSUES.md #0: `Raman ∘ D ≠ D ∘ Raman`, and the sign of
the discrepancy. On the scalar Kerr/dispersion phases of this file the
operators *do* commute (`two_phases_add` is commutative in its conclusions),
so the whole commutator budget sits in the nonlocal operator.

The two-bin grid is the smallest instance that captures it. A convolution `R`
(the shape of the delayed Raman response, which mixes frequency bins) and a
diagonal `d` (the dispersion operator, unequal across bins) fail to commute
exactly when the convolution is off-diagonal — and the witness is the field
concentrated in one bin.
-/

/-- A two-bin convolution operator: `(R·A) i = ∑_j R i j · A j`. This is the
shape of any delayed-response operator, Raman included: nonlocal in the
frequency index. -/
noncomputable def convOp (R : Matrix (Fin 2) (Fin 2) ℂ) (A : Fin 2 → ℂ) : Fin 2 → ℂ :=
  fun i => R i 0 * A 0 + R i 1 * A 1

/-- A two-bin diagonal operator — the dispersion operator restricted to two
bins. -/
noncomputable def diagOp (d : Fin 2 → ℂ) (A : Fin 2 → ℂ) : Fin 2 → ℂ :=
  fun i => d i * A i

/--
**Composition of the split-step maps is associative.** The alternated sequence
`D ∘ K ∘ D ∘ K ∘ …` is therefore unambiguous without parentheses — the
composition theorem of `PLAN.md` §9.1 in its trivial half.
-/
theorem split_comp_assoc (D K : ℂ → ℂ) :
    (D ∘ K) ∘ D = D ∘ (K ∘ D) := Function.comp_assoc D K D

/--
**A convolution does not commute with an unequal diagonal.**

If `R 1 0 ≠ 0` — the convolution genuinely feeds bin 0 into bin 1, which is the
shape of a delayed kernel — and `d 0 ≠ d 1` — the dispersion differs across the
bins, which is the whole point of a dispersive propagator — then the two
orders differ. The witness is the field `e₀` concentrated in bin 0:

- `(R ∘ D) e₀` at bin 1 is `R 1 0 · d 0`: the dispersed bin-0 field, then
  convolved;
- `(D ∘ R) e₀` at bin 1 is `d 1 · R 1 0`: the convolved field, then dispersed.

They agree only when `R 1 0 · (d 0 − d 1) = 0`, which the hypotheses rule out.

This is the formal statement of why the convention swap of ISSUES.md #0 had to
change the Raman operator and the propagator *together*: they do not commute,
so changing one without the other is not a change of gauge. It is also why the
Raman term — not the dispersion, not the Kerr phase — is the piece of
`gnlse.py` that can move what `splitSeq_norm` proved is otherwise fixed.
-/
theorem conv_diag_ne_commute (R : Matrix (Fin 2) (Fin 2) ℂ) (d : Fin 2 → ℂ)
    (hR : R 1 0 ≠ 0) (hd : d 0 ≠ d 1) :
    convOp R ∘ diagOp d ≠ diagOp d ∘ convOp R := by
  intro h
  -- The witness: the field concentrated in bin 0, evaluated at bin 1.
  have hcomposed : R 1 0 * (d 0 * 1) + R 1 1 * (d 1 * 0)
      = d 1 * (R 1 0 * 1 + R 1 1 * 0) := by
    have h' := congrFun (congrFun h (fun i => if i = 0 then (1:ℂ) else 0)) 1
    simp only [Function.comp, convOp, diagOp] at h'
    exact h'
  have hlt : R 1 0 * d 0 = d 1 * R 1 0 := by simpa using hcomposed
  have hz : R 1 0 * (d 0 - d 1) = 0 := by
    have hcomm : R 1 0 * d 0 = R 1 0 * d 1 := by rw [hlt, mul_comm]
    rw [mul_sub, hcomm, sub_self]
  rcases mul_eq_zero.mp hz with h | h
  · exact hR h
  · exact hd (sub_eq_zero.mp h)

end Solvers
end Photonics