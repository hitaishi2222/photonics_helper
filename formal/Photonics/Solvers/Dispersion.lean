import Mathlib
import Photonics.Solvers.Strang

/-!
# The linear (dispersion) operator and norm conservation of a split step

The dispersion sub-step of `gnlse.py` is a diagonal multiplication in the
frequency domain, `A(Ω) ↦ A(Ω)·exp(i·β(Ω)·z)`. Transcribed here as a phase
rotation, which is all the structure the conservation argument needs.

The theorems discharge the `‖D‖ = 1` obligation that `PLAN.md` §6.2 has
carried since Stage C and never settled.

**`splitStep_norm` is the one that matters.** The dispersion operator and the
Kerr phase operator are both unit-modulus, so their composition is too. A full
split-step *without* the shock term therefore conserves photon number exactly.
Since `gnlse.py` does report photon-number drift, that drift is attributable to
the shock correction, to the delayed Raman term, or to discretisation — and
never to the dispersion or the Kerr phase. That is a useful constraint on any
future investigation, and it is now proved rather than argued.
-/

namespace Photonics
namespace Solvers

open Complex
open scoped ComplexConjugate

/--
The dispersion operator over a propagation distance `z` with phase
`β(Ω)·z`: `A ↦ A·exp(i·β·z)`.
-/
noncomputable def dispersionOp (beta z : ℝ) (A : ℂ) : ℂ := A * exp (I * (beta * z))

theorem dispersionOp_eq (beta z : ℝ) (A : ℂ) :
    dispersionOp beta z A = A * exp (I * (beta * z)) := rfl

/-- **The linear operator is lossless.** `|exp(iβz)| = 1`, so the dispersion step
preserves the field magnitude. This is the `‖D‖ = 1` of `PLAN.md` §6.2. -/
theorem dispersionOp_norm (beta z : ℝ) (A : ℂ) :
    ‖dispersionOp beta z A‖ = ‖A‖ := by
  rw [dispersionOp_eq, Complex.norm_mul, Complex.norm_exp]
  norm_num [Complex.mul_re, Complex.I_re, Complex.I_im]

/-- A zero phase thickness is the identity: no propagation, no change. -/
theorem dispersionOp_zero (beta : ℝ) (A : ℂ) (h : beta = 0) :
    dispersionOp beta 0 A = A := by
  rw [dispersionOp_eq, h]
  norm_num

/-- Two dispersion steps compose by adding their phase thicknesses. -/
theorem dispersionOp_add (b₁ b₂ z : ℝ) (A : ℂ) :
    dispersionOp b₂ z (dispersionOp b₁ z A) = dispersionOp (b₁ + b₂) z A := by
  rw [dispersionOp_eq, dispersionOp_eq, dispersionOp_eq, mul_assoc]
  rw [← Complex.exp_add]
  congr 1
  push_cast
  ring

/--
**A split step without the shock term conserves photon number exactly.**

The dispersion operator is unitary and the Kerr phase operator is unitary, so
the composition of the two is unitary. Since `gnlse.py` does report
photon-number drift, that drift cannot originate in these two terms.
-/
theorem splitStep_norm (A : ℂ) (beta z gamma P dz : ℝ) :
    ‖kerrStep (dispersionOp beta z A) gamma P dz‖ = ‖A‖ := by
  rw [kerrStep_norm, dispersionOp_norm]

/--
`n` alternated dispersion/Kerr steps. The shock term is deliberately absent: it
is the term that can change norm, which is the point of isolating the others.
-/
noncomputable def alternating (beta z gamma P dz : ℝ) : ℕ → ℂ → ℂ
  | 0, A => A
  | n + 1, A =>
      kerrStep (dispersionOp beta z (alternating beta z gamma P dz n A)) gamma P dz

theorem alternating_zero (beta z gamma P dz : ℝ) (A : ℂ) :
    alternating beta z gamma P dz 0 A = A := rfl

theorem alternating_succ (beta z gamma P dz : ℝ) (n : ℕ) (A : ℂ) :
    alternating beta z gamma P dz (n + 1) A =
      kerrStep (dispersionOp beta z (alternating beta z gamma P dz n A)) gamma P dz := rfl

/--
**Any number of alternated dispersion and Kerr steps conserves norm.**

By induction on the number of steps, so the conservation is not an artefact of
taking a single step of each kind.
-/
theorem splitSeq_norm (A : ℂ) (beta z gamma P dz : ℝ) (n : ℕ) :
    ‖alternating beta z gamma P dz n A‖ = ‖A‖ := by
  induction n with
  | zero => simp [alternating]
  | succ n ih =>
      rw [alternating_succ, kerrStep_norm, dispersionOp_norm]
      exact ih

end Solvers
end Photonics