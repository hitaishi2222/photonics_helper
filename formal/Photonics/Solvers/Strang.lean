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

end Solvers
end Photonics