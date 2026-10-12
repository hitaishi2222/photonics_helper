/-
Copyright (c) 2026 Hitaishi V. All rights reserved.
Released under MIT license. See LICENSE for details.

# Stage S5 — RK4IP: the interaction-picture algebra Lean can check

The Python side (`photonics_helper.gnlse`, `scheme="rk4ip"`) advances the GNLSE
`∂_z A = L·A + N(A)` with Hult's fourth-order Runge–Kutta *in the interaction
picture* (J. Hult, *J. Lightwave Technol.* **25**(12), 3770 (2007), Eq. (12)):

    A_I = exp(−(z − z̄)L̂)·A                 (transform to the picture)
    ∂_I A_I = N̂_I A_I                        (the linear term is transformed away)
    A⁺  = exp(h L̂)·A_I(z + h)               (transform back)

`L̂` is diagonal in the Fourier basis, so on the grid each bin is the scalar
multiplication `exp(i·φ·h)` — the `_linear_step` of `gnlse.py`. What is
checkable here (PLAN §17.5, §18) is the *algebra*: that the transform pair is an
involution, that a purely linear flow is integrated exactly, that with `L = 0`
the scheme collapses to the classical RK4 update whose order-4 condition
`RK4.lean` proves, and that the continuous interaction-picture transform removes
the linear term. The empirical order and any error magnitude are
floating-point and are deliberately not claimed (PLAN §11).

This file is the *per-bin complex, general-`N`* transcription of the shipped
`:meth:`SplitStepEngine._rk4ip_step`. `RK4.lean` already proves the scalar
constant-coefficient factorisation `ipStep = exp(λh)·R(μh)` (rows H4/H7 in
`CONFORMANCE.md`); the theorems below are the ones specific to the outer-level
GNLSE step: exactness of the linear half and the `N = 0` collapse to classical
RK4.
-/

import Mathlib
import Photonics.Solvers.RK4

namespace Photonics
namespace Solvers

open Complex

/-! ## The diagonal linear propagator

One scalar per bin: the GNLSE linear operator is `i·φ` with
`φ(Ω) = Σ_{k≥2} βₖΩᵏ/k!` (loss is a real damping factor and does not affect the
algebra). `linStep φ h A` is what `_linear_step` applies.
-/

/-- The exact per-bin linear propagator: multiplication by `exp(i·φ·h)`. -/
noncomputable def linStep (φ h : ℝ) (A : ℂ) : ℂ := A * exp (I * (φ * h))

/-- The transform pair is an involution: `linStep φ h ∘ linStep φ (−h) = id`.

This is what makes "transform to the picture, advance, transform back" not bias
the field by itself. -/
theorem linStep_comp_inv (φ h : ℝ) (A : ℂ) :
    linStep φ h (linStep φ (-h) A) = A := by
  simp only [linStep]
  rw [mul_assoc, ← exp_add]
  have hsum : I * (↑φ * ↑(-h)) + I * (↑φ * ↑h) = 0 := by push_cast; ring
  rw [hsum, exp_zero, mul_one]

/-- Composition of two linear propagators adds their steps: `h ↦ linStep φ h`
is a one-parameter group. With `linStep_comp_inv` this is the property
`rk4ipStep_linear` exploits to cancel the two half transforms. -/
theorem linStep_comp (φ h₁ h₂ : ℝ) (A : ℂ) :
    linStep φ h₁ (linStep φ h₂ A) = linStep φ (h₁ + h₂) A := by
  simp only [linStep]
  rw [mul_assoc, ← exp_add]
  congr 1
  push_cast
  ring_nf

/-! ## The RK4IP step

Transcribed from `_rk4ip_step` on the grid, with the *grid FFT operators
replaced by their scalar symbols* (`_linear_step → linStep φ ·`,
`_nonlinear_rhs → N`). Because the linear operator is diagonal, every bin is
independent and the whole scheme is this complex map. -/

/-- One RK4IP step (Hult 2007, Eq. (12)), per bin. -/
noncomputable def rk4ipStep (φ h : ℝ) (N : ℂ → ℂ) (A : ℂ) : ℂ :=
  let E := fun X => linStep φ (h / 2) X
  let A_I := E A
  let k1 := E (h * N A)
  let k2 := h * N (A_I + k1 / 2)
  let k3 := h * N (A_I + k2 / 2)
  let k4 := h * N (E (A_I + k3))
  E (A_I + (k1 + 2 * k2 + 2 * k3) / 6) + k4 / 6

/-- **A purely linear flow is integrated exactly.** With `N = 0` the
interaction-picture transform carries the linear operator *without*
approximation: the scheme returns `exp(i·φ·h)·A = linStep φ h A`. There is no
splitting error for the part of the flow the transform removes — the Python test
`test_rk4ip_is_exact_for_a_purely_linear_flow` pins the same fact to
round-off. -/
theorem rk4ipStep_linear (φ h : ℝ) (A : ℂ) :
    rk4ipStep φ h (fun _ => 0) A = linStep φ h A := by
  unfold rk4ipStep linStep
  simp only [mul_zero, zero_mul, add_zero, zero_div]
  have harg : I * (↑φ * ↑(h / 2)) + I * (↑φ * ↑(h / 2)) = I * (↑φ * ↑h) := by
    push_cast; ring
  rw [mul_assoc, ← exp_add, harg]

/-- The classical RK4 update on `∂_z A = N(A)` (no linear part), as the
combination `A + (k₁ + 2k₂ + 2k₃ + k₄)/6`. This is the integrator `RK4.lean`'s
`rk4Step` is the real-valued version of. -/
noncomputable def classicalRK4 (h : ℝ) (N : ℂ → ℂ) (A : ℂ) : ℂ :=
  let k1 := h * N A
  let k2 := h * N (A + k1 / 2)
  let k3 := h * N (A + k2 / 2)
  let k4 := h * N (A + k3)
  A + (k1 + 2 * k2 + 2 * k3 + k4) / 6

/-- `linStep 0 ·` is the identity — the half transform with no dispersion. -/
@[simp] theorem linStep_zero (h : ℝ) (A : ℂ) : linStep 0 h A = A := by
  simp [linStep]

/-- **With `L = 0` the scheme *is* classical RK4.** All half transforms are the
identity, so RK4IP collapses to the standard combination whose order-4
consistency `RK4.lean` proves. This is the bridge between the
interaction-picture formulation and the classical integrator the library
already trusts. -/
theorem rk4ipStep_no_dispersion (h : ℝ) (N : ℂ → ℂ) (A : ℂ) :
    rk4ipStep 0 h N A = classicalRK4 h N A := by
  unfold rk4ipStep classicalRK4
  simp only [linStep_zero]
  ring

/-! ## The interaction picture removes the linear term

The transform's whole purpose, stated for the *continuous* flow: if `A`
satisfies `A' = i·φ·A + N(A)`, then `A_I(z) = exp(−i·φ·(z − z₀))·A(z)` satisfies
`A_I'(z) = exp(−i·φ·(z − z₀))·N(A(z))` — the `i·φ·A` term cancels exactly. -/

/-- Derivative of the interaction-picture phase factor
`z ↦ exp(−i·φ·(z − z₀))`. -/
theorem hasDerivAt_ipPhase (φ z₀ z : ℝ) :
    HasDerivAt (fun w : ℝ => exp (-(I * (φ : ℂ)) * ((w : ℂ) - (z₀ : ℂ))))
      (exp (-(I * (φ : ℂ)) * ((z : ℂ) - (z₀ : ℂ))) * (-(I * (φ : ℂ)))) z := by
  have h1 : HasDerivAt (fun w : ℝ => (w : ℂ) - (z₀ : ℂ)) 1 z :=
    ofRealCLM.hasDerivAt.sub_const (z₀ : ℂ)
  have hinner :
      HasDerivAt (fun w : ℝ => -(I * (φ : ℂ)) * ((w : ℂ) - (z₀ : ℂ)))
        (-(I * (φ : ℂ))) z := by
    have h := h1.const_mul (-(I * (φ : ℂ)))
    simpa using h
  simpa using hinner.cexp

/-- **The interaction-picture transformation removes the linear term.**
If `A' = i·φ·A + N(A)` then the picture field
`A_I(z) = exp(−i·φ·(z − z₀))·A(z)` obeys the *linear-free* equation
`A_I' = exp(−i·φ·(z − z₀))·N(A)`. -/
theorem interactionPicture_removes_linear
    (φ z₀ z : ℝ) (A : ℝ → ℂ) (N : ℂ → ℂ)
    (hA : HasDerivAt A (I * (φ : ℂ) * A z + N (A z)) z) :
    HasDerivAt (fun w : ℝ => A w * exp (-(I * (φ : ℂ)) * ((w : ℂ) - (z₀ : ℂ))))
      (N (A z) * exp (-(I * (φ : ℂ)) * ((z : ℂ) - (z₀ : ℂ)))) z := by
  have hphase := hasDerivAt_ipPhase φ z₀ z
  have hprod := hA.mul hphase
  have hval :
      (I * (φ : ℂ) * A z + N (A z))
          * exp (-(I * (φ : ℂ)) * ((z : ℂ) - (z₀ : ℂ)))
        + A z
          * (exp (-(I * (φ : ℂ)) * ((z : ℂ) - (z₀ : ℂ))) * (-(I * (φ : ℂ))))
      = N (A z) * exp (-(I * (φ : ℂ)) * ((z : ℂ) - (z₀ : ℂ))) := by
    ring
  rwa [hval] at hprod

end Solvers
end Photonics
