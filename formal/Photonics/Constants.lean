/-
Copyright (c) 2026 photonics_helper contributors.
Released under the repository licence (see LICENSE).

# Physical constants as relations, not values

`photonics_helper.core.constants` exports CODATA numbers from `scipy`.
Lean proves nothing about CODATA — it cannot, and should not. What Lean *can*
do is prove the **relations** those numbers must satisfy for the library's
formulas to mean what they say.

So every constant is an opaque real, and every theorem here is a relation
between them. That is exactly the content needed by
`photonics_helper/base.py`'s `Wavelength`/`Frequency`/`AngularFrequency`/
`Wavenumber`/`Energy` conversions.
-/

import Photonics.Conventions

namespace Photonics
namespace Constants

open Conventions

/-! ## The constants -/

/-- Speed of light in vacuum, m/s. Opaque. -/
axiom c : ℝ

/-- Planck's constant, J·s. Opaque. -/
axiom hPlanck : ℝ

/-- Vacuum permittivity, F/m. Opaque. -/
axiom eps0 : ℝ

/-- Vacuum permeability, H/m. Opaque. -/
axiom mu0 : ℝ

/-- Fundamental charge, C. Opaque. -/
axiom qe : ℝ

/-- Permittivity of free space is strictly positive. -/
axiom eps0_pos : 0 < eps0

/-- Permittivity of free space is strictly positive. -/
axiom mu0_pos : 0 < mu0

/-- The speed of light is strictly positive. -/
axiom c_pos : 0 < c

/-- Planck's constant is strictly positive. -/
axiom hPlanck_pos : 0 < hPlanck

/-! ## The relations the library's conversions depend on

Each theorem below is the *definition* of a conversion implemented in
`photonics_helper/base.py`, restated as a theorem with its domain hypothesis
made explicit. The hypotheses are not decoration: `Wavelength.to_freq` cannot
be stated without `lam > 0`, and making that visible is the point.
-/

/-- `Wavelength.to_freq`: lam·f = c. -/
theorem wavelength_freq (lam f : ℝ) (hlam : 0 < lam) (h : lam * f = c) : f = c / lam := by
  field_simp
  nlinarith [h]

/-- The frequency of a positive wavelength is positive. -/
theorem freq_of_wavelength_pos (lam : ℝ) (f : ℝ) (hlam : 0 < lam) (h : lam * f = c) :
    0 < f := by
  rw [wavelength_freq lam f hlam h]
  exact div_pos c_pos hlam

/-- `Frequency.to_omega`: angular frequency is the frequency scaled by `2π`. -/
noncomputable def omega (f : ℝ) : ℝ := 2 * Real.pi * f

theorem omega_eq (f : ℝ) : omega f = 2 * Real.pi * f := rfl

/-- `omega > 0` whenever `f > 0`: required to divide by `ω₀`, which every
nonlinear coefficient (`γ`, the shock term, the SPM detuning) does. -/
theorem omega_pos (f : ℝ) (h : 0 < f) : 0 < omega f := by
  unfold omega
  positivity

/-- `AngularFrequency.to_wavenumber`: the wavenumber is `k = ω/c`. -/
noncomputable def k (ω : ℝ) : ℝ := ω / c

theorem k_eq (ω : ℝ) : k ω = ω / c := rfl

/--
**Wavenumber in cycles per metre**, `kt = 1/lam = w/(2*pi*c)`.

This is a *different quantity* from `k` above, by a factor of `2*pi`, and it is
the one the Python library actually computes: `Wavelength.to_wn` returns
`1/as_m` and `AngularFrequency.to_wn` returns `as_rad_s/(2*PI*C_MS)`, both of
which give cycles per metre. See `CONFORMANCE.md` F2.
-/
noncomputable def kTilde (ω : ℝ) : ℝ := ω / (2 * Real.pi * c)

theorem kTilde_eq (ω : ℝ) : kTilde ω = ω / (2 * Real.pi * c) := rfl

/-- The two wavenumber conventions differ by exactly `2*pi`. -/
theorem k_of_kTilde (ω : ℝ) (hc : c ≠ 0) : k ω = 2 * Real.pi * kTilde ω := by
  unfold k kTilde
  field_simp

/-- The cycles-per-metre wavenumber is the reciprocal of the wavelength. -/
theorem kTilde_of_wavelength (ω : ℝ) (lam : ℝ) (h : lam * ω = 2 * Real.pi * c)
    (hpos : lam ≠ 0) (hc : c ≠ 0) :
    kTilde ω = 1 / lam := by
  unfold kTilde
  field_simp
  ring_nf
  nlinarith [h]

/-- The wavenumber of a positive angular frequency is positive. -/
theorem k_pos (ω : ℝ) (h : 0 < ω) (hc : c > 0) : 0 < k ω := by
  unfold k
  exact div_pos h hc

-- `Wavelength.to_energy` is e = h·c/lam. As a *definition* of the energy
-- of a wavelength it needs no separate theorem; the positivity fact below is
-- the part that the library relies on (energies are used as positive scales).

/-- Energy of a positive wavelength is positive. -/
theorem energy_pos (lam : ℝ) (hlam : 0 < lam) (hh : 0 < hPlanck) (hc : 0 < c) :
    0 < hPlanck * c / lam := by
  exact div_pos (mul_pos hh hc) hlam

/-! ## Vacuum impedance

`base.py:18` computes the vacuum impedance as `Z0 = 1/(eps0*c)`. Its agreement
with the permeability rests on the identity `mu0*eps0*c^2 = 1`, which is a
physical fact about measured constants, not a theorem: it holds because the
constants are what they are. It is therefore recorded as an explicit axiom
rather than silently assumed or, worse, pretended to be provable.

Without this, the three constants are independent opaque reals and nothing in
the specification would notice a future edit that broke the consistency.
See `CONFORMANCE.md` F3.
-/

/--
Consistency of the three electromagnetic constants.

**This is an assumption about nature, not a theorem.** CODATA's values satisfy
it to measurement precision; it cannot be proved from anything else in this
file.
-/
axiom mu0_eps0_c : mu0 * eps0 * c ^ 2 = 1

/-- Vacuum impedance, as the library computes it: `Z0 = 1/(eps0*c)`. -/
noncomputable def Z0 : ℝ := 1 / (eps0 * c)

theorem Z0_eq : Z0 = 1 / (eps0 * c) := rfl

/-- `Z0` is positive, being a reciprocal of two positive constants. -/
theorem Z0_pos : 0 < Z0 := by
  unfold Z0
  exact div_pos one_pos (mul_pos eps0_pos c_pos)

/-- `Z0^2 = mu0/eps0`, the textbook form. Follows from the physical
consistency axiom. -/
theorem Z0_sq (heps : eps0 ≠ 0) (hc : c ≠ 0) : Z0 ^ 2 = mu0 / eps0 := by
  have h3 : mu0 = 1 / (eps0 * c ^ 2) := by
    field_simp
    nlinarith [mu0_eps0_c]
  calc Z0 ^ 2 = 1 / (eps0 ^ 2 * c ^ 2) := by unfold Z0; field_simp
    _ = mu0 / eps0 := by rw [h3]; field_simp

/-- The identity that would fail if one of the constants were edited
independently. -/
theorem mu0_eq (heps : eps0 ≠ 0) (hc : c ≠ 0) :
    mu0 = (1 / (eps0 * c)) ^ 2 * eps0 := by
  have h3 : mu0 = 1 / (eps0 * c ^ 2) := by
    field_simp
    nlinarith [mu0_eps0_c]
  rw [h3]
  field_simp

/-! ## The carrier relation

`lam₀` (carrier wavelength) and `ω₀` (carrier angular frequency) are related by
`ω₀ = 2πc/lam₀`. The nonlinear coefficients (`γ`, the shock operator, the lam map
on the spectral grid) are all written in terms of `ω₀`, and `ISSUES.md`-class
sign and scaling errors all trace back to this relation being applied with the
wrong wavelength.
-/

/-- Carrier angular frequency for a carrier wavelength. -/
noncomputable def omega0 (lam₀ : ℝ) : ℝ := 2 * Real.pi * c / lam₀

theorem omega0_eq (lam₀ : ℝ) : omega0 lam₀ = 2 * Real.pi * c / lam₀ := rfl

/-- `ω₀ > 0` for a positive carrier wavelength — the hypothesis under which
`γ = n₂ω₀Γ/(c·A_eff)` and the shock operator are well defined. -/
theorem omega0_pos (lam₀ : ℝ) (hlam : 0 < lam₀) (hc : 0 < c) : 0 < omega0 lam₀ := by
  unfold omega0
  positivity

/--
Recovering the carrier wavelength from the carrier angular frequency: `ω₀·lam₀`
is `2πc`.

(An earlier formulation asserted `c/ω₀·2 = 2·lam₀`, which is false — it drops
the `π`. The product form is the relation the library actually uses.)
-/
theorem lambda0_of_omega0 (lam₀ : ℝ) (hlam : 0 < lam₀) :
    omega0 lam₀ * lam₀ = 2 * Real.pi * c := by
  unfold omega0
  field_simp

end Constants
end Photonics