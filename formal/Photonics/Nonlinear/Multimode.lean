import Mathlib

/-!
# Multimode GNLSE: coefficient models and conservation identities

The algebra the Python multimode engine relies on, stated over `ℝ` / `ℂ`:

* the three SPM/XPM/FWM coefficient models, including the **co-polarized
  spatial-mode** limit (`SPM 1, XPM 2, FWM 1`) that the 2026-10 audit added;
* the FWM angular-momentum gate is symmetric under exchanging the two arms
  of an exchange pair (`ℓ_m = 2ℓ_n − ℓ_q ↔ ℓ_q = 2ℓ_n − ℓ_m`);
* the depleted three-mode FWM set conserves `Σ|A|²` exactly — the algebraic
  content of the engine's `fwm_pump_depletion` and its RK4 regression;
* the shock symbol applied **twice** is `(1+Ωτ)²`, not `(1+Ωτ)` — the
  operator-application bug the step10 regression pins.

These are coefficient/algebra statements. The modal-overlap integrals
themselves are measure-theoretic and remain outside this development
(`CONFORMANCE.md`, G12).
-/

namespace Photonics
namespace Nonlinear

open Complex
open scoped ComplexConjugate

/-! ## The coefficient models -/

/-- The engine's `coef_model` choices. `lpDegenerate` is the two-polarization
vector-GNLSE strip-down; `coPolarized` is the scalar co-polarized spatial-mode
limit. -/
inductive CoeffModel where
  | lpDegenerate
  | isotropic
  | coPolarized
  deriving DecidableEq, Repr

/-- SPM/XPM weight of the off-diagonal slot (`i ≠ j`). -/
noncomputable def xpmFactor : CoeffModel → ℝ
  | .lpDegenerate => 2 / 3
  | .isotropic => 1
  | .coPolarized => 2

/-- Pump-driven inter-modal FWM weight. -/
noncomputable def fwmFactor : CoeffModel → ℝ
  | .lpDegenerate => 2 / 3
  | .isotropic => 1
  | .coPolarized => 1

theorem xpmFactor_lpDegenerate : xpmFactor .lpDegenerate = 2 / 3 := rfl

theorem xpmFactor_isotropic : xpmFactor .isotropic = 1 := rfl

/-- **Co-polarized spatial modes have XPM = 2.** Two equal XPM index
permutations in the MM-GNLSE contraction; SPM stays 1. This is the limit the
engine's default `lpDegenerate` model does *not* represent. -/
theorem xpmFactor_coPolarized : xpmFactor .coPolarized = 2 := rfl

theorem fwmFactor_coPolarized : fwmFactor .coPolarized = 1 := rfl

/-- SPM is always one in every model (the overall `γ` carries the scale). -/
noncomputable def spmFactor (_ : CoeffModel) : ℝ := 1

theorem spmFactor_eq (m : CoeffModel) : spmFactor m = 1 := rfl

/--
The co-polarized instantaneous phase rate of channel `i`:
`γ·(P_i + 2·P_j)`. The engine's `_diagonal_phase` forms exactly this with
`coef_model="co_polarized"` and `xpm_weights=None`.
-/
noncomputable def coPolarizedRate (gamma Pi Pj : ℝ) : ℝ := gamma * (Pi + 2 * Pj)

theorem coPolarizedRate_eq (gamma Pi Pj : ℝ) :
    coPolarizedRate gamma Pi Pj = gamma * (Pi + 2 * Pj) := rfl

/-- Swapping the two channels swaps self and cross consistently: the pair
`(γ(P_i+2P_j), γ(P_j+2P_i))` is the same rule read from either channel. -/
theorem coPolarizedRate_swap (gamma Pi Pj : ℝ) :
    coPolarizedRate gamma Pj Pi = gamma * (Pj + 2 * Pi) := rfl

/-- A zero cross term reduces the two-mode rate to single-mode SPM. -/
theorem coPolarizedRate_no_cross (gamma Pi : ℝ) :
    coPolarizedRate gamma Pi 0 = gamma * Pi := by
  unfold coPolarizedRate
  ring

/-! ## The FWM selection rule -/

/-- The engine's angular-momentum gate for the `→ m` arm `A_n A_p A_q*`:
`ℓ_m = ℓ_n + ℓ_p − ℓ_q`. -/
def fwmAllowed (lm ln lp lq : ℤ) : Prop := lm = ln + lp - lq

/--
**The two arms of an exchange pair satisfy the same gate.** For a pump `n`
and pair `(m, q)` the engine checks `ℓ_m = 2ℓ_n − ℓ_q` once and integrates
both `dA_m` and `dA_q`; this theorem says the partner arm is automatically
allowed under `ℓ_q = 2ℓ_n − ℓ_m`.
-/
theorem fwmAllowed_partner {lm ln lq : ℤ}
    (h : fwmAllowed lm ln ln lq) : fwmAllowed lq ln ln lm := by
  unfold fwmAllowed at *
  omega

/-- The gate is invariant when the two pump indices are exchanged. -/
theorem fwmAllowed_pump_swap {lm ln lp lq : ℤ}
    (h : fwmAllowed lm ln lp lq) : fwmAllowed lm lp ln lq := by
  unfold fwmAllowed at *
  omega

/-! ## Depleted FWM conserves `Σ|A|²` -/

/--
**The three-mode depleted FWM set is exactly photon-conserving.**

With real coupling `c = γ·f` (the default, no imaginary gain tensor) and

* `dA_m = i·c·A_n²·A_q*`
* `dA_q = i·c·A_n²·A_m*`
* `dA_n = 2i·c·|A_n|²`-weighted back-conversion `2i·c·A_m·A_q·A_n*`,

the sum of the `Re(conj(A)·dA)` terms is zero.  `Σ|A|²` is therefore
stationary, which is what `tests/test_multimode_v2.py
::test_depleted_conserves_energy_machine_precision` measures to RK4
round-off.  (The factor two on the pump arm is the two pump photons per
exchange event; without it the sum is not zero.)
-/
theorem depletedFwm_energy_conserved (a b n : ℂ) (c : ℝ) :
    (star a * (I * (c : ℂ) * n ^ 2 * star b)).re
      + (star b * (I * (c : ℂ) * n ^ 2 * star a)).re
      + (star n * (2 * I * (c : ℂ) * star n * a * b)).re = 0 := by
  simp only [pow_two, Complex.mul_re, Complex.mul_im, Complex.star_def,
    Complex.conj_re, Complex.conj_im, Complex.I_re, Complex.I_im,
    Complex.ofReal_re, Complex.ofReal_im]
  norm_num
  ring

/-- The same statement as the derivative of the norm-squared sum with the
explicit factor `2` that `d|A|²/dz = 2·Re(conj A · dA)` carries. -/
theorem depletedFwm_normSq_rate (a b n : ℂ) (c : ℝ) :
    2 * (star a * (I * (c : ℂ) * n ^ 2 * star b)).re
      + 2 * (star b * (I * (c : ℂ) * n ^ 2 * star a)).re
      + 2 * (star n * (2 * I * (c : ℂ) * star n * a * b)).re = 0 := by
  have h := depletedFwm_energy_conserved a b n c
  linarith

/-- **The pairing alone is not photon-conserving.** A concrete witness with
unit pump and a relative phase between the two arms gives a non-zero pair
rate; the pump back-conversion arm is what cancels it, which is why
`fwm_pump_depletion` is the Manley–Rowe-consistent option. -/
theorem fwm_pair_only_not_conserved :
    ∃ (a b : ℂ),
      (star a * (I * (1 : ℂ) * (1 : ℂ) ^ 2 * star b)).re
        + (star b * (I * (1 : ℂ) * (1 : ℂ) ^ 2 * star a)).re ≠ 0 := by
  refine ⟨1, I, ?_⟩
  norm_num [Complex.ext_iff]

/-! ## The shock operator: applied once -/

/-- The per-bin shock symbol `ω/ω₀ = 1 + Ωτ` of the codebase's `e^{-iΩt}`
convention (`Nonlinear.Shock::code_symbol_is_exact`). -/
noncomputable def shockSymbol (tau Omega : ℝ) : ℝ := 1 + tau * Omega

theorem shockSymbol_eq (tau Omega : ℝ) : shockSymbol tau Omega = 1 + tau * Omega := rfl

/-- Applying the shock operator twice squares its symbol. -/
theorem shockSymbol_sq (tau Omega : ℝ) :
    shockSymbol tau Omega ^ 2 = 1 + 2 * (tau * Omega) + (tau * Omega) ^ 2 := by
  unfold shockSymbol
  ring

/-- The difference between the double- and single-application symbols. -/
theorem shockSymbol_double_sub (tau Omega : ℝ) :
    shockSymbol tau Omega ^ 2 - shockSymbol tau Omega
      = (tau * Omega) * (1 + tau * Omega) := by
  unfold shockSymbol
  ring

/--
**Applying shock twice is not the identity of applying it once.** For a
physical bin (`τΩ ≠ 0`, and not the negative-frequency root `τΩ = -1`)
the squared symbol differs — the step10 regression's `(1+iτΩ)²` bug.
-/
theorem shockSymbol_double_ne_single {tau Omega : ℝ}
    (h0 : tau * Omega ≠ 0) (h1 : tau * Omega ≠ -1) :
    shockSymbol tau Omega ^ 2 ≠ shockSymbol tau Omega := by
  intro h
  have hsub : shockSymbol tau Omega ^ 2 - shockSymbol tau Omega = 0 := sub_eq_zero.mpr h
  rw [shockSymbol_double_sub] at hsub
  rcases mul_eq_zero.mp hsub with hz | hm
  · exact h0 hz
  · exact h1 (by linarith)

end Nonlinear
end Photonics
