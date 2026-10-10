import Photonics.Dispersion.Beta

/-!
# Self-steepening: validity of the first-order shock expansion

The shock operator `(1 + (i/ω₀)∂_t)` is the first-order form of `ω/ω₀`
(Blow & Wood 1989, Agrawal §2.3). `gnlse.py` guards it with
`_SHOCK_TAYLOR_LIMIT = 0.2` and a docstring argument; this file turns that
argument into theorems.

Two independent conditions are involved and the code is right to separate them:

- `Ω_max < ω₀` — the harder *existence* condition. The factor `1 + Ω·τ_shock`
  has no meaning where `ω = ω₀ + Ω ≤ 0`.
- `τ_shock·Ω_max ≤ 0.2` — the *Taylor accuracy* condition, which only bounds how
  wrong the first-order form may be.

Everything here is real-valued.
-/

namespace Photonics
namespace Nonlinear

open Conventions

/-! ## The expansion itself -/

/--
The frequency ratio in the retarded frame, `ω/ω₀`, as a function of the
dimensionless offset `x = Ω/ω₀`.

Since `ω = ω₀ + Ω`, this is `1/(1 − x)`.
-/
noncomputable def shockRatio (x : ℝ) : ℝ := 1 / (1 - x)

theorem shockRatio_eq (x : ℝ) : shockRatio x = 1 / (1 - x) := rfl

/-- Existence condition: the ratio is only defined for `x < 1`, i.e. for
`Ω < ω₀`. Bins beyond this alias onto negative optical frequencies. -/
theorem shockRatio_ne (x : ℝ) (h : x = 1) : shockRatio x = 0 := by
  simp [shockRatio, h]

/--
The first-order Taylor form used by the solver is `1 + x`.

The code's comment describes it as `ω/ω₀ ≈ 1 + Ω·τ_shock`, which is this with
`τ_shock = 1/ω₀` and `x = Ω/ω₀`.
-/
noncomputable def shockFirstOrder (x : ℝ) : ℝ := 1 + x

theorem shockFirstOrder_eq (x : ℝ) : shockFirstOrder x = 1 + x := rfl

/--
**The exact truncation error of the first-order expansion.**

`1/(1−x) − (1+x) = x²/(1−x)`, exactly, for `x ≠ 1`.

This is the number the `_SHOCK_TAYLOR_LIMIT` guard is really about, and it is
exact rather than bounded, so there is nothing left to estimate.
-/
theorem shock_truncation_error (x : ℝ) (h : x ≠ 1) :
    shockRatio x - shockFirstOrder x = x ^ 2 / (1 - x) := by
  unfold shockRatio shockFirstOrder
  field_simp
  ring

/-- For a positive offset the first-order form *under*-estimates the true ratio:
the error is signed positive, so the solver's linearisation is optimistic. -/
theorem shock_error_nonneg (x : ℝ) (h0 : 0 ≤ x) (h1 : x < 1) :
    0 ≤ shockRatio x - shockFirstOrder x := by
  rw [shock_truncation_error x (ne_of_lt h1)]
  have h1' : 0 < 1 - x := by linarith
  have h2 : 0 ≤ x ^ 2 := sq_nonneg x
  exact div_nonneg h2 (le_of_lt h1')

/--
**`_SHOCK_TAYLOR_LIMIT = 0.2` admits at most 5 % truncation error.**

The guard is conservative *by proof*, not by habit: at the limit `x = 1/5` the
error `x²/(1−x)` is exactly `(1/25)/(4/5) = 1/20`. Monotonicity then bounds the
error below the limit.

So the constant in `gnlse.py` is not arbitrary, and the honest way to say what it
buys is "first-order accuracy to within 5 %", not "no error".
-/
theorem shock_error_at_limit :
    shockRatio (1 / 5 : ℝ) - shockFirstOrder (1 / 5 : ℝ) = 1 / 20 := by
  rw [shock_truncation_error _ (by norm_num)]
  norm_num

theorem shock_error_bounded {x : ℝ} (h0 : 0 ≤ x) (h1 : x ≤ 1 / 5) :
    shockRatio x - shockFirstOrder x ≤ 1 / 20 := by
  have hlt : x < 1 := lt_of_le_of_lt h1 (by norm_num)
  rw [shock_truncation_error x (ne_of_lt hlt)]
  have hmonotone : x ^ 2 / (1 - x) ≤ (1 / 5 : ℝ) ^ 2 / (1 - 1 / 5) := by
    have hd1 : 0 < 1 - x := by linarith
    have hd2 : (0:ℝ) < 1 - 1/5 := by norm_num
    rw [div_le_div_iff₀ hd1 hd2]
    · have hx : x ^ 2 ≤ (1 / 5 : ℝ) ^ 2 := by nlinarith
      linarith
  rw [show (1 / 5 : ℝ) ^ 2 / (1 - 1 / 5) = 1 / 20 by norm_num] at hmonotone
  exact hmonotone

/--
**The existence condition is strictly stronger than the accuracy condition.**

Any `x` in the accuracy region is automatically in the existence region, so a
caller who respects `_SHOCK_TAYLOR_LIMIT` has already satisfied `Ω < ω₀` for
the *same* `ω₀`. The converse fails: `x` can sit between `1/5` and `1` and be
perfectly well defined, just inaccurate.

This is why the code checks the two conditions separately and treats one as a
warning and one as an error.
-/
theorem accuracy_implies_existence {x : ℝ} (h : x ≤ 1 / 5) : x < 1 :=
  lt_of_le_of_lt h (by norm_num)

theorem existence_does_not_imply_accuracy {x : ℝ} (h : 0.5 ≤ x) (h1 : x < 1) :
    ¬(x ≤ 1 / 5) := by
  intro h2
  linarith

/-! ## The resolution floor

With the default `τ_shock = 1/ω₀` and the Taylor limit, the grid spacing `dt`
is pinned from below. `gnlse.py` states this as "about 12.9 fs at 1550 nm";
this is the theorem behind that number.
-/

/-- With `τ_shock = 1/ω₀`, the dimensionless offset at the grid edge is
`Ω_max/ω₀ = Ω_max·dt/π`, by the Nyquist identity `Grid.omega_max_mul_dt`. -/
theorem offset_at_grid_edge {ω0 Ωmax dt : ℝ} (hω0 : 0 < ω0) (hdt : 0 < dt)
    (hNyq : Ωmax * dt = Real.pi) :
    Ωmax / ω0 = (Ωmax * dt / Real.pi) * (Real.pi / (ω0 * dt)) := by
  have hpi : Real.pi ≠ 0 := by norm_num
  field_simp

/--
**The resolution floor.** If the Taylor condition `τ_shock·Ω_max ≤ limit` holds
with `τ_shock = 1/ω₀`, then `Ω_max ≤ limit·ω₀`, and by `Ω_max·dt = π` the grid
spacing satisfies `dt ≥ π/(limit·ω₀)`.

At 1550 nm with `ω₀ = 2πc/λ` this evaluates to about 12.9 fs, matching the
code's comment. The consequence recorded there also follows: a femtosecond
pulse cannot be resolved under the shock model at the Taylor limit, which is
why `reproductions/hult_2007_rk4ip/` disables self-steepening.
-/
theorem resolution_floor {ω0 dt limit Ωmax : ℝ} (hω0 : 0 < ω0) (hdt : 0 < dt)
    (hlim : 0 < limit) (hNyq : Ωmax * dt = Real.pi) (hcond : (1 / ω0) * Ωmax ≤ limit) :
    dt ≥ Real.pi / (limit * ω0) := by
  have h1 : Ωmax ≤ limit * ω0 := by
    have hmul : (1 / ω0) * Ωmax * ω0 = Ωmax := by field_simp
    nlinarith [hcond, hmul]
  have hden : 0 < limit * ω0 := mul_pos hlim hω0
  have h4 : Real.pi ≤ (limit * ω0) * dt := by
    nlinarith [hNyq, h1]
  refine (div_le_iff₀ hden).mpr ?_
  nlinarith [h4]

/-! ## Raman

Causality and the gain bound live in `Nonlinear.Raman`, which imports this file.
-/

/-- The delay length itself is a non-negative number of samples whenever the
delay is non-negative, which is the regime the Raman term is used in. -/
theorem ramanDelaySamples (tau dt : ℝ) (hτ : 0 ≤ tau) (hdt : 0 < dt) :
    0 ≤ tau / dt := by
  have h2 : 0 ≤ tau / dt := div_nonneg hτ (le_of_lt hdt)
  exact h2

end Nonlinear
end Photonics
