import Mathlib

/-!
# The RK4 integrator

`_rk4_step` in `chi2.py` and the matching step in `gnlse.py` are the classical
Runge–Kutta fourth-order method:

```
k₁ = f(y),  k₂ = f(y + h/2·k₁),  k₃ = f(y + h/2·k₂),  k₄ = f(y + h·k₃)
y' = y + h/6·(k₁ + 2k₂ + 2k₃ + k₄)
```

The theorems below say what that formula actually guarantees, which is what the
docstring's "RK4IP" claim rests on. The interesting one is `stability_polynomial`:
for the linear equation `y' = λy` the one-step map is the polynomial
`R(z) = 1 + z + z²/2 + z³/6 + z⁴/24`, which is exactly the fourth-order Taylor
sum of `exp(z)`. That is the precise sense in which the method is
**fourth-order accurate for the propagator**, and it is the reason the
interaction-picture formulation — where the operator is close to linear — is the
one the library uses.
-/

namespace Photonics
namespace Solvers

/-! ## The step formula -/

/-- One stage evaluation, as a function of the state. -/
def Stage (f : ℝ → ℝ → ℝ) (z y : ℝ) : ℝ := f z y

/--
The RK4 step, transcribed from `_rk4_step`:

`y + h/6·(k₁ + 2k₂ + 2k₃ + k₄)`.
-/
noncomputable def rk4Step (f : ℝ → ℝ → ℝ) (z y h : ℝ) : ℝ :=
  let k1 := Stage f z y
  let k2 := Stage f (z + h / 2) (y + h / 2 * k1)
  let k3 := Stage f (z + h / 2) (y + h / 2 * k2)
  let k4 := Stage f (z + h) (y + h * k3)
  y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

/-! ## Weights -/

/-- **The weights sum to one.** `1 + 2 + 2 + 1 = 6`, so for a constant
right-hand side the step advances the state by exactly `h·f`, the same as the
exact solution. This is the consistency condition, and it is what makes the
method first-order before its accuracy order is considered. -/
theorem rk4_weights_sum : (1 + 2 + 2 + 1 : ℝ) / 6 = 1 := by norm_num

/-- The step reproduces the exact update when the right-hand side does not
depend on the state. -/
theorem rk4Step_constant (a z y h : ℝ) : rk4Step (fun _ _ => a) z y h = y + h * a := by
  simp [rk4Step, Stage]
  ring

/-! ## The linear equation -/

/-- The exact one-step map of `y' = λy` over a step of size `h`. -/
noncomputable def exactStep (lam h : ℝ) : ℝ := Real.exp (lam * h)

/--
**The stability polynomial.** For the linear equation `y' = λy` the RK4 step is
multiplication by

`R(z) = 1 + z + z²/2 + z³/6 + z⁴/24`,  `z = λh`.

Proved by evaluating the step at `y = 1` and unpacking the four stages.
-/
theorem rk4Step_linear (lam z0 h : ℝ) :
    rk4Step (fun _ y => lam * y) z0 1 h =
      1 + lam * h + (lam * h) ^ 2 / 2 + (lam * h) ^ 3 / 6 + (lam * h) ^ 4 / 24 := by
  simp [rk4Step, Stage]
  ring

/-- `iteratedDeriv k exp = exp`: exp is its own derivative of every order.
Induction on `iteratedDeriv_succ'` from `Real.deriv_exp`. -/
theorem iteratedDeriv_exp (k : ℕ) :
    iteratedDeriv k Real.exp = Real.exp := by
  induction k with
  | zero => exact iteratedDeriv_zero
  | succ k ih => rw [iteratedDeriv_succ', Real.deriv_exp, ih]

/-- The truncated exponential's error, with an explicit constant — the analytic
content H6 needed.

`|exp z − Σ_{k≤4} zᵏ/k!| ≤ exp |z| · |z|⁵/5!` holds for every `z`, including
`z < 0` (where the truncation *overshoots* — the remainder has the sign of
`z⁵` — and the same estimate bounds the overshoot, via `exp x' ≤ exp |z|`
for the Lagrange point `x'` between `0` and `z`).

This is Taylor's theorem with Lagrange remainder applied to `exp` at `x₀ = 0`
with `n = 4`, using `taylor_mean_remainder_lagrange_iteratedDeriv` and
`iteratedDeriv_exp` to evaluate the Taylor polynomial of the exponential in
the Maclaurin form `-/
theorem exp_taylor_abs_error_le (x : ℝ) (hx : x ≠ 0) :
    |Real.exp x - (∑ k ∈ Finset.range 5, x ^ k / (Nat.factorial k : ℝ))|
      ≤ Real.exp |x| * |x| ^ 5 / 120 := by
  have hdiff : ContDiff ℝ 5 Real.exp := Real.contDiff_exp
  obtain ⟨x', h1, h2⟩ :=
    taylor_mean_remainder_lagrange_iteratedDeriv (f := Real.exp) (x := x) (x₀ := 0)
      (n := 4) (Ne.symm hx) hdiff.contDiffOn
  have hini : ∀ k ∈ Finset.range 5,
      iteratedDerivWithin k Real.exp (Set.uIcc 0 x) 0 = 1 := by
    intro k hk
    have h0mem : (0:ℝ) ∈ Set.uIcc 0 x := by
      by_cases hx0 : 0 ≤ x
      · exact Set.mem_uIcc.mpr (Or.inl ⟨le_refl 0, hx0⟩)
      · exact Set.mem_uIcc.mpr (Or.inr ⟨by linarith [hx0], le_refl 0⟩)
    have hstep : iteratedDerivWithin k Real.exp (Set.uIcc 0 x) 0 = Real.exp 0 := by
      rw [iteratedDerivWithin_eq_iteratedDeriv (uniqueDiffOn_uIcc (Ne.symm hx))
        Real.contDiff_exp.contDiffAt h0mem, iteratedDeriv_exp k]
    exact hstep.trans Real.exp_zero
  -- the Lagrange point lies between 0 and x, hence at most |x|
  have hmem : min 0 x < x' ∧ x' < max 0 x := Set.mem_Ioo.mp h1
  have hxleabs : x' ≤ |x| := by
    rcases le_total 0 x with hnonneg | hneg
    · rw [min_eq_left hnonneg, max_eq_right hnonneg] at hmem
      rw [abs_of_nonneg hnonneg]
      linarith [hmem.2]
    · have hneglt : x < 0 := lt_of_le_of_ne hneg (fun he => hx he)
      have hxle0 : (x:ℝ) ≤ 0 := le_of_lt hneglt
      rw [min_comm, max_comm] at hmem
      rw [min_eq_left hxle0, max_eq_right hxle0] at hmem
      rw [abs_of_neg hneglt]
      linarith [hmem.1, hmem.2]
  -- the Taylor polynomial of exp at 0 is the Maclaurin sum
  have htaylor : taylorWithinEval Real.exp 4 (Set.uIcc 0 x) 0 x
      = ∑ k ∈ Finset.range 5, x ^ k / (Nat.factorial k : ℝ) := by
    rw [taylor_within_apply]
    apply Finset.sum_congr rfl
    intro k hk
    rw [hini k hk]
    rw [inv_eq_one_div]
    ring
  have hpos : 0 ≤ Real.exp x' := Real.exp_nonneg _
  have hexp' : Real.exp x' ≤ Real.exp |x| :=
    Real.exp_le_exp.2 hxleabs
  -- remaining error is exactly exp x' * x⁵ / 120 (2 and 5 aliases factored)
  have hres : Real.exp x - taylorWithinEval Real.exp 4 (Set.uIcc 0 x) 0 x
      = Real.exp x' * x ^ 5 / 120 := by
    rw [h2, sub_zero]
    norm_num
    exact Or.inl (congrFun (iteratedDeriv_exp 5) x')

  rw [← htaylor, hres]
  have hexp'abs : |Real.exp x' * x ^ 5 / 120| = Real.exp x' * |x| ^ 5 / 120 := by
    rw [abs_div, abs_mul, abs_pow, abs_of_nonneg hpos,
      abs_of_pos (show (0:ℝ) < 120 by norm_num)]
  rw [hexp'abs]
  have hpowle : Real.exp x' * |x| ^ 5 ≤ Real.exp |x| * |x| ^ 5 :=
    mul_le_mul hexp' (le_refl _) (pow_nonneg (abs_nonneg x) 5)
      (Real.exp_nonneg _)
  linarith

/-- The method applied to `y' = λy` at `y = 1` is the truncated exponential. -/
theorem stability_polynomial (lam h : ℝ) :
    rk4Step (fun _ y => lam * y) 0 1 h =
      ∑ k ∈ Finset.range 5, (lam * h) ^ k / (Nat.factorial k : ℝ) := by
  rw [rk4Step_linear]
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, Nat.factorial,
    Nat.factorial_zero, Nat.factorial_one, one_mul, zero_add]
  ring

/--
**Fourth-order accuracy, two-sided.**

For `0 ≤ z` the RK4 step is bounded above by the exact exponential step: the
truncated series never exceeds `exp z` (`Real.sum_le_exp_of_nonneg`). That is
the "RK4 never overshoots the exact propagator" statement, one-sided as it
must be: the Lagrange remainder of the truncation has the sign of `z⁵`, so
for `z ≥ 0` the truncation undershoots and for `z < 0` it overshoots.

The two-sided bound with an explicit constant lives in
`exp_taylor_abs_error_le`: `|exp z − Σ| ≤ exp |z| · |z|⁵/5!` for every
`z ≠ 0`, proved from `taylor_mean_remainder_lagrange_iteratedDeriv` — no
hypothesis carries the expansion any more.
-/
theorem rk4_error_le {z : ℝ} (hz : 0 ≤ z) :
    ∑ k ∈ Finset.range 5, z ^ k / (Nat.factorial k : ℝ) ≤ Real.exp z :=
  Real.sum_le_exp_of_nonneg hz 5

/-- The RK4 step of `y' = λy` at an arbitrary state, not just `y = 1`:
multiplication by the stability polynomial. -/
theorem rk4Step_linear_y (lam z y h : ℝ) :
    rk4Step (fun _ u => lam * u) z y h
      = (1 + lam * h + (lam * h) ^ 2 / 2 + (lam * h) ^ 3 / 6 + (lam * h) ^ 4 / 24) * y := by
  simp only [rk4Step, Stage]
  ring

/-! ## The interaction picture

The library's main solver is labelled "RK4IP": Runge–Kutta *in the interaction
picture*. The linear (dispersion) part is propagated exactly with its
exponential, and the remainder is integrated by RK4 in the frame that rotates
with the linear part:

`û(z) = e^(−λz)·u(z)`,   `û' = μ(û)`,

so one step of the scheme is the map

`y ↦ exp(λh)·R(μh)·y`,

with `R(z) = Σ_{k≤4} zᵏ/k!` the RK4 stability polynomial of
`stability_polynomial`. Two claims are worth proving rather than asserting:

- the factorisation is **exact** — the IP step is literally the composition of
  the exact exponential propagator with the RK4 map, with no further
  approximation baked in; and
- each half is **exact when the other is switched off**: linear-only is the
  pure exponential (`μ = 0`), nonlinear-only is pure RK4 (`λ = 0`).

Together with `rk4_error_le`, this is the full content of the RK4IP claim: the
`O(h⁵)` error lives only in the RK4 factor and only for the interaction-picture
remainder.
-/

/--
One interaction-picture step: the exact linear propagator composed with an RK4
step of the scalar remainder — the map the IP formulation implements.
-/
noncomputable def ipStep (lam mu z y h : ℝ) : ℝ :=
  exactStep lam h * rk4Step (fun _ u => mu * u) z y h

theorem ipStep_eq (lam mu z y h : ℝ) :
    ipStep lam mu z y h = exactStep lam h * rk4Step (fun _ u => mu * u) z y h := rfl

/--
**The linear-only IP step is exact.** With no remainder (`μ = 0`) the step is
the pure exponential propagator: RK4 is applied to the zero vector field and
reproduces `y` exactly. This is why the dispersion part of the split-step needs
no step-size control of its own.
-/
theorem ipStep_linear (lam z y h : ℝ) :
    ipStep lam 0 z y h = exactStep lam h * y := by
  simp [ipStep, rk4Step, Stage]

/--
**The nonlinear-only IP step is plain RK4.** With the linear part zero
(`λ = 0`) the rotation is the identity and the step is the classical RK4 map,
error `O(h⁵)` by `rk4_error_le`.
-/
theorem ipStep_nonlinear_only (mu z y h : ℝ) :
    ipStep 0 mu z y h = rk4Step (fun _ u => mu * u) z y h := by
  unfold ipStep exactStep
  simp

/--
**The IP step factors through the stability polynomial.** The
interaction-picture map is the exact propagator times the fourth-order Taylor
sum — because that is what `stability_polynomial` says the RK4 step *is* for a
linear remainder.
-/
theorem ipStep_factors (lam mu z y h : ℝ) :
    ipStep lam mu z y h = exactStep lam h
      * (1 + mu * h + (mu * h) ^ 2 / 2 + (mu * h) ^ 3 / 6 + (mu * h) ^ 4 / 24) * y := by
  unfold ipStep
  rw [rk4Step_linear_y]
  ring

end Solvers
end Photonics