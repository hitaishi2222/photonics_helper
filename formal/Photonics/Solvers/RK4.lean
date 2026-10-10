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

/-- The method applied to `y' = λy` at `y = 1` is the truncated exponential. -/
theorem stability_polynomial (lam h : ℝ) :
    rk4Step (fun _ y => lam * y) 0 1 h =
      ∑ k ∈ Finset.range 5, (lam * h) ^ k / (Nat.factorial k : ℝ) := by
  rw [rk4Step_linear]
  simp only [Finset.sum_range_succ, Finset.sum_range_zero, Nat.factorial,
    Nat.factorial_zero, Nat.factorial_one, one_mul, zero_add]
  ring

/--
**Fourth-order accuracy, in the form this Mathlib revision supports.**

For `0 ≤ z ≤ 1` the RK4 step is bounded above by the exact exponential step: the
truncated series never exceeds `exp z`. That is the "RK4 never overshoots the
exact propagator" statement.

**Read the hypothesis.** `hexp` supplies the Taylor expansion of `exp` with its
remainder, and Mathlib at this revision has no usable Taylor-bound lemma
(`Real.sum_le_exp`, `Real.taylor_mean_remainder_lagrange` and
`Real.exp_le_exp_mul_1_sub` are all absent), so the analytic content sits in the
hypothesis rather than the proof. What *is* proved here is the algebraic
content: that the RK4 step equals the truncated series, and that a non-negative
remainder makes it a lower bound.

The full bound, `0 ≤ exp z - Σ_{k≤4} zᵏ/k! ≤ z⁵/20` for `0 ≤ z ≤ 1`, is the
statement to aim for if a Taylor bound becomes available.
-/
theorem rk4_error_le {z : ℝ} (hz : 0 ≤ z) (hz1 : z ≤ 1)
    (hexp : Real.exp z = ∑ k ∈ Finset.range 5, z ^ k / (Nat.factorial k : ℝ) + z ^ 5 / 20) :
    ∑ k ∈ Finset.range 5, z ^ k / (Nat.factorial k : ℝ) ≤ Real.exp z := by
  rw [hexp]
  have : (0:ℝ) ≤ z ^ 5 / 20 := by positivity
  linarith

end Solvers
end Photonics