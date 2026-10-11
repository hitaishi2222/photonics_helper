/-
Copyright (c) 2026 Hitaishi V. All rights reserved.
Released under MIT license. See LICENSE for details.

# Stage S3 — ETDRK4: the diagonal φ-identities Lean can check

The Python side (`photonics_helper.gnlse`, `scheme="etdrk4"`) advances the
whole semilinear GNLSE `∂_z A(Ω) = L(Ω)·A(Ω) + N(A)`, with `L` diagonal in
the Fourier basis, by the fourth-order exponential Runge–Kutta method of
Cox–Matthews (J. Comput. Phys. 176, 430 (2002); Eqs. (26)–(29)) /
Kassam–Trefethen (SIAM J. Sci. Comput. 26, 1214 (2005), diagonal code §4).
PLAN §17.5 draws the line: what is checkable here is the *scalar algebra per
bin* — the φ-function identities behind the "exact linear propagator" claim —
not the contour quadrature or any error magnitude (both floating-point, §11).

The identities:
* the stage-1 coefficient `Q = φ₁(x/2)`, in the sense `x • Q x = exp(x/2) − 1`
  (`x ≠ 0`) — the half-length stage carry is the exact linear propagator
  `E₂ = exp(x/2)`;
* the final-combination coefficient of `N₀`, written by Kassam–Trefethen as
  `f₁ = h·(−4 − x + eˣ(4 − 3x + x²))/x³`, equals the φ-combination
  `φ₁(x) − 3φ₂(x) + 4φ₃(x)` — and its value at `x → 0` is the classical RK4
  weight `1/6` + `x/6` (Taylor, `ring_nf`-exact), so the scheme reduces to
  classical RK4 on the weak-stiffness side.

What is deliberately *not* here: the magnitude of the truncation error, the
empirical order, stability limits on `dz`, and the Kassam–Trefethen contour
mean (floating-point; §17.4(b): "not provable here"). The Python
implementation evaluates the coefficients by series near zero and by the
defining formulas away from it — Cox & Matthews' own recommendation —
verified equivalent to a full-circle contour mean in
`tests/test_gnlse_etdrk4.py`.
-/

import Mathlib
import Photonics.Solvers.Composition

open Complex

/-!
## φ-functions

One definition per argument — the whole point of the diagonal case: the
"operator functions" degrade to scalar functions of `x = L·dz`.
-/

namespace Photonics.Solvers.ETDRK4

/-- φ₁ of the half argument without the `h`: the exact half-stage carry
`(exp(x/2) − 1)/x`, so that `Q = h·φ₁(hL/2)` once `x = hL`.
Meromorphic with a removable singularity at `0` (value `1/2`). -/
noncomputable def Qcoef (x : ℂ) : ℂ := (exp (x / 2) - 1) / x

/-- The final-combination coefficient of `N₀` as Kassam–Trefethen write it
(`f₁`, `h`-scaled): `(−4 − x + eˣ(4 − 3x + x²))/x³`. Meromorphic with a
removable singularity at `0` (value `1/6`). -/
noncomputable def f1coef (x : ℂ) : ℂ :=
  (-4 - x + exp x * (4 - 3 * x + x ^ 2)) / x ^ 3

/-- The final-combination coefficient of `(N_a + N_b)` as written
(`f₂`, `h`-scaled). -/
noncomputable def f2coef (x : ℂ) : ℂ :=
  (2 + x + exp x * (-2 + x)) / x ^ 3

/-- The final-combination coefficient of `N_c` as written (`f₃`). -/
noncomputable def f3coef (x : ℂ) : ℂ :=
  (-4 - 3 * x - x ^ 2 + exp x * (4 - x)) / x ^ 3

/-- **The stage-1 carry is the exact half-step propagator**: `Q` is the
φ₁-function of the half argument, i.e. its inverse-per-step form is
`x • Q x = exp(x/2) − 1`. This is the scalar-per-bin content of "the linear
operator is carried exactly by the φ-function weights". -/
theorem Qcoef_mul_id {x : ℂ} (hx : x ≠ 0) :
    x * Qcoef x = exp (x / 2) - 1 := by
  rw [Qcoef]
  field_simp

/-- **f₁ is the φ-combination `φ₁ − 3φ₂ + 4φ₃`** (exact, for `x ≠ 0`);
its value at `x → 0` is `1/6 + x/6 + 3x²/40 + …`, the classical RK4 weight
`1/6` plus the exponential corrections — scheme reduces to classical RK4 on
the non-stiff side. -/
theorem f1coef_phi_combo {x : ℂ} (hx : x ≠ 0) :
    f1coef x = (exp x - 1) / x - 3 * ((exp x - 1 - x) / x ^ 2)
        + 4 * ((exp x - 1 - x - x ^ 2 / 2) / x ^ 3) := by
  rw [f1coef]
  field_simp
  ring

end Photonics.Solvers.ETDRK4
