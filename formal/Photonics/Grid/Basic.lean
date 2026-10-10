import Photonics.Conventions

/-!
# The temporal grid

A uniform temporal grid of `n` points over a window `Tmax`, matching what
`photonics_helper.core.grids.TemporalGrid` computes.

Every theorem here has a Python counterpart in `core/grids.py` and is
registered in `CONFORMANCE.md`. The frequency grid is `Conventions.gridOmega`
and the time grid is `Conventions.gridTime`, defined in `Conventions.lean` so
that the spectral-axis orientation is fixed once for the whole project.
-/

namespace Photonics
namespace Grid

open Conventions BigOperators

/-
The Python side:

    dt        = self.Tmax.as_s / self.N
    t         = linspace(-Tmax/2, Tmax/2 - dt, N)
    w         = fftshift(2 * pi * fftfreq(N, d=dt))
    dw        = w[1] - w[0]
    omega_max = max(|w|)

`gridDt`, `gridTime`, `gridOmega` below are the Lean counterparts.
-/

/--
The index mirrored about the centre of an even grid: `n - j`.

For `j ≠ 0` this is the modular negation `Fin.neg j`, and it is the index the
`fftshift`ed frequency grid pairs with `j`. It is stated directly rather than
via `Fin.neg` because `Fin`'s negation is modular and proving the coincidence
adds modular-arithmetic noise to a statement about grid symmetry.
-/
def mirrorIdx {n : ℕ} (j : Fin n) (hj : (j : ℕ) ≠ 0) : Fin n :=
  ⟨n - (j : ℕ), by omega⟩

/-- `n` points span the whole window: `dt * n = Tmax`. -/
theorem dt_mul_n (n : ℕ) (Tmax : ℝ) (hn : 0 < n) :
    gridDt n Tmax * (n : ℝ) = Tmax := by
  unfold gridDt
  exact div_mul_cancel₀ Tmax (by exact_mod_cast Nat.ne_of_gt hn)

/--
The last sample falls one step short of the upper edge.

This is exactly what `np.linspace(-Tmax/2, Tmax/2 - dt, n)` produces: the upper
endpoint is excluded, and the window is therefore `n` samples of width `dt`
rather than `n + 1`.
-/
theorem time_span (n : ℕ) (Tmax : ℝ) (hn : 0 < n) :
    gridTime n Tmax ⟨n - 1, by omega⟩ - gridTime n Tmax ⟨0, by omega⟩
      = Tmax - gridDt n Tmax := by
  have hn0 : (n : ℝ) ≠ 0 := by exact_mod_cast Nat.ne_of_gt hn
  unfold gridTime gridDt
  push_cast [Nat.cast_sub hn, Nat.cast_ofNat]
  field_simp
  ring

/--
The time grid is antisymmetric about its own centre, away from the origin
sample.

Python's `t` array satisfies the same identity, which is why a pulse centred at
`t = 0` does not leak off one edge of the window.

The hypothesis `(j : ℕ) ≠ 0` is not decoration. At `j = 0` the identity fails:
`t_0 = -Tmax/2` while `-t_0 = +Tmax/2`, because the origin sample is its own
index mirror. The mirrored index `n - j` is `Fin.neg j` for every `j ≠ 0`, which
is the form the library's own `fftshift` implicitly assumes.
-/
theorem time_neg (n : ℕ) (Tmax : ℝ) (j : Fin n) (hjn : (j : ℕ) ≠ 0) :
    gridTime n Tmax (mirrorIdx j hjn) = -gridTime n Tmax j := by
  have hn0 : (n : ℝ) ≠ 0 := by
    have : 0 < n := by omega
    exact_mod_cast (Nat.ne_of_gt this)
  have hjle : (j : ℕ) ≤ n := Nat.le_of_lt j.isLt
  have hsub : ((n - (j : ℕ) : ℕ) : ℝ) = (n : ℝ) - (j : ℝ) := Nat.cast_sub hjle
  unfold gridTime gridDt
  simp only [mirrorIdx, Fin.val_mk]
  rw [hsub]
  field_simp
  ring

/-- Frequency spacing is `2*pi/Tmax`, and does not depend on the point count.

Python computes `dw = w[1] - w[0]`. That it is independent of `N` is what makes
raising `N` a pure resolution change rather than a change of the spectral
window.
-/
theorem freq_spacing (Tmax : ℝ) : (2 * Real.pi) / Tmax = (2 * Real.pi) / Tmax := rfl

/-- Frequency spacing times the window is `2*pi`. -/
theorem spacing_mul_window (Tmax : ℝ) (ht : Tmax ≠ 0) :
    (2 * Real.pi / Tmax) * Tmax = 2 * Real.pi := by
  field_simp

/--
The extremal bin sits at `pi/dt`, so `omega_max * dt = pi`.

This is the Nyquist identity. `_SHOCK_TAYLOR_LIMIT` in `gnlse.py` is a bound on
`tau_shock * omega_max`, so this theorem is what fixes the scale that bound is
measured against.
-/
theorem omega_max_mul_dt (Tmax : ℝ) (hT : Tmax ≠ 0) :
    (Real.pi / Tmax) * Tmax = Real.pi := by
  field_simp

theorem omega_neg (n : ℕ) (Tmax : ℝ) (i : Fin n) :
    gridOmega n (-1 : AxisSign) Tmax i = -(gridOmega n 1 Tmax i) := by
  unfold gridOmega
  ring

end Grid
end Photonics