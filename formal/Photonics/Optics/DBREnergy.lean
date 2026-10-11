import Mathlib
import Photonics.Optics.DBR

/-!
# Energy conservation through a lossless stack

The second of the two properties of the characteristic-matrix formalism:

**every layer matrix has determinant `1`** — proved in `DBR.lean` for any
medium, absorbing or not; and

**for a lossless stack, `R + T = 1`** — proved here, for lossless media.

The two are not the same statement. The determinant survives absorption
(`CONFORMANCE.md` F13 has the numeric counterexample: an absorbing layer keeps
`det = 1` while `R + T = 0.9292`); the energy identity does not. What the
energy identity needs, beyond `det M = 1`, is exactly the *shape* a lossless
stack has:

```
M = [[a, i·β], [i·γ, δ]]   a, β, γ, δ ∈ ℝ
```

The rest is one line of arithmetic. Writing the reflection numerator and
denominator of `dbr.py`'s `spectrum` as

```
D = (η₀a + η_x δ) + i(η₀η_x β + γ)
N = (η₀a − η_x δ) + i(η₀η_x β − γ)
```

one gets

```
|D|² − |N|² = 4·η₀·η_x·(a·δ + β·γ)
```

and `a·δ + β·γ = det M` because `i² = −1`. With `det M = 1` and real admittances
`η₀, η_x > 0`, the Python quantities `R = |r|²` and `T = (η_x/η₀)·|t|²` sum to
exactly `1`.

**Why this file is separate.** A previous session recorded the claim that the
identity "does not follow from `det M = 1`" and is "a different argument". Both
halves are half-true: the determinant alone is not enough, and the conjugate
form is needed — but the conjugate form is *this* argument, extended by the
lossless shape, and it is short. The finding is written up in `CHANGELOG.md`
(2026-10-25). See `F13` for the absorbing counterexample that separates the two
statements.
-/

namespace Photonics
namespace Optics

open Matrix Complex
open scoped ComplexConjugate

/-! ## The lossless shape -/

/--
**A lossless-shaped characteristic matrix:** the diagonal is real and the
off-diagonal purely imaginary. This is exactly the shape `layerMatrix d e` has
when `d` and `e` are real (`e = η`, the admittance of a lossless medium).
-/
def IsLossless (M : Matrix (Fin 2) (Fin 2) ℂ) : Prop :=
  (∃ a : ℝ, M 0 0 = (a : ℂ)) ∧ (∃ b : ℝ, M 0 1 = (b : ℂ) * I) ∧
  (∃ c : ℝ, M 1 0 = (c : ℂ) * I) ∧ (∃ d : ℝ, M 1 1 = (d : ℂ))

theorem IsLossless_one : IsLossless (1 : Matrix (Fin 2) (Fin 2) ℂ) := by
  refine ⟨⟨1, by simp⟩, ⟨0, by simp⟩, ⟨0, by simp⟩, ⟨1, by simp⟩⟩

/-- A real admittance gives a lossless-shaped layer matrix: `cos δ`, `sin δ` are
real for real `δ`, so the diagonal is real and the off-diagonal is a pure
multiple of `i`. -/
theorem layerMatrix_lossless (d e : ℝ) : IsLossless (layerMatrix (d : ℂ) (e : ℂ)) := by
  refine ⟨⟨Real.cos d, ?_⟩, ⟨-(Real.sin d / e), ?_⟩, ⟨-(e * Real.sin d), ?_⟩,
    ⟨Real.cos d, ?_⟩⟩
  · simp only [layerMatrix, Fin.isValue]
    exact (Complex.ofReal_cos d).symm
  · simp only [layerMatrix, Fin.isValue]
    push_cast
    ring
  · simp only [layerMatrix, Fin.isValue]
    push_cast
    ring
  · simp only [layerMatrix, Fin.isValue]
    exact (Complex.ofReal_cos d).symm

/-- Products of lossless-shaped matrices are lossless-shaped: `real·real` stays
real, and `(i·β)·(i·γ) = −β·γ` stays real. -/
theorem IsLossless_mul (M N : Matrix (Fin 2) (Fin 2) ℂ) (hM : IsLossless M)
    (hN : IsLossless N) : IsLossless (M * N) := by
  obtain ⟨⟨a, ha⟩, ⟨b, hb⟩, ⟨c, hc⟩, ⟨d, hd⟩⟩ := hM
  obtain ⟨⟨x, hx⟩, ⟨y, hy⟩, ⟨z, hz⟩, ⟨w, hw⟩⟩ := hN
  refine ⟨⟨a * x - b * z, ?_⟩, ⟨a * y + b * w, ?_⟩, ⟨c * x + d * z, ?_⟩,
    ⟨d * w - c * y, ?_⟩⟩
  · show (M * N) 0 0 = _
    rw [Matrix.mul_apply]
    simp only [Fin.sum_univ_two, ha, hx, hb, hz]
    push_cast
    ring_nf
    rw [show (I : ℂ) ^ 2 = -1 by norm_num]
    ring
  · show (M * N) 0 1 = _
    rw [Matrix.mul_apply]
    simp only [Fin.sum_univ_two, ha, hy, hb, hw]
    push_cast
    ring
  · show (M * N) 1 0 = _
    rw [Matrix.mul_apply]
    simp only [Fin.sum_univ_two, hc, hx, hd, hz]
    push_cast
    ring
  · show (M * N) 1 1 = _
    rw [Matrix.mul_apply]
    simp only [Fin.sum_univ_two, hc, hy, hd, hw]
    push_cast
    ring_nf
    rw [show (I : ℂ) ^ 2 = -1 by norm_num]
    ring

private theorem stack_lossless_aux (l : List (Matrix (Fin 2) (Fin 2) ℂ))
    (acc : Matrix (Fin 2) (Fin 2) ℂ) (hacc : IsLossless acc)
    (h : ∀ M ∈ l, IsLossless M) :
    IsLossless (l.foldl (fun acc M => acc * M) acc) := by
  induction l generalizing acc with
  | nil => exact hacc
  | cons a l ih =>
      rw [List.foldl_cons]
      exact ih (acc * a) (IsLossless_mul acc a hacc (h a (by simp)))
        (fun M hM => h M (by simp [hM]))

/-- **A stack of lossless media is lossless-shaped.** The fold of
`transfer_matrix` preserves the shape, so the whole stack carries it — which is
what the energy identity below is stated against. -/
theorem stackMatrix_lossless (l : List (Matrix (Fin 2) (Fin 2) ℂ))
    (h : ∀ M ∈ l, IsLossless M) : IsLossless (stackMatrix l) :=
  stack_lossless_aux l 1 IsLossless_one h

/-! ## The amplitude coefficients -/

/-- The amplitude reflection coefficient, exactly as `spectrum` computes it
from the stack matrix and the two media admittances. -/
noncomputable def reflCoeff (M : Matrix (Fin 2) (Fin 2) ℂ) (eta0 etaX : ℝ) : ℂ :=
  ((eta0 : ℂ) * M 0 0 + (eta0 : ℂ) * etaX * M 0 1 - M 1 0 - (etaX : ℂ) * M 1 1) /
    ((eta0 : ℂ) * M 0 0 + (eta0 : ℂ) * etaX * M 0 1 + M 1 0 + (etaX : ℂ) * M 1 1)

/-- The amplitude transmission coefficient: `t = 2·η₀ / denom`. -/
noncomputable def transCoeff (M : Matrix (Fin 2) (Fin 2) ℂ) (eta0 etaX : ℝ) : ℂ :=
  (2 * (eta0 : ℂ)) /
    ((eta0 : ℂ) * M 0 0 + (eta0 : ℂ) * etaX * M 0 1 + M 1 0 + (etaX : ℂ) * M 1 1)

/-- `|u + i·v|² = u² + v²` for real `u`, `v`. -/
theorem normSq_real_plus_imag (u v : ℝ) :
    ((u : ℂ) + (I : ℂ) * (v : ℂ)).normSq = u ^ 2 + v ^ 2 := by
  rw [Complex.normSq_apply]
  simp only [map_mul, Complex.conj_I, Complex.conj_ofReal, mul_neg,
    Complex.add_re, Complex.mul_re, Complex.I_re, Complex.I_im, Complex.ofReal_re,
    Complex.ofReal_im, Complex.add_im, Complex.mul_im]
  ring

/-! ## Energy conservation -/

/--
**Energy conservation.** For a lossless stack — lossless shape and real,
positive media admittances — the reflectance and transmittance of `dbr.py`'s
`spectrum` sum to exactly one:

`R + T = 1`,  `R = |r|²`,  `T = (η_x/η₀)·|t|²`.

The determinant `det M = 1` enters through the lossless shape: writing
`D = (η₀a + η_x δ) + i(η₀η_x β + γ)` and `N = (η₀a − η_x δ) + i(η₀η_x β − γ)`,
`|D|² − |N|² = 4·η₀·η_x·(a·δ + β·γ) = 4·η₀·η_x·det M = 4·η₀·η_x`.

The hypothesis set is the physical regime, not a convenience: lossless media
(real `n`, `α = 0`) give the shape; absorbing media break it and the identity
fails with them (F13).
-/
theorem energyConservation (M : Matrix (Fin 2) (Fin 2) ℂ) (hM : IsLossless M)
    (hdet : M.det = 1) (eta0 etaX : ℝ) (h0 : 0 < eta0) (hX : 0 < etaX) :
    (reflCoeff M eta0 etaX).normSq
      + (etaX / eta0) * (transCoeff M eta0 etaX).normSq = 1 := by
  obtain ⟨⟨a, ha⟩, ⟨b, hb⟩, ⟨c, hc⟩, ⟨d, hd⟩⟩ := hM
  -- From the shape and `det M = 1`: `a·δ + β·γ = 1`.
  have hdetR : a * d + b * c = 1 := by
    have h2 : M.det = ((a * d + b * c : ℝ) : ℂ) := by
      rw [Matrix.det_fin_two, ha, hb, hc, hd]
      push_cast
      ring_nf
      rw [show (I : ℂ) ^ 2 = -1 by norm_num]
      ring
    rw [h2] at hdet
    exact Complex.ofReal_inj.mp hdet
  -- The denominator and numerator of `r`, in real-plus-imaginary form.
  have hD : ((eta0 : ℂ) * M 0 0 + (eta0 : ℂ) * etaX * M 0 1 + M 1 0 + (etaX : ℂ) * M 1 1)
      = ((eta0 * a + etaX * d : ℝ) : ℂ) + (I : ℂ) * ((eta0 * etaX * b + c : ℝ) : ℂ) := by
    rw [ha, hb, hc, hd]
    push_cast
    ring
  have hN : ((eta0 : ℂ) * M 0 0 + (eta0 : ℂ) * etaX * M 0 1 - M 1 0 - (etaX : ℂ) * M 1 1)
      = ((eta0 * a - etaX * d : ℝ) : ℂ) + (I : ℂ) * ((eta0 * etaX * b - c : ℝ) : ℂ) := by
    rw [ha, hb, hc, hd]
    push_cast
    ring
  have hsqD : ((eta0 : ℂ) * M 0 0 + (eta0 : ℂ) * etaX * M 0 1 + M 1 0 + (etaX : ℂ) * M 1 1).normSq
      = (eta0 * a + etaX * d) ^ 2 + (eta0 * etaX * b + c) ^ 2 := by
    rw [hD, normSq_real_plus_imag]
  have hsqN : ((eta0 : ℂ) * M 0 0 + (eta0 : ℂ) * etaX * M 0 1 - M 1 0 - (etaX : ℂ) * M 1 1).normSq
      = (eta0 * a - etaX * d) ^ 2 + (eta0 * etaX * b - c) ^ 2 := by
    rw [hN, normSq_real_plus_imag]
  -- `|D|² − |N|² = 4·η₀·η_x·(a·δ + β·γ)`.
  have hkey : (eta0 * a + etaX * d) ^ 2 + (eta0 * etaX * b + c) ^ 2
      = (eta0 * a - etaX * d) ^ 2 + (eta0 * etaX * b - c) ^ 2
        + 4 * eta0 * etaX * (a * d + b * c) := by ring
  -- The denominator cannot vanish: `D = 0` would force `a·δ + β·γ = 0`.
  have hDne : ((eta0 : ℂ) * M 0 0 + (eta0 : ℂ) * etaX * M 0 1 + M 1 0 + (etaX : ℂ) * M 1 1) ≠ 0 := by
    intro hz
    have hzero : ((eta0 : ℂ) * M 0 0 + (eta0 : ℂ) * etaX * M 0 1 + M 1 0 + (etaX : ℂ) * M 1 1).normSq
        = 0 := by
      rw [hz]
      simp
    rw [hsqD] at hzero
    have hU : eta0 * a + etaX * d = 0 := by nlinarith [hzero]
    have hV : eta0 * etaX * b + c = 0 := by nlinarith [hzero]
    have hUd : eta0 * a * d + etaX * d * d = 0 := by
      have h1 : eta0 * a = -etaX * d := by linarith [hU]
      rw [h1]
      ring
    have hVb' : eta0 * b * c + eta0 * (eta0 * etaX) * b * b = 0 := by
      have h2 : c = -eta0 * etaX * b := by linarith [hV]
      rw [h2]
      ring
    have h4 : eta0 * (a * d + b * c) + etaX * d ^ 2 + eta0 ^ 2 * etaX * b ^ 2 = 0 := by
      nlinarith [hUd, hVb']
    have h5 : (0:ℝ) < eta0 * (a * d + b * c) + etaX * d ^ 2 + eta0 ^ 2 * etaX * b ^ 2 := by
      rw [hdetR]
      have hsq1 : (0:ℝ) ≤ etaX * d ^ 2 := mul_nonneg hX.le (pow_two_nonneg d)
      have hsq2 : (0:ℝ) ≤ eta0 ^ 2 * etaX * b ^ 2 :=
        mul_nonneg (mul_nonneg (pow_two_nonneg eta0) hX.le) (pow_two_nonneg b)
      linarith
    linarith [h4, h5]
  have hsqDpos : (0:ℝ) < (eta0 * a + etaX * d) ^ 2 + (eta0 * etaX * b + c) ^ 2 := by
    have : ((eta0 : ℂ) * M 0 0 + (eta0 : ℂ) * etaX * M 0 1 + M 1 0 + (etaX : ℂ) * M 1 1).normSq
        = (eta0 * a + etaX * d) ^ 2 + (eta0 * etaX * b + c) ^ 2 := hsqD
    have hpos : (0:ℝ) < ((eta0 : ℂ) * M 0 0 + (eta0 : ℂ) * etaX * M 0 1 + M 1 0
        + (etaX : ℂ) * M 1 1).normSq :=
      Complex.normSq_pos.2 hDne
    rw [this] at hpos
    exact hpos
  -- Assemble: `R + T = (|N|² + 4·η₀·η_x) / |D|² = |D|² / |D|² = 1`.
  show (reflCoeff M eta0 etaX).normSq + (etaX / eta0) * (transCoeff M eta0 etaX).normSq = 1
  have h2eta : Complex.normSq (2 * (eta0 : ℂ)) = (2 * eta0) * (2 * eta0) := by
    rw [show (2 * (eta0 : ℂ)) = ((2 * eta0 : ℝ) : ℂ) from by push_cast; ring,
      Complex.normSq_ofReal]
  unfold reflCoeff transCoeff
  rw [Complex.normSq_div, Complex.normSq_div, hsqN, hsqD]
  rw [hkey, hdetR, h2eta]
  field_simp [h0.ne', hX.ne']
  norm_num

end Optics
end Photonics
