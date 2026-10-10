import Mathlib

/-!
# The characteristic matrix of a thin-film stack

The Macleod optical-admittance formalism, as `dbr.py` implements it.

The property proved here is the one the whole formalism rests on:

**every layer matrix has determinant `1`, so every product of them does too.**

It holds for absorbing layers as well as lossless ones, because the underlying
identity is `cos²δ + sin²δ = 1`, and that is a *complex* trigonometric identity,
not a real one. Energy conservation (`R + T = 1`) is a separate and strictly
stronger statement: it additionally needs the media to be lossless. See
`CONFORMANCE.md` F13 for the absorbing counterexample.
-/

namespace Photonics
namespace Optics

open Matrix Complex
open scoped ComplexConjugate

/--
One layer's characteristic matrix in the admittance formalism:

`[[cos δ, −i·sin δ/η], [−i·η·sin δ, cos δ]]`

This is exactly the 2×2 array `_layer_matrix` builds in `dbr.py`, with `δ` its
`delta` and `η` its `eta`.
-/
noncomputable def layerMatrix (d e : ℂ) : Matrix (Fin 2) (Fin 2) ℂ :=
  fun i j => match i, j with
    | 0, 0 => cos d
    | 0, 1 => -(I) * sin d / e
    | 1, 0 => -(I) * e * sin d
    | _, _ => cos d

/--
**Every layer matrix has determinant one.**

The off-diagonal product is `(−i·sin δ/e)·(−i·e·sin δ) = sin²δ`, because the two
minus signs and `i²` cancel, so the determinant is `cos²δ + sin²δ = 1`.

No hypothesis on `d` at all, and on `e` only nonvanishing. In particular
**nothing here assumes the layer is lossless**: the identity is complex, so it
survives `Im n ≠ 0`.

Verified numerically against `dbr.py` for a lossless stack (`n = 2.30`,
`d = λ/4n`): layer and stack determinants are `1.000000000000` at 400, 550, 800
and 1550 nm, for both TE and TM. For an absorbing layer (`k = 0.02`) the
determinant is *still* `1.000000000000` — which is the point.
-/
theorem layerMatrix_det (d e : ℂ) (he : e ≠ 0) : (layerMatrix d e).det = 1 := by
  rw [Matrix.det_fin_two]
  simp only [layerMatrix, Fin.isValue]
  field_simp
  rw [show I ^ 2 = -1 by norm_num]
  calc cos d ^ 2 - (-1) * sin d ^ 2 = cos d ^ 2 + sin d ^ 2 := by ring
    _ = 1 := Complex.cos_sq_add_sin_sq d

/--
The stack matrix, as `transfer_matrix` builds it: a **left fold** of
`M₁ @ M₂ @ … @ Mₙ` starting from the identity.

Stated with `foldl` rather than a `∏` because `Finset.prod` and `List.prod`
both require a `CommMonoid` at this Mathlib revision, and matrix multiplication
is not commutative. The fold is also the more faithful statement: it is exactly
what the Python does.
-/
noncomputable def stackMatrix (l : List (Matrix (Fin 2) (Fin 2) ℂ)) :
    Matrix (Fin 2) (Fin 2) ℂ :=
  l.foldl (fun acc M => acc * M) 1

private theorem stack_det_aux (l : List (Matrix (Fin 2) (Fin 2) ℂ))
    (acc : Matrix (Fin 2) (Fin 2) ℂ) (hacc : acc.det = 1)
    (h : ∀ M ∈ l, M.det = 1) : (l.foldl (fun acc M => acc * M) acc).det = 1 := by
  induction l generalizing acc with
  | nil => exact hacc
  | cons a l ih =>
      have hstep : ((acc:Matrix (Fin 2) (Fin 2) ℂ) * a).det = 1 := by
        rw [Matrix.det_mul, hacc, h a (by simp)]
        ring
      rw [List.foldl_cons, ih (acc := (acc:Matrix (Fin 2) (Fin 2) ℂ) * a) hstep
        (fun M hM => h M (by simp [hM]))]

/--
**A stack inherits determinant `1` from its layers.**

`transfer_matrix` returns `M₁ @ M₂ @ … @ Mₙ`, so this says the whole stack
matrix is unimodular whatever the layer count.

Verified numerically against `dbr.py`: a lossless 8-layer TiO₂/SiO₂ stack at
1550 nm gives `det = 1.000000000000` for both TE and TM, and so does an
absorbing stack.
-/
theorem stack_det (l : List (Matrix (Fin 2) (Fin 2) ℂ)) (h : ∀ M ∈ l, M.det = 1) :
    (stackMatrix l).det = 1 := by
  exact stack_det_aux l 1 (by simp) h

