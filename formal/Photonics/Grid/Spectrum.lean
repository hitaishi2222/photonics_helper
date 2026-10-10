import Photonics.Grid.Basic

/-!
# The Fourier pair

The machinery behind `TemporalGrid.fft` / `TemporalGrid.ifft`.

`PLAN.md` §5.2 allows a concrete-`n` fallback, but this file takes the general
route, because the Raman `H(Ω)`-realness argument in Stage D needs the general
statement and a concrete instance would not compose with it.

## Status

**Proved:** the ingredients the pairing theorem rests on.

- `geom_sum_zero` — a finite geometric sum of a non-trivial root of unity
  vanishes.
- `rootOfUnity_pow`, `rootOfUnity_ne_one` — the `n`-th root built from an integer
  frequency offset is a genuine `n`-th root of unity, and is trivial only when
  the offset is zero.
- `exponent_closed_form` — the closed form for `ω_k t_j − ω_i t_k`, which is what
  makes the summand a geometric progression at all.
- `exponent_closed_formC`, `kernel_term` — the same closed form in complex form,
  and one entry of the analysis-then-synthesis product written as a single
  exponential. Together these reduce the pairing to a geometric series.

**Still open:** `Conventions.dft_pair`, `conj_dft`, `hermitian`.

The remaining step is mechanical rather than deep, and is written out as a
`dft_pair_entry` statement at the bottom of this file so the next sitting has a
concrete target rather than three opaque names. What is left is: rewrite the
closed form through `Complex.exp_add` and `Complex.exp_nat_mul` to factor the
summand as `C * ζ^k`, apply `Finset.sum_mul` to lift `C` out of the sum, and
finish with `geom_sum_zero`.
-/

namespace Photonics
namespace Grid

open Conventions Matrix Complex
open scoped ComplexConjugate

/-! ## Finite sums over `Fin n`, transported to `Finset.range n`

The root-of-unity argument needs sums indexed by natural numbers, while the
matrices are indexed by `Fin n`.
-/

theorem finSum_to_range {n : ℕ} (g : ℕ → ℂ) :
    ∑ m ∈ Finset.range n, g m = ∑ k : Fin n, g (k : ℕ) := by
  classical
  symm
  exact Finset.sum_bij (fun k (_ : k ∈ (Finset.univ : Finset (Fin n))) => (k : ℕ))
    (fun k _ => Finset.mem_range.mpr (show k.val < n from k.isLt))
    (fun a _ b _ h => Fin.ext h)
    (fun m hm => ⟨⟨m, Finset.mem_range.mp hm⟩, Finset.mem_univ _, rfl⟩)
    (fun k _ => rfl)

/-! ## The geometric-sum lemma -/

private lemma geom_sum_div (ζ : ℂ) (N : ℕ) (hζ : ζ ≠ 1) :
    ∑ k ∈ Finset.range N, ζ ^ k = (ζ ^ N - 1) / (ζ - 1) := by
  induction N with
  | zero => simp
  | succ m ih =>
      rw [Finset.sum_range_succ]
      rw [ih]
      field_simp
      ring

/--
A finite geometric sum of a non-trivial root of unity vanishes.

This is the whole content of Fourier orthogonality. Mathlib has no ready-made
lemma for it at this revision, so it is proved here by induction.
-/
theorem geom_sum_zero (ζ : ℂ) (N : ℕ) (hN : ζ ^ N = 1) (hζ : ζ ≠ 1) :
    ∑ k ∈ Finset.range N, ζ ^ k = 0 := by
  rw [geom_sum_div ζ N hζ, hN]
  simp

/-! ## Roots of unity -/

/--
The `N`-th root associated with an integer frequency offset `m`.

`rootOfUnity m N` is what the DFT kernel accumulates along `k` when the output
and input bins differ by `m`.
-/
noncomputable def rootOfUnity (m : ℤ) (N : ℕ) : ℂ :=
  Complex.exp (Complex.I * (-2 * Real.pi * (m : ℝ) / (N : ℝ)))

/-- It is an `N`-th root of unity. -/
theorem rootOfUnity_pow (m : ℤ) (N : ℕ) (hN : 0 < N) : rootOfUnity m N ^ N = 1 := by
  have hN0 : (N : ℝ) ≠ 0 := by exact_mod_cast (Nat.ne_of_gt hN)
  rw [rootOfUnity, ← Complex.exp_nat_mul]
  have hexp : (N : ℂ) * (Complex.I * (-2 * Real.pi * (m : ℝ) / (N : ℝ)))
      = (-m : ℂ) * (2 * (Real.pi : ℂ) * Complex.I) := by
    have hc : ((N : ℕ) : ℂ) ≠ 0 := by exact_mod_cast (Nat.ne_of_gt hN)
    push_cast
    field_simp [hc]
  rw [hexp, Complex.exp_eq_one_iff]
  refine ⟨-m, ?_⟩
  push_cast
  ring

/--
It is not trivial unless `m` is zero.

Proof strategy: `Complex.exp_eq_one_iff` forces `m/N` to be an integer, and an
integer strictly smaller than `N` in magnitude cannot be a nonzero multiple of
`N`. Both halves are checked explicitly; the hypothesis `m.natAbs < N` is what
the caller gets from `i ≠ j` with `i, j < n`.
-/
theorem rootOfUnity_ne_one {m : ℤ} {N : ℕ} (hN : 0 < N) (hm : m ≠ 0)
    (habs : m.natAbs < N) :
    rootOfUnity m N ≠ 1 := by
  intro h
  rw [rootOfUnity] at h
  obtain ⟨k, hk⟩ := Complex.exp_eq_one_iff.mp h
  -- only the *imaginary* parts carry information: both real parts are zero,
  -- since both sides are imaginary
  obtain ⟨-, him⟩ := Complex.ext_iff.mp hk
  simp at him
  have him' := him
  have hN0 : (N : ℝ) ≠ 0 := by exact_mod_cast (Nat.ne_of_gt hN)
  field_simp [hN0] at him'
  have him'' : (-(m : ℤ) = (N : ℤ) * (k : ℤ)) := by exact_mod_cast him'
  have hmk : (-(m : ℤ) = (k : ℤ) * (N : ℤ)) := by
    calc -(m : ℤ) = (N : ℤ) * (k : ℤ) := him''
      _ = (k : ℤ) * (N : ℤ) := Int.mul_comm _ _
  have hab : m.natAbs = k.natAbs * N := by
    calc m.natAbs = (-(m : ℤ)).natAbs := by rw [Int.natAbs_neg]
      _ = ((k : ℤ) * (N : ℤ)).natAbs := by rw [hmk]
      _ = k.natAbs * (N : ℤ).natAbs := Int.natAbs_mul _ _
      _ = k.natAbs * N := by simp
  have hab' : k.natAbs * N < N := by
    calc k.natAbs * N = m.natAbs := hab.symm
      _ < N := habs
  have hkn : k.natAbs = 0 := by
    have h1 : N * k.natAbs < N * 1 := by simpa [Nat.mul_comm] using hab'
    have h1' : k.natAbs < 1 := (Nat.mul_lt_mul_left (Nat.succ_le_of_lt hN)).mp h1
    exact Nat.lt_one_iff.mp h1'
  have hk0 : k = 0 := by
    rw [Int.natAbs_eq_zero] at hkn
    omega
  subst hk0
  have hm0 : m = 0 := by omega
  exact absurd hm0 hm

/-! ## The kernel exponent

The identity that turns the DFT product into a geometric series.
-/

/--
`ω_k t_j − ω_i t_k = s·π·(i − j)·(1 − 2k/N)`.

Verified numerically before being proved, which is why the shape of the
expression is what it is: the `−Tmax/2` offset on the time grid and the
`i − N/2` offset on the frequency grid combine to give this product, and the
offset terms cancel exactly.
-/
theorem exponent_closed_form (n : ℕ) (Tmax : ℝ) (s : AxisSign) (hn : 0 < n)
    (hT : Tmax ≠ 0) (i j k : Fin n) :
    gridOmega n s Tmax k * gridTime n Tmax j - gridOmega n s Tmax i * gridTime n Tmax k
      = (s : ℝ) * Real.pi * ((i : ℤ) - (j : ℤ)) * (1 - 2 * ((k : ℕ) : ℝ) / (n : ℝ)) := by
  have hn0 : (n : ℝ) ≠ 0 := by exact_mod_cast (Nat.ne_of_gt hn)
  unfold gridOmega gridTime gridDt
  push_cast
  field_simp
  ring


/--
The complex form of `exponent_closed_form`, as a difference of complex numbers.

Proved by `congrArg` from the real statement and then `convert ... using 1`.
The reason a separate complex corollary exists is that Lean distributes the
`ℝ → ℂ` coercions differently in a `ℂ` goal than in the real one, so the real
statement will not `rw` into the complex goal directly.
-/
theorem exponent_closed_formC (n : ℕ) (Tmax : ℝ) (s : AxisSign) (hn : 0 < n)
    (hT : Tmax ≠ 0) (i j k : Fin n) :
    (Complex.I * (gridOmega n s Tmax k * gridTime n Tmax j)
      - Complex.I * (gridOmega n s Tmax i * gridTime n Tmax k))
      = Complex.I * ((s : ℝ) * Real.pi * ((i : ℤ) - (j : ℤ))
          * (1 - 2 * ((k : ℕ) : ℝ) / (n : ℝ))) := by
  have h := congrArg (fun x : ℝ => Complex.I * (x : ℂ))
    (exponent_closed_form n Tmax s hn hT i j k)
  push_cast at h
  convert h using 1 <;> ring

/--
A single entry of the analysis-then-synthesis product.

Writing the two exponentials as one exponential of their exponent difference is
what lets `exponent_closed_form` apply; this is where the `dt` scaling of
`TemporalGrid.fft`/`ifft` drops out (it is carried by the pair's `1/n`).
-/
theorem kernel_term (n : ℕ) (Tmax : ℝ) (s : AxisSign) (hn : 0 < n) (hT : Tmax ≠ 0)
    (i j k : Fin n) :
    idftMatrix n s Tmax i k * dftMatrix n s Tmax k j
      = Complex.exp (Complex.I * ((s : ℝ) * Real.pi * ((i : ℤ) - (j : ℤ))
          * (1 - 2 * ((k : ℕ) : ℝ) / (n : ℝ)))) := by
  simp only [idftMatrix, dftMatrix]
  rw [← Complex.exp_add]
  have hre : -(Complex.I * ((gridOmega n s Tmax i * gridTime n Tmax k) : ℂ))
      + Complex.I * ((gridOmega n s Tmax k * gridTime n Tmax j) : ℂ)
      = (Complex.I * (gridOmega n s Tmax k * gridTime n Tmax j)
        - Complex.I * (gridOmega n s Tmax i * gridTime n Tmax k)) := by
    push_cast
    ring
  rw [hre, exponent_closed_formC n Tmax s hn hT i j k]


/-! ## Reindexing under negation

The machinery needed for conjugate symmetry (`conj_dft`) and hence for the
Raman `H(Ω)`-realness argument.

`Fin n` carries a cyclic group structure, so negation is a genuine permutation
of the bins, and Mathlib already packages it as `Equiv.neg`. That single fact
gives the reindexing of a sum for free.
-/

/-- **Sums are invariant under negating the bin index.** -/
theorem sum_neg_eq {n : ℕ} (f : Fin n → ℂ) :
    ∑ j : Fin n, f (-j : Fin n) = ∑ j : Fin n, f j :=
  Equiv.sum_comp (Equiv.neg (Fin n)) f

/-- Modular negation is ordinary reflection, away from the origin bin. -/
theorem neg_eq_sub {n : ℕ} (j : Fin n) (hj : (j : ℕ) ≠ 0) :
    (-j : Fin n) = ⟨n - (j : ℕ), by omega⟩ := by
  have hjlt : (j : ℕ) < n := j.isLt
  have hj0 : 0 < (j : ℕ) := Nat.pos_of_ne_zero hj
  have hnpos : 0 < n := by omega
  rw [show (-j : Fin n) = ⟨(n - (j : ℕ)) % n, Nat.mod_lt _ hnpos⟩ from Fin.neg_def j]
  congr 1
  exact Nat.mod_eq_of_lt (show n - (j : ℕ) < n by omega)

/--
**The frequency grid is antisymmetric under reflection**, for even `n` and away
from the origin bin.

`ω_{n-i} = -ω_i`. This is the ω-level statement that `conj_dft` needs; the
origin bin `i = 0` is excluded because it sits at `-πn/T` and has no partner
inside the grid. The exclusion costs nothing: that bin is handled separately.
-/
theorem omega_reflect (n : ℕ) (Tmax : ℝ) (hn : Even n) (hT : Tmax ≠ 0) (i : Fin n)
    (hi : (i : ℕ) ≠ 0) :
    gridOmega n (1 : AxisSign) Tmax (mirrorIdx i hi) = -(gridOmega n 1 Tmax i) := by
  have hhalf : (2 : ℝ) * ((n / 2 : ℕ) : ℝ) = (n : ℝ) := by
    have key : 2 * (n / 2) = n := by
      obtain ⟨m, hm⟩ := hn
      have hm2 : n / 2 = m := by omega
      rw [hm2, Nat.two_mul, hm]
    exact_mod_cast key
  unfold gridOmega
  simp only [mirrorIdx, Fin.val_mk]
  push_cast [Nat.cast_sub (Nat.le_of_lt i.isLt), Nat.cast_ofNat]
  have hT0 : (Tmax : ℝ) ≠ 0 := hT
  field_simp
  linarith [hhalf]

/-! ## The orientation-aware root of unity -/

/--
The `N`-th root the DFT kernel actually accumulates along `k`.

`rootOfUnity` above carries no orientation factor, which is fine for `s = 1` but
**wrong for `s = −1`**: reversing the spectral axis reverses the sign of the
`k`-dependent phase. That was a false statement on my part until it was checked
numerically; see `CHANGELOG.md` finding 4.
-/
noncomputable def rootOfUnityS (m : ℤ) (s : AxisSign) (N : ℕ) : ℂ :=
  Complex.exp (Complex.I * (-2 * Real.pi * (m : ℝ) * s / (N : ℝ)))

/--
**The summand factors as a constant times a geometric progression.**

This is the step that turns the DFT pairing into a geometric series. It is
proved by first rewriting the summand with `kernel_term` (which is what avoids
the ℝ/ℂ coercion problem that blocked it earlier), then splitting the exponent
with `Complex.exp_add` and `Complex.exp_nat_mul`. The remaining obligation is
purely arithmetic in `ℂ` and closes with `push_cast; ring`.

With this in hand the entrywise theorem needs only:
`rootOfUnityS_pow`, `rootOfUnityS_ne_one`, `Finset.sum_mul`, `geom_sum_zero`.
-/
theorem kernel_factors (n : ℕ) (Tmax : ℝ) (s : AxisSign) (hn : 0 < n) (hT : Tmax ≠ 0)
    (i j k : Fin n) :
    idftMatrix n s Tmax i k * dftMatrix n s Tmax k j
      = Complex.exp (Complex.I * ((s : ℝ) * Real.pi * (((i : ℤ) : ℝ) - ((j : ℤ) : ℝ))))
        * rootOfUnityS ((i : ℤ) - (j : ℤ)) s n ^ (k : ℕ) := by
  rw [kernel_term n Tmax s hn hT i j k, rootOfUnityS, ← Complex.exp_nat_mul,
    ← Complex.exp_add]
  congr 1
  push_cast
  ring


/--
`|i − j| < n` for distinct bins inside an `n`-point grid.

The last arithmetic input to the off-diagonal half of the pairing. The obvious
lemma, `Int.natAbs_sub_le`, is far too weak (`≤ |i| + |j|`). The working route
normalises the sign twice with `Int.natAbs_neg`, so that whichever ordering holds
the difference is finally the one with a non-negative `ℤ` value, then lifts the
bound from `ℕ` to `ℤ` by `exact_mod_cast`.
-/
theorem fin_diff_natAbs_lt {n : ℕ} (i j : Fin n) (hij : i ≠ j) :
    ((i : ℤ) - (j : ℤ)).natAbs < n := by
  have h1 := i.isLt
  have h2 := j.isLt
  have hne : i.val ≠ j.val := by
    intro hc
    apply hij
    exact Fin.ext (Nat.cast_injective hc)
  rcases Nat.lt_or_gt_of_ne hne with hlt | hgt
  · rw [show ((i : ℤ) - (j : ℤ)) = -((j : ℤ) - (i : ℤ)) by ring, Int.natAbs_neg]
    have hnn : 0 ≤ (j : ℤ) - (i : ℤ) := by omega
    have hk : (((j : ℤ) - (i : ℤ)).natAbs : ℤ) = (j : ℤ) - (i : ℤ) :=
      Int.natAbs_of_nonneg hnn
    have hz : (((j : ℤ) - (i : ℤ)).natAbs : ℤ) < (n : ℤ) := by
      rw [hk]
      push_cast
      omega
    exact_mod_cast hz
  · rw [show ((i : ℤ) - (j : ℤ)) = -((j : ℤ) - (i : ℤ)) by ring, Int.natAbs_neg,
        show ((j : ℤ) - (i : ℤ)) = -((i : ℤ) - (j : ℤ)) by ring, Int.natAbs_neg]
    have hnn : 0 ≤ (i : ℤ) - (j : ℤ) := by omega
    have hk : (((i : ℤ) - (j : ℤ)).natAbs : ℤ) = (i : ℤ) - (j : ℤ) :=
      Int.natAbs_of_nonneg hnn
    have hz : (((i : ℤ) - (j : ℤ)).natAbs : ℤ) < (n : ℤ) := by
      rw [hk]
      push_cast
      omega
    exact_mod_cast hz

/-! ## The two orientations, proved separately

`rcases (validAxis s) with rfl | rfl` substitutes a bare `1` into
`rootOfUnityS`, and the substitution lands as a *natural* cast inside the
exponent. That makes `rw [key, …]` miss and leaves a `ring` goal containing
`↑1` that will not close.

Stating each orientation with an explicit `(1 : ℝ)` literal and never
substituting avoids the problem entirely. These are the two cases
`rootOfUnityS_pow` and `rootOfUnityS_ne_one` are assembled from.
-/

theorem rootOfUnityS_pow_pos (m : ℤ) (N : ℕ) (hN : 0 < N) :
    (Complex.exp (Complex.I * ((-2 * Real.pi * (m : ℝ) * (1 : ℝ)) / (N : ℝ)))) ^ N = 1 := by
  rw [← Complex.exp_nat_mul]
  have hc : (N : ℂ) ≠ 0 := by exact_mod_cast (Nat.ne_of_gt hN)
  have key : (N : ℂ) * (Complex.I * ((-2 * Real.pi * (m : ℝ) * (1 : ℝ)) / (N : ℝ)))
      = (-m : ℂ) * (2 * (Real.pi : ℂ) * Complex.I) := by
    push_cast
    field_simp [hc]
  rw [key, Complex.exp_eq_one_iff]
  refine ⟨-m, ?_⟩
  push_cast
  ring

theorem rootOfUnityS_pow_neg (m : ℤ) (N : ℕ) (hN : 0 < N) :
    (Complex.exp (Complex.I * ((-2 * Real.pi * (m : ℝ) * (-1 : ℝ)) / (N : ℝ)))) ^ N = 1 := by
  rw [← Complex.exp_nat_mul]
  have hc : (N : ℂ) ≠ 0 := by exact_mod_cast (Nat.ne_of_gt hN)
  have key : (N : ℂ) * (Complex.I * ((-2 * Real.pi * (m : ℝ) * (-1 : ℝ)) / (N : ℝ)))
      = (m : ℂ) * (2 * (Real.pi : ℂ) * Complex.I) := by
    push_cast
    field_simp [hc]
  rw [key, Complex.exp_eq_one_iff]
  refine ⟨m, ?_⟩
  push_cast
  ring

/-- **The orientation-aware root is an `N`-th root of unity**, for either
admissible orientation. -/
theorem rootOfUnityS_pow (m : ℤ) (s : AxisSign) (hs : validAxis s) (N : ℕ)
    (hN : 0 < N) : rootOfUnityS m s N ^ N = 1 := by
  rcases hs with rfl | rfl
  · simpa [rootOfUnityS] using rootOfUnityS_pow_pos m N hN
  · simpa [rootOfUnityS] using rootOfUnityS_pow_neg m N hN



/-! ## Non-triviality of the orientation-aware root

The last input the pairing needs: `ζ ≠ 1` when the bin offset is non-zero.

**The route that works.** `Complex.exp_eq_one_iff` gives an integer `k`; take
imaginary parts; `field_simp` at that real identity already clears the
denominator and yields `-(m : ℝ) = (N : ℝ)·(k : ℝ)`; `exact_mod_cast` lifts it
back to `ℤ`; and `fin_diff_natAbs_lt` closes it off.

The key is to stay on the **hand-substituted** statement, where the real
scalars are written as `(1 : ℝ)` rather than reached by `rcases … with rfl` on
`validAxis`. Routing the same argument through `rootOfUnityS` is what produced
the `↑1` artefact recorded in `CONFORMANCE.md` F20.
-/

theorem rootOfUnityS_ne_pos {m : ℤ} {N : ℕ} (hN : 0 < N) (hm : m ≠ 0)
    (habs : m.natAbs < N) :
    Complex.exp (Complex.I * ((-2 * Real.pi * (m : ℝ) * (1 : ℝ)) / (N : ℝ))) ≠ 1 := by
  intro h1
  obtain ⟨k, hk⟩ := Complex.exp_eq_one_iff.mp h1
  obtain ⟨-, him⟩ := Complex.ext_iff.mp hk
  simp at him
  have hN0 : (N : ℝ) ≠ 0 := by exact_mod_cast (Nat.ne_of_gt hN)
  field_simp at him
  have hmk : m = -(k * (N : ℤ)) := by
    have hreal : (m : ℝ) = -((k : ℝ) * (N : ℝ)) := by linarith
    exact_mod_cast hreal
  have hab : m.natAbs = k.natAbs * N := by
    calc m.natAbs = (-(k * (N : ℤ))).natAbs := by rw [hmk]
      _ = (k * (N : ℤ)).natAbs := by rw [Int.natAbs_neg]
      _ = k.natAbs * (N : ℤ).natAbs := Int.natAbs_mul _ _
      _ = k.natAbs * N := by simp
  have hkn : k.natAbs = 0 := by
    rw [hab] at habs
    by_contra hne
    have hne' : 0 < k.natAbs := Nat.pos_of_ne_zero hne
    have h1k : (1 : ℕ) ≤ k.natAbs := Nat.succ_le_iff.mpr hne'
    have h2k : (1 : ℕ) * N ≤ k.natAbs * N := Nat.mul_le_mul_right N h1k
    omega
  have hk0 : k = 0 := by rw [Int.natAbs_eq_zero] at hkn; omega
  have hm0 : m = 0 := by rw [hmk, hk0]; simp
  exact hm hm0

theorem rootOfUnityS_ne_neg {m : ℤ} {N : ℕ} (hN : 0 < N) (hm : m ≠ 0)
    (habs : m.natAbs < N) :
    Complex.exp (Complex.I * ((-2 * Real.pi * (m : ℝ) * (-1 : ℝ)) / (N : ℝ))) ≠ 1 := by
  intro h1
  obtain ⟨k, hk⟩ := Complex.exp_eq_one_iff.mp h1
  obtain ⟨-, him⟩ := Complex.ext_iff.mp hk
  simp at him
  have hN0 : (N : ℝ) ≠ 0 := by exact_mod_cast (Nat.ne_of_gt hN)
  field_simp at him
  have hmk : m = k * (N : ℤ) := by
    have hreal : (m : ℝ) = (k : ℝ) * (N : ℝ) := by linarith
    exact_mod_cast hreal
  have hab : m.natAbs = k.natAbs * N := by
    calc m.natAbs = (k * (N : ℤ)).natAbs := by rw [hmk]
      _ = k.natAbs * (N : ℤ).natAbs := Int.natAbs_mul _ _
      _ = k.natAbs * N := by simp
  have hkn : k.natAbs = 0 := by
    rw [hab] at habs
    by_contra hne
    have hne' : 0 < k.natAbs := Nat.pos_of_ne_zero hne
    have h1k : (1 : ℕ) ≤ k.natAbs := Nat.succ_le_iff.mpr hne'
    have h2k : (1 : ℕ) * N ≤ k.natAbs * N := Nat.mul_le_mul_right N h1k
    omega
  have hk0 : k = 0 := by rw [Int.natAbs_eq_zero] at hkn; omega
  have hm0 : m = 0 := by rw [hmk, hk0]; simp
  exact hm hm0

/-- **The root is trivial only when the bin offset is zero.** -/
theorem rootOfUnityS_ne_one {m : ℤ} {N : ℕ} {s : AxisSign} (hs : validAxis s)
    (hN : 0 < N) (hm : m ≠ 0) (habs : m.natAbs < N) : rootOfUnityS m s N ≠ 1 := by
  rcases hs with rfl | rfl
  · simpa [rootOfUnityS] using rootOfUnityS_ne_pos hN hm habs
  · simpa [rootOfUnityS] using rootOfUnityS_ne_neg hN hm habs

/-! ## The pairing -/

/-- **The entrywise Fourier pairing.** Analysis then synthesis is `n` on the
diagonal and `0` everywhere else.

This is `dft_pair` in entrywise form: it is what makes the `dt` and `1/dt`
scalings in `TemporalGrid.fft` / `ifft` a matched pair rather than a convention
to be remembered.
-/
theorem dft_pair_entry (n : ℕ) (Tmax : ℝ) (s : AxisSign) (hs : validAxis s)
    (hn : 0 < n) (hT : Tmax ≠ 0) (i j : Fin n) :
    (idftMatrix n s Tmax * dftMatrix n s Tmax) i j = if i = j then (n : ℂ) else 0 := by
  classical
  by_cases hij : i = j
  · subst hij
    simp only [if_pos rfl]
    rw [Matrix.mul_apply]
    have hk : ∀ k : Fin n,
        idftMatrix n s Tmax i k * dftMatrix n s Tmax k i = 1 := by
      intro k
      rw [kernel_term n Tmax s hn hT i i k]
      congr 1
      push_cast
      ring
      simp [Complex.exp_zero]
    rw [Finset.sum_congr rfl (fun k _ => hk k)]
    simp
  · simp only [if_neg hij]
    rw [Matrix.mul_apply]
    rw [Finset.sum_congr rfl (fun k _ => kernel_factors n Tmax s hn hT i j k)]
    rw [← Finset.mul_sum]
    refine mul_eq_zero.mpr (Or.inr ?_)
    rw [← finSum_to_range]
    exact geom_sum_zero (rootOfUnityS ((i : ℤ) - (j : ℤ)) s n) n
      (rootOfUnityS_pow ((i : ℤ) - (j : ℤ)) s hs n hn)
      (rootOfUnityS_ne_one hs hn (by omega) (fin_diff_natAbs_lt i j hij))

/--
**The analysis-then-synthesis pair is `n` times the identity.**

`idft · dft = n · 1`. This is what makes the `dt` and `1/dt` scalings in
`TemporalGrid.fft` and `ifft` a matched pair rather than a convention to be
remembered, and it holds for both admissible spectral-axis orientations.
-/
theorem dft_pair (n : ℕ) (Tmax : ℝ) (s : AxisSign) (hs : validAxis s) (hn : 0 < n)
    (hT : Tmax ≠ 0) :
    idftMatrix n s Tmax * dftMatrix n s Tmax
      = (n : ℂ) • (1 : Matrix (Fin n) (Fin n) ℂ) := by
  classical
  ext i j
  rw [dft_pair_entry n Tmax s hs hn hT i j]
  by_cases hij : i = j
  · subst hij
    simp
  · simp [hij]

end Grid
end Photonics