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
- `kernel_factors`, `dft_pair_entry`, `dft_pair` — the pairing itself, for either
  admissible spectral-axis orientation.
- `exp_two_pi_int_mul`, `origin_bin_exp` — the origin-bin identity that makes
  conjugate symmetry hold at the DC bin.
- `omega_reflect_s`, `omega_reflect_sum` — the frequency grid is antisymmetric
  under reflection away from the origin bin.
- `conj_dft`, `hermitian` — conjugate symmetry and the Hermitian symmetry of a
  real input's spectrum.

**Revision notes.** This file was written against an earlier Mathlib and three
things had moved; see `CHANGELOG.md` (2026-10-24) for the full account.

- `dotProduct` lives in the **root** namespace, not `Matrix`. `simp only`
  silently skips an unknown lemma, so the old `Matrix.dotProduct` left the
  goal in `⋅`-form and the error surfaced much later, at `Finset.sum_congr`.
- `Complex.exp_conj` is `exp (conj x) = conj (exp x)` here, so rewriting
  `conj(exp z)` to the right form needs `←`.
- A `Fin n` literal (`0`) needs `NeZero n`, and that instance is not available
  while a *statement* is being elaborated. DC-bin indices are written
  `⟨0, by omega⟩` against an explicit `hn0 : 0 < n`.
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

/-! ## Conjugate symmetry

The machinery needed for `conj_dft`: the frequency grid's behavior under
reflection, and the origin-bin identity that handles the self-paired DC bin.
-/

/-- `exp(I·2π·k) = 1` for any integer `k`. -/
theorem exp_two_pi_int_mul (k : ℤ) :
    Complex.exp (Complex.I * (2 * Real.pi * (k : ℝ))) = 1 := by
  rw [Complex.exp_eq_one_iff]
  refine ⟨k, ?_⟩
  push_cast
  ring

/-- **The frequency grid is antisymmetric under reflection**, for either
orientation and away from the origin bin. Generalizes `omega_reflect` (which
fixes `s = 1`) to arbitrary `s`. -/
theorem omega_reflect_s (n : ℕ) (Tmax : ℝ) (s : AxisSign) (hs : validAxis s)
    (hn : Even n) (hT : Tmax ≠ 0) (i : Fin n) (hi : (i : ℕ) ≠ 0) :
    gridOmega n s Tmax (mirrorIdx i hi) = -(gridOmega n s Tmax i) := by
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

/-- **The origin-bin identity.** For even `n`, the double-frequency at bin 0
satisfies `exp(I·(ω_0 + ω_0)·t_j) = 1`. This is what makes conjugate symmetry
hold at the DC bin, where `ω_{-0} = ω_0` and the pairing is trivial.

The arithmetic: `(ω_0 + ω_0)·t_j = 2π·s·(n/2 − j)`, and the whole exponent is
`2π·k` with `k = ±(n/2 − j) ∈ ℤ`, so it is a vanishing `2π` multiple.

The bin index is written `⟨0, by omega⟩` rather than the literal `0`: a
`Fin n` literal needs a `NeZero n` instance, which is not available while the
*statement* is being elaborated. Writing the index out keeps the hypothesis
`hn0 : 0 < n` doing that job explicitly, which is also what callers have.
-/
theorem origin_bin_exp (n : ℕ) (Tmax : ℝ) (s : AxisSign) (hs : validAxis s)
    (hn : Even n) (hn0 : 0 < n) (hT : Tmax ≠ 0) (j : Fin n) :
    Complex.exp (Complex.I * ((gridOmega n s Tmax ⟨0, by omega⟩ + gridOmega n s Tmax ⟨0, by omega⟩)
      * gridTime n Tmax j)) = 1 := by
  haveI : NeZero n := ⟨Nat.ne_of_gt hn0⟩
  obtain ⟨m, hm⟩ := hn
  have hm2 : (2 : ℝ) * (m : ℝ) = (n : ℝ) := by push_cast [hm]; ring
  have hT0 : (Tmax : ℝ) ≠ 0 := hT
  -- The DC bin sits at `ω_0 = −s·2π·(n/2)/Tmax`.
  have hzero : gridOmega n s Tmax ⟨0, by omega⟩ = -(s * (2 * Real.pi) * (m : ℝ)) / Tmax := by
    unfold gridOmega
    simp only [Nat.cast_zero, zero_sub]
    field_simp [hT0]
    rw [← hm2]
    ring
  have htime : gridTime n Tmax j = (Tmax : ℝ) * (-(1:ℝ)/2 + (j : ℝ) / n) := by
    unfold gridTime gridDt
    field_simp [hT0]
  -- The exponent is `I · 2π · s·(m − j)`.
  have hexp : Complex.I * (((↑(gridOmega n s Tmax ⟨0, by omega⟩)) + ↑(gridOmega n s Tmax ⟨0, by omega⟩))
      * ↑(gridTime n Tmax j)) = Complex.I * (2 * Real.pi * (s * ((m : ℝ) - (j : ℝ)))) := by
    rw [hzero, htime]
    push_cast
    field_simp [hT0, hn0.ne']
    push_cast [hm]
    ring
  rw [hexp]
  rcases hs with rfl | rfl
  · rw [Complex.exp_eq_one_iff]
    refine ⟨m - (j : ℕ), ?_⟩
    push_cast
    ring
  · rw [Complex.exp_eq_one_iff]
    refine ⟨-((m : ℤ) - (j : ℕ)), ?_⟩
    push_cast
    ring

/-- **The reflected frequency grid sums to zero** for either orientation and
away from the origin bin: `ω_i + ω_{-i} = 0`. -/
theorem omega_reflect_sum (n : ℕ) (Tmax : ℝ) (s : AxisSign) (hs : validAxis s)
    (hn : Even n) (hT : Tmax ≠ 0) (i : Fin n) (hi : (i : ℕ) ≠ 0) :
    gridOmega n s Tmax i + gridOmega n s Tmax (-i) = 0 := by
  have hneg : gridOmega n s Tmax (-i) = -gridOmega n s Tmax i := by
    have : (-i : Fin n) = mirrorIdx i hi := by
      rw [neg_eq_sub i hi]
      rfl
    rw [this]
    exact omega_reflect_s n Tmax s hs hn hT i hi
  rw [hneg]
  ring

/-- **Conjugation identity for the DFT.**

Conjugating the analysis transform equals analysing the conjugate, with the
sign of the frequency index reversed.

This is the algebraic content of the conjugated-transform-pair fast path
`conj(DFT(conj(·)))`, and it holds because the kernel is a real-exponential.

The proof: the summands match iff `exp(I·(ω_i + ω_{-i})·t_j) = 1`. For
`i ≠ 0`, `ω_i + ω_{-i} = 0` by `omega_reflect_sum`. For `i = 0`, the origin-bin
identity applies. Both cases are exact — no `i ≠ 0` hypothesis is needed.
-/
theorem conj_dft (n : ℕ) (Tmax : ℝ) (s : AxisSign) (hs : validAxis s)
    (hn : Even n) (hT : Tmax ≠ 0) (A : Fin n → ℂ) (i : Fin n) :
    (dftMatrix n s Tmax *ᵥ (fun j => conj (A j))) i
      = conj ((dftMatrix n s Tmax *ᵥ A) (-i)) := by
  -- Expand both sides. In this Mathlib revision `dotProduct` lives in the root
  -- namespace (there is no `Matrix.dotProduct`), so `Matrix.mulVec` unfolds
  -- to a root-`dotProduct`, which `simp only [dotProduct]` then unfolds to
  -- the `Finset.sum` the rest of the proof is written against.
  simp only [dftMatrix, Matrix.mulVec, dotProduct]
  -- LHS: ∑_j exp(I·ω_i·t_j) · conj(A_j)
  -- RHS: conj(∑_j exp(I·ω_{-i}·t_j) · A_j)
  --
  -- Step 1: pointwise identity exp(I·ω_i·t_j) = exp(-I·ω_{-i}·t_j)
  have hpt : ∀ j : Fin n,
      Complex.exp (Complex.I * ((gridOmega n s Tmax i * gridTime n Tmax j : ℝ) : ℂ))
        = Complex.exp (-(Complex.I * ((gridOmega n s Tmax (-i) * gridTime n Tmax j : ℝ) : ℂ))) := by
    intro j
    have hn0 : 0 < n := Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt
    -- exp(I·ω_i·t_j) = exp(-I·ω_{-i})·t_j) iff exp(I·(ω_i + ω_{-i})·t_j) = 1
    have key : Complex.exp (Complex.I * ((gridOmega n s Tmax i + gridOmega n s Tmax (-i))
        * gridTime n Tmax j)) = 1 := by
      by_cases hi0 : (i : ℕ) = 0
      · -- i = 0: ω_{-0} = ω_0, so ω_i + ω_{-i} = 2ω_0
        haveI : NeZero n := ⟨by omega⟩
        have hi0' : i = 0 := Fin.val_eq_zero_iff.mp hi0
        subst hi0'
        simpa using origin_bin_exp n Tmax s hs hn hn0 hT j
      · -- i ≠ 0: ω_{-i} = -ω_i, so ω_i + ω_{-i} = 0
        have hi' : (i : ℕ) ≠ 0 := by omega
        have hneg : gridOmega n s Tmax (-i) = -gridOmega n s Tmax i := by
          have : (-i : Fin n) = mirrorIdx i hi' := by
            rw [neg_eq_sub i hi']
            rfl
          rw [this]
          exact omega_reflect_s n Tmax s hs hn hT i hi'
        have hzero : (gridOmega n s Tmax i + gridOmega n s Tmax (-i)) * gridTime n Tmax j = 0 := by
          rw [hneg]; ring
        -- `key` carries split casts; `hzero` is stated unsplit. Bridge with
        -- the cast lemmas, then finish with `mul_zero` + `Complex.exp_zero`.
        rw [← Complex.ofReal_add, ← Complex.ofReal_mul, hzero]
        show cexp (I * (0 : ℂ)) = 1
        rw [mul_zero, Complex.exp_zero]
    -- Derive exp(I·ω_i·t_j) = exp(-I·ω_{-i}·t_j) from key
    have hsum : Complex.I * ((gridOmega n s Tmax i + gridOmega n s Tmax (-i))
        * gridTime n Tmax j)
        = Complex.I * (gridOmega n s Tmax i * gridTime n Tmax j)
          + Complex.I * (gridOmega n s Tmax (-i) * gridTime n Tmax j) := by
      push_cast; ring
    rw [hsum] at key
    rw [Complex.exp_add] at key
    -- key : exp(I·ω_i·t_j) · exp(I·ω_{-i}·t_j) = 1
    -- Goal: exp(I·ω_i·t_j) = exp(-I·ω_{-i}·t_j)
    have h1 : ∀ (x : ℝ), Complex.exp (Complex.I * (x : ℂ))
        * Complex.exp (-(Complex.I * (x : ℂ))) = 1 := by
      intro x
      rw [← Complex.exp_add, add_neg_cancel, Complex.exp_zero]
    calc Complex.exp (Complex.I * ((gridOmega n s Tmax i * gridTime n Tmax j : ℝ) : ℂ))
        = Complex.exp (Complex.I * ((gridOmega n s Tmax i * gridTime n Tmax j : ℝ) : ℂ)) * 1 := by ring
      _ = Complex.exp (Complex.I * ((gridOmega n s Tmax i * gridTime n Tmax j : ℝ) : ℂ))
          * (Complex.exp (Complex.I * ((gridOmega n s Tmax (-i) * gridTime n Tmax j : ℝ) : ℂ))
            * Complex.exp (-(Complex.I * ((gridOmega n s Tmax (-i) * gridTime n Tmax j : ℝ) : ℂ)))) := by
        rw [h1]
      _ = (Complex.exp (Complex.I * ((gridOmega n s Tmax i * gridTime n Tmax j : ℝ) : ℂ))
            * Complex.exp (Complex.I * ((gridOmega n s Tmax (-i) * gridTime n Tmax j : ℝ) : ℂ)))
          * Complex.exp (-(Complex.I * ((gridOmega n s Tmax (-i) * gridTime n Tmax j : ℝ) : ℂ))) := by ring
      _ = 1 * Complex.exp (-(Complex.I * ((gridOmega n s Tmax (-i) * gridTime n Tmax j : ℝ) : ℂ))) := by
        -- `key` carries split casts (`↑a * ↑b`) while this goal carries unsplit ones
        -- (`↑(a*b)`); `Complex.ofReal_mul` bridges the two forms.
        rw [Complex.ofReal_mul, Complex.ofReal_mul, key]
      _ = Complex.exp (-(Complex.I * ((gridOmega n s Tmax (-i) * gridTime n Tmax j : ℝ) : ℂ))) := by ring
  -- Step 2: the summand identity, stated in the split-cast form that
  -- `simp only [dftMatrix, Matrix.mulVec, dotProduct]` produces (the cast
  -- lemmas push `↑(a·b)` into `↑a * ↑b`), so the two sums match syntactically.
  have hterm : ∀ j : Fin n,
      Complex.exp (Complex.I * ((↑(gridOmega n s Tmax i) * ↑(gridTime n Tmax j)) : ℂ)) * conj (A j)
        = conj (Complex.exp (Complex.I * ((↑(gridOmega n s Tmax (-i)) * ↑(gridTime n Tmax j)) : ℂ)) * A j) := by
    intro j
    -- Bridge the split casts back to `hpt`'s unsplit form, where it applies.
    have hkey : Complex.exp (Complex.I * ((↑(gridOmega n s Tmax i) * ↑(gridTime n Tmax j)) : ℂ))
        = Complex.exp (-(Complex.I * ((↑(gridOmega n s Tmax (-i)) * ↑(gridTime n Tmax j)) : ℂ))) := by
      have h1 : Complex.exp (Complex.I * ((↑(gridOmega n s Tmax i) * ↑(gridTime n Tmax j)) : ℂ))
          = Complex.exp (Complex.I * ((gridOmega n s Tmax i * gridTime n Tmax j : ℝ) : ℂ)) := by
        rw [← Complex.ofReal_mul]
      have h2 : Complex.exp (-(Complex.I * ((↑(gridOmega n s Tmax (-i)) * ↑(gridTime n Tmax j)) : ℂ)))
          = Complex.exp (-(Complex.I * ((gridOmega n s Tmax (-i) * gridTime n Tmax j : ℝ) : ℂ))) := by
        rw [← Complex.ofReal_mul]
      rw [h1, h2, hpt j]
    rw [hkey, map_mul, ← Complex.exp_conj]
    have hconjI : ∀ (x y : ℝ), conj (Complex.I * ((↑x * ↑y) : ℂ))
        = -(Complex.I * ((↑x * ↑y) : ℂ)) := by
      intro x y
      rw [map_mul, Complex.conj_I, map_mul, Complex.conj_ofReal, Complex.conj_ofReal]
      ring
    rw [hconjI]
  -- Step 3: rewrite the LHS summand
  rw [Finset.sum_congr rfl (fun j _ => hterm j)]
  -- Step 4: move conj inside the RHS sum. In this Mathlib revision `conj` is
  -- `starRingEnd ℂ`, which is `Star.star` definitionally, so `star_sum`'s
  -- conclusion unifies with the goal; its arguments are explicit: `(s) (f)`.
  exact (star_sum _ _).symm

/--
**Hermitian symmetry.** A real time-domain input has a spectrum satisfying
`A_{-i} = conj(A_i)`.

This is the physical statement that the Raman gain spectrum `H(Ω)` is real
(PLAN.md §7.1): `H` is the transform of a real response, so its bins pair off
as conjugates.

The proof is `conj_dft` specialised to real input — it is not an independent
result. `conj_dft` says `DFT(conj A) i = conj(DFT A (-i))`; with `A` real the
left side is `DFT A i`, and conjugating both sides gives `DFT A (-i) =
conj(DFT A i)`.
-/
theorem hermitian (n : ℕ) (Tmax : ℝ) (s : AxisSign) (hs : validAxis s)
    (hn : Even n) (hT : Tmax ≠ 0) (A : Fin n → ℂ)
    (hreal : ∀ j, A j = ((A j).re : ℂ)) (i : Fin n) :
    (dftMatrix n s Tmax *ᵥ A) (-i) = conj ((dftMatrix n s Tmax *ᵥ A) i) := by
  have hc : (fun j => conj (A j)) = A := by
    funext j
    rw [hreal j, Complex.conj_ofReal]
  have h := conj_dft n Tmax s hs hn hT A i
  rw [hc] at h
  have h' := congrArg (fun z : ℂ => conj z) h
  simpa using h'.symm

end Grid
end Photonics