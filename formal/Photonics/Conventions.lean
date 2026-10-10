/-
Copyright (c) 2026 photonics_helper contributors.
Released under the repository licence (see LICENSE).

# Conventions

The single source of truth for every sign, orientation, and ordering choice
in the formal development.

Every later file imports this one. Nothing downstream re-derives a sign.

The two decisions recorded here are the ones the whole library depends on:

1. **Spectral-axis orientation.** A grid's angular-frequency axis may be
   oriented as `w = +2π·fftfreq` or `w = -2π·fftfreq`. Nothing about the grid
   is asymmetric; but *odd-order* dispersion is antisymmetric in it. So the
   orientation is not a cosmetic choice — it is part of the physics, and it is
   recorded here once so that no sub-operator can pick its own.

2. **Fourier pairing.** The analysis and synthesis kernels are a matched pair.
   Changing one without the other changes the meaning of a convolution, so the
   pair is stated as a unit.

Throughout, physical constants are **opaque reals**. We prove relations
between them, never their numeric values.
-/

import Mathlib

namespace Photonics
namespace Conventions

open Matrix BigOperators Complex
-- In this Mathlib revision complex conjugation is the `star` operation, exposed
-- as the notation `conj` in the scope `ComplexConjugate`.
open scoped ComplexConjugate

universe u

/-! ## Spectral-axis orientation -/

/--
Orientation of the spectral axis, as a real sign.

`axisSign = 1` is the standard FFT orientation (`w = +2π·fftfreq`);
`axisSign = -1` is its mirror image.
-/
abbrev AxisSign := ℝ

/--
The two admissible orientations. A spectral axis is a choice between these and
nothing else — this is what "picking a convention" means formally.
-/
def validAxis (s : AxisSign) : Prop := s = 1 ∨ s = -1

/--
Dispersion coefficient `k` as seen on an axis with orientation `s`.

The `k`-th order coefficient transforms as `s^k`: an odd-order term is
antisymmetric under axis reversal, an even-order term is invariant. This is the
formal content of "the axis orientation is part of the physics".
-/
def betaAxis (s : AxisSign) (k : ℕ) (b : ℕ → ℝ) : ℝ := s ^ k * b k

/--
**Axis reversal flips exactly the odd-order dispersion terms.**

This is the central convention theorem of the project. It is stated for an
arbitrary order `k` and an arbitrary coefficient sequence `b`, so it applies to
every dispersion coefficient in the library without exception.

Note the precise form. The comparison is between the **reversed** axis and the
**identity** axis — it is not a statement about an arbitrary orientation `s`.
A general-`s` version would assert `b k = -(b k)` at `s = 1` and odd `k`,
which is false; see `CHANGELOG.md` for that correction.
-/
theorem beta_flip (k : ℕ) (b : ℕ → ℝ) :
    betaAxis (-1 : AxisSign) k b = (-1 : ℝ) ^ k * betaAxis 1 k b := by
  simp [betaAxis]

/-- `(r^m)^2 = (r^2)^m` — squaring a power is a power of the square. -/
private lemma sq_pow (r : ℝ) (m : ℕ) : (r ^ m) ^ 2 = (r ^ 2) ^ m := by
  rw [← pow_mul]
  ring_nf

/--
Even-order coefficients are invariant under axis reversal.

This one *is* a statement about an arbitrary orientation, because for even `k`
the `s^k` factor is `1` for both admissible orientations. The hypothesis
`validAxis s` is doing real work: without it, `s = 2` would refute it.
-/
theorem beta_flip_even (s : AxisSign) (k : ℕ) (b : ℕ → ℝ) (h : validAxis s)
    (hk : Even k) : betaAxis s k b = betaAxis 1 k b := by
  obtain ⟨m, rfl⟩ := hk
  show s ^ (m + m) * b (m + m) = 1 ^ (m + m) * b (m + m)
  have hsq : (s ^ m) ^ 2 = 1 := by
    rcases h with rfl | rfl
    · simp
    · rw [sq_pow]; norm_num
  rw [pow_add, ← sq, hsq, one_mul]
  simp

/--
Odd-order coefficients are negated under axis reversal.
-/
theorem beta_flip_odd (k : ℕ) (b : ℕ → ℝ) (hk : Odd k) :
    betaAxis (-1 : AxisSign) k b = -(betaAxis 1 k b) := by
  obtain ⟨m, rfl⟩ := hk
  simp only [betaAxis]
  have key : (-1 : ℝ) ^ (2 * m + 1) = -1 := by
    induction m with
    | zero => norm_num
    | succ m ih =>
        rw [show (2 : ℕ) * (m + 1) + 1 = 2 * m + 1 + 2 by ring, pow_succ, pow_succ, ih]
        ring
  rw [key]
  ring

/--
Reversing the spectral axis twice is the identity.

Applied to a coefficient sequence, the transformation `b ↦ (k ↦ betaAxis s k b)`
is an involution for either orientation `s`: the `s^k` factors multiply out to
`s^(2k) = 1`.
-/
theorem axis_reversal_is_involution (k : ℕ) (b : ℕ → ℝ) :
    betaAxis (-1 : AxisSign) k (fun j => betaAxis (-1 : AxisSign) j b) = b k := by
  simp only [betaAxis]
  rw [← mul_assoc, ← mul_pow, show ((-1 : ℝ) * (-1 : ℝ)) = 1 by norm_num, one_pow,
    one_mul]

/-! ## The Fourier pair

A uniform temporal grid of `n` points over a window `Tmax`:

- sample spacing `dt = Tmax / n`
- time samples `t_j = -Tmax/2 + j·dt`
- angular frequencies `ω_i = s · 2π · (i - n/2) / Tmax` (the `fftshift`ed
  `fftfreq`, so that bin index is signed)

Only the *shape* of the pair is fixed here; the arithmetic identities live in
`Photonics.Grid.Spectrum`.
-/

/-- Sample spacing of a uniform temporal grid: `dt = Tmax / n`. -/
noncomputable def gridDt (n : ℕ) (Tmax : ℝ) : ℝ := Tmax / n

/-- Time sample `j` of a uniform grid: `t_j = -Tmax/2 + j·dt`, matching the
endpoint-excluded `np.linspace(-Tmax/2, Tmax/2 - dt, n)` the library uses. -/
noncomputable def gridTime (n : ℕ) (Tmax : ℝ) (j : Fin n) : ℝ :=
  -Tmax / 2 + (j : ℝ) * gridDt n Tmax

/-- Angular-frequency sample `i`, in the `fftshift`ed `fftfreq` ordering so that
the bin index is signed: `ω_i = s·2π·(i - n/2)/Tmax`. -/
noncomputable def gridOmega (n : ℕ) (s : AxisSign) (Tmax : ℝ) (i : Fin n) : ℝ :=
  s * (2 * Real.pi) * ((i : ℝ) - n / 2) / Tmax

/--
The DFT matrix at a given spectral-axis orientation: entry `i j` is the
analysis kernel `exp(i·ω_i·t_j)`.

We take the **analysis kernel to be `e^{+iΩt}`** and the synthesis kernel
`e^{-iΩt}`, which is the Agrawal pairing for the envelope convention
`A(z,T) = ∫ Ã(Ω) e^{-iΩT}`.
-/
noncomputable def dftMatrix (n : ℕ) (s : AxisSign) (Tmax : ℝ) : Matrix (Fin n) (Fin n) ℂ :=
  fun i j => Complex.exp (Complex.I * ((gridOmega n s Tmax i) * gridTime n Tmax j))

/--
The inverse (synthesis) matrix: entry `i j` is `exp(-i·ω_i·t_j)`.
-/
noncomputable def idftMatrix (n : ℕ) (s : AxisSign) (Tmax : ℝ) : Matrix (Fin n) (Fin n) ℂ :=
  fun i j => Complex.exp (-(Complex.I * ((gridOmega n s Tmax i) * gridTime n Tmax j)))

/--
**The pair is matched.** Composing analysis with synthesis returns the
identity up to the `1/n` normalisation that `TemporalGrid.fft` and
`TemporalGrid.ifft` supply via `dt`.

Stating this as a theorem — rather than as a comment in the docstring — is what
makes the pairing a fact the library can rely on, rather than a convention
someone has to remember.

**Status: open (Stage B).** The general form is a statement about sums of
roots of unity; see `PLAN.md` §5.2 for the concrete-`n` fallback. This is
tracked as an open statement, not silently left `sorry`.
-/
theorem dft_pair (n : ℕ) (s : AxisSign) (Tmax : ℝ) (h : 0 < n) (hT : 0 < Tmax) :
    (idftMatrix n s Tmax * dftMatrix n s Tmax) = (n : ℂ) • (1 : Matrix (Fin n) (Fin n) ℂ) := by
  sorry

/--
Conjugation identity: conjugating the analysis transform equals analysing the
conjugate, with the sign of the frequency index reversed.

This is the algebraic content of the conjugated-transform-pair fast path
`conj(DFT(conj(·)))`, and it holds because the kernel is a real-exponential,
not because of anything specific to the implementation.

**Status: open (Stage B).** See `PLAN.md` §5.2.
-/
theorem conj_dft (n : ℕ) (s : AxisSign) (Tmax : ℝ) (A : Fin n → ℂ) (hn : Even n)
    (i : Fin n) :
    (dftMatrix n s Tmax *ᵥ (fun j => conj (A j))) i
      = conj ((dftMatrix n s Tmax *ᵥ A) ((-i : Fin n) : Fin n)) := by
  sorry

/--
Hermitian symmetry: a real-valued field has a conjugate-symmetric spectrum.

Needed downstream to show that the Raman gain spectrum `H(Ω)` is real.

**Status: open (Stage B).** See `PLAN.md` §5.2.
-/
theorem hermitian (n : ℕ) (s : AxisSign) (Tmax : ℝ) (A : Fin n → ℂ) (hn : Even n)
    (hA : ∀ j, conj (A j) = A j) (i : Fin n) :
    conj ((dftMatrix n s Tmax *ᵥ A) i)
      = (dftMatrix n s Tmax *ᵥ (fun j => conj (A j))) i := by
  sorry

/-! ## Layer ordering -/


/--
A DBR stack is a finite sequence of layers, **ordered from the incidence
medium outward**. The characteristic matrix of the stack is the ordered product
`M₁ · M₂ · … · M_N` in that same order (Macleod's convention).

Recorded once, here, because the product is order-sensitive and nothing in the
type of a stack would otherwise say which order was meant.
-/
def layerIndex (N : ℕ) (j : Fin N) : ℕ := j

theorem layerIndex_lt (N : ℕ) (j : Fin N) : layerIndex N j < N :=
  j.isLt

/-! ## Units -/

/--
Dimensional bookkeeping: a vector of exponents over
`length, time, mass, current`.

Charge does not get its own coordinate — it enters through `ε₀` and `μ₀`, which
are treated as dimensionless references. Products of dimensions add their
exponent vectors.
-/
structure Dim where
  length : ℤ
  time : ℤ
  mass : ℤ
  current : ℤ
  deriving DecidableEq, Repr

instance : One Dim := ⟨⟨0, 0, 0, 0⟩⟩
instance : Add Dim := ⟨fun a b => ⟨a.length + b.length, a.time + b.time, a.mass + b.mass, a.current + b.current⟩⟩
instance : Mul Dim := ⟨fun a b => ⟨a.length * b.length, a.time * b.time, a.mass * b.mass, a.current * b.current⟩⟩

/-- `Pow` on dimensions is exponent multiplication, so that `ℓ^3` reads as
"three powers of length" the way it does physically. -/
instance : Pow Dim Nat := ⟨fun d k => ⟨d.length * k, d.time * k, d.mass * k, d.current * k⟩⟩

namespace Dim

/-- One dimension, i.e. dimensionless. -/
def one : Dim := 1

/-- Length: base unit `m`. -/
def len : Dim := ⟨1, 0, 0, 0⟩

/-- Time: base unit `s`. -/
def tim : Dim := ⟨0, 1, 0, 0⟩

/-- Mass: base unit `kg`. -/
def mss : Dim := ⟨0, 0, 1, 0⟩

/-- Current: base unit `A`. -/
def crnt : Dim := ⟨0, 0, 0, 1⟩

/-- Reciprocal of a dimension. -/
def inv (d : Dim) : Dim := ⟨-d.length, -d.time, -d.mass, -d.current⟩

/-- Power is mass·length²·time⁻³. -/
def power : Dim := mss * len ^ 2 * inv (tim ^ (3 : Nat))

end Dim



end Conventions
end Photonics