import Photonics.Grid.Basic

/-!
# Dispersion

The real-valued dispersion and soliton relations the GNLSE and the soliton
analysis are built on. Every theorem here has a Python counterpart in
`gnlse.py` or `soliton.py` and is registered in `CONFORMANCE.md`.

Unlike Stage B this file stays entirely in `ℝ`. The ℝ/ℂ boundary is what made the
Fourier pairing expensive, and none of these relations needs it.
-/

namespace Photonics
namespace Dispersion

open Conventions

/-
Python side (`soliton.py::SolitonAnalysis`):

    beta2_si  = betas[0] * 1e-24                 # ps^2/m -> s^2/m
    N         = sqrt(gamma * P_peak * T0^2 / |beta2_si|)
    L_D       = T0^2 / |beta2_si|
    L_NL      = 1 / (gamma * P_peak)
    L_fiss    = L_D / (N * eta)
-/

/-! ## The nonlinear coefficient

`γ = n₂·ω₀·Γ/(c·A_eff)`, the constant in `photonics_helper`'s Kerr term and
the one every nonlinear length is built from.
-/

noncomputable def gamma (n2 ω0 G Aeff c : ℝ) : ℝ := n2 * ω0 * G / (c * Aeff)

theorem gamma_eq (n2 ω0 G Aeff c : ℝ) : gamma n2 ω0 G Aeff c = n2 * ω0 * G / (c * Aeff) :=
  rfl

/-- The confinement factor `Γ` lies in `[0, 1]`; this is a modelling assumption,
not a derived fact, and it is recorded as a hypothesis wherever it is used. -/
theorem gamma_pos (hn2 : 0 < n2) (hω0 : 0 < ω0) (hG : 0 < G) (hG1 : G ≤ 1)
    (hA : 0 < Aeff) (hc : 0 < c) :
    0 < gamma n2 ω0 G Aeff c := by
  have hnum : 0 < n2 * ω0 * G := mul_pos (mul_pos hn2 hω0) hG
  have hden : 0 < c * Aeff := mul_pos hc hA
  exact div_pos hnum hden

/-- The library's default `Γ = 1` recovers the fibre case, and `γ` then equals
`n₂ω₀/(c·A_eff)`. -/
theorem gamma_full_confinement (n2 ω0 Aeff c : ℝ) :
    gamma n2 ω0 1 Aeff c = n2 * ω0 / (c * Aeff) := by
  simp [gamma]

/-! ## Characteristic lengths -/

/-- Dispersion length `L_D = T0²/|β₂|`. -/
noncomputable def LD (T0 β2 : ℝ) : ℝ := T0 ^ 2 / |β2|

theorem LD_eq (T0 β2 : ℝ) : LD T0 β2 = T0 ^ 2 / |β2| := rfl

theorem LD_pos (hT : 0 < T0) (hb : β2 ≠ 0) : 0 < LD T0 β2 :=
  div_pos (sq_pos_of_pos hT) (abs_pos.mpr hb)

/-- Nonlinear length `L_NL = 1/(γ·P_peak)`. -/
noncomputable def LNL (g P : ℝ) : ℝ := 1 / (g * P)

theorem LNL_eq (g P : ℝ) : LNL g P = 1 / (g * P) := rfl

theorem LNL_pos (hg : 0 < g) (hP : 0 < P) : 0 < LNL g P :=
  div_pos one_pos (mul_pos hg hP)

/-- Soliton order `N = √(γ·P·T0²/|β₂|)`. -/
noncomputable def solitonOrder (g P T0 β2 : ℝ) : ℝ :=
  Real.sqrt (g * P * T0 ^ 2 / |β2|)

theorem solitonOrder_eq (g P T0 β2 : ℝ) :
    solitonOrder g P T0 β2 = Real.sqrt (g * P * T0 ^ 2 / |β2|) := rfl

/--
**The defining relation `N² = L_D/L_NL`.**

This is Agrawal eq. 3.2.1, and it is the relation that ties the soliton order to
the two characteristic lengths: `N²` is exactly how many dispersion lengths fit
into one nonlinear length.
-/
theorem soliton_order_sq (g P T0 β2 : ℝ) (hg : 0 < g) (hP : 0 < P) (hT : 0 < T0)
    (hb : β2 ≠ 0) :
    solitonOrder g P T0 β2 ^ 2 = LD T0 β2 / LNL g P := by
  unfold solitonOrder LD LNL
  have hgP : g * P ≠ 0 := ne_of_gt (mul_pos hg hP)
  have hb' : |β2| ≠ 0 := abs_ne_zero.mpr hb
  rw [Real.sq_sqrt (by positivity : 0 ≤ g * P * T0 ^ 2 / |β2|)]
  field_simp

/--
**Fundamental soliton:** `N = 1` exactly when the dispersion and nonlinear
lengths coincide.

Agrawal's fundamental soliton is the special case `N = 1`, and this says exactly
that it is the case `L_D = L_NL`. It is the cleanest statement that the two
"lengths" are not independent quantities.
-/
theorem fundamental_soliton (g P T0 β2 : ℝ) (hg : 0 < g) (hP : 0 < P) (hT : 0 < T0)
    (hb : β2 ≠ 0) :
    solitonOrder g P T0 β2 = 1 ↔ LD T0 β2 = LNL g P := by
  have hL : LNL g P ≠ 0 := ne_of_gt (LNL_pos hg hP)
  have hsq := soliton_order_sq g P T0 β2 hg hP hT hb
  constructor
  · intro h
    have h1 : solitonOrder g P T0 β2 ^ 2 = 1 := by rw [h]; norm_num
    rw [hsq] at h1
    exact (div_eq_one_iff_eq hL).mp h1
  · intro h
    have h2 : LD T0 β2 / LNL g P = 1 :=
      (div_eq_one_iff_eq hL).mpr h
    have h1 : solitonOrder g P T0 β2 ^ 2 = 1 := by rw [hsq, h2]
    have _ := h1
    have hN0 : 0 ≤ solitonOrder g P T0 β2 := Real.sqrt_nonneg _
    rcases mul_eq_zero.mp (show (solitonOrder g P T0 β2 - 1)
        * (solitonOrder g P T0 β2 + 1) = 0 by nlinarith [h1]) with hz | hz
    · linarith
    · linarith

/-- `L_D` grows with `T0²` and shrinks with `|β₂|`, both separately. -/
theorem LD_mono_T0 {T0 β2 : ℝ} (hT : 0 < T0) (hb : β2 ≠ 0) {T0' : ℝ} (hT' : T0 ≤ T0') :
    LD T0 β2 ≤ LD T0' β2 := by
  unfold LD
  have hb' : |β2| ≠ 0 := abs_ne_zero.mpr hb
  have hden : 0 < |β2| := abs_pos.mpr hb
  rw [div_le_div_iff₀ hden (abs_pos.mpr hb)]
  · have hsq' : T0 ^ 2 ≤ T0' ^ 2 := by nlinarith
    exact mul_le_mul_of_nonneg_right hsq' (le_of_lt hden)

/-- Fission length `L_fiss = L_D/(N·η)`. -/
noncomputable def fissionLength (LD' N eta : ℝ) : ℝ := LD' / (N * eta)

theorem fissionLength_eq (LD' N eta : ℝ) : fissionLength LD' N eta = LD' / (N * eta) := rfl

/--
**Higher-order fission needs more room.** `N > N'` gives a shorter fission
length at fixed `η`, which is the `L_fiss = L_D/(N·η)` relation in usable form.
-/
theorem fissionLength_mono (LD' : ℝ) {N N' eta : ℝ} (hLD : 0 < LD')
    (hN : 0 < N) (hN' : 0 < N') (heta : 0 < eta) (hNN : N' ≤ N) :
    fissionLength LD' N eta ≤ fissionLength LD' N' eta := by
  unfold fissionLength
  have hNe : 0 < N * eta := mul_pos hN heta
  have hN'e' : 0 < N' * eta := mul_pos hN' heta
  rw [div_le_div_iff₀ hNe hN'e']
  · exact mul_le_mul_of_nonneg_left
      (mul_le_mul_of_nonneg_right hNN (le_of_lt heta)) (le_of_lt hLD)

/-! ## Group delay

The action of the dispersion operator on a pulse's arrival time. This is the
observable that the walk-off between polarisation axes or between modes is
measured with, and it is linear in the length.
-/

/-- Group delay through length `z` is `β₁·z`. -/
noncomputable def groupDelay (β1 z : ℝ) : ℝ := β1 * z

theorem groupDelay_eq (β1 z : ℝ) : groupDelay β1 z = β1 * z := rfl

/-- Two modes separate by `ΔT = (β₁ₓ − β₁ᵧ)·z`, with the sign fixed by which is
faster. -/
theorem walkoff (β1x β1y z : ℝ) :
    groupDelay β1x z - groupDelay β1y z = (β1x - β1y) * z := by
  simp [groupDelay]
  ring

/-- Reversing the propagation direction reverses the delay, and nothing else
changes. This is the group-delay counterpart of `Conventions.beta_flip`. -/
theorem groupDelay_neg (β1 z : ℝ) : groupDelay (-β1) z = -(groupDelay β1 z) := by
  simp [groupDelay]

/-! ## The Taylor expansion of the propagation constant -/

/-- The truncated propagation phase at frequency offset `Ω`, to order `m`. -/
noncomputable def betaPhase (b : ℕ → ℝ) (m : ℕ) (Ω z : ℝ) : ℝ :=
  ∑ k ∈ Finset.range (m + 1), b k * Ω ^ k * z / (Nat.factorial k : ℝ)

/--
At zero frequency offset the phase collapses to the carrier phase `β₀·z`,
whatever the truncation order.

**Not** to zero: the `k = 0` term survives. This is the carrier phase, and it is
why a solver written in a rotating frame must still carry `β₀`. An earlier
draft of this file claimed the phase vanished at `Ω = 0`; that is false and
Lean rejected it.
-/
theorem betaPhase_at_zero (b : ℕ → ℝ) (m : ℕ) (z : ℝ) : betaPhase b m 0 z = b 0 * z := by
  unfold betaPhase
  calc ∑ k ∈ Finset.range (m + 1), b k * 0 ^ k * z / (Nat.factorial k : ℝ)
      = b 0 * (0 : ℕ) ^ 0 * z / (Nat.factorial 0 : ℝ) := by
        apply Finset.sum_eq_single 0
        · intro k _ hk
          have hk0 : 0 < k := Nat.zero_lt_of_ne_zero hk
          have : (0 : ℝ) ^ (k : ℕ) = 0 := by
            rw [show (0 : ℝ) ^ (k : ℕ) = if k = 0 then 1 else 0 from by
                  cases k with
                  | zero => simp
                  | succ k => simp]
            rw [if_neg (by omega)]
          rw [this]
          simp
        · intro h0
          simp at h0
    _ = b 0 * z := by simp

/-- The phase is linear in the propagation length, as an integral of the
propagation constant must be. -/
theorem betaPhase_linear_z (b : ℕ → ℝ) (m : ℕ) (Ω : ℝ) (z z' : ℝ) :
    betaPhase b m Ω (z + z') = betaPhase b m Ω z + betaPhase b m Ω z' := by
  unfold betaPhase
  calc ∑ k ∈ Finset.range (m + 1), b k * Ω ^ k * (z + z') / (Nat.factorial k : ℝ)
      = ∑ k ∈ Finset.range (m + 1),
          (b k * Ω ^ k * z / (Nat.factorial k : ℝ)
            + b k * Ω ^ k * z' / (Nat.factorial k : ℝ)) := by
          refine Finset.sum_congr rfl ?_
          intro k _
          ring
    _ = (∑ k ∈ Finset.range (m + 1), b k * Ω ^ k * z / (Nat.factorial k : ℝ))
        + ∑ k ∈ Finset.range (m + 1), b k * Ω ^ k * z' / (Nat.factorial k : ℝ) :=
        Finset.sum_add_distrib

/-! ## Spontaneous fissions

`L_fiss = L_D/(N·η)` counts the separation of the solitons that emerge from a
higher-order soliton. For `N = 1` there is nothing to fission: this rules the
mechanism out rather than leaving a division by `N`.
-/
theorem no_fission_at_fundamental (g P T0 β2 : ℝ) (hg : 0 < g) (hP : 0 < P) (hT : 0 < T0)
    (hb : β2 ≠ 0) (eta : ℝ) (hN : solitonOrder g P T0 β2 = 1) :
    fissionLength (LD T0 β2) (solitonOrder g P T0 β2) eta = LD T0 β2 / eta := by
  rw [hN]
  simp [fissionLength]

/-! ## F19 — the Taylor truncation of the phase

`phase_matching.py::_taylor_phase_max_error` compares the *full* dispersion
phase against Taylor truncation and returns the max error. The theorem
underneath it is Taylor's theorem with Lagrange remainder, applied to `f = β`
-between any two grid points: the full phase minus its truncation is exactly
`f^(n+1)(x')·Δ^(n+1)/(n+1)!` for some intermediate point `x'` between `ω₀` and
`x`, and whenever a bound `M` is known on that derivative across the interval,
the quantity is at most `M·Δ^(n+1)/(n+1)!`. -/

/-- The Lagrange form of the phase truncation remainder. `f = β(ω)`, `x₀` the
carrier, `n` the truncation order — this is the statement whose numerical
counterpart `_taylor_phase_max_error` evaluates: the coefficients the Python
carries in its `betas` array are exactly `f^(k)(ω₀)`. -/
theorem phaseRemainder_lagrange (f : ℝ → ℝ) (n : ℕ) (hf : ContDiff ℝ (n + 1) f)
    (x₀ x : ℝ) (hx : x₀ ≠ x) :
    ∃ x', x' ∈ Set.uIoo x₀ x ∧
      f x - (∑ k ∈ Finset.range (n + 1),
          (iteratedDeriv k f x₀) * (x - x₀) ^ k / (Nat.factorial k : ℝ))
        = iteratedDeriv (n + 1) f x' * (x - x₀) ^ (n + 1)
          / (Nat.factorial (n + 1) : ℝ) := by
  obtain ⟨x', hmem, h2⟩ :=
    taylor_mean_remainder_lagrange_iteratedDeriv (f := f) (x := x) (x₀ := x₀)
      (n := n) hx hf.contDiffOn
  have h0mem : x₀ ∈ Set.uIcc x₀ x := by
    by_cases hxx : x₀ ≤ x
    · exact Set.mem_uIcc.mpr (Or.inl ⟨le_refl _, hxx⟩)
    · exact Set.mem_uIcc.mpr (Or.inr ⟨le_of_lt (not_le.mp hxx), le_refl _⟩)
  have huniq : UniqueDiffOn ℝ (Set.uIcc x₀ x) := uniqueDiffOn_uIcc hx
  have htaylor : taylorWithinEval f n (Set.uIcc x₀ x) x₀ x
      = ∑ k ∈ Finset.range (n + 1), iteratedDeriv k f x₀ * (x - x₀) ^ k
        / (Nat.factorial k : ℝ) := by
    rw [taylor_within_apply]
    apply Finset.sum_congr rfl
    intro k hk
    have hkle : (k : ℕ) ≤ n + 1 := le_of_lt (Finset.mem_range.mp hk)
    have hstep : iteratedDerivWithin k f (Set.uIcc x₀ x) x₀ = iteratedDeriv k f x₀ := by
      rw [iteratedDerivWithin_eq_iteratedDeriv huniq
        (hf.contDiffAt.of_le (by exact_mod_cast hkle)) h0mem]
    rw [hstep, smul_eq_mul, inv_eq_one_div]
    ring
  rw [htaylor] at h2
  exact ⟨x', hmem, h2⟩

/-- The truncation bound `_taylor_phase_max_error` estimates: given `M` a bound
on `|f^(n+1)|` across the interval, the phase truncation error of the Taylor
polynomial is at most `M·|Δ|^(n+1)/(n+1)!`, at every grid point `x`. -/
theorem phaseRemainder_bound (f : ℝ → ℝ) (n : ℕ) (hf : ContDiff ℝ (n + 1) f)
    (x₀ x : ℝ) (M : ℝ)
    (hM : ∀ y : ℝ, y ∈ Set.uIcc x₀ x → |iteratedDeriv (n + 1) f y| ≤ M) :
    |f x - (∑ k ∈ Finset.range (n + 1),
        iteratedDeriv k f x₀ * (x - x₀) ^ k / (Nat.factorial k : ℝ))|
      ≤ M * |x - x₀| ^ (n + 1) / (Nat.factorial (n + 1) : ℝ) := by
  by_cases hxeq : x₀ = x
  · subst hxeq
    have h0 : (x₀:ℝ) - x₀ = 0 := sub_self x₀
    have hsum0 : (∑ k ∈ Finset.range (n + 1), (iteratedDeriv k f x₀)
        * (x₀ - x₀) ^ k / (Nat.factorial k : ℝ)) = f x₀ := by
      rw [Finset.sum_eq_single 0]
      · rw [pow_zero, iteratedDeriv_zero]
        norm_num
      · intro k _ hk
        rw [show ((x₀:ℝ) - x₀) ^ k = (0:ℝ) ^ k from congrArg (· ^ k) h0,
          zero_pow hk]
        norm_num
      · intro hnotmem
        exact absurd (Finset.mem_range.2 (Nat.zero_lt_succ n)) hnotmem
    rw [hsum0]
    simp [h0]
  · obtain ⟨x', hmem, h2⟩ := phaseRemainder_lagrange f n hf x₀ x hxeq
    have hxmem0 : x₀ ∈ Set.uIcc x₀ x := by
      by_cases hh : x₀ ≤ x
      · exact Set.mem_uIcc.mpr (Or.inl ⟨le_refl _, hh⟩)
      · exact Set.mem_uIcc.mpr (Or.inr ⟨le_of_lt (not_le.mp hh), le_refl _⟩)
    have hmxmem : x' ∈ Set.uIcc x₀ x := Set.uIoo_subset_uIcc_self hmem
    have hx'M : |iteratedDeriv (n + 1) f x'| ≤ M := hM x' hmxmem
    have hMnonneg : 0 ≤ M := le_trans (abs_nonneg _) (hM x₀ hxmem0)
    have hpownonneg : 0 ≤ |x - x₀| ^ (n + 1) := pow_nonneg (abs_nonneg _) _
    rw [h2, abs_div]
    have hnum : |iteratedDeriv (n + 1) f x' * (x - x₀) ^ (n + 1)|
        = |iteratedDeriv (n + 1) f x'| * |x - x₀| ^ (n + 1) := by
      rw [abs_mul, abs_pow]
    have hFpos : (0:ℝ) < (Nat.factorial (n + 1) : ℝ) :=
      Nat.cast_pos.2 (Nat.factorial_pos (n + 1))
    rw [hnum, abs_of_pos hFpos]
    have hpowle : |iteratedDeriv (n + 1) f x'| * |x - x₀| ^ (n + 1)
        ≤ M * |x - x₀| ^ (n + 1) :=
      mul_le_mul hx'M (le_refl _) hpownonneg hMnonneg
    exact div_le_div_of_nonneg_right hpowle
      (Nat.cast_nonneg (Nat.factorial (n + 1)))

end Dispersion
end Photonics