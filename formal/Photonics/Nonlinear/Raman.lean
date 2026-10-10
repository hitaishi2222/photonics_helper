import Photonics.Grid.Basic

/-!
# Raman response: causality and gain

The Raman term's two structural properties that the code relies on: the
response is causal, and it cannot amplify beyond unity.

The physically richer statements — the gain spectrum `H(Ω)` being real, and
photon-number conservation of the sub-operators — are complex-valued, and are
blocked on the same coercion problem documented in `Spectrum.lean`. What is here
is the real-valued content, which is not a consolation prize: the gain bound is
what stops a mis-set delay table from amplifying the solution without limit.
-/

namespace Photonics
namespace Nonlinear

open Conventions BigOperators

/-! ## Causality

`gnlse.py` convolves the intensity with a *causal* `h_R`. Causality is not an
optimisation: the retarded contribution at time `t` is drawn entirely from
intensities at times `≤ t`, so a pulse cannot drive the Raman response before
the delay has elapsed.
-/

/-- `h_R` vanishes on every sample before the delay `start`. -/
def ramanCausal {n : ℕ} (hR : Fin n → ℝ) (start : ℕ) : Prop :=
  ∀ j, j.val < start → hR j = 0

theorem ramanCausal_of_all {n : ℕ} {hR : Fin n → ℝ} {start : ℕ}
    (h : ∀ j, j.val < start → hR j = 0) : ramanCausal hR start := h

/-- Causality and non-negativity together: the response is zero early and
non-negative afterwards. This is the shape of every physical `h_R`. -/
theorem ramanShape {n : ℕ} {hR : Fin n → ℝ} {start : ℕ}
    (hc : ramanCausal hR start) (hpos : ∀ j, start ≤ j.val → 0 ≤ hR j)
    (j : Fin n) (hlate : start ≤ j.val) : 0 ≤ hR j := hpos j hlate

theorem ramanShape_early {n : ℕ} {hR : Fin n → ℝ} {start : ℕ}
    (hc : ramanCausal hR start) (j : Fin n) (hearly : j.val < start) : hR j = 0 :=
  hc j hearly

/-! ## The delayed convolution -/

/--
The delayed Raman drive at sample `j`: the intensity weighted by the response,
sampled `delay` earlier. Indices below the delay contribute nothing, which is
causality again, expressed as a definition that cannot be evaluated out of
range.
-/
noncomputable def ramanDrive (n : ℕ) (hn : 0 < n) (delay : ℕ) (P : Fin n → ℝ)
    (hR : Fin n → ℝ) (j : Fin n) : ℝ :=
  ∑ i : Fin n,
    if i.val + delay ≤ j.val then
      hR ⟨j.val - delay - i.val, by have hjlt := j.isLt; omega⟩ * P i
    else 0

/-- **The drive vanishes before the delay has elapsed.** Causality as an
equation rather than a side condition. -/
theorem ramanDrive_early {n delay : ℕ} (hn : 0 < n) (hdelay : 0 < delay) {P : Fin n → ℝ}
    {hR : Fin n → ℝ} {j : Fin n} (hearly : j.val < delay) :
    ramanDrive n hn delay P hR j = 0 := by
  unfold ramanDrive
  apply Finset.sum_eq_zero
  intro i _
  by_cases hc : i.val + delay ≤ j.val
  · omega
  · simp [hc]

/-! ## The gain bound -/

/--
**The Raman drive is non-negative.**

Each surviving summand is `h_R(...) * P(i)`, and both factors are non-negative,
so the drive cannot be negative. Combined with the gain bound below this pins
the Raman term's whole effect on the intensity: it adds energy, by at most a
bounded amount.
-/
theorem ramanDrive_nonneg {n : ℕ} (hn : 0 < n) (hR : Fin n → ℝ) {P : Fin n → ℝ}
    (hP : ∀ i, 0 ≤ P i) (hRpos : ∀ i, 0 ≤ hR i) {j : Fin n} :
    0 ≤ ramanDrive n hn 0 P hR j := by
  unfold ramanDrive
  apply Finset.sum_nonneg
  intro i _
  by_cases hc : i.val + 0 ≤ j.val
  · simp only [hc, ↓reduceIte]
    apply mul_nonneg
    · exact hRpos _
    · exact hP i
  · have hc' : ¬ (i.val ≤ j.val) := by omega
    simp [hc']

/--
**The Raman term cannot amplify past unity.** Stated as a target for the
remaining Stage D work; the ℝ-valued statement is within reach but the
`Fin`-indexed bookkeeping needed for a clean proof was not finished in the
session that added this file.

The claim: with a non-negative response of total weight at most `1`, the drive
at any sample is at most the total input. That is what makes the Raman step
safe to iterate — a mis-set delay table can distort a solution but cannot drive
it exponentially.
-/
theorem ramanDrive_le_max {n : ℕ} (hn : 0 < n) (hR : Fin n → ℝ)
    (hnorm : (∑ i : Fin n, hR i) ≤ 1)
    (hRpos : ∀ i, 0 ≤ hR i)
    {P : Fin n → ℝ} (hP : ∀ i, 0 ≤ P i) {j : Fin n} :
    ramanDrive n hn 0 P hR j ≤ ∑ i : Fin n, P i := by
  sorry

end Nonlinear
end Photonics
