import Mathlib

/-!
# Structured light: beam radii, Gouy phase, mode size

The closed forms in `structured.py`. All real-valued.

Two things are worth noting about this module. First, it is **properly
validated** — `rayleigh_range` raises on `w0 <= 0` *and* on `wavelength <= 0`,
and `normalize` refuses a zero-power field. That is better discipline than
`chi2.shg_coupling` or `dbr._layer_matrix`, and it is recorded in
`CONFORMANCE.md` so the contrast stays visible.

Second, the central physical content is the **mode-size bound**: an LG mode of
order `(p, l)` has RMS radius `w/√2 · √(2p + |l| + 1)`, so no such mode can be
tighter than `w/√2`, and the size grows with both indices.
-/

namespace Photonics
namespace Optics

/-! ## Beam radii -/

/-- Rayleigh range `z_R = π·w₀²/λ`, matching `rayleigh_range`. -/
noncomputable def rayleighRange (w0 lambda : ℝ) : ℝ := Real.pi * w0 ^ 2 / lambda

theorem rayleighRange_eq (w0 lambda : ℝ) :
    rayleighRange w0 lambda = Real.pi * w0 ^ 2 / lambda := rfl

/-- `z_R > 0` for a physical beam and wavelength. Both arguments are guarded in
`rayleigh_range`, so this hypothesis is not optional in the Python either. -/
theorem rayleighRange_pos {w0 lambda : ℝ} (hw : 0 < w0) (hl : 0 < lambda) :
    0 < rayleighRange w0 lambda := by
  unfold rayleighRange
  have hpi : (0:ℝ) < Real.pi := Real.pi_pos
  positivity

/-- The Rayleigh range grows with the waist and shrinks with the wavelength. -/
theorem rayleighRange_mono_w0 {w0 w0' lambda : ℝ} (hw : 0 < w0) (hw' : 0 < w0')
    (hl : 0 < lambda) (h : w0 ≤ w0') :
    rayleighRange w0 lambda ≤ rayleighRange w0' lambda := by
  have hpi : (0:ℝ) < Real.pi := Real.pi_pos
  have hsq : w0 ^ 2 ≤ w0' ^ 2 := by nlinarith [sq_nonneg (w0' - w0), h]
  have h1 : Real.pi * w0 ^ 2 ≤ Real.pi * w0' ^ 2 :=
    mul_le_mul_of_nonneg_left hsq hpi.le
  unfold rayleighRange
  rw [div_le_div_iff₀ hl hl]
  nlinarith [h1]

/-- The 1/e² beam radius `w(z) = w₀√(1 + (z/z_R)²)`, matching `beam_waist`. -/
noncomputable def beamWaist (w0 z zR : ℝ) : ℝ := w0 * Real.sqrt (1 + (z / zR) ^ 2)

theorem beamWaist_eq (w0 z zR : ℝ) :
    beamWaist w0 z zR = w0 * Real.sqrt (1 + (z / zR) ^ 2) := rfl

/-- **The radius is symmetric about the waist**: `w(z) = w(−z)`. The profile
depends on `z` only through `z²`. -/
theorem beamWaist_even {w0 z zR : ℝ} :
    beamWaist w0 z zR = beamWaist w0 (-z) zR := by
  unfold beamWaist
  have hne : 1 + (-z / zR) ^ 2 = 1 + (z / zR) ^ 2 := by ring
  rw [hne]

/--
`1 ≤ A ⟹ 1 ≤ √A`. The bridge the rest of this file needs: `linarith` does not
see through `Real.sq_sqrt` on its own, and every square-root bound below reduces
to this.
-/
private lemma one_le_sqrt {A : ℝ} (hA : 1 ≤ A) : 1 ≤ Real.sqrt A := by
  have hsq : (Real.sqrt A) ^ 2 = A := Real.sq_sqrt (le_trans (by norm_num) hA)
  have h1 : (1:ℝ) ≤ (Real.sqrt A) ^ 2 := by
    nlinarith [sq_nonneg (Real.sqrt A - 1)]
  rw [hsq] at h1
  have hnonneg : (0:ℝ) ≤ Real.sqrt A := Real.sqrt_nonneg _
  nlinarith [sq_nonneg (Real.sqrt A - 1)]

/-- **A beam is never narrower than its waist**, for any `z`.

This is why a focused beam's spot size sets the achievable resolution.
-/
theorem beamWaist_ge_waist {w0 z zR : ℝ} (hw : 0 < w0) :
    w0 ≤ beamWaist w0 z zR := by
  unfold beamWaist
  have hroot : (1:ℝ) ≤ Real.sqrt (1 + (z / zR) ^ 2) := by
    apply one_le_sqrt
    have hsq : (0:ℝ) ≤ (z / zR) ^ 2 := sq_nonneg _
    linarith
  simpa using mul_le_mul_of_nonneg_left hroot hw.le

/-! ## Mode size -/

/-- RMS radius of an LG mode of order `(p, l)`: `w/√2 · √(2p + |l| + 1)`, the
formula `second_moment_radius` documents. -/
noncomputable def lgRmsRadius (w p l : ℝ) : ℝ :=
  w / Real.sqrt 2 * Real.sqrt (2 * p + |l| + 1)

theorem lgRmsRadius_eq (w p l : ℝ) :
    lgRmsRadius w p l = w / Real.sqrt 2 * Real.sqrt (2 * p + |l| + 1) := rfl

/--
**No LG mode of order `(p, l)` is tighter than `w/√2`.**

The fundamental mode `(0, 0)` attains the bound; every higher order is wider.
-/
theorem lgRmsRadius_ge_fundamental {w p l : ℝ} (hw : 0 < w) (hp : 0 ≤ p) :
    w / Real.sqrt 2 ≤ lgRmsRadius w p l := by
  unfold lgRmsRadius
  have hroot : (1:ℝ) ≤ Real.sqrt (2 * p + |l| + 1) := by
    apply one_le_sqrt
    have hp2 : (0:ℝ) ≤ 2 * p := by positivity
    have hl2 : (0:ℝ) ≤ |l| := abs_nonneg l
    linarith
  have hne : (0:ℝ) < Real.sqrt 2 := Real.sqrt_pos.2 (by norm_num)
  have hwne : (0:ℝ) < w / Real.sqrt 2 := div_pos hw hne
  have hmul := mul_le_mul_of_nonneg_right hroot (le_of_lt hwne)
  nlinarith [hmul]

/-- The fundamental mode `(p, l) = (0, 0)` attains the bound. -/
theorem lgRmsRadius_fundamental (w : ℝ) (hw : 0 < w) :
    lgRmsRadius w 0 0 = w / Real.sqrt 2 := by
  unfold lgRmsRadius
  rw [abs_zero]
  have : Real.sqrt (2 * (0:ℝ) + 0 + 1) = 1 := by norm_num
  rw [this, mul_one]

/-! ## Gouy phase -/

/--
The Gouy phase `order·arctan(z/z_R)`, matching `gouy_phase`.

The coefficient `2p + |l| + 1` is the same bracket that sets the mode size, which
is why higher-order modes accumulate phase faster along the beam.
-/
noncomputable def gouyPhase (order z zR : ℝ) : ℝ := order * Real.arctan (z / zR)

theorem gouyPhase_eq (order z zR : ℝ) :
    gouyPhase order z zR = order * Real.arctan (z / zR) := rfl

/-- **The Gouy phase is zero at the waist**, where the beam is planar. -/
theorem gouyPhase_zero (order zR : ℝ) : gouyPhase order 0 zR = 0 := by
  unfold gouyPhase
  rw [show (0:ℝ) / zR = 0 by simp, Real.arctan_zero, mul_zero]

/-- **The Gouy phase advances monotonically along the beam** for positive order:
the axial phase rotation distinguishing a focused beam from a plane wave. -/
theorem gouyPhase_mono {order z z' zR : ℝ} (ho : 0 < order) (hR : 0 < zR)
    (h : z ≤ z') :
    gouyPhase order z zR ≤ gouyPhase order z' zR := by
  unfold gouyPhase
  have hatan : Real.arctan (z / zR) ≤ Real.arctan (z' / zR) := by
    apply Real.arctan_mono
    exact div_le_div_of_nonneg_right h hR.le
  nlinarith [ho]

/-- A negative order runs the phase backwards: the same rotation seen from the
other direction. -/
theorem gouyPhase_neg_order (order z zR : ℝ) :
    gouyPhase (-order) z zR = -(gouyPhase order z zR) := by
  simp [gouyPhase]

end Optics
end Photonics