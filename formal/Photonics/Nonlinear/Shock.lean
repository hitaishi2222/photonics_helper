import Photonics.Dispersion.Beta

/-!
# Self-steepening: the validity regime of the shock operator

The shock operator `(1 + (i/ω₀)∂_t)` has symbol `1 + Ω·τ_shock`, which with
`τ_shock = 1/ω₀` **is** `ω/ω₀ = (ω₀ + Ω)/ω₀` bin-by-bin. There is no Taylor
expansion in this realization: `shockRatio x = 1 + x` and the solver applies
exactly that (`code_symbol_is_exact`). This was corrected on 2026-10-27 — an
earlier draft of this file identified the ratio as `1/(1 − x)` and derived a
"first-order truncation error" `x²/(1 − x)` against the wrong exact quantity.
The reciprocal theorems are kept below, relabelled, because the guard's history
(and CONFORMANCE rows D3–D6) refers to them; they bound an error that does not
occur here.

What the guard *is* about, per the primary source:

- `Ω_max < ω₀` — the hard *positivity* condition: the symbol `ω/ω₀` is only
  meaningful where the absolute frequency `ω = ω₀ + Ω > 0`. This is the
  existence condition and the code raises on violation.
- `τ_shock·Ω_max ≤ 0.2` — the *derivation regime* condition. Blow & Wood
  derive the GNLS assuming "the bandwidth of U(ω') is less than ω₀/3"; the
  guard keeps the simulated window at `2x·ω₀ ≤ 2ω₀/5` (see
  `window_of_guard`). Note `2/5 > 1/3`: strict adherence to the ω₀/3
  assumption would need `x < 1/6` (`bw_regime_needs_sharper`). The code keeps
  `0.2`; the difference is recorded in CONFORMANCE F11 rather than silently
  absorbed.

Everything here is real-valued.
-/

namespace Photonics
namespace Nonlinear

open Conventions

/-! ## The expansion itself -/

/--
The frequency ratio in the retarded frame, `ω/ω₀`, as a function of the
dimensionless offset `x = Ω/ω₀`.

Since `ω = ω₀ + Ω`, this is `1 + x` — exactly, with no expansion.
-/
noncomputable def shockRatio (x : ℝ) : ℝ := 1 + x

theorem shockRatio_eq (x : ℝ) : shockRatio x = 1 + x := rfl

/-- The quotient form: with `x = Ω/ω₀`, `(ω₀ + Ω)/ω₀ = 1 + Ω/ω₀`. The symbol the
solver applies per bin is the frequency ratio itself. -/
theorem shockRatio_quotient {ω0 Ω : ℝ} (hω0 : 0 < ω0) :
    (ω0 + Ω) / ω0 = 1 + Ω / ω0 :=
  by field_simp

/-- Positivity: the symbol is positive exactly where the absolute frequency is,
`ω₀ + Ω > 0`, i.e. `x > −1`. On the code's symmetric grid `Ω ∈ (−ω₀, ω₀)`, the
guard region `x ≤ 1/5` is comfortably inside. -/
theorem shockRatio_pos (x : ℝ) (h : -1 < x) : 0 < shockRatio x :=
  by linarith [shockRatio_eq x]

/--
The first-order (and, here, exact) form used by the solver is `1 + x`.

The code's comment describes it as `1 + Ω·τ_shock`, which is this with
`τ_shock = 1/ω₀` and `x = Ω/ω₀`.
-/
noncomputable def shockFirstOrder (x : ℝ) : ℝ := 1 + x

theorem shockFirstOrder_eq (x : ℝ) : shockFirstOrder x = 1 + x := rfl

/--
**The solver's shock symbol is the exact frequency ratio — no truncation.**

`ω/ω₀ = (ω₀ + Ω)/ω₀ = 1 + Ω/ω₀ = 1 + x` is an identity, not an approximation,
so the operator the code applies in the frequency domain has no first-order
truncation error against the paper's `ω/ω₀`. (The piece the code *does* drop is
the material dispersion of `N/n_eff` in Blow & Wood's `τ_shock` — "the frequency
dependence of the coefficient `n_eff/N` is extremely weak and, to a good
approximation, this dependence can be neglected" — a different, stated
approximation, not a Taylor remainder of `ω/ω₀`.)
-/
theorem code_symbol_is_exact (x : ℝ) : shockRatio x = shockFirstOrder x := rfl

/--
The reciprocal ratio `ω₀/(ω₀ − Ω) = 1/(1 − x)` — **not** the code's operator.

An early draft of this file identified this as "the" frequency ratio and
derived a truncation error against it. Kept (and relabelled) because the guard's
history refers to it: the theorems below bound the error a reciprocal
realization *would* have, which this codebase does not exhibit.
-/
noncomputable def reciprocalRatio (x : ℝ) : ℝ := 1 / (1 - x)

theorem reciprocalRatio_eq (x : ℝ) : reciprocalRatio x = 1 / (1 - x) := rfl

/--
The error a *reciprocal* realization would incur against the `1 + x` symbol:
`1/(1−x) − (1+x) = x²/(1−x)`, exactly, for `x ≠ 1`.

**Counterfactual, not a description of this code.** See `code_symbol_is_exact`.
-/
theorem shock_truncation_error (x : ℝ) (h : x ≠ 1) :
    reciprocalRatio x - shockFirstOrder x = x ^ 2 / (1 - x) := by
  unfold reciprocalRatio shockFirstOrder
  field_simp
  ring

/-- For a positive offset a reciprocal realization would *under*-estimate its
true ratio (counterfactual; kept for the register's history). -/
theorem shock_error_nonneg (x : ℝ) (h0 : 0 ≤ x) (h1 : x < 1) :
    0 ≤ reciprocalRatio x - shockFirstOrder x := by
  rw [shock_truncation_error x (ne_of_lt h1)]
  have h1' : 0 < 1 - x := by linarith
  have h2 : 0 ≤ x ^ 2 := sq_nonneg x
  exact div_nonneg h2 (le_of_lt h1')

/--
**`_SHOCK_TAYLOR_LIMIT = 0.2` measured against the counterfactual: at `x = 1/5`
the reciprocal error would be exactly `1/20`.**
-/
theorem shock_error_at_limit :
    reciprocalRatio (1 / 5 : ℝ) - shockFirstOrder (1 / 5 : ℝ) = 1 / 20 := by
  rw [shock_truncation_error _ (by norm_num)]
  norm_num

theorem shock_error_bounded {x : ℝ} (h0 : 0 ≤ x) (h1 : x ≤ 1 / 5) :
    reciprocalRatio x - shockFirstOrder x ≤ 1 / 20 := by
  have hlt : x < 1 := lt_of_le_of_lt h1 (by norm_num)
  rw [shock_truncation_error x (ne_of_lt hlt)]
  have hmonotone : x ^ 2 / (1 - x) ≤ (1 / 5 : ℝ) ^ 2 / (1 - 1 / 5) := by
    have hd1 : 0 < 1 - x := by linarith
    have hd2 : (0:ℝ) < 1 - 1/5 := by norm_num
    rw [div_le_div_iff₀ hd1 hd2]
    · have hx : x ^ 2 ≤ (1 / 5 : ℝ) ^ 2 := by nlinarith
      linarith
  rw [show (1 / 5 : ℝ) ^ 2 / (1 - 1 / 5) = 1 / 20 by norm_num] at hmonotone
  exact hmonotone

/--
**The guard region implies the positivity (existence) condition.**

Any non-negative `x ≤ 1/5` satisfies `x < 1`, so a caller who respects
`_SHOCK_TAYLOR_LIMIT` has already satisfied `Ω_max < ω₀` (the bin-positivity
existence condition) for the *same* `ω₀` — `_validate_shock_grid` enforces
`Ω_max ≥ 0`, hence `h₀`. The converse fails: `x` can sit between `1/5` and
`1` and the symbol stays perfectly well defined — which is why the code
treats one as a warning and one as an error.
-/
theorem accuracy_implies_existence {x : ℝ} (h0 : 0 ≤ x) (h : x ≤ 1 / 5) :
    x < 1 := by
  have h5 : (1 : ℝ) > 1 / 5 := by norm_num
  linarith

theorem existence_does_not_imply_accuracy {x : ℝ} (h : 0.5 ≤ x) (h1 : x < 1) :
    ¬(x ≤ 1 / 5) := by
  intro h2
  have h5 : (1 / 5 : ℝ) < 0.5 := by norm_num
  have := lt_of_le_of_lt h2 h5
  linarith [h]

/-! ## The derivation-regime (bandwidth) conditions

Blow & Wood derive the GNLS assuming the envelope bandwidth is less than
`ω₀/3`. In terms of the dimensionless offset, the guard `x ≤ 1/5` bounds the
simulated *window* (edge to edge) by `2x·ω₀`. -/

/-- The guard bounds the simulated band edge: `ω₀ + Ω_max = ω₀(1 + x)` with
`x ≤ 1/5`, so the edge sits at most at `6ω₀/5`. -/
theorem guard_band_edge (x : ℝ) (h : x ≤ 1 / 5) : x + 1 ≤ 6 / 5 := by linarith

/-- The guard bounds the full window by `2ω₀/5`. -/
theorem window_of_guard (x : ℝ) (h : x ≤ 1 / 5) : 2 * x ≤ 2 / 5 := by linarith

/-- Blow & Wood's derivation regime `bandwidth < ω₀/3` in terms of the offset:
the full window `2x` must stay under `1/3`, i.e. `x < 1/6`. The code's guard
`1/5` sits outside that; recorded, not silently absorbed (CONFORMANCE F11). -/
theorem bw_regime_needs_sharper (x : ℝ) (h : 2 * x < 1 / 3) : x < 1 / 6 := by
  linarith

/-
## The resolution floor

With the default `τ_shock = 1/ω₀` and the derivation-regime guard, the grid
spacing `dt` is pinned from below. `gnlse.py` states this as "about 12.9 fs at
1550 nm"; this is the theorem behind that number.
-/

/-- With `τ_shock = 1/ω₀`, the dimensionless offset at the grid edge is
`Ω_max/ω₀ = Ω_max·dt/π`, by the Nyquist identity `Grid.omega_max_mul_dt`. -/
theorem offset_at_grid_edge {ω0 Ωmax dt : ℝ} (hω0 : 0 < ω0) (hdt : 0 < dt)
    (hNyq : Ωmax * dt = Real.pi) :
    Ωmax / ω0 = (Ωmax * dt / Real.pi) * (Real.pi / (ω0 * dt)) := by
  have hpi : Real.pi ≠ 0 := by norm_num
  field_simp

/--
**The resolution floor.** If the derivation-regime condition
`τ_shock·Ω_max ≤ limit` holds with `τ_shock = 1/ω₀`, then
`Ω_max ≤ limit·ω₀`, and by `Ω_max·dt = π` the grid spacing satisfies
`dt ≥ π/(limit·ω₀)`.

At 1550 nm with `ω₀ = 2πc/λ` this evaluates to about 12.9 fs, matching the
code's comment. The consequence recorded there also follows: a femtosecond
pulse cannot be resolved inside the shock model's derivation regime, which is
why `reproductions/hult_2007_rk4ip/` disables self-steepening.
-/
theorem resolution_floor {ω0 dt limit Ωmax : ℝ} (hω0 : 0 < ω0) (hdt : 0 < dt)
    (hlim : 0 < limit) (hNyq : Ωmax * dt = Real.pi) (hcond : (1 / ω0) * Ωmax ≤ limit) :
    dt ≥ Real.pi / (limit * ω0) := by
  have h1 : Ωmax ≤ limit * ω0 := by
    have hmul : (1 / ω0) * Ωmax * ω0 = Ωmax := by field_simp
    nlinarith [hcond, hmul]
  have hden : 0 < limit * ω0 := mul_pos hlim hω0
  have h4 : Real.pi ≤ (limit * ω0) * dt := by
    nlinarith [hNyq, h1]
  refine (div_le_iff₀ hden).mpr ?_
  nlinarith [h4]


open Complex
open scoped ComplexConjugate

/-! ## The sign convention

`gnlse.py` uses the shock factor `1 + Ω·τ_shock`. `Dudley, Genty & Coen` Eq. (2),
p. 1144, writes `1 + iτ_shock ∂_∂T` — a **plus** sign. `Blow & Wood` Eq. (18)
writes `1 − iτ_shock ∂_∂t` — apparently the opposite. Both were read directly
from the PDFs in `formal/refs/` during the §17 audit.

**They agree, and the difference is purely the Fourier convention.** The shock
factor is the *symbol* of a differential operator, and the symbol of `∂_t`
depends on which exponential the forward transform uses:

- `A(t) = ∫ Ã(Ω) e^{-iΩt} dΩ` (the codebase's convention — the grid FFT uses
  `e^{-iΩt}` and bins map to `ω = ω₀ + Ω`) gives `∂_t ↦ -iΩ`, hence
  `1 + iτ∂_t ↦ 1 + τΩ`.
- the opposite convention gives `∂_t ↦ +iΩ`, hence `1 - τΩ`.

`convention_difference` below shows the two symbols differ by **exactly `2τΩ`**
and by nothing else. That is the entire content of the sign audit that
`gnlse.py`'s docstring argues numerically at length — now a theorem.

`shock_skews_blue` is the physical consequence the audit also checks: for
`τ > 0` and `Ω > 0` the factor exceeds `1`, i.e. **the blue side is amplified**,
which is the skew the code's sign audit reports observing.
-/
/-- Symbol of `∂_t` on a bin at offset `Ω`, for the convention
`Ã(Ω) = ∫ A(t) e^{+iΩt} dt` (inverse `A(t) = ∫ Ã e^{-iΩt} dΩ`) — the forward-FFT
convention `core/grids.py` uses, whose bins map to `ω = ω₀ + Ω`. -/
noncomputable def derivSymbolNeg (O : ℝ) : ℂ := -(I * (O : ℂ))

/-- Symbol of `∂_t` under the opposite convention, `A(t) = ∫ Ã(Ω) e^{+iΩt} dΩ`. -/
noncomputable def derivSymbolPos (O : ℝ) : ℂ := I * (O : ℂ)

theorem derivSymbolNeg_eq (O : ℝ) : derivSymbolNeg O = -(I * (O : ℂ)) := rfl
theorem derivSymbolPos_eq (O : ℝ) : derivSymbolPos O = I * (O : ℂ) := rfl

/-- `I · (−I · z) = z`. This single identity is the whole sign flip: it is what
turns the `+Ω` of one Fourier convention into the `−Ω` of the other. -/
theorem I_mul_negI (z : ℂ) : I * (-(I * z)) = z := by
  rw [show I * (-(I * z)) = -(I * (I * z)) from by ring, ← mul_assoc, Complex.I_mul_I]
  ring

theorem shock_factor_neg (tau O : ℝ) :
    1 + I * (tau : ℂ) * derivSymbolNeg O = 1 + tau * (O : ℂ) := by
  simp only [derivSymbolNeg_eq]
  rw [show (I * (tau : ℂ)) * -(I * (O : ℂ)) = (tau : ℂ) * (I * -(I * (O : ℂ))) from by ring,
      I_mul_negI]

/-- `I · (I · z) = −z`, the mirror of `I_mul_negI`. -/
theorem I_mul_I_mul (z : ℂ) : I * (I * z) = -z := by
  rw [← mul_assoc, Complex.I_mul_I]
  simp

theorem shock_factor_pos (tau O : ℝ) :
    1 + I * (tau : ℂ) * derivSymbolPos O = 1 - tau * (O : ℂ) := by
  simp only [derivSymbolPos_eq]
  rw [show (I * (tau : ℂ)) * (I * (O : ℂ)) = (tau : ℂ) * (I * (I * (O : ℂ))) from by ring,
      I_mul_I_mul]
  ring

/-- The two conventions differ by exactly `2τΩ`. -/
theorem convention_difference (tau O : ℝ) :
    (1 + I * (tau : ℂ) * derivSymbolNeg O) - (1 + I * (tau : ℂ) * derivSymbolPos O)
      = 2 * (tau * O : ℝ) := by
  rw [shock_factor_neg, shock_factor_pos]
  push_cast
  ring

/-- Self-steepening amplifies the blue side under this codebase's convention. -/
theorem shock_skews_blue (tau O : ℝ) (ht : 0 < tau) (hO : 0 < O) :
    1 < 1 + tau * O := by nlinarith

theorem omega_offset_pos (w0 O : ℝ) (_h0 : 0 < w0) (hO : 0 < O) : w0 < w0 + O := by linarith


/-! ## Raman

Causality and the gain bound live in `Nonlinear.Raman`, which imports this file.
-/

/-- The delay length itself is a non-negative number of samples whenever the
delay is non-negative, which is the regime the Raman term is used in. -/
theorem ramanDelaySamples (tau dt : ℝ) (hτ : 0 ≤ tau) (hdt : 0 < dt) :
    0 ≤ tau / dt := by
  have h2 : 0 ≤ tau / dt := div_nonneg hτ (le_of_lt hdt)
  exact h2

end Nonlinear
end Photonics
