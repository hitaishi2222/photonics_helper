# Reproduction — optical wave breaking (Tomlinson, Stolen & Johnson, 1985)

**Reference:** W. J. Tomlinson, R. H. Stolen, A. M. Johnson, "Optical wave
breaking of pulses in nonlinear optical fibers," *Optics Letters* **10**, 457
(1985).
**DOI:** [10.1364/OL.10.000457](https://doi.org/10.1364/OL.10.000457)

> **PDF note:** this paper is paywalled and could not be downloaded here, so no
> `paper_pages/` are included. The analytic criterion below is derived from the
> standard SPM chirp (Agrawal, *Nonlinear Fiber Optics*, Sec. 4.1.3) and
> validated numerically. If the PDF is supplied, the paper's exact experimental
> parameters can be substituted.

## Result reproduced

In the **normal-dispersion** regime (β₂ > 0) self-phase modulation chirps the
pulse and group-velocity dispersion converts that chirp into a time shift. For a
Gaussian input `A(0,T) = √P₀ exp(−T²/2T₀²)` the SPM chirp after distance z is

```
δω(T) = −(2 γ P₀ z / T₀²) · T exp(−T²/T₀²),
```

so each spectral component is shifted to `T′ = T + β₂ z δω(T)`. Wave breaking is
where this map becomes non-monotonic, `1 + β₂ z ∂δω/∂T = 0`, giving

```
z_WB = T₀ / √(2 β₂ γ P₀) = (e^{3/4}/2) √(L_D L_NL),
L_D = T₀²/β₂,   L_NL = 1/(γP₀).
```

The pulse first acquires **steep (shock) edges**, then a **flat-topped
profile**, and finally **oscillations** a few `√(L_D L_NL)` later.

This reproduction validates three things with the GNLSE solver:

1. the **strengthening onset** — the edge steepness
   `S = max|∂I/∂T|·T₀ / I_peak` departs from the Gaussian value
   `√2 e^{−1/2} ≈ 0.858` near `z_WB` and peaks around `1.5√(L_D L_NL)`;
2. **oscillations** appear within a few `√(L_D L_NL)` of `z_WB`;
3. the **scaling law** `z_WB ∝ √(L_D L_NL) ∝ P₀^{−1/2}` (the decisive test that
   does not depend on the O(1) prefactor).

## Parameters (representative silica fibre, normal dispersion)

| Quantity | Value |
|---|---|
| Wavelength | 1060 nm |
| β₂ | +20 ps²/km (normal) |
| γ | 1.5 W⁻¹km⁻¹ |
| T₀ | 10 ps |
| P₀ | 10 W (swept 5–40 W) |
| Derived L_D / L_NL / √(L_D L_NL) / z_WB | 5000 m / 66.7 m / 577.4 m / 611 m |

## Outcome

```
steepening onset       348 m  (0.57 z_WB)
peak edge steepness    2.90× the Gaussian value
first oscillations     2212 m (3.83 √(L_D L_NL))
scaling                z_onset ∝ P₀^(−0.54)  (theory −0.5)
                       z_onset/√(L_D L_NL) = 0.63, 0.60, 0.59, 0.58 for
                       P₀ = 5, 10, 20, 40 W  (spread 1.09)
```

Figure: `wave_breaking.png` — (a) profiles developing steep edges and a flat
top, (b) edge steepness vs distance with `z_WB` marked, (c) `P₀^{−1/2}` scaling,
(d) SPM spectral broadening.

## Findings

- The GNLSE solver reproduces the full wave-breaking sequence: steep edges →
  flat top → oscillations, with the onset at `0.57 z_WB` and the steepness
  maximum at `~1.4 z_WB`, bracketing the analytic chirp-folding estimate.
- The **scaling exponent is the cleanest validation**: sweeping P₀ over 8×
  gives `z ∝ P₀^{−0.54}`, and `z/√(L_D L_NL)` is constant to 9 %, confirming
  `z_WB ∝ √(L_D L_NL)`.
- In a purely symmetric NLSE (no odd-order terms or self-steepening) the
  spectrum stays symmetric; the asymmetric spectral tail seen in some
  wave-breaking figures requires the shock/higher-order terms and is not
  claimed here.

## ISSUES

- The Opt. Lett. **PDF was not available**, so the parameter set is
  representative rather than the paper's exact experimental values, and the
  analytic prefactor `(e^{3/4}/2) ≈ 1.06` is derived here rather than quoted
  from the paper. The measured onset (`0.57 z_WB`) and steepness maximum
  (`~1.4 z_WB`) bracket it, but no sub-percent agreement with the paper's
  reported distances is claimed.
- The "steepest edge" diagnostic becomes unreliable once fine oscillations
  develop (grid-limited gradients), so the reproduction validates the onset and
  the scaling rather than the peak-steepness distance at large z.
- Normal dispersion is required; the routine is not valid for β₂ ≤ 0.
