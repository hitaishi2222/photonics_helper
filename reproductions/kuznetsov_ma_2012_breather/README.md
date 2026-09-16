# Reproduction — Kuznetsov–Ma soliton dynamics (Kibler et al., 2012)

**Reference:** B. Kibler, J. Fatome, C. Finot, G. Millot, G. Genty, B. Wetzel,
N. Akhmediev, F. Dias, J. M. Dudley, "Observation of Kuznetsov-Ma soliton
dynamics in optical fibre," *Scientific Reports* **2**, 463 (2012).
**DOI:** [10.1038/srep00463](https://doi.org/10.1038/srep00463)

## Result reproduced

The paper reports the first experimental confirmation of the **Kuznetsov–Ma
(KM) soliton**, the periodic-on-a-finite-background solution of the focusing
nonlinear Schrödinger equation. In dimensionless form (paper Eq. 1)

```
i ψ_ξ + (1/2) ψ_ττ + |ψ|² ψ = 0,
```

the general soliton-on-finite-background (SFB) solution is (paper Eq. 2)

```
ψ(ξ,τ) = e^{iξ} [ 1 + ( 2(1−2a) cosh(bξ) + i b sinh(bξ) )
                      / ( √(2a) cos(ντ) − cosh(bξ) ) ],
b = √(8a(1−2a)),   ν = 2√(1−2a).
```

For `a > 1/2` (the KM regime) `b → iB` and `ν → i/Δ`, giving the periodic KM
soliton; its minimum- and maximum-compression profiles are (paper Eqs. 4–5)

```
ψ_min(τ) = 1 + 2(2a−1) / ( √(2a) cosh(τ/Δ) + 1 ),
ψ_max(τ) = 1 − 2(2a−1) / ( √(2a) cosh(τ/Δ) − 1 ),
B = √(8a(2a−1)),   Δ = 1/(2√(2a−1)),   period Δξ = 2π/B.
```

The paper's SMF-28 mapping (Methods) is

```
A(z,T) = √P₀ ψ,   T = τ T₀,   z = ξ L_NL + z_p/2,
L_NL = 1/(γP₀),   T₀ = √(|β₂| L_NL),   z_p = L_NL · Δξ.
```

This reproduction builds the **exact** KM field at its minimum-intensity point
and propagates it for one full KM period with the library split-step GNLSE
solver in the pure-NLSE limit, then compares the result with the analytic
solution. A second run adds the paper's 0.2 dB/km fibre loss.

## Parameters (paper Methods, SMF-28)

| Quantity | Value |
|---|---|
| Wavelength | 1554.9 nm |
| β₂ | −21.8 ps²/km |
| β₃ | 0.012 ps³/km |
| γ | 1.3 W⁻¹km⁻¹ |
| Loss | 0.2 dB/km |
| Background power P₀ | 0.7 W |
| KM parameter `a` | 0.66 |
| Derived L_NL / T₀ / z_p | 1098.9 m / 4.894 ps / 5.312 km |

## Ground truth

The exact SFB solution above. Independent checks used while implementing:

- Eq. (2) was verified to solve Eq. (1) by finite differences
  (residual ≈ 2×10⁻⁶; the sign of the `sin` term in the extracted Eq. 3 is a
  text-extraction artefact — the correct sign is **−**).
- The reference values quoted by the paper: initial centre power **1.2 W**
  (background 0.7 W), peak-to-background ratio from Eq. (5).

## Outcome

```
T₀ = 4.894 ps,  L_NL = 1098.9 m,  z_p = 5.3119 km  (a = 0.66)
centre power: 1.1790 W (z=0) → 7.6129 W (z_p/2), analytic 7.6130 W
peak relative error         3.9e-05
max centre-power abs error  3.9e-03 W
temporal intensity L2 error 9.1e-04
spectrum L2 error (DC off)  2.1e-04
lossy (0.2 dB/km) peak      5.95 W  (21.9 % below the ideal KM curve)
```

Initial centre power 1.179 W ↔ paper's 1.2 W. The lossy compression peak lies
below the ideal curve, exactly as in the paper's Fig. 4b, where loss is the
dominant deviation between the ideal KM theory and the experiment/simulation.

Figure: `kuznetsov_ma_breather.png` (evolution, centre-power breathing vs
analytic, profiles at the two extrema). Paper page images in `paper_pages/`.

## Findings

- The library's split-step GNLSE engine reproduces the **exact** KM breathing to
  ~10⁻⁴ relative accuracy — a strong test of the dispersion and Kerr operators
  for a field with a **non-zero (CW) background**, a regime not exercised by the
  previous soliton/SCG reproductions.
- The KM soliton is the first SFB validated here whose amplitude evolves
  *periodically with distance* (contrast this with the Akhmediev breather, which
  grows and decays once).
- The lossy curve is a useful sanity check for the paper's experiment: 21.9 %
  peak reduction at `z_p/2`, consistent with the reported theory–experiment gap.

## ISSUES

- This reproduces the paper's **ideal KM theory and its numerical simulation**,
  not the fibre experiment itself (no experimental data are digitised). The
  paper's experimental input is a strongly modulated CW that approximates the KM
  soliton over a single modulation cycle; that approximation is described in the
  paper's Fig. 2 and is not reproduced here.
- The paper's numerical simulation also included β₃ and a 250 dB noise
  background from the EDFA; only β₃ and loss are treated here, and the exact-KM
  validation sets them to zero (the analytic solution is for pure NLSE).
- `GNLSESolver` has no fixed-step option, but its adaptive heuristic works here
  (and, contrary to an earlier note, for the CW field in the Närhi MI reproduction
  as well).
