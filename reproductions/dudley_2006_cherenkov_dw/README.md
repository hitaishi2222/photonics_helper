# Reproduction — Cherenkov dispersive wave (Dudley et al., 2006)

**Reference:** J. M. Dudley, G. Genty, S. Coen, "Supercontinuum generation in
photonic crystal fiber," *Rev. Mod. Phys.* **78**, 1135 (2006) — the PCF
parameter set is the **Fig. 3 configuration** (835 nm, β₂ = −11.83e-3 ps²/m,
β₃ = 8.10e-5 ps³/m, γ = 0.11 /W/m, 10 kW, 15 cm), as also distributed with
laserfun's `NLSE_dudley`. Theory: N. Akhmediev and M. Karlsson, *Phys. Rev. A*
**51**, 2602 (1995).

> **Scope / honesty note:** this validates the **analytic Cherenkov
dispersive-wave physics** (phase matching, `dispersive_wave_roots`, and the
GNLSE DW peak) using the Dudley Fig. 3 parameter set. It is **not** a
digitized, figure-for-figure reproduction of Dudley et al. Fig. 3 — that would
require extracting the spectral evolution from the paper and validating the
full SCG (soliton fission + Raman + self-steepening). The full configuration is
available in `examples/21_gnlse_nlse_dudley.py`.

## Result reproduced

A soliton sheds a dispersive wave where the linear phase mismatch vanishes. For a
PCF with β₂ < 0 and β₃ > 0 the lowest-order root is

```
Ω_DW = −3 β₂ / β₃        → λ_DW ≈ 699 nm for the Dudley PCF
```

## Ground truth

The analytic root above, plus the full-β(ω) root found by
`phase_matching.dispersive_wave_roots`.

## Outcome

```
analytic -3β₂/β₃        = 699.3 nm
dispersive_wave_roots   = 699.3 nm   (exact)
GNLSE blue DW peak      = 702.3 nm   (0.44% from analytic)
```

Figure: `cherenkov_dw.png`.

This reproduction also exercises the **corrected GNLSE dispersion convention**
(β₂ < 0 → soliton), without which no stable soliton — and hence no Cherenkov
radiation — forms.
