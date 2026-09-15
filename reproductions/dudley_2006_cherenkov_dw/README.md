# Reproduction — Cherenkov dispersive wave (Dudley et al., 2006)

**Reference:** J. M. Dudley, G. Genty, S. Coen, "Supercontinuum generation in
photonic crystal fibers," *Rev. Mod. Phys.* **78**, 1135 (2006) — the PCF
parameter set is the **Fig. 3 configuration** (835 nm, β₂ = −11.83e-3 ps²/m,
β₃ = 8.10e-5 ps³/m, γ = 0.11 /W/m, 10 kW, 15 cm), as also distributed with
laserfun's `NLSE_dudley`.
**DOI:** [10.1103/RevModPhys.78.1135](https://doi.org/10.1103/RevModPhys.78.1135)
Theory: N. Akhmediev and M. Karlsson, "Cherenkov radiation emitted by solitons
in optical fibers," *Phys. Rev. A* **51**, 2602 (1995),
[10.1103/PhysRevA.51.2602](https://doi.org/10.1103/PhysRevA.51.2602).

> **See also** [`../dudley_2006_scg/`](../dudley_2006_scg/) for the full set of
> figure-by-figure Dudley reproductions (Figs. 3, 4, 5, 6, 7, 8, 9, 10, 23),
> including the Kodama–Hasegawa fission check and the full-β dispersive-wave
> phase matching used below.

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

## Findings

- The root finder `dispersive_wave_roots` returns the analytic ``−3β₂/β₃``
  value **exactly** (699.3 nm) for the two-term dispersion used here.
- The GNLSE blue DW peak (702.3 nm) is within 0.44 % of the analytic root,
  confirming that the sign/scale conventions of the dispersion operator are
  consistent with the phase-matching formula.
- This case is the *linear* (β₂, β₃ only) limit. The new
  `dudley_2006_scg/fig08_dispersive_wave.py` shows that adding the higher-order
  Table I terms and the nonlinear `(1−f_R)γP_s` term moves the root from 699 nm
  to ≈ 663 nm and the simulated DW to 645 nm — i.e. this reproduction is a
  special case, not the final word on Dudley's DW.

## ISSUES

- **Two-term dispersion only.** The Fig. 3 PCF has β₄…β₁₀ terms that shift the
  DW by ~5 % (see above); this script intentionally validates the textbook
  `−3β₂/β₃` limit but should not be quoted as the paper's DW wavelength.
- **Output peak matching is heuristic.** `validate` takes the strongest blue
  peak within 20 nm of the pump-conjugate side; for a different grid/step count
  the DW fine structure changes (the GNLSE DW has SPM sidebands), so the 5 %
  tolerance is deliberately loose.
- **No Raman/self-steepening** in this run, matching the Akhmediev–Karlsson
  analytic condition but not the full Fig. 3 simulation.
