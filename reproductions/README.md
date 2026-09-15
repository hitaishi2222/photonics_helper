# Paper reproductions

Self-contained, validated reproductions of results from the literature. Each
folder contains:

- `parameters.json` — every physical input (documented, reproducible).
- `reproduce.py` — the script (`python .../reproduce.py`), which asserts its
  result against an analytic/closed-form reference and writes a figure PNG.
- `README.md` — what was reproduced, the ground truth, and the outcome.

Run all reproduction tests with:

```bash
python -m pytest tests/test_reproductions.py
```

## Available

| Reproduction | Reference (DOI) | Module stack | Status |
|---|---|---|---|
| [`stolen_lin_1978_spm`](stolen_lin_1978_spm/) | Stolen & Lin, *Phys. Rev. A* **17**, 1448 (1978) · [10.1103/PhysRevA.17.1448](https://doi.org/10.1103/PhysRevA.17.1448) | GNLSE (Kerr) + pulse | ✅ spectrum matches closed form to ~1e-13; peak-count rule exact |
| [`macleod_quarter_wave_dbr`](macleod_quarter_wave_dbr/) | Macleod, *Thin-Film Optical Filters*; Born & Wolf (textbook) | `dbr` TMM | ✅ peak reflectance matches exact closed form to <1e-6; stopband width within 7% |
| [`gordon_1986_ssfs`](gordon_1986_ssfs/) | Gordon, *Opt. Lett.* **11**, 662 (1986) · [10.1364/OL.11.000662](https://doi.org/10.1364/OL.11.000662) | GNLSE + Raman + soliton | ✅ measured +1.08 nm / 20 m vs Gordon +0.91 nm (ratio 1.19); Stokes gain, red-shift |
| [`dudley_2006_cherenkov_dw`](dudley_2006_cherenkov_dw/) | Akhmediev & Karlsson, *Phys. Rev. A* **51**, 2602 (1995) · [10.1103/PhysRevA.51.2602](https://doi.org/10.1103/PhysRevA.51.2602) | phase_matching + GNLSE | ✅ `dispersive_wave_roots` = analytic 699.3 nm; GNLSE DW peak 702.3 nm (0.44%). *Linear two-term limit.* |
| [`dudley_2006_scg`](dudley_2006_scg/) | Dudley, Genty & Coen, *Rev. Mod. Phys.* **78**, 1135 (2006) · [10.1103/RevModPhys.78.1135](https://doi.org/10.1103/RevModPhys.78.1135) | GNLSE + Raman + phase_matching + soliton | ✅ **9 figure scripts** (Figs. 3, 4, 5, 6, 7, 8, 9, 10, 23); Kodama–Hasegawa fission to 4%, DW phase matching to 2.6%, `z_sol` periodicity 0.998 |

## Findings surfaced while reproducing (see `../REPORT.md`)

Reproducing papers is also a bug hunt. The Gordon reproduction exposed **two
fundamental GNLSE bugs**, both fixed:

- **Inverted dispersion sign** — `_linear_step` formed a bright soliton for
  β₂ > 0 (and inverted the group delay). Fixed to the standard convention
  (β₂ < 0 → soliton).
- **Raman response direction** — the delayed convolution amplified the
  anti-Stokes sideband; corrected to give Stokes gain and a red-shifting soliton.

Still open: the **self-steepening RK4 energy drift** (≈5% alone, ≈12% with
Raman; the Raman term itself conserves energy exactly).

The **FROG chirped-pulse retrieval** is now resolved: `retrieve()` uses the
principal-component projection over all delays with seeded random restarts, so
chirped pulses retrieve at high fidelity across window sizes.

The **MI gain convention** is now resolved: `mi_gain_spectrum`,
`mi_gain_spectrum_extended`, and `mi_sideband_frequencies` use the exact
linear-stability result Ω_c² = 4γP/|β₂| (g_max = 2γP at Ω = √(2γP/|β₂|)).

## Findings across reproductions

- **Analytic ground truth is decisive.** Where a closed form exists (SPM
  Fourier integral, quarter-wave DBR, Gordon SSFS, Cherenkov root,
  Kodama–Hasegawa solitons, MI gain) the reproductions match it to
  `1e-13`–few %. Solver bugs that survived the unit tests (inverted dispersion
  sign, Raman sideband direction, DBR matrix order, `_estimate_beta2`) were all
  exposed only by the literature comparisons.
- **Grid/step convergence matters more than expected.** Raman fission and full
  SCG drift by tens of percent if `dt > ~3 fs` or if the self-steepening step is
  under-resolved. The scripts that depend on this (Dudley Figs. 3/6) pin the
  grid explicitly.
- **The Taylor reconstruction reproduces the paper's ZDW.** From the Table I
  coefficients alone, β₂(λ) = 0 at 779.9 nm, matching Dudley's Fig. 2 and
  enabling the MI-gain figure without digitising the dispersion curve.

## ISSUES (cross-cutting)

1. **Self-steepening time scale.** The engine has no `τ_shock` override, so
   Dudley's effective-area-corrected 0.56 fs cannot be used; the default is
   1/ω₀ = 0.443 fs (~26 % smaller). High-order SCG bandwidths are therefore
   approximate. See `REPORT.md` (RK4 energy drift).
2. **No quantum/shot noise.** The Dudley coherence figures (18, 20–22) are not
   reproducible without a stochastic Raman source; the deterministic
   reproductions are single-shot only.
3. **Full GVD curves are not digitised.** Where a paper's figure needs β(ω)
   across wavelengths (Dudley Fig. 23), this repo uses a Taylor
   reconstruction; far-from-835-nm results are qualitative.
4. **Reproductions validate physics, not pixels.** No figure is digitised
   curve-for-curve; tolerances and caveats are stated in each folder's README
   rather than hidden.
