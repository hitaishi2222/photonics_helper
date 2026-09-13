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

| Reproduction | Reference | Module stack | Status |
|---|---|---|---|
| [`stolen_lin_1978_spm`](stolen_lin_1978_spm/) | Stolen & Lin, *Phys. Rev. A* **17**, 1448 (1978) | GNLSE (Kerr) + pulse | ✅ spectrum matches closed form to ~1e-13; peak-count rule exact |
| [`macleod_quarter_wave_dbr`](macleod_quarter_wave_dbr/) | Macleod, *Thin-Film Optical Filters*; Born & Wolf | `dbr` TMM | ✅ peak reflectance matches exact closed form; stopband width within 7% |
| [`gordon_1986_ssfs`](gordon_1986_ssfs/) | Gordon, *Opt. Lett.* **11**, 662 (1986); Mitschke & Mollenauer (1986) | GNLSE + Raman + soliton | ✅ measured +1.08 nm / 20 m vs Gordon +0.91 nm (ratio 1.19); Stokes gain, red-shift |
| [`dudley_2006_cherenkov_dw`](dudley_2006_cherenkov_dw/) | Akhmediev & Karlsson, *Phys. Rev. A* **51**, 2602 (1995); PCF params from Dudley et al. Fig. 3 config | phase_matching + GNLSE | ✅ `dispersive_wave_roots` = analytic 699.3 nm; GNLSE DW peak 702.3 nm (0.44%). *Physics validated; not a digitized Fig. 3 reproduction.* |

## Findings surfaced while reproducing (see `../REPORT.md`)

Reproducing papers is also a bug hunt. The Gordon reproduction exposed **two
fundamental GNLSE bugs**, both fixed:

- **Inverted dispersion sign** — `_linear_step` formed a bright soliton for
  β₂ > 0 (and inverted the group delay). Fixed to the standard convention
  (β₂ < 0 → soliton).
- **Raman response direction** — the delayed convolution amplified the
  anti-Stokes sideband; corrected to give Stokes gain and a red-shifting soliton.

Still open: **MI gain √2 convention**, **FROG chirped-pulse retrieval**,
**TMM exit medium (air assumed)**, and the **self-steepening RK4 energy drift**
(≈5% alone, ≈12% with Raman; the Raman term itself conserves energy exactly).
