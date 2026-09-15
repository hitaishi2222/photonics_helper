# Reproduction — Raman soliton self-frequency shift (Gordon, 1986)

**Reference:** J. P. Gordon, "Theory of the soliton self-frequency shift,"
*Opt. Lett.* **11**, 662 (1986); experiment: F. M. Mitschke and
L. F. Mollenauer, *Opt. Lett.* **11**, 659 (1986).
**DOI:** [10.1364/OL.11.000662](https://doi.org/10.1364/OL.11.000662)
(experiment: [10.1364/OL.11.000659](https://doi.org/10.1364/OL.11.000659))

## Result reproduced

A fundamental soliton in a silica fibre red-shifts because the delayed Raman
response transfers energy from the blue (leading) to the red (trailing) part of
the pulse. The analytic rate is

```
dΩ/dz = − 8·|β₂|·T_R / (15·T₀⁴),      T_R = f_R ∫ t·h_R(t) dt.
```

For λ = 1.5 µm, D = 15 ps/(nm·km) and a 250 fs soliton this is ≈ +0.9 nm / 20 m
(≈ one spectral width per 100 m, as stated in the paper).

## Ground truth

The analytic Gordon rate above, evaluated with the library's silica `h_R`.

## Outcome

```
measured redshift = +1.0793 nm @ 20 m
Gordon analytic   = +0.9083 nm   (ratio 1.19)
```

Also checked: the Stokes (red) Raman sideband gains while the anti-Stokes loses;
the Raman term is exactly energy-conserving.

Figure: `ssfs_spectrum.png`.

## Findings

- The measured redshift is **1.19×** the Gordon analytic rate over 20 m. The
  residual factor is consistent with the finite propagation distance and the
  perturbative nature of the analytic formula, and the *sign* (red shift) and
  the Stokes/anti-Stokes asymmetry are reproduced exactly.
- The Raman term is **exactly energy-conserving** in the engine (E ratio =
  1.00000 with steepening off), which is the key internal check that the
  delayed convolution is implemented as a norm-preserving product in frequency
  space.
- This reproduction exposed **two fundamental GNLSE bugs**, both fixed in the
  archived `fix-raman-ssfs` change:
  1. the engine's dispersion sign was inverted (bright soliton formed at β₂ > 0,
     and the group delay was inverted);
  2. the delayed Raman term amplified the anti-Stokes sideband instead of the
     Stokes sideband.
  Without those fixes the soliton did not form and the shift was ≈ 0 and blue.

## ISSUES

- The ratio 1.19 is within the reproduction's tolerance (0.5–2.0) but is not a
  sub-percent validation; the analytic Gordon rate is a perturbation result and
  the library uses a single-Lorentzian `h_R`, whereas Gordon's `T_R` is an
  integral over the full measured Raman cross section.
- The report's still-open **self-steepening RK4 energy drift** does not affect
  this run (steepening off), but it does affect high-order SCG runs.
