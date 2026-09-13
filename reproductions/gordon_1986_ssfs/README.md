# Reproduction — Raman soliton self-frequency shift (Gordon, 1986)

**Reference:** J. P. Gordon, "Theory of the soliton self-frequency shift,"
*Opt. Lett.* **11**, 662 (1986); experiment: F. M. Mitschke and
L. F. Mollenauer, *Opt. Lett.* **11**, 659 (1986).

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

## Note

This reproduction exposed **two fundamental GNLSE bugs**, both fixed in the
archived `fix-raman-ssfs` change:
1. the engine's dispersion sign was inverted (bright soliton formed at β₂ > 0);
2. the delayed Raman term amplified the anti-Stokes sideband.

Without those fixes the soliton did not form and the shift was ~0 and blue.
