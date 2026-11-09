---
quadrant: reference
---

# Spectral span metrics

Bandwidth-at-level ("N octaves at −M dB") with the numeraire, the level and the
interpolation rule carried in the result. Docstrings are the source of truth,
rendered with [mkdocstrings](https://mkdocstrings.github.io/) (numpydoc style).

Every group that quotes a supercontinuum span writes it slightly differently:
which density (per Hz or per nm), which level reference, and how the threshold
crossing is interpolated. `span_at_level` makes all three arguments and returns
them, so the number can be checked instead of trusted.

The one figure worth remembering: on a wavelength axis the density is
`S_λ = S_ν·|dω/dλ| = S_ν·2πc/λ²`. Skipping that Jacobian moves a −50 dB
crossing by up to `10·log10((9/2)²) ≈ 13 dB` between 2 µm and 9 µm — i.e. it
changes the octave count. `numeraire="frequency"` applies the Jacobian;
`numeraire="wavelength"` says the spectrum already carries it.
`to_wavelength_density` is the same Jacobian the GNLSE resamplers use, so a span
computed here and a spectrum plotted there cannot disagree.

The octave count is `log2(λ_long / λ_short)`; a crossing the axis does not reach
is reported as `nan`, never guessed or clipped to the axis edge. The answer is
resolution-dependent by construction — two grids over one field can differ by
tenths of an octave — so `bin_nm` travels with the result.

::: photonics_helper.metrics
