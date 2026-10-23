# Phase-matching diagnostics (χ⁽³⁾)

FWM detuning, modulation-instability gain, dispersive-wave roots, readiness reports. Docstrings are the source of truth, rendered with
[mkdocstrings](https://mkdocstrings.github.io/) (numpydoc style).

## Beat length

`beat_length(delta_n_eff, wavelength)` returns `λ/|Δn_eff|` as a `Length`;
`wavelength` may be a `Wavelength` or an SI float (metres). `Δn_eff = 0`
raises. Use it instead of hand-written `λ/Δn` expressions (GAP A5).

::: photonics_helper.phase_matching
