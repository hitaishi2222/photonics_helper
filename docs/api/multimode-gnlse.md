# Multimode (few-mode) coupled GNLSE

Split-step Fourier solver for N coupled modal channels: per-mode dispersion
and group delay, LP/isotropic SPM-XPM coefficients, opt-in pump-driven
inter-modal FWM with angular-momentum gating. Docstrings are the source of
truth, rendered with [mkdocstrings](https://mkdocstrings.github.io/)
(numpydoc style). See [multimode GNLSE](../multimode-gnlse.md) for the
physics and references.

::: photonics_helper.multimode_gnlse

## OAM power budget (GAP A4)

`MultimodeSplitStepEngine.power_by_l()` returns `∫|A_m|²dt` summed per
`oam_l` class (W); `power_by_l_vs_z` returns `(z, labels, powers)` recorded at
every saved step of the last `propagate`. With the FWM gate off, each class
must be flat. A changing budget is an allowed exchange or a symmetry break,
not numerical "physics". Both require `oam_l` labels and raise otherwise.
