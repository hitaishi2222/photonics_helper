---
quadrant: reference
---

# Noise sources

Amplitude-contrast, spectral-level and quantum shot-noise seeding. The broadband one-photon-per-mode seed (`opm_noise_field` / `add_opm_noise`) injects a full `ℏω` quantum per mode once at the input; `raman_noise_field` is a different source, the per-step Raman Langevin term, which uses the `ℏω/2` vacuum-fluctuation convention (a 3 dB difference in variance). Docstrings are the source of truth, rendered with
[mkdocstrings](https://mkdocstrings.github.io/) (numpydoc style).

::: photonics_helper.noise
