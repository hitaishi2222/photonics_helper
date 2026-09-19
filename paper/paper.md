---
title: "photonics-helper: a unit-safe foundation for photonics computation and validated nonlinear-optics modelling"
tags:
  - Python
  - photonics
  - nonlinear optics
  - supercontinuum generation
  - numerical simulation
  - materials data
authors:
  - name: Hitaishi V
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 19 September 2026
bibliography: paper.bib
---

# Summary

`photonics-helper` is a Python library for photonics, fiber optics and
nonlinear-optics computation. It is organised around an explicit **foundation
layer** — type-safe physical units and constants, temporal/spectral grids with a
documented FFT convention, and an auditable optical-material database — with the
physics and simulation tooling layered on top.

The foundation is deliberately small and dependency-light: importing
`photonics_helper.core` pulls in only NumPy, SciPy and Pydantic, and the package
itself is imported lazily so that `import photonics_helper` does not load a
plotting or web stack. This makes the library usable as a base for other tools,
not only as an end-user application.

On top of the core, the library provides a solver suite:

- **GNLSE propagation** (split-step, adaptive stepping, z-dependent dispersion,
  tapered waveguides) with Kerr, delayed Raman, self-steepening, two-photon
  absorption and a stochastic Raman noise source;
- **soliton and breather analysis** (soliton order, fission length, dispersive
  waves, Raman self-frequency shift, Kuznetsov–Ma / Peregrine / Akhmediev
  solutions);
- **Raman response and material data** (time- and frequency-domain responses,
  pumped Stoke/anti-Stokes explorers, 44 materials with provenance);
- **phase-matching diagnostics** for four-wave mixing, modulation instability
  and Cherenkov dispersive waves; **χ⁽²⁾ mixing** (SHG/SFG/DFG with
  quasi-phase-matching and modal-overlap coupling); **DBR/TMM** stacks;
  **FROG** trace generation and retrieval; **structured light** (Laguerre–Gaussian
  / OAM modes); and optical wave-breaking diagnostics.

Two properties distinguish the library from a collection of scripts. First,
**units are first-class**: physical quantities are constructed as typed objects
(`Wavelength(1550, "nm")`) and converted explicitly, which removes the
factor-of-1000 class of error that dominates photonics code. Second, the
library's trust model is **agreement with the literature** rather than agreement
with another code: `reproductions/` contains self-contained studies that
reproduce published results and assert them against closed-form references,
including supercontinuum generation, Raman soliton self-frequency shift,
Cherenkov dispersive waves, modulation instability, breathers and dispersive
broadening.

# Statement of need

Photonics research code is routinely written as one-off scripts in which
wavelengths are bare floats, dispersion signs and FFT scalings are re-derived
each time, and material parameters are copied without provenance. The result is
that *plumbing* — unit handling, grid conventions, data lookup — dominates the
effort, and silent errors are common and expensive. Two failure modes recur:
unit mistakes that produce physically plausible but wrong numbers, and
convention mistakes (dispersion sign, FFT normalisation, Raman response
direction) that survive unit tests and only appear when a result is compared
with an experiment or a published figure.

`photonics-helper` addresses this by making the plumbing a stable, documented,
tested foundation and by treating published results as the acceptance criterion.
Specifically it provides:

- **A frozen, dependency-light core** (`photonics_helper.core`) with a published
  [stability contract](https://hitaishi2222.github.io/photonics_helper/stability/),
  machine-checked by snapshot tests on symbols and signatures, so downstream
  projects can depend on it. A worked example builds an external tool on the core
  alone.
- **Auditable material data**: the bundled database records per-row provenance
  and licence, a central provenance registry, and both drift tests (database vs
  its canonical Python source) and golden-value tests pinning refractive-index,
  Raman and phonon values. A data correction cannot change physics silently.
- **Reproductions as regression tests**: each study ships parameters, a script
  that asserts against an analytic or published ground truth, and a figure. The
  workflow has already found and fixed real bugs — an inverted GNLSE dispersion
  sign and an inverted Raman response direction both survived the unit-test suite
  and were exposed only by literature comparisons.

The intended users are photonics researchers and educators who need trustworthy
building blocks: students learning pulse propagation with correct conventions,
and researchers who want to prototype a supercontinuum, sensing or
structured-light study without rebuilding units, grids and material data. While
general-purpose scientific packages provide the numerical substrate
[@harris2020numpy; @virtanen2020scipy; @hunter2007matplotlib], and
photonic-specific projects exist for layout or mode solving, this library
targets the *propagation and material-data* layer with an explicit stability and
provenance model.

# Acknowledgements

The author thanks the maintainers and community of the open-source scientific
Python ecosystem, and the authors of the reproduced literature whose papers made
the validation possible.

# References
