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
date: 22 October 2026
bibliography: paper.bib
---

# Summary

`photonics-helper` is a Python library for photonics, fiber optics and
nonlinear-optics computation. It is built around an explicit foundation layer:
typed physical units and constants, temporal and spectral grids with a
documented FFT convention, and an optical-material database that records where
each value came from. The physics and simulation tools sit on top of that layer.

The foundation stays small. Importing `photonics_helper.core` loads only NumPy,
SciPy and Pydantic, and `import photonics_helper` is lazy, so it pulls in no
plotting or web stack. Another tool can build on the core without inheriting an
application.

On top of the core the library provides a solver suite:

- GNLSE propagation (split-step, adaptive stepping, z-dependent dispersion,
  tapered waveguides) with Kerr, delayed Raman, self-steepening, two-photon
  absorption and a stochastic Raman noise source [@blow1989theoretical],
  integrated with the fourth-order RK4IP scheme [@hult2007fourthorder];
- soliton and breather analysis (soliton order, fission length, dispersive
  waves, Raman self-frequency shift [@gordon1986theory], Kuznetsov-Ma,
  Peregrine and Akhmediev solutions);
- Raman response and material data (time- and frequency-domain responses,
  pumped Stokes and anti-Stokes explorers, 45 materials with per-row
  provenance);
- phase-matching diagnostics for four-wave mixing, modulation instability and
  Cherenkov dispersive waves, plus χ^(2)^ mixing (SHG, SFG and DFG with
  quasi-phase-matching and modal-overlap coupling), DBR/TMM stacks, FROG trace
  generation and retrieval, structured light (Laguerre-Gaussian and OAM modes),
  and optical wave-breaking diagnostics.

Two properties separate the library from a collection of scripts. Units are
first-class objects. `Wavelength(1550, "nm")` carries its unit and dimension,
conversions are explicit, and the factor-of-1000 errors that dominate photonics
code do not compile as silently. The library also measures trust against the
literature rather than against another code. `reproductions/` holds
self-contained studies that reproduce published results and assert them against
closed-form references, including supercontinuum generation
[@dudley2006supercontinuum], Raman soliton self-frequency shift
[@gordon1986theory], self-phase modulation [@stolen1978self], Cherenkov
dispersive waves, modulation instability, breathers and dispersive broadening.
The archived release described here is available from Zenodo [@zenodo2026].

# Statement of need

Photonics research code is routinely written as one-off scripts: wavelengths are
bare floats, dispersion signs and FFT scalings get re-derived each time, and
material parameters are copied without a record of where they came from. Unit
handling, grid conventions and data lookup then take most of the effort, and
silent errors are expensive. Two failure modes recur. Unit mistakes produce
numbers that look physically plausible, and convention mistakes (dispersion
sign, FFT normalisation, Raman response direction) survive unit tests and appear
only when a result is compared with an experiment or a published figure.

`photonics-helper` makes that layer stable, documented and tested, and it treats
published results as the acceptance criterion. It provides:

- a frozen, dependency-light core (`photonics_helper.core`) with a published
  [stability contract](https://hitaishi2222.github.io/photonics_helper/stability/),
  checked by snapshot tests on symbols and signatures, so downstream projects
  can depend on it; a worked example builds an external tool on the core alone;
- auditable material data: the database records provenance and licence per row,
  keeps a central provenance registry, and is pinned by drift tests (database
  against its canonical Python source) and golden-value tests for
  refractive-index, Raman and phonon values, so a data correction cannot change
  physics silently;
- reproductions as regression tests: each study ships its parameters, a script
  that asserts against an analytic or published ground truth, and a figure; the
  procedure has already found two real bugs, an inverted GNLSE dispersion sign
  and an inverted Raman response direction, both of which survived the
  unit-test suite;
- a machine-checked specification in `formal/`: a Lean 4 development that
  proves the sign conventions, analytic limits and conservation laws behind the
  numerical kernels. A test can confirm that a computation ran, but not that
  the convention it used was the intended one. The formal layer states each
  convention once as a theorem: reversing the spectral axis negates exactly the
  odd-order dispersion terms, the Fourier analysis and synthesis kernels are a
  matched pair, a real input has a conjugate-paired spectrum, and a normalised
  Raman response cannot amplify past unity. Where proof and code disagree, the
  disagreement is recorded and resolved against the primary literature, and the
  register is public.

The intended users are photonics researchers and educators who need trustworthy
building blocks: students learning pulse propagation with correct conventions
[@agrawal2013nonlinear], and researchers who want to prototype a
supercontinuum, sensing or structured-light study without rebuilding units,
grids and material data.

# State of the field

General-purpose scientific libraries supply the arrays, transforms and solvers
that photonics code runs on [@harris2020numpy; @virtanen2020scipy;
@hunter2007matplotlib]. Pulse propagation itself has several open
implementations. PyNLO [@pynlo] provides GNLSE pulse propagation in Python,
laserfun [@laserfun] collects Python functions for simulating laser pulses,
including a GNLSE solver, gnlse-python [@gnlsepython] implements a GNLSE solver
in Python, and the supercontinuum literature distributes MATLAB implementations
with its tutorial code [@dudley2006supercontinuum].

Those packages concentrate on the propagation kernel and the inputs it needs.
This library was built for the layer underneath. Wavelengths, grid conventions,
FFT scaling and material parameters are what break silently, and to the author's
knowledge no shared material database with per-row provenance was available to
work from. Reusing one of the existing solvers would have meant adopting its
conventions and its data model throughout, and the conventions are exactly what
the reproductions and the Lean development exist to pin down.

The work did not fork those projects. The author contributed a custom Raman
response implementation and example updates to laserfun [@laserfun] before
developing this library, and keeps the propagation code here as one consumer of
a shared foundation rather than as a competing solver. A user who needs only a
kernel should use one of the packages above. A user who needs the conventions
and the material data pinned down, or who wants to build another tool on the
same base, has a reason to use this one.

# Software design

Four decisions shape the code.

The foundation is small and frozen. `photonics_helper.core` depends on NumPy,
SciPy and Pydantic only, and the package imports submodules lazily, so a
plotting or web stack loads only when a caller asks for it. The alternative was
one importable package with every feature wired in, which is simpler to write
and worse to depend on. The small core costs extra plumbing between layers, and
that cost buys a core a downstream tool can adopt without adopting the
application.

Units are objects, not floats. `Wavelength(1550, "nm")` carries its unit and
dimension, and the author writes every conversion explicitly. Bare floats are
shorter to type, and most published photonics code uses them. The trade-off was
accepted because a unit error raises no exception: it returns a wrong number
that looks right, which is the failure this library exists to prevent.

Conventions are stated once and checked twice. Each convention is a Lean 4
theorem in `formal/` and a convention test in Python. A proof costs more than a
comment above the code, and the two representations can drift. In exchange, a
changed convention fails the build, and the theorem is a statement about the
algebra the code actually performs rather than a claim in a docstring.

Validation is a set of runnable studies, not a fixture set. Each study in
`reproductions/` commits its parameters and asserts its figure against an
analytic limit or a published value, so the literature is the acceptance test
for the physics. The studies run longer than the unit tests and run separately
in CI. That runtime bought the two convention bugs the unit tests missed.

Material data carries its own audit trail. Each database row records its
provenance and licence, a central registry maps entries to published sources,
and drift tests compare the database with the canonical Python source. A
correction to a refractive index or a Raman parameter therefore arrives with a
citation and a failing test, so it cannot change a simulation silently.

# Research impact statement

The library is in use in the author's own research workflow, as the environment
in which numerical results are checked against the literature. That use is
verifiable in the repository: the `reproductions/` studies run in CI, assert
against closed-form references or published figures, and have twice found
convention bugs the unit-test suite missed, an inverted GNLSE dispersion sign
and an inverted Raman response direction. The author also contributed a custom
Raman response implementation and example updates upstream to laserfun.

Near-term significance rests on reproducible materials rather than on citations.
The repository ships the parameters and assertions behind every reproduced
result, a Lean 4 development that checks the conventions those results depend
on, benchmarks, and a published stability contract that lets another project
depend on the core. A reviewer or a new user can run all of it.

No external publication cites the library yet, and it has no external adopters
at the time of submission. The impact claim is therefore about the material it
provides for verification, and not about citations.

# AI usage disclosure

Generative AI tools were used in the development of this software. They were
LLM coding agents driven through the `pi` coding-agent harness, whose session
logs record the model name and version for each session.

Their use covered code generation and refactoring of solver modules, examples
and tests; debugging and code review, including audits of numerical kernels for
convention and sign errors; drafting of docstrings, documentation pages and
repository markdown; and drafting and copy-editing of this paper.

The author reviewed, edited and validated every AI-assisted output, and made the
design decisions the library rests on: the typed-units foundation, the FFT and
dispersion-sign conventions, the stability contract, the material-data
provenance and licence model, and the scope of the Lean 4 specification.
Physics-facing code is checked by the `reproductions/` studies against published
results and closed-form ground truth, by the Lean development for the
conventions and conservation laws, and by the test suite and CI on every change.

# Acknowledgements

The author thanks the maintainers and community of the open-source scientific
Python ecosystem, and the authors of the reproduced literature whose papers made
the validation possible.

# References
