---
quadrant: explanation
---

# What this library does not do

Read this before you start, so you find out now rather than after a week of
modelling.

`photonics_helper` covers one-way propagation of optical envelopes in guided
structures, plus the supporting calculations around it: dispersion from material
data, mode analysis, phase matching, Raman gain, multilayer stacks and fitting.
Everything outside that list belongs to another tool.

## Not a beam-propagation or diffraction solver

The engines are one-way (z) envelope models. The transverse structure is frozen
into mode profiles, overlaps and γ. Transverse diffraction, self-focusing,
multi-mode imaging and anything where the beam profile evolves as it propagates
needs a different code.

If your question is "what does the field look like across the waveguide", you
want a mode solver (see below). If it is "how does the beam reshape in free
space", you want a beam-propagation method implementation.

## Not bidirectional

Propagation is one-way. Backscattering is not modelled, with one exception: the
opt-in multimode Rayleigh term. Distributed feedback, random lasing and
reflection from a grating inside the propagation direction are outside the
model.

## Not a general mode solver

[`vector_modes`](api/vector-modes.md) gives exact analytic modes for a circular
step-index guide. [`mode_analysis`](api/mode-analysis.md) consumes exports from
an external solver. Neither replaces a full-vector finite-element solve for an
arbitrary cross-section.

Use femwell or tidy3d for the eigenmode problem, then import the result. The
[how-to guide](how-to/import-a-mode.md) walks through that path.

## Not a temporal χ⁽²⁾ propagator

The [`chi2`](api/chi2.md) functions are z-only continuous-wave models, not
temporal envelope propagators. For χ⁽²⁾ plus χ⁽³⁾ pulse dynamics use the
[cascaded model](cascaded-chi23.md), or an external solver.

## Not a full-vector Maxwell solver for propagation

The vector and multimode engines propagate a small number of modal envelopes.
They do not solve Maxwell's equations on a grid at every step. If you need the
field everywhere, you are in FDTD or FEM territory (Meep, tidy3d).

## Where the boundary is enforced

`GNLSESolver.verify_convergence()` and the reductions described in
[choosing a solver](choose-a-solver.md) tell you whether the model you picked is
adequate. They cannot tell you whether the model applies at all. That is what
this page is for.
