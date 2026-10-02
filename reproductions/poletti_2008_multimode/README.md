# Reference only — Poletti & Horak (2008)

**Reference:** F. Poletti and P. Horak, "Description of ultrashort pulse
propagation in multimode optical fibers", *J. Opt. Soc. Am. B* **25**(10),
1645 (2008). File in this folder: `JOSAB.25.001645.pdf`.

**This folder holds the paper as reference material. It is not a numerical
reproduction** — there is no `parameters.json`, no `reproduce.py` and no
`validate()`, so it is deliberately absent from the main table in
`reproductions/README.md` (the same convention as
`dudley_2014_breathers_review/`).

**Why it is here.** This is the extended-GNLSE paper for multimode fibers:
polarization effects, high-order dispersion, Kerr and Raman, self-steepening,
and **wavelength-dependent mode coupling and nonlinear coefficients** — i.e. the
direct specification for the `multimode_gnlse` engine, which implements the
LP (`1, 2/3`) / isotropic SPM–XPM coefficient sets, per-mode dispersion and
group delay, and optional inter-modal FWM.

**What a deck here would add.** The paper's own analysis of the symmetry
properties of the nonlinear coupling coefficients (step-index, circularly
symmetric and hexagonally symmetric microstructured fibers) is a
self-contained analytical target that the engine could be checked against
without new solver machinery — the natural first candidate. The paper also
discusses computational complexity, which would inform the engine's step-size
policy.

If a deck is built here, it must add `parameters.json` + `reproduce.py` +
`README.md` and a row in the main table, per the registration convention in
`reproductions/README.md`.