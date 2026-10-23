---
quadrant: home
---

# photonics-helper

Unit-safe types, dispersion, pulse envelopes, the GNLSE, Raman modelling,
χ⁽²⁾ mixing, DBR/TMM stacks and literature reproductions, for photonics and
nonlinear optics in Python.

```bash
pip install photonics-helper
```

```python
from photonics_helper import Wavelength

wl = Wavelength(1550, "nm")
print(wl.to_freq().as_THz)     # 193.4
print(wl.to_energy().as_eV)    # 0.8
```

## Start here

**[Tutorial: one fiber, one pulse, one answer you can check](tutorial.md)** is
the place to begin. You launch a soliton into 10 mm of fiber, predict the output
with three multiplications, and confirm the code agrees. That takes about ten
minutes and it teaches the unit conventions the rest of the library uses.

Before you invest a week in a model, read
[what this library does not do](scope.md). It is one page and it will save you
from building on a solver that does not cover your regime.

## Then follow your question

**[How-to guides](install.md)** answer a task: pick a grid, check convergence,
add a material, import a mode, reproduce a paper, read a spectrum. Start with
[install](install.md) if you have not run anything yet.

**[Examples](gallery.md)** is every runnable script in the repository, 48 of
them, each with the figure it produces and its full source. If you want to see
what a solver does before reading about it, this is the page.

**[Reference](api/index.md)** states what exists: one page per module, plus the
[stability contract](stability.md) and the [database schema](data-schema.md).

**[Explanation](gnlse-physics.md)** covers why the numerics work: the GNLSE
physics, the Raman model, the formal Lean specification, and the provenance of
the material data.

## Conventions

All internal math is SI (m, s, rad/s, W, J). Unit classes convert on
construction and expose `as_*` accessors. The GNLSE takes `betas` in `ps^k/m`
with detuning `Ω` in `rad/ps`, unless you pass `betas_unit="s^k/m"`.

Detuning is always measured from the carrier, so a spectrum's `grid.w` starts at
zero at your central wavelength. [Read a spectrum](how-to/read-a-spectrum.md)
walks through the conversion that most often goes wrong.
