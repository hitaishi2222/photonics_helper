---
quadrant: reference
---

# Materials and the n/k database

RefractiveIndex: Sellmeier, tabulated nk sources and the built-in materials.db. Docstrings are the source of truth, rendered with
[mkdocstrings](https://mkdocstrings.github.io/) (numpydoc style).

## Extrapolation is explicit

An index table is valid only over the wavelengths it was sampled on. That window
is data: `RefractiveIndex.valid_range_um` and `Material.valid_range_um` return the
`(min, max)` bounds in μm without raising, so a caller can tag the value it got.
What happens outside the window is a per-call policy:

| `extrapolate=` | Outside the window |
| --- | --- |
| `"raise"` (default) | `ValueError` naming the window; the `Material` wrapper adds the material name |
| `"warn"` | the value, plus one `UserWarning` naming the window (and the material) |
| `"allow"` | the value, silently |

The default is unchanged, so `nk_data`'s "out-of-range evaluation is rejected"
contract and every catalogue audit keep their behaviour. The warning text is
constant per object (it does not name the wavelength), so Python's default
filter registry collapses a wavelength sweep into one line; with
`simplefilter("always")` it fires per call.

## A missing material stays missing

The catalogue carries an entry only when a primary index source exists. A
composition named only in citing papers is **absent** rather than filled with an
untraceable fit: `AsSe2` has no row, so `material("AsSe2")` fails by name, lists
the near matches (`As2Se3`, `AgGaSe2`, …) and says why the entry is not there.
Use `As2Se3` or `GeAsSe` when a measured chalcogenide curve is what the deck
needs, and record the substitution in the write-up.

## Registering your own n(λ) table

`AsSe2` is the standing example of a *data gap*: Naskar et al., *Phys. Scr.*
**101** 385502 (2026) publish the functional form of their Sellmeier fit
(eq. 2) and one calibration point (`n(2 µm) = 2.818`), but no coefficients, and
the refs they cite for the curves ([25, 34]) do not carry an AsSe₂ set either.
Inventing `A_j, B_j` to hit that point would be calibration dressed as data, so
no row is shipped.

What you *can* do is hold the numbers you can cite and wrap them once:

```python
from photonics_helper.materials import RefractiveIndex
from photonics_helper.core.materials import Material

index = RefractiveIndex.from_samples(wl_um, n, k=None)   # ≥ 4 points, λ increasing
material = Material(name="AsSe2", index=index, source="<citation for the table>")
material.n_func(2.0)            # splined, range-checked (μm)
material.valid_range_um         # (min, max) — the data's window, not a guess
```

The result is a first-class index: it splines, it enforces
`extrapolate=` outside its window, and it feeds the dispersion path
(`Dispersion.from_neff(index.n, index.wl, Wavelength(2.0, "um"))`). Nothing is
written to `materials.db`, so a user table can never be mistaken for a
catalogue entry, and the citation you attach is its only provenance.

::: photonics_helper.materials
