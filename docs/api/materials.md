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

::: photonics_helper.materials
