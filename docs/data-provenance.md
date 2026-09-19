# Data provenance

`photonics-helper` bundles a numerical material database
(`photonics_helper/materials.db`) alongside the code. This page documents where
that data comes from, the terms it is redistributed under, and how to audit it.
The canonical notice also ships as [`NOTICE`](https://github.com/hitaishi2222/photonics_helper/blob/dev/NOTICE)
in the repository.

## Code vs data

| Component | Licence |
|---|---|
| `photonics_helper` source code | MIT (see `LICENSE`) |
| `photonics_helper/materials.db` | compilation of literature data — see below |

## Tables and provenance columns

| Table | Rows | Contents | Provenance |
|---|---|---|---|
| `nk_data` | 24 767 | tabulated `n`, `k` vs λ (µm) for 12 materials | `source`, `citation` (24 760 / 24 767 populated) |
| `sellmeier` | 39 | Sellmeier coefficients + validity window | `source` (39 / 39) |
| `raman_specs` | 44 | Raman shift, linewidth, `f_R`, `n₂`, gain | `references` (44 / 44) |
| `phonon_modes` | 0 | multi-phonon modes | empty (roadmap Phase 2) |

Inspect any row directly:

```python
import sqlite3, importlib.resources

with importlib.resources.as_file(
    importlib.resources.files("photonics_helper") / "materials.db"
) as db:
    con = sqlite3.connect(db)
    for row in con.execute("SELECT material, source, citation FROM nk_data LIMIT 3"):
        print(row)
```

## Where the data comes from

- **`nk_data`** — collected from [refractiveindex.info](https://refractiveindex.info/)
  via its public sitemap (`nk_datasets/collect.py`, `nk_datasets/manifest.json`).
  The site publishes its compilation under **CC0 1.0 Universal**. Individual
  datasets originate in specific publications, so each row keeps a `citation`
  naming the original source; keep that column if you repackage the database.
  Seven rows are currently provenance-incomplete.
- **`sellmeier`** — dispersion equations taken from the cited papers
  (Malitson 1965, Edwards & Lawrence 1984, Luke et al. 2015, …); `source`
  carries the attribution.
- **`raman_specs`** — a literature compilation original to this project, with
  per-row `references` to the underlying paper or textbook.

## Licensing caveat

CC0 applies to the refractiveindex.info aggregation. Because datasets are
contributed from many publications, a small number may carry the terms of their
original source. The `citation` column exists so a redistributor can verify
each one. If you believe an entry cannot be redistributed, open an issue and it
will be corrected or removed.

## Roadmap

Data-layer consolidation (single material contract, populated `phonon_modes`,
removal of the duplicated `RAMAN_MATERIALS` dict, explicit per-row licence) is
Phase 2 of the Foundation Backbone roadmap; see `REPORT.md`.
