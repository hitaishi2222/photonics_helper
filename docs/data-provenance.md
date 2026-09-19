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
| `nk_data` | 24 767 | tabulated `n`, `k` vs λ (µm) for 12 materials | `source`, `citation` (24 760 / 24 767 populated), `license` |
| `sellmeier` | 39 | Sellmeier coefficients + validity window | `source` (39 / 39), `license` |
| `raman_specs` | 44 | Raman shift, linewidth, `f_R`, `n₂`, gain | `references` (44 / 44), `license` |
| `phonon_modes` | 52 | multi-mode phonon data for 10 crystals | `note`, `license` |
| `provenance` | 30 | central registry joinable via `nk_data.source` | `source_key`, `citation`, `doi`, `url`, `license` |

Every data row also carries a `license`. Concrete identifiers are used only
where a source declares one (`nk_data` is `CC0-1.0`); elsewhere the value is a
documented sentinel (`see-source-publication`, `see-references`, `see-note`)
pointing at the row's citation. See the [database schema](data-schema.md) for the
full model.

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
- **`phonon_modes`** — multi-mode phonon data whose canonical source is
  `photonics_helper.phonon.PHONON_MATERIALS` (per-mode literature notes).
- **`provenance`** — a registry table joining on `nk_data.source`, backfilled
  from the tabulated data so the database is auditable without external files.
  Its `doi` column is populated by extracting the identifier from each
  citation, so consumers do not have to parse citation strings.

To list everything that ships (with wavelength range, DOI and licence), use
`material_catalog()` / `print_material_catalog()` — see the
[database schema](data-schema.md#discovering-the-data) page.

## Licensing caveat

CC0 applies to the refractiveindex.info aggregation. Because datasets are
contributed from many publications, a small number may carry the terms of their
original source. The `citation` column exists so a redistributor can verify
each one. If you believe an entry cannot be redistributed, open an issue and it
will be corrected or removed.

## Status

The data-layer consolidation (Phase 2) is complete:

- `phonon_modes` is populated and reachable through `RamanDatabase` and
  `PhononResponse.from_material()` — the same interface as n/k data.
- Every row in every data table carries a `license`; the `provenance` registry
  is populated and queryable (`RamanDatabase.get_provenance()`).
- The shipped database is verified against the canonical Python tables by
  drift tests, and key values are pinned by golden tests.

SQLite access is lazy: constructing a `RamanDatabase` has no filesystem side
effects; the schema/seed step runs on first use.

Splitting the data into a separate distribution or a download-on-demand extra
remains a future option; the database stays bundled for now.

`REPORT.md` (git-ignored local notes) tracks the wider "Foundation Backbone"
roadmap.
