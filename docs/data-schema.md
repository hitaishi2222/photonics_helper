---
quadrant: reference
---

# Database schema

`photonics_helper/materials.db` is a SQLite database bundled with the package
(and listed in `package-data`). This page documents its tables, the provenance
and licence model, and how the database is regenerated from the canonical
Python tables.

## Single source of truth

The **canonical data lives in Python**, not in the database:

| Canonical table | Module | Contents |
|---|---|---|
| `RAMAN_MATERIALS` | `photonics_helper/raman/reference.py` | 32 Raman specs |
| `THORLABS_SUBSTRATE_MATERIALS` | `photonics_helper/raman/reference.py` | 12 substrate specs |
| `PHONON_MATERIALS` | `photonics_helper/phonon.py` | 10 materials, 52 modes |
| `SELLMEIER_MATERIALS` | `seed_db.py` | Sellmeier coefficients |
| `nk_datasets/manifest.json` | (not in the repo) | tabulated n/k datasets |

`materials.db` is a **derived artefact**. Drift between the database and the
canonical tables is a test failure
(`tests/test_material_data_drift.py`), and the physics values are pinned by
`tests/test_material_data_golden.py`, so a regeneration cannot change them
silently.

## Tables

### `raman_specs`
One row per material: `name` (PK), `crystal`, `bandgap_eV`, `n2`,
`raman_shift_cm`, `raman_linewidth_cm`, `fR`, `gain_coeff`, `tau1`, `tau2`,
`lo_phonon_cm`, `to_phonon_cm`, `references`, `license`.

### `nk_data`
Tabulated refractive index vs wavelength: `material`, `wavelength_um`, `n`, `k`,
`source` (a `material-author` provenance key, e.g. `si-aspnes`), `citation`,
`license`.

Unique per `(material, wavelength_um, COALESCE(source, ''))` through the
`nk_data_unique` index, so `get_nk_data` returns one sample per wavelength.
Re-adding a point replaces it (`add_nk_data` is `INSERT OR REPLACE`); a file
that predates the index is deduped on first use (highest rowid wins).

### Writing rows
`add_material` and `import_from_dict` are **partial upserts**: only the supplied
columns are written, so `add_material({"name": "Silica"})` leaves `fR`,
`raman_shift_cm` and `n2` intact, and a new row gets NULL only for fields never
supplied. Unknown keys raise `ValueError` (same contract as `update_material`).

### `sellmeier`
Sellmeier coefficients: `material` (PK), `form` (`standard`/`alt`), `a0`,
`coefficients` (JSON), `wavelengths` (JSON), `valid_from_um`, `valid_to_um`,
`source`, `license`.

### `phonon_modes`
Multi-mode phonon data: `material`, `shift_cm`, `linewidth_cm`,
`fwhm_gauss_cm` (NULL = pure Lorentzian), `symmetry`, `relative_strength`,
`lo_phonon_cm`, `to_phonon_cm`, `note`, `license`.
Primary key `(material, shift_cm, symmetry)`.

### `provenance`
Central registry, joinable via `nk_data.source`:

| Column | Meaning |
|---|---|
| `source_key` | the `material-author` key (PK), e.g. `si-aspnes` |
| `kind` | `nk` (currently the only kind seeded) |
| `citation` | full reference the values were taken from |
| `doi` | DOI when known |
| `url` | source URL when known |
| `license` | licence identifier or sentinel |

## Licence model

Every data row carries a `license`. Concrete identifiers are used only where a
source genuinely declares one; otherwise the value is a **sentinel** pointing at
the row's citation column:

| Value | Applies to | Meaning |
|---|---|---|
| `CC0-1.0` | `nk_data` | refractiveindex.info compilation dedication |
| `see-source-publication` | `sellmeier` | coefficients are facts from the cited paper |
| `see-references` | `raman_specs` | literature-compiled values |
| `see-note` | `phonon_modes` | per-mode literature note |

Sentinels are deliberately not valid SPDX identifiers, so they cannot be
mistaken for a blanket licence grant. See also [Data provenance](data-provenance.md)
and `NOTICE`.

## Lazy access

`RamanDatabase` resolves its path on construction but does **not** open, create,
migrate or seed the database until the first query. Constructing a database
handle is therefore free, and read-only consumers (for example a provenance
lookup in `core.materials`) do not touch the file until they need data.

```python
from photonics_helper.raman import RamanDatabase

db = RamanDatabase()  # no filesystem access
db.list_provenance()  # first use: schema ensured, then the query
```

## Read-only installs

The migration is check-first: the bundled file is opened with SQLite's
`mode=ro` and its tables, columns, licence backfills and the
`phonon_modes_unique` / `nk_data_unique` indexes are inspected. When nothing is
missing the file is **never written**, so a root-owned conda install, a
read-only container image
or an HPC module can serve every read path (`material()`,
`RamanSpec.from_database`, phonon modes, provenance). Only a file that really
lacks a schema object is opened read-write; if that is impossible the error
names the read-only state and the missing object instead of calling the file
corrupt. Schema work is cached per resolved path for the process, so a second
`MaterialsDatabase()` for the same file does not repeat it.

## Discovering the data

The catalogue lists every dataset the database ships, tabulated n/k spectra and
Sellmeier equations together, with the columns you need to choose one:

```python
from photonics_helper import material_catalog, print_material_catalog

print_material_catalog()  # rich table: material, type, λ range, DOI, licence
print_material_catalog("sil")  # case-insensitive name filter

rows = material_catalog()  # programmatic: list[MaterialDataset]
```

Each `MaterialDataset` carries `material`, `kind` (`"tabulated"` or
`"sellmeier"`), `axis` (for birefringent sub-rows), `source`, `wl_min_um` /
`wl_max_um`, `n_points`, `doi`, `citation` and `license`.

Under the hood it reads `RamanDatabase.list_nk_dataset_summaries()` (tabulated)
and `RamanDatabase.list_sellmeier_datasets()` (Sellmeier). The latter exists
because Sellmeier rows are **not** all present in `raman_specs`: Silicon,
Sapphire, Germanium and the `LiNbO3_er` / `LiNbO3_or` sub-rows have no Raman
spec, so building a catalogue from `list_materials()` alone would silently drop
them.

## Regenerating the database

From the repository root:

```sh
# 1. Raman specs + phonon modes (does not touch nk_data/sellmeier)
python seed_db.py

# 2. Optionally re-seed Sellmeier + tabulated nk from nk_datasets/ (external)
python seed_db.py --nk
```

Step 1 is the safe, routine path: it re-writes `raman_specs` from
`reference.py`, seeds `phonon_modes` from `PHONON_MATERIALS`, and runs the
additive schema migration (licence columns + `provenance` registry). Step 2
requires the `nk_datasets/manifest.json` collection, which is intentionally not
in the repository; the shipped `nk_data` table is authoritative. The whole
manifest is validated before anything is deleted, and the surviving rows are
written in one transaction; if every dataset fails validation the shipped rows
are left untouched.

After any regeneration, run:

```sh
python -m pytest tests/test_material_data_drift.py tests/test_material_data_golden.py
```

## Querying provenance

```python
from photonics_helper.raman import RamanDatabase

db = RamanDatabase()
record = db.get_provenance("si-aspnes")
print(record["citation"], record["license"])
```
