# Stability contract

`photonics_helper` aims to be a **foundation other photonics projects build on**.
A foundation is only useful if you can depend on it without fear, so this page
states exactly what is stable, what a version number means, and how anything
gets retired.

If you are building on this library, read this page once and then build with
confidence against the **stable surface** below.

## The stable surface

| Layer | Modules | Guarantee |
|---|---|---|
| **Stable** | `photonics_helper.core.units`, `.constants`, `.grids`, `.materials`; the database **schema** | Frozen contract. Changes are additive within a major version; removals/renames follow the [deprecation lifecycle](#deprecation-lifecycle). |
| **Provisional** | `gnlse`, `chi2`, `phase_matching`, `fiber`, `pulse`, `raman`, `soliton`, `dbr`, `breathers`, `noise`, `wave_breaking`, `structured`, `phonon`, `materials`, `dashboard`, `_fftw` | First-class and tested, but may change behaviour or API in any **minor** release as the physics and features develop. |

"Stable" means the **symbol must keep working as documented**: same name, same
call signature, same meaning. It does *not* mean the numbers can never be
corrected — see [numerical corrections](#versioning).

### What counts as public

- A name in a module's `__all__`.
- A documented class, function or attribute (including properties) of a stable
  module.
- Names starting with `_` are **private** and may change at any time.

Every existing import path is part of the contract: `photonics_helper.base`,
`photonics_helper.pulse.TemporalGrid`, and `photonics_helper.core.grids.TemporalGrid`
refer to the same objects and all keep working.

## Versioning

The project follows [semantic versioning](https://semver.org/) with the usual
pre-1.0 caveat that the **minor** number carries breaking changes until 1.0.

| Change | Pre-1.0 | Post-1.0 |
|---|---|---|
| Add a symbol to the stable surface | MINOR | MINOR |
| Add an *optional* parameter to a stable function | MINOR | MINOR |
| Change a satellite's behaviour/API | MINOR | MINOR |
| Remove/rename a stable symbol (after deprecation) | MINOR | MAJOR |
| Correct a wrong numerical result or data value | PATCH¹ | PATCH¹ |

¹ **Numerical corrections may ship in a PATCH**, but the changelog MUST mark
them `BREAKING` and cite the test or reference that justifies them. This is a
deliberate departure from strict semver: keeping a known-wrong coefficient,
sign or convention in the field because the version number is inconvenient
would be worse than the break. The 0.1.1 χ⁽²⁾ coupling fix (a 2× correction) is
the canonical example.

## Deprecation lifecycle

When a stable symbol has to be removed or renamed:

1. **Mark** — the symbol keeps working and emits a `DeprecationWarning` naming
   the replacement and the release in which it will be removed. The warning is
   emitted **once per process**.
2. **Support** — the deprecated symbol stays for **at least one minor release**
   and one full release cycle in the documentation.
3. **Remove** — removal ships in a minor release (pre-1.0) or a major release
   (post-1.0), with a **migration note** in the changelog.

The runtime helper is `photonics_helper._deprecation`:

```python
from photonics_helper._deprecation import deprecated

@deprecated("Use spectral_grid() instead.", replacement="spectral_grid",
            since="0.1.7", removed_in="0.3.0")
def old_grid(*args, **kwargs):
    ...
```

Nothing in the library is deprecated today; the machinery exists so the next
rename follows the policy instead of breaking callers.

## Support window

- **The latest minor release receives fixes** (bugs, data corrections,
  security).
- The **previous minor** receives critical correctness/security fixes only.
- Older releases are unsupported; upgrade is the supported path.
- **Python**: the versions exercised in CI (currently 3.12 and 3.13) are
  supported. Dropping a Python version is a minor-release change announced in
  the changelog.
- **Optional extras** (`fftw`, `plotting`, `webapp`, `extras`, `mode-export`)
  are tested in CI where practical; `pyfftw` is tested both present and absent.

## Database schema

The bundled `materials.db` is covered by the same contract:

- Schema changes are **additive** (`ALTER TABLE ... ADD COLUMN`,
  `CREATE TABLE IF NOT EXISTS`) and idempotent, so older code can read a newer
  database.
- **Removing a table or column, or changing the meaning of an existing value,
  is a breaking change** and follows the versioning and deprecation rules.
- Physical values are pinned by golden tests; a correction follows the
  patch-release rule above.
- Canonical data lives in Python (`raman/reference.py`, `phonon.PHONON_MATERIALS`)
  and the database is verified against it — see [Database schema](data-schema.md)
  and [Data provenance](data-provenance.md).

## How the contract is enforced

The policy is not just prose; it is tested:

| Guarantee | Test |
|---|---|
| Stable symbols are not removed/renamed | `tests/test_core_api_surface.py` (symbol snapshot) |
| Stable signatures are not broken | `tests/test_core_api_surface.py` (parameter-shape snapshot) |
| The core imports without the plotting stack | `tests/test_import_budget.py`, `tests/test_core.py` |
| The public top-level surface does not shrink | `tests/test_lazy_import.py` (`public_api_snapshot.json`) |
| The database matches its canonical source | `tests/test_material_data_drift.py` |
| Physical values are unchanged | `tests/test_material_data_golden.py` |
| Deprecations warn once and keep working | `tests/test_deprecation.py` |

After an **intentional** stable-surface change, regenerate the snapshot so the
diff is explicit and reviewable:

```sh
python scripts/update_api_snapshot.py
```

## Proposing a change

- **Bug / data correction** — open an issue with the citation or reference that
  establishes the correct value.
- **New feature** — satellites are added freely; additions to the stable core
  are reviewed against whether the project can support them indefinitely.
- **Breaking a stable symbol** — the proposal must include the deprecation plan
  (replacement, `since`, `removed_in`).
- **Contributing code** — see [CONTRIBUTING.md](https://github.com/hitaishi2222/photonics_helper/blob/main/CONTRIBUTING.md).

## See also

- [Foundation core](core.md) — what the stable surface contains and how to use
  it.
- [Building on the core](building-on-core.md) — a worked external example.
- [Database schema](data-schema.md) · [Data provenance](data-provenance.md).
