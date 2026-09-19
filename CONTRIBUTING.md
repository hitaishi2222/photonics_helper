# Contributing to photonics-helper

Thanks for helping. This project aims to be a **foundation** for photonics
code, which means correctness, clear units and backward compatibility matter
more than feature count. Contributions are accepted under the MIT licence (see
`LICENSE`).

## Setup

```sh
git clone https://github.com/hitaishi2222/photonics_helper
cd photonics_helper
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,plotting,webapp,examples]"
```

Optional: `pip install -e ".[dev,all,fftw]"` to exercise the FFTW backend.

## Gates

Every change must pass all three; CI runs them across Python 3.12/3.13 and with
and without `pyfftw`.

```sh
ruff check .            # lint
mypy photonics_helper   # types (0 errors expected)
pytest -q               # tests
```

Also useful before a release:

```sh
python -m build --wheel && python scripts/check_wheel.py
```

## Proposing a change

This repository uses [OpenSpec](https://github.com/Fission-AI/OpenSpec)-style
change proposals under `openspec/changes/` (spec-driven: why → what → design →
tasks). For anything beyond a trivial fix:

1. Describe the *why* and the affected capability.
2. Keep implementation details in the design section.
3. Land the change and tests together.

Bug reports are welcome as issues — for **data or physics** bugs, include the
citation or reference that establishes the correct value.

## Stability rules (important)

Read [`docs/stability.md`](docs/stability.md) before changing a public symbol.

- `photonics_helper.core.*` is the **stable** surface. Removing or renaming a
  stable symbol requires a deprecation cycle; use
  `photonics_helper._deprecation` to warn.
- After an **intentional** stable-surface change, regenerate the guard snapshot:
  ```sh
  python scripts/update_api_snapshot.py
  ```
  Review the diff — it should contain exactly the change you intended.
- Satellites (solvers, plotting, dashboard) may evolve, but document behaviour
  changes in `CHANGELOG.md`.
- Numerical corrections are allowed in a patch release, but must be marked
  `BREAKING` in the changelog and backed by a test or reference.

## Changing material data

Canonical data lives in Python:

- `photonics_helper/raman/reference.py` — Raman/substrate specs
- `photonics_helper/phonon.py` — `PHONON_MATERIALS`
- `seed_db.py` — Sellmeier coefficients

After editing, regenerate and verify:

```sh
python seed_db.py        # updates raman_specs + phonon_modes
pytest tests/test_material_data_drift.py tests/test_material_data_golden.py
```

If a corrected value changes a golden test, update the test **and** explain the
correction in the changelog.

## Adding a reproduction

Reproductions live under `reproductions/<name>/` with a `parameters.json`, a
`reproduce.py` that checks against an analytic or published reference, a figure
and a `README.md`. Do not add comparisons against other simulation packages;
the point is agreement with physics, not with another code.

## Style

- Type hints everywhere; the package ships `py.typed` and must keep `mypy` clean.
- Units are classes — accept and return them at public boundaries (see the
  README's "Units are classes" section).
- Prefer analytic checks and limiting cases in tests over snapshot-only tests.
- Keep new dependencies optional unless the core genuinely needs them; `core`
  must stay numpy/scipy/pydantic only.
