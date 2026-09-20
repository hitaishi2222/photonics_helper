"""Tests for the ``photonics_helper.core`` foundation namespace.

Covers the core-backbone contract: the namespace is dependency-light (no
plotting/web stack, no solver satellites), it exposes the primitives, the moved
``TemporalGrid`` keeps its identity and import path, and the material facade
delegates to the existing database lookup.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

from photonics_helper.base import Time

HEAVY = ("matplotlib", "plotly", "dash")
SATELLITES = (
    "photonics_helper.gnlse",
    "photonics_helper.chi2",
    "photonics_helper.dbr",
    "photonics_helper.pulse",
    "photonics_helper.phase_matching",
    "photonics_helper.raman",
    "photonics_helper.dashboard",
    "photonics_helper.structured",
)


def _probe(statement: str) -> tuple[list[str], list[str]]:
    """Run ``statement`` in a fresh interpreter; return (heavy, satellites)."""
    snippet = (
        "import json, sys\n"
        f"{statement}\n"
        "print('@@' + json.dumps({"
        f"'heavy': sorted(m for m in {HEAVY!r} if m in sys.modules), "
        f"'sat': sorted(m for m in {SATELLITES!r} if m in sys.modules)"
        "}))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", snippet], capture_output=True, text=True, check=True
    )
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("@@"))
    payload = json.loads(line[2:])
    return payload["heavy"], payload["sat"]


# ─── Import boundary ──────────────────────────────────────────────────────


def test_core_import_is_dependency_light() -> None:
    heavy, _ = _probe("import photonics_helper.core")
    assert heavy == [], f"importing photonics_helper.core loaded {heavy}"


def test_core_import_does_not_load_satellites() -> None:
    _, satellites = _probe("import photonics_helper.core")
    assert satellites == [], f"importing photonics_helper.core loaded {satellites}"


# ─── Namespace surface ────────────────────────────────────────────────────


def test_core_exposes_primitives() -> None:
    from photonics_helper.core import constants, grids, materials, units

    assert units.Wavelength(1550, "nm").to_freq().as_THz == 193.41448903225802
    assert constants.C_MS > 2.99e8
    assert grids.TemporalGrid is not None
    assert materials.OpticalMaterial is not None


def test_temporal_grid_identity_is_preserved() -> None:
    from photonics_helper.core.grids import TemporalGrid as FromCore
    from photonics_helper.pulse import TemporalGrid as FromPulse

    assert FromCore is FromPulse
    assert "TemporalGrid" in __import__("photonics_helper").__all__


def test_legacy_import_paths_still_work() -> None:
    from photonics_helper import Wavelength  # noqa: F401
    from photonics_helper.base import Wavelength as FromBase

    assert Wavelength is FromBase


def test_temporal_grid_no_longer_in_pulse_source() -> None:
    """The class definition should live in core.grids, not pulse."""
    import photonics_helper.core.grids as grids
    import photonics_helper.pulse as pulse

    assert grids.TemporalGrid.__module__ == "photonics_helper.core.grids"
    assert pulse.TemporalGrid.__module__ == "photonics_helper.core.grids"


# ─── Materials interface ──────────────────────────────────────────────────


def test_material_satisfies_protocol() -> None:
    from photonics_helper.core.materials import OpticalMaterial, material

    mat = material("Silica")
    assert isinstance(mat, OpticalMaterial)


def test_material_lookup_delegates_to_existing_behaviour() -> None:
    from photonics_helper.core.materials import material
    from photonics_helper.materials import RefractiveIndex

    mat = material("Silica")
    reference = RefractiveIndex.from_material_database("Silica")

    assert np.allclose(mat.n, reference.n)
    assert np.allclose(mat.k, reference.k)
    assert abs(mat.n_func(1.55) - reference.n_func(1.55)) < 1e-12
    assert mat.source  # provenance is populated for Silica


def test_material_protocol_rejects_incomplete_object() -> None:
    from photonics_helper.core.materials import OpticalMaterial

    class NotAMaterial:
        name = "nope"

    assert not isinstance(NotAMaterial(), OpticalMaterial)


# ─── Grid usability without plotting ──────────────────────────────────────


def test_grid_fft_roundtrip_without_matplotlib() -> None:
    from photonics_helper.core.grids import TemporalGrid

    grid = TemporalGrid(N=256, Tmax=Time(1.0, "ps"))
    field = np.random.default_rng(0).normal(size=grid.N) + 0j
    assert np.allclose(grid.ifft(grid.fft(field)), field)


def test_core_data_does_not_import_satellites_at_module_level():
    """The materials DB backend (core.data) must stay a core primitive.

    Satellite leaves (``raman`` reference tables, ``phonon`` types) are
    method-level lazy imports only — importing ``core.data`` must not
    pull any satellite module, preserving the foundation import budget.
    """
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import photonics_helper.core.data\n"
        "mods = [m for m in sys.modules if 'raman' in m or 'phonon' in m]\n"
        "assert not mods, mods\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_core_data_is_the_material_database_backend():
    """The historical RamanDatabase path is an alias of core.data."""
    from photonics_helper.core.data import MaterialsDatabase
    from photonics_helper.raman.db import RamanDatabase

    assert RamanDatabase is MaterialsDatabase
