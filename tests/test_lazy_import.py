"""Tests for the lazy package surface (PEP 562).

The package ``__init__`` resolves public names from ``_LAZY_MODULES`` on first
access. These tests pin that contract:

- every name in ``__all__`` resolves to the same object as its defining module;
- ``from photonics_helper import *`` works and binds every name;
- the public surface matches a committed snapshot, so an accidental removal (or
  a silent addition) fails CI and must be a deliberate change;
- importing the package does not import the heavy satellite modules.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import photonics_helper

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = Path(__file__).resolve().parent / "public_api_snapshot.json"


def test_lazy_map_covers_all_exactly() -> None:
    """Every exported name has a lazy source, and the map has no extras."""
    assert set(photonics_helper._LAZY_MODULES) == set(photonics_helper.__all__)


def test_every_public_name_resolves_to_defining_module() -> None:
    for name in photonics_helper.__all__:
        module_name = photonics_helper._LAZY_MODULES[name]
        attr = photonics_helper._LAZY_ATTR.get(name, name)
        module = importlib.import_module(module_name, "photonics_helper")
        assert getattr(photonics_helper, name) is getattr(module, attr), name


def test_star_import_binds_every_name() -> None:
    namespace: dict[str, object] = {}
    exec("from photonics_helper import *", namespace)  # noqa: S102 - intentional
    bound = {k for k in namespace if not k.startswith("__")}
    assert bound == set(photonics_helper.__all__)


def test_dir_matches_all() -> None:
    assert dir(photonics_helper) == sorted(photonics_helper.__all__)


def test_public_surface_matches_snapshot() -> None:
    expected = json.loads(SNAPSHOT_PATH.read_text())
    assert sorted(photonics_helper.__all__) == expected, (
        "The public surface changed. If intentional, regenerate "
        "tests/public_api_snapshot.json; if not, restore the removed name."
    )


def test_unknown_attribute_raises_attribute_error() -> None:
    try:
        photonics_helper.definitely_not_a_public_name  # noqa: B018
    except AttributeError as exc:
        assert "definitely_not_a_public_name" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected AttributeError for an unknown attribute")


def test_package_import_does_not_load_satellite_modules() -> None:
    """A bare package import must not pull in the simulation/plotting modules."""
    satellites = (
        "photonics_helper.gnlse",
        "photonics_helper.chi2",
        "photonics_helper.dbr",
        "photonics_helper.raman",
        "photonics_helper.dashboard",
        "photonics_helper.structured",
        "photonics_helper.pulse",
        "photonics_helper.phase_matching",
    )
    snippet = (
        "import sys, photonics_helper; "
        f"print([m for m in {satellites!r} if m in sys.modules])"
    )
    proc = subprocess.run(
        [sys.executable, "-c", snippet], capture_output=True, text=True, check=True
    )
    assert proc.stdout.strip() == "[]", f"import pulled in {proc.stdout.strip()}"
