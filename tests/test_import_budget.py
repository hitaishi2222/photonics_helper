"""Import-cost and packaging guards for the foundation layer.

The library's stated goal is to provide dependency-light foundation primitives
(units, constants, grids, materials) that other projects can build on — see the
"Foundation Backbone" roadmap in ``REPORT.md``. Two properties matter:

1. Importing the package or its foundation module must not drag in the
   plotting/web stack (matplotlib, plotly, dash). ``photonics_helper`` is a
   lazy package (PEP 562), so this holds.

2. The built wheel must contain its data files (``materials.db``, ``py.typed``)
   and licences, and must not leak non-package top-level directories.

Note: ``rich`` may be pulled in transitively by pydantic when the optional
``logfire`` package is installed (pydantic's dataclass plugin imports it), so it
is deliberately not asserted here.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Top-level third-party modules that must NOT be pulled in by the foundation.
HEAVY_MODULES = ("matplotlib", "plotly", "dash")

# Deliberately loose: this is a smoke budget, not a benchmark.
IMPORT_BUDGET_S = 1.0


def _cold_import(statement: str) -> tuple[float, list[str]]:
    """Import ``statement`` in a fresh interpreter, return (seconds, heavy mods)."""
    snippet = textwrap.dedent(
        f"""
        import json, sys, time
        _t0 = time.perf_counter()
        {statement}
        _elapsed = time.perf_counter() - _t0
        _heavy = sorted(m for m in {HEAVY_MODULES!r} if m in sys.modules)
        print("@@" + json.dumps({{"elapsed": _elapsed, "heavy": _heavy}}))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        check=True,
    )
    line = next(line for line in proc.stdout.splitlines() if line.startswith("@@"))
    payload = json.loads(line[2:])
    return float(payload["elapsed"]), list(payload["heavy"])


# ─── Import budget ─────────────────────────────────────────────────────────


def test_base_import_does_not_load_plotting() -> None:
    elapsed, heavy = _cold_import("import photonics_helper.base")
    assert heavy == [], f"importing photonics_helper.base loaded {heavy}"
    assert elapsed < IMPORT_BUDGET_S, f"photonics_helper.base import took {elapsed:.2f}s"


def test_package_import_does_not_load_plotting() -> None:
    elapsed, heavy = _cold_import("import photonics_helper")
    assert heavy == [], f"importing photonics_helper loaded {heavy}"
    assert elapsed < IMPORT_BUDGET_S, f"photonics_helper import took {elapsed:.2f}s"


# ─── Packaging guards (green now; protect the Phase 0 work) ────────────────


def test_license_and_notice_exist() -> None:
    assert (REPO_ROOT / "LICENSE").is_file(), "MIT LICENSE file is missing"
    assert (REPO_ROOT / "NOTICE").is_file(), "data-provenance NOTICE is missing"


def test_pyproject_declares_wheel_contents() -> None:
    import tomllib

    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    project = data["project"]

    # PEP 639 licence expression + files (drives dist-info/licenses/*).
    assert project["license"] == "MIT"
    assert project["license-files"] == ["LICENSE", "NOTICE"]

    # Data shipped inside the package.
    package_data = data["tool"]["setuptools"]["package-data"]["photonics_helper"]
    assert "materials.db" in package_data
    assert "py.typed" in package_data

    # Flat layout: only photonics_helper ships (not reproductions/, examples/…).
    find = data["tool"]["setuptools"]["packages"]["find"]
    assert find["include"] == ["photonics_helper*"]


def test_public_alias_spelling() -> None:
    import photonics_helper

    assert "Permeability" in photonics_helper.__all__
    assert "Permiability" not in photonics_helper.__all__
