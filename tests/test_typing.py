"""Regression guard for PEP 561 typing metadata (review N5).

The package previously shipped hand-maintained `.pyi` stubs that had drifted
from the runtime API (e.g. a `DispersiveWaveResult` with non-existent fields).
It now ships inline annotations plus a `py.typed` marker.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "photonics_helper"
PYPROJECT = ROOT / "pyproject.toml"


def test_py_typed_marker_exists():
    assert (PKG / "py.typed").is_file()


def test_py_typed_is_declared_as_package_data():
    text = PYPROJECT.read_text()
    assert "py.typed" in text


def test_no_stale_stub_files():
    stubs = list(PKG.glob("*.pyi"))
    assert stubs == [], f"stale .pyi stubs found: {stubs}"
