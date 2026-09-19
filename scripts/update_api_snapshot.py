"""Regenerate the core API-surface snapshot.

Records, for every stable ``photonics_helper.core`` submodule, its exported
symbol set and — for callables and classes — the *shape* of the public
signature (parameter names, kinds and whether each has a default). The shape is
deliberately annotation-free so dependency/churn does not cause false failures.

Run after an **intentional** change to the stable surface::

    python scripts/update_api_snapshot.py

The diff is then reviewable: only what changed appears.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
# Make the working tree importable when run as `python scripts/...` (sys.path[0]
# is then scripts/, not the repo root).
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SNAPSHOT_PATH = ROOT / "tests" / "core_api_snapshot.json"

CORE_MODULES = (
    "photonics_helper.core.units",
    "photonics_helper.core.constants",
    "photonics_helper.core.grids",
    "photonics_helper.core.materials",
)


def _param_shape(obj: Any) -> list[list[Any]]:
    """Return ``[[name, kind, has_default], ...]`` for a callable."""
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):
        return []
    return [
        [p.name, p.kind.name, p.default is not inspect.Parameter.empty]
        for p in signature.parameters.values()
    ]


def _member_shape(member: Any, owner: type) -> dict[str, Any]:
    if isinstance(member, (property, functools.cached_property)):
        return {"kind": "property"}
    if isinstance(member, (classmethod, staticmethod)):
        return {"kind": "callable", "params": _param_shape(getattr(owner, member.__func__.__name__))}
    if callable(member):
        return {"kind": "callable", "params": _param_shape(member)}
    return {"kind": "value"}


def _symbol_entry(obj: Any) -> dict[str, Any]:
    if isinstance(obj, type):
        members = {
            name: _member_shape(member, obj)
            for name, member in obj.__dict__.items()
            if not name.startswith("_")
        }
        return {"kind": "class", "params": _param_shape(obj), "members": members}
    if callable(obj):
        return {"kind": "function", "params": _param_shape(obj)}
    return {"kind": "value"}


def build_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for module_name in CORE_MODULES:
        module = importlib.import_module(module_name)
        exported = sorted(getattr(module, "__all__", []))
        snapshot[module_name] = {
            "all": exported,
            "symbols": {
                name: _symbol_entry(getattr(module, name)) for name in exported
            },
        }
    return snapshot


def main() -> int:
    snapshot = build_snapshot()
    SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    total = sum(len(v["all"]) for v in snapshot.values())
    print(f"Wrote {SNAPSHOT_PATH} ({len(snapshot)} modules, {total} symbols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
