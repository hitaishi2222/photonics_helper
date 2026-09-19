"""Guard the stable core API: symbol set and public signature *shape*.

The committed snapshot (`tests/core_api_snapshot.json`) is the contract. The
guard fails when the live API:

- removes or adds an exported symbol (additions must be deliberate, so they
  also require a regenerated snapshot);
- removes, renames or reorders a parameter, changes a parameter's kind, or adds
  a new **required** parameter.

Adding an optional parameter is allowed (backwards compatible). Annotations and
docstrings are not compared, so dependency churn does not cause false failures.

After an intentional change: ``python scripts/update_api_snapshot.py``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = Path(__file__).resolve().parent / "core_api_snapshot.json"

import sys  # noqa: E402

if str(ROOT) not in sys.path:  # pragma: no cover - test-time bootstrap
    sys.path.insert(0, str(ROOT))

from scripts.update_api_snapshot import CORE_MODULES, build_snapshot  # noqa: E402


def _compare_params(
    expected: list[list[Any]],
    actual: list[list[Any]],
    context: str,
    errors: list[str],
) -> None:
    expected_names = [p[0] for p in expected]
    actual_names = [p[0] for p in actual]

    if actual_names[: len(expected_names)] != expected_names:
        errors.append(
            f"{context}: parameters changed: expected {expected_names}, got {actual_names}"
        )
        return

    for extra in actual[len(expected_names) :]:
        if not extra[2]:
            errors.append(f"{context}: new required parameter {extra[0]!r}")

    for exp, act in zip(expected, actual):
        if exp[2] and not act[2]:
            errors.append(f"{context}: parameter {exp[0]!r} became required")
        if exp[1] != act[1]:
            errors.append(
                f"{context}: parameter {exp[0]!r} kind changed {exp[1]} -> {act[1]}"
            )


def collect_errors(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    """Return a list of contract violations (empty means the API is compatible)."""
    errors: list[str] = []

    for module_name in expected:
        if module_name not in actual:
            errors.append(f"module missing from the live API: {module_name}")
            continue

        expected_all = set(expected[module_name]["all"])
        actual_all = set(actual[module_name]["all"])
        for missing in sorted(expected_all - actual_all):
            errors.append(f"{module_name}: exported symbol removed: {missing}")
        for added in sorted(actual_all - expected_all):
            errors.append(
                f"{module_name}: exported symbol added: {added} "
                "(regenerate the snapshot if intentional)"
            )

        for symbol, expected_entry in expected[module_name]["symbols"].items():
            actual_entry = actual[module_name]["symbols"].get(symbol)
            if actual_entry is None:
                continue  # already reported as removed
            context = f"{module_name}.{symbol}"
            if expected_entry["kind"] != actual_entry["kind"]:
                errors.append(
                    f"{context}: kind changed {expected_entry['kind']} -> {actual_entry['kind']}"
                )
                continue

            _compare_params(
                expected_entry.get("params", []),
                actual_entry.get("params", []),
                context,
                errors,
            )

            expected_members = expected_entry.get("members")
            if expected_members is None:
                continue
            actual_members = actual_entry.get("members", {})
            for member_name, expected_member in expected_members.items():
                actual_member = actual_members.get(member_name)
                if actual_member is None:
                    errors.append(f"{context}.{member_name}: member removed")
                    continue
                if expected_member["kind"] != actual_member["kind"]:
                    errors.append(
                        f"{context}.{member_name}: kind changed "
                        f"{expected_member['kind']} -> {actual_member['kind']}"
                    )
                    continue
                _compare_params(
                    expected_member.get("params", []),
                    actual_member.get("params", []),
                    f"{context}.{member_name}",
                    errors,
                )

    return errors


def _load_snapshot() -> dict[str, Any]:
    return json.loads(SNAPSHOT_PATH.read_text())


def _mutated() -> dict[str, Any]:
    """A copy of the committed snapshot, to be used as the 'live' API."""
    return copy.deepcopy(_load_snapshot())


# ─── The live API matches the committed contract ──────────────────────────


def test_live_core_api_matches_snapshot() -> None:
    errors = collect_errors(_load_snapshot(), build_snapshot())
    assert errors == [], "Stable core API changed:\n  " + "\n  ".join(errors)


def test_snapshot_covers_the_stable_modules() -> None:
    assert set(_load_snapshot()) == set(CORE_MODULES)


# ─── The guard actually catches breaking changes ──────────────────────────


def _first_module() -> str:
    return CORE_MODULES[0]


def _first_class_symbol(module: str) -> str:
    for name, entry in _load_snapshot()[module]["symbols"].items():
        if entry["kind"] == "class" and entry.get("params"):
            return name
    raise AssertionError("no class symbol with parameters in the snapshot")


def test_guard_detects_removed_symbol() -> None:
    expected = _load_snapshot()
    live = _mutated()
    module = _first_module()
    live[module]["all"] = live[module]["all"][1:]
    errors = collect_errors(expected, live)
    assert any("exported symbol removed" in e for e in errors)


def test_guard_detects_added_symbol() -> None:
    expected = _load_snapshot()
    live = _mutated()
    module = _first_module()
    live[module]["all"] = live[module]["all"] + ["ZzNewSymbol"]
    errors = collect_errors(expected, live)
    assert any("exported symbol added" in e for e in errors)


def test_guard_detects_removed_parameter() -> None:
    expected = _load_snapshot()
    live = _mutated()
    module = _first_module()
    symbol = _first_class_symbol(module)
    live[module]["symbols"][symbol]["params"] = live[module]["symbols"][symbol]["params"][:-1]
    live[module]["symbols"][symbol]["members"] = {}  # avoid member noise
    errors = collect_errors(expected, live)
    assert any("parameters changed" in e for e in errors)


def test_guard_detects_renamed_parameter() -> None:
    expected = _load_snapshot()
    live = _mutated()
    module = _first_module()
    symbol = _first_class_symbol(module)
    live[module]["symbols"][symbol]["params"][0][0] = "renamed"
    live[module]["symbols"][symbol]["members"] = {}
    errors = collect_errors(expected, live)
    assert any("parameters changed" in e for e in errors)


def test_guard_detects_new_required_parameter() -> None:
    expected = _load_snapshot()
    live = _mutated()
    module = _first_module()
    symbol = _first_class_symbol(module)
    live[module]["symbols"][symbol]["params"].append(["extra", "POSITIONAL_OR_KEYWORD", False])
    live[module]["symbols"][symbol]["members"] = {}
    errors = collect_errors(expected, live)
    assert any("new required parameter" in e for e in errors)


def test_guard_allows_optional_parameter_addition() -> None:
    expected = _load_snapshot()
    live = _mutated()
    module = _first_module()
    symbol = _first_class_symbol(module)
    live[module]["symbols"][symbol]["params"].append(
        ["extra", "POSITIONAL_OR_KEYWORD", True]
    )
    errors = collect_errors(expected, live)
    assert errors == []


def test_guard_detects_removed_member() -> None:
    expected = _load_snapshot()
    live = _mutated()
    module = "photonics_helper.core.grids"
    entry = live[module]["symbols"]["TemporalGrid"]["members"]
    entry.pop("fft")
    errors = collect_errors(expected, live)
    assert any("member removed" in e for e in errors)
