"""Validate the built wheel under ``dist/``.

Checks that the wheel ships the package data and licences that the packaging
config promises, and that setuptools' flat-layout discovery has not leaked
non-package top-level directories (a real failure mode when a stale ``build/``
tree is present: ``build/lib/reproductions`` once landed in the wheel).

Usage::

    python -m build --wheel
    python scripts/check_wheel.py
"""

from __future__ import annotations

import glob
import sys
import zipfile

REQUIRED_FILES = (
    "photonics_helper/materials.db",
    "photonics_helper/py.typed",
)

# Top-level directories that must never appear inside the distribution.
FORBIDDEN_TOP_LEVEL = (
    "reproductions/",
    "examples/",
    "tests/",
    "benchmarks/",
    "openspec/",
    "site/",
    "docs/",
)


def check(wheel_path: str) -> list[str]:
    names = zipfile.ZipFile(wheel_path).namelist()
    problems: list[str] = []

    for required in REQUIRED_FILES:
        if required not in names:
            problems.append(f"missing required file: {required}")

    for licence in ("LICENSE", "NOTICE"):
        if not any(n.endswith(f"licenses/{licence}") for n in names):
            problems.append(f"missing licence file in dist-info: {licence}")

    for forbidden in FORBIDDEN_TOP_LEVEL:
        leaked = [n for n in names if n.startswith(forbidden)]
        if leaked:
            problems.append(
                f"non-package directory leaked into wheel: {forbidden} "
                f"(e.g. {leaked[0]}); remove stale build/ and rebuild"
            )

    return problems


def main() -> int:
    wheels = sorted(glob.glob("dist/*.whl"))
    if not wheels:
        print("no wheel found under dist/", file=sys.stderr)
        return 2

    wheel = wheels[-1]
    problems = check(wheel)
    if problems:
        print(f"FAILED: {wheel}")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print(f"OK: {wheel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
