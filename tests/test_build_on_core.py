"""The build-on-core example must run on the core alone and be correct.

Runs `examples/33_build_on_core.py` in a fresh interpreter and asserts that it
(a) imports no satellite module and (b) reproduces the closed-form dispersive
broadening ``sqrt(1 + (z/L_D)^2)``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "33_build_on_core.py"

SATELLITES = (
    "photonics_helper.gnlse",
    "photonics_helper.chi2",
    "photonics_helper.dbr",
    "photonics_helper.raman",
    "photonics_helper.pulse",
    "photonics_helper.phase_matching",
    "photonics_helper.structured",
    "photonics_helper.dashboard",
)


def _run_in_subprocess() -> dict[str, object]:
    code = (
        "import json, runpy, sys\n"
        f"module = runpy.run_path({str(EXAMPLE)!r})\n"
        "module['main']()\n"
        "sat = [m for m in "
        f"{SATELLITES!r} if m in sys.modules]\n"
        "result = module['run']()\n"
        "print('@@' + json.dumps({'sat': sat, 'max_error': result['max_error'], "
        "'is_material': result['is_optical_material']}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("@@"))
    return json.loads(line[2:])


def test_example_imports_no_satellites() -> None:
    payload = _run_in_subprocess()
    assert payload["sat"] == [], f"example imported satellites: {payload['sat']}"


def test_example_matches_analytic_broadening() -> None:
    payload = _run_in_subprocess()
    assert payload["max_error"] < 1e-9, payload


def test_example_material_satisfies_protocol() -> None:
    payload = _run_in_subprocess()
    assert payload["is_material"] is True
