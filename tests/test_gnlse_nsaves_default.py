"""The cheap two-snapshot default and the explicit all-step diagnostic."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from photonics_helper.base import Length, Time, Wavelength
from photonics_helper.gnlse import Fiber, SplitStepEngine
from photonics_helper.pulse import Envelope, TemporalGrid, Wave


def _engine(steps: int, n: int = 256):
    grid = TemporalGrid(N=n, Tmax=Time(8e-12, "s"))
    env = Envelope.from_energy(
        shape="sech", energy=10e-12, fwhm=Time(200e-15, "s")
    )
    wave = Wave(
        grid=grid, envelope=env, central_wavelength=Wavelength(1550e-9, "m")
    )
    fiber = Fiber(gamma=0.05, length=Length(1e-3, "m"))
    return SplitStepEngine(
        wave,
        fiber,
        betas=np.array([-20.0]),
        include_raman=False,
        step_size=Length(fiber.length.as_m / steps, "m"),
    )


def test_default_is_two_snapshots_and_final_result_is_unchanged():
    default = _engine(50)
    diagnostic = _engine(50)
    default.propagate(50)
    diagnostic.propagate(50, nsaves=None)
    assert len(default.evolution) == 2
    assert len(diagnostic.evolution) == 51
    assert np.array_equal(np.asarray(default.A), np.asarray(diagnostic.A))
    assert np.array_equal(default.energy_vs_z[-1], diagnostic.energy_vs_z[-1])


def test_explicit_none_and_nsaves_validation_are_preserved():
    engine = _engine(5)
    engine.propagate(5, nsaves=None)
    assert len(engine.evolution) == 6
    with pytest.raises(ValueError, match="nsaves must be >= 2"):
        _engine(5).propagate(5, nsaves=1)


def test_default_peak_rss_is_flat_in_step_count():
    code = """
import json, resource
import numpy as np
from photonics_helper.base import Length, Time, Wavelength
from photonics_helper.gnlse import Fiber, SplitStepEngine
from photonics_helper.pulse import Envelope, TemporalGrid, Wave
steps = int(__import__('sys').argv[1])
grid = TemporalGrid(N=2048, Tmax=Time(8e-12, 's'))
env = Envelope.from_energy(shape='sech', energy=1e-12, fwhm=Time(200e-15, 's'))
wave = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(1550e-9, 'm'))
fiber = Fiber(gamma=0.0, length=Length(1e-3, 'm'))
engine = SplitStepEngine(wave, fiber, betas=np.array([0.0]),
    step_size=Length(fiber.length.as_m / steps, 'm'))
engine.propagate(steps)
print(json.dumps({'rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  'snapshots': len(engine.evolution)}))
"""
    results = []
    for steps in (200, 2000):
        proc = subprocess.run(
            [sys.executable, "-c", code, str(steps)],
            check=True,
            capture_output=True,
            text=True,
        )
        results.append(json.loads(proc.stdout))
    assert all(item["snapshots"] == 2 for item in results)
    ratio = results[1]["rss_kb"] / results[0]["rss_kb"]
    assert ratio < 1.2, f"peak RSS grew {ratio:.3f}x: {results}"
