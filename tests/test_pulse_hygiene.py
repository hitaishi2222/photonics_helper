"""Regression tests for pulse/visualization hygiene (review items + N8)."""

from pathlib import Path

import numpy as np
import pytest

from photonics_helper.base import Frequency, Time, Wavelength
from photonics_helper.pulse import Envelope, TemporalGrid, Wave

ROOT = Path(__file__).resolve().parents[1]


def _grid(N=2**10, Tmax=20e-12):
    return TemporalGrid(N=N, Tmax=Time(Tmax, "s"))


def test_with_field_sets_envelope_field():
    grid = _grid()
    wave = Wave(
        grid=grid,
        envelope=Envelope(
            shape="gaussian", peak_amplitude=1.0, pulse_width=Time(50e-15, "s")
        ),
        central_wavelength=Wavelength(1550, "nm"),
    )
    custom = np.exp(-((grid.t / 1e-13) ** 2)).astype(complex)
    returned = wave.with_field(custom)
    assert returned is wave
    np.testing.assert_allclose(wave.envelope_field, custom)


def test_pulse_train_uses_first_class_field():
    grid = _grid(N=2**12, Tmax=40e-12)
    env = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=Time(50e-15, "s"))
    wave = Wave.from_pulse_train(
        envelope=env,
        central_wavelength=Wavelength(1550, "nm"),
        grid=grid,
        repetition_rate=Frequency(50, "GHz"),
        n_pulses=5,
    )
    field = wave.envelope_field
    assert field.shape == grid.t.shape
    # Multiple distinct pulses should be present (not the single-envelope field).
    assert not np.allclose(field, env.field(grid.t))
    assert np.count_nonzero(np.abs(field) > 1e-6) > 0


def test_xy_backend_removed():
    """The undeclared ``xy`` backend was dropped; selecting it now errors."""
    env = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=Time(50e-15, "s"))
    with pytest.raises(ValueError, match="plotly"):
        env.visualize_2d(backend="xy")  # type: ignore[arg-type]
    assert not hasattr(env, "_visualize_2d_xy")


def test_fiber_has_no_rich_traceback_side_effect():
    src = (ROOT / "photonics_helper" / "fiber.py").read_text()
    assert "rich.traceback" not in src
