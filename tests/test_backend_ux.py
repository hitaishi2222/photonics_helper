"""Backing-backend UX tests: ``_ensure_showable_backend`` behavior.

A user script that forces ``matplotlib.use("Agg")`` (or inherits
``MPLBACKEND=Agg``) silently no-ops ``plt.show()``, which used to look like
"visualize() didn't open a window". The visualizers now auto-switch to an
interactive backend when the intent is display (no ``save_path``), and leave
the headless backend untouched when saving. These tests pin that contract.
"""

import sys
import tempfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from photonics_helper import Frequency, TemporalGrid, Wavelength
from photonics_helper.pulse import (
    HEADLESS_BACKENDS,
    INTERACTIVE_BACKEND_CANDIDATES,
    Envelope,
    FROGTrace,
    Time,
    Wave,
)


@pytest.fixture()
def restore_backend():
    """Save/restore the session backend around each test."""
    backend = matplotlib.get_backend()
    yield backend
    plt.switch_backend(backend)


@pytest.fixture()
def forced_agg(restore_backend):
    """Run the test under an explicit headless backend."""
    matplotlib.use("Agg", force=True)
    return "Agg"


def _interactive_backend_available() -> bool:
    """True if at least one candidate toolkit can actually be imported."""
    import importlib.util

    mods = {
        "qtagg": "qtpy",
        "TkAgg": "tkinter",
        "gtk4agg": "gtk",
        "gtk3agg": "gi",
        "wxagg": "wx",
    }
    return any(
        importlib.util.find_spec(m) is not None for m in mods.values()
    ) or sys.platform == "darwin"  # macosx built-in


requires_interactive_toolkit = pytest.mark.skipif(
    not _interactive_backend_available(),
    reason="no interactive matplotlib backend is installed; switching is "
    "expected to fail and leave the backend headless",
)


def _make_wave():
    enp = Envelope.from_fwhm("sech", 1.0, Time(50, "fs"))
    grid = TemporalGrid(2**8, Time(2e-12, "s"))
    return Wave.from_pulse_train(
        envelope=enp,
        central_wavelength=Wavelength(1550, "nm"),
        grid=grid,
        repetition_rate=Frequency(50, "GHz"),
        n_pulses=1,
    )


class TestEnsureShowableBackend:
    @requires_interactive_toolkit
    def test_switches_off_agg_when_displaying(self, forced_agg):
        assert matplotlib.get_backend().lower() == "agg"
        w = _make_wave()
        w.visualize(save_path=None)
        assert matplotlib.get_backend().lower() not in HEADLESS_BACKENDS

    def test_save_path_keeps_headless_backend(self, forced_agg):
        """Saving is display-agnostic: no backend side effects and the file exists."""
        w = _make_wave()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "wave.png"
            w.visualize(save_path=str(out))
            assert out.exists()
            assert matplotlib.get_backend().lower() == "agg"

    def test_noop_on_interactive_backend(self, restore_backend):
        """An already-interactive backend must not be switched away from.

        The session may legitimately start headless (other test modules run
        under Agg), so first park it on an interactive backend we control,
        then require visualize() not to change it.
        """
        before = matplotlib.get_backend()
        if before.lower() in HEADLESS_BACKENDS:
            before = "TkAgg"  # tkinter ships with CPython — guaranteed present
            plt.switch_backend(before)
        w = _make_wave()
        w.visualize()
        assert matplotlib.get_backend() == before

    @requires_interactive_toolkit
    def test_frog_visualize_switches_too(self, forced_agg):
        N, T0 = 256, 50e-15
        dt = 10 * T0 / N
        t = np.arange(N) * dt - N * dt / 2
        E = np.exp(-(t**2) / (2 * T0**2))
        trace = FROGTrace.from_field(E, dt=dt)
        trace.visualize()
        assert matplotlib.get_backend().lower() not in HEADLESS_BACKENDS

    def test_default_candidates_start_with_qt(self):
        """Qt binds to the user's desktop session most broadly (incl. Wayland)."""
        assert INTERACTIVE_BACKEND_CANDIDATES[0].lower().startswith("qt")
