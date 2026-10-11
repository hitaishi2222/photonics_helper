"""Backing-backend UX tests: ``_ensure_showable_backend`` behavior.

A user script that forces ``matplotlib.use("Agg")`` (or inherits
``MPLBACKEND=Agg``) silently no-ops ``plt.show()``, which used to look like
"visualize() didn't open a window". The visualizers now auto-switch to an
interactive backend when the intent is display (no ``save_path``), and leave
the headless backend untouched when saving. These tests pin that contract.
"""

import subprocess
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


_SWITCH_PROBE = """
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

for candidate in {candidates!r}:
    try:
        plt.switch_backend(candidate)
    except Exception:
        continue
    raise SystemExit(0)
raise SystemExit(1)
"""


def _interactive_backend_available() -> bool:
    """True if at least one candidate backend can actually be loaded here.

    Module presence (``find_spec``) is not enough: tkinter ships with
    CPython but ``switch_backend`` still raises ``ImportError`` on a headless
    machine, so probe a real switch in a throwaway subprocess instead.
    """
    probe = _SWITCH_PROBE.format(candidates=INTERACTIVE_BACKEND_CANDIDATES)
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        timeout=60,
    )
    return result.returncode == 0


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

    def test_noop_on_interactive_backend(self, monkeypatch, restore_backend):
        """An already-interactive backend must not be switched away from.

        The no-op contract is tested deterministically by reporting an
        interactive backend and recording any switch attempt, so it runs
        even on headless machines where no real interactive backend exists.
        """
        monkeypatch.setattr(plt, "get_backend", lambda: "TkAgg")
        calls: list = []
        monkeypatch.setattr(
            plt, "switch_backend", lambda *a, **k: calls.append((a, k))
        )
        from photonics_helper.pulse import _ensure_showable_backend

        _ensure_showable_backend()
        assert calls == []

    @requires_interactive_toolkit
    def test_noop_on_interactive_backend_end_to_end(self, restore_backend):
        """End-to-end: visualize() leaves a working interactive backend alone."""
        before = matplotlib.get_backend()
        if before.lower() in HEADLESS_BACKENDS:
            for candidate in INTERACTIVE_BACKEND_CANDIDATES:
                try:
                    plt.switch_backend(candidate)
                    before = candidate
                    break
                except Exception:
                    continue
            else:  # pragma: no cover - gated by availability probe
                pytest.fail("availability probe passed but no backend switchable")
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
