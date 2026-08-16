"""Tests for the FFTW3 backend (:mod:`photonics_helper._fftw`).

Verifies that the backend (a) matches numpy.fft conventions exactly, (b) round
trips through :class:`TemporalGrid`, (c) reproduces scipy's fftconvolve, and
(d) produces bit-comparable GNLSE results to the pure-numpy path.
"""

import numpy as np
import pytest

from photonics_helper import _fftw
from photonics_helper._fftw import (
    available,
    backend_name,
    convolve_full,
    fft,
    ifft,
    set_backend,
)
from photonics_helper.base import Time
from photonics_helper.pulse import TemporalGrid

SIZES = [64, 127, 128, 255, 256, 512]


def _random_complex(n: int) -> np.ndarray:
    rng = np.random.default_rng(42 + n)
    return rng.standard_normal(n) + 1j * rng.standard_normal(n)


@pytest.mark.parametrize("n", SIZES)
def test_fft_matches_numpy(n: int):
    A = _random_complex(n)
    expected = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(A)))
    assert np.allclose(fft(A), expected, atol=1e-12)


@pytest.mark.parametrize("n", SIZES)
def test_ifft_matches_numpy(n: int):
    A = _random_complex(n)
    expected = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(A)))
    assert np.allclose(ifft(A), expected, atol=1e-12)


@pytest.mark.parametrize("n", SIZES)
def test_round_trip(n: int):
    A = _random_complex(n)
    assert np.max(np.abs(ifft(fft(A)) - A)) < 1e-11


def test_real_input_upcast():
    A = np.random.default_rng(0).standard_normal(128)
    assert fft(A).dtype == np.complex128


def test_temporal_grid_round_trip():
    """The solver-facing entry point must stay exact (<1e-10 per test_pulse)."""
    grid = TemporalGrid(N=1024, Tmax=Time(200e-12, "s"))
    t0 = 20e-12
    A_t = np.exp(-(grid.t**2) / (2 * t0**2))
    A_w = grid.fft(A_t)
    A_rec = grid.ifft(A_w)
    assert np.max(np.abs(A_rec - A_t)) < 1e-10


@pytest.mark.parametrize("la,lb", [(256, 64), (255, 63), (128, 128), (1024, 7)])
def test_convolve_full_matches_scipy(la: int, lb: int):
    scipy_signal = pytest.importorskip("scipy.signal")
    rng = np.random.default_rng(la + lb)
    a = rng.standard_normal(la)
    b = rng.standard_normal(lb)
    expected = scipy_signal.fftconvolve(a, b, mode="full")
    assert np.allclose(convolve_full(a, b), expected, atol=1e-11)

    ac = a + 1j * rng.standard_normal(la)
    bc = b + 1j * rng.standard_normal(lb)
    expected_c = scipy_signal.fftconvolve(ac, bc, mode="full")
    assert np.allclose(convolve_full(ac, bc), expected_c, atol=1e-11)


def test_backend_reports_fftw_when_installed():
    """Auto-selection picks the fastest available backend."""
    try:
        import pyfftw  # noqa: F401

        assert available() is True
        assert "FFTW3" in backend_name()
    except ImportError:
        assert available() is True
        assert backend_name() in ("scipy.fft (pocketfft)", "scipy.fft (pocketfft, workers=1)")


def _parity(name: str):
    """Assert the named backend matches numpy.fft conventions."""
    set_backend(name)
    for n in [64, 127, 255]:
        A = _random_complex(n)
        assert np.allclose(fft(A), np.fft.fftshift(np.fft.fft(np.fft.ifftshift(A))), atol=1e-12)
        assert np.allclose(ifft(A), np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(A))), atol=1e-12)
        assert np.max(np.abs(ifft(fft(A)) - A)) < 1e-11


@pytest.mark.parametrize("name", ["fftw", "scipy", "numpy"])
def test_backend_selection_parity(name: str):
    """Every selectable backend honours the same shifted conventions."""
    try:
        _parity(name)
    finally:
        set_backend(None)  # restore auto-selection


@pytest.mark.parametrize("name", ["fftw", "scipy", "numpy"])
def test_backend_convolve_parity(name: str):
    """Every backend's full convolution matches scipy.signal.fftconvolve."""
    scipy_signal = pytest.importorskip("scipy.signal")
    rng = np.random.default_rng(7)
    a = rng.standard_normal(256)
    b = rng.standard_normal(64)
    expected = scipy_signal.fftconvolve(a, b, mode="full")
    try:
        set_backend(name)
        assert np.allclose(convolve_full(a, b), expected, atol=1e-11)
    finally:
        set_backend(None)


def test_invalid_backend_raises():
    with pytest.raises(ValueError, match="Unknown FFT backend"):
        set_backend("bogus")


def test_fallback_chain_without_pyfftw(monkeypatch):
    """Simulating a pyfftw-less install: scipy (or numpy) takes over silently."""
    try:
        import pyfftw  # noqa: F401
    except ImportError:
        pytest.skip("pyfftw not installed — nothing to simulate")

    monkeypatch.setattr(_fftw, "_HAS_PYFFTW", False)
    set_backend(None)
    try:
        assert backend_name().startswith("scipy")
        A = np.random.default_rng(0).standard_normal(128)
        assert np.max(np.abs(ifft(fft(A)) - A)) < 1e-12
    finally:
        monkeypatch.undo()
        set_backend(None)


def test_env_override(monkeypatch):
    """PHOTONICS_FFT_BACKEND forces a specific backend on import."""
    monkeypatch.setenv("PHOTONICS_FFT_BACKEND", "numpy")
    # Re-run module selection in a clean subprocess to mimic a fresh import.
    import subprocess
    import sys

    code = (
        "import photonics_helper._fftw as f; "
        "print(f.backend_name()); "
        "assert 'numpy' in f.backend_name()"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "numpy" in result.stdout


def test_gnlse_matches_numpy_path():
    """FFTW and numpy paths must give near-identical propagation results."""
    try:
        import pyfftw  # noqa: F401
    except ImportError:
        pytest.skip("pyfftw not installed")

    from photonics_helper.gnlse import FiberProfile, GNLSESolver
    from photonics_helper.pulse import Envelope, Wave, TemporalGrid
    from photonics_helper.base import Wavelength, Time, Length, Area
    from photonics_helper.raman import RamanResponse, RamanSpec

    grid = TemporalGrid(N=2**11, Tmax=Time(10e-12, "s"))
    env = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=Time(100, "fs"))
    wl = Wavelength(1550, "nm")
    pulse = Wave(grid=grid, envelope=env, central_wavelength=wl)

    spec = RamanSpec(name="Silica", raman_shift_cm=440, raman_linewidth_cm=45, fR=0.18)
    raman = RamanResponse(spec=spec, grid=grid)

    fiber = FiberProfile(
        n2=2.6e-20,
        alpha=0.0,
        A_eff=Area(80e-12, "m^2"),
        length=Length(10e-3, "m"),
        raman_response=raman,
    )
    betas = np.array([-0.02, 1e-3])  # β₂, β₃ in ps²/m, ps³/m

    def run():
        solver = GNLSESolver(
            pulse=pulse,
            fiber=fiber,
            betas=betas,
            include_raman=True,
            include_self_steepening=True,
        )
        solver.propagate(200, nsaves=20)
        return solver.evolution[-1].envelope_field

    set_backend("fftw")
    try:
        fftw_result = run()
        set_backend("numpy")
        numpy_result = run()
    finally:
        set_backend(None)

    err = np.max(np.abs(fftw_result - numpy_result))
    # FFTW vs pocketfft rounding: ~1e-14 relative per transform, accumulated
    # over 200 steps × 2 linear + Raman FFTs. 2e-8 is generous but still proves
    # the solver follows the same trajectory.
    assert err / np.max(np.abs(numpy_result)) < 2e-8
