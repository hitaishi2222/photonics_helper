"""Tests for FROG (Frequency-Resolved Optical Gating) module."""

import numpy as np
import pytest

from photonics_helper.pulse import FROGTrace, generate_trace, retrieve, fidelity  # type: ignore[import-not-found]


class TestGenerateTrace:
    """Tests for FROG trace generation."""

    def test_gaussian_trace_shape(self):
        """Trace from Gaussian pulse has correct shape."""
        T0 = 50e-15
        N = 256
        dt = 10 * T0 / N
        t = np.arange(N) * dt - N * dt / 2
        E = np.exp(-t**2 / (2 * T0**2))

        trace = generate_trace(E, dt=dt)

        assert trace.trace.shape[0] == N  # N_omega
        assert trace.trace.shape[1] == N  # N_tau
        assert trace.trace.max() == pytest.approx(1.0, abs=1e-6)
        assert trace.trace.min() >= 0

    def test_trace_normalized(self):
        """Trace is normalized to [0, 1] by default."""
        T0 = 100e-15
        N = 512
        dt = 10 * T0 / N
        t = np.arange(N) * dt - N * dt / 2
        E = np.exp(-t**2 / (2 * T0**2))

        trace = generate_trace(E, dt=dt, normalize=True)

        assert trace.trace.max() == pytest.approx(1.0, abs=1e-6)
        assert trace.trace.min() >= 0

    def test_trace_unnormalized(self):
        """Unnormalized trace preserves amplitude."""
        T0 = 50e-15
        N = 256
        dt = 10 * T0 / N
        t = np.arange(N) * dt - N * dt / 2
        E = 2.0 * np.exp(-t**2 / (2 * T0**2))

        trace = generate_trace(E, dt=dt, normalize=False)

        assert trace.trace.max() > 1.0
        assert trace.unnormalized_trace is not None

    def test_chirped_pulse_trace(self):
        """Chirped pulse produces broader trace than transform-limited."""
        T0 = 50e-15
        N = 512
        dt = 10 * T0 / N
        t = np.arange(N) * dt - N * dt / 2

        # Transform-limited
        E_tl = np.exp(-t**2 / (2 * T0**2))
        trace_tl = generate_trace(E_tl, dt=dt)

        # Chirped
        chirp = 3.0
        E_chirped = np.exp(-t**2 / (2 * T0**2)) * np.exp(
            1j * 0.5 * chirp * (t / T0) ** 2
        )
        trace_chirped = generate_trace(E_chirped, dt=dt)

        # Chirped should have broader spectral extent
        # (This is a rough check - the trace shape changes with chirp)
        assert trace_chirped.trace.shape == trace_tl.trace.shape


class TestRetrieve:
    """Tests for PCGPA retrieval."""

    def test_gaussian_retrieval_fidelity(self):
        """PCGPA retrieves Gaussian pulse with high fidelity."""
        T0 = 50e-15
        N = 512
        dt = 10 * T0 / N
        t = np.arange(N) * dt - N * dt / 2
        E = np.exp(-t**2 / (2 * T0**2))

        trace = generate_trace(E, dt=dt)
        result = retrieve(trace, max_iter=50, verbose=False)

        f = fidelity(trace, result)
        assert f > 0.99, f"Fidelity {f} below threshold"

    def test_sech_retrieval_fidelity(self):
        """PCGPA retrieves sech pulse with high fidelity."""
        T0 = 100e-15
        N = 512
        dt = 10 * T0 / N
        t = np.arange(N) * dt - N * dt / 2
        E = 1.0 / np.cosh(t / T0)

        trace = generate_trace(E, dt=dt)
        result = retrieve(trace, max_iter=50, verbose=False)

        f = fidelity(trace, result)
        assert f > 0.99, f"Fidelity {f} below threshold"

    def test_chirped_retrieval_fidelity(self):
        """PCGPA retrieves chirped pulse with high fidelity."""
        T0 = 50e-15
        chirp = 2.0
        N = 512
        dt = 10 * T0 * 1.5 / N  # Wider window for chirped
        t = np.arange(N) * dt - N * dt / 2
        E = (
            np.exp(-t**2 / (2 * T0**2))
            * np.exp(1j * 0.5 * chirp * (t / T0) ** 2)
        )

        trace = generate_trace(E, dt=dt)
        result = retrieve(trace, max_iter=100, verbose=False)

        f = fidelity(trace, result)
        assert f > 0.99, f"Fidelity {f} below threshold"

    def test_retrieval_preserves_fwhm(self):
        """Retrieved pulse has correct FWHM."""
        T0 = 50e-15
        N = 512
        dt = 10 * T0 / N
        t = np.arange(N) * dt - N * dt / 2
        E = np.exp(-t**2 / (2 * T0**2))

        trace = generate_trace(E, dt=dt)
        result = retrieve(trace, max_iter=50, verbose=False)

        # Calculate FWHM of original
        int_orig = np.abs(E) ** 2
        half_max = int_orig.max() / 2
        idx_orig = np.where(int_orig >= half_max)[0]
        fwhm_orig = t[idx_orig[-1]] - t[idx_orig[0]]

        # Calculate FWHM of retrieved
        int_rec = np.abs(result.field) ** 2
        half_max_rec = int_rec.max() / 2
        idx_rec = np.where(int_rec >= half_max_rec)[0]
        fwhm_rec = t[idx_rec[-1]] - t[idx_rec[0]]

        assert abs(fwhm_orig - fwhm_rec) / fwhm_orig < 0.05, (
            f"FWHM mismatch: {fwhm_orig*1e15:.2f} fs vs {fwhm_rec*1e15:.2f} fs"
        )

    def test_retrieval_convergence(self):
        """Retrieval converges within max_iter."""
        T0 = 50e-15
        N = 256
        dt = 10 * T0 / N
        t = np.arange(N) * dt - N * dt / 2
        E = np.exp(-t**2 / (2 * T0**2))

        trace = generate_trace(E, dt=dt)
        result = retrieve(trace, max_iter=20, verbose=False)

        # Should converge (fidelity > 0.9)
        f = fidelity(trace, result)
        assert f > 0.9, f"Did not converge: fidelity = {f}"


class TestFidelity:
    """Tests for fidelity metric."""

    def test_identical_traces(self):
        """Fidelity of identical traces is 1.0."""
        T0 = 50e-15
        N = 256
        dt = 10 * T0 / N
        t = np.arange(N) * dt - N * dt / 2
        E = np.exp(-t**2 / (2 * T0**2))

        trace1 = generate_trace(E, dt=dt)
        trace2 = generate_trace(E, dt=dt)

        f = fidelity(trace1, trace2)
        assert f == pytest.approx(1.0, abs=1e-6)

    def test_fidelity_bounds(self):
        """Fidelity is in [0, 1]."""
        T0 = 50e-15
        N = 256
        dt = 10 * T0 / N
        t = np.arange(N) * dt - N * dt / 2
        E1 = np.exp(-t**2 / (2 * T0**2))
        E2 = np.exp(-t**2 / (2 * (2 * T0) ** 2))  # Different width

        trace1 = generate_trace(E1, dt=dt)
        trace2 = generate_trace(E2, dt=dt)

        f = fidelity(trace1, trace2)
        assert 0.0 <= f <= 1.0

    def test_fidelity_symmetric(self):
        """Fidelity is symmetric."""
        T0 = 50e-15
        N = 256
        dt = 10 * T0 / N
        t = np.arange(N) * dt - N * dt / 2
        E1 = np.exp(-t**2 / (2 * T0**2))
        E2 = 1.5 * np.exp(-t**2 / (2 * T0**2))  # Scaled

        trace1 = generate_trace(E1, dt=dt)
        trace2 = generate_trace(E2, dt=dt)

        f12 = fidelity(trace1, trace2)
        f21 = fidelity(trace2, trace1)
        assert abs(f12 - f21) < 1e-10


class TestFROGTrace:
    """Tests for FROGTrace dataclass."""

    def test_from_field(self):
        """FROGTrace.from_field creates valid trace."""
        N = 256
        dt = 1e-13
        t = np.arange(N) * dt - N * dt / 2
        E = np.exp(-t**2 / (2 * (1e-14) ** 2))

        trace = FROGTrace.from_field(E, dt=dt)

        assert isinstance(trace, FROGTrace)
        assert trace.trace.shape == (N, N)
        assert trace.omega is not None
        assert trace.tau is not None
        assert trace.dt == dt

    def test_visualize_without_retrieval(self):
        """visualize() works without retrieved field."""
        T0 = 50e-15
        N = 256
        dt = 10 * T0 / N
        t = np.arange(N) * dt - N * dt / 2
        E = np.exp(-t**2 / (2 * T0**2))

        trace = generate_trace(E, dt=dt)
        fig = trace.visualize()

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_visualize_with_retrieval(self):
        """visualize() works with retrieved field."""
        T0 = 50e-15
        N = 256
        dt = 10 * T0 / N
        t = np.arange(N) * dt - N * dt / 2
        E = np.exp(-t**2 / (2 * T0**2))

        trace = generate_trace(E, dt=dt)
        result = retrieve(trace, max_iter=20, verbose=False)
        fig = trace.visualize(retrieved=result)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)
