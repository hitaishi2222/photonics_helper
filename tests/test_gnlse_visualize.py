"""Tests for photonics_helper.gnlse.visualize."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from photonics_helper.base import Area, Length, Time, Wavelength
from photonics_helper.gnlse import (
    FiberProfile,
    GNLSESolver,
    plot_intensity_metrics,
    plot_spectrum_vs_distance,
    plot_waterfall,
)
from photonics_helper.pulse import Envelope, TemporalGrid, Wave


@pytest.fixture
def solver():
    """Create a propagated solver for testing."""
    grid = TemporalGrid(N=256, Tmax=Time(20e-12, "s"))
    env = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=Time(1, "ps"))
    pulse = Wave(
        grid=grid,
        envelope=env,
        central_wavelength=Wavelength(1550, "nm"),
    )
    fiber = FiberProfile(
        n2=1e-19, alpha=1e-5, A_eff=Area(5e-11, "m^2"), length=Length(1e-3, "m")
    )
    betas = np.array([0.02])  # 20 ps²/km
    solver = GNLSESolver(pulse=pulse, fiber=fiber, betas=betas, include_raman=False)
    solver.propagate(num_steps=10)
    return solver


def test_waterfall_plot_returns_figure(solver):
    """waterfall plot returns a matplotlib Figure."""
    fig = plot_waterfall(solver)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_waterfall_plot_has_axis_labels(solver):
    """waterfall plot has axis labels."""
    fig = plot_waterfall(solver)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Time (ps)"
    assert ax.get_ylabel() == "Propagation distance (z)"
    # Each y-tick is labeled with its trace distance; colorbar repeats it.
    cb_labels = [a.get_ylabel() for a in fig.axes[1:]]
    assert any("Propagation distance" in lbl for lbl in cb_labels)
    plt.close(fig)


def test_spectrum_vs_distance_plot_returns_figure(solver):
    """spectrum vs distance plot returns a matplotlib Figure."""
    fig = plot_spectrum_vs_distance(solver)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_spectral_evolution_plot_returns_figure(solver):
    """spectral evolution contour returns a matplotlib Figure."""
    from photonics_helper.gnlse import plot_spectral_evolution

    fig = plot_spectral_evolution(solver, wl_min=1400, wl_max=1700, dynamic_range_db=30)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_spectral_evolution_grid_shape(solver):
    """spectral_evolution_on_wavelength_grid returns consistent shapes."""
    from photonics_helper.gnlse import spectral_evolution_on_wavelength_grid

    z, wl, spec = spectral_evolution_on_wavelength_grid(solver, 1400, 1700, 100)
    assert len(z) == spec.shape[0]
    assert len(wl) == spec.shape[1] == 100


def test_spectrum_vs_distance_has_axis_labels(solver):
    """spectrum vs distance plot has axis labels."""
    fig = plot_spectrum_vs_distance(solver)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Wavelength (nm)"
    assert ax.get_ylabel() == "Propagation distance (mm)"
    plt.close(fig)


def test_intensity_metrics_returns_figure(solver):
    """intensity metrics plot returns a matplotlib Figure."""
    fig = plot_intensity_metrics(solver)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_intensity_metrics_has_two_subplots(solver):
    """intensity metrics plot has two subplots."""
    fig = plot_intensity_metrics(solver)
    assert len(fig.axes) == 2
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cross-correlation spectrogram (gnlse_spectrogram / plot_spectrogram)
# ---------------------------------------------------------------------------


def test_spectrogram_shapes(solver):
    """gnlse_spectrogram returns (delay, wavelength, trace) on the requested grid."""
    from photonics_helper.gnlse import gnlse_spectrogram

    delays, wl, trace = gnlse_spectrogram(
        solver, n_delays=21, wl_bounds=(1400.0, 1700.0), n_wavelength=128
    )
    assert delays.shape == (21,)
    assert wl.shape == (128,)
    assert trace.shape == (21, 128)
    assert wl[0] == pytest.approx(1400.0)
    assert wl[-1] == pytest.approx(1700.0)
    assert np.all(np.diff(wl) > 0)
    assert trace.max() > 0.0


def test_spectrogram_peaks_at_zero_delay_for_initial_pulse(solver):
    """Self-gating the initial pulse puts the spectrogram peak at zero delay."""
    from photonics_helper.gnlse import gnlse_spectrogram

    delays, _wl, trace = gnlse_spectrogram(
        solver,
        snapshot=0,
        n_delays=41,
        delay_span_ps=5.0,
        wl_bounds=(1400.0, 1700.0),
    )
    i, _j = np.unravel_index(np.argmax(trace), trace.shape)
    assert abs(delays[i]) < 0.5


def test_spectrogram_gate_shape_validation(solver):
    """A gate of the wrong length raises a clear error."""
    from photonics_helper.gnlse import gnlse_spectrogram

    with pytest.raises(ValueError, match="gate shape"):
        gnlse_spectrogram(solver, gate=np.ones(3))


def test_plot_spectrogram_returns_figure(solver):
    """plot_spectrogram returns a Figure with delay/wavelength labels."""
    from photonics_helper.gnlse import plot_spectrogram

    fig = plot_spectrogram(solver, n_delays=21, wl_bounds=(1400, 1700))
    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Delay (ps)"
    assert ax.get_ylabel() == "Wavelength (nm)"
    plt.close(fig)


def test_plot_spectrogram_with_projections(solver):
    """The projected layout adds spectral and temporal marginal axes."""
    from photonics_helper.gnlse import plot_spectrogram

    fig = plot_spectrogram(
        solver, n_delays=21, with_projections=True, wl_bounds=(1400, 1700)
    )
    assert isinstance(fig, plt.Figure)
    # main + spectral projection + temporal projection + colorbar
    assert len(fig.axes) >= 4
    plt.close(fig)


# ---------------------------------------------------------------------------
# Four-panel summary (plot_spectral_temporal_summary)
# ---------------------------------------------------------------------------


def test_summary_plot_layout_and_order(solver):
    """Summary has line profiles on top and (taller) contours below."""
    from photonics_helper.gnlse import plot_spectral_temporal_summary

    fig = plot_spectral_temporal_summary(
        solver, wl_bounds=(1400, 1700), t_bounds=(-5, 5)
    )
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 4
    titles = [ax.get_title() for ax in fig.axes]
    assert titles == [
        "(a) Intensity (dB) vs wavelength",
        "(b) Intensity (dB) vs time",
        "(c) Spectral evolution",
        "(d) Temporal evolution",
    ]
    # Contour row must be taller than the line row.
    top_h = fig.axes[0].get_position().height
    bottom_h = fig.axes[2].get_position().height
    assert bottom_h > top_h
    plt.close(fig)


def test_summary_plot_validates_bounds(solver):
    """Inconsistent wavelength/time bounds raise a clear error."""
    from photonics_helper.gnlse import plot_spectral_temporal_summary

    with pytest.raises(ValueError, match="wl_bounds"):
        plot_spectral_temporal_summary(solver, wl_bounds=(1700, 1400))
    with pytest.raises(ValueError, match="t_bounds"):
        plot_spectral_temporal_summary(solver, t_bounds=(5, -5))


def test_summary_plotly_dashboard_traces(solver):
    """plotly=True returns line-top / heatmap-bottom traces with hover labels."""
    pytest.importorskip("plotly.graph_objects")
    from photonics_helper.gnlse import plot_spectral_temporal_summary

    fig = plot_spectral_temporal_summary(
        solver,
        wl_bounds=(1480, 1620),
        t_bounds=(-5, 5),
        plotly=True,
    )
    assert fig.__class__.__module__.startswith("plotly")
    types = [trace.type for trace in fig.data]
    assert types[:4] == ["scatter", "scatter", "heatmap", "heatmap"]
    assert "Feature: %{customdata}" in fig.data[2].hovertemplate
    labels = set(np.asarray(fig.data[3].customdata).ravel().tolist())
    assert labels  # non-empty feature labels


def test_save_summary_html(tmp_path, solver):
    """save_summary_html writes a standalone interactive HTML document."""
    pytest.importorskip("plotly")
    from photonics_helper.gnlse import save_summary_html

    out = save_summary_html(
        solver,
        tmp_path / "nested" / "summary.html",
        wl_bounds=(1480, 1620),
        t_bounds=(-5, 5),
    )
    assert out.exists()
    assert out.stat().st_size > 0
    text = out.read_text(encoding="utf-8")
    assert "plotly" in text.lower()
    assert "Feature:" in text or "customdata" in text
