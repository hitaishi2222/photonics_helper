"""Tests for photonics_helper.gnlse.visualize."""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from photonics_helper.gnlse import (
    FiberProfile,
    GNLSESolver,
    plot_intensity_metrics,
    plot_spectrum_vs_distance,
    plot_waterfall,
)
from photonics_helper.pulse import Wave, Envelope, TemporalGrid
from photonics_helper.base import Wavelength, Time, Length, Area


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
    fiber = FiberProfile(n2=1e-19, alpha=1e-5, A_eff=Area(5e-11, "m^2"), length=Length(1e-3, "m"))
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
    assert ax.get_ylabel() == "Propagation distance (mm)"
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
