"""Auto grid sizing (for_gnlse) and converge-and-retry (propagate_auto)."""

import warnings

import numpy as np
import pytest

from photonics_helper import (
    Envelope,
    Fiber,
    GNLSESolver,
    Length,
    Time,
    Wave,
    Wavelength,
)
from photonics_helper.core.grids import TemporalGrid


def test_for_gnlse_spm_only_gives_compact_window():
    grid = TemporalGrid.for_gnlse(Time(50, "fs"), 0.1, 0.0)
    assert grid.Tmax.as_s == pytest.approx(10 * Time(50, "fs").as_s)


def test_for_gnlse_fission_case_is_wide():
    pulse = Envelope.from_fwhm("sech", np.sqrt(10e3), Time(50, "fs"))
    grid = TemporalGrid.for_gnlse(
        pulse.pulse_width, 0.1, -1.276e-2, gamma=0.045, peak_power=10e3
    )
    assert grid.Tmax.as_ps > 1.0  # must hold the fission walk-off


def test_propagate_auto_converges_small_case():
    warnings.filterwarnings("ignore")
    # N=2 soliton: fast (narrow grid, short fiber), exercises drift doubling.
    beta2 = -2.0e-2  # ps^2/m
    gamma = 0.02
    T0 = Time(100, "fs").as_s
    beta2_si = abs(beta2) * 1e-24
    P0 = beta2_si / (gamma * T0**2)  # N=1 power
    fiber = Fiber.from_gamma(gamma, length=Length(0.05, "m"))
    pulse = Envelope(
        shape="sech", peak_amplitude=np.sqrt(4 * P0), pulse_width=Time(100, "fs")
    )
    grid = TemporalGrid.for_gnlse(
        pulse.pulse_width, fiber.length.as_m, beta2, gamma=gamma, peak_power=4 * P0
    )
    wave = Wave.from_envelope(pulse, Wavelength(1550, "nm"), grid=grid)
    gs = GNLSESolver(
        wave, fiber, np.array([beta2]), scheme="strang", include_raman=False
    )
    report = gs.propagate_auto()
    assert report["converged"]
    assert report["drift"] <= 0.01
    assert np.all(np.isfinite(np.abs(gs.evolution[-1].envelope_field)))


def test_propagate_auto_widens_wrapped_grid():
    warnings.filterwarnings("ignore")
    # Narrow window + third-order dispersion: the shed dispersive wave runs
    # to the window edge, so the edge detector must fire and widen the grid.
    import math

    betas = np.array([-2.0e-2, 1.0e-3])
    gamma = 0.045
    fiber = Fiber.from_gamma(gamma, length=Length(0.1, "m"))
    T0 = Time(50, "fs").as_s / (2 * math.acosh(math.sqrt(2)))
    P0 = abs(betas[0]) * 1e-24 / (gamma * T0**2)
    pulse = Envelope.from_fwhm("sech", np.sqrt(9 * P0), Time(50, "fs"))
    narrow = TemporalGrid(N=2**11, Tmax=Time(10 * T0, "s"))
    wave = Wave.from_envelope(pulse, Wavelength(1550, "nm"), grid=narrow)
    gs = GNLSESolver(wave, fiber, betas, scheme="strang", include_raman=False)
    report = gs.propagate_auto(max_rounds=4)
    assert report["widened"]
    assert gs.pulse.grid.Tmax.as_s > narrow.Tmax.as_s


def test_propagate_auto_uses_photon_drift_with_shock():
    warnings.filterwarnings("ignore")
    # Raman red-shift legitimately changes the unweighted energy (several %)
    # while the B&W (A5) photon number is conserved: the auto driver must
    # converge on the invariant in round 1, not double steps chasing physics.
    from photonics_helper.raman import RamanSpec

    beta2 = -2.0e-2
    gamma = 0.045
    grid = TemporalGrid(N=2**12, Tmax=Time(30 * Time(50, "fs").as_s, "s"))
    spec = RamanSpec(
        name="Silica", raman_shift_cm=440.0, raman_linewidth_cm=45.0, fR=0.18
    )
    fiber = Fiber.from_gamma(
        gamma,
        alpha=0.0,
        length=Length(0.1, "m"),
        raman_response=spec.response(grid),
    )
    T0 = Time(50, "fs").as_s
    P0 = abs(beta2) * 1e-24 / (gamma * T0**2)
    pulse = Envelope(
        shape="sech", peak_amplitude=np.sqrt(9 * P0), pulse_width=Time(50, "fs")
    )
    wave = Wave.from_envelope(pulse, Wavelength(1550, "nm"), grid=grid)
    gs = GNLSESolver(
        wave,
        fiber,
        np.array([beta2]),
        scheme="strang",
        include_raman=True,
        include_self_steepening=True,
    )
    report = gs.propagate_auto(max_rounds=2)
    assert report["drift_metric"] == "photon"
    assert report["converged"]
    assert report["rounds"] == 1
    assert report["photon_drift"] <= 0.01
    assert report["drift"] > 0.01  # energy moved: old code doubled here
