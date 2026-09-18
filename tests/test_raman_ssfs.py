"""Regression tests for the Raman sign convention and soliton self-frequency shift.

These guard the two bugs fixed together: the engine's dispersion-sign convention
(bright soliton for anomalous β₂) and the Raman response direction (Stokes gain,
red-shifting solitons).
"""

import numpy as np

from photonics_helper.base import C_MS, Length, Time, Wavelength
from photonics_helper.gnlse import FiberProfile, GNLSESolver
from photonics_helper.pulse import Envelope, TemporalGrid, Wave
from photonics_helper.raman import RamanResponse, RamanSpec
from reproductions.gordon_1986_ssfs.reproduce import validate as validate_gordon


def _raman(grid):
    spec = RamanSpec(
        name="Silica", raman_shift_cm=440.0, raman_linewidth_cm=45.0, fR=0.18
    )
    return RamanResponse(spec=spec, fR=0.18, tau1=12.2e-15, tau2=32e-15, grid=grid)


def test_raman_amplifies_stokes_not_anti_stokes():
    """Physical Raman gain: the Stokes (red) sideband grows, the anti-Stokes decays."""
    wl0 = 1.55e-6
    omega_r = 2 * np.pi * 13.2e12
    grid = TemporalGrid(N=2**13, Tmax=Time(20e-12, "s"))
    t, w = grid.t, grid.w
    i_p = int(np.argmin(np.abs(w - omega_r)))
    i_m = int(np.argmin(np.abs(w + omega_r)))

    env = Envelope(
        shape="gaussian", peak_amplitude=np.sqrt(5.0), pulse_width=Time(1.0, "s")
    )
    pulse = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(wl0, "m"))
    field = np.sqrt(5.0) * (1 + 1e-3 * np.cos(omega_r * t))
    pulse = pulse.with_field(field)
    fiber = FiberProfile.from_gamma(
        gamma=2.0e-3,
        n2=2.6e-20,
        omega0=pulse.central_frequency,
        length=Length(2.0, "m"),
        raman_response=_raman(grid),
    )
    solver = GNLSESolver(
        pulse=pulse, fiber=fiber, betas=np.array([0.0]), include_raman=True
    )
    solver.propagate(num_steps=500)

    W0 = np.abs(grid.fft(field)) ** 2
    W1 = np.abs(grid.fft(solver.evolution[-1].envelope_field)) ** 2
    assert W1[i_m] / W0[i_m] > 1.0, "Stokes sideband must gain"
    assert W1[i_p] / W0[i_p] < 1.0, "anti-Stokes sideband must lose"


def test_fundamental_soliton_is_stable_and_red_shifts():
    wl0 = 1.5e-6
    D = 15e-6
    beta2 = -D * wl0**2 / (2 * np.pi * C_MS)
    T0 = 250e-15 / 1.7627
    gamma = 1.3e-3
    P0 = abs(beta2) / (gamma * T0**2)
    grid = TemporalGrid(N=2**12, Tmax=Time(20e-12, "s"))
    env = Envelope(shape="sech", peak_amplitude=np.sqrt(P0), pulse_width=Time(T0, "s"))
    pulse = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(wl0, "m"))
    fiber = FiberProfile.from_gamma(
        gamma=gamma,
        n2=2.6e-20,
        omega0=pulse.central_frequency,
        length=Length(20.0, "m"),
        raman_response=_raman(grid),
    )
    solver = GNLSESolver(
        pulse=pulse, fiber=fiber, betas=np.array([beta2 * 1e24]), include_raman=True
    )
    solver.propagate(num_steps=4000)
    A = solver.evolution[-1].envelope_field
    intensity = np.abs(A) ** 2
    idx = np.where(intensity / intensity.max() >= 0.5)[0]
    fwhm = (grid.t[idx[-1]] - grid.t[idx[0]]) * 1e15
    assert fwhm < 2.0 * 234.375, "fundamental soliton must stay intact (no dispersion)"
    W = np.abs(grid.fft(A)) ** 2
    wl = 2 * np.pi * C_MS / (2 * np.pi * C_MS / wl0 + grid.w)
    assert wl[int(np.argmax(W))] > wl0, "Raman soliton must red-shift"


def test_gordon_ssfs_reproduction():
    result = validate_gordon(make_plot=False)
    assert result["measured_nm"] > 0.0
    assert 0.5 <= result["ratio"] <= 2.0
