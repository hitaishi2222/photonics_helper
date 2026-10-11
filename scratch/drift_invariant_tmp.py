"""Is the -14% drift just the wrong invariant?

Blow & Wood 1989 Appendix, Eq. (A5): the conserved quantity of the GNLS is
  Q = integral |Uhat(w)|^2 / (w0 + w) dw        [photon number]
NOT the unweighted Parseval norm integral |Uhat|^2 dw (= energy). With
tau_shock = 1/w0 (the code's default), (A5) is exactly the classical photon
number. If Q is conserved while the unweighted norm drifts -14%, the drift is
the difference between energy and photon number, not a defect.
"""
import warnings

import numpy as np

from photonics_helper.base import C_MS, Length, Time, Wavelength
from photonics_helper.gnlse import FiberProfile, SplitStepEngine
from photonics_helper.pulse import Envelope, TemporalGrid, Wave
from photonics_helper.raman import RamanResponse, RamanSpec

WL0 = 1550e-9
OMEGA0 = 2 * np.pi * C_MS / WL0
BETA2 = -21e-27
GAMMA = 2.0
T0 = 50e-15 / 1.763
PEAK = 9.0 * abs(BETA2) / (GAMMA * T0**2)
LENGTH = 2.0 * (np.pi / 2) * T0**2 / abs(BETA2)
BETAS = np.array([BETA2 * 1e24])


def raman(grid):
    spec = RamanSpec(name="Silica", raman_shift_cm=440.0, raman_linewidth_cm=45.0, fR=0.18)
    return RamanResponse(spec=spec, fR=0.18, tau1=12.2e-15, tau2=32e-15, grid=grid)


def invariants(A, grid, shock_only=False):
    Aw = grid.fft(A)
    dw = float(grid.dw)
    w = grid.w
    omega_abs = OMEGA0 + w
    unweighted = float(np.sum(np.abs(Aw) ** 2) * dw)
    photon = float(np.sum(np.abs(Aw) ** 2 / omega_abs) * dw)
    return unweighted, photon


def run(N, Tmax, n_steps, shock, raman_on):
    grid = TemporalGrid(N=N, Tmax=Time(Tmax, "s"))
    env = Envelope(shape="sech", peak_amplitude=np.sqrt(PEAK), pulse_width=Time(T0, "s"))
    pulse = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(WL0, "m"))
    resp = raman(grid) if raman_on else None
    fiber = FiberProfile.from_gamma(
        gamma=GAMMA, n2=2.6e-20, omega0=OMEGA0, length=Length(LENGTH, "m"),
        raman_response=resp,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine = SplitStepEngine(
            pulse=pulse, fiber=fiber, betas=BETAS,
            include_raman=raman_on, include_self_steepening=shock,
            step_size=Length(LENGTH / n_steps, "m"),
        )
        A0 = np.array(pulse.envelope_field, dtype=complex)
        engine.propagate(num_steps=n_steps)
    A1 = np.array(engine.A, dtype=complex)
    u0, p0 = invariants(A0, grid)
    u1, p1 = invariants(A1, grid)
    return (u1 - u0) / u0 * 100, (p1 - p0) / p0 * 100


print(f"{'config':44s} {'dE_unw/E':>10s} {'dQ_phot/Q':>10s}")
for label, (shock, raman_on) in {
    "raman only (no shock)": (False, True),
    "shock only (no raman)": (True, False),
    "raman + shock, Tmax=48ps (tauOm=0.88)": (True, True),
    "raman + shock, Tmax=96ps (tauOm=0.44)": (True, True),
    "raman + shock, Tmax=144ps (tauOm=0.29)": (True, True),
    "raman + shock, Tmax=264ps (tauOm=0.16)": (True, True),
    "raman + shock, Tmax=528ps (tauOm=0.08)": (True, True),
}.items():
    if "48ps" in label:
        N, T = 16384, 48e-12
    elif "96ps" in label:
        N, T = 16384, 96e-12
    elif "144ps" in label:
        N, T = 16384, 144e-12
    elif "264ps" in label:
        N, T = 16384, 264e-12
    elif "528ps" in label:
        N, T = 16384, 528e-12
    else:
        N, T = 16384, 48e-12
    du, dp = run(N, T, 800, shock, raman_on)
    print(f"{label:44s} {du:+9.4f}% {dp:+9.4f}%")
