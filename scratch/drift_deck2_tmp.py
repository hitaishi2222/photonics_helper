"""Confirm B&W photon-number invariant on the Hult/Dudley PCF deck + yoshida4."""
import warnings

import numpy as np

from photonics_helper.base import Length, Time, Wavelength
from photonics_helper.gnlse import FiberProfile, SplitStepEngine
from photonics_helper.pulse import Envelope, TemporalGrid, Wave
from photonics_helper.raman import RamanResponse, RamanSpec

WL0_NM = 835.0
OMEGA0 = 2 * np.pi * 3e8 / (WL0_NM * 1e-9)


def _wave(n=8192, tmax_s=14e-12):
    grid = TemporalGrid(N=n, Tmax=Time(tmax_s, "s"))
    env = Envelope.from_fwhm("sech", peak_amplitude=np.sqrt(10e3), fwhm=Time(50e-15, "s"))
    return Wave(grid=grid, envelope=env, central_wavelength=Wavelength(WL0_NM, "nm"))


def _fiber(wave, raman):
    grid = wave.grid
    if raman:
        spec = RamanSpec(name="Silica", raman_shift_cm=440.0, raman_linewidth_cm=45.0, fR=0.18)
        resp = RamanResponse(spec=spec, fR=0.18, tau1=12.2e-15, tau2=32e-15, grid=grid)
    else:
        resp = None
    return FiberProfile.from_gamma(
        gamma=0.11, n2=2.6e-20, omega0=OMEGA0, length=Length(0.15, "m"),
        raman_response=resp,
    )


BETAS = np.array([-0.0755, -1.5, 3.0])


def run(scheme, n_steps=1500):
    wave = _wave()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eng = SplitStepEngine(
            pulse=wave, fiber=_fiber(wave, True), betas=BETAS,
            include_raman=True, include_self_steepening=True,
            tau_shock=0.56e-15,
            step_size=Length(0.15 / n_steps, "m"),
            scheme=scheme,
        )
        A0 = np.array(wave.envelope_field, dtype=complex)
        eng.propagate(num_steps=n_steps)
    A1 = np.array(eng.A, dtype=complex)
    g = eng.grid
    Aw = g.fft(A1)
    dw = float(g.dw)
    wabs = OMEGA0 + g.w
    u0 = np.sum(np.abs(g.fft(A0)) ** 2) * dw
    u1 = np.sum(np.abs(Aw) ** 2) * dw
    p0 = np.sum(np.abs(g.fft(A0)) ** 2 / wabs) * dw
    p1 = np.sum(np.abs(Aw) ** 2 / wabs) * dw
    print(f"scheme={scheme:9s} dE={((u1-u0)/u0)*100:+.4f}%  dQ_photon={((p1-p0)/p0)*100:+.6f}%")


for scheme in ("strang", "yoshida4"):
    run(scheme)
