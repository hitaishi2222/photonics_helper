"""Measure B&W photon-number vs energy drift on the ISSUES #1 fissioning deck
(Hult/Dudley Table-I PCF params, 850 nm, 500 fs, N_sol~3, L=100 m)."""
import warnings

import numpy as np

from photonics_helper.base import Length, Time, Wavelength
from photonics_helper.gnlse import FiberProfile, SplitStepEngine
from photonics_helper.pulse import Envelope, TemporalGrid, Wave
from photonics_helper.raman import RamanResponse, RamanSpec

B2 = -0.01276e-24
GAMMA = 0.045
T0 = 500e-15
P0 = 40.0
L = 100.0
OMEGA0 = 2 * np.pi * 3e8 / 850e-9


def raman(grid):
    spec = RamanSpec(name="Silica", raman_shift_cm=440.0, raman_linewidth_cm=45.0, fR=0.18)
    return RamanResponse(spec=spec, fR=0.18, tau1=12.2e-15, tau2=32e-15, grid=grid)


def run(n_pts):
    grid = TemporalGrid(N=n_pts, Tmax=Time(20e-12, "s"))
    env = Envelope(shape="sech", peak_amplitude=np.sqrt(P0), pulse_width=Time(T0, "s"))
    wave = Wave(envelope=env, central_wavelength=Wavelength(850.0, "nm"), grid=grid)
    fiber = FiberProfile.from_gamma(
        gamma=GAMMA, n2=2.6e-20, omega0=OMEGA0, alpha=0.0,
        length=Length(L, "m"), raman_response=raman(grid),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eng = SplitStepEngine(
            pulse=wave, fiber=fiber, betas=np.array([B2]),
            include_raman=True, include_self_steepening=True,
        )
        eng.propagate(num_steps=4000, nsaves=3)
    e = np.asarray(eng.energy_vs_z, float)
    p = np.asarray(eng.photon_number_vs_z, float)
    tau_om = (1 / OMEGA0) * float(grid.omega_max)
    print(f"N={n_pts:5d} tauOm={tau_om:.3f}: dE={100*(e[-1]/e[0]-1):+.3f}%  "
          f"dQ_photon={100*(p[-1]/p[0]-1):+.5f}%", flush=True)


for n in (1024, 4096):
    run(n)
