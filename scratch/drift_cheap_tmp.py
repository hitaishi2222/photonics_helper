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


def raman(grid):
    spec = RamanSpec(name="Silica", raman_shift_cm=440.0, raman_linewidth_cm=45.0, fR=0.18)
    return RamanResponse(spec=spec, fR=0.18, tau1=12.2e-15, tau2=32e-15, grid=grid)


def run(N, Tmax, n_steps, shock, raman_on, scheme="strang"):
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
        eng = SplitStepEngine(
            pulse=pulse, fiber=fiber, betas=np.array([BETA2 * 1e24]),
            include_raman=raman_on, include_self_steepening=shock,
            step_size=Length(LENGTH / n_steps, "m"), scheme=scheme,
        )
        eng.propagate(num_steps=n_steps, nsaves=5)
    e = np.asarray(eng.energy_vs_z, float)
    p = np.asarray(eng.photon_number_vs_z, float)
    tau_om = (1 / OMEGA0) * float(grid.omega_max)
    print(f"N={N} T={Tmax*1e12:.0f}ps tauOm={tau_om:.2f}: "
          f"dE={((e[-1]-e[0])/e[0])*100:+.3f}% dQ={((p[-1]-p[0])/p[0])*100:+.5f}%")


import time
for (N, T, ns) in [(2048, 24e-12, 400), (4096, 24e-12, 400), (4096, 48e-12, 800)]:
    t0 = time.time()
    run(N, T, ns, True, True)
    print(f"   ({time.time()-t0:.1f}s)")
run(4096, 24e-12, 400, True, False)
run(4096, 24e-12, 400, False, True)
