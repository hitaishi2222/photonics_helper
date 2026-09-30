"""Diagnostics: kibler AB evolution probe — where does the compression go?"""
import importlib.util as ilu
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = ilu.spec_from_file_location("rep", HERE.parent / "reproduce.py")
rep = ilu.module_from_spec(spec); spec.loader.exec_module(rep)

from photonics_helper.base import Length
from photonics_helper.breathers import general_sfb
from photonics_helper.gnlse import GNLSESolver
from photonics_helper.pulse import Envelope, TemporalGrid, Wave, Wavelength

b2_si = -8.85e-27
gamma, P0, a, xi_d = 0.01, 0.30, 0.42, 2.5
L_NL = 1/(gamma*P0); T0 = np.sqrt(abs(b2_si)*L_NL)
from photonics_helper.base import Time
grid = TemporalGrid(N=8192, Tmax=Time(100e-12, 's'))
psi0 = np.asarray(general_sfb(-xi_d, grid.t/T0, a)) * np.sqrt(P0)

def env_of(field):
    def f(t, _T0, _A0):
        return np.interp(t, grid.t, field)
    return Envelope(shape="custom", peak_amplitude=float(np.max(np.abs(field))),
                    pulse_width=Time(T0, 's'), func=f)

wave = Wave(grid=grid, envelope=env_of(psi0), central_wavelength=Wavelength(1550e-9, "m"))
fiber = rep.FiberProfile.from_gamma if hasattr(rep, 'FiberProfile') else None
from photonics_helper.gnlse import FiberProfile
fiber = FiberProfile.from_gamma(gamma=gamma, n2=2.7e-20,
                                omega0=float(wave.central_frequency),
                                length=Length(2*xi_d*L_NL, "m"))
solver = GNLSESolver(pulse=wave, fiber=fiber, betas=np.array([b2_si*1e24]),
                     include_raman=False, include_self_steepening=False,
                     step_size=Length(0.5, "m"))
n = int(2*xi_d*L_NL/0.5)
solver.propagate(num_steps=n, nsaves=min(n, 80))
I_all = np.array([np.abs(w.envelope_field)**2 for w in solver.evolution])
z = np.asarray(solver._z_positions)[:len(solver.evolution)]
print("z range:", z[0], z[-1], "nz:", len(z))
peaks = I_all.max(axis=1)/P0
for k in range(0, len(z), max(1, len(z)//20)):
    xi = z[k]/L_NL - xi_d
    Ia = np.abs(general_sfb(xi, grid.t/T0, a))**2
    print(f"z={z[k]:8.1f} m xi={xi:6.2f} sim_peak={peaks[k]:8.2f} ana_peak={Ia.max():8.2f}")
print("max sim peak:", peaks.max(), " ana at xi=0:", float((np.abs(general_sfb(0.0, grid.t/T0, a))**2).max()))
