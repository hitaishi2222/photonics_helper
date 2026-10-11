"""Drift diagnosis, round 2: pin down the mechanism.

Config: N=3 soliton, 1550 nm, beta2=-21 ps^2/km, gamma=2, T0=50/1.763 fs, L~12cm.
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


def build(N, Tmax, raman_on):
    grid = TemporalGrid(N=N, Tmax=Time(Tmax, "s"))
    env = Envelope(shape="sech", peak_amplitude=np.sqrt(PEAK), pulse_width=Time(T0, "s"))
    pulse = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(WL0, "m"))
    resp = raman(grid) if raman_on else None
    fiber = FiberProfile.from_gamma(
        gamma=GAMMA, n2=2.6e-20, omega0=OMEGA0, length=Length(LENGTH, "m"),
        raman_response=resp,
    )
    return pulse, fiber, grid


def run(N, Tmax, n_steps, shock, raman_on, conserving=False, beta2=True):
    pulse, fiber, grid = build(N, Tmax, raman_on)
    betas = BETAS if beta2 else np.zeros(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine = SplitStepEngine(
            pulse=pulse, fiber=fiber, betas=betas,
            include_raman=raman_on,
            include_self_steepening=shock,
            conserving_shock=conserving,
            step_size=Length(LENGTH / n_steps, "m"),
        )
        engine.propagate(num_steps=n_steps)
    e = np.asarray(engine.energy_vs_z, dtype=float)
    return (e[-1] - e[0]) / e[0] * 100, engine, grid


print("=== A. 2x2 triad (N=16384, Tmax=48ps, 800 steps) ===")
for shock in (False, True):
    for raman_on in (False, True):
        d, _, _ = run(16384, 48e-12, 800, shock, raman_on)
        print(f"shock={shock!s:5s} raman={raman_on!s:5s}  drift={d:+.4f}%")

print("\n=== B. Tmax ladder at N=16384 (vary tau*Omega_max) ===")
for tmax_ps in [48, 96, 144, 192, 211, 264, 528]:
    d, _, g = run(16384, tmax_ps * 1e-12, 800, True, True)
    tau_om = (1 / OMEGA0) * float(g.omega_max)
    print(f"Tmax={tmax_ps:4d}ps dt={g.dt:.2e} tau*Om={tau_om:.3f}  drift={d:+.4f}%")

print("\n=== C. conserving_shock=True (gamma>0, expected no-op) ===")
for shock in (False, True):
    d, _, _ = run(16384, 48e-12, 800, shock, True, conserving=True)
    print(f"conserving, shock={shock!s:5s}  drift={d:+.4f}%")

print("\n=== D. spectral split of energy (N=16384, Tmax=48ps, raman+shock) ===")
pulse, fiber, grid = build(16384, 48e-12, True)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    engine = SplitStepEngine(
        pulse=pulse, fiber=fiber, betas=BETAS,
        include_raman=True, include_self_steepening=True,
        step_size=Length(LENGTH / 800, "m"),
    )
    A0 = np.array(pulse.envelope_field, dtype=complex)
    engine.propagate(num_steps=800)
A1 = np.array(engine.A, dtype=complex)
w = grid.w
dt = grid.dt


def split(A):
    Aw = grid.fft(A)
    e_tot = np.sum(np.abs(A) ** 2) * dt
    e_blue = np.sum(np.abs(Aw[w > 0]) ** 2) * grid.dw
    e_red = np.sum(np.abs(Aw[w < 0]) ** 2) * grid.dw
    return e_tot, e_blue, e_red


e0 = split(A0)
e1 = split(A1)
print("total change: {:+.3f}%   blue change: {:+.3f}%   red change: {:+.3f}%".format(
    (e1[0] - e0[0]) / e0[0] * 100, (e1[1] - e0[1]) / e0[1] * 100, (e1[2] - e0[2]) / e0[2] * 100))

print("\n=== E. model sink-rate check at the final state ===")
# dE/dz of the shock operator alone = -2*gamma*tau*int P_NL Im(A* dA/dt) dt
A = A1
gamma = GAMMA
tau = 1.0 / OMEGA0
dt_s = grid.dt
dAdt = grid.ifft((-1j * w) * grid.fft(A))  # d/dt symbol -i*Omega (codebase convention)
P_NL = engine._nl_intensity(np.abs(A) ** 2, engine._get_h_R_fft())
sink_per_m = -2.0 * gamma * tau * float(np.sum(P_NL * np.imag(np.conj(A) * dAdt)) * dt_s)
print(f"predicted dE/dz at final state = {sink_per_m:+.6e} /m")
dz = LENGTH / 800
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    A_before = np.array(A, dtype=complex)
    A_after, _ = engine._nonlinear_step(A_before, dz)
dE_obs = float(np.sum(np.abs(A_after) ** 2 - np.abs(A_before) ** 2) * dt_s)
print(f"observed   dE over one dz  = {dE_obs:+.6e}  (dz={dz:.2e} m)")
print(f"ratio obs/pred = {dE_obs / (sink_per_m * dz):.4f}")
