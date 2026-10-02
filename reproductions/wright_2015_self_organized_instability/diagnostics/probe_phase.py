"""Verify corrected phase matching: coherent pair gain at new/old roots."""

import importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location(
    "rep", "reproductions/planned/wright_2015_self_organized_instability/reproduce.py"
)
rep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rep)
from photonics_helper.base import Area, Length, Time, Wavelength
from photonics_helper.gnlse import FiberProfile
from photonics_helper.multimode_gnlse import MultimodeSplitStepEngine
from photonics_helper.pulse import TemporalGrid, Wave, Envelope

N, WIN = 16384, 25.0
ap = np.sqrt(rep.P0_W)
b2 = rep.beta2_silica(rep.LAMBDA0_NM)
GAMMA = rep.gamma_per_w = 2 * np.pi * rep.N2 / rep.LAMBDA0 / rep.A_EFF
gf_full = GAMMA * rep.P0_W  # 4.70 /m (pump diag, full gamma)
gf = rep.gamma_eff() * rep.P0_W  # 3.13 /m=(2/3)gamma P0 (coupling & XPM)
grid = TemporalGrid(N=N, Tmax=Time(WIN * 1e-12, "s"))
tt = grid.t
tt = tt.as_s if hasattr(tt, "as_s") else tt
w_thz = np.abs(grid.w) / (2 * np.pi * 1e12)
N_ord = 2


def delta_eff(f_thz):
    sym = (
        rep.beta0(rep.PUMP_THZ + f_thz)
        + rep.beta0(rep.PUMP_THZ - f_thz)
        - 2 * rep.beta0(rep.PUMP_THZ)
    )
    return 0.5 * sym - N_ord * rep.KAPPA - GAMMA * rep.P0_W / 3.0


def root_new():
    xs = np.linspace(1.0, 270.0, 12000)
    vals = np.array([delta_eff(x) for x in xs])
    i = np.where(np.diff(np.sign(vals)) != 0)[0]
    j = i[0]
    lo, hi = xs[j], xs[j + 1]
    # bisect
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if delta_eff(lo) * delta_eff(mid) < 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


fn, fo = root_new(), rep.stmi_shift_thz(2)
print(
    f"old root {fo if False else fo:.3f} vs new root {fn:.3f} THz; "
    f"delta_eff(old)={delta_eff(fo):.2f}, delta_eff(new)={delta_eff(fn):.2e}"
)
L, DZ = 0.3, 4e-5
STEPS = int(round(L / DZ))


def probe(f_target, seed="growing", offs_sign=-1):
    k = int(np.argmin(np.abs(w_thz - f_target)))
    om = grid.w[k]
    k_neg = int(np.argmin(np.abs(grid.w + grid.w[k])))
    offs = [0.0, offs_sign * N_ord * rep.KAPPA, offs_sign * N_ord * rep.KAPPA]
    waves = []
    a0 = 1e-5 * ap
    phi0 = 0.0
    for ch in range(3):
        env = Envelope(
            shape="gaussian", peak_amplitude=ap, pulse_width=Time(WIN * 1e-12 / 4, "s")
        )
        W = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(532.0, "nm"))
        if ch == 0:
            field = np.full(N, ap, dtype=complex)
        elif ch == 1:
            field = a0 * np.exp(1j * (om * tt + phi0))
        else:
            # relative phase scan: 0, pi/2, pi, 3pi/2 handled via seed variants
            field = (
                a0 * np.exp(1j * (-om * tt)) if False else a0 * np.exp(-1j * om * tt)
            )
        waves.append(W.with_field(np.asarray(field, complex)))
    fiber = FiberProfile(
        n2=rep.N2, alpha=0.0, A_eff=Area(rep.A_EFF, "m^2"), length=Length(L, "m")
    )
    eng = MultimodeSplitStepEngine(
        waves,
        fiber,
        betas=[[b2]] * 3,
        betas_unit="s^k/m",
        phase_offsets=offs,
        oam_l=[0, 0, 0],
        coef_model="lp_degenerate",
        include_fwm=True,
        fwm_pump_depletion=True,
        step_size=Length(DZ, "m"),
    )
    s10 = abs(grid.fft(np.asarray(eng.A[1], complex)))[k] ** 2
    s20 = abs(grid.fft(np.asarray(eng.A[2], complex)))[k_neg] ** 2
    eng.propagate(STEPS, nsaves=2)
    s1 = abs(grid.fft(np.asarray(eng.A[1], complex)))[k] ** 2
    s2 = abs(grid.fft(np.asarray(eng.A[2], complex)))[k_neg] ** 2
    return s1 / s10, s2 / s20


for rel in [0.0, np.pi / 2, np.pi, -np.pi / 2]:
    r1, r2 = probe(fn)
    break
# scanning relative phase via phi0 patch: simpler - loop phi
for phi in [0.0, np.pi / 2, np.pi, 3 * np.pi / 2, np.pi / 4, -np.pi / 4]:
    k = int(np.argmin(np.abs(w_thz - fn)))
    om = grid.w[k]
    k_neg = int(np.argmin(np.abs(grid.w + grid.w[k])))
    waves = []
    a0 = 1e-5 * ap
    for ch in range(3):
        env = Envelope(
            shape="gaussian", peak_amplitude=ap, pulse_width=Time(WIN * 1e-12 / 4, "s")
        )
        W = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(532.0, "nm"))
        if ch == 0:
            field = np.full(N, ap, dtype=complex)
        elif ch == 1:
            field = a0 * np.exp(1j * om * tt)
        else:
            field = a0 * np.exp(1j * (-om * tt + phi))
        waves.append(W.with_field(np.asarray(field, complex)))
    fiber = FiberProfile(
        n2=rep.N2, alpha=0.0, A_eff=Area(rep.A_EFF, "m^2"), length=Length(L, "m")
    )
    eng = MultimodeSplitStepEngine(
        waves,
        fiber,
        betas=[[b2]] * 3,
        betas_unit="s^k/m",
        phase_offsets=[0.0, -N_ord * rep.KAPPA, -N_ord * rep.KAPPA],
        oam_l=[0, 0, 0],
        coef_model="lp_degenerate",
        include_fwm=True,
        fwm_pump_depletion=True,
        step_size=Length(DZ, "m"),
    )
    s10 = abs(grid.fft(np.asarray(eng.A[1], complex)))[k] ** 2
    s20 = abs(grid.fft(np.asarray(eng.A[2], complex)))[k_neg] ** 2
    eng.propagate(STEPS, nsaves=2)
    s1 = abs(grid.fft(np.asarray(eng.A[1], complex)))[k] ** 2
    s2 = abs(grid.fft(np.asarray(eng.A[2], complex)))[k_neg] ** 2
    print(
        f"new-root rel phase {phi:+.3f}: P1 out/in = {s1 / s10:8.4f}, "
        f"P2 out/in = {s2 / s20:8.4f}"
    )
