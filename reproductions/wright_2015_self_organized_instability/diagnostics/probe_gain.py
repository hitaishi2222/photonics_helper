"""Probe the engine's parametric gain with a coherent conjugate tone pair."""
import importlib.util
import numpy as np
import time

spec = importlib.util.spec_from_file_location(
    "rep", "reproductions/planned/wright_2015_self_organized_instability/reproduce.py")
rep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rep)
from photonics_helper.base import Area, Length, Time, Wavelength
from photonics_helper.gnlse import FiberProfile
from photonics_helper.multimode_gnlse import MultimodeSplitStepEngine
from photonics_helper.pulse import TemporalGrid, Wave, Envelope

N, WIN = 16384, 25.0
ap = np.sqrt(rep.P0_W)
b2 = rep.beta2_silica(rep.LAMBDA0_NM)
GAMMA_F = rep.gamma_eff()
gf = GAMMA_F * rep.P0_W
L, DZ = 0.3, 4e-5
STEPS = int(round(L / DZ))

grid = TemporalGrid(N=N, Tmax=Time(WIN * 1e-12, "s"))
tt = grid.t
tt = tt.as_s if hasattr(tt, "as_s") else tt
w_thz = np.abs(grid.w) / (2 * np.pi * 1e12)

root2 = rep.stmi_shift_thz(2)
print(f"gf={gf:.3f} /m, root2={root2:.3f} THz, L={L} m, expected amp growth"
      f" cosh(gf*L)={np.cosh(gf * L):.3f}")


def probe(f_target, mode_offsets="minus", ch2="conj"):
    k = int(np.argmin(np.abs(w_thz - f_target)))
    Om = 2 * np.pi * w_thz[k] * 1e12
    k_m = int(np.argmin(np.abs(w_thz + (w_thz[k])))) if False else None
    # negative-frequency index of -f
    wn = grid.w  # signed
    k_neg = int(np.argmin(np.abs(wn + grid.w[k])))
    offs = {"minus": [0, -2 * rep.KAPPA, -2 * rep.KAPPA],
            "plus": [0, +2 * rep.KAPPA, +2 * rep.KAPPA],
            "none": None}[mode_offsets]
    waves = []
    a0 = 1e-4 * ap
    for ch in range(3):
        env = Envelope(shape="gaussian", peak_amplitude=ap,
                       pulse_width=Time(WIN * 1e-12 / 4, "s"))
        W = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(532.0, "nm"))
        if ch == 0:
            field = np.full(N, ap, dtype=complex)
        elif ch == 1:
            field = a0 * np.exp(1j * Om * tt)
        else:
            field = (a0 * np.exp(-1j * Om * tt) if ch2 == "conj"
                     else np.zeros(N, complex))
        waves.append(W.with_field(np.asarray(field, complex)))
    fiber = FiberProfile(n2=rep.N2, alpha=0.0, A_eff=Area(rep.A_EFF, "m^2"),
                         length=Length(L, "m"))
    eng = MultimodeSplitStepEngine(
        waves, fiber, betas=[[b2]] * 3, betas_unit="s^k/m",
        phase_offsets=offs, oam_l=[0, 0, 0], coef_model="lp_degenerate",
        include_fwm=True, fwm_pump_depletion=True, step_size=Length(DZ, "m"))
    s0 = np.abs(grid.fft(np.asarray(eng.A[1], complex)))[k] ** 2
    s0n = np.abs(grid.fft(np.asarray(eng.A[2], complex)))[k_neg] ** 2
    eng.propagate(STEPS, nsaves=2)
    s1 = np.abs(grid.fft(np.asarray(eng.A[1], complex)))[k] ** 2
    s1n = np.abs(grid.fft(np.asarray(eng.A[2], complex)))[k_neg] ** 2
    return dict(f=w_thz[k], g_ch1=s1 / s0, g_ch2=s1n / max(s0n, 1e-300),
                G_sqrt=np.sqrt(s1 / s0), en_drift=abs(eng.energy_vs_z[-1] / eng.energy_vs_z[0] - 1))


t0 = time.time()
for f in [root2 - 0.05, root2, root2 + 0.05, 60.0]:
    r = probe(f)
    print(f"f={r['f']:8.3f} THz  amp-gain ch1={r['G_sqrt']:8.4f}  "
          f"ch2spont={r['g_ch2']:9.2e}  drift={r['en_drift']:.2e}  "
          f"({time.time()-t0:.0f}s)")
r = probe(root2, mode_offsets="none")
print(f"offset-none control: amp-gain {r['G_sqrt']:.4f}")
r = probe(root2, mode_offsets="plus")
print(f"offset-plus: amp-gain {r['G_sqrt']:.4f}")
r = probe(root2, ch2="zero")
print(f"decayed (ch2=0): ch2 spontaneous = {r['g_ch2']:.3e}")
