"""Exact-B bin map from an FFT'd tone: no grid-rounding assumptions."""

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
GAMMA = 2 * np.pi * rep.N2 / rep.LAMBDA0 / rep.A_EFF
P0 = rep.P0_W
grid = TemporalGrid(N=N, Tmax=Time(WIN * 1e-12, "s"))
tt = grid.t.as_s if hasattr(grid.t, "as_s") else grid.t
w = np.asarray(grid.w)


def binof(f_thz, sign):
    """Bin index where a tone exp(+/-i om t) has its (real) FFT peak."""
    tone = np.exp(sign * 1j * 2 * np.pi * f_thz * 1e12 * tt)
    S = np.abs(np.asarray(grid.fft(tone)))
    return int(np.argmax(S))


def run(f_thz, offs, L=0.3, use2x2=False, step_m=4e-5):
    k = binof(f_thz, +1)
    om = w[k]
    k_neg = binof(f_thz, -1)
    a0 = 1e-5 * ap
    tone = np.exp(1j * om * tt)

    def make(s1, s2):
        waves = []
        for ch in range(3):
            env = Envelope(
                shape="gaussian",
                peak_amplitude=ap,
                pulse_width=Time(WIN * 1e-12 / 4, "s"),
            )
            W = Wave(
                grid=grid, envelope=env, central_wavelength=Wavelength(532.0, "nm")
            )
            if ch == 0:
                field = np.full(N, ap, dtype=complex)
            elif ch == 1:
                field = a0 * s1 * tone
            else:
                field = a0 * s2 * np.conj(tone)
            waves.append(W.with_field(np.asarray(field, complex)))
        return MultimodeSplitStepEngine(
            waves,
            FiberProfile(
                n2=rep.N2,
                alpha=0.0,
                A_eff=Area(rep.A_EFF, "m^2"),
                length=Length(L, "m"),
            ),
            betas=[[b2]] * 3,
            betas_unit="s^k/m",
            phase_offsets=offs,
            oam_l=[0, 0, 0],
            coef_model="lp_degenerate",
            include_fwm=True,
            fwm_pump_depletion=True,
            step_size=Length(step_m, "m"),
        )

    # readout norms: fwm-off reference run of each single-seed column
    a1_in = abs(np.asarray(grid.fft(np.asarray(make(1.0, 0.0).A[1], complex)))[k])
    a2_in = abs(np.asarray(grid.fft(np.asarray(make(0.0, 1.0).A[2], complex)))[k_neg])

    def read(s1, s2):
        eng = make(s1, s2)
        eng.propagate(int(round(L / step_m)), nsaves=2)
        b1 = abs(np.asarray(grid.fft(np.asarray(eng.A[1], complex)))[k])
        b2amp = abs(np.asarray(grid.fft(np.asarray(eng.A[2], complex)))[k_neg])
        return b1 / a1_in, b2amp / a2_in

    A = read(1.0, 0.0)  # col A: (b1, b2) from seeding a1
    B = read(0.0, 1.0)  # col B: (b1, b2) from seeding a2
    M = np.array([[A[0], B[0]], [A[1], B[1]]])
    ev = np.linalg.eigvals(M)
    zs = L
    return M, np.log(np.abs(ev)) / zs, ev


N_ord = 2
OFF = [0.0, -N_ord * rep.KAPPA, -N_ord * rep.KAPPA]
f_root = rep.stmi_shift_thz(N_ord)


def d_new(f):
    sym = (
        rep.beta0(rep.PUMP_THZ + f)
        + rep.beta0(rep.PUMP_THZ - f)
        - 2 * rep.beta0(rep.PUMP_THZ)
    )
    return 0.5 * sym - N_ord * rep.KAPPA - GAMMA * P0 / 3.0


c = (2 / 3) * GAMMA * P0
for f in [f_root - 0.05, f_root, f_root + 0.05]:
    M, lnabs, ev = run(float(f), OFF)
    g1, g2 = lnabs
    g1, g2 = ev
    print(f"f={f:8.3f} THz  d_pred={d_new(float(f)):9.2f} rad/m")
    print(
        f"   M = [[{M[0, 0]:.5f},{M[0, 1]:.5f}],[{M[1, 0]:.5f},{M[1, 1]:.5f}]]"
        f"  ln|eig|/L = {g1:9.4f} {g2:9.4f}"
        f"  (cosh-model at d: g={np.sqrt(max(c * c - d_new(float(f)) ** 2, 0)):.3f})"
    )
