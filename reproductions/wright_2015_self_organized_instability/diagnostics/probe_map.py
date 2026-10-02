"""Extract the engine's per-bin-pair 2x2 linear map empirically and compare
with closed form from the engine's own operators."""

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
F_REP = 100.0
grid = TemporalGrid(N=N, Tmax=Time(WIN * 1e-12, "s"))
tt = grid.t
tt = tt.as_s if hasattr(tt, "as_s") else tt
w_thz = np.abs(grid.w) / (2 * np.pi * 1e12)
kf = int(np.argmin(np.abs(w_thz - F_REP)))
kn = int(np.argmin(np.abs(grid.w + grid.w[kf])))
om = grid.w[kf]
# (scratch var removed)
xpm_rate = GAMMA * (2 / 3) * P0  # 3.13 rad/m
coupling_rate = xpm_rate  # c/|ap|^2
K, dz = 5000, 1e-5  # 50 m?? no: dz*K = 0.05 m... keep small: K=5000 steps=0.05 m
K = 5000  # 0.05 m: coupling response g*z = 3.13*0.05 = 0.157

off_N = 2


def engine(fwm, s1, s2):
    waves = []
    for ch in range(3):
        env = Envelope(
            shape="gaussian", peak_amplitude=ap, pulse_width=Time(WIN * 1e-12 / 20, "s")
        )
        W = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(532.0, "nm"))
        if ch == 0:
            field = np.full(N, ap, dtype=complex)
        elif ch == 1:
            field = s1 * np.exp(1j * om * tt)
        else:
            field = s2 * np.exp(-1j * om * tt)
        waves.append(W.with_field(np.asarray(field, complex)))
    fiber = FiberProfile(
        n2=rep.N2, alpha=0.0, A_eff=Area(rep.A_EFF, "m^2"), length=Length(dz * K, "m")
    )
    return MultimodeSplitStepEngine(
        waves,
        fiber,
        betas=[[b2]] * 3,
        betas_unit="s^k/m",
        phase_offsets=[0.0, -off_N * rep.KAPPA, -off_N * rep.KAPPA],
        oam_l=[0, 0, 0],
        coef_model="lp_degenerate",
        include_fwm=fwm,
        fwm_pump_depletion=fwm,
        step_size=Length(dz, "m"),
    )


def b_amp(eng, ch, k):
    return complex(grid.fft(np.asarray(eng.A[ch], complex))[k])


a_sig = 1e-5 * ap
reads = {}
for name, (s1v, s2v), fwm in [
    ("A", (a_sig, 0.0), True),
    ("B", (0.0, a_sig), True),
    ("refA", (a_sig, 0.0), False),
    ("refB", (0.0, a_sig), False),
]:
    eng = engine(fwm, s1v, s2v)
    eng.propagate(K, nsaves=2)
    reads[name] = (b_amp(eng, 1, kf), b_amp(eng, 2, kn))

# input amplitudes in the same readout norm:
a_in_read = abs(b_amp((lambda: None)() or engine(False, 1e-300, 0.0), 1, kf)) * 0
eng0 = engine(False, a_sig, 0.0)
a1_in = b_amp(eng0, 1, kf)
eng0 = engine(False, 0.0, a_sig)
a2_in = b_amp(eng0, 2, kn)
print("readout norms: a1_in =", a1_in, " a2_in =", a2_in)

# Remove the reference (fwm-off) rotation from the FWM-on reads:
col1 = (reads["A"][0] / a1_in, reads["A"][1] / a2_in)
col2 = (reads["B"][0] / a1_in, reads["B"][1] / a2_in)
M = np.array([[col1[0], col2[0]], [col1[1], col2[1]]])
# subtract pure linear rotation from reference runs:
refA = (reads["refA"][0] / a1_in, reads["refA"][1] / a2_in)
refB = (reads["refB"][0] / a1_in, reads["refB"][1] / a2_in)
# M_expected (no FWM): diagonal e^{i u dz K} but per channel: ch1 rotated u1, ch2 by u2:
# compare measured M off vs predicted separate diagonals:
print("measured FWM-ON couplings (normalized by input tones):")
print(
    " col A (a1 seed):", [f"{x:.6f}" if isinstance(x, float) else x for x in reads["A"]]
)
print(
    " col B (a2 seed):",
    [
        complex(round(x.real, 6), round(x.imag, 6)) if isinstance(x, complex) else x
        for x in reads["B"]
    ],
)
print(
    " FWM-OFF refs:",
    [complex(round(x.real, 12), round(x.imag, 12)) for x in reads["refA"]],
    [complex(round(x.real, 12), round(x.imag, 12)) for x in reads["refB"]],
)
grow = np.array(
    [
        [reads["A"][0] / a1_in, reads["B"][0] / a1_in],
        [reads["A"][1] / a2_in, reads["B"][1] / a2_in],
    ]
)
ev, evec = np.linalg.eig(grow)
zs = dz * K
print(f"growth rates ln|eig|/{zs} m = {np.log(np.abs(ev)) / zs} /m")
print(f"analytic pair coupling gfP0 = {(2 / 3) * GAMMA * P0:.4f} /m")
print(
    "mismatch Xi = (2/3)gP0 - (b2 om^2 + 2off): "
    + str((2 / 3) * GAMMA * P0 - (b2 * (om * 1e-12) ** 2 / 2 * 2 + 0.0))
    + " rad/m "
    "(b2 rotation per-metre: beta2*om_ps^2 =" + str(b2 * (om * 1e-12) ** 2) + ")"
)
