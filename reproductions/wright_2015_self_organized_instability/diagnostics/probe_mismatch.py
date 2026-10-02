"""Measure the engine's effective FWM pair mismatch delta_eff(z-oscillation)."""

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
gf = rep.gamma_eff() * rep.P0_W
grid = TemporalGrid(N=N, Tmax=Time(WIN * 1e-12, "s"))
tt = grid.t
tt = tt.as_s if hasattr(tt, "as_s") else tt
w_thz = np.abs(grid.w) / (2 * np.pi * 1e12)
k = int(np.argmin(np.abs(w_thz - rep.stmi_shift_thz(2))))
Om = 2 * np.pi * w_thz[k] * 1e12

for tag, offs in [
    ("minus", [0, -2 * rep.KAPPA, -2 * rep.KAPPA]),
    ("plus", [0, +2 * rep.KAPPA, +2 * rep.KAPPA]),
    ("half", [0, -rep.KAPPA, -rep.KAPPA]),
    ("none", None),
]:
    waves = []
    L, dz, steps = 0.3, 4e-5, None
    steps = int(round(L / dz))
    a0 = 1e-4 * ap
    for ch, phi in zip(range(3), [0.0] * 3):
        env = Envelope(
            shape="gaussian", peak_amplitude=ap, pulse_width=Time(WIN * 1e-12 / 4, "s")
        )
        W = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(532.0, "nm"))
        if ch == 0:
            field = np.full(N, ap, dtype=complex)
        elif ch == 1:
            field = a0 * np.exp(1j * Om * tt)
        else:
            field = a0 * np.exp(-1j * Om * tt)
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
        step_size=Length(dz, "m"),
    )
    eng.propagate(steps, nsaves=301)
    zs = np.linspace(0, L, 301)
    # amplitude of ch1 tone at bin k vs z (project onto the tone)
    amps = []
    for snap in eng.evolution:
        A = np.asarray(snap[1]._pulse_train_field, complex)
        amps.append(
            abs(A @ np.conj(np.exp(1j * Om * tt)))
            / N
            * (
                1.0 / (grid.dt.as_s if hasattr(grid.dt, "as_s") else grid.dt) / N
                if False
                else 1.0
            )
        )
    amps = np.array(amps) / abs(amps[0])
    # locate oscillation period: peaks of amps
    # FFT of amps over z
    fa = np.fft.rfft(amps * np.hanning(len(amps)))
    freqs = np.fft.rfftfreq(len(amps), d=(zs[1] - zs[0]))
    ipk = np.argmax(np.abs(fa[1:]) + 1) + 1
    if freqs[ipk] > 0:
        period = 1.0 / freqs[ipk]
        delta_eff = np.pi / period
    else:
        period, delta_eff = np.inf, 0.0
    # theoretical candidates
    sym = (
        rep.beta0(rep.PUMP_THZ + w_thz[k])
        + rep.beta0(rep.PUMP_THZ - w_thz[k])
        - 2 * rep.beta0(rep.PUMP_THZ)
    )
    print(
        f"{tag:5s}: dominant z-osc period {period * 1e3:8.3f} mm -> "
        f"delta_eff = {delta_eff:12.1f} rad/m (|delta/pi per m|); amp range "
        f"{amps.min():.3f}..{amps.max():.3f}"
    )
    print(
        f"      candidates: 2Nkappa={4 * rep.KAPPA:.0f}, 2Nkappa-2gf*P0/="
        f" {4 * rep.KAPPA:.0f}, sym-2Nkappa+2gf*P0={sym - 4 * rep.KAPPA + 4 * gf:.0f},"
        f" sym+2Nkappa={sym + 4 * rep.KAPPA:.0f}, sym={sym:.1f}"
    )
