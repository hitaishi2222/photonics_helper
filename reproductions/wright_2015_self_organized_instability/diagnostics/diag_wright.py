"""Diagnose missing multimode FWM/MI gain at the GRIN ladder resonance."""
import importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location(
    "rep", "reproductions/planned/wright_2015_self_organized_instability/reproduce.py")
rep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rep)

from photonics_helper.base import Area, Length, Time, Wavelength
from photonics_helper.gnlse import FiberProfile
from photonics_helper.multimode_gnlse import MultimodeSplitStepEngine
from photonics_helper.pulse import TemporalGrid, Wave, Envelope


def build(mode, n_ord=2, L=0.3, N=16384, win=25.0, dz=4e-5):
    grid = TemporalGrid(N=N, Tmax=Time(win * 1e-12, "s"))
    ap = np.sqrt(rep.P0_W)
    b2 = rep.beta2_silica(rep.LAMBDA0_NM)
    KAP = rep.KAPPA
    if mode.startswith("none"):
        offs = None
    else:
        s = {"minus": -1.0, "plus": +1.0, "split": None}[mode.split("_")[0]]
        if s is None:
            offs = [0.0, -n_ord * KAP, +n_ord * KAP]
        else:
            offs = [0.0, s * n_ord * KAP, s * n_ord * KAP]
    iso = "iso" in mode
    waves = []
    rng = np.random.default_rng(7)
    for ch in range(3):
        env = Envelope(shape="gaussian", peak_amplitude=ap,
                       pulse_width=Time(win * 1e-12 / 4, "s"))
        w = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(532.0, "nm"))
        if ch == 0:
            field = np.full(N, ap, dtype=complex)
        else:
            field = (rng.normal(0, 1e-4 * ap, N)
                     + 1j * rng.normal(0, 1e-4 * ap, N))
        waves.append(w.with_field(field))
    fiber = FiberProfile(n2=rep.N2, alpha=0.0, A_eff=Area(rep.A_EFF, "m^2"),
                         length=Length(L, "m"))
    eng = MultimodeSplitStepEngine(
        waves, fiber, betas=[[b2]] * 3, betas_unit="s^k/m",
        phase_offsets=offs, oam_l=[0, 0, 0],
        coef_model=("isotropic" if iso else "lp_degenerate"),
        include_fwm=True, fwm_pump_depletion=True,
        step_size=Length(dz, "m"))
    return eng, grid


def gain_report(eng, grid, n_ord, tag):
    f = np.abs(grid.w) / (2 * np.pi * 1e12)
    s0 = np.abs(grid.fft(np.asarray(eng.A[1], complex))) ** 2
    eng.propagate(int(round(eng.fiber.length.as_m / 4e-5)), nsaves=2)
    s1 = np.abs(grid.fft(np.asarray(eng.A[1], complex))) ** 2

    def br(lo, hi):
        m = (f >= lo) & (f < hi)
        return float(s1[m].sum() / max(s0[m].sum(), 1e-300))

    On = rep.stmi_shift_thz(n_ord)
    print(f"{tag:12s} root={On:7.2f} THz  rootband={br(On-0.05, On+0.05):9.4f}  "
          f"elsewhere={br(20, 90):7.4f}  drift={abs(eng.energy_vs_z[-1]/eng.energy_vs_z[0]-1):.2e}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "scan":
        eng, grid = build("minus")
        f = np.abs(grid.w) / (2 * np.pi * 1e12)
        s0 = np.abs(grid.fft(np.asarray(eng.A[1], complex))) ** 2
        eng.propagate(7500, nsaves=2)
        s1 = np.abs(grid.fft(np.asarray(eng.A[1], complex))) ** 2
        ratio = s1 / np.minimum(np.maximum(s0, 1e-300), 1e300)
        pos = f >= 0
        idx = np.argsort(ratio[pos])[::-1][:12]
        print("top gain bins (freq THz, ratio, s0):")
        for i in idx:
            print(f"  {f[pos][i]:9.3f}  {ratio[pos][i]:12.4f}  {s0[pos][i]:.3e}")
        # pump + sideband band energies
        for ch in range(3):
            E = float(np.sum(np.abs(np.asarray(eng.A[ch], complex)) ** 2))
            print(f"channel {ch} final energy(sum|A|^2): {E:.6e}")
    else:
        for mode in ["minus", "plus", "split", "none", "minus_iso"]:
            eng, grid = build(mode)
            gain_report(eng, grid, 2, mode)
