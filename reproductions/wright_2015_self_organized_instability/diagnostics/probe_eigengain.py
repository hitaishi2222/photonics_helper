"""Session-2 bisect: verify the analytic reduced pair model against the engine.

Derived from the engine's own operator algebra (small signal, pump const):

    pair variables (b1, b2*) in the sideband diagonal frames;
    d b1/dz = i c e^{i Xi z} b2*,  d b2*/dz = -i c e^{-i Xi z} b1,
    c   = (2/3) gamma P0                (lp_degenerate FWM arm, f = 2/3)
    Xi  = 2 alpha0 - alpha1 - alpha2
        = (2/3) gamma P0 - sym(Omega) - 2*phase_offset
    sym = 2*phi_b (even Taylor pair sum) = material sum if betas fit it
    alpha_m = linear per-metre phase of channel m + diagonal SPM/XPM rate
    (pump alpha0 = gamma P0; sidebands alpha = phi_b + db0 + (2/3) gamma P0).

Gain peak: Xi = 0  ->  0.5*sym(Omega) = N kappa + gamma P0/3.
Gain: g = sqrt(c^2 - (Xi/2)^2); on-resonance amplitude growth cosh(cL).

Probes (all short, N=8192):
  A. betas_taylor fidelity: sym_engine vs Sellmeier sym (whole sweep).
  B. mismatch probe at the OLD root: predicted Xi = 10.09; previous session
     measured 10.4 by z-oscillation -> theory/experiment agreement.
  C. eigen-growth at the corrected root over L=1 m vs cosh(cL)=11.5.
  D. off-resonance control (+/-0.06 THz): growth ~ 1, g_analytic ~ 0.
"""
import importlib.util
from math import factorial

import numpy as np

spec = importlib.util.spec_from_file_location(
    "rep", "reproductions/planned/wright_2015_self_organized_instability/reproduce.py")
rep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rep)

N = 8192
WIN = 25.0
P0 = rep.P0_W
AP = np.sqrt(P0)
GAMMA = 2 * np.pi * rep.N2 / rep.LAMBDA0 / rep.A_EFF
C = (2 / 3) * GAMMA * P0
NORD = 2
BETAS = rep.betas_taylor()
L = 1.0
DZ = 5e-4
K = int(round(L / DZ))

# ---------------------------------------------------------------- A. fit
f_check = np.linspace(30, 250, 23)
om = 2 * np.pi * f_check * 1e-12  # rad/ps
phi = np.zeros_like(om)
for k, b in enumerate(BETAS, start=2):
    phi += (b * 1e12**k) * om**k / factorial(k)
sym_engine = 2 * phi
sym_true = np.array([rep.beta0(rep.PUMP_THZ + f) + rep.beta0(rep.PUMP_THZ - f)
                     - 2 * rep.beta0(rep.PUMP_THZ) for f in f_check])
err = sym_engine - sym_true
print(f"[A] betas_taylor sym error: max|d| = {np.max(np.abs(err)):.3e} rad/m "
      f"(typical |Xi| scale ~ 3-10 rad/m)")
bad = f_check[np.abs(err) > 0.5]
if bad.size:
    print("    THz points with |err|>1:", bad)

# ------------------------------------------------------- analytic targets
f_old = rep.stmi_shift_thz(2)          # old (wrong-Kerr) condition
f_new = 0.5 * (rep.stmi_shift_thz(2) + f_old)  # placeholder, computed below


def sym_at(f_thz: float) -> float:
    return rep.beta0(rep.PUMP_THZ + f_thz) + rep.beta0(rep.PUMP_THZ - f_thz) \
        - 2 * rep.beta0(rep.PUMP_THZ)


from scipy.optimize import brentq

f_new = brentq(lambda f: 0.5 * sym_at(f) - NORD * rep.KAPPA - GAMMA * P0 / 3,
               f_old - 1.0, f_old + 1.0)
print(f"    old root (reproduce's Kerr term) f = {f_old:.3f} THz")
print(f"    corrected root (0.5 sym = N k + gP0/3) f = {f_new:.3f} THz")

xi_old = C - sym_at(f_old) + 2 * NORD * rep.KAPPA
xi_new = C - sym_at(f_new) + 2 * NORD * rep.KAPPA
print(f"[B] predicted engine Xi at OLD root = {xi_old:+.3f} rad/m "
      "(previous session z-osc probe measured 10.4)")


def xi_model(f_thz: float) -> float:
    return C - sym_at(f_thz) + 2 * NORD * rep.KAPPA


def g_model(f_thz: float) -> float:
    x = xi_model(f_thz)
    return float(np.sqrt(max(C * C - (x / 2) ** 2, 0.0)))


print(f"    Xi(corrected root) = {xi_new:+.2e}; g_model(root) = {g_model(f_new):.3f} /m")

# ---------------------------------------------------------------- engine
from photonics_helper.base import Area, Length, Wavelength
from photonics_helper.gnlse import FiberProfile
from photonics_helper.multimode_gnlse import MultimodeSplitStepEngine
from photonics_helper.pulse import Envelope, TemporalGrid, Wave

grid = TemporalGrid(N=N, Tmax=rep.Time(WIN * 1e-12, "s"))
tt = grid.t
tt = tt.as_s if hasattr(tt, "as_s") else tt
w_thz = np.abs(grid.w) / (2 * np.pi * 1e12)


def engine(fwm, s1, s2, f_thz, off, L_m=L, dz=DZ):
    omp = 2 * np.pi * f_thz * 1e12
    tone = np.exp(1j * omp * tt)
    waves = []
    for ch in range(3):
        ww = Wave(grid=grid, envelope=Envelope(
            shape="gaussian", peak_amplitude=AP,
            pulse_width=rep.Time(WIN * 1e-12 / 4.0, "s")),
            central_wavelength=Wavelength(rep.LAMBDA0_NM, "nm"))
        if ch == 0:
            field = np.full(N, AP, dtype=complex)
        elif ch == 1:
            field = s1 * tone
        else:
            field = s2 * np.conj(tone)
        waves.append(ww.with_field(np.asarray(field, complex)))
    fiber = FiberProfile(n2=rep.N2, alpha=0.0, A_eff=Area(rep.A_EFF, "m^2"),
                         length=Length(L_m, "m"))
    return MultimodeSplitStepEngine(
        waves, fiber, betas=[BETAS] * 3, betas_unit="s^k/m",
        phase_offsets=[0.0, -off * rep.KAPPA, -off * rep.KAPPA],
        oam_l=[0, 0, 0], coef_model="lp_degenerate", include_fwm=fwm,
        fwm_pump_depletion=fwm, step_size=Length(dz, "m"))


def bin_amps(eng, f_thz):
    sp1 = grid.fft(np.asarray(eng.A[1], complex))
    sp2 = grid.fft(np.asarray(eng.A[2], complex))
    kf = int(np.argmin(np.abs(w_thz - f_thz)))
    kn = int(np.argmin(np.abs(grid.w + grid.w[kf])))
    return abs(sp1[kf]), abs(sp1[kn]), abs(sp2[kf]), abs(sp2[kn])


def eigen_growth(f_thz, off, verbose=True):
    """Measured 2x2 map eigenvalue -> growth rate over L (FWM minus ref)."""
    a = 1e-6 * AP
    e = engine(False, a, 0.0, f_thz, off)
    e.propagate(K, nsaves=2)
    r1 = bin_amps(e, f_thz)
    e = engine(False, 0.0, a, f_thz, off)
    e.propagate(K, nsaves=2)
    r2 = bin_amps(e, f_thz)
    e = engine(True, a, 0.0, f_thz, off)
    e.propagate(K, nsaves=2)
    sA = bin_amps(e, f_thz)
    driftA = e.energy_vs_z[-1] / e.energy_vs_z[0]
    e = engine(True, 0.0, a, f_thz, off)
    e.propagate(K, nsaves=2)
    sB = bin_amps(e, f_thz)
    drift = max(driftA, e.energy_vs_z[-1] / e.energy_vs_z[0])
    G = np.array([[sA[0] / a, sB[0] / a],
                  [sA[3] / a, sB[3] / a]], dtype=complex)
    G[0, 0] *= np.exp(-1j * np.angle(r1[0] / a))
    G[0, 1] *= np.exp(-1j * np.angle(r2[0] / a))
    G[1, 0] *= np.exp(-1j * np.angle(r1[3] / a))
    G[1, 1] *= np.exp(-1j * np.angle(r2[3] / a))
    ev = np.linalg.eigvals(G)
    mu = float(np.max(np.log(np.abs(np.maximum(ev, 1e-300)))) / L)
    if verbose:
        print(f"    f={f_thz:7.3f} off={off:+d}: |eig|="
              f"{[abs(x) for x in ev]}  mu={mu:+.3f}/m  "
              f"g_model={g_model(f_thz):.3f}  E-ratio={drift:.8f}")
    return mu


print("[C] eigen-growth scan around corrected root (L=1 m):")
for f in [f_new - 0.06, f_new - 0.03, f_new, f_new + 0.03, f_new + 0.06]:
    eigen_growth(f, NORD)
print("[D] control at old root (should be near-gainless):")
eigen_growth(f_old, NORD)
print(f"    g_model(old root) = {g_model(f_old):.4f} /m -> "
      f"cosh({g_model(f_old)*L:.2f}) = {np.cosh(g_model(f_old)*L):.3f}")

print(f"\ntheory check: c = {C:.4f} /m;  cosh(c*L) = {np.cosh(C*L):.2f} at root")
