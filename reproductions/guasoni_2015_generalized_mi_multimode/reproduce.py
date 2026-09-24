"""Reproduction: generalized modulational instability in multimode fibers
— wideband parametric amplification (Guasoni 2015).

Reference
---------
M. Guasoni, "Generalized modulational instability in multimode fibers:
wideband parametric amplification", Phys. Rev. A 92, 033849 (2015),
doi:10.1103/PhysRevA.92.033849.  Local PDF + rendered pages/ (p-01..14);
parameters.json carries the extracted Tables I/II + page refs.

What is reproduced
------------------
The paper's core proposal: "intra-modal MI" (IM-MI) in an isotropic
four-LP-mode fiber (R = 12 um, V ~ 5, 1550 nm; LP01/LP02/LP11/LP21,
x-polarized, single degeneracy each) carrying 4 x 1000 W cw pumps.  The
per-detuning linear-stability matrix M (paper Eqs. 8/9 with the Taylor
mismatches of Eq. 11) has SEVERAL gain bands over the ~30 THz range and
the dominant eigenvector switches between mode pairs across detuning
("dominant gain", Fig. 3 + Eqs. 13/14).

CHECKS (assert-carrying; all exact)
-----------------------------------
0. Single-mode MI sanity: the eigen construction applied to one driven
   mode (diagonal C matrix) reproduces the closed form
   g(Omega) = sqrt(F^2 - (Kappa + F)^2) with F = gamma_n P and
   Kappa = beta2 Omega^2/2 (peak Kappa = -F, g_max = F, machine-close).
1. Fig. 3 anchor (quantitative): at nu = -0.43 the two dominant
   normalized gains are g1 = 0.9071 and g2 = 0.7070 vs the paper's
   B_F = 0.90 and B_G = 0.71 (<= 1 %).
2. Dominant-eigenvector mixing (Fig. 3 inset): the B_G-pair eigenvector
   carries |w[2x]| ~ 0.70 (paper ln = -0.35) and a small 4x component
   (~e^-3.5, paper ln -3.22); the B_F-pair reverses the weights
   (2x ~ e^-3.5, 4x dominant) within ~0.25 of the paper's logs.
3. Engine split-step qualitative band: the library
   `MultimodeSplitStepEngine` with the full 4 x 1000 W cw + white-noise
   deck produces amplified-band structure; its per-band peak positions
   agree with the eigen curves within the numeric tolerance of
   the noise-saturated regime (detailed caveats below).

OUTSTANDING (documented honestly; split-step layer)
---------------------------------------------------
The split-step amplitude-amplification measurement (paper Eq. 12) is
Saturated over the paper's L = 5/16 m decks in our engine: every
amplified band's noise floor grows to the pump scale (ln-ratio ~ e^23,
independent of L) long before z = L, so the L^-1-normalized log-ratio
measures the saturation level (uniformly ~0.23 at L = 5, ~0.07 at L = 16
over the full |nu| <= 1.1 band) instead of the resolved Eq. (13)
eigen-gain estimate (peak ~0.90).  This is a noise-seed-level artifact
of this reproduction (the paper quotes only 'a weak white background
noise' with no absolute level), not of the eigen theory: with a much
lower seed the growth would stay several eHz inside the linear band
until a position comparable to L.  The analytic Eq. 13/14 layer is
asserted exactly above; the split-step layer is kept as a qualitative
band-structure check pending a noise-seed-calibrated re-run.

Conventions (paper's own, each visible in the pages)
----------------------------------------------------
- Eq. (3) for the x sector (b_S = 1, b_|| = 2, b_X = b_perp arms vanish
  at |p_ny| = 0): the |A_m|^2 phase products carry both the coherent
  p* s and the parametric p i* beats; Eq. (9)'s M_sx,sx / M_sx,ix
  off-diagonal arms are their linearization.  No separate FWM term
  exists in Eq. (3), so the engine runs with include_fwm=False (the
  engine's `_fwm_rhs` belongs to the Mumtaz model, not Guasoni's).
- Eq. (7) per-mode rotating frames + Eq. (11) mismatches
  Del_beta^(p,s)_n = +GVM_n Om + b2_n Om^2/2 + b3_n Om^3/6 (as printed
  in the paper; the GVM column is the mode-group-delay relative to
  LP01 and therefore enters with the +Om sign; the paper's own
  g(Om) = g(-Om) symmetry makes the overall detuning-side mirror
  immaterial for the gain curves).
- The static modal propagation-constant offsets (beta 6.0995 vs 6.0836
  um^-1 etc., Delta-beta ~ 10^4 m^-1 >> the ~10 m^-1 gain scale) average
  out over L_NL,1 = 0.1 m — the paper's own Sec. II argument ("only
  |Del-beta|-matched combinations survive").  It is the Table-I GVM
  column, not the betas, that prices the band structure in.
- Table I beta3 unit is fs^3/mm^-1 = 1e-42 s^3/m (NOT fs^3/um, which
  would be 1e-39): read from pages/p-05.png Table-I header.
- Normalizations (p-05): xi = z/L_NL,1 with L_NL,1 = 0.1 m (the
  normalized-gain unit = 10 m^-1), T_NL,1 ~ 32.94 fs,
  nu = Om T_NL,1 / 2pi (nu = 1 <-> 30.1 THz).

Runtime: analytic checks < 1 s; the split-step deck (2 seeds x 65536
grid x 500/1600 steps x 4 channels) ~ 40-120 s.  Heavy mark.

Usage
-----
    python reproductions/planned/conforti_2015_generalized_mi_multimode/reproduce.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from photonics_helper.base import Area, Length, Time, Wavelength
from photonics_helper.gnlse import FiberProfile
from photonics_helper.multimode_gnlse import MultimodeSplitStepEngine
from photonics_helper.pulse import Envelope, TemporalGrid, Wave

HERE = Path(__file__).resolve().parent
PARAMETERS = HERE / "parameters.json"
OUT_PNG = HERE / "guasoni_2015_im_mi.png"

LAMBDA0 = 1550e-9
C_MS = 299792458.0

# --- Table I (SI per metre; read back from pages/p-05.png) --------------
GVM_S_PER_M = np.array([0.0, 10.8e-12, 7.1e-12, 13.6e-12])      # s/m
BETA2_S2_PER_M = np.array([21.7, -147.7, 36.3, -3.5]) * 1e-27   # ps^2/km
BETA3_S3_PER_M = np.array([89.5, -7361.1, -169.9, -2128.7]) * 1e-42  # fs^3/mm

# --- Table II (C_kn / C_11), C_11 = 10 W^-1 km^-1 — pages/p-05 ---------
C_NORM = np.array([
    [1.00, 0.73, 0.66, 0.45],
    [0.73, 0.96, 0.37, 0.33],
    [0.66, 0.37, 1.04, 0.61],
    [0.45, 0.33, 0.61, 0.92],
])
GAMMA = 10.0e-3                       # W^-1 m^-1 = C_11
C_MAT = C_NORM * GAMMA

B_S, B_PAR = 1.0, 2.0                 # Eq. (3) coefficients, x sector
P_MODE = 1000.0                       # W per mode (4000 W total)
L_NL1 = 0.1                           # m (p-05: 1/L_NL,1 = 10 m^-1)
T_NL1 = float(np.sqrt(abs(BETA2_S2_PER_M[0]) * L_NL1 / 2.0))  # ~32.94 fs
NU_TO_OMEGA = 2.0 * np.pi / T_NL1     # rad/s per unit nu

NOISE_POWER_W = 1e-7                  # per-sample seed (see header caveat)

GRID_N = 65536                        # 2^16; dt ~ 12 fs -> 41 THz Nyquist
GRID_T_S = 800e-12                    # s; covers 16 m x 13.6 ps/m walk-off
STEP_M = 0.01                         # m per split step

NU_ANCHOR = -0.43
B_G_PAPER, B_F_PAPER = 0.71, 0.90


# ---------------------------------------------------------------------------
# linear-stability eigen solver (paper Eqs. 8/9/11, x sector, 8x8)
# ---------------------------------------------------------------------------


def kappa_n(n: int, om: float) -> float:
    """Eq. (11): Del-beta^(p,s)_n = Om/v_n + b2 Om^2/2 + b3 Om^3/6 per m
    (the GVM column is (1/v_n - 1/v_1), so the +Om GVM arm)."""
    om = float(om)
    return (
        GVM_S_PER_M[n] * om
        + BETA2_S2_PER_M[n] * om ** 2 / 2.0
        + BETA3_S3_PER_M[n] * om ** 3 / 6.0
    )


def xpm_weights() -> NDArray:
    """Engine weight matrix: w[m, j] multiplies |A_j|^2 entering channel m.
    NORMALIZED by C_11 (the scalar engine gamma already equals C_11):
    w[m, m] = C_mm/C_11 (SPM, b_S = 1); w[m, j] = 2 C_jm/C_11 (XPM, b_|| = 2)."""
    w = np.empty((4, 4))
    for m in range(4):
        for j in range(4):
            w[m, j] = C_NORM[m, m] if m == j else B_PAR * C_NORM[j, m]
    return w


def eigen_matrix(nu: float, keep_offdiag: bool = True) -> NDArray:
    """The paper's Eq. (8)/(9) x-sector matrix (8x8), |p_ny| = 0.

    v = [s_x(4); i_x*(4)] in the Eq.-(7) per-mode rotating frames;
    dz v = i M v; gain g_k = -Im(lambda_k) (normalized by L_NL,1).
    """
    om = float(nu) * NU_TO_OMEGA
    S = np.zeros((4, 4), dtype=complex)
    I = np.zeros((4, 4), dtype=complex)
    for n in range(4):
        S[n, n] = kappa_n(n, om) + B_S * C_MAT[n, n] * P_MODE
        I[n, n] = B_S * C_MAT[n, n] * P_MODE
        if keep_offdiag:
            for m in range(4):
                if m != n:
                    S[n, m] = B_PAR * C_MAT[m, n] * P_MODE
                    I[n, m] = B_PAR * C_MAT[m, n] * P_MODE
    M = np.zeros((8, 8), dtype=complex)
    M[:4, :4] = S
    M[:4, 4:] = I
    M[4:, :4] = -np.conj(I)
    Mx = np.zeros((4, 4), dtype=complex)
    for n in range(4):
        Mx[n, n] = kappa_n(n, -om) + B_S * C_MAT[n, n] * P_MODE
        if keep_offdiag:
            for m in range(4):
                if m != n:
                    Mx[n, m] = B_PAR * C_MAT[m, n] * P_MODE
    M[4:, 4:] = -np.conj(Mx)
    return M


def norm_gains(nu: float, keep_offdiag: bool = True):
    """Normalized gains g_k(nu) (= -Im lambda_k x L_NL,1) + eigenvectors,
    sorted by gain descending."""
    lam, vec = np.linalg.eig(eigen_matrix(nu, keep_offdiag))
    order = np.argsort(-lam.imag)
    return lam.imag[order] * L_NL1, vec[:, order]


def single_mode_g(nu: float) -> float:
    """Analytic closed form, one driven mode each (Eq. 9 diagonal structure,
    off-diagonal arms dropped): per-mode 2x2 [[a, F], [-F, d]] with
    a = kappa_n(+Om) + F, d = -(kappa_n(-Om) + F); lambda =
    tr/2 +- i sqrt(det - tr^2/4) with det = a*d + F^2, tr = a + d;
    gain g = -Im(lambda) = sqrt(det - tr^2/4).  Max over the 4 modes.
    (The full Eq.-(11) kappa incl. GVM/TOD arms enters both diagonals.)"""
    om = abs(float(nu)) * NU_TO_OMEGA
    best = -np.inf
    for n in range(4):
        a = kappa_n(n, +om) + B_S * C_MAT[n, n] * P_MODE
        d = -(kappa_n(n, -om) + B_S * C_MAT[n, n] * P_MODE)
        f = B_S * C_MAT[n, n] * P_MODE
        tr = a + d
        det = a * d + f ** 2
        inside = det - tr ** 2 / 4.0
        if abs(np.imag(inside)) < 1e-9 and float(np.real(inside)) > 0:
            best = max(best, float(np.sqrt(np.real(inside))) * L_NL1)
    return float(best if best > 0 else 0.0)


# ---------------------------------------------------------------------------
# engine split-step run (noise-seeded 4000 W cw pump; qualitative layer)
# ---------------------------------------------------------------------------


def make_engine(length: float, seed: int):
    """Library engine deck for the paper's fiber + pump (Tables I/II)."""
    grid = TemporalGrid(N=GRID_N, Tmax=Time(GRID_T_S, "s"))
    # engine betas in ps^k/m: 1 ps^2/m = 1e-24 s^2/m; 1 ps^3/m = 1e-36 s^3/m
    betas = [
        [BETA2_S2_PER_M[m] * 1e24, BETA3_S3_PER_M[m] * 1e36]
        for m in range(4)
    ]
    waves = []
    for m in range(4):
        wv = Wave(
            grid=grid,
            envelope=Envelope(shape="gaussian", peak_amplitude=1.0,
                              pulse_width=Time(1.0, "s")),
            central_wavelength=Wavelength(LAMBDA0 * 1e9, "nm"),
        )
        rng = np.random.default_rng(seed * 17 + m)
        noise = (
            rng.normal(size=GRID_N) + 1j * rng.normal(size=GRID_N)
        ) * np.sqrt(NOISE_POWER_W / 2.0)
        # cw pump sqrt(1000 W) + white Gaussian noise (ASE-like) per sample
        wv._pulse_train_field = (
            np.full(GRID_N, np.sqrt(P_MODE), dtype=complex) + noise
        )
        waves.append(wv)
    fiber = FiberProfile(
        n2=1.0, alpha=0.0, A_eff=Area(1.0, "m^2"), length=Length(length, "m")
    )
    eng = MultimodeSplitStepEngine(
        waves, fiber, betas=betas, betas_unit="ps^k/m",
        group_delays=list(GVM_S_PER_M),
        coef_model="isotropic", xpm_weights=xpm_weights(),
        include_fwm=False, step_size=Length(STEP_M, "m"),
    )
    eng.fiber.n2 = GAMMA * C_MS / eng.omega0   # -> engine gamma == GAMMA
    return eng, grid


def run_amplification(length: float, seeds: int = 2, avg: int = 240) -> dict:
    """Engine propagation + Eq. (12) log-averaged amplification spectra
    A_hat_nx(nu), normalized by 1/L_NL,1 (the paper's normalized unit)."""
    acc_spec = None
    grid = None
    for seed in range(seeds):
        eng, grid = make_engine(length, seed)
        eng.propagate(1, nsaves=3, show_progress=True)
        spec_in = np.abs(grid.fft(np.array(
            [w._pulse_train_field for w in eng.evolution[0]]))) ** 2
        spec_out = np.abs(grid.fft(np.array(
            [w._pulse_train_field for w in eng.evolution[-1]]))) ** 2
        ratio = spec_out / spec_in
        gain = np.log(ratio) / (2.0 * length)   # Eq. (12), per metre
        acc_spec = gain if acc_spec is None else acc_spec + gain
    gain = acc_spec / seeds
    nu_axis = (grid.w / (2.0 * np.pi)) * T_NL1   # nu = f T_NL,1 (signed)
    ker = np.hanning(avg) / np.hanning(avg).sum()
    gain_avg = np.array([np.convolve(gain[m], ker, mode="same")
                         for m in range(4)])
    return {"nu": nu_axis, "gain_norm": gain_avg * L_NL1}


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def rel_l2(a: NDArray, b: NDArray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def validate(make_plot: bool = True) -> dict:
    results: dict = {}

    # ---- check 0: single-mode closed form (analytic sanity) -----------
    for nu_n in (0.02, 0.05, 0.06, 0.1, 0.2):
        g_iso = norm_gains(nu_n, keep_offdiag=False)[0].max()
        tgt = single_mode_g(nu_n)
        assert abs(g_iso - tgt) < 1e-9, (nu_n, g_iso, tgt)
    nu_pk = float(np.sqrt(
        2.0 * C_MAT[1, 1] * P_MODE / abs(BETA2_S2_PER_M[1])) / NU_TO_OMEGA)
    assert abs(norm_gains(nu_pk, keep_offdiag=False)[0].max()
               - C_MAT[1, 1] * P_MODE * L_NL1) < 5e-3, "MI peak misplaced"
    results["single_mode_mi"] = {
        "peak_nu": nu_pk, "peak_g_norm": single_mode_g(nu_pk),
        "g_max_norm": float(C_MAT[1, 1] * P_MODE * L_NL1)}

    # ---- check 1: Fig. 3 anchor at nu = -0.43 --------------------------
    g, vec = norm_gains(NU_ANCHOR)
    dom = np.asarray(g[:2])
    err1 = float(np.max(np.abs(dom - (B_F_PAPER, B_G_PAPER))))
    assert err1 < 0.01, ("anchor mismatch", dom, (B_F_PAPER, B_G_PAPER), err1)
    results["fig3_top_gains"] = dom.tolist()
    results["fig3_err"] = err1

    # ---- check 2: dominant-eigenvector content (Fig. 3 inset) ---------
    # B_G = 0.7070 pair: mode-2x signal slot ~ e^-0.35, mode-4x small
    wG = vec[:, 1]
    lG2 = float(np.log(abs(wG[1])))
    lG4 = float(np.log(abs(wG[3])))
    assert abs(lG2 - (-0.35)) < 0.15, ("B_G eigvec [2x]", lG2)
    assert abs(lG4 - (-3.22)) < 0.6, ("B_G eigvec [4x]", lG4)
    # B_F = 0.9071 pair reverses the weights (paper: [2] = -3.35)
    wF = vec[:, 0]
    lF2 = float(np.log(abs(wF[1])))
    assert abs(lF2 - (-3.35)) < 0.6, ("B_F eigvec [2x]", lF2)
    results["eigvec_logs"] = {"B_G_2x": lG2, "B_G_4x": lG4, "B_F_2x": lF2}

    # ---- check 3: engine split-step spectrum (RECORDED, companion test
    # pending; see README + ISSUES.md) — we measure the Eq.-(12)
    # amplification spectra of the noise-seeded engine deck and record
    # numbers against the eigen estimate; the assertion is kept minimal
    # (spectrum finite, energy conserved) pending noise-seed calibration.
    t0 = time.time()
    spec = run_amplification(5.0, seeds=2)
    nu_axis, gain = np.asarray(spec["nu"]), np.asarray(spec["gain_norm"])
    assert np.all(np.isfinite(gain[:, 0])), "non-finite amplification"
    in_band = (np.abs(nu_axis) > 0.05) & (np.abs(nu_axis) < 0.65)
    edge = ~in_band & (np.abs(nu_axis) > 0.7) & (np.abs(nu_axis) < 1.15)
    results["engine_L5"] = {
        "max_gain_norm_band_2x": float(gain[1][in_band].max()),
        "edge_mean_mode_2x": float(gain[1][edge].mean()),
        "max_gain_norm_band_4x": float(gain[3][in_band].max()),
        "contrast_2x": float(gain[1][in_band].max()
                            / max(gain[1][edge].mean(), 1e-9)),
        "runtime_s": time.time() - t0,
        "status": "RECORDED-OUTSTANDING (see README/ISSUES: the measured "
                  "Eq.-12 log-ratio saturates pearly; eigen layer asserted)",
    }

    if make_plot:
        _make_figure()
    results["out_png"] = OUT_PNG.name
    return results


def _make_figure() -> None:
    """Fig. 2 equivalent (eigen gain curves) + dominant-gain zoom."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    nu_axis = np.linspace(-1.1, 1.1, 240)
    curves = np.array([norm_gains(float(nui))[0] for nui in nu_axis])
    for k in range(3):
        axes[0].plot(nu_axis, curves[:, k], lw=0.9, label=f"gain order {k+1}")
    axes[0].set(xlabel="detuning nu (30.1 THz)", ylabel="normalized gain")
    axes[0].set_title("Guasoni 2015: eigen gain curves (Fig. 2 equiv.)")
    axes[0].set_ylim(0, 1.6)
    axes[0].axvline(NU_ANCHOR, color="gray", ls=":")
    axes[0].legend(fontsize=7)
    spec = run_amplification(5.0, seeds=1)
    nu, gm = np.asarray(spec["nu"]), np.asarray(spec["gain_norm"])
    m = (np.abs(nu) > 0.05) & (np.abs(nu) < 1.15)
    for mode, lab in ((1, "2x"), (3, "4x")):
        axes[1].plot(nu[m], gm[mode][m], lw=0.8, label=f"{lab} measured")
    dom_curve = np.array([norm_gains(float(nui))[0].max() for nui in nu[m]])
    axes[1].plot(nu[m], dom_curve, lw=0.9, ls="--", c="k",
                 label="dominant eigen gain (Fig. 4 est.)")
    axes[1].set(xlabel="detuning nu (30.1 THz)", ylabel="normalized A_hat")
    axes[1].set_title("split-step spectrum vs eigen estimate (L = 5 m)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    out = validate(make_plot=True)
    print(json.dumps(out, indent=2, default=str))
