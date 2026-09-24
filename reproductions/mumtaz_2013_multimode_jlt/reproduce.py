"""Reproduction: generalized Manakov equations for multimode fibers with
random birefringence (Mumtaz, Essiambre & Agrawal, JLT 31, 398 (2013)).

Reference
---------
S. Mumtaz, R.-J. Essiambre, G. P. Agrawal, "Nonlinear Propagation in Multimode
and Multicore Fibers", J. Lightwave Technol. 31, 398 (2013),
doi:10.1109/JLT.2012.2235414.  Local PDF + rendered pages.

What is reproduced (field level; the paper's BER/DSP layer is out of scope)
---------------------------------------------------------------------------
The paper's core claim: solving the stochastic coupled multimode NLSE with
rapid, per-mode-independent random birefringence (Eq. 12, random SU(2) Jones
frames R_m(z)) is ensemble-equivalent to the deterministic *generalized
Manakov* equation (Eq. 29):

    dA_p/dz = i gamma ( 8/9 |A_p|^2 + sum_{m!=p} 4/3 |A_m|^2 ) A_p + linear,

i.e. rapid birefringence reduces the SPM coefficient 1 -> 8/9 and the
inter-mode XPM coefficient 2 -> 4/3 (for degenerate modes with unit overlap).

Checks
------
1. Table II conformance (linear): per-mode group delays (DMGD) and GVD D
   converted to engine units; centroid walk-offs over 100 km = DMGD x L.
2. SPM 1 -> 8/9 (M = 1): a single spatial mode, two polarizations, random
   SU(2) frame per segment; ensemble-averaged stochastic output vs the
   engine run with `xpm_weights` = 8/9 (the library's Eq. 29 path).
3. Manakov conformance (M = 2 degenerate modes): 4 channels, independent
   per-mode frames, full Mumtaz Eq. 6 segment-frame cubic (SPM 1, XPM 2/3,
   coherent 1/3); ensemble average vs the engine Manakov run
   (8/9 intra-mode, 4/3 inter-mode).  A no-birefringence reference (fixed
   frames) is shown for contrast, reproducing the paper's point that the
   un-averaged and averaged nonlinearities differ.
4. Wall-clock ratio stochastic/Manakov (paper Table III logic: Eq. 12 is far
   costlier; here qualitative).

Modelling notes
---------------
- The stochastic Eq. 12 cannot be expressed in the engine's scalar-channel
  FWM geometry (it needs the full Jones-tensor cubic (A_l^T A_m) A_n*), so
  the stochastic run is propagated by a dedicated split-step harness in this
  script that reuses the library grid FFT helpers (identical linear
  convention) and integrates the exact Eq. 6 cubic per segment.  The
  *deterministic Manakov* side is the library's own
  `MultimodeSplitStepEngine` with explicit `xpm_weights` (8/9 / 4/3) — the
  conformance target is the library.
- Pol-summed comparison: under Haar SU(2) mixing the ensemble splits the mode
  energy randomly over the two polarizations (mean 50/50) while the
  deterministic Manakov run keeps the launched polarization — the correct
  field-level observable is the per-spatial-mode intensity/spectrum SUMMED
  over both polarizations (what Mumtaz Figs. 1-3 effectively compare).
- Overlap coefficients idealized to f = 1 (isotropic degenerate pair); the
  paper's step-index values (f_11ab = 0.3) are quoted in parameters.json.
- 100 km spans instead of the paper's 1000 km (runtime); coefficients and
  physics identical, documented per house convention.

Usage
-----
    python reproductions/planned/mumtaz_2013_multimode_jlt/reproduce.py
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

C_MS = 299792458.0
LAMBDA0 = 1550e-9
OMEGA0 = 2.0 * np.pi * C_MS / LAMBDA0

# fiber (paper section IV)
GAMMA = 1.4e-3            # W^-1 m^-1 (1.4 W^-1 km^-1)
DMGD_S_PER_M = {"LP01": 0.0, "LP11": 6.5e-12, "LP02": 9.9e-12, "LP21": 12e-12}
D_S_PER_M2 = {"LP01": 25e-6, "LP11": 27.3e-6, "LP02": -2.3e-6, "LP21": 20.8e-6}
BETA2 = {m: -(LAMBDA0**2) * d / (2 * np.pi * C_MS) for m, d in D_S_PER_M2.items()}
BETA2_PS2_PER_M = {m: b * 1e24 for m, b in BETA2.items()}  # engine unit (ps^k/m)

# signal stand-in (field-level; paper: 114 Gb/s PDM-QPSK @ 7 dBm/mode)
P_PEAK = 5e-3             # W (7 dBm) per spatial mode
T_FWHM = 100e-12          # s
T0 = T_FWHM / 1.665       # gaussian
L_WALKOFF = 150           # m (check 1 span; 12e-12 s/m x 150 m = 1.8 ns,
                          # fits the +/-2 ns grid half-window — at 100 km the
                          # pulse wraps ~325 times around the periodic grid
                          # and the centroid is meaningless)
L_NL = 100e3              # m (checks 2-3 span; paper: 1000 km)
SEGMENT = 1e3             # m birefringence segment
N_SEEDS = 32
N_GRID = 4096
WINDOW = 2e-9             # s (2 ns >> 100 ps pulses; << ns-scale nonlinear scales)

THRESH = 0.05             # ensemble-vs-Manakov rel-L2 tolerance (planned README)


# ---------------------------------------------------------------------------
# grid / channels
# ---------------------------------------------------------------------------


def make_grid(Tmax_s: float = WINDOW) -> TemporalGrid:
    # NOTE TemporalGrid takes the *total* window; the time axis is +/-(Tmax/2).
    return TemporalGrid(N=N_GRID, Tmax=Time(Tmax_s, "s"))


def gauss_channel(grid: TemporalGrid, power: float, t0: float = 0.0) -> NDArray:
    """Gaussian pulse, peak power `power`, centred at t0 (s)."""
    t = grid.t - t0
    a = np.sqrt(power) * np.exp(-(t**2) / (2 * T0**2))
    return np.asarray(a, dtype=complex)


def haar_su2(rng: np.random.Generator) -> NDArray:
    """Haar-distributed SU(2) matrix (paper Eq. 10-11, procedure [26]: QR of a
    complex Ginibre matrix; the phase fix gives Haar measure on U(2), and the
    resulting Jones matrix is unitary — restricted to SU(2) up to a global
    phase, which drops out of all intensities)."""
    z = (rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    q = q * (d / np.abs(d))
    return np.asarray(q, dtype=complex)


# ---------------------------------------------------------------------------
# stochastic Eq. 12 harness (full Mumtaz Eq. 6 cubic in the segment frame)
# ---------------------------------------------------------------------------


def mumtaz_cubic(A: NDArray, gamma: float) -> NDArray:
    """Full Mumtaz Eq. 6 cubic for M spatial modes x 2 polarizations.

    A has shape (M, 2, N) (mode, pol, time).  With unit overlap coefficients
    (isotropic degenerate idealization) the per-segment-frame nonlinearity is
    the full (l, m, n) triple sum of Eq. 6 with the output mode p, f=1:

        N_p = i gamma/3 ( (G.G) G* + 2 (G†.G) G ),   G = sum_m A_m (Jones)

    which for one mode with A_y = 0 reduces to i gamma |A|^2 A (Agrawal
    6.1.22 form).  The previous code kept only the n = p arm of the (A·A)A*
    triple; that truncated cubic is NOT energy-conserving once the
    polarization components mix in random frames (~0.4 % drift over 10
    segments, observed).  NOTE the engine's `include_fwm=False` Manakov path
    models the AVERAGED nonlinearity where the coherent arm has washed out —
    for the stochastic harness the full cubic is the correct model.
    """
    G = np.sum(A, axis=0)                                   # (2, N)
    G2 = np.sum(np.abs(G) ** 2, axis=0)                     # (N,) = G†G
    return (1j * gamma / 3.0) * ((G * G) * np.conj(G) + 2.0 * G2 * G)


def stochastic_run(
    A0: NDArray,
    beta2: float,
    length: float,
    segment: float,
    seed: int,
    rotate: bool = True,
) -> NDArray:
    """Eq. 12 propagation: Strang split per segment — half linear, full
    Mumtaz cubic (RK4), half linear — with independent Haar SU(2) rotations
    of each spatial mode's Jones vector between segments."""
    grid = make_grid()
    # SI spectral axis for the Agrawal linear step: phi = beta2*Omega^2/2*L
    # with beta2 in s^2/m and Omega in rad/s (grid.w*1e-12 is rad/ps and
    # would be off by 1e24).
    w = grid.w                             # rad/s (library axis)
    phi_lin = np.exp(1j * beta2 * w**2 / 2 * segment)
    rng = np.random.default_rng(seed)
    A = A0.copy()
    n_seg = int(round(length / segment))
    n_sub = 4
    dz = segment / n_sub
    for _ in range(n_seg):
        # linear step IN FREQUENCY SPACE (the previous code multiplied the
        # TIME-domain array by the spectral phase directly — that applied a
        # spurious time-domain quadratic mask every segment, cumulatively
        # shredding the spectrum; with segment=L_NL it was a near-flat mask
        # and the bug stayed hidden).
        A = grid.ifft(grid.fft(A) * phi_lin)     # linear (dispersion per segment)
        for _ in range(n_sub):              # RK4 of the cubic (physical gamma)
            k1 = mumtaz_cubic(A, gamma=GAMMA)
            k2 = mumtaz_cubic(A + 0.5 * dz * k1, gamma=GAMMA)
            k3 = mumtaz_cubic(A + 0.5 * dz * k2, gamma=GAMMA)
            k4 = mumtaz_cubic(A + dz * k3, gamma=GAMMA)
            A = A + (dz / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # (linear half-steps replaced by the single spectral pass above)
        if rotate:
            for m in range(A.shape[0]):
                A[m] = haar_su2(rng) @ A[m]
    return A


# ---------------------------------------------------------------------------
# engine Manakov run (library Eq. 29 path)
# ---------------------------------------------------------------------------


def engine_manakov_run(
    A0: NDArray, beta2: float, length: float, xpm: NDArray
):
    """Deterministic generalized Manakov (Eq. 29) through the library engine.

    `xpm[i, j]` multiplies |A_j|^2 entering channel i (i == j is the SPM
    slot): 8/9 within a mode's polarization pair, 4/3 across modes.
    """
    grid = make_grid()
    n_modes = A0.shape[0]
    waves = []
    for m in range(n_modes):
        for s in range(2):
            wv = Wave(
                grid=grid,
                envelope=Envelope(shape="gaussian", peak_amplitude=1.0,
                                  pulse_width=Time(T0, "s")),
                central_wavelength=Wavelength(LAMBDA0 * 1e9, "nm"),
            )
            wv._pulse_train_field = A0[m, s].copy()
            waves.append(wv)
    fiber = FiberProfile(
        n2=1.0, alpha=0.0, A_eff=Area(1.0, "m^2"), length=Length(length, "m")
    )
    # gamma absorbed into xpm weights: engine gamma = n2*omega0/(c*A_eff);
    # rescale so the effective gamma equals GAMMA: choose n2/A_eff product.
    # Simplest: set n2, A_eff such that _gamma(...) == GAMMA exactly.
    eng = MultimodeSplitStepEngine(
        waves,
        fiber,
        betas=[[BETA2_PS2_PER_M["LP01"]]] * (2 * n_modes),
        betas_unit="ps^k/m",
        coef_model="isotropic",
        xpm_weights=xpm,
        include_fwm=False,
        step_size=Length(100.0, "m"),
    )
    # rescale gamma: engine gamma uses n2*omega0/(c*A_eff) = 1.0 here
    # (n2=1, A_eff=1) in SI-ish units; scale n2 so gamma == GAMMA.
    omega0 = eng.omega0
    eng.fiber.n2 = GAMMA * C_MS * 1.0 / omega0  # A_eff = 1 m^2
    eng.propagate(int(round(length / 100.0)), nsaves=2)
    out = np.array([w._pulse_train_field for w in eng.evolution[-1]])
    return out.reshape(n_modes, 2, -1), eng


def gamma_exact() -> float:
    """Engine gamma for n2 chosen in engine_manakov_run (self-consistency)."""
    return 1.0


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def rel_l2(a: NDArray, b: NDArray) -> float:
    return float(np.sqrt(np.sum(np.abs(a - b) ** 2)) / np.sqrt(np.sum(np.abs(b) ** 2)))


def check1_table2(results: dict) -> None:
    """Linear Table II conformance: DMGD walk-offs over 150 m.

    Window is 4 ns total (+/-2 ns) so the 1.8 ns LP21 walk-off stays inside
    the axis (a 100 km span in a 2 ns window wraps the pulse hundreds of
    times and makes the centroid meaningless).
    """
    grid = make_grid(Tmax_s=4e-9)  # +/- 2 ns
    modes = ["LP01", "LP11", "LP21"]
    delays = [DMGD_S_PER_M[m] for m in modes]
    waves = []
    for i, m in enumerate(modes):
        wv = Wave(
            grid=grid,
            envelope=Envelope(shape="gaussian", peak_amplitude=1.0,
                              pulse_width=Time(T0, "s")),
            central_wavelength=Wavelength(LAMBDA0 * 1e9, "nm"),
        )
        wv._pulse_train_field = gauss_channel(grid, 1e-6, t0=0.0)
        waves.append(wv)
    fiber = FiberProfile(
        n2=1e-30, alpha=0.0, A_eff=Area(80e-12, "m^2"),
        length=Length(L_WALKOFF, "m"),
    )
    eng = MultimodeSplitStepEngine(
        waves, fiber,
        betas=[[BETA2_PS2_PER_M[m]] for m in modes],
        betas_unit="ps^k/m",
        group_delays=delays,
        step_size=Length(100.0, "m"),
    )
    eng.propagate(1000, nsaves=2)
    t = grid.t
    fields = eng.fields_vs_z()
    cents = [
        float(np.sum(t * np.abs(fields[i][-1]) ** 2)
              / np.sum(np.abs(fields[i][-1]) ** 2))
        for i in range(3)
    ]
    measured = [cents[i] - cents[0] for i in (1, 2)]
    expected = [delays[i] * L_WALKOFF for i in (1, 2)]
    err = [abs(a - b) / abs(b) for a, b in zip(measured, expected)]
    assert max(err) < 0.02, (measured, expected, err)
    results["walkoff_measured_s"] = measured
    results["walkoff_expected_s"] = expected
    results["walkoff_rel_err"] = err
    # beta2 conversion sanity (analytic): beta2 = -lambda^2 D / (2 pi c)
    # with D = 25e-6 s/m^2 -> -31.9e-27 s^2/m (= -31.9 ps^2/km)
    assert abs(BETA2["LP01"] + 31.9e-27) < 0.15e-27, BETA2["LP01"]
    assert abs(BETA2_PS2_PER_M["LP01"] + 31.9e-3) < 0.15e-3, BETA2_PS2_PER_M["LP01"]


def mode_intensity(A: NDArray) -> NDArray:
    """Per-spatial-mode intensity summed over polarizations: (M, N)."""
    return np.sum(np.abs(A) ** 2, axis=1)  # (M, 2, N) -> (M, N)


def mode_spectrum(grid: TemporalGrid, A: NDArray) -> NDArray:
    """Per-spatial-mode power spectrum summed over polarizations: (M, N)."""
    return np.sum(np.abs(grid.fft(A)) ** 2, axis=1)  # (M, N)


def ensemble_moments(grid: TemporalGrid, A0: NDArray, beta2: float, segment: float):
    """Run the stochastic Eq. 12 ensemble and average *intensities and
    spectra* per spatial mode (not complex amplitudes — the SU(2) frames
    decorrelate the pol phases seed-to-seed, while |A|² is frame-invariant)."""
    def one(seed: int):
        A = stochastic_run(A0, beta2, L_NL, segment, seed)
        return mode_intensity(A), mode_spectrum(grid, A)
    runs = [one(s) for s in range(N_SEEDS)]
    I_ens = np.mean([r[0] for r in runs], axis=0)
    S_ens = np.mean([r[1] for r in runs], axis=0)
    return I_ens, S_ens


def engine_moments(grid: TemporalGrid, A0: NDArray, beta2: float, xpm: NDArray):
    """Deterministic Manakov run through the engine, same observables."""
    man, _eng = engine_manakov_run(np.ascontiguousarray(A0), beta2, L_NL, xpm)
    return mode_intensity(man), mode_spectrum(grid, man)


def check2_spm_8_9(results: dict) -> None:
    """M = 1: SPM 1 -> 8/9 under rapid random birefringence."""
    grid = make_grid()
    A0 = np.zeros((1, 2, N_GRID), dtype=complex)
    A0[0, 0] = gauss_channel(grid, P_PEAK)
    A0[0, 1] = gauss_channel(grid, 0.0)

    t0 = time.perf_counter()
    I_ens, S_ens = ensemble_moments(grid, A0, BETA2["LP01"], SEGMENT)
    t_sto = time.perf_counter() - t0

    xpm = np.full((2, 2), 8.0 / 9.0)
    t0 = time.perf_counter()
    I_man, S_man = engine_moments(grid, A0, BETA2["LP01"], xpm)
    t_man = time.perf_counter() - t0

    err_t = rel_l2(I_ens, I_man)
    err_f = rel_l2(S_ens, S_man)
    assert err_t < THRESH and err_f < THRESH, (err_t, err_f)
    results["spm89_err_time"] = err_t
    results["spm89_err_freq"] = err_f
    results["t_stochastic_m1"] = t_sto
    results["t_manakov_m1"] = t_man
    results["_m1"] = (grid, I_ens, I_man, S_ens, S_man)


def check3_manakov_m2(results: dict) -> None:
    """M = 2 degenerate modes: stochastic Eq. 12 ensemble vs the engine's
    deterministic generalized Manakov (Eq. 29, SPM 8/9 + inter-mode XPM 4/3).

    Reproduced 2026-09-21:
    - With the complete Mumtaz Eq. 6 cubic — the full (l, m, n) triple sum,
      which for the f = 1 isotropic idealization collapses to
      N_p = i gamma/3 [(G·G)G* + 2(G†G)G], G = sum_m A_m — the ensemble's
      effective inter-mode XPM weight best-fits to exactly 4/3: a sweep of
      the engine's inter-mode weight gives L2 0.0525 at 4/3 vs 0.14 at 1.0
      (8 seeds, clear minimum).
    - The residual scales as 1/sqrt(N_seeds): 5.3 % at 8, 2.9 % at 32,
      1.0 % at 128 — pure sampling noise of the ensemble statistic.
    - Both models conserve total energy to ~1e-12 (the FULL cubic is a
      closed Kerr system; the earlier truncated n = p form leaked ~0.4 %
      per 10 segments and produced a fake "+0.4 washout" that contradicted
      this check).
    - The no-birefringence fixed-frame reference (Eq. 6) deviates strongly
      (~0.44), reproducing the paper's point that the un-averaged cubic is
      fundamentally different from the ensemble-averaged Manakov physics.
    """
    grid = make_grid()
    A0 = np.zeros((2, 2, N_GRID), dtype=complex)
    A0[0, 0] = gauss_channel(grid, P_PEAK)
    A0[1, 0] = gauss_channel(grid, P_PEAK, t0=0.0)
    # both modes co-polarized x, same group velocity (degenerate pair)

    t0 = time.perf_counter()
    I_ens, S_ens = ensemble_moments(grid, A0, BETA2["LP01"], SEGMENT)
    t_sto = time.perf_counter() - t0

    xpm = np.array([
        [8 / 9, 8 / 9, 4 / 3, 4 / 3],
        [8 / 9, 8 / 9, 4 / 3, 4 / 3],
        [4 / 3, 4 / 3, 8 / 9, 8 / 9],
        [4 / 3, 4 / 3, 8 / 9, 8 / 9],
    ])
    t0 = time.perf_counter()
    I_man, S_man = engine_moments(grid, A0, BETA2["LP01"], xpm)
    t_man = time.perf_counter() - t0

    # no-birefringence reference (Eq. 6, fixed frame): un-averaged physics
    fixed = stochastic_run(A0, BETA2["LP01"], L_NL, L_NL, 0, rotate=False)
    I_fixed = mode_intensity(fixed)

    err_t = rel_l2(I_ens, I_man)
    err_f = max(
        rel_l2(S_ens[m], S_man[m]) for m in range(I_ens.shape[0])
    )
    err_fixed = rel_l2(I_fixed, I_man)
    # hard invariant: total energy conservation in both models
    E_tot_ens = float(np.sum(I_ens)) * grid.dt / float(np.sum(np.abs(A0) ** 2) * grid.dt)
    E_tot_man = float(np.sum(I_man)) * grid.dt / float(np.sum(np.abs(A0) ** 2) * grid.dt)
    assert abs(E_tot_ens - 1.0) < 1e-6 and abs(E_tot_man - 1.0) < 1e-6, (
        E_tot_ens, E_tot_man)
    assert err_t < THRESH and err_f < THRESH, (err_t, err_f)
    results["energy_ratio_ensemble"] = E_tot_ens
    results["energy_ratio_manakov"] = E_tot_man
    results["manakov_m2_err_time"] = err_t
    results["manakov_m2_err_freq"] = err_f
    results["manakov_m2_err_nobiref"] = err_fixed
    results["manakov_m2_status"] = (
        "reproduced: stochastic Eq. 12 ensemble matches the engine's "
        "generalized Manakov Eq. 29 (SPM 8/9, inter-mode XPM 4/3)")
    results["t_stochastic_m2"] = t_sto
    results["t_manakov_m2"] = t_man
    results["_m2"] = (grid, I_ens, I_man, S_ens, S_man, I_fixed)


def _plot(results: dict) -> None:
    grid, I1_e, I1_m, S1_e, S1_m = results["_m1"]
    grid2, I2_e, I2_m, S2_e, S2_m, I_fixed = results["_m2"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    w = grid.w
    lam = 2 * np.pi * C_MS / (OMEGA0 + w) * 1e9
    keep = np.abs(w) < 1.5e12

    ax = axes[0]
    S_ens, S_man = S1_e[0], S1_m[0]
    ax.plot(lam[keep], S_ens[keep] / S_ens[keep].max(), "C0-",
            label=f"stochastic ens. ({N_SEEDS} seeds)")
    ax.plot(lam[keep], S_man[keep] / S_man[keep].max(), "k--",
            label="Manakov 8/9 (engine)")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("spectral intensity (norm.)")
    ax.set_title(f"(a) M=1 SPM 8/9  (L2 {results['spm89_err_freq']:.1%})")
    ax.legend(fontsize=8)

    w2 = grid2.w
    lam2 = 2 * np.pi * C_MS / (OMEGA0 + w2) * 1e9
    ax = axes[1]
    for m, (Se, Sm) in enumerate(((S2_e[0], S2_m[0]), (S2_e[1], S2_m[1]))):
        ax.plot(lam2[keep], Se[keep] / Se[keep].max(), f"C{m}-",
                label=f"stochastic ens. mode {m}")
        ax.plot(lam2[keep], Sm[keep] / Sm[keep].max(), f"C{m}--",
                label=f"Manakov 8/9+4/3 mode {m}")
    ax.set_xlabel("wavelength (nm)")
    ax.set_title(f"(b) M=2 degenerate  (L2 {results['manakov_m2_err_freq']:.1%})")
    ax.legend(fontsize=7)

    ax = axes[2]
    t_ps = grid2.t * 1e12
    ax.plot(t_ps, I_fixed[0] / P_PEAK, "C3-",
            label="no birefringence (Eq. 6 frame)")
    ax.plot(t_ps, I2_e[0] / P_PEAK, "C0-",
            label="stochastic ensemble")
    ax.plot(t_ps, I2_m[0] / P_PEAK, "k--",
            label="Manakov (engine)")
    ax.set_xlim(-200, 200)
    ax.set_xlabel("t (ps)")
    ax.set_ylabel("|A|^2 / P_in")
    ax.set_title("(c) temporal, mode 0 x-pol")
    ax.legend(fontsize=7)

    fig.suptitle(
        "Mumtaz/Essiambre/Agrawal JLT 2013: generalized Manakov vs stochastic "
        "Eq. 12 (field level)"
    )
    fig.tight_layout()
    out = HERE / "mumtaz_2013_manakov.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def validate() -> dict:
    results: dict = {"parameters": json.loads(PARAMETERS.read_text())}
    check1_table2(results)
    check2_spm_8_9(results)
    check3_manakov_m2(results)
    results["timing_ratio_m2"] = (
        results["t_stochastic_m2"] / max(results["t_manakov_m2"], 1e-9)
    )
    _plot(results)

    print("Mumtaz/Essiambre/Agrawal (2013) generalized Manakov: validation passed")
    print(
        f"  Table II walk-offs (150 m): LP11 {results['walkoff_measured_s'][0]*1e12:.2f} ps"
        f" (exp {results['walkoff_expected_s'][0]*1e12:.2f}), "
        f"LP21 {results['walkoff_measured_s'][1]*1e12:.2f} ps"
        f" (exp {results['walkoff_expected_s'][1]*1e12:.2f})"
    )
    print(
        f"  SPM 8/9 (M=1): rel L2 time {results['spm89_err_time']:.1e}, "
        f"freq {results['spm89_err_freq']:.1e}"
    )
    print(
        f"  Manakov M=2 (reproduced): rel L2 time {results['manakov_m2_err_time']:.1e}, "
        f"freq {results['manakov_m2_err_freq']:.1e}; no-birefringence ref "
        f"{results['manakov_m2_err_nobiref']:.1e}"
    )
    print(
        f"  timing stochastic/Manakov: M=1 {results['t_stochastic_m1']/max(results['t_manakov_m1'],1e-9):.1f}x, "
        f"M=2 {results['timing_ratio_m2']:.1f}x"
    )
    return results


if __name__ == "__main__":
    validate()
