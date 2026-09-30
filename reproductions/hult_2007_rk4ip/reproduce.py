"""Hult 2007 (JLT 25, 3770) — RK4IP validation reproductions.

Two decks from the paper:

A. Second-order soliton (Sec. III.A): anomalous fiber beta2 = -0.01 ps^2/m,
   gamma = 0.01 W^-1 m^-1; N = 2 soliton, T_FWHM = 100 fs (T0 = 56.7 fs,
   P0 = 1.24 kW), one soliton period z_sol = 0.506 m, N = 2^12 points. The
   higher-order soliton recurrence returns the input envelope after one
   period, so the paper's average relative intensity error (Eq. 13) needs
   no closed form: A_true(z_sol) = A(0). We run the paper's coarse step
   ladder through the engine's RK4IP integrator and assert the paper's
   Fig. 1: fourth-order convergence (slope -4) to the accuracy floor
   (paper: epsilon <= 1e-10 at ~1000 steps; we assert slope -4 and that
   the ladder reaches <= 1e-7).

B. SCG in the Table-I PCF (Sec. III.B; the Dudley & Coen scenario at
   850 nm): P0 = 10 kW sech, T0 = 28.4 fs, gamma = 0.045 W^-1 m^-1,
   beta2..beta7 per Table I, L = 0.1 m, fR = 0.18, shock on, N = 2^13
   points, soliton order ~ 5. Asserts the paper's Fig. 2 morphology
   (fission, Raman red-shift of the ejected solitons, blue-side dispersive
   wave) plus a coarse-to-fine convergence ladder against a fine
   reference (fourth-order slope; the paper's Fig. 3 reaches ~1e-8 at
   ~3e4 steps — our ladder is a subset of that span).

Recorded deviation: the paper uses the Hollenbeck-Cantrell modal Raman
response; the engine ships the standard tau1/tau2 two-exponential silica
response (the repository-wide dudley_2006_scg convention) — both soliton
decks are otherwise the paper's operator set exactly.

Run from the repo root:  python reproductions/hult_2007_rk4ip/reproduce.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PARAMS = json.loads((HERE / "parameters.json").read_text())
OUT_PNG = HERE / "hult_2007_rk4ip.png"

# -- deck A: second-order soliton convergence ladder --------------------------


def _build_soliton(n_points: int, z_len_m: float):
    import warnings
    warnings.filterwarnings("ignore")
    from photonics_helper.base import Length, Time, Wavelength
    from photonics_helper.gnlse import FiberProfile, GNLSESolver
    from photonics_helper.pulse import Envelope, TemporalGrid, Wave

    d = PARAMS["soliton_deck"]
    grid = TemporalGrid(N=n_points, Tmax=Time(2e-12, "s"))
    wave = Wave(
        grid=grid,
        envelope=Envelope(
            shape="sech",
            peak_amplitude=float(np.sqrt(d["P0_W"])),
            pulse_width=Time(d["T0_fs"], "fs"),
        ),
        central_wavelength=Wavelength(850.0, "nm"),
    )
    fiber = FiberProfile.from_gamma(
        gamma=d["gamma_per_Wm"],
        n2=2.5e-20,
        omega0=float(wave.central_frequency),
        alpha=0.0,
        length=Length(z_len_m, "m"),
        raman_response=None,
    )
    return GNLSESolver(
        pulse=wave,
        fiber=fiber,
        betas=np.array([d["beta2_ps2_per_m"]]),
        include_raman=False,
        include_self_steepening=False,
        include_tpa=False,
    )


def average_relative_intensity_error(A_comp: np.ndarray,
                                     A_true: np.ndarray) -> float:
    """Paper Eq. (13)."""
    Ic, It = np.abs(A_comp) ** 2, np.abs(A_true) ** 2
    return float(np.mean(np.abs(Ic - It)) / np.max(It))


def soliton_ladder() -> dict:
    d = PARAMS["soliton_deck"]
    z_sol = d["z_sol_m"]
    A_true = None
    rows = []
    for n_steps in (40, 80, 160, 320, 640, 1280):
        solver = _build_soliton(d["n_points"], z_sol)
        solver.propagate(num_steps=n_steps, nsaves=2)
        A0 = np.asarray(solver.evolution[0].envelope_field, dtype=complex)
        A1 = np.asarray(solver.evolution[-1].envelope_field, dtype=complex)
        if A_true is None:
            A_true = A0        # exact recurrence: A(z_sol) = A(0)
        eps = average_relative_intensity_error(A1, A_true)
        rows.append({"n_steps": n_steps, "epsilon": eps})
    xs = np.log10(np.array([r["n_steps"] for r in rows], dtype=float))
    ys = np.log10(np.array([r["epsilon"] for r in rows], dtype=float))
    slope, _intercept = np.polyfit(xs, ys, 1)
    T0_s = d["T0_fs"] * 1e-15
    L_D = T0_s ** 2 / abs(d["beta2_ps2_per_m"]) * 1e24
    L_NL = 1.0 / (d["gamma_per_Wm"] * d["P0_W"])
    return {
        "rows": rows,
        "slope": float(slope),
        "best_epsilon": rows[-1]["epsilon"],
        "z_sol_m": z_sol,
        "L_D_m": L_D,
        "N_order": float(np.sqrt(L_D / L_NL)),
    }


# -- deck B: SCG (Table I) ------------------------------------------------------

_cache: dict = {}


def _run_scg(n_steps: int, n_points: int = 8192, key: str = ""):
    import warnings
    warnings.filterwarnings("ignore")
    from photonics_helper.base import Length, Time, Wavelength
    from photonics_helper.gnlse import FiberProfile, GNLSESolver
    from photonics_helper.pulse import Envelope, TemporalGrid, Wave
    from photonics_helper.raman import RamanResponse, RamanSpec

    ck = (n_steps, n_points, key)
    if ck in _cache:
        return _cache[ck]
    d = PARAMS["scg_deck"]
    grid = TemporalGrid(N=n_points, Tmax=Time(4e-12, "s"))
    wave = Wave(
        grid=grid,
        envelope=Envelope.from_fwhm(
            "sech",
            peak_amplitude=float(np.sqrt(d["P0_W"])),
            fwhm=Time(1.763 * d["T0_fs"], "fs"),
        ),
        central_wavelength=Wavelength(d["central_wavelength_nm"], "nm"),
    )
    spec = RamanSpec(name="Silica", raman_shift_cm=440.0,
                     raman_linewidth_cm=45.0, fR=d["fR"])
    resp = RamanResponse(spec=spec, fR=d["fR"], tau1=12.2e-15,
                         tau2=32e-15, grid=grid)
    fiber = FiberProfile.from_gamma(
        gamma=d["gamma_per_Wm"],
        n2=2.5e-20,
        omega0=float(wave.central_frequency),
        alpha=0.0,
        length=Length(d["fiber_length_m"], "m"),
        raman_response=resp,
    )
    solver = GNLSESolver(
        pulse=wave,
        fiber=fiber,
        betas=np.asarray(d["betas_psN_per_m"], dtype=float),
        include_raman=True,
        include_self_steepening=True,
        include_tpa=False,
    )
    solver.propagate(num_steps=n_steps, nsaves=2, show_progress=False)
    out = (np.asarray(solver.evolution[0].envelope_field, dtype=complex),
           np.asarray(solver.evolution[-1].envelope_field, dtype=complex),
           grid)
    _cache[ck] = out
    return out


def _engine_band_spectrum(u: np.ndarray, grid) -> tuple[np.ndarray, np.ndarray]:
    """|A~|^2 sorted by wavelength, with the engine-convention kernel
    (analysis e^{+i}; raw np.fft mirrors complex wideband fields — the
    ISSUES.md #0 resolution addendum)."""
    spec = np.fft.fftshift(np.conj(np.fft.fft(np.conj(np.fft.ifftshift(u)))))
    psd = np.abs(spec) ** 2
    w0 = 2 * np.pi * 299792458.0 / (PARAMS["scg_deck"]["central_wavelength_nm"]
                                    * 1e-9)
    w = np.fft.fftshift(np.fft.fftfreq(u.size, d=float(grid.dt))) * 2 * np.pi
    lam = 2 * np.pi * 299792458.0 / (w0 + w) * 1e9
    order = np.argsort(lam)
    return lam[order], psd[order]


def strongest_peak_in_band(lam: np.ndarray, psd: np.ndarray,
                           lo: float, hi: float) -> tuple[float, float]:
    from scipy.signal import find_peaks
    m = (lam >= lo) & (lam <= hi)
    band_lam, band_v = lam[m], psd[m]
    if band_v.size == 0:
        return float("nan"), 0.0
    pks, _ = find_peaks(band_v, height=band_v.max() * 0.02,
                        distance=max(1, 4 * (band_v.size // 500)))
    idx = int(pks[np.argmax(band_v[pks])]) if pks.size else \
        int(np.argmax(band_v))
    return float(band_lam[idx]), float(band_v[idx])


def soliton_scales_scg() -> dict:
    d = PARAMS["scg_deck"]
    T0_s = d["T0_fs"] * 1e-15
    b2_si = abs(d["betas_psN_per_m"][0]) * 1e-24
    L_D = T0_s ** 2 / b2_si
    L_NL = 1.0 / (d["gamma_per_Wm"] * d["P0_W"])
    return {"L_D_cm": L_D * 100, "L_NL_mm": L_NL * 1e3,
            "N": float(np.sqrt(L_D / L_NL)),
            "z_sol_cm": 0.5 * np.pi * L_D * 100}


def scg_physics() -> dict:
    """Table-I SCG deck anchors (paper Fig. 2)."""
    d = PARAMS["scg_deck"]
    A0, A1, grid = _run_scg(2048, key="c2048")
    lam, psd = _engine_band_spectrum(A1, grid)
    lam0, psd0 = _engine_band_spectrum(A0, grid)
    peak = psd.max()
    # input spectrum -20 dB upper edge (the red Raman-shifted solitons live
    # beyond it by construction once Raman acts)
    mask0 = psd0 >= psd0.max() * 0.01
    red_input = float(lam0[mask0].max())
    # output: strongest red-shifted soliton peak and blue-side DW
    sol_lam, sol_v = strongest_peak_in_band(lam, psd, red_input, 1400.0)
    dw_band_floor = 0.95 * d["central_wavelength_nm"]
    dw_lam, dw_v = strongest_peak_in_band(lam, psd, 400.0, dw_band_floor)
    # temporal fission: intensity peaks above 2 % of max
    from scipy.signal import find_peaks
    I = np.abs(A1) ** 2
    pks, _ = find_peaks(I, height=I.max() * 0.02, distance=8)
    # -20 dB spectral span (paper Fig. 2a: ~480..1450 nm at 10 cm)
    mask20 = psd >= peak * 0.01
    return {
        "input_red_edge_nm": red_input,
        "raman_soliton_peak_nm": sol_lam,
        "raman_redshift_nm": sol_lam - red_input,
        "dw_side_nm": dw_lam,
        "dw_side_power_rel": float(dw_v / peak),
        "n_temporal_peaks": int(pks.size),
        "span_minus20dB_nm": (float(lam[mask20].min()),
                              float(lam[mask20].max())),
        "span_ratio": float(lam[mask20].max() / lam[mask20].min()),
    }


def scg_ladder() -> dict:
    """Coarse ladder vs fine reference (paper Fig. 3, subset span)."""
    rows = []
    for n_steps in (256, 512, 1024, 2048):
        _, _, _ = _run_scg(n_steps, key=f"c{n_steps}")
    _, _, _ = _run_scg(10240, key="ref")
    ref_A1 = _cache[(10240, 8192, "ref")][1]
    for n_steps in (256, 512, 1024, 2048):
        A1 = _cache[(n_steps, 8192, f"c{n_steps}")][1]
        eps = average_relative_intensity_error(A1, ref_A1)
        rows.append({"n_steps": n_steps, "epsilon_ref": eps})
    xs = np.log10(np.array([r["n_steps"] for r in rows], dtype=float))
    ys = np.log10(np.array([r["epsilon_ref"] for r in rows], dtype=float))
    slope, _ = np.polyfit(xs, ys, 1)
    return {"rows": rows, "slope": float(slope),
            "best_epsilon_ref": rows[-1]["epsilon_ref"]}


def validate() -> dict:
    results: dict = {}
    tol = PARAMS["accept_tolerances"]
    scales = soliton_scales_scg()
    results["scg_scales"] = scales
    assert abs(scales["N"] - 5.0) / 5.0 < 0.25, scales

    # --- deck A ---
    A = soliton_ladder()
    results["soliton"] = A
    assert A["slope"] < 0 and abs(A["slope"] - (-4.0)) < tol["soliton_slope_abs"], \
        (A["slope"], "RK4IP must reproduce the paper's 4th-order convergence")
    assert A["best_epsilon"] < tol["soliton_best_epsilon"], A["best_epsilon"]
    assert abs(A["z_sol_m"] - 0.506) / 0.506 < 0.01
    assert abs(A["N_order"] - 2.0) < 1e-9

    # --- deck B physics ---
    B = scg_physics()
    results["scg_physics"] = B
    assert B["n_temporal_peaks"] >= 2, B["n_temporal_peaks"]
    assert B["raman_redshift_nm"] > 30.0, B["raman_redshift_nm"]
    d = PARAMS["scg_deck"]
    assert B["dw_side_nm"] < 0.95 * d["central_wavelength_nm"], \
        (B["dw_side_nm"], "blue DW must sit short of the pump")
    assert B["dw_side_power_rel"] > 0.01, B["dw_side_power_rel"]
    assert B["span_ratio"] > 1.2, B["span_ratio"]

    # --- deck B convergence ---
    C = scg_ladder()
    results["scg_convergence"] = C
    assert abs(C["slope"] - (-4.0)) < tol["scg_slope_abs"], C["slope"]
    assert C["best_epsilon_ref"] < tol["scg_best_epsilon"], C["best_epsilon_ref"]
    return results


def make_fig(results) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    A = results["soliton"]
    xs = [r["n_steps"] for r in A["rows"]]
    ys = [r["epsilon"] for r in A["rows"]]
    axes[0].loglog(xs, ys, "ko-", label="engine RK4IP")
    ord4 = ys[-1] * np.array([(n / xs[-1]) ** (-4.0) for n in xs])
    axes[0].loglog(xs, ord4, "k--", lw=0.8, label="slope -4")
    axes[0].set_xlabel("steps"); axes[0].set_ylabel("avg rel intensity error")
    axes[0].set_title("soliton ladder (paper Fig. 1)")
    axes[0].legend(fontsize=7)
    C = results["scg_convergence"]
    xs2 = [r["n_steps"] for r in C["rows"]]
    ys2 = [r["epsilon_ref"] for r in C["rows"]]
    axes[1].loglog(xs2, ys2, "ko-", label="vs fine ref")
    ord4b = ys2[-1] * np.array([(n / xs2[-1]) ** (-4.0) for n in xs2])
    axes[1].loglog(xs2, ord4b, "k--", lw=0.8, label="slope -4")
    axes[1].set_xlabel("steps"); axes[1].set_title("SCG ladder (paper Fig. 3)")
    axes[1].legend(fontsize=7)
    # spectra panel
    A0, A1, grid = _run_scg(2048, key="c2048")
    lam0, psd0 = _engine_band_spectrum(A0, grid)
    lam, psd = _engine_band_spectrum(_cache[(2048, 8192, "c2048")][1], grid)
    axes[2].plot(lam0, 10 * np.log10(psd0 / psd0.max()), "k-", lw=0.8,
                 label="input")
    axes[2].plot(lam, 10 * np.log10(psd / psd.max()), "r-", lw=0.8,
                 label="10 cm (paper Fig. 2a)")
    axes[2].set_xlim(350, 1500); axes[2].set_ylim(-45, 2)
    axes[2].set_xlabel("wavelength (nm)"); axes[2].set_ylabel("dB")
    axes[2].legend(fontsize=7)
    fig.suptitle("Hult 2007 RK4IP — engine reproduction")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    out = validate()
    out["soliton"].pop("rows", None)
    out["scg_convergence"].pop("rows", None)
    print(json.dumps(out, indent=1, default=float))
    make_fig(out)
    (HERE / "validation_results.json").write_text(
        json.dumps(out, indent=1, default=float))
    print("VALIDATION OK")
