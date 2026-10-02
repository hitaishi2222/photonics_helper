"""Task 7.3 — per-channel pump-sideband walk-off audit at detuning (ISSUES #8).

Deterministic, LINEAR probe of the engine's per-channel delay/walk-off arms
against Eq. (11) of Guasoni 2015:

    Delta_beta_n(+Om) = GVM_n*Om + b2_n*Om^2/2 + b3_n*Om^3/6     (paper)

The engine deck (reproduce.py::make_engine) supplies per-channel betas
(b2_n, b3_n) and group_delays = Table-I GVM (1/v_n - 1/v_1).  What the flat
Eq.-12 readout question hinges on is the SIGNS of these arms as the engine
combined them post-#0 (conjugated transform pair, +Dbeta1 walk-off).

Measurement (time domain, sub-0.5 % deterministic):
seed channel n with g(t)*e^{i Om t} (g: 20 ps gaussian), channel 0 with
g(t); propagate L; the centroid shift of |A_n(L)| relative to |A_0(L)|
gives the sideband-vs-reference walk-off tau(n, Om).  Modulating with an
analytic-sign tone never shifts a magnitude, so the readout is
kernel-sign-safe; the group-velocity arm (b2 dOm/dt) shifts the envelope,
the pure GVM arm shifts everything.

Analytic hypotheses (both recorded; engine measured value decides):
    A (GVM arm as GVM_n * Om as rate -> walk-off = GVM_n * L * (dOm/dwolka))
      we test the two standard readings:
      hyp-GVM:   tau = +GVM_n * L            (group-delay offset, no Om dependence)
      hyp-GVD:   tau = (b2_n - b2_0)*Om + (b3_n - b3_0)*Om^2/2) * L, SI
      hyp-SUM:   the paper's Eq.-(11) reading combined.

Outputs (incremental, crash-safe):
    diagnostics/walkoff_13thz_points.jsonl  - one line per (n, Om) measurement
    diagnostics/walkoff_13thz_summary.md    - rewritten at the end

Run from the repo root:
    python reproductions/guasoni_2015_generalized_mi_multimode/ \
        diagnostics/probe_walkoff_13thz.py
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
JSONL = HERE / "walkoff_13thz_points.jsonl"
SUM_MD = HERE / "walkoff_13thz_summary.md"

spec = importlib.util.spec_from_file_location("rep", str(HERE.parent / "reproduce.py"))
rep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rep)

from photonics_helper.base import Area, Length, Time, Wavelength
from photonics_helper.gnlse import FiberProfile
from photonics_helper.multimode_gnlse import MultimodeSplitStepEngine
from photonics_helper.pulse import Envelope, TemporalGrid, Wave

N = 8192
T_PS = 200.0  # dt = 24.4 fs -> Nyquist 20.5 THz: the +-13 THz tones unaliased
L_M = 0.5
NM = 4
LAMBDA0_NM = rep.LAMBDA0 * 1e9
CONV = 1e-6  # fs^3/mm (=1e-42 s^3/m) -> ps^3/m (=1e-36 s^3/m)


def gauss(tt_s, width_ps, center_ps):
    return np.exp(-(((tt_s - center_ps * 1e-12) / (width_ps * 1e-12)) ** 2))


def run_single(n: int, om_thz: float, with_gvm: bool) -> dict:
    grid = TemporalGrid(N=N, Tmax=Time(T_PS * 1e-12, "s"))
    tt = np.asarray(grid.t.as_s if hasattr(grid.t, "as_s") else grid.t)
    # envelope centered at t=0 (the window center); fringes at the
    # circular seam (+-T/2) are e^-100 — a seam-centered probe biased
    # every shift measurement by wrap spill (v1 audit artifact, fixed).
    g = gauss(tt, 20.0, 0.0)
    om = 2.0 * np.pi * om_thz * 1e12
    amp = 1e-3
    waves = []
    for m in range(NM):
        wv = Wave(
            grid=grid,
            envelope=Envelope(
                shape="gaussian", peak_amplitude=1.0, pulse_width=Time(1.0, "s")
            ),
            central_wavelength=Wavelength(LAMBDA0_NM, "nm"),
        )
        if m == 0:
            field = amp * g
        elif m == n:
            field = amp * g * np.exp(1j * om * tt)
        else:
            field = np.zeros(N, dtype=complex)
        waves.append(wv.with_field(np.asarray(field, complex)))
    b2_ps = list(rep.BETA2_S2_PER_M * 1e24)  # ps^2/m
    b3_ps = [float(v) * CONV for v in rep.BETA3_S3_PER_M]  # ps^3/m
    fiber = FiberProfile(
        n2=1.0, alpha=0.0, A_eff=Area(1.0, "m^2"), length=Length(L_M, "m")
    )
    kw = dict(group_delays=list(rep.GVM_S_PER_M)) if with_gvm else {}
    eng = MultimodeSplitStepEngine(
        waves,
        fiber,
        betas=[[b2_ps[m], b3_ps[m]] for m in range(NM)],
        betas_unit="ps^k/m",
        coef_model="isotropic",
        include_fwm=False,
        step_size=Length(1e-3, "m"),
        **kw,
    )
    eng.fiber.n2 = 1e-9  # linear audit: Kerr off
    eng.propagate(int(round(L_M / 1e-3)), nsaves=2)
    out0 = np.abs(np.asarray(eng.A[0], complex))
    outn = np.abs(np.asarray(eng.A[n], complex))
    # centroid shift of channel-n envelope relative to channel-0, ps
    c0 = float(np.sum(tt * out0) / np.sum(out0))
    cn = float(np.sum(tt * outn) / np.sum(outn))
    tau_meas = float((cn - c0) * 1e12)
    oms = om
    b2_n, b2_0 = rep.BETA2_S2_PER_M[n], rep.BETA2_S2_PER_M[0]
    b3_n, b3_0 = (rep.BETA3_S3_PER_M[n], rep.BETA3_S3_PER_M[0])  # SI
    ana_gvd = ((b2_n - b2_0) * oms + (b3_n - b3_0) * oms**2 / 2.0) * L_M
    ana_gvm = rep.GVM_S_PER_M[n] * L_M
    return {
        "type": "point",
        "n": n,
        "om_thz": om_thz,
        "gvm": with_gvm,
        "tau_meas_ps": round(tau_meas, 6),
        "tau_hyp_gvm_ps": round(ana_gvm * 1e12, 6),
        "tau_hyp_gvd_ps": round(ana_gvd * 1e12, 6),
        "tau_hyp_sum_ps": round((ana_gvm + ana_gvd) * 1e12, 6),
        "energy_drift_pct": round(
            100.0 * abs(eng.energy_vs_z[-1] / eng.energy_vs_z[0] - 1.0), 9
        ),
    }


def sweep() -> int:
    om_list = [0.0, -13.0, 13.0, -5.0, 5.0]
    jobs = [(n, om, g) for n in (1, 2, 3) for om in om_list for g in (True, False)]
    n_meas = 0
    with tqdm(jobs, desc="walk-off probes", file=sys.stdout) as bar:
        for n, om, g in bar:
            rec_start = {"type": "start", "n": n, "om_thz": om, "gvm": g}
            with open(JSONL, "a") as fh:
                fh.write(json.dumps(rec_start) + "\n")
                fh.flush()
            t0 = time.time()
            try:
                r = run_single(n, om, g)
                r["elapsed_s"] = round(time.time() - t0, 2)
                n_meas += 1
                bar.set_postfix(tau=r["tau_meas_ps"])
            except Exception as e:
                r = {"type": "error", "n": n, "om_thz": om, "gvm": g, "error": repr(e)}
            with open(JSONL, "a") as fh:
                fh.write(json.dumps(r, default=float) + "\n")
                fh.flush()
    return n_meas


def write_summary() -> None:
    pts = []
    for line in JSONL.read_text().splitlines():
        r = json.loads(line)
        if r.get("type") == "point" and "tau_meas_ps" in r:
            pts.append(r)
    out = [
        "# Task 7.3 — per-channel walk-off audit (probe_walkoff_13thz.py)",
        "",
        "tau_meas = centroid shift of channel n envelope vs channel 0",
        f"after L = {L_M} m (ps), linear-only deck.",
        "",
        "| n | Om (THz) | GVM arm | tau_meas (ps) | hyp GVM (ps) | "
        "hyp GVD (ps) | hyp SUM (ps) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(pts, key=lambda x: (x["n"], x["om_thz"], not x["gvm"])):
        out.append(
            f"| {r['n']} | {r['om_thz']} | {r['gvm']} | {r['tau_meas_ps']} | "
            f"{r['tau_hyp_gvm_ps']} | {r['tau_hyp_gvd_ps']} | "
            f"{r['tau_hyp_sum_ps']} |"
        )
    SUM_MD.write_text("\n".join(out) + "\n")


def main() -> None:
    with open(JSONL, "a") as fh:
        fh.write(
            json.dumps(
                {"type": "meta", "N": N, "T_ps": T_PS, "L_m": L_M, "ts": time.time()}
            )
            + "\n"
        )
    total = sweep()
    write_summary()
    tqdm.write(f"WALK-OFF AUDIT DONE: {total} points -> {JSONL.name} / {SUM_MD.name}")


if __name__ == "__main__":
    main()
