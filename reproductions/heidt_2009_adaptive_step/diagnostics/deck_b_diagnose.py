"""Deck-B diagnosis (task-2): two fundamental solitons, +/-400 GHz about the
band centre, 100 ps apart, 400 km, RK4IP dz = 25 m.

Recorded symptom: the propagated two-soliton field disperses into a low
smooth pedestal by 40 km. Probes:
  1. one detuned soliton alone at +400 GHz and at -400 GHz — a fundamental
     soliton must keep its shape (the carrier detuning only changes its
     group velocity through the beta1 term the engine drops... note: the
     operator has NO beta1 term, so a detuned soliton also slides in
     retarded time at beta2*detuning*z — that drift itself may wrap the
     pulse toward the window edge).
  2. both together, field snapshots at 0/50/100/200/400 km with peak
     positions, to see whether the "pedestal" is actually the two pulses
     walking apart/wrapping rather than a numerical failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))

from heidt_adaptive import RK4IPIntegrator  # noqa: E402
from reproduce import PARAMS, deck_b_setup  # noqa: E402


def run(op, A0, length_m, dz, probe_every=None):
    integ = RK4IPIntegrator(op)
    A = A0.copy()
    n = int(round(length_m / dz))
    probes = {}
    for i in range(n):
        if probe_every is not None and (i * dz) in probe_every:
            probes[f"{i * dz:g}"] = describe(op, A)
        A = integ.step(A, dz)
    if probe_every is not None:
        probes["end"] = describe(op, A)
    return A, probes


def describe(op, A):
    t = np.asarray(op.grid.t)
    P = np.abs(A) ** 2
    if P.max() < 1e-30:
        return {"peak": 0.0}
    # split-window peak finding: local maxima above 10 % of max
    from scipy.signal import find_peaks

    pk, _ = find_peaks(P, height=0.1 * P.max(), distance=8)
    return {
        "peak": float(P.max()),
        "energy": float(np.sum(P)) * float(op.grid.dt),
        "peaks_t_ps": [round(float(t[i]) * 1e12, 1) for i in pk[:6]],
        "peaks_rel": [round(float(P[i] / P.max()), 3) for i in pk[:6]],
        "edge_frac": float((P[:64].sum() + P[-64:].sum()) / P.sum()),
    }


def main() -> None:
    dz = 25.0
    marks = {k * 1e3: None for k in (40, 100, 200, 300, 400)}
    out = {}
    # single detuned solitons
    for sign in (+1, -1):
        op, _ = deck_b_setup(False)
        d = PARAMS["deck_b_soliton_collision"]
        p = d["pulses"]
        T0 = float(p["T0_s"])
        t = np.asarray(op.grid.t)
        det = sign * 0.5 * float(p["detuning_Hz"])
        a = (np.sqrt(float(p["P0_W"])) / np.cosh(t / T0)).astype(complex)
        a = a * np.exp(-1j * 2 * np.pi * det * t)
        A, probes = run(op, a, 4e5, dz, probe_every=marks)
        out[f"single_{sign:+d}"] = probes
        print(f"single {sign:+d}: end", json.dumps(probes["end"]))
    # both together
    op, A0 = deck_b_setup(False)
    A, probes = run(op, A0, 4e5, dz, probe_every=marks)
    out["pair"] = probes
    for k, v in probes.items():
        print("pair", k, json.dumps(v))
    (HERE / "diagnostics" / "deck_b_diagnose.json").write_text(
        json.dumps(out, indent=2)
    )


if __name__ == "__main__":
    main()
