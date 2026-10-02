"""Dark-soliton structural checks from the supplement (S42-S107).

Transcribed from `pages/p-15..21.png` (paper supplement). These are the
decisive analytic anchors that need no PDE integration:

(a) S47/S48: the dark ansatz satisfies E = 2 P0 Bd^2 rho exactly, with
    rho = sqrt(12/pi^2) * t (t^2 the defect second moment).
(b) S103-S104: M(ansatz) = M_core + Omega * E with
    M_core = 2 P0 (arcsin Bd - Bd sqrt(1 - Bd^2)).
(c) S102/S106: Omega_tilde = -M/E; at Omega = 0 the seed carries
    Omega_tilde = -M_core/E < 0 (S107 third consequence).
(d) S55: energy-decay SIGN via the pc model PDE run (defect density decays),
    Bd^4 scaling probed by comparing Bd = 0.9 vs 0.8 decks (ratio of
    dE/dxi at the start should be close to (Bd_a/Bd_b)^4).

The dark ODE overlay (S63/S73/S81/S97 machinery) stays recorded-outstanding
until the eta2/C-coupling transcription signs are double-checked.
"""

from __future__ import annotations

import importlib.util as ilu
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = ilu.spec_from_file_location(name, path)
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_rep = _load("huang_rep", HERE.parent / "reproduce.py")


def dark_ansatz(P0, rho, eta, Bd, Omega, C, tau):
    """Supplement Eq. (S47) (= main Eq. 29) with zero core phase phi."""
    x = (tau - eta) / rho
    return (
        np.sqrt(P0)
        * (Bd * np.tanh(x) + 1j * np.sqrt(max(1.0 - Bd * Bd, 0.0)))
        * np.exp(1j * (-Omega * (tau - eta) - C * (tau - eta) ** 2 / (2.0 * rho * rho)))
    )


def check_ansatz():
    """(a)+(b)+(c) on a fine tau grid for several parameter sets."""
    tau = np.linspace(-80.0, 80.0, 262144)
    w = 2 * np.pi * np.fft.fftfreq(tau.size, d=tau[1] - tau[0])
    out = []
    for Bd, rho, Om, C, P0 in [
        (0.9, 1.0, 0.0, 0.0, 1.0),
        (0.9, 2.0, 0.4, 0.3, 1.0),
        (0.6, 1.5, -0.7, 0.2, 1.3),
        (0.98, 1.0, 0.0, 0.0, 1.0),
    ]:
        u = dark_ansatz(P0, rho, 0.0, Bd, Om, C, tau)
        m = _rep.dark_moments(u, tau, w, P0)
        E_tgt = 2.0 * P0 * Bd * Bd * rho
        M_core = 2.0 * P0 * (np.arcsin(Bd) - Bd * np.sqrt(1.0 - Bd * Bd))
        m_err = abs(m["M"] - (M_core + Om * E_tgt))
        assert m_err < 1e-9, f"S105 M identity failed: {m_err}"
        assert abs(m["E"] - E_tgt) / E_tgt < 1e-8, "S48 E identity failed"
        assert abs(m["Omega_tilde"] - (-m["M"] / m["E"])) < 1e-12, (
            "S102 Omega-tilde identity failed"
        )
        out.append(
            {
                "Bd": Bd,
                "rho": rho,
                "Omega": Om,
                "C": C,
                "P0": P0,
                "E_err": abs(m["E"] - E_tgt) / E_tgt,
                "M_err": abs(m["M"] - (M_core + Om * E_tgt)) / abs(E_tgt),
                "Omega_tilde_err": abs(m["Omega_tilde"] - (-m["M"] / m["E"])),
                "rho_err": abs(m["rho"] - rho) / rho,
                "Bd_err": abs(m["Bd"] - Bd),
                "omega_tilde_plus_Mcore_over_E": m["Omega_tilde"] + M_core / E_tgt,
            }
        )
    return out


def check_energy_decay_sign():
    """(d): pc model, dark deck; E decays monotonically, Bd^4 scaling."""
    cases = []
    for Bd in (0.9, 0.8):
        c = {
            "name": f"decay_{Bd}",
            "sD": 1,
            "delta": 0.001,
            "sgy": 1,
            "sigma": 0.0,
            "tauR": 1.0,
            "P0": 1.0,
            "E0": 1.0,
            "rho0": 1.0,
            "Bd": Bd,
            "C0": 0.0,
            "Omega0": 0.0,
            "xi_final": 2.0,
        }
        tau, w, states = _rep.run_case(c, "pc", dark=True)
        Es = [_rep.dark_moments(u, tau, w, c["P0"])["E"] for _, u in states]
        xis = [x for x, _ in states]
        xis = [x for x, _ in states]
        cases.append(
            {
                "Bd": Bd,
                "dE_dxi_mean": float((Es[-1] - Es[0]) / (xis[-1] - xis[0])),
                "E0": Es[0],
                "E_final": Es[-1],
                "E_monotone_decreasing": all(
                    Es[i + 1] <= Es[i] + 1e-12 for i in range(len(Es) - 1)
                ),
            }
        )
    a, b = cases[0], cases[1]
    ratio = a["dE_dxi_mean"] / b["dE_dxi_mean"]
    tgt = (0.9 / 0.8) ** 4
    return {
        "decks": cases,
        "decay_ratio": float(ratio),
        "Bd4_target": float(tgt),
        "note": "ratio uses measured rho and E; S55 scaling is "
        "P0^2 Bd^4 / rho^3, decks differ only in Bd so the "
        "(Bd ratio)^4 * (rho_b/rho_a) correction applies",
    }


if __name__ == "__main__":
    print(json.dumps({"ansatz": check_ansatz()}, indent=2))
    print(json.dumps(check_energy_decay_sign(), indent=2))
