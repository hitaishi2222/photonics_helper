"""Reproduction: textbook SHG conversion efficiency and QPM recovery.

Reference
---------
R. W. Boyd, *Nonlinear Optics* (3rd ed.), Ch. 2. The coupled-wave equations
for second-harmonic generation with pump depletion have the exact
perfect-phase-matching solution

    eta(L) = |A_sh(L)|^2 / |A_f(0)|^2 = tanh^2(kappa L),
    kappa = sigma sqrt(P0),

where ``sigma`` is the effective coupling and ``P0`` the input fundamental
power.  First-order quasi-phase-matching (Fejer et al., IEEE JQE 28, 2631
(1992)) recovers the same curve for a mismatched interaction when the
square-wave poling period is ``Lambda = 2 pi / |Delta k|``; the 50 % duty-cycle
grating reduces the effective coupling by ``2/pi``, i.e. ``eta =
tanh^2((2/pi) kappa L)``.

What is reproduced
------------------
The library's ``photonics_helper.chi2`` RK4IP solver against both closed
forms above for a 1550 nm waveguide example, and the QPM recovery of a
deliberately mismatched interaction.

Usage
-----
    python reproductions/shg_textbook/reproduce.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from photonics_helper.base import Area, Wavelength
from photonics_helper.chi2 import Lambda_qpm, shg_coupling, solve_shg

HERE = Path(__file__).resolve().parent
PARAMETERS = HERE / "parameters.json"


def validate(params: dict | None = None, make_plot: bool = True) -> dict:
    params = params or json.loads(PARAMETERS.read_text())
    wl = Wavelength(params["wavelength_nm"], "nm")
    A_eff = Area(params["A_eff_um2"], "um^2")
    d_eff = params["d_eff_pm_per_V"] * 1e-12
    n = params["n"]
    P0 = params["P0_mW"] * 1e-3
    tol = params["efficiency_tolerance"]
    n_steps = params["n_steps"]

    sigma = shg_coupling(wl, d_eff, n=n, A_eff=A_eff)
    kappa = sigma * np.sqrt(P0)

    # 1) eta = tanh^2(kappa L) at a set of target kappa L values.
    checks = []
    for kL in params["kappaL_values"]:
        length = kL / kappa
        result = solve_shg(length=length, P0=P0, sigma=sigma, n_steps=n_steps)
        eta = float(result.efficiency()[-1])
        expected = float(np.tanh(kL) ** 2)
        rel_err = abs(eta - expected) / expected
        checks.append((float(kL), eta, expected, rel_err))
    max_err = max(c[3] for c in checks)
    assert max_err < tol, checks

    # 2) QPM recovery of a mismatched interaction.
    qpm = params["qpm"]
    period = qpm["period_um"] * 1e-6
    delta_k = 2.0 * np.pi / period
    assert abs(Lambda_qpm(delta_k) - period) / period < 1e-12

    L_qpm = qpm["kappaL"] / kappa
    off = solve_shg(
        length=L_qpm, P0=P0, sigma=sigma, n_steps=n_steps, delta_k=delta_k
    )
    on = solve_shg(
        length=L_qpm,
        P0=P0,
        sigma=sigma,
        n_steps=n_steps,
        delta_k=delta_k,
        qpm_period=period,
    )
    eta_off = float(off.efficiency()[-1])
    eta_on = float(on.efficiency()[-1])
    eta_qpm_expected = float(np.tanh(2.0 / np.pi * kappa * L_qpm) ** 2)
    qpm_rel_err = abs(eta_on - eta_qpm_expected) / eta_qpm_expected
    assert eta_off < qpm["off_efficiency_max"], eta_off
    assert qpm_rel_err < tol, (eta_on, eta_qpm_expected)

    if make_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Continuous curves over one solve each.
        L_max = 2.5 / kappa
        curve = solve_shg(length=L_max, P0=P0, sigma=sigma, n_steps=6000)
        qpm_curve = solve_shg(
            length=L_qpm,
            P0=P0,
            sigma=sigma,
            n_steps=8000,
            delta_k=delta_k,
            qpm_period=period,
        )
        kL = kappa * curve.z
        kL_qpm = kappa * qpm_curve.z

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

        ax1.plot(kL, curve.efficiency(), "C0-", lw=2, label="solver (RK4IP)")
        ax1.plot(
            kL,
            np.tanh(kL) ** 2,
            "k--",
            lw=1.2,
            label=r"$\tanh^2(\kappa L)$",
        )
        ax1.set_xlabel(r"$\kappa L$")
        ax1.set_ylabel(r"SHG efficiency $\eta$")
        ax1.set_title(r"Perfect phase matching ($\Delta k = 0$)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(
            kL_qpm,
            qpm_curve.efficiency(),
            "C2-",
            lw=2,
            label=r"QPM, $\Lambda = 2\pi/|\Delta k|$",
        )
        ax2.plot(
            kL_qpm,
            np.tanh(2.0 / np.pi * kL_qpm) ** 2,
            "k--",
            lw=1.2,
            label=r"$\tanh^2((2/\pi)\kappa L)$",
        )
        ax2.axhline(
            eta_off,
            color="C3",
            ls=":",
            lw=1.2,
            label=rf"no QPM ($\eta={eta_off:.1e}$)",
        )
        ax2.set_xlabel(r"$\kappa L$")
        ax2.set_ylabel(r"SHG efficiency $\eta$")
        ax2.set_title(r"Quasi-phase-matching recovery ($\Delta k \neq 0$)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        out = HERE / "shg_efficiency.png"
        fig.savefig(out, dpi=150)
        print(f"wrote {out}")

    print("SHG textbook validation passed:")
    for kL, eta, expected, rel_err in checks:
        print(
            f"  kappa L={kL:.2f}: eta={eta:.6f} "
            f"(tanh^2 {expected:.6f}, err {rel_err:.2e})"
        )
    print(
        f"  QPM: eta_off={eta_off:.3e}, eta_on={eta_on:.6f} "
        f"(tanh^2((2/pi)kL) {eta_qpm_expected:.6f}, err {qpm_rel_err:.2e})"
    )
    return {
        "sigma": sigma,
        "kappa": kappa,
        "checks": checks,
        "max_rel_error": max_err,
        "eta_off_qpm": eta_off,
        "eta_on_qpm": eta_on,
        "eta_qpm_analytic": eta_qpm_expected,
        "qpm_rel_error": qpm_rel_err,
    }


if __name__ == "__main__":
    validate()
