"""Reproduction: quarter-wave dielectric mirror stopband (Macleod / Born & Wolf).

Reference
---------
H. A. Macleod, *Thin-Film Optical Filters* (4th ed.), Ch. 5; the quarter-wave
stack (HL)^N is the canonical DBR.  The first-order stopband full width is

    Delta_lambda / lambda0 = (4/pi) arcsin((nH - nL)/(nH + nL)),

and the peak reflectance at lambda0 has the exact characteristic-matrix form

    r = (eta0 (-etaL/etaH)^N - etas (-etaH/etaL)^N)
        / (eta0 (-etaL/etaH)^N + etas (-etaH/etaL)^N),   R = |r|^2,

with normal-incidence optical admittances eta = n.

What is reproduced
------------------
The reflectance spectrum R(lambda) of a quarter-wave DBR computed with the
library's TMM, checked against the closed forms above.  This exercises the
transfer-matrix path fixed in the `fix-review-findings` change (correct layer
ordering and complex-index handling).

Usage
-----
    python reproductions/macleod_quarter_wave_dbr/reproduce.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from photonics_helper.base import Length, Wavelength, WavelengthArray
from photonics_helper.dbr import TMM, Block, Material, Pattern

HERE = Path(__file__).resolve().parent
PARAMETERS = HERE / "parameters.json"


def _constant_material(name: str, n: float, wl_um: np.ndarray) -> Material:
    return Material(
        _name=name,
        n=np.full(len(wl_um), n),
        k=np.zeros(len(wl_um)),
        wl=WavelengthArray(wl_um, "um"),
    )


def _pattern(nH, nL, n_periods, lam0_m) -> Pattern:
    wl_um = np.linspace(0.4, 3.2, 101)
    dH = lam0_m / (4 * nH)
    dL = lam0_m / (4 * nL)
    mapping = {
        "H": Block(length=Length(dH, "m"), material=_constant_material("H", nH, wl_um)),
        "L": Block(length=Length(dL, "m"), material=_constant_material("L", nL, wl_um)),
    }
    return Pattern(
        style="HL" * n_periods,
        mapping=mapping,
        central_wavelength=Wavelength(lam0_m, "m"),
    )


def _peak_reflectance_closed_form(n0, ns, nH, nL, n_periods) -> float:
    r = (n0 * (-nL / nH) ** n_periods - ns * (-nH / nL) ** n_periods) / (
        n0 * (-nL / nH) ** n_periods + ns * (-nH / nL) ** n_periods
    )
    return float(abs(r) ** 2)


def validate(params: dict | None = None, make_plot: bool = True) -> dict:
    params = params or json.loads(PARAMETERS.read_text())
    nH, nL = params["n_high"], params["n_low"]
    n0, ns = params["n_incident"], params["n_substrate"]
    lam0 = params["central_wavelength_nm"] * 1e-9
    N = params["n_periods"]

    # 1) Peak reflectance vs the exact closed form for several period counts.
    peak_checks = []
    for n_per in params["small_n_periods"]:
        pat = _pattern(nH, nL, n_per, lam0)
        tmm = TMM(pattern=pat, angle_of_incidence=0.0, polarisation="TE")
        R, _ = tmm.spectrum(
            WavelengthArray(np.array([params["central_wavelength_nm"]]), "nm")
        )
        r_expected = _peak_reflectance_closed_form(n0, ns, nH, nL, n_per)
        peak_checks.append((n_per, float(R[0]), r_expected))
    max_peak_err = max(abs(r - e) for _, r, e in peak_checks)
    assert max_peak_err < params["reflectance_tolerance"], peak_checks

    # 2) Stopband spectrum and first-order width.
    pat = _pattern(nH, nL, N, lam0)
    tmm = TMM(pattern=pat, angle_of_incidence=0.0, polarisation="TE")
    wl = WavelengthArray(
        np.linspace(*params["wavelength_range_nm"], params["n_points"]), "nm"
    )
    R, T = tmm.spectrum(wl)
    R = np.asarray(R)
    lam = wl.as_nm
    # high-reflectance band (R > 0.5) containing lambda0
    high = R > 0.5
    i0 = int(np.argmin(np.abs(lam - params["central_wavelength_nm"])))
    assert high[i0], "lambda0 is not inside the stopband"
    lo = i0
    while lo > 0 and high[lo - 1]:
        lo -= 1
    hi = i0
    while hi < len(high) - 1 and high[hi + 1]:
        hi += 1
    measured_width = float(lam[hi] - lam[lo])
    frac = (4 / np.pi) * np.arcsin((nH - nL) / (nH + nL))
    analytic_width = float(frac * params["central_wavelength_nm"])
    width_err = abs(measured_width - analytic_width) / analytic_width
    assert width_err < params["stopband_width_tolerance"], (
        measured_width,
        analytic_width,
    )

    # 3) R(lambda0) saturates to unity for the large stack.
    peak = float(R[i0])
    assert peak > 0.999, peak

    if make_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(lam, R, label="R (TMM)")
        ax.plot(lam, T, label="T (TMM)")
        ax.axvline(
            params["central_wavelength_nm"],
            color="k",
            ls=":",
            lw=1,
            label=r"$\lambda_0$",
        )
        ax.axvspan(lam[lo], lam[hi], color="C0", alpha=0.1)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Reflectance / Transmittance")
        ax.set_title(f"Quarter-wave DBR stopband (nH={nH}, nL={nL}, N={N})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = HERE / "dbr_spectrum.png"
        fig.savefig(out, dpi=150)
        print(f"wrote {out}")

    print("DBR validation passed:")
    for n_per, r, e in peak_checks:
        print(f"  N={n_per}: R(lambda0)={r:.6f} (closed form {e:.6f})")
    print(f"  N={N}: R(lambda0)={peak:.6f}")
    print(
        f"  stopband width (R>0.5) = {measured_width:.1f} nm, "
        f"analytic = {analytic_width:.1f} nm (err {width_err * 100:.1f}%)"
    )
    return {
        "peak_checks": peak_checks,
        "max_peak_error": max_peak_err,
        "stopband_width_nm": measured_width,
        "analytic_width_nm": analytic_width,
        "width_rel_error": width_err,
        "R_lambda0": peak,
    }


if __name__ == "__main__":
    validate()
