"""Reproduction: SHG in x-cut LNOI (Wang et al., Opt. Express 25(6), 6963).

Modal overlap coupling ``g`` (uniform waveguide) and periodically-grooved LN
``g'`` (PGLN), evaluated with the :mod:`photonics_helper.chi2` mode-overlap
APIs (``shg_coupling_overlap`` / ``pgln_overlap``).

``_n_e`` below is a placeholder interpolation between the article-validated
LiNbO₃ extraordinary-index anchors; the openspec change
``fix-shg-replication-gaps`` (Group 4) seeds the real birefringent Sellmeier
curve in ``materials.db`` and this module then reads it via
``RefractiveIndex.from_material_database("LiNbO3", axis="extraordinary")``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from photonics_helper.base import Area, Wavelength
from photonics_helper.chi2 import pgln_overlap, shg_coupling, shg_coupling_overlap

RESULTS = Path(__file__).parent / "results.json"
WL_PUMP = Wavelength(1550, "nm")
WL_SH = Wavelength(775, "nm")
D33 = 27e-12  # LiNbO₃ d33 (m/V), x-cut, e-axis (E_z dominant)

# article anchors (Wang et al. §2/§3)
PAPER_G = 77.4   # 0.774 W^-1/2 cm⁻¹ → 77.4 1/(m·√W)
PAPER_GP = 34.5  # 0.345 W^-1/2 cm⁻¹ → 34.5 1/(m·√W)
QPM_LAMBDA = 2.77e-6

DX = 4e-9
NX = 1000  # ±2 µm window

_N_E_ANCHORS = ((0.775, 2.187), (1.55, 2.139))  # Edwards & Lawrence, n_e


def _n_e(wl_um: float) -> float:
    """Extraordinary index of x-cut LiNbO₃ near the working band (see note)."""
    return float(
        np.interp(
            wl_um,
            [a[0] for a in _N_E_ANCHORS],
            [a[1] for a in _N_E_ANCHORS],
        )
    )


def main() -> dict:
    """Compute both overlaps for the article geometry and dump results.json."""
    coords = np.linspace(-2e-6, 2e-6, NX)
    X, Z = np.meshgrid(coords, coords, indexing="ij")

    # wt / wb / t from the article; wb is determined by the etch wall angle
    wt, _wb, t = 600e-9, 1270e-9, 400e-9
    theta = np.deg2rad(40.0)
    in_ln = (np.abs(X) <= (wt / 2 + np.maximum(-Z, 0.0) * np.tan(theta))) & (
        (Z >= -t) & (Z <= 0)
    )

    ne_p = _n_e(1.55)
    ne_s = _n_e(0.775)
    n_clad = 1.44

    # Scalar (e-component) toy modes standing in for the FEM fields; the FEM
    # anchor is exercised in the shg_solve study (~/Research/sim/shg_solve).
    w_p = 260e-9
    pump = np.exp(-(X**2 / w_p**2 + (Z + 120e-9) ** 2 / w_p**2))
    sh3 = (
        np.cos(3.0 * np.pi * X / 0.4e-6) * np.exp(-(X**2) / (210e-9) ** 2)
    ) * np.exp(-((Z + 110e-9) ** 2) / (200e-9) ** 2)

    g = shg_coupling_overlap(
        pump, sh3, DX, DX,
        wavelength=WL_PUMP, d=D33, n_pump=ne_p, n_sh=ne_s,
    )

    # PGLN (50 % duty grooves, 80 nm depth): d^(1) & Δε₁ first harmonic
    strip = in_ln & (np.abs(X) > (wt - 80e-9) / 2)
    d1 = np.where(strip, D33 / np.pi, 0.0)
    de1_p = np.where(strip, (ne_p**2 - n_clad**2) / np.pi, 0.0)
    de1_s = np.where(strip, (ne_s**2 - n_clad**2) / np.pi, 0.0)
    delta_k = 2 * np.pi / QPM_LAMBDA  # article-anchored residual mismatch

    pg = pgln_overlap(
        pump, sh3, DX, DX,
        wavelength=WL_PUMP,
        d0=D33,
        d1=d1,
        delta_eps1_pump=de1_p,
        delta_eps1_sh=de1_s,
        delta_k=delta_k,
        n_pump=ne_p,
        n_sh=ne_s,
    )

    sigma_plane = shg_coupling(WL_PUMP, D33, n=ne_p, A_eff=Area(0.52, "um^2"))
    out = {
        "sigma_plane_m": float(sigma_plane),
        "g_uniform_m": float(g),
        "ratio_g_to_paper": float(g / PAPER_G),
        "g_eff_m": float(pg["g_eff"]),
        "ratio_gprime_to_paper": float(pg["g_eff"] / PAPER_GP),
        "phi": float(pg["phi"]),
        "g_nl_0_m": float(pg["g_nl_0"]),
        "g_L_w": float(pg["g_L_w"]),
        "g_L_2w": float(pg["g_L_2w"]),
    }
    RESULTS.write_text(json.dumps(out, indent=2))
    return out


def validate() -> None:
    """Test hook — math sanity only; quantitative FEM anchors live in the
    standalone shg_solve study, not in this lightweight reproduction."""
    out = main()
    assert np.isfinite(out["g_uniform_m"]) and out["g_uniform_m"] > 0
    assert np.isfinite(out["g_eff_m"])
    assert abs(out["phi"]) > 0


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
