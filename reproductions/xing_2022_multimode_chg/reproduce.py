"""Reproduction: Xing 2022 As2Se3/As2S3 multimode chalcogenide fibre SCG.

Reference
---------
D.-K. Xing, J.-Y. He, P. Wang, H.-Y. Guo, K. Chang, C.-C. Liu, Y.-G. Liu,
Z. Wang, "Intermodal dispersive waves and soliton collision during multimode
supercontinuum generation in chalcogenide glass fiber", *Laser Phys.* **32**,
115401 (2022), doi:10.1088/1555-6611/ac979a.  Not open access; the inputs in
``parameters.json`` were read from the IOP full text (see the folder README).

What is reproduced
------------------
The paper's **inputs**, re-solved from scratch, so its **outputs** can be
checked.  This deck is deliberately a reproduction of the *dispersion* layer
(material + modes + the intermodal phase-matching condition of Eq. 7); it does
**not** run the 6-mode, 50 mm, 1 kW MM-GN LSE propagation, which is out of scope
here (see the README's "scope" note).

CHECKS (assert-carrying)
------------------------
0. **Eq. (5)** (As2S3) reproduces the repository's validated As2S3 row, and
   **Eq. (6)** (As2Se3) *as printed* is broken -- the middle ``B`` coefficient
   lost its square and reads ``19`` (pole at sqrt(19) = 4.359 um); restored to
   ``19^2 = 361`` it reproduces the repository's validated As2Se3 row exactly
   (n(4 um) = 2.783, material ZDW = 7.22 um).
1. **Modes.**  The exact vector solver guides HE11, TE01, TM01 and HE21 at the
   paper's 6.104 um pump; TE01 and TM01 are **non-degenerate** (the scalar LP
   model cannot do this), and the HE21 cutoff sits within 0.5 um of the paper's
   11.722 um.
2. **Dispersion -- negative (two ways).**  (a) The exact-vector HE11 ZDW is
   ~1.3 um short of the paper's 6.104 um.  (b) The paper states the higher-order
   modes (TE01/TM01/HE21) stay *normal* across the band; the solver puts all
   three **anomalous** at the pump, with ZDWs at ~3.8-3.9 um -- so the
   staggered-ZDW picture its multimode SC scenario rests on is not recoverable
   from the stated inputs.  (The vector solver is self-consistent: its HE11
   matches the scalar-LP LP01 ZDW to <0.2 um, and its TE01 ZDW equals LP11.)
3. **Intermodal DW at 10.8 um -- negative.**  **No pair** has an Eq. (7) root in
   the 8-13 um red window, so the paper's 10.8 um DW is not supported by the
   stated inputs.  The paper's own pair *does* phase-match, but only a **blue** DW
   at ~1.94 um (HE11-soliton partners ~2.24 um).  The intermodal condition is
   solved from scratch (the repository's ``phase_matching.dispersive_wave_roots``
   handles only the *intramodal* case -- see ``check_intermodal_dw``), and
   re-checked with a different beta1 derivative scheme.

CAVEAT (recorded, not asserted)
-------------------------------
The residual HE11-ZDW gap is plausibly the idealised-Gaussian/analytic-mode
treatment versus the paper's COMSOL full-vector FEM on a high-Δ guide; the
stock fibre may differ from the stated geometry.  What is robust is that the
paper's numbers cannot be recovered from its *printed* inputs -- which is the
evidential point (Gate 0.3 in the originating project).

Usage
-----
    python reproductions/xing_2022_multimode_chg/reproduce.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reproductions.xing_2022_multimode_chg import xing2022 as X  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_PNG = HERE / "xing_2022_dispersion.png"

LAMBDA_PUMP_UM = 6.104  # paper: HE11 ZDW / pump
PAPER_HE11_ZDW_UM = 6.104
PAPER_HE21_CUTOFF_UM = 11.722
PAPER_DW_UM = 10.8
V_HE21_CUTOFF = 3.8317  # j_{1,1}


# ---------------------------------------------------------------------------
# 0. material
# ---------------------------------------------------------------------------
def check_material() -> dict:
    from photonics_helper.materials import RefractiveIndex

    ri_se = RefractiveIndex.from_material_database("As2Se3")
    ri_s3 = RefractiveIndex.from_material_database("As2S3")

    lams = np.linspace(1.5, 6.5, 60)
    db_se = np.array([float(ri_se.n_func(lam)) for lam in lams])
    db_s3 = np.array([float(ri_s3.n_func(lam)) for lam in lams])

    n_eq6 = np.array([X.AS2SE3_FIXED.n(lam) for lam in lams])
    n_eq5 = np.array([X.AS2S3.n(lam) for lam in lams])
    err_eq6 = float(np.max(np.abs(n_eq6 - db_se) / db_se))
    err_eq5 = float(np.max(np.abs(n_eq5 - db_s3) / db_s3))

    # corrected Eq. (6) must match the repo row to ~1e-5 (same Thompson-2008 fit)
    assert err_eq6 < 2e-5, err_eq6
    assert err_eq5 < 2e-5, err_eq5

    # Eq. (6) AS PRINTED has a pole at sqrt(19) = 4.359 um
    pole = float(np.sqrt(19.0))
    assert abs(pole - 4.359) < 1e-3, pole
    n_at_pole = X.AS2SE3_PRINTED.n(pole * 1.001)
    assert n_at_pole > 4.0, n_at_pole  # diverging, unphysical
    # ... and a wrong material ZDW
    zdw_fixed = X.material_zdw(X.AS2SE3_FIXED)
    zdw_printed = X.material_zdw(X.AS2SE3_PRINTED)
    assert abs(zdw_fixed[0] - 7.22) < 0.05, zdw_fixed
    assert not zdw_printed or abs(zdw_printed[0] - 7.22) > 0.2, zdw_printed

    return {
        "eq5_as2s3_vs_repo_max_relerr": err_eq5,
        "eq6_fixed_as2se3_vs_repo_max_relerr": err_eq6,
        "eq6_printed_pole_um": round(pole, 3),
        "material_zdw_fixed_um": zdw_fixed,
        "material_zdw_printed_um": zdw_printed,
        "n_eq6_fixed_at_4um": round(X.AS2SE3_FIXED.n(4.0), 5),
    }


# ---------------------------------------------------------------------------
# 1. modes
# ---------------------------------------------------------------------------
def check_modes() -> dict:
    n_eff = {k: X.vector_neff(k, LAMBDA_PUMP_UM) for k in X.VECTOR_KINDS}
    for k, v in n_eff.items():
        assert np.isfinite(v), (k, v)

    # HE11 is the fundamental; TE01/TM01/HE21 are higher-order
    assert n_eff["HE11"] == max(n_eff.values()), n_eff
    assert n_eff["HE11"] > n_eff["TE01"] and n_eff["HE11"] > n_eff["HE21"]

    # the exact vector solver splits TE01 / TM01 (the scalar LP model cannot)
    split = abs(n_eff["TE01"] - n_eff["TM01"])
    assert split > 1e-4, split

    # HE21 cutoff near the paper's 11.722 um
    cut = X.cutoff_wavelength(V_HE21_CUTOFF)
    assert abs(cut - PAPER_HE21_CUTOFF_UM) < 0.5, (cut, PAPER_HE21_CUTOFF_UM)

    return {
        "V_at_pump": round(X.v_number(LAMBDA_PUMP_UM), 4),
        "n_core_at_pump": round(X.AS2SE3_FIXED.n(LAMBDA_PUMP_UM), 5),
        "n_clad_at_pump": round(X.AS2S3.n(LAMBDA_PUMP_UM), 5),
        "neff": {k: round(v, 5) for k, v in n_eff.items()},
        "TE01_TM01_neff_split": split,
        "HE21_cutoff_um": round(cut, 3),
        "HE21_cutoff_paper_um": PAPER_HE21_CUTOFF_UM,
    }


# ---------------------------------------------------------------------------
# 2. HE11 ZDW -- the negative
# ---------------------------------------------------------------------------
def check_zdw() -> dict:
    zdws = {k: X.vector_zdws(k) for k in X.VECTOR_KINDS}
    b2 = {k: X.vector_beta2(k, LAMBDA_PUMP_UM) for k in X.VECTOR_KINDS}

    # (a) The paper puts the HE11 ZDW at the pump, 6.104 um.  It is NOT there.
    he11 = zdws["HE11"]
    assert he11, "no HE11 ZDW found -- unexpected"
    nearest = min(he11, key=lambda z: abs(z - PAPER_HE11_ZDW_UM))
    gap = abs(nearest - PAPER_HE11_ZDW_UM)
    assert gap > 0.5, (he11, PAPER_HE11_ZDW_UM)

    # (b) The paper states TE01/TM01/HE21 stay NORMAL across the band.  The
    # exact-vector solver contradicts that: they are anomalous at the pump,
    # with ZDWs at ~3.8-3.9 um -- exactly where the scalar-LP solver puts LP11
    # (3.901 um), so this is a property of the guide, not a solver artefact.
    for k in ("TE01", "TM01", "HE21"):
        assert b2[k] < 0.0, (k, b2[k])
        assert min(zdws[k]) < 4.5, (k, zdws[k])

    # solver self-consistency: the vector HE11 agrees with the scalar LP01
    lp01 = X.scalar_lp_zdws(0, 1)
    assert lp01 and abs(lp01[0] - he11[0]) < 0.2, (lp01, he11)

    return {
        "he11_zdws_um": he11,
        "paper_he11_zdw_um": PAPER_HE11_ZDW_UM,
        "zdw_gap_um": round(gap, 3),
        "scalar_lp01_zdw_um": lp01,
        "beta2_at_pump_ps2_per_km": {k: round(v, 1) for k, v in b2.items()},
        "higher_order_zdws_um": {k: zdws[k] for k in ("TE01", "TM01", "HE21")},
        "higher_order_anomalous_at_pump": True,
        "verdict": "HE11 ZDW NOT reproduced; higher-order modes are anomalous at "
        "the pump (paper claims normal) -- the paper's dispersion "
        "picture is not recoverable from the stated inputs",
    }


# ---------------------------------------------------------------------------
# 3. intermodal dispersive wave -- the negative
# ---------------------------------------------------------------------------
def check_intermodal_dw() -> dict:
    """Eq. (7) intermodal dispersive waves for the paper's mode set.

    The paper's headline is a **red** DW at 10.8 um (TE01 <-> HE21b).  It does
    not exist: **no** pair has an Eq. (7) root in the 8-13 um window.  What the
    paper's own pair *does* phase-match is a **blue** DW at ~1.94 um -- reported,
    not hidden (the HE11-soliton partners phase-match blue too, ~2.24 um).
    """
    pairs = [
        ("HE11", "HE21"),
        ("TE01", "HE21"),
        ("HE11", "TE01"),
        ("HE11", "TM01"),
        ("TM01", "HE21"),
    ]
    roots = {f"{s}->{cr}": X.intermodal_dw_roots(s, cr) for s, cr in pairs}
    # (a) the paper's headline is a RED DW at 10.8 um -- no pair has a root in
    #     the 8-13 um window.  That DW does not exist.
    for name, r in roots.items():
        assert not any(8.0 < z < 13.0 for z in r), (name, r)
    # (b) the paper's own pair *does* phase-match, but only a BLUE DW (~1.94 um)
    assert roots["TE01->HE21"], roots["TE01->HE21"]
    assert all(z < 2.5 for z in roots["TE01->HE21"]), roots["TE01->HE21"]
    # (c) HE11-soliton partners phase-match blue too (~2.24 um) -- sanity
    blue = [z for z in roots["HE11->HE21"] if 2.0 < z < 3.0]
    assert blue, roots["HE11->HE21"]

    # Independent robustness check on Eq. (7) itself.  The repository's
    # phase_matching.dispersive_wave_roots solves the *intramodal* Cherenkov
    # condition beta(w_DW) = beta(w_s) + beta1(w_s)(w_DW - w_s): it reads
    # beta_sol, beta(w) and beta1(w_s) all from one ``beta_fn``.  An *intermodal*
    # DW needs beta(w) from the partner (HE21) but beta_sol and beta1 from the
    # soliton (HE11); the API cannot express that, and if fed the partner it
    # returns the soliton wavelength itself as a spurious root (beta_sol :=
    # beta_partner(w_s) makes f(w_s) = 0 identically).  So we do not use it here;
    # we instead re-derive the zero with a *different* derivative scheme and
    # confirm the root location is scheme-independent.
    def delta_beta_2pt(lam):
        ws = 2.0 * np.pi * X.C_UM / LAMBDA_PUMP_UM
        w = 2.0 * np.pi * X.C_UM / lam
        f_sol = lambda o: X.vector_beta("HE11", 2.0 * np.pi * X.C_UM / o)  # noqa: E731
        d = 1e6
        b1 = (f_sol(ws + d) - f_sol(ws - d)) / (2.0 * d)  # 2-point FD
        return (
            X.vector_beta("HE21", lam)
            - X.vector_beta("HE11", LAMBDA_PUMP_UM)
            - b1 * (w - ws)
        )

    ls2 = np.linspace(1.5, 4.0, 500)
    v2 = np.array([delta_beta_2pt(lam) for lam in ls2])
    cross2 = [
        round(float(ls2[i] + (ls2[i + 1] - ls2[i]) * (-v2[i]) / (v2[i + 1] - v2[i])), 3)
        for i in range(len(ls2) - 1)
        if np.isfinite(v2[i]) and np.isfinite(v2[i + 1]) and v2[i] * v2[i + 1] < 0
    ]
    assert cross2 and abs(cross2[0] - blue[0]) < 0.01, (cross2, blue)

    return {
        "eq7_roots_um": roots,
        "eq7_blue_root_2pt_recheck_um": cross2,
        "paper_dw_um": PAPER_DW_UM,
        "repo_api_note": "photonics_helper.phase_matching.dispersive_wave_roots "
        "solves the intramodal condition only; it cannot express "
        "an intermodal (multimode) Cherenkov DW, because beta_sol, "
        "beta(w) and beta1 all come from one beta_fn",
        "verdict": "red intermodal DW at 10.8 um NOT reproduced -- no Eq. 7 root "
        "in 8-13 um for any pair; the paper's pair instead "
        "phase-matches a BLUE DW at ~1.94 um",
    }


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def make_figure(mat: dict, mode: dict, zdw: dict, dw: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))

    lams = np.linspace(1.5, 8.0, 200)
    ax[0].plot(
        lams, [X.AS2SE3_FIXED.n(lam) for lam in lams], label="Eq. (6) fixed (361)"
    )
    ax[0].plot(
        lams,
        [X.AS2SE3_PRINTED.n(lam) for lam in lams],
        "--",
        label="Eq. (6) printed (19)",
    )
    ax[0].axvline(np.sqrt(19.0), color="k", lw=0.7, ls=":")
    ax[0].set_ylim(2.4, 4.0)
    ax[0].set_xlabel("λ (µm)")
    ax[0].set_ylabel("n")
    ax[0].set_title("Sellmeier: Eq. (6) as printed vs corrected")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    for k in X.VECTOR_KINDS:
        ls = np.linspace(3.0, 11.0, 120)
        b2 = [X.vector_beta2(k, lam) for lam in ls]
        ax[1].plot(ls, b2, label=k)
    ax[1].axhline(0, color="k", lw=0.7)
    ax[1].axvline(LAMBDA_PUMP_UM, color="r", lw=0.7, ls=":")
    ax[1].set_ylim(-500, 3000)
    ax[1].set_xlabel("λ (µm)")
    ax[1].set_ylabel("β₂ (ps²/km)")
    ax[1].set_title("Vector-mode β₂ (red: paper pump 6.104 µm)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    ls = np.linspace(2.2, 14.0, 500)
    for s, cr in [("TE01", "HE21"), ("HE11", "HE21")]:
        db = [X.intermodal_delta_beta(s, cr, lam) for lam in ls]
        ax[2].plot(ls, db, label=f"{s} → {cr}")
    ax[2].axhline(0, color="k", lw=0.7)
    ax[2].axvline(PAPER_DW_UM, color="r", lw=0.7, ls=":")
    ax[2].set_xlabel("λ (µm)")
    ax[2].set_ylabel("Δβ (1/m)")
    ax[2].set_title("Eq. (7) phase mismatch (red: paper's 10.8 µm)")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def validate(make_plot: bool = True) -> dict:
    out = {
        "material": check_material(),
        "modes": check_modes(),
        "he11_zdw": check_zdw(),
        "intermodal_dw": check_intermodal_dw(),
    }
    if make_plot:
        make_figure(
            out["material"], out["modes"], out["he11_zdw"], out["intermodal_dw"]
        )
        out["figure"] = str(OUT_PNG)
    return out


if __name__ == "__main__":
    result = validate(make_plot=True)
    print(json.dumps(result, indent=2, default=str))
