#!/usr/bin/env python3
"""Xing 2022 — As₂Se₃/As₂S₃ step-index MMF dispersion deck.

Reference
---------
D.-K. Xing, J.-Y. He, P. Wang, H.-Y. Guo, K. Chang, C.-C. Liu, Y.-G. Liu,
Z. Wang, *"Intermodal dispersive waves and soliton collision during multimode
supercontinuum generation in chalcogenide glass fiber"*, **Laser Phys. 32,
115401 (2022)**, doi:10.1088/1555-6611/ac979a.  Not open access; the numbers in
``parameters.json`` were read from the IOP full text (see the folder README).

What is here
------------
The paper's *inputs* re-solved from scratch so its *outputs* can be checked:

* the two Sellmeier equations — Eq. (5) As₂S₃, Eq. (6) As₂Se₃ — as printed
  **and** with the dropped superscript restored (``19`` → ``19²`` = 361);
* the **scalar LP** modes of the 10 µm-core / 246 µm-cladding fibre
  (weakly-guiding model, solved with the sibling Poletti deck's ``lp_roots``);
* the **exact vector** modes (HE₁₁, TE₀₁, TM₀₁, HE₂₁) from the closed-form
  eigenvalue equations (Okamoto, *Fundamentals of Optical Waveguides*, Eqs.
  3.97-3.99) — needed because the scalar model cannot split TE₀₁/TM₀₁/HE₂₁;
* the intermodal dispersive-wave phase-matching condition, the paper's Eq. (7):
  ``Δβ(ω) = β_cr(ω) - β_s(ω_s) - β₁_s (ω - ω_s) - γP/2 = 0``.

The verdict (``reproduce.py``) is **negative** on the paper's headline numbers;
see ``logs/phase0_xing2022_repro.md`` in the originating project.

Usage
-----
    python reproductions/xing_2022_multimode_chg/reproduce.py
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import jn_zeros, jv, jvp, kv, kvp

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent  # .../reproductions/<this>/ -> repo root
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from reproductions.poletti_2008_multimode.overlap import lp_roots  # noqa: E402

C_UM = 299792458.0 * 1e6  # speed of light in um/s

# --------------------------------------------------------------------------
# 1. The paper's Sellmeier equations  (Eq. 5 = As2S3, Eq. 6 = As2Se3)
# --------------------------------------------------------------------------
# Standard form  n^2 = 1 + sum_j A_j lambda^2 / (lambda^2 - B_j),  lambda in um.
# Eq. (5), As2S3 (5 terms).
SELLMEIER_AS2S3 = [
    (1.8983678, 0.0225),
    (1.9222979, 0.0625),
    (0.8765134, 0.1225),
    (0.1188704, 0.2025),
    (0.9569903, 750.0),
]
# Eq. (6), As2Se3 (3 terms) AS PRINTED -- the second B coefficient lost its
# square and reads ``lambda^2 - 19``.
SELLMEIER_AS2SE3_PRINTED = [
    (4.9948719, 0.0584),
    (0.1207152, 19.0),
    (1.7123685, 0.2336),
]
# Eq. (6) CORRECTED: the Thompson-2008 As2Se3 fit it is copied from has
# B = 19^2 = 361.  This is the row the repository's material database carries.
SELLMEIER_AS2SE3_FIXED = [
    (4.9948719, 0.0584),
    (0.1207152, 361.0),
    (1.7123685, 0.2336),
]

CORE_RADIUS_UM = 5.0  # 10 um core diameter
CLADDING_RADIUS_UM = 123.0  # 246 um cladding diameter


def n_sellmeier(lam, terms) -> np.ndarray:
    """Refractive index from a ``(A_j, B_j)`` Sellmeier sum (lambda in um)."""
    l2 = np.asarray(lam, float) ** 2
    s = np.ones_like(l2)
    for a, b in terms:
        s = s + a * l2 / (l2 - b)
    with np.errstate(invalid="ignore"):
        return np.sqrt(s)  # NaN where the sum goes negative (inside a pole)


class _Glass:
    """Callable Sellmeier glass: ``glass(lam_um) -> n`` (scalar in, scalar out)."""

    def __init__(self, terms, name: str = ""):
        self.terms = list(terms)
        self.name = name

    def n(self, lam):
        return float(n_sellmeier(lam, self.terms))

    def __call__(self, lam):
        return self.n(lam)


AS2S3 = _Glass(SELLMEIER_AS2S3, "As2S3 (eq. 5)")
AS2SE3_FIXED = _Glass(SELLMEIER_AS2SE3_FIXED, "As2Se3 (eq. 6 fixed)")
AS2SE3_PRINTED = _Glass(SELLMEIER_AS2SE3_PRINTED, "As2Se3 (eq. 6 as printed)")


def material_zdw(glass, lo=2.0, hi=14.0, n=400) -> list[float]:
    """Zero-material-dispersion wavelengths of a Sellmeier glass (um).

    Material dispersion ``D_m = -(lambda/c) d^2 n / d lambda^2`` vanishes where
    ``d^2 n / d lambda^2`` changes sign.
    """
    lams = np.linspace(lo, hi, n)
    ns = np.array([glass.n(lam) for lam in lams])
    d1 = np.gradient(np.gradient(ns, lams), lams)
    crosses = []
    for i in range(n - 1):
        if np.isfinite(d1[i]) and np.isfinite(d1[i + 1]) and d1[i] * d1[i + 1] < 0:
            crosses.append(
                round(
                    float(
                        lams[i]
                        + (lams[i + 1] - lams[i]) * (-d1[i]) / (d1[i + 1] - d1[i])
                    ),
                    3,
                )
            )
    return crosses


# --------------------------------------------------------------------------
# 2. Step-index fibre: V number and the scalar LP modes
# --------------------------------------------------------------------------
def v_number(lam, core=AS2SE3_FIXED, clad=AS2S3, a=CORE_RADIUS_UM) -> float:
    n_co, n_cl = core.n(lam), clad.n(lam)
    return 2.0 * np.pi / lam * a * np.sqrt(max(n_co * n_co - n_cl * n_cl, 0.0))


@functools.lru_cache(maxsize=None)
def scalar_lp_beta(
    lam, ell, m_rad, core=AS2SE3_FIXED, clad=AS2S3, a=CORE_RADIUS_UM
) -> float:
    """β of scalar LP_{ell,m} in **rad/um**; NaN if the mode is not guided."""
    V = v_number(lam, core, clad, a)
    if not np.isfinite(V) or V < 1e-9:
        return np.nan
    try:
        roots = lp_roots(V, l_max=2, m_max=2)
    except (ValueError, ZeroDivisionError):
        return np.nan
    if (ell, m_rad) not in roots:
        return np.nan
    k0 = 2.0 * np.pi / lam
    return float(np.sqrt((k0 * core.n(lam)) ** 2 - (roots[(ell, m_rad)] / a) ** 2))


def scalar_lp_beta2(
    lam0, ell, m_rad, core=AS2SE3_FIXED, clad=AS2S3, h_rel=0.01
) -> float:
    """d²β/dω² in ps²/km (5-point stencil); NaN if any sample is unguided."""
    om0 = 2.0 * np.pi * C_UM / lam0
    h = h_rel * om0
    oms = om0 + h * np.array([-2, -1, 0, 1, 2])
    b = (
        np.array(
            [
                scalar_lp_beta(2.0 * np.pi * C_UM / o, ell, m_rad, core, clad)
                for o in oms
            ]
        )
        * 1e6
    )  # rad/m
    if not np.all(np.isfinite(b)):
        return np.nan
    return (-b[0] + 16 * b[1] - 30 * b[2] + 16 * b[3] - b[4]) / (12 * h * h) * 1e27


def scalar_lp_zdws(
    ell=0, m_rad=1, core=AS2SE3_FIXED, clad=AS2S3, lo=2.0, hi=14.0, n=120
) -> list[float]:
    """Zero-dispersion wavelengths of scalar LP_{ell,m} (um)."""
    lams = np.linspace(lo, hi, n)
    v = np.array([scalar_lp_beta2(lam, ell, m_rad, core, clad) for lam in lams])
    out = []
    for i in range(n - 1):
        if np.isfinite(v[i]) and np.isfinite(v[i + 1]) and v[i] * v[i + 1] < 0:
            out.append(
                round(
                    float(
                        lams[i] + (lams[i + 1] - lams[i]) * (-v[i]) / (v[i + 1] - v[i])
                    ),
                    3,
                )
            )
    return out


# --------------------------------------------------------------------------
# 3. Exact vector modes of a circular step-index fibre  (Okamoto 3.97-3.99)
# --------------------------------------------------------------------------
VECTOR_KINDS = ("HE11", "TE01", "TM01", "HE21")


def res_hybrid(nu, u, V, n1, n2) -> float:
    w2 = V * V - u * u
    if w2 <= 0:
        return np.nan
    w = np.sqrt(w2)
    r = (n2 / n1) ** 2
    X = jvp(nu, u) / (u * jv(nu, u))
    Y = kvp(nu, w) / (w * kv(nu, w))
    return (X + Y) * (X + r * Y) - nu * nu * (1 / u**2 + 1 / w**2) * (
        1 / u**2 + r / w**2
    )


def res_te(u, V) -> float:
    w = np.sqrt(V * V - u * u)
    return jv(1, u) / (u * jv(0, u)) + kv(1, w) / (w * kv(0, w))


def res_tm(u, V, n1, n2) -> float:
    w = np.sqrt(V * V - u * u)
    return jv(1, u) / (u * jv(0, u)) + (n2 / n1) ** 2 * kv(1, w) / (w * kv(0, w))


def _bracket(fn, guess, V, poles, half=0.5):
    """Sign-change bracket for ``fn`` near ``guess``, skipping Bessel poles."""
    lo = max(1e-8, guess - half)
    hi = min(V * (1 - 1e-9), guess + half)
    us = np.linspace(lo, hi, 300)
    vals = np.array([fn(u) for u in us])
    for k in range(len(us) - 1):
        a, b = vals[k], vals[k + 1]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if any(us[k] < p < us[k + 1] for p in poles):
            continue
        if a * b < 0 and abs(a) < 1e4 and abs(b) < 1e4:
            return us[k], us[k + 1]
    return None


@functools.lru_cache(maxsize=None)
def vector_u(kind, lam, core=AS2SE3_FIXED, clad=AS2S3, a=CORE_RADIUS_UM):
    """Normalised transverse wavenumber ``u`` of a vector mode; NaN if cut off."""
    V = v_number(lam, core, clad, a)
    n1, n2 = core.n(lam), clad.n(lam)
    if kind in ("HE11", "HE21"):
        nu = 1 if kind == "HE11" else 2
        poles = jn_zeros(nu, 12)
        fn = lambda u: res_hybrid(nu, u, V, n1, n2)  # noqa: E731
        key = (0, 1) if kind == "HE11" else (1, 1)
    elif kind == "TE01":
        poles = jn_zeros(0, 12)
        fn = lambda u: res_te(u, V)  # noqa: E731
        key = (1, 1)
    elif kind == "TM01":
        poles = jn_zeros(0, 12)
        fn = lambda u: res_tm(u, V, n1, n2)  # noqa: E731
        key = (1, 1)
    else:
        raise ValueError(f"unknown vector kind {kind!r}")

    try:
        sc = lp_roots(V, l_max=2, m_max=2)
    except (ValueError, ZeroDivisionError):
        return np.nan
    guess = sc.get(key)
    if guess is None:
        return np.nan
    br = _bracket(fn, guess, V, poles)
    if br is None:
        return np.nan
    try:
        return float(brentq(fn, br[0], br[1], xtol=1e-14, rtol=1e-15))
    except ValueError:
        return np.nan


@functools.lru_cache(maxsize=None)
def vector_beta(kind, lam, core=AS2SE3_FIXED, clad=AS2S3, a=CORE_RADIUS_UM) -> float:
    """β of a vector mode in **rad/m**; NaN if cut off."""
    u = vector_u(kind, lam, core, clad, a)
    if not np.isfinite(u):
        return np.nan
    k0 = 2.0 * np.pi / lam
    arg = (k0 * core.n(lam)) ** 2 - (u / a) ** 2
    return float(np.sqrt(arg) * 1e6) if arg > 0 else np.nan


def vector_neff(kind, lam, **kw) -> float:
    b = vector_beta(kind, lam, **kw)
    return b / (2.0 * np.pi / (lam * 1e-6)) if np.isfinite(b) else np.nan


def vector_beta2(kind, lam0, h_rel=0.01, **kw) -> float:
    """d²β/dω² in ps²/km (5-point stencil in omega); NaN if any sample fails."""
    om0 = 2.0 * np.pi * C_UM / lam0
    h = h_rel * om0
    oms = om0 + h * np.array([-2, -1, 0, 1, 2])
    b = np.array([vector_beta(kind, 2.0 * np.pi * C_UM / o, **kw) for o in oms])
    if not np.all(np.isfinite(b)):
        return np.nan
    return (-b[0] + 16 * b[1] - 30 * b[2] + 16 * b[3] - b[4]) / (12 * h * h) * 1e27


def vector_zdws(kind, lo=2.0, hi=14.0, n=120, **kw) -> list[float]:
    lams = np.linspace(lo, hi, n)
    v = np.array([vector_beta2(kind, lam, **kw) for lam in lams])
    out = []
    for i in range(n - 1):
        if np.isfinite(v[i]) and np.isfinite(v[i + 1]) and v[i] * v[i + 1] < 0:
            out.append(
                round(
                    float(
                        lams[i] + (lams[i + 1] - lams[i]) * (-v[i]) / (v[i + 1] - v[i])
                    ),
                    3,
                )
            )
    return out


def cutoff_wavelength(
    V_cut, core=AS2SE3_FIXED, clad=AS2S3, a=CORE_RADIUS_UM, lo=4.0, hi=20.0
) -> float:
    """λ (um) where ``V(λ)`` falls to ``V_cut`` — a mode's long-λ cutoff."""
    f = lambda lam: v_number(lam, core, clad, a) - V_cut  # noqa: E731
    try:
        return float(brentq(f, lo, hi, xtol=1e-12))
    except ValueError:
        return float("nan")


# --------------------------------------------------------------------------
# 4. Intermodal dispersive-wave phase matching  (the paper's Eq. 7)
# --------------------------------------------------------------------------
def intermodal_delta_beta(kind_s, kind_cr, lam, lam_s=6.104, gamma_p=0.0, **kw):
    """Δβ(ω) of Eq. (7) [1/m]; a zero is an intermodal DW."""
    ws = 2.0 * np.pi * C_UM / lam_s
    h = 0.005 * ws
    oms = ws + h * np.array([-2, -1, 0, 1, 2])
    bs = np.array([vector_beta(kind_s, 2.0 * np.pi * C_UM / o, **kw) for o in oms])
    if not np.all(np.isfinite(bs)):
        return np.nan
    b1s = (bs[0] - 8 * bs[1] + 8 * bs[3] - bs[4]) / (12 * h)
    bS = vector_beta(kind_s, lam_s, **kw)
    bcr = vector_beta(kind_cr, lam, **kw)
    if not np.isfinite(bcr):
        return np.nan
    return bcr - bS - b1s * (2.0 * np.pi * C_UM / lam - ws) - 0.5 * gamma_p * 1e-6


def intermodal_dw_roots(
    kind_s, kind_cr, lam_s=6.104, gamma_p=0.0, lo=1.5, hi=14.0, n=800, **kw
) -> list[float]:
    """Wavelengths where Eq. (7) is satisfied (intermodal DWs), in um."""
    lams = np.linspace(lo, hi, n)
    db = np.array(
        [
            intermodal_delta_beta(kind_s, kind_cr, lam, lam_s, gamma_p, **kw)
            for lam in lams
        ]
    )
    out = []
    for i in range(n - 1):
        if np.isfinite(db[i]) and np.isfinite(db[i + 1]) and db[i] * db[i + 1] < 0:
            out.append(
                round(
                    float(
                        lams[i]
                        + (lams[i + 1] - lams[i]) * (-db[i]) / (db[i + 1] - db[i])
                    ),
                    3,
                )
            )
    return out


# --------------------------------------------------------------------------
# 5. Adaptor: expose a vector mode to the repository's phase-matching API
# --------------------------------------------------------------------------
class VectorBetaModel:
    """``beta_fn`` for ``photonics_helper.phase_matching`` — β(ω) of one mode.

    Callable of angular frequency in rad/s, with a ``beta1`` method so the
    repository's ``dispersive_wave_roots`` uses an analytic-ish derivative
    (central finite difference) instead of its coarse fallback.
    """

    def __init__(self, kind, **kw):
        self.kind = kind
        self.kw = kw

    def __call__(self, omega):
        omega = np.asarray(omega, dtype=float)
        lam = 2.0 * np.pi * C_UM / omega
        b = np.array(
            [
                vector_beta(self.kind, float(lam), **self.kw)
                for lam in np.atleast_1d(lam)
            ]
        )
        return float(b[0]) if np.isscalar(omega) or omega.ndim == 0 else b

    def beta1(self, omega):
        omega = float(np.asarray(omega, dtype=float))
        d = 1e10
        return (float(self(omega + d)) - float(self(omega - d))) / (2.0 * d)
