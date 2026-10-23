"""Exact vector eigenmodes of a circular step-index fibre (GAP A1).

Closed-form solution of Maxwell's equations for a circular step-index guide:
the hybrid ``HE``/``EH`` modes and the ``TE``/``TM`` modes, with no mesh, no
weakly-guiding approximation, and no materials attached. This is the exact
``C_∞v`` reference that symmetry-sensitive work (OAM splitting, polarisation
splitting, cross-solver truth checks) needs.

Equation grounding
------------------
Okamoto, *Fundamentals of Optical Waveguides*, Eqs. 3.97 to 3.99 (the same
equations already used by ``reproductions/xing_2022_multimode_chg``). With
``V = 2π a √(n_core² − n_clad²)/λ``, ``u`` the normalised transverse
wavenumber, ``w = √(V² − u²)`` and ``r = (n_clad/n_core)²``:

``HE_ν``/``EH_ν``::

    X = J'_ν(u)/(u J_ν(u)),     Y = K'_ν(w)/(w K_ν(w))
    (X + Y)(X + rY) − ν²(1/u² + 1/w²)(1/u² + r/w²) = 0

For ``ν = 0`` the two factors are the ``TE`` (``X + Y = 0``) and ``TM``
(``X + rY = 0``) equations. The transverse field's azimuthal order is
``ℓ = ν − 1`` for ``HE_ν``, ``ℓ = ν + 1`` for ``EH_ν``, and ``ℓ = 1`` for
``TE``/``TM``.

Why this module exists
----------------------
The reproduction above hard-codes four kinds and ±0.5-around-the-scalar-guess
bracketing, which misses the vector root once the guide is strongly guiding
(measured: ``HE21`` → NaN at Δn = 1.65). Here the root is found by a full
pole-aware sign-change scan and matched to the scalar LP root of its own group,
and any order is available. The scalar LP seeding is implemented locally so the
library does not import from ``reproductions/``.

Weakly-guiding limit
--------------------
``u`` reduces to the scalar ``lp_roots`` value for the matching LP group as
Δn → 0 (tested to < 1e-4 at Δn = 1e-4). ``n_eff² = n_core² −
(n_core² − n_clad²)(u/V)²`` and ``β = (2π/λ) n_eff`` (rad/m); the reported
``β`` is exact, not the weakly-guiding approximation.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.optimize import brentq
from scipy.special import jn_zeros, jv, jvp, kv, kvp

__all__ = ["VectorMode", "vector_modes"]

_FAMILIES = ("HE", "EH", "TE", "TM")


class VectorMode(NamedTuple):
    """One exact vector mode of a circular step-index guide.

    ``kind`` is the mode name (``"HE11"``, ``"EH21"``, ``"TE01"``, …),
    ``ell`` the azimuthal order of the transverse field, ``nu`` the order in
    the characteristic equation (``0`` for ``TE``/``TM``), ``m`` the radial
    order, ``u`` the normalised transverse wavenumber, ``n_eff`` the effective
    index, and ``beta`` the propagation constant in rad/m. Cut-off modes carry
    NaN for ``u``, ``n_eff``, and ``beta``.
    """

    kind: str
    ell: int
    nu: int
    m: int
    u: float
    n_eff: float
    beta: float


def _check_inputs(
    n_core: float, n_clad: float, radius_um: float, wavelength_um: float
) -> None:
    if not np.isfinite(n_core) or not np.isfinite(n_clad):
        raise ValueError("n_core and n_clad must be finite")
    if not n_core > n_clad > 0:
        raise ValueError(f"require n_core > n_clad > 0, got {n_core=}, {n_clad=}")
    if not np.isfinite(radius_um) or radius_um <= 0:
        raise ValueError(f"radius_um must be positive, got {radius_um!r}")
    if not np.isfinite(wavelength_um) or wavelength_um <= 0:
        raise ValueError(f"wavelength_um must be positive, got {wavelength_um!r}")


def _v_number(
    n_core: float, n_clad: float, radius_um: float, wavelength_um: float
) -> float:
    return float(
        2.0
        * np.pi
        / wavelength_um
        * radius_um
        * np.sqrt(n_core * n_core - n_clad * n_clad)
    )


def _bessel_zeros(order: int, count: int) -> np.ndarray:
    return np.asarray(jn_zeros(order, count), dtype=float)


def _lp_residual(ell: int, u, V: float):
    """Scalar LP characteristic equation (same as the Poletti reproduction)."""
    u_arr = np.asarray(u, dtype=float)
    w2 = V * V - u_arr * u_arr
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.sqrt(w2)
        res = (
            u_arr * jv(ell - 1, u_arr) / jv(ell, u_arr)
            + w * kv(ell + 1, w) / kv(ell, w)
            - 2.0 * ell
        )
    return np.where(w2 > 0, res, np.nan)


def _lp_u(V: float, ell: int, m: int) -> float:
    """Scalar LP root ``u_{ℓm}`` for seeding; NaN when the mode is cut off."""
    if ell == 0:
        lo = 0.0 if m == 1 else float(_bessel_zeros(0, m)[m - 2])
        hi = float(_bessel_zeros(0, m)[m - 1])
    else:
        lo = float(_bessel_zeros(ell - 1, m)[m - 1])
        hi = float(_bessel_zeros(ell, m)[m - 1])
    hi = min(hi, V * (1.0 - 1e-12))
    if hi <= lo:
        return float("nan")
    try:
        return float(
            brentq(
                lambda u: float(_lp_residual(ell, u, V)),
                lo + 1e-12,
                hi - 1e-12,
                xtol=1e-14,
                rtol=1e-15,
            )
        )
    except ValueError:
        return float("nan")


def _hybrid_residual(nu: int, u, V: float, r: float):
    u_arr = np.asarray(u, dtype=float)
    w2 = V * V - u_arr * u_arr
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        w = np.sqrt(w2)
        X = jvp(nu, u_arr) / (u_arr * jv(nu, u_arr))
        Y = kvp(nu, w) / (w * kv(nu, w))
        res = (X + Y) * (X + r * Y) - nu * nu * (1.0 / u_arr**2 + 1.0 / w**2) * (
            1.0 / u_arr**2 + r / w**2
        )
    return np.where(w2 > 0, res, np.nan)


def _te_residual(u, V: float):
    u_arr = np.asarray(u, dtype=float)
    w2 = V * V - u_arr * u_arr
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.sqrt(w2)
        res = jv(1, u_arr) / (u_arr * jv(0, u_arr)) + kv(1, w) / (w * kv(0, w))
    return np.where(w2 > 0, res, np.nan)


def _tm_residual(u, V: float, r: float):
    u_arr = np.asarray(u, dtype=float)
    w2 = V * V - u_arr * u_arr
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.sqrt(w2)
        res = jv(1, u_arr) / (u_arr * jv(0, u_arr)) + r * kv(1, w) / (w * kv(0, w))
    return np.where(w2 > 0, res, np.nan)


def _sign_change_roots(
    fn, V: float, poles: np.ndarray, n_scan: int = 4000
) -> list[float]:
    """All finite sign changes of ``fn`` on ``(0, V)``, skipping Bessel poles."""
    us = np.linspace(1e-9, V * (1.0 - 1e-12), n_scan)
    vals = np.asarray(fn(us), dtype=float)
    roots: list[float] = []
    for i in range(us.size - 1):
        a, b = vals[i], vals[i + 1]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if a * b >= 0.0 or abs(a) > 1e4 or abs(b) > 1e4:
            continue
        if np.any((poles > us[i]) & (poles <= us[i + 1])):
            continue
        try:
            roots.append(
                float(
                    brentq(
                        lambda u: float(fn(np.asarray([u]))[0]),
                        us[i],
                        us[i + 1],
                        xtol=1e-14,
                        rtol=1e-15,
                    )
                )
            )
        except ValueError:  # pragma: no cover - numerical guard at a pole
            continue
    return roots


def _hybrid_u(nu: int, V: float, r: float, target: float) -> float:
    """Vector root nearest the scalar LP root of the mode's own LP group."""
    if not np.isfinite(target):
        return float("nan")
    nz = max(8, int(V) + 3)
    poles = _bessel_zeros(nu, nz)
    poles = poles[poles < V]
    roots = _sign_change_roots(lambda u: _hybrid_residual(nu, u, V, r), V, poles)
    if not roots:
        return float("nan")
    return float(min(roots, key=lambda u: abs(u - target)))


def _te_tm_u(kind: str, V: float, r: float, m: int) -> float:
    lo = float(_bessel_zeros(0, m)[m - 1])
    hi = min(float(_bessel_zeros(1, m)[m - 1]), V * (1.0 - 1e-12))
    if hi <= lo:
        return float("nan")
    fn = (
        (lambda u: _te_residual(u, V))
        if kind == "TE"
        else (lambda u: _tm_residual(u, V, r))
    )
    try:
        a, b = (
            float(fn(np.asarray([lo + 1e-10]))[0]),
            float(fn(np.asarray([hi - 1e-10]))[0]),
        )
        if not (np.isfinite(a) and np.isfinite(b)) or a * b >= 0.0:
            return float("nan")
        return float(
            brentq(
                lambda u: float(fn(np.asarray([u]))[0]),
                lo + 1e-10,
                hi - 1e-10,
                xtol=1e-14,
                rtol=1e-15,
            )
        )
    except ValueError:  # pragma: no cover - cut-off guard
        return float("nan")


def _parse_kind(kind: str) -> tuple[str, int, int, int, int]:
    """``("HE21") -> (family, nu, m, ell, equation-order)``."""
    if not isinstance(kind, str):
        raise ValueError(f"mode kind must be a string, got {kind!r}")
    upper = kind.strip().upper()
    family = upper[:2]
    if family not in _FAMILIES:
        raise ValueError(
            f"unknown mode family in {kind!r}; expected one of {_FAMILIES}"
        )
    digits = upper[2:]
    if len(digits) < 2 or not digits.isdigit():
        raise ValueError(
            f"cannot parse {kind!r}; use <family><order><m>, e.g. 'HE11', 'EH21', 'TE01'"
        )
    nu = int(digits[:-1])
    m = int(digits[-1])
    if m < 1:
        raise ValueError(f"radial order must be >= 1 in {kind!r}")
    if family in ("TE", "TM"):
        if nu != 0:
            raise ValueError(f"{family} modes have order 0, got {kind!r}")
        return family, 0, m, 1, 0
    if nu < 1:
        raise ValueError(f"{family} order must be >= 1, got {kind!r}")
    ell = nu - 1 if family == "HE" else nu + 1
    return family, nu, m, ell, nu


def _default_kinds() -> list[str]:
    kinds: list[str] = []
    for nu in range(1, 5):
        for m in range(1, 4):
            kinds.extend((f"HE{nu}{m}", f"EH{nu}{m}"))
    for m in range(1, 4):
        kinds.extend((f"TE0{m}", f"TM0{m}"))
    return kinds


def vector_modes(
    n_core: float,
    n_clad: float,
    radius_um: float,
    wavelength_um: float,
    kinds: list[str] | tuple[str, ...] | None = None,
) -> list[VectorMode]:
    """Exact vector modes of a circular step-index guide.

    Parameters
    ----------
    n_core, n_clad
        Core and cladding refractive indices (dimensionless, exact values). Dispersion is the caller's problem.
    radius_um, wavelength_um
        Core radius and vacuum wavelength in µm (the convention of the
        reproduction this promotes).
    kinds
        Mode names ``"<family><order><m>"``, e.g. ``["HE11", "TE01", "TM01",
        "HE21", "EH11", "HE31", "EH21"]``. ``None`` enumerates guided
        ``HE``/``EH`` (orders 1 to 4, radial 1 to 3) and ``TE``/``TM`` (radial 1 to 3).

    Returns
    -------
    One :class:`VectorMode` per requested kind, in the requested order. An
    explicitly requested cut-off mode is returned with NaN ``u``/``n_eff``/
    ``beta``; with ``kinds=None`` cut-off candidates are omitted.
    """
    _check_inputs(n_core, n_clad, radius_um, wavelength_um)
    V = _v_number(n_core, n_clad, radius_um, wavelength_um)
    if V <= 0.0:
        raise ValueError(f"guide has no guidance (V = {V:.3g})")
    r = (n_clad / n_core) ** 2
    k0 = 2.0 * np.pi / (wavelength_um * 1e-6)

    requested = _default_kinds() if kinds is None else list(kinds)
    out: list[VectorMode] = []
    for kind in requested:
        family, nu, m, ell, nu_eq = _parse_kind(kind)
        target = _lp_u(V, ell, m)
        if family in ("TE", "TM"):
            u = _te_tm_u(family, V, r, m)
        else:
            u = _hybrid_u(nu_eq, V, r, target)
        if not np.isfinite(u):
            if kinds is None:
                continue
            out.append(VectorMode(kind, ell, nu_eq, m, np.nan, np.nan, np.nan))
            continue
        arg = n_core * n_core - (n_core * n_core - n_clad * n_clad) * (u / V) ** 2
        if arg <= 0.0:  # pragma: no cover - only at V -> 0
            n_eff = beta = float("nan")
        else:
            n_eff = float(np.sqrt(arg))
            beta = float(k0 * n_eff)
        out.append(VectorMode(kind.upper(), ell, nu_eq, m, u, n_eff, beta))
    return out
