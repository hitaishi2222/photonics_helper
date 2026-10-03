"""Overlap coefficients of the multimode GNLSE (Poletti & Horak 2008, Eq. 7).

The nonlinear coupling of the multimode GNLSE of F. Poletti and P. Horak,
*J. Opt. Soc. Am. B* **25**, 1645 (2008), runs through the four-mode overlap
coefficients

    Q^(1)_plmn = C / (N_p N_l N_m N_n) INT [F_p* . F_l ][F_m . F_n*] dx dy
    Q^(2)_plmn = C / (N_p N_l N_m N_n) INT [F_p* . F_l*][F_m . F_n ] dx dy

with ``C = eps0 n0^2 c^2 / 12`` and ``N_k = INT |F_k|^2 dx dy`` the mode-power
constants.  This module evaluates those integrals for the mode set of a
cylindrically symmetric (step-index) fiber and exposes the paper's symmetry
results as checks:

* Eq. (18) — the *spatial* selection rules, which here **emerge** from the
  azimuthal quadrature rather than being hard-coded;
* Eq. (19) — the weakly-guiding *polarisation* selection rules (the circular
  basis vectors obey ``e_a* . e_b = delta_ab`` exactly);
* Eq. (16) — the exact permutation / complex-conjugation identities of
  ``Q^(1)`` and ``Q^(2)``;
* the Sec. 3 statement that for *real-valued* mode functions
  ``Q^(1) = Q^(2)``, which holds in the real (cos/sin) LP basis and fails in
  the helical Eq. (17) basis.

Mode basis
----------
Both azimuthal conventions the paper uses are supported, selected by
``basis``:

``"helical"``
    ``F_p(r, phi) = R_p(r) exp(i m_p phi)`` — Eq. (17), the basis in which the
    selection rules Eq. (18) are written down.
``"real"``
    the physical LP basis, ``F = R(r) cos(m phi)`` and ``R(r) sin(m phi)``
    (a single shape for ``m = 0``).  ``F_l* = F_l`` here, so the two integrands
    of Eq. (7) coincide and ``Q^(1) = Q^(2)`` element-wise.

Both bases carry the *same* integer angular labels and the same ordering, so
the two tensors can be compared index by index (label ``+m`` = the cos copy,
``-m`` = the sin copy).

Polarisation is treated analytically: with ``F = R(r) a(phi) e_sigma`` and
``e_+ = (ex + i ey)/sqrt(2)``, the Cartesian dot product is
``F_p* . F_l = R_p R_l a_p* a_l e_{sigma_p}* . e_{sigma_l} = 0`` unless
``sigma_p == sigma_l`` — an exact identity of the circular basis, not an
approximation.  It is applied as a hard mask; everything else (radial profile,
azimuthal product, both tensor types) comes from quadrature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import jn_zeros, jv, kv

__all__ = [
    "LPMode",
    "StepIndexFiber",
    "angular_profile",
    "complexity_sweep",
    "engine_weights",
    "expected_survivors",
    "lp_roots",
    "magnitude_distribution",
    "measure_selection_rules",
    "mode_set_10",
    "overlap_tensor",
    "real_basis_equality",
    "selection_rule_masks",
    "symmetry_identity_errors",
    "v_number",
]


# --------------------------------------------------------------------------
# 1. LP-mode eigenproblem of a step-index fiber
# --------------------------------------------------------------------------
def v_number(core_radius_um: float, na: float, wavelength_um: float) -> float:
    """Normalised frequency ``V = 2 pi a NA / lambda`` (paper Sec. 4.A)."""
    return 2.0 * np.pi * core_radius_um * na / wavelength_um


def _eigen_eq(ell: int, u: float, V: float) -> float:
    """Scalar LP eigenvalue equation for a step-index fiber.

    Continuity of the tangential field at ``r = a`` gives
    ``u J'_l(u)/J_l(u) = w K'_l(w)/K_l(w)`` with ``w = sqrt(V^2 - u^2)``, and

        u J'_l/J_l = u J_{l-1}/J_l - l
        w K'_l/K_l = -w K_{l+1}/K_l + l

    so the residual is ``u J_{l-1}/J_l + w K_{l+1}/K_l - 2l``.  For ``l = 0``
    this is the standard LP01 equation ``u J_1(u)/J_0(u) = w K_1(w)/K_0(w)``.
    """
    w2 = V * V - u * u
    if w2 <= 0.0:
        return np.inf
    w = np.sqrt(w2)
    return u * jv(ell - 1, u) / jv(ell, u) + w * kv(ell + 1, w) / kv(ell, w) - 2.0 * ell


def lp_roots(V: float, l_max: int = 4, m_max: int = 3) -> dict[tuple[int, int], float]:
    """Guided LP radial roots ``{(l, m_radial): u}`` of a step-index fiber.

    ``LP01`` is guided for any ``V`` (root bracketed in ``(0, j_{0,1})``).  For
    ``l >= 1`` the ``m``-th LP mode of azimuthal index ``l`` cuts off at the
    ``m``-th zero of ``J_{l-1}`` and its root sits between that zero and the
    ``m``-th zero of ``J_l`` (a pole of the residual).
    """
    roots: dict[tuple[int, int], float] = {}
    z0 = jn_zeros(0, 2)
    hi = min(z0[0], V - 1e-12)
    if hi > 1e-9:
        roots[(0, 1)] = brentq(
            lambda u: _eigen_eq(0, u, V), 1e-12, hi, xtol=1e-15, rtol=1e-15
        )
    for ell in range(1, l_max + 1):
        z_prev = jn_zeros(ell - 1, m_max + 1)
        z_curr = jn_zeros(ell, m_max)
        for m in range(1, m_max + 1):
            if z_prev[m - 1] >= V - 1e-12:  # below cutoff
                continue
            lo = z_prev[m - 1] + 1e-12
            up = min(z_curr[m - 1] - 1e-12, V - 1e-12)
            if up <= lo:
                continue
            try:
                roots[(ell, m)] = brentq(
                    lambda u: _eigen_eq(ell, u, V), lo, up, xtol=1e-15, rtol=1e-15
                )
            except ValueError:  # pragma: no cover - bracketing guard
                continue
    return roots


# --------------------------------------------------------------------------
# 2. Mode set
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LPMode:
    """One guided LP mode: radial profile + azimuthal label + polarisation.

    Attributes
    ----------
    ell, m_radial : int
        LP indices (azimuthal and radial order).
    label : int
        Angular label: ``0`` for the axisymmetric mode, ``+m`` / ``-m`` for the
        two azimuthal copies (helical: ``exp(-/+ i m phi)``; real:
        ``cos(m phi)`` / ``sin(m phi)``).
    sigma : {"+", "-"}
        Circular polarisation state ``e_sigma``.
    """

    ell: int
    m_radial: int
    label: int
    sigma: str
    u: float
    w: float
    R: NDArray
    power: float

    @property
    def m(self) -> int:
        """Azimuthal order ``|label|``."""
        return abs(self.label)

    @property
    def name(self) -> str:
        return f"LP{self.ell}{self.m_radial}[{self.label:+d}]{self.sigma}"

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"LPMode({self.name}, u={self.u:.6f}, W={self.w:.6f})"


def _radial_grid(n_core: int, n_clad: int, r_max: float) -> NDArray:
    """Polar radial grid: linear through the core, log beyond ``r = 1``.

    ``r = 1`` (the core/cladding interface) is a grid point, which is where
    the profile's first derivative jumps.
    """
    r_core = np.linspace(0.0, 1.0, n_core)
    r_clad = np.exp(np.linspace(np.log(1.0 + 1e-3), np.log(r_max), n_clad))
    return np.concatenate([r_core, r_clad])


def _trapezoid_weights(x: NDArray) -> NDArray:
    """Trapezoidal quadrature weights for a non-uniform grid."""
    w = np.zeros_like(x)
    dx = np.diff(x)
    w[:-1] += 0.5 * dx
    w[1:] += 0.5 * dx
    return w


def angular_profile(labels: list[int], phi: NDArray, basis: str) -> NDArray:
    """Azimuthal profile ``a(phi)`` of every mode, shape ``(N, len(phi))``."""
    m = np.abs(np.asarray(labels, dtype=int))
    out = np.empty((len(labels), phi.size), dtype=complex)
    for i, (mi, li) in enumerate(zip(m, labels)):
        if mi == 0:
            out[i] = 1.0
        elif basis == "helical":
            out[i] = np.exp(1j * li * phi)
        elif basis == "real":
            out[i] = np.cos(mi * phi) if li > 0 else np.sin(mi * phi)
        else:  # pragma: no cover - guarded by the caller
            raise ValueError(f"unknown basis {basis!r}")
    return out


class StepIndexFiber:
    """Weakly-guiding LP mode set of a step-index fiber.

    Parameters
    ----------
    V : float
        Normalised frequency, :func:`v_number`.
    basis : {"helical", "real"}
        Azimuthal convention (see the module docstring).
    n_modes : int, optional
        Keep only the first ``n_modes`` in the cutoff ordering
        ``(u, l, m_radial, |label|, sigma)``.  The ordering is independent of
        the basis, so both bases describe the same ``n_modes`` modes.
    single_pol : {"+", "-", None}
        Keep one circular polarisation state only.
    l_max, m_max : int
        LP orders solved for before the truncation.
    n_core, n_clad, r_max, n_phi : int
        Quadrature resolution (core radius is the length unit).
    """

    def __init__(
        self,
        V: float,
        *,
        basis: str = "helical",
        n_modes: int | None = None,
        single_pol: str | None = None,
        l_max: int = 4,
        m_max: int = 3,
        n_core: int = 601,
        n_clad: int = 1400,
        r_max: float = 20.0,
        n_phi: int = 256,
    ) -> None:
        if basis not in {"helical", "real"}:
            raise ValueError(f"unknown basis {basis!r}")
        if single_pol not in {None, "+", "-"}:
            raise ValueError(f"unknown single_pol {single_pol!r}")
        self.V = float(V)
        self.basis = basis
        self.n_phi = int(n_phi)
        self.r = _radial_grid(n_core, n_clad, r_max)
        self.phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
        self.radial_weights = _trapezoid_weights(self.r)
        self.phi_weights = np.full(n_phi, 2.0 * np.pi / n_phi)

        roots = lp_roots(V, l_max=l_max, m_max=m_max)
        order = sorted(roots.items(), key=lambda kv_: (kv_[1], kv_[0][0], kv_[0][1]))

        modes: list[LPMode] = []
        for (ell, m_radial), u in order:
            w = float(np.sqrt(V * V - u * u))
            core = jv(ell, u * self.r) / jv(ell, u)
            clad = kv(ell, w * self.r) / kv(ell, w)
            R = np.where(self.r <= 1.0, core, clad)
            # unit-power normalisation, N_p = INT |F_p|^2 dA = 1 (the paper's
            # N_k, so Eq. (7)'s prefactor 1/(N_p N_l N_m N_n) is 1)
            norm = float(2.0 * np.pi * np.sum(self.radial_weights * R * R * self.r))
            R = R / np.sqrt(norm)
            power = 1.0
            labels = [0] if ell == 0 else [-ell, +ell]
            for label in labels:
                for sigma in ("+", "-"):
                    if single_pol is not None and sigma != single_pol:
                        continue
                    modes.append(
                        LPMode(
                            ell=ell,
                            m_radial=m_radial,
                            label=label,
                            sigma=sigma,
                            u=float(u),
                            w=w,
                            R=R,
                            power=power,
                        )
                    )

        if n_modes is not None:
            modes = modes[:n_modes]
        self.modes = modes
        self.labels = [mo.label for mo in modes]
        self.sigmas = [mo.sigma for mo in modes]
        self.angular = angular_profile(self.labels, self.phi, basis)

    def __len__(self) -> int:
        return len(self.modes)

    def unit_power_error(self) -> float:
        """Max ``|INT |F_p|^2 dA - 1|`` over the mode set."""
        return max(
            abs(
                2.0 * np.pi * float(np.sum(self.radial_weights * mo.R * mo.R * self.r))
                - 1.0
            )
            for mo in self.modes
        )


# --------------------------------------------------------------------------
# 3. Eq. (7) — the overlap tensors
# --------------------------------------------------------------------------
def _radial_tensor(fiber: StepIndexFiber) -> NDArray:
    """``INT R_p R_l R_m R_n dA`` for unit-power radial profiles."""
    R = np.array([mo.R for mo in fiber.modes])
    wr = fiber.radial_weights * fiber.r
    return np.einsum("ai,bi,ci,di,i->abcd", R, R, R, R, 2.0 * np.pi * wr, optimize=True)


def _angular_tensor(fiber: StepIndexFiber, typ: int) -> NDArray:
    """Azimuthal factor of Eq. (7) for one coefficient type.

    ``typ = 1``: ``[F_p* . F_l][F_m . F_n*]`` -> ``a_p* a_l a_m a_n*``
    ``typ = 2``: ``[F_p* . F_l*][F_m . F_n]``  -> ``a_p* a_l* a_m a_n``
    """
    a = fiber.angular
    dphi = fiber.phi_weights
    if typ == 1:
        return np.einsum(
            "ai,bi,ci,di,i->abcd",
            np.conj(a),
            a,
            a,
            np.conj(a),
            dphi,
            optimize=True,
        )
    if typ == 2:
        return np.einsum(
            "ai,bi,ci,di,i->abcd",
            np.conj(a),
            np.conj(a),
            a,
            a,
            dphi,
            optimize=True,
        )
    raise ValueError(f"typ must be 1 or 2, got {typ!r}")


def _polarisation_mask(fiber: StepIndexFiber) -> NDArray:
    """Eq. (19) mask: ``e_{sigma_p}* . e_{sigma_l}`` x ``e_{sigma_m} . e_{sigma_n}``."""
    s = np.array(fiber.sigmas)
    d = (s[:, None] == s[None, :]).astype(float)  # (p, l)
    return np.einsum("pl,mn->plmn", d, d, optimize=True)


def overlap_tensor(
    fiber: StepIndexFiber, typ: int, *, apply_pol: bool = True
) -> NDArray:
    """Normalised ``Q^(typ)`` tensor of Eq. (7) for ``fiber``'s mode set.

    The prefactor ``C = eps0 n0^2 c^2 / 12`` and the mode-power constants are
    dropped: the profiles are unit-power (``N_k = 1``) and the result is
    returned divided by ``Q^(typ)_{0000}``, the self-coupling of the
    fundamental mode.  That fixes the arbitrary overall scale of Eq. (7) and
    makes the tensor directly comparable across the two bases and types.

    With ``apply_pol=False`` the polarisation mask Eq. (19) is *not* applied,
    which is what the spatial-rule check of Eq. (18) needs: it must be able to
    see the quadruples the weakly-guiding polarisation rule also forbids.
    """
    q = _radial_tensor(fiber) * _angular_tensor(fiber, typ)
    if apply_pol:
        q = q * _polarisation_mask(fiber)
    q0 = q[0, 0, 0, 0]
    if abs(q0) < 1e-300:  # pragma: no cover - defensive
        raise ValueError("self-coupling Q[0,0,0,0] vanished; check the mode set")
    return q / q0


# --------------------------------------------------------------------------
# 4. Selection rules (Eq. 18 / Eq. 19) — predicted *and* measured
# --------------------------------------------------------------------------
def selection_rule_masks(fiber: StepIndexFiber) -> dict[str, NDArray]:
    """Boolean masks (True = allowed by the analytic rule) for Eq. (18)/(19).

    ``spatial_1``  : ``-m_p + m_l + m_m - m_n = 0``   (Eq. 18, type 1)
    ``spatial_2``  : ``-m_p - m_l + m_m + m_n = 0``   (Eq. 18, type 2)
    ``pol_1`` / ``pol_2``: ``sigma_p = sigma_l`` and ``sigma_m = sigma_n``
    ``fully_1`` / ``fully_2``: spatial **and** polarisation rule
    """
    m = np.array(fiber.labels, dtype=int)  # signed label; the Eq. (18) rules
    s = np.array(fiber.sigmas)  # are written for the helical exp(i m phi) form
    p_ = m[:, None, None, None]
    l_ = m[None, :, None, None]
    m_ = m[None, None, :, None]
    n_ = m[None, None, None, :]
    same = s[:, None] == s[None, :]
    pol = np.einsum("ab,cd->abcd", same, same)
    return {
        "spatial_1": (-p_ + l_ + m_ - n_) == 0,
        "spatial_2": (-p_ - l_ + m_ + n_) == 0,
        "pol_1": pol,
        "pol_2": pol,
    }


def expected_survivors(fiber: StepIndexFiber, typ: int) -> dict[str, int]:
    """Non-zero entry counts predicted by the analytic rules alone.

    Counts the mode quadruples admitted by Eq. (18) (spatial only) and by
    Eq. (18) + Eq. (19) (the weakly-guiding set, the dash-dotted curve of the
    paper's Fig. 1).
    """
    masks = selection_rule_masks(fiber)
    n = len(fiber)
    spatial = masks[f"spatial_{typ}"]
    pol = masks[f"pol_{typ}"]
    return {
        "n_modes": n,
        "total": n**4,
        "spatial": int(spatial.sum()),
        "spatial_and_pol": int((spatial & pol).sum()),
    }


def measure_selection_rules(fiber: StepIndexFiber, tensors: dict[int, NDArray]) -> dict:
    """Check the emergent zeros of Eq. (18)/(19) against the analytic rules.

    ``tensors`` must be the *polarisation-unmasked* tensors
    (``overlap_tensor(..., apply_pol=False)``): the spatial rule Eq. (18) has
    to be able to see the quadruples that the weakly-guiding polarisation rule
    Eq. (19) forbids as well.  ``n_dropped_by_pol_rule`` and
    ``max_abs_dropped_by_pol_rule`` describe that second population, i.e. how
    much coupling the polarisation rule actually discards on top of the spatial
    rule.  Applying the Eq. (19) mask is the caller's step
    (:func:`overlap_tensor` with ``apply_pol=True``).
    """
    masks = selection_rule_masks(fiber)
    out: dict[str, object] = {}
    for typ, q in tensors.items():
        m_spatial = masks[f"spatial_{typ}"]
        m_pol = masks[f"pol_{typ}"]
        num = np.abs(q) > 1e-12
        dropped = m_spatial & ~m_pol
        exp_full = expected_survivors(fiber, typ)["spatial_and_pol"]
        out[f"type_{typ}"] = {
            "n_total": int(q.size),
            "n_surviving_numeric": int(num.sum()),
            "expected_spatial": expected_survivors(fiber, typ)["spatial"],
            "expected_spatial_and_pol": exp_full,
            "max_abs_violating_spatial": float(np.abs(q[~m_spatial]).max()),
            "matched_spatial": bool(
                num.sum() == expected_survivors(fiber, typ)["spatial"]
            ),
            "n_dropped_by_pol_rule": int(dropped.sum()),
            "max_abs_dropped_by_pol_rule": float(np.abs(q[dropped]).max()),
        }
    return out


# --------------------------------------------------------------------------
# 5. Eq. (16) — the exact symmetry identities
# --------------------------------------------------------------------------
def _permute(q: NDArray, order: tuple[int, int, int, int]) -> NDArray:
    """``q`` re-indexed by ``order`` (a permutation of ``(0, 1, 2, 3)``)."""
    return q.transpose(order).copy()  # type: ignore[return-value]


def symmetry_identity_errors(tensors: dict[int, NDArray]) -> list[dict[str, float]]:
    """Max ``|LHS - RHS|`` for the Eq. (16) identities of each type.

    As printed in the paper:

    ``Q^(1)``  ``Q_plmn = Q_nmlp = Q_lpnm* = Q_mnpl* = Q_nmlp*``
    ``Q^(2)``  ``Q_plmn = Q_lpmn = Q_plnm = Q_lpnm = Q_mnpl* = Q_mnlp*
               = Q_nmpl* = Q_nmlp*``
    """
    ident = {
        1: [
            ("Q1: Qplmn = Qnmlp", (3, 2, 1, 0), False),
            ("Q1: Qplmn = Qlpnm*", (1, 0, 3, 2), True),
            ("Q1: Qplmn = Qmnpl*", (2, 3, 0, 1), True),
            ("Q1: Qplmn = Qnmlp*", (3, 2, 1, 0), True),
        ],
        2: [
            ("Q2: Qplmn = Qlpmn", (1, 0, 2, 3), False),
            ("Q2: Qplmn = Qplnm", (0, 1, 3, 2), False),
            ("Q2: Qplmn = Qlpnm", (1, 0, 3, 2), False),
            ("Q2: Qplmn = Qmnpl*", (2, 3, 0, 1), True),
            ("Q2: Qplmn = Qmnlp*", (2, 3, 1, 0), True),
            ("Q2: Qplmn = Qnmpl*", (3, 2, 0, 1), True),
            ("Q2: Qplmn = Qnmlp*", (3, 2, 1, 0), True),
        ],
    }
    out: list[dict[str, float]] = []
    for typ, rules in ident.items():
        q = tensors[typ]
        for name, order, conj in rules:
            rhs = _permute(q, order)
            if conj:
                rhs = np.conj(rhs)
            out.append({"identity": name, "max_err": float(np.abs(q - rhs).max())})
    return out


def real_basis_equality(he: NDArray, re_: NDArray) -> dict[str, float]:
    """``Q^(1) = Q^(2)`` — asserted for the real basis (paper Sec. 3)."""
    return {
        "max_abs_diff": float(np.abs(he - re_).max()),
        "max_rel_diff": float(np.abs(he - re_).max() / max(np.abs(he).max(), 1e-300)),
    }


# --------------------------------------------------------------------------
# 6. The paper's 10-mode set and the Fig. 1 distribution
# --------------------------------------------------------------------------
def mode_set_10(V: float, **kwargs) -> StepIndexFiber:
    """The first ten modes of the paper's step-index fiber (V = 4.27).

    At ``V = 4.2726`` exactly three LP families are guided — LP01, LP11 and
    LP21 — carrying ``2 + 4 + 4 = 10`` modes in the weakly-guiding
    (circularly polarised) description, i.e. ``2 x 10^4`` coefficients for the
    two types of Eq. (7).  This is the set behind Fig. 1 of the paper.
    """
    kwargs.setdefault("n_modes", 10)
    return StepIndexFiber(V, **kwargs)


def magnitude_distribution(
    q: NDArray, *, n_bins: int = 200, floor: float = 1e-12
) -> dict[str, NDArray | float]:
    """Histogram of ``|Q|`` on a log axis, plus the Fig. 1 threshold counts.

    The paper reads the number of "small but non-zero" coefficients off the
    ``|Q| < 0.01 max`` part of the distribution; those counts are returned
    explicitly so the reproduction can be compared with Fig. 1 numerically.
    """
    a = np.abs(q).ravel()
    amax = float(a.max())
    nonzero = a[a > floor]
    edges = np.logspace(np.log10(max(nonzero.min(), 1e-16)), np.log10(amax), n_bins + 1)
    hist, _ = np.histogram(nonzero, bins=edges)
    centres = np.sqrt(edges[:-1] * edges[1:])
    return {
        "centres": centres,
        "counts": hist.astype(float),
        "max": amax,
        "n_total": int(a.size),
        "n_zero": int((a <= floor).sum()),
        "n_below_1pct": int((a < 0.01 * amax).sum()),
        "n_below_1pct_nonzero": int(((a > floor) & (a < 0.01 * amax)).sum()),
    }


# --------------------------------------------------------------------------
# 7. Mapping Eq. (7) onto the library engine
# --------------------------------------------------------------------------
def engine_weights(fiber: StepIndexFiber) -> dict[str, NDArray | list[int]]:
    """The ``MultimodeSplitStepEngine`` coefficient slots read off Eq. (7).

    ``xpm_weights[i, j]``
        ``Q^(1)_{i i j j}`` — the instantaneous Kerr coefficient of the
        intensity of channel ``j`` acting on channel ``i`` (diagonal = SPM).
    ``fwm_weights``
        ``Q^(2)_{m n n q}`` — the degenerate (pump-driven) slice of Eq. (7)
        that the engine's ``A_n^2 A_q*`` FWM term implements; the non-degenerate
        arms ``l != p`` of Eq. (6) are outside the engine's channel geometry.
    ``oam_l``
        the azimuthal labels.  Note that the engine's angular-momentum gate
        ``l_m = l_n + l_p - l_q`` is the *isotropic photon-momentum* rule
        ``l_m + l_q = 2 l_n``, which on the degenerate slice is a strict
        **superset** of Eq. (18) — see ``check_engine`` for the measured count.
        The paper's tensor stays the authoritative gate because it is fed in
        through ``fwm_weights``.

    The engine stores both coefficient slots as real matrices, so a complex
    ``Q`` would lose its phase there; in the real LP basis ``Q^(1) = Q^(2)`` is
    real by construction, which is why the deck drives the engine from that
    basis.  The complex structure of the helical Eq. (17) basis (the
    conjugation identities of Eq. (16)) is exercised on the tensors directly.

    The tensors come from the real LP basis, in which ``Q^(1) = Q^(2)``
    (paper Sec. 3) and the polarisation rules Eq. (19) hold.
    """
    q1 = overlap_tensor(fiber, 1)
    q2 = overlap_tensor(fiber, 2)
    n = len(fiber)
    idx = np.arange(n)
    xpm = q1[idx[:, None], idx[:, None], idx[None, :], idx[None, :]]
    fwm = q2.copy()
    return {
        "xpm_weights": np.real_if_close(xpm).astype(float),
        "fwm_weights": np.real_if_close(fwm).astype(complex),
        "oam_l": list(fiber.labels),
    }


# --------------------------------------------------------------------------
# 8. Sec. 5 — computational complexity of the nonzero coefficients
# --------------------------------------------------------------------------
def complexity_sweep(
    v_values: list[float], n_mode_list: list[int], *, basis: str = "real"
) -> dict[str, NDArray]:
    """Nonzero-coefficient count and cost reduction as a function of ``M``.

    Sec. 5 of the paper: the three-term product of Eq. (6) "scales as
    ``O(M^4 N)`` if no constraints on the coupling coefficients are imposed",
    and the selection rules "reduce the number of three-term multiplications".
    This sweep measures the surviving coefficient count per ``M`` and the
    reduction factor ``M^4 / n_nonzero``.
    """
    m_list: list[int] = []
    n_nonzero: list[int] = []
    n_all: list[int] = []
    for v, m in zip(v_values, n_mode_list, strict=True):
        fib = StepIndexFiber(
            v, basis=basis, n_modes=m, l_max=8, m_max=4, n_core=401, n_clad=900
        )
        if len(fib) != m:  # pragma: no cover - guarded by the caller
            raise ValueError(
                f"V={v} only supports {len(fib)} modes, asked for {m}; "
                "raise the V value or lower n_modes"
            )
        q = np.abs(overlap_tensor(fib, 1))
        m_list.append(m)
        n_nonzero.append(int((q > 1e-12).sum()))
        n_all.append(2 * m**4)
    m_arr = np.array(m_list, dtype=float)
    nz = np.array(n_nonzero, dtype=float)
    allc = np.array(n_all, dtype=float)
    return {
        "M": m_arr,
        "n_nonzero": nz,
        "n_all": allc,
        "reduction_factor": allc / nz,
        "fit_exponent": float(np.polyfit(np.log(m_arr), np.log(nz), 1)[0]),
    }
