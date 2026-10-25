"""Generalized nonlinear Schrödinger equation (GNLSE) solver.

Split-step Fourier solver for pulse propagation through nonlinear media,
supporting Kerr, Raman, self-steepening, and two-photon absorption effects.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from math import factorial
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from photonics_helper.base import C_MS, AngularFrequency, Area, Length, Time

# Accepted unit strings for dispersion coefficients passed to the solvers.
BetasUnit = Literal["ps^k/m", "s^k/m", "SI"]
_BETAS_UNITS: tuple[str, ...] = ("ps^k/m", "s^k/m", "SI")

# Bandwidth limit of the GNLS derivation (Blow & Wood 1989: the GNLS assumes
# the envelope bandwidth is less than ω₀/3). Exceeding it warns; it does not
# raise, because the shock symbol ω/ω₀ = 1 + Ω·τ_shock stays exact and
# well-defined up to the harder Ω_max < ω₀ positivity condition checked in
# `SplitStepEngine._validate_shock_grid`. Renamed in spirit 2026-10-25: this
# is not a Taylor-truncation limit — (1+Ωτ) IS ω/ω₀ bin-by-bin — it guards the
# small-bandwidth regime the equation was derived in.
_SHOCK_TAYLOR_LIMIT = 0.2

# Suzuki triple-jump weights for a 4th-order symmetric composition of Strang
# steps:  S2(w0*dz) S2(w1*dz) S2(w0*dz).
#
# Yoshida (1990), Phys. Lett. A 150(5-7), 262, Eqs. (4.1), (4.5), (4.6):
#   consistency      2*w0 + w1 = 1
#   cubic cancellation 2*w0**3 + w1**3 = 0
# solved by w0 = 1/(2 - 2**(1/3)), w1 = -2**(1/3)/(2 - 2**(1/3)).
#
# NOTE ON NAMING: Yoshida writes the same composition with his w1 as the OUTER
# weight and his w0 as the middle, i.e. the transpose of the weights below.
# The order conditions above are stated in THIS file's convention (w0 outer).
# Both residues were checked numerically (2e-16, 9e-16); the transposed
# condition w0**3 + 2*w1**3 = 0 evaluates to about -7.4 and is NOT satisfied.
YOSHIDA4_W0 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
YOSHIDA4_W1 = -(2.0 ** (1.0 / 3.0)) / (2.0 - 2.0 ** (1.0 / 3.0))
YOSHIDA4_WEIGHTS = (YOSHIDA4_W0, YOSHIDA4_W1, YOSHIDA4_W0)

#: Number of contour points for the Kassam–Trefethen complex means used by
#: the DC-reconciliation test (their §4 diagonal examples use M = 32).
ETDRK4_M = 48

#: Below this ``|L·dz|`` the ETDRK4 coefficients switch from the defining
#: formulas to their Taylor series (Cox & Matthews' own recommendation when
#: ``|λh|`` is small: "the Taylor series for the coefficients should be used
#: rather than the explicit formulae themselves"). At ``|x| ≥ 0.5`` the
#: cancellation in the formulas is ~machine-epsilon level.
ETDRK4_SMALL = 0.5


def _etdrk4_scalar_coeffs(
    x: NDArray, h: float
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """ETDRK4 coefficients ``(Q, f1, f2, f3)`` per bin (Cox & Matthews 2002).

    Evaluated exactly for each ``x = λ·h`` per bin: the defining formulas

    ``Q  = h·(e^{x/2} − 1)/x``
    ``f1 = h·(−4 − x + e^x(4 − 3x + x²))/x³``
    ``f2 = h·(2 + x + e^x(x − 2))/x³``
    ``f3 = h·(−4 − 3x − x² + e^x(4 − x))/x³``

    for ``|x| ≥ 0.5``, and their Taylor series (order ``x⁸``, truncation well
    below round-off there) for ``|x|`` below that — the DC-bin failure mode
    both Cox & Matthews and Kassam–Trefethen identify. The Taylor coefficients
    are:

    ``Q  = 1/2 + x/8 + x²/48 + x³/384 + x⁴/3840 + x⁵/46080 + x⁶/645120 + …``
    ``f1 = 1/6 + x/6 + 3x²/40 + x³/45 + 5x⁴/1008 + x⁵/1120 + …``
    ``f2 = 1/6 + x/12 + x²/40 + x³/180 + x⁴/1008 + …``
    ``f3 = 1/6 − x²/120 − x³/360 − x⁴/1680 − …``
    """
    x = np.asarray(x, dtype=complex)
    small = np.abs(x) < ETDRK4_SMALL
    xs = x[small]

    Q = np.empty_like(x)
    f1 = np.empty_like(x)
    f2 = np.empty_like(x)
    f3 = np.empty_like(x)

    if xs.size:
        x1 = xs
        x2 = x1 * x1
        x3 = x2 * x1
        x4 = x3 * x1
        x5 = x4 * x1
        x6 = x5 * x1
        x7 = x6 * x1
        x8 = x7 * x1
        Q[small] = (
            0.5
            + x1 / 8.0
            + x2 / 48.0
            + x3 / 384.0
            + x4 / 3840.0
            + x5 / 46080.0
            + x6 / 645120.0
            + x7 / 10321920.0
            + x8 / 185794560.0
        )
        f1[small] = (
            1.0 / 6.0
            + x1 / 6.0
            + 3.0 * x2 / 40.0
            + x3 / 45.0
            + 5.0 * x4 / 1008.0
            + x5 / 1120.0
            + 7.0 * x6 / 51840.0
            + x7 / 56700.0
            + x8 / 492800.0
        )
        f2[small] = (
            1.0 / 6.0
            + x1 / 12.0
            + x2 / 40.0
            + x3 / 180.0
            + x4 / 1008.0
            + x5 / 6720.0
            + x6 / 51840.0
            + x7 / 453600.0
            + x8 / 4435200.0
        )
        f3[small] = (
            1.0 / 6.0
            - x2 / 120.0
            - x3 / 360.0
            - x4 / 1680.0
            - x5 / 10080.0
            - x6 / 72576.0
            - x7 / 604800.0
            - x8 / 5702400.0
        )

    big = ~small
    xb = x[big]
    if xb.size:
        exb = np.exp(xb)
        Q[big] = (np.exp(xb * 0.5) - 1.0) / xb
        f1[big] = (-4.0 - xb + exb * (4.0 - 3.0 * xb + xb**2)) / xb**3
        f2[big] = (2.0 + xb + exb * (-2.0 + xb)) / xb**3
        f3[big] = (-4.0 - 3.0 * xb - xb**2 + exb * (4.0 - xb)) / xb**3

    return Q * h, f1 * h, f2 * h, f3 * h


#: Valid ``scheme`` values for :class:`SplitStepEngine`.
Scheme: TypeAlias = Literal["strang", "yoshida4", "etdrk4", "rk4ip"]

_SCHEMES = ("strang", "yoshida4", "etdrk4", "rk4ip")


def _validate_betas_unit(betas_unit: str) -> str:
    """Return *betas_unit* if it is an accepted unit string, else raise."""
    if betas_unit not in _BETAS_UNITS:
        raise ValueError(
            f"Unknown betas_unit {betas_unit!r}; accepted values are "
            f"{', '.join(repr(u) for u in _BETAS_UNITS)}."
        )
    return betas_unit


def _full_beta_phase(full_beta, grid, omega0: float) -> tuple[NDArray, NDArray, float]:
    """Validate a static ``β(ω)`` curve and build the retarded-frame phase.

    Mirrors the multimode engine's coercion (``_mm_coerce_full_beta``):
    ``full_beta`` is a callable ``β(ω_abs)`` or an array sampled on the
    grid's ``Ω`` axis.  The phase returned is

        φ(Ω) = [β(ω₀+Ω) − β(ω₀)] − β₁(ω₀)·Ω

    with a 4th-order centre slope and a cubic centre value.  Nearest-sample
    estimates leave O(0.1) rad/m at a broad band edge, which would break the
    narrowband agreement contract with an order-10 Taylor expansion.

    Returns ``(curve, phase, beta2_si)`` where ``beta2_si`` is
    ``d²β/dω²`` at the carrier in s²/m, used only by the adaptive step
    heuristic.
    """
    w = np.asarray(grid.w, dtype=float)
    w0 = float(omega0)
    if callable(full_beta):
        curve = np.asarray(full_beta(w + w0), dtype=float)
    else:
        curve = np.asarray(full_beta, dtype=float)
    if curve.shape != w.shape:
        raise ValueError(
            "full_beta must match the grid "
            f"({w.shape}), got {curve.shape}; pass a callable β(ω_abs) or "
            "an array sampled on grid.w."
        )
    if not np.all(np.isfinite(curve)):
        raise ValueError("full_beta contains non-finite values.")

    i0 = int(np.argmin(np.abs(w)))
    if w.shape[0] < 5:
        raise ValueError(
            "full_beta needs at least 5 grid samples for the centre "
            f"β₀/β₁ estimates, got {w.shape[0]}."
        )
    lo = max(i0 - 2, 0)
    if lo + 5 > w.shape[0]:  # near the top edge
        lo = w.shape[0] - 5
    seg_w = w[lo : lo + 5]
    seg_b = curve[lo : lo + 5]
    # Cubic Lagrange value at Ω = 0.
    b_at_0 = 0.0
    for j in range(5):
        lj = 1.0
        for k in range(5):
            if k != j:
                lj *= (0.0 - seg_w[k]) / (seg_w[j] - seg_w[k])
        b_at_0 += seg_b[j] * lj
    dw = float(w[1] - w[0])
    ci = i0 - lo
    # 4th-order central slope at the centre sample.
    dB0 = float(
        (-seg_b[ci + 2] + 8.0 * seg_b[ci + 1] - 8.0 * seg_b[ci - 1] + seg_b[ci - 2])
        / (12.0 * dw)
    )
    # β₂ at the carrier for the adaptive dispersion-length estimate.
    beta2_si = float((seg_b[ci + 1] - 2.0 * seg_b[ci] + seg_b[ci - 1]) / dw**2)
    phase = curve - float(b_at_0) - dB0 * w
    return curve, phase, beta2_si


def _normalize_betas(betas, betas_unit: BetasUnit = "ps^k/m") -> NDArray:
    """Validate and normalise Taylor dispersion coefficients to ``ps^k/m``.

    The GNLSE dispersion operator uses the Taylor expansion
    ``β(ω) = Σ_k β_k (ω−ω₀)^k / k!`` with ``β_k = d^kβ/dω^k`` (units
    ``s^k/m``; Agrawal, *Nonlinear Fiber Optics*, 5th ed., §2.3.1). The solver
    evaluates this with ``Ω`` in ``rad/ps``, so coefficients are stored in
    ``ps^k/m``. Element ``i`` of *betas* is order ``k = i + 2``
    (beta2, beta3, ...). Because ``Ω_ps = 10^{−12} Ω_SI``, matching the two
    forms of ``β_k Ω^k/k!`` gives the per-order conversion

        β_k[ps^k/m] = β_k[s^k/m] · 10^{12k}

    (for ``k = 2`` this is the familiar ``1 ps²/m = 10^{−24} s²/m`` used by
    :meth:`~photonics_helper.fiber.Dispersion.get_betas`).
    """
    _validate_betas_unit(betas_unit)
    try:
        arr = np.asarray(betas, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"betas must be a real numeric array; got {type(betas).__name__} ({exc})."
        ) from exc
    if arr.ndim != 1:
        raise ValueError(f"betas must be 1-D, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        bad = np.where(~np.isfinite(arr))[0].tolist()
        raise ValueError(f"betas contains non-finite values at index/indices {bad}.")
    if betas_unit == "ps^k/m":
        return arr
    out = arr.copy()
    for i in range(out.shape[0]):
        k = i + 2
        out[i] = out[i] * 10.0 ** (12 * k)
    return out


if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib import pyplot as plt
    from matplotlib.figure import Figure as MplFigure
    from plotly.graph_objects import Figure as PlotlyFigure

    from photonics_helper.fiber import ZDependentDispersion
    from photonics_helper.pulse import TemporalGrid, Wave

    from .phase_matching import SimulationReadinessReport

    # Plotting helpers return a matplotlib figure by default and a plotly figure
    # when ``plotly=True``; plotly is an optional dependency.
    FigureLike: TypeAlias = MplFigure | PlotlyFigure

__all__ = [
    "ConvergenceDiagnostics",
    "DRIFT_WARN_PCT",
    "Fiber",
    "FiberProfile",
    "GNLSESolver",
    "SplitStepEngine",
    "TaperedGNLSESolver",
    "gnlse_spectrogram",
    "plot_intensity_metrics",
    "plot_scg_dashboard",
    "plot_spectral_evolution",
    "plot_spectral_temporal_summary",
    "plot_spectrogram",
    "plot_spectrum_vs_distance",
    "plot_temporal_evolution",
    "plot_waterfall",
    "save_summary_html",
    "spectral_evolution_on_frequency_grid",
    "spectral_evolution_on_wavelength_grid",
    "temporal_evolution_intensity",
]


# ---------------------------------------------------------------------------
# Fiber (formerly FiberProfile)
# ---------------------------------------------------------------------------


def _coerce_omega0(omega0: float | AngularFrequency) -> float:
    """Carrier angular frequency in rad/s from float or :class:`AngularFrequency`."""
    if isinstance(omega0, AngularFrequency):
        return omega0.as_rad_s
    return float(omega0)


@dataclass
class Fiber:
    """Optical fiber/waveguide parameters for GNLSE propagation.

    Preferred name is :class:`Fiber` (``FiberProfile`` is kept as a
    deprecated alias). Either supply ``n2`` + ``A_eff`` (γ derived as
    ``n₂·ω₀·Γ/(c·A_eff)``) or supply ``gamma`` directly — when ``gamma``
    is set the solver uses it and ``n2``/``A_eff`` are only kept for
    bookkeeping, so ``from_gamma`` no longer needs ``n2``.

    Attributes:
        n2: Nonlinear refractive index n₂ (m²/W). Optional when ``gamma`` given.
        alpha: Fiber loss coefficient α (1/m).
        A_eff: Effective mode area. Optional when ``gamma`` is given directly.
        length: Fiber length.
        gamma: Nonlinear coefficient γ in 1/(W·m). When set, takes
            precedence over the ``n2``/``A_eff`` derivation.
        confinement_factor: Waveguide confinement factor Γ (1.0 for fibers, <1.0 for waveguides).
        sigma_tpa: Two-photon absorption cross-section (m²·W⁻¹). Default 0.
        carrier_lifetime: Carrier recombination lifetime (s). Default None.
        raman_response: Raman response function. Default None.
        beta_tpa: Two-photon absorption coefficient β_TPA (m/W) for the
            time-resolved free-carrier model. Default 0.
        sigma_fca: Free-carrier absorption cross-section σ_FCA (m²) for the
            time-resolved free-carrier model. Default 0.
        group_velocity: Envelope group velocity v (m/s) converting the carrier
            lifetime to a propagation length. Default None → C_MS.
        a_eff: Reference effective mode area for the **nonlinear absorption
            coefficients**. When given, ``beta_tpa`` / ``sigma_tpa`` (and the
            engine's ``three_photon_alpha_3``) are interpreted as *physical*
            coefficients in ``m/W`` / ``m²/W`` / ``m³/W²`` and converted to the
            engine's power convention internally (``β/A_eff``, ``σ/A_eff``,
            ``α₃/A_eff²``). When ``None`` (default) every coefficient is taken
            as **already engine-normalised** — ``|A|²`` in watts — exactly as
            before this field existed. Deliberately separate from
            :attr:`A_eff`, which only feeds the ``n₂`` → γ derivation.
    """

    n2: float = 0.0
    alpha: float = 0.0
    A_eff: Area | None = None
    length: Length = field(default_factory=lambda: Length(1.0, "m"))
    gamma: float | None = None
    confinement_factor: float = 1.0
    sigma_tpa: float = 0.0
    carrier_lifetime: Time | None = None
    raman_response: object | None = None
    beta_tpa: float = 0.0
    sigma_fca: float = 0.0
    group_velocity: float | None = None
    #: Opt-in area hook: when set, ``beta_tpa``/``sigma_tpa`` (and the engine's
    #: ``three_photon_alpha_3``) are *physical* coefficients and are divided by
    #: ``a_eff**1`` / ``a_eff**2`` internally. ``None`` → engine-normalised.
    a_eff: Area | None = None

    def _nl_area_factor(self, power: int) -> float:
        """``1/a_eff**power``, or exactly ``1.0`` when no area hook is set."""
        if self.a_eff is None:
            return 1.0
        a = self.a_eff.as_m2
        if a <= 0:
            raise ValueError(f"Fiber.a_eff must be positive, got {a!r} m^2")
        return a ** -power

    @property
    def beta_tpa_engine(self) -> float:
        """β_TPA in the engine convention (W⁻¹m⁻¹): ``beta_tpa/a_eff``.

        ``a_eff=None`` returns ``beta_tpa`` unchanged (bit-identical), i.e. the
        stored value is already ``β_TPA/A_eff``.
        """
        return self.beta_tpa * self._nl_area_factor(1)

    @property
    def sigma_tpa_engine(self) -> float:
        """σ_TPA in the engine convention: ``sigma_tpa/a_eff`` (see
        :attr:`beta_tpa_engine`)."""
        return self.sigma_tpa * self._nl_area_factor(1)

    def effective_gamma(self, omega0: float | AngularFrequency) -> float:
        """Nonlinear coefficient γ in 1/(W·m).

        Returns the stored :attr:`gamma` when set, else derives it from
        ``n₂·ω₀·Γ/(c·A_eff)``. Accepts a raw float (rad/s) or an
        :class:`AngularFrequency`.
        """
        if self.gamma is not None:
            return float(self.gamma)
        if self.A_eff is None:
            raise ValueError(
                "Fiber has neither gamma nor A_eff: supply gamma=... directly "
                "or n2 + A_eff to derive it."
            )
        a_eff: Area = self.A_eff
        return _gamma(self.n2, _coerce_omega0(omega0), a_eff, self.confinement_factor)

    @classmethod
    def from_gamma(
        cls,
        gamma: float,
        n2: float | None = None,
        omega0: float | AngularFrequency | None = None,
        alpha: float = 0.0,
        length: Length | None = None,
        confinement_factor: float = 1.0,
        sigma_tpa: float = 0.0,
        carrier_lifetime: Time | None = None,
        raman_response: object | None = None,
        beta_tpa: float = 0.0,
        sigma_fca: float = 0.0,
        group_velocity: float | None = None,
        A_eff: Area | None = None,
        a_eff: Area | None = None,
    ) -> Fiber:
        """Create a Fiber from a target nonlinear coefficient γ.

        Gamma-first form needs no ``n2``::

            Fiber.from_gamma(0.045, A_eff=Area(80, "um^2"), length=Length(0.1, "m"))
            Fiber.from_gamma(0.045, length=Length(0.1, "m"))  # A_eff unknown, gamma stored

        Legacy form (``n2`` + ``omega0``) still works and back-computes
        ``A_eff = n₂·ω₀·Γ/(c·γ)`` with a deprecation warning. ``omega0``
        accepts a float (rad/s) or :class:`AngularFrequency`.

        Parameters
        ----------
        gamma : float — target nonlinear coefficient γ in 1/(W·m).
        n2 : float | None — nonlinear refractive index n₂ in m²/W.
            Only needed for the legacy A_eff back-computation.
        omega0 : float | AngularFrequency | None — carrier angular frequency.
            Only needed with ``n2`` (legacy path).
        A_eff : Area | None — effective mode area. Preferred with gamma-first.
        alpha : float — fiber loss coefficient α in 1/m. Default 0.
        length : Length — fiber length. Default 1 m.
        confinement_factor : float — waveguide confinement factor Γ. Default 1.0.
        sigma_tpa : float — two-photon absorption cross-section. Default 0.
        carrier_lifetime : Time — carrier recombination lifetime. Default None.
        raman_response : object — Raman response function. Default None.
        beta_tpa : float — two-photon absorption coefficient β_TPA in m/W.
            Used by the time-resolved free-carrier model
            (``SplitStepEngine(include_free_carriers=True)``). Default 0.
        sigma_fca : float — free-carrier absorption cross-section σ_FCA in m².
            Used by the time-resolved free-carrier model. Default 0.
        group_velocity : float | None — envelope group velocity v in m/s used
            to convert the retarded-time carrier lifetime into a propagation
            length. Default ``None`` → the vacuum-light-speed constant.
        a_eff : Area | None — nonlinear-coefficient area hook (see
            :class:`Fiber`): when given, ``beta_tpa``/``sigma_tpa`` are
            *physical* (m/W, m²/W) and are converted to the engine convention
            internally. Default ``None`` → engine-normalised.

        Returns
        -------
        Fiber — with ``gamma`` stored (and ``A_eff`` when known).
        """
        if length is None:
            length = Length(1.0, "m")
        if A_eff is not None:
            return cls(
                n2=n2 if n2 is not None else 0.0,
                alpha=alpha,
                A_eff=A_eff,
                length=length,
                gamma=gamma,
                confinement_factor=confinement_factor,
                sigma_tpa=sigma_tpa,
                carrier_lifetime=carrier_lifetime,
                raman_response=raman_response,
                beta_tpa=beta_tpa,
                sigma_fca=sigma_fca,
                group_velocity=group_velocity,
                a_eff=a_eff,
            )
        if n2 is not None and omega0 is not None:
            warnings.warn(
                "from_gamma(gamma, n2, omega0) back-computes A_eff; prefer "
                "from_gamma(gamma, A_eff=...) or from_gamma(gamma, ...) "
                "which needs no n2.",
                DeprecationWarning,
                stacklevel=2,
            )
            A_eff_m2 = n2 * _coerce_omega0(omega0) * confinement_factor / (C_MS * gamma)
            return cls(
                n2=n2,
                alpha=alpha,
                A_eff=Area(A_eff_m2, "m^2"),
                length=length,
                gamma=gamma,
                confinement_factor=confinement_factor,
                sigma_tpa=sigma_tpa,
                carrier_lifetime=carrier_lifetime,
                raman_response=raman_response,
                beta_tpa=beta_tpa,
                sigma_fca=sigma_fca,
                group_velocity=group_velocity,
                a_eff=a_eff,
            )
        # Gamma-only: no n2, no A_eff needed for propagation.
        return cls(
            n2=n2 if n2 is not None else 0.0,
            alpha=alpha,
            A_eff=None,
            length=length,
            gamma=gamma,
            confinement_factor=confinement_factor,
            sigma_tpa=sigma_tpa,
            carrier_lifetime=carrier_lifetime,
            raman_response=raman_response,
            beta_tpa=beta_tpa,
            sigma_fca=sigma_fca,
            group_velocity=group_velocity,
            a_eff=a_eff,
        )

    @classmethod
    def from_n2(
        cls,
        n2: float,
        A_eff: Area,
        length: Length | None = None,
        alpha: float = 0.0,
        confinement_factor: float = 1.0,
        sigma_tpa: float = 0.0,
        carrier_lifetime: Time | None = None,
        raman_response: object | None = None,
        beta_tpa: float = 0.0,
        sigma_fca: float = 0.0,
        group_velocity: float | None = None,
        a_eff: Area | None = None,
    ) -> Fiber:
        """Create a Fiber from ``n2`` + ``A_eff`` (γ derived at run time)."""
        if length is None:
            length = Length(1.0, "m")
        return cls(
            n2=n2,
            alpha=alpha,
            A_eff=A_eff,
            length=length,
            gamma=None,
            confinement_factor=confinement_factor,
            sigma_tpa=sigma_tpa,
            carrier_lifetime=carrier_lifetime,
            raman_response=raman_response,
            beta_tpa=beta_tpa,
            sigma_fca=sigma_fca,
            group_velocity=group_velocity,
            a_eff=a_eff,
        )


#: Deprecated alias — use :class:`Fiber` instead.
FiberProfile = Fiber


# ---------------------------------------------------------------------------
# Nonlinear effect functions
# ---------------------------------------------------------------------------


# Reduced Planck constant (J·s) — used by the time-resolved free-carrier model.
HBAR_J_S = 1.0545718e-34


def _gamma(
    n2: float,
    omega0: float | AngularFrequency,
    A_eff: Area,
    confinement_factor: float = 1.0,
) -> float:
    """Nonlinear coefficient γ = n₂·ω₀·Γ / (c·A_eff).

    Parameters
    ----------
    n2 : float — Nonlinear refractive index n₂ (m²/W).
    omega0 : float | AngularFrequency — Carrier angular frequency (float in
        rad/s, or unit-bearing :class:`AngularFrequency`).
    A_eff : Area — Effective mode area.
    confinement_factor : float — Waveguide confinement factor Γ (1.0 for fibers).
    """
    return n2 * _coerce_omega0(omega0) * confinement_factor / (C_MS * A_eff.as_m2)


def _fiber_gamma(fiber: Fiber, omega0: float | AngularFrequency) -> float:
    """Resolve the effective γ for *fiber*: stored ``gamma`` wins, else derive."""
    return fiber.effective_gamma(omega0)


def kerr_step(
    A: NDArray,
    fiber: FiberProfile,
    grid: TemporalGrid,
    dz: float,
    omega0: float | AngularFrequency,
) -> NDArray:
    """Apply instantaneous Kerr effect: A ← A · exp(i·γ·|A|²·Δz).

    Parameters
    ----------
    A : complex array — pulse envelope.
    fiber : FiberProfile — fiber parameters.
    grid : TemporalGrid — time grid.
    dz : float — step size (m).
    omega0 : float — carrier frequency (rad/s).

    Returns
    -------
    A : updated complex array.
    """
    gamma = _fiber_gamma(fiber, omega0)
    phase = np.exp(1j * gamma * np.abs(A) ** 2 * dz)
    return np.asarray(A * phase)


def _raman_polarization(
    intensity: NDArray,
    fR: float,
    grid: TemporalGrid,
    h_R_fft: NDArray | None,
) -> NDArray:
    """Compute Raman nonlinear polarization P_NL = (1-fR)|A|² + fR·(h_R ⊗ |A|²).

    Shared helper used by both `raman_step` (standalone) and
    `SplitStepEngine._nl_intensity` (cached engine path).

    Parameters
    ----------
    intensity : |A(t)|² on the simulation grid.
    fR : float — Raman fraction.
    grid : TemporalGrid — time grid for FFT.
    h_R_fft : FFT of h_R(t) on ``grid.t``, or None when Raman is off.

    Returns
    -------
    P_NL : Nonlinear driving intensity for Kerr+Raman phase.
    """
    P_inst = (1.0 - fR) * intensity
    if h_R_fft is not None:
        I_fft = grid.fft(intensity)
        # The physical Raman response is causal and must amplify the Stokes
        # (red) sideband.  Under the post-#0 transform pair (analysis
        # ``e^{+iΩt}``, synthesis ``e^{−iΩt}``), the delayed (non-instantaneous)
        # Raman polarization is the plain convolution with the causal h_R:
        # ``ifft( fft(h_R) · fft(I) )``.  (Before the #0 swap the mirrored
        # backend pair required ``conj`` here; that would now amplify the
        # anti-Stokes band and blue-shift solitons.)  DC unchanged (H(0) = 1).
        P_delayed = fR * np.real(grid.ifft(h_R_fft * I_fft))
    else:
        P_delayed = np.zeros_like(intensity)
    return np.asarray(P_inst + P_delayed)


def _delayed_h_R(response: object, t: NDArray) -> NDArray:
    """Return the un-scaled delayed Raman response ``h_R(t)`` from a response.

    Accepts any object exposing a public ``h_R(t)`` — the single-mode
    :class:`~photonics_helper.raman.RamanResponse` (Blow–Wood / Agrawal form)
    and the multi-mode :class:`~photonics_helper.phonon.PhononResponse`
    (damped-oscillator superposition) — falling back to the legacy private
    ``_h_R`` for backwards compatibility.

    References
    ----------
    Agrawal, *Nonlinear Fiber Optics*, 5th ed., Sec. 2.3.2 (delayed-response
    normalization); Hollenbeck & Cantrell, *J. Opt. Soc. Am. B* **19**, 2886
    (2002) (multi-mode response).
    """
    h_R = getattr(response, "h_R", None)
    if callable(h_R):
        return np.asarray(h_R(t), dtype=float)
    legacy = getattr(response, "_h_R", None)
    if callable(legacy):
        return np.asarray(legacy(t), dtype=float)
    raise TypeError(
        "the fiber's Raman response must provide an h_R(t) method; got "
        f"{type(response).__name__}"
    )


def _response_fR(response: object) -> float:
    """Return ``fR`` from a Raman response, or raise a clear error."""
    fR = getattr(response, "fR", None)
    if fR is None:
        raise ValueError(
            "Raman is enabled but the response object has no fR. Pass "
            "fR=<material fR> (e.g. via RamanSpec.phonon_response) or use "
            "RamanResponse(spec)."
        )
    return float(fR)


def raman_step(
    A: NDArray,
    fiber: FiberProfile,
    grid: TemporalGrid,
    dz: float,
    include_raman: bool,
    omega0: float | AngularFrequency = 0.0,
) -> NDArray:
    """Apply Raman convolution: P_NL = (1-fR)|A|² + fR·(h_R ⊗ |A|²).

    Uses FFT-based circular convolution on the centered ``grid.t`` axis with
    the same ``grid.fft`` / ``grid.ifft`` convention as the split-step engine.

    Parameters
    ----------
    A : complex array — pulse envelope.
    fiber : FiberProfile — fiber parameters.
    grid : TemporalGrid — time grid.
    dz : float — step size (m).
    include_raman : bool — whether to include Raman.
    omega0 : float — carrier angular frequency (rad/s).

    Returns
    -------
    A : updated complex array (with Raman nonlinear polarization added).

    Raises
    ------
    ValueError : if include_raman is True but fiber.raman_response is None.
    """
    if not include_raman:
        return A

    if fiber.raman_response is None:
        raise ValueError(
            "include_raman=True but fiber.raman_response is None. "
            "Provide a RamanResponse or set include_raman=False."
        )

    intensity = np.abs(A) ** 2
    fR = _response_fR(fiber.raman_response)

    # Compute h_R_fft on the fly (no cache available for standalone function)
    h_R = _delayed_h_R(fiber.raman_response, grid.t)  # causal: h_R[t<0] = 0
    h_R_fft = grid.fft(h_R)

    P_Raman = _raman_polarization(intensity, fR, grid, h_R_fft)

    gamma = _fiber_gamma(fiber, omega0)
    raman_phase = np.exp(1j * gamma * P_Raman * dz)
    return np.asarray(A * raman_phase)


def free_carrier_step(
    A: NDArray,
    fiber: FiberProfile,
    grid: TemporalGrid,
    dz: float,
    N: NDArray | None,
    omega0: float | AngularFrequency,
) -> tuple[NDArray, NDArray | None]:
    """Time-resolved TPA / free-carrier absorption step (opt-in).

    Models the per-sample carrier population over the retarded-time grid
    (Soref & Bennett 1987; Cowan, Rieger & Young, *Appl. Phys. Lett.* **82**,
    1745 (2003); Yin & Agrawal, *Opt. Lett.* **32**, 2951 (2007)):

    ``∂A/∂z|_TPA = −(β_TPA/2) |A|² A`` — the standard TPA attenuation;
    in intensity form ``d|A|²/dz = −β_TPA |A|⁴``, solved *exactly* per
    sample for a constant intensity within the step:
    ``|A|²(z+dz) = |A|² / (1 + β_TPA |A|² dz)``.

    ``∂N/∂z = β_TPA |A|⁴/(2ħω₀) − N/(τ_c v)`` — carrier generation by TPA
    and recombination (``v`` the envelope group velocity converts the
    retarded-time lifetime into a propagation length); integrated exactly
    per sample: ``N = N e^{−dz/L} + S·L·(1 − e^{−dz/L})`` with ``L = τ_c v``.

    ``∂A/∂z|_FCA = −(σ_FCA/2) N A`` — free-carrier absorption, applied with
    the mid-step carrier density.

    Out of scope (documented, not implemented): free-carrier refraction (the
    ``μ`` index term), carrier diffusion and drift.

    Parameters
    ----------
    A : complex array — pulse envelope.
    fiber : FiberProfile — carries ``beta_tpa``, ``sigma_fca``,
        ``carrier_lifetime`` and ``group_velocity``.
    grid : TemporalGrid — time grid (defines the sample axis of ``N``).
    dz : float — step size (m).
    N : NDArray | None — per-sample carrier density state from the previous
        steps, on the same grid. ``None`` (start of propagation init).
    omega0 : float — carrier angular frequency (rad/s).

    Returns
    -------
    (A, N) — updated field and carrier states.
    """
    beta = fiber.beta_tpa_engine
    sigma = fiber.sigma_fca
    omega0_f = _coerce_omega0(omega0)
    if beta <= 0 and sigma <= 0:
        return A, N

    if N is None:
        N = np.zeros(grid.N, dtype=float)

    intensity = np.abs(A) ** 2

    # Exact TPA attenuation: dI/dz = −β I²  ⇒  I⁺ = I / (1 + β I dz).
    if beta > 0:
        eps = 1e-300
        denom = 1.0 + beta * intensity * dz
        I_new = intensity / np.maximum(denom, eps)
        A = A * np.sqrt(I_new / np.maximum(intensity, eps))

    # Carrier population: exact integration of dN/dz = S − N/L over the step
    # with S frozen at the step-opening intensity (operator-split convention).
    tau_c = fiber.carrier_lifetime.as_s if fiber.carrier_lifetime is not None else 1e-9
    v = fiber.group_velocity if fiber.group_velocity is not None else C_MS
    L = max(tau_c * v, 1e-300)  # m — carrier recombination length
    decay = np.exp(-dz / L)
    source = beta * intensity**2 / (2.0 * HBAR_J_S * omega0_f) if beta > 0 else 0.0
    N_mid = N * decay + source * L * (1.0 - decay)

    # FCA loss with the mid-step carrier density.
    if sigma > 0:
        A = A * np.exp(-0.5 * sigma * N_mid * dz)

    return A, N_mid


def tpa_step(
    A: NDArray,
    fiber: FiberProfile,
    grid: TemporalGrid,
    dz: float,
    include_tpa: bool,
    U: float = 0.0,
    omega0: float | AngularFrequency = 0.0,
) -> tuple[NDArray, float]:
    """Apply two-photon absorption with carrier dynamics.

    dA/dz = -σ·U·A
    dU/dz = σ·|A|²/(2ħω) - U/τ_c

    Applies ``fiber.sigma_tpa`` only. ``fiber.beta_tpa`` is **not** consumed
    here — β_TPA is carried by :func:`free_carrier_step`, i.e. by
    ``include_free_carriers=True``, not by ``include_tpa``.

    Parameters
    ----------
    A : complex array — pulse envelope.
    fiber : FiberProfile — fiber parameters.
    grid : TemporalGrid — time grid.
    dz : float — step size (m).
    include_tpa : bool — whether to include TPA.
    U : float — current carrier density (W⁻¹·m⁻³). Default 0.
    omega0 : float — carrier angular frequency (rad/s). Required only when the
        step actually acts (``sigma_tpa > 0``).

    Returns
    -------
    A : updated complex array.
    U_new : updated carrier density.

    Raises
    ------
    ValueError : if the step is active (``sigma_tpa > 0``) and omega0 <= 0.
    """
    if not include_tpa:
        return A, U

    sigma = fiber.sigma_tpa_engine
    tau_c = fiber.carrier_lifetime.as_s if fiber.carrier_lifetime is not None else 1e-9

    # Inactive step: no carrier frequency is needed for an operation that
    # does nothing (the check used to precede this return).
    if sigma <= 0:
        return A, U

    omega0_f = _coerce_omega0(omega0)
    if omega0_f <= 0:
        raise ValueError(
            "tpa_step: omega0 (carrier angular frequency in rad/s) is required when include_tpa=True"
        )

    hbar = 1.0545718e-34  # J·s
    hbar_omega = hbar * omega0_f

    # Intensity |A|^2 (W/m² for proper TPA)
    intensity = np.abs(A) ** 2

    # Carrier density update (spatially-averaged, explicit Euler)
    U_avg = np.mean(intensity) * sigma / (2 * hbar_omega) * dz - U / tau_c * dz
    U_new = U + U_avg

    # Field attenuation (using spatially-averaged carrier density)
    attenuation = np.exp(-sigma * max(U_new, 0.0) * dz)
    A_new = A * attenuation

    return A_new, max(U_new, 0.0)  # carrier density can't be negative


# ---------------------------------------------------------------------------
# SplitStepEngine
# ---------------------------------------------------------------------------


class SplitStepEngine:
    """Split-step Fourier engine for GNLSE propagation.

    Alternates linear (dispersion) and nonlinear (Kerr/Raman/steepening/TPA)
    steps in Fourier space with adaptive step sizing.

    Parameters
    ----------
    pulse : Wave
        Input pulse.
    fiber : FiberProfile
        Fiber parameters.
    betas : array_like
        Dispersion coefficients [beta2, beta3, ...] in ``ps^k/m`` (with ``Ω``
        in ``rad/ps``), as returned by ``Dispersion.get_betas()``. Pass
        ``betas_unit="s^k/m"`` (or ``"SI"``) to supply SI coefficients
        instead; they are converted to ``ps^k/m`` internally.
    betas_unit : {"ps^k/m", "s^k/m", "SI"}
        Unit of *betas*. Default ``"ps^k/m"`` (native internal form).
    full_beta : array_like or callable, optional
        Static full ``β(ω)`` curve (rad/m) — a callable ``β(ω_abs)`` or an
        array sampled on the grid's ``Ω=ω−ω₀`` axis. When supplied, *betas*
        is ignored (it may be ``None``) and the linear step applies the
        retarded-frame phase
        ``φ(Ω) = [β(ω₀+Ω)−β(ω₀)] − β₁Ω`` with 4th-order centre
        value/slope. This is the scalar analogue of the multimode
        ``full_beta`` path and avoids the truncated-Taylor error on
        octave-spanning spectra. Mutually exclusive with
        ``dispersion_profile`` (that is the z-dependent path). Supported by
        ``scheme="etdrk4"`` as well.
    raman_response : object, optional
        Raman response override (any object exposing ``h_R(t)`` and ``fR``;
        :class:`~photonics_helper.raman.RamanResponse` or
        :class:`~photonics_helper.phonon.PhononResponse`). Falls back to
        ``fiber.raman_response`` when omitted.
    include_raman : bool
        Include Raman. Default False.
    include_self_steepening : bool
        Include self-steepening (shock term). Default ``False``.

        When enabled, the shock operator ``(1 + (i/ω₀)∂t)`` of Blow & Wood
        (1989) and Dudley–Genty–Coen RMP 78, 1135 (2006), Eq. (3), is
        integrated with an interaction-picture scheme (``τ_shock`` settable,
        default ``1/ω₀``): the exactly-integrable Kerr/Raman phase
        ``exp(iγP_NL Δz)`` is factored out and only the shock correction
        ``iγτ_shock ∂_t(A·P_NL)`` is advanced with frequency-domain RK4 (the
        RK4IP idea of Hult 2007 / Hochbruck & Ostermann 2010). The shock
        factor is never clamped (see :meth:`_validate_shock_grid`). Enable
        explicitly for cross-library comparisons. laserfun defaults to shock
        on and uses the same physical factor ``ω/ω₀ = 1 + Ω/ω₀``
        (``laserfun/laserfun/nlse.py`` sets ``gamma = gamma/ω₀`` and
        ``w = Ω + ω₀``); the two libraries agree on the shock sign once both
        are run at a converged step size.

        Sign audit (2026): the multiplier ``1 + Ω·τ_shock`` is the correct
        physical factor ``ω/ω₀`` under this codebase's FFT convention. The
        grid FFT uses the numpy ``e^{−iΩt}`` kernel while the linear
        dispersion step, all spectral plots, and the DW/MI utilities map bin
        ``W`` to optical frequency ``ω₀ + W``; that fixes the carrier
        convention to ``e^{+iω₀t}`` and makes ``W > 0`` the blue side. The
        convention set was verified numerically to be self-consistent:
        (i) Raman-only soliton propagation red-shifts under the same mapping
        (Gordon SSFS), (ii) a fundamental soliton with shock and β₂ only
        develops a blue-shifted centroid and positive spectral skew with
        ``1 + Ω·τ`` and the mirror-image red shift with ``1 − Ω·τ``, and
        (iii) energy is conserved.
    include_tpa : bool
        Include TPA (legacy spatially-averaged carrier model). Default False.
        This flag applies ``fiber.sigma_tpa`` only; ``fiber.beta_tpa`` is
        carried by ``include_free_carriers`` (see below), so a fiber with
        ``beta_tpa > 0`` and ``sigma_tpa == 0`` warns instead of silently
        doing nothing.
    include_free_carriers : bool
        Include the opt-in time-resolved TPA / free-carrier model: the carrier
        population ``N(t)`` is resolved over the retarded-time grid instead of
        spatially averaged, driven by ``β_TPA |A|⁴/(2ħω₀)`` generation and
        ``N/(τ_c v)`` recombination, with the field attenuated by the TPA term
        ``−(β_TPA/2)|A|²A`` and free-carrier absorption ``−(σ_FCA/2)N·A``
        (Soref & Bennett 1987; Cowan et al. 2003; Yin & Agrawal 2007). Requires
        ``fiber.beta_tpa`` / ``fiber.sigma_fca``; free-carrier refraction (μ),
        diffusion and drift are out of scope. Default False.

    Coefficient units
        The engine takes ``|A|²`` in **watts**, so ``fiber.beta_tpa`` and
        ``fiber.sigma_tpa`` must be normalised values ``β_TPA/A_eff`` and
        ``σ_TPA/A_eff`` and ``three_photon_alpha_3`` must be ``α₃/A_eff²``.
        Pass a *physical* coefficient (m/W, m²/W, m³/W²) together with
        ``Fiber(a_eff=Area(...))`` and the conversion happens internally;
        without ``a_eff`` the stored numbers are used as they are. Field
        amplitudes follow the same rule through ``Wave.to_engine_units``.
    tau_shock : float | None
        Shock (self-steepening) timescale in seconds (SI). ``None``
        (default) uses ``1/ω₀`` at the pulse carrier frequency, matching
        Agrawal and the uncorrected Dudley Eq. (3) value (0.443 fs at
        835 nm). Pass an explicit value for the effective-area-corrected
        timescale (Dudley–Genty–Coen RMP 78, 1135 (2006), Sec. V.B:
        ``tau_shock = 0.56e-15`` for the Fig. 3 PCF config). Must be > 0.
    conserving_shock : bool
        Enable the photon-conserving (pcGNLSE) shock operator. Default
        ``False``. The modification substitutes ``|γ|`` for ``γ`` on the
        delayed (Raman) arm only; the instantaneous SPM/SS arm keeps the
        signed ``γ``. **Note:** for any fiber with ``γ > 0`` (every ordinary
        dielectric), ``abs(gamma) == gamma`` and the two paths are identical
        — the flag is a no-op. It only has an effect when ``γ < 0`` (the
        sign bench used in ``reproductions/huang_202x_pcgnlse_attractors/``).
        Requires ``include_raman=True`` and a Raman response.
    scheme : {"strang", "yoshida4", "etdrk4", "rk4ip"}
        Composition/integration scheme. ``"strang"`` (default) is the
        second-order symmetric splitting of Blow & Wood (1989) Eq. (17);
        ``"yoshida4"`` composes three Strang steps with Suzuki weights for
        fourth order; ``"etdrk4"`` is the fourth-order exponential RK4 of
        Cox & Matthews (2002) — no splitting; ``"rk4ip"`` is Hult's
        fourth-order Runge–Kutta in the interaction picture (2007) — also
        no splitting, with the linear propagator applied exactly and the
        nonlinear remainder advanced by classical RK4. See :meth:`_advance`
        for the algebra and the trade-offs. With ``gamma_omega`` the Strang
        nonlinear sub-step is an explicit midpoint (RK2) advance, so
        ledger-grade ``gamma_omega`` runs are ``"etdrk4"`` (Strang carries
        ~2e-3 in the energy at production step counts).
    step_size : Length | None
        Fixed step size (m). If None, adaptive stepping is used.
    gamma_omega : array_like or callable, optional
        First-class **per-bin** nonlinear coefficient ``γ(Ω)`` (1/(W·m)): an
        array sampled on the grid's ``Ω = ω−ω₀`` axis (rad/s) or a callable
        ``γ(Ω)``. The Kerr/Raman term is then applied in the energy-conserving
        **symmetrised** form

        ``N(A) = i√γ(Ω)·F{ P_NL(|A|²)·F⁻¹[√γ(Ω)·A] }``

        (Hermitian per bin: ``√γ`` on both sides of the pointwise
        polarisation), which conserves ``∫|A|² dt`` exactly for a real Raman
        response. For a spectrally narrow field this reduces to the scalar
        ``iγ(Ω)|A|²A``. The **naive** ``iγ(Ω)·F{P·A}`` form does not conserve
        energy (~10 % drift on a 20 mm SCG deck, independent of ``dz``) and is reachable
        only through ``gamma_omega_naive=True``, which warns. Combines with
        ``full_beta``, Raman and ``alpha_fn``; ``include_self_steepening`` with
        ``gamma_omega`` is refused (not implemented). ``gamma_fn`` / ``a_eff_fn``
        are **ignored** when ``gamma_omega`` is given (the per-bin operator takes
        precedence), and the engine warns rather than dropping them silently.
    gamma_omega_naive : bool
        Diagnostic only: apply the non-Hermitian ``iγ(Ω)·F{P·A}`` form instead
        of the default symmetrised one. Default ``False``. Requires
        ``gamma_omega``; emits a ``UserWarning``.
    alpha_w : float, array_like or callable, optional
        Per-bin linear field-amplitude loss α(Ω) in 1/m, sampled on
        ``grid.w`` or returned by a callable α(Ω). It is applied exactly as
        ``exp(-alpha_w*dz/2)`` in the linear operator, including ETDRK4.
        This is an amplitude coefficient exactly like ``fiber.alpha``: do not
        double it when converting a power attenuation or ``loss_db_cm``.
        Combining it with nonzero ``fiber.alpha`` is refused to prevent double
        counting. The public ``alpha_w`` field records that spectral loss was
        accounted for in the run.
    three_photon_alpha_3 : float or callable, optional
        Three-photon absorption coefficient α₃ with field form
        ``−α₃|A|⁴A/2`` (``I = |A|²``) — i.e. ``dI/dz = −α₃I³`` per sample, closed
        form ``I(z) = I₀/√(1+2α₃I₀²z)``. Applied by every scheme: the Strang
        path solves that per-sample ODE exactly in a sub-step placed
        symmetrically around the Kerr phase (so the composition stays second
        order), while ``etdrk4``/``rk4ip`` carry it as the pointwise functional
        it is. **Residual:** no 3PA-generated free carriers (no FCA, no carrier
        recombination) — that is ``include_free_carriers``' TPA model, which is
        separate. A callable α₃(Ω) is applied as the per-bin weight
        ``−½·F⁻¹[α₃(Ω)·F(|A|⁴A)]``; a float is applied pointwise, exactly as
        written. ``0.0`` is a no-op (bit-identical to omitting the term). The
        accumulated integrals are exposed after :meth:`propagate` as
        :attr:`w3pa` (``∫∫|A|⁴dt dz``, plus :attr:`w3pa_vs_z`) and
        :attr:`w3pa_energy` (``∫∫|A|⁶dt dz``), whose constant-α₃ closure is
        ``E_out = E_in − α₃·w3pa_energy``. :attr:`w3pa_spectral` additionally
        accumulates ``∫ -2 Re⟨A, N_3PA(A)⟩ dt dz`` and closes the energy budget
        for both scalar and spectral α₃ as ``E_out = E_in − w3pa_spectral``.

        **Units:** the value acts on ``|A|²`` in watts, so it must be
        ``α₃/A_eff²``. Pass the *physical* α₃ (m³/W²) with
        ``Fiber(a_eff=Area(...))`` and the engine divides by ``A_eff²`` for
        you; without ``a_eff`` the number is used exactly as given.
    min_shrink_factor : float
        Floor for the gradient-based step shrink factor in z-dependent mode.
        Must be in (0, 1]. Default 0.1.
    """

    def __init__(
        self,
        pulse: Wave,
        fiber: FiberProfile,
        betas: NDArray | None = None,
        include_raman: bool = False,
        include_self_steepening: bool = False,
        include_tpa: bool = False,
        include_free_carriers: bool = False,
        conserving_shock: bool = False,
        tau_shock: float | None = None,
        step_size: Length | None = None,
        scheme: Scheme = "strang",
        dispersion_profile: (
            ZDependentDispersion | Callable[[NDArray, float], NDArray] | None
        ) = None,
        a_eff_fn: Callable[[float], float] | None = None,
        alpha_fn: Callable[[float], float] | None = None,
        gamma_fn: Callable[[float], float] | None = None,
        min_shrink_factor: float = 0.1,
        betas_unit: BetasUnit = "ps^k/m",
        full_beta=None,
        raman_response=None,
        gamma_omega: NDArray | Callable[[NDArray], NDArray] | None = None,
        gamma_omega_naive: bool = False,
        three_photon_alpha_3: float | Callable[[NDArray], NDArray] | None = None,
        alpha_w: (
            float | NDArray | Callable[[NDArray], NDArray] | None
        ) = None,
    ):
        self.pulse = pulse
        self.fiber = fiber
        convention = getattr(
            getattr(pulse, "envelope", None), "amplitude_convention", "power"
        )
        if convention == "field":
            raise ValueError(
                "pulse envelope is in field units (V/m), but the GNLSE engine "
                "uses |A|² in watts. Convert it first with "
                "Wave.to_engine_units(A_eff) (or build it with "
                "Envelope.from_energy)."
            )
        if full_beta is not None and dispersion_profile is not None:
            raise ValueError(
                "full_beta and dispersion_profile are mutually exclusive: "
                "full_beta is the static scalar β(ω) operator, while "
                "dispersion_profile is the z-dependent β(ω,z) path."
            )
        if betas is None and full_beta is None:
            raise ValueError("pass either betas or full_beta")
        self.full_beta = full_beta
        # get_betas() returns betas in ps^(k)/m with omega in rad/ps.
        # We keep them in native units and convert omega to rad/ps in _linear_step.
        self.betas = (
            _normalize_betas(betas, betas_unit)
            if full_beta is None
            else np.array([], dtype=float)
        )
        self.include_raman = include_raman
        # Raman-response override (parity with MultimodeSplitStepEngine); the
        # fiber attribute remains the fallback.
        self.raman_response = (
            raman_response if raman_response is not None else fiber.raman_response
        )
        self.include_self_steepening = include_self_steepening
        self.include_tpa = include_tpa
        self.include_free_carriers = include_free_carriers
        if (
            self.include_tpa
            and self.fiber.beta_tpa > 0.0
            and self.fiber.sigma_tpa <= 0.0
            and not self.include_free_carriers
        ):
            # Gap M (§17.1): include_tpa consumes sigma_tpa only, so the flag a
            # user reaches for when they mean "TPA on" was a silent no-op.
            warnings.warn(
                "include_tpa=True applies fiber.sigma_tpa only and this fiber "
                "has sigma_tpa == 0, so it is a no-op: fiber.beta_tpa is not "
                "consumed by this flag. beta_tpa is carried by "
                "include_free_carriers=True (the time-resolved TPA/FCA "
                "model). Set include_free_carriers=True to get the beta_tpa "
                "absorption you asked for, give the fiber a sigma_tpa, or "
                "drop include_tpa.",
                UserWarning,
                stacklevel=2,
            )

        # Opt-in photon-conserving nonlinear operator (pcGNLSE; time-domain
        # form, Huang et al. arXiv:2607.05244): |γ| in the delayed-Raman
        # phase arm and in the SS–Raman dissipative cross term (ISSUES.md #1).
        self.conserving_shock = bool(conserving_shock)
        if tau_shock is not None and not tau_shock > 0.0:
            raise ValueError(
                f"tau_shock must be positive (seconds, SI), got {tau_shock!r}"
            )
        self._tau_shock_override = float(tau_shock) if tau_shock is not None else None
        self.step_size = step_size

        if not (0.0 < min_shrink_factor <= 1.0):
            raise ValueError(
                f"min_shrink_factor must be in (0, 1], got {min_shrink_factor!r}"
            )
        self._min_shrink_factor = min_shrink_factor
        self._h_R_fft_cache: NDArray | None = None

        if scheme not in _SCHEMES:
            raise ValueError(
                f"Unknown scheme {scheme!r}; expected one of {', '.join(_SCHEMES)}."
            )
        #: Composition scheme; see :meth:`_advance`.
        self.scheme: Scheme = scheme

        # z-dependent hooks (all optional; None → uniform behavior)
        self._dispersion_profile = dispersion_profile
        self._a_eff_fn = a_eff_fn
        self._alpha_fn = alpha_fn
        self.gamma_fn = gamma_fn
        self._is_z_dependent = dispersion_profile is not None

        # Set up temporal grid from pulse
        self.grid: TemporalGrid = pulse.grid
        self.omega0 = pulse.central_frequency

        #: Per-bin field-amplitude loss in 1/m; its presence names spectral
        #: loss in the run record. ``None`` keeps the legacy scalar-loss path.
        self.alpha_w = alpha_w
        self._alpha_w: NDArray | None = None
        if alpha_w is not None:
            if fiber.alpha != 0.0:
                raise ValueError(
                    "alpha_w cannot be combined with nonzero fiber.alpha: "
                    "both account for linear loss and would double count it."
                )
            a_w = np.asarray(
                alpha_w(self.grid.w) if callable(alpha_w) else alpha_w,
                dtype=float,
            )
            if a_w.ndim == 0:
                a_w = np.full(self.grid.w.shape, float(a_w))
            if a_w.shape != self.grid.w.shape:
                raise ValueError(
                    "alpha_w must match the grid "
                    f"({self.grid.w.shape}), got {a_w.shape}; pass a scalar, "
                    "a callable α(Ω), or an array sampled on grid.w."
                )
            if not np.all(np.isfinite(a_w)) or np.any(a_w < 0.0):
                raise ValueError(
                    "alpha_w must be finite and non-negative (it is a linear "
                    "field-amplitude absorption coefficient)."
                )
            self._alpha_w = a_w

        #: Frequency-dependent nonlinear coefficient γ(Ω), resolved on
        #: ``grid.w`` (rad/s); ``None`` keeps the scalar-γ path untouched.
        self.gamma_omega = gamma_omega
        self._gamma_omega: NDArray | None = None
        if gamma_omega is not None:
            if include_self_steepening:
                raise NotImplementedError(
                    "gamma_omega= with include_self_steepening=True is not "
                    "implemented: the shock operator (1 + (i/ω₀)∂t) needs the "
                    "γ-weighted split form applied to its ∂_t source as well, "
                    "and no test pins that composition. Run the shock deck "
                    "with a scalar γ (omit gamma_omega=)."
                )
            g_w = (
                np.asarray(gamma_omega(self.grid.w), dtype=float)
                if callable(gamma_omega)
                else np.asarray(gamma_omega, dtype=float)
            )
            if g_w.shape != self.grid.w.shape:
                raise ValueError(
                    "gamma_omega must match the grid "
                    f"({self.grid.w.shape}), got {g_w.shape}; pass a callable "
                    "γ(Ω) or an array sampled on grid.w."
                )
            if not np.all(np.isfinite(g_w)) or np.any(g_w < 0.0):
                raise ValueError(
                    "gamma_omega must be finite and non-negative (the "
                    "symmetrised split operator takes √γ per bin)."
                )
            self._gamma_omega = g_w
            # Same defect shape as Gap M: a hook that reads active but is
            # ignored, because the per-bin operator takes precedence.
            ignored = [
                name
                for name, hook in (("gamma_fn", gamma_fn), ("a_eff_fn", a_eff_fn))
                if hook is not None
            ]
            if ignored:
                warnings.warn(
                    f"gamma_omega= is set, so {', '.join(ignored)} "
                    f"{'is' if len(ignored) == 1 else 'are'} ignored: the "
                    "per-bin γ(Ω) operator takes precedence over the scalar "
                    "γ hooks. Pass the z-dependence in γ(Ω) (or drop "
                    f"gamma_omega=) if you meant {', '.join(ignored)} to act.",
                    UserWarning,
                    stacklevel=2,
                )
        #: Diagnostic-only opt-in for the *naive* (non-Hermitian)
        #: ``iγ(Ω)·F{P·A}`` form; the default symmetrised form is the one
        #: that conserves energy.
        self._gamma_omega_naive = bool(gamma_omega_naive)
        if self._gamma_omega_naive:
            if self._gamma_omega is None:
                raise ValueError(
                    "gamma_omega_naive=True requires gamma_omega=...; it only "
                    "selects between two forms of the per-bin γ operator."
                )
            warnings.warn(
                "gamma_omega_naive=True applies the non-Hermitian "
                "iγ(Ω)·F{P·A} form, which does not conserve energy (the "
                "project measured an 11% drift on a 40 mm SCG deck). It is a "
                "diagnostic, not a physical operator: use the default "
                "symmetrised i√γ·F{P·F⁻¹[√γ·A]} form for production runs.",
                UserWarning,
                stacklevel=2,
            )

        #: Three-photon absorption coefficient α₃ (float or callable α₃(Ω)),
        #: with field form ``−α₃|A|⁴A/2`` and no free carriers (the residual:
        #: 3PA-generated carriers/FCA are not modelled). Recorded on the
        #: engine as part of the run's operator set.
        self.three_photon_alpha_3 = three_photon_alpha_3
        self._alpha3_scalar: float | None = None
        self._alpha3_w: NDArray | None = None
        if three_photon_alpha_3 is not None:
            if callable(three_photon_alpha_3) or np.ndim(three_photon_alpha_3) > 0:
                a3_w = np.asarray(
                    three_photon_alpha_3(self.grid.w)
                    if callable(three_photon_alpha_3)
                    else three_photon_alpha_3,
                    dtype=float,
                )
                if a3_w.shape != self.grid.w.shape:
                    raise ValueError(
                        "three_photon_alpha_3 must match the grid "
                        f"({self.grid.w.shape}), got {a3_w.shape}; pass a "
                        "callable α₃(Ω) or an array sampled on grid.w."
                    )
                if not np.all(np.isfinite(a3_w)) or np.any(a3_w < 0.0):
                    raise ValueError(
                        "three_photon_alpha_3 must be finite and non-negative "
                        "(it is an absorption coefficient)."
                    )
                if np.any(a3_w != 0.0):
                    self._alpha3_w = a3_w * self.fiber._nl_area_factor(2)
            else:
                a3 = float(three_photon_alpha_3)
                if not np.isfinite(a3) or a3 < 0.0:
                    raise ValueError(
                        "three_photon_alpha_3 must be finite and non-negative "
                        f"(it is an absorption coefficient), got {a3!r}."
                    )
                # α₃ = 0 is a no-op: bit-identical to omitting the term.
                if a3 != 0.0:
                    self._alpha3_scalar = a3 * self.fiber._nl_area_factor(2)
        #: Three-photon absorption ledger ``∫∫|A|⁴ dt dz`` (the brief's
        #: ``∫∫I²dt dz``) accumulated by :meth:`propagate` (0.0 when the term is
        #: off).
        self.w3pa: float = 0.0
        #: The integral that closes the **energy** budget for the field form
        #: ``−α₃|A|⁴A/2``: ``dE/dz = −α₃∫|A|⁶dt``, so
        #: ``E_out = E_in − α₃·w3pa_energy`` exactly (``w3pa`` above is the
        #: intensity-squared action, whose ``exp(−α₃w3pa)`` form holds per
        #: sample / in the small-loss limit only).
        self.w3pa_energy: float = 0.0
        #: Coefficient-weighted energy-loss ledger. For scalar α₃ this equals
        #: ``α₃*w3pa_energy``; for spectral α₃ it accumulates the exact
        #: instantaneous loss ``-2 Re⟨A, N_3PA(A)⟩`` of the per-bin operator.
        self.w3pa_spectral: float = 0.0
        self._w3pa_vs_z: list[float] | None = None
        self._full_beta_curve: NDArray | None = None
        self._full_phi: NDArray | None = None
        self._full_beta2_si: float | None = None
        if full_beta is not None:
            (
                self._full_beta_curve,
                self._full_phi,
                self._full_beta2_si,
            ) = _full_beta_phase(full_beta, self.grid, self.omega0)
        self.A = np.array(pulse.envelope_field, dtype=complex)
        self.evolution: list[Wave] = []
        self._z_positions: list[float] = [0.0]
        self._current_z: float = 0.0
        self._spectra_vs_z: tuple[NDArray, NDArray] | None = None
        self._energy_vs_z: list[float] | None = None
        #: Blow & Wood (1989) App. Eq. (A5) photon number Σ|Â(ω)|²/(ω₀+ω)dω at
        #: each saved step; only recorded with self-steepening, where the grid
        #: validation guarantees ω₀+Ω > 0 (see :meth:`_photon_number`).
        self._photon_vs_z: list[float] | None = None
        self._U = 0.0  # carrier density for TPA
        self._N: NDArray | None = None  # time-resolved carrier population

        if self.include_self_steepening:
            self._validate_shock_grid()
        self._warn_z_dependent_hooks_etdrk4()

    def _warn_z_dependent_hooks_etdrk4(self) -> None:
        """Warn when ETDRK4 freezes a z-varying hook (Gap N, §18.4).

        ``gamma_fn`` / ``alpha_fn`` / ``a_eff_fn`` are evaluated once per step
        and held across the φ-function stages. That is exact for a static hook
        and silently drops the scheme below fourth order for a z-dependent one;
        ``dispersion_profile`` and ``include_free_carriers`` are *refused* for
        the same class of reason, but a constant callable is a legitimate
        input, so these only warn. The hooks are probed at three positions —
        a hook that raises is left alone.
        """
        if self.scheme != "etdrk4":
            return
        length = float(self.fiber.length.as_m)
        zs = (0.0, 0.5 * length, length)
        varying: list[str] = []
        for name, hook in (
            ("gamma_fn", self.gamma_fn),
            ("alpha_fn", self._alpha_fn),
            ("a_eff_fn", self._a_eff_fn),
        ):
            if hook is None:
                continue
            try:
                vals = [float(hook(z)) for z in zs]
            except Exception:
                continue
            if max(vals) - min(vals) > 1e-12 * max(abs(v) for v in vals):
                varying.append(name)
        if varying:
            warnings.warn(
                f"scheme='etdrk4' freezes {', '.join(varying)} at the step's "
                "z: the φ-function weights hold a z-dependent hook constant "
                "across a step, so the scheme silently drops below fourth "
                "order (dispersion_profile and include_free_carriers are "
                "refused for the same reason). Use scheme='strang' or "
                "'yoshida4' for a z-dependent hook, or pass a static value.",
                UserWarning,
                stacklevel=3,
            )

    def _photon_number(self, A: NDArray) -> float:
        """Blow & Wood (1989) photon number: the Appendix Eq. (A5) invariant
        of the GNLS, ``Σ_ω |Â(ω)|² / (ω₀ + ω) · dω``.

        This — not the unweighted Parseval norm ``Σ|A|²dt`` — is what the
        GNLSE with self-steepening conserves. With ``τ_shock = 1/ω₀`` the
        frequency-dependent prefactor ``n_eff/N`` may be taken outside the
        integral, and B&W state the conserved quantity (A5) "is then the
        obvious classical analog of the total photon number in the optical
        wave". The unweighted energy is *not* an invariant: "Optical energy
        is not conserved by the equation, as indeed it is not conserved in
        stimulated Raman scattering".

        Only meaningful when every bin satisfies ``ω₀ + Ω > 0``, which
        :meth:`_validate_shock_grid` enforces whenever self-steepening is on.
        """
        w_abs = self.omega0 + self.grid.w
        return float(np.sum(np.abs(self.grid.fft(A)) ** 2 / w_abs) * self.grid.dw)

    def photon_number(self, A: NDArray | None = None) -> float:
        """Blow & Wood (1989) Eq. (A5) photon number of ``A``.

        The per-run ``Q_in``/``Q_out`` a validation gate needs; see
        :meth:`_photon_number` for the invariant and its domain. ``A=None``
        (default) uses the engine's current field, so after :meth:`propagate`
        this is ``Q_out``; pass the launch field for ``Q_in``.
        """
        field = self.A if A is None else A
        return self._photon_number(np.asarray(field, dtype=complex))

    @property
    def tau_shock(self) -> float:
        """Shock timescale in seconds (SI): explicit override or ``1/ω₀``."""
        if self._tau_shock_override is not None:
            return self._tau_shock_override
        return 1.0 / float(self.omega0)

    def _validate_shock_grid(self) -> None:
        """Reject grids that push the ``ω/ω₀`` shock factor outside its domain.

        Two independent conditions, both from the primary source (Blow & Wood
        1989):

        1. **Positivity of the optical frequency (hard).** The shock operator
           ``(1 + (i/ω₀)∂_t)`` has symbol ``ω/ω₀ = (ω₀ + Ω)/ω₀`` — exact, with no
           Taylor expansion in this realization. It is only meaningful for
           channels with positive *absolute* frequency, ``ω = ω₀ + Ω > 0``. On
           a symmetric grid spanning ``±Ω_max`` this requires ``Ω_max < ω₀``.
           Bins beyond that would alias onto negative optical frequencies where
           the factor has no physical meaning; the historical code clamped
           them to zero, which silently corrupted the shock physics. Failing
           loudly is the honest alternative (design decision D2).

        2. **Bandwidth of the GNLS derivation (soft, warned).** Blow & Wood
           derive the GNLS assuming "the bandwidth of U(ω') is less than
           ω₀/3" — a small-fractional-bandwidth expansion of the wave
           equation, not a property of any integrator. ``τ_shock·Ω_max ≤
           _SHOCK_TAYLOR_LIMIT = 0.2`` keeps the simulated bandwidth at
           ω₀/5, inside that bound with margin. This is a warning, not a
           rejection: exceeding it widens the band past the equation's
           derivation, it does not make the operator ill-defined.

        What this guard is *not* about (a corrected misdiagnosis,
        ``CHANGELOG.md`` 2026-10-27): a ``≤0.3 % … −5.2 %`` table of
        "photon-number drift vs τ·Ω_max" was once quoted here as evidence of a
        first-order-truncation grid artifact. That drift was the *unweighted*
        energy ``Σ|A|²dt`` changing, which the GNLSE does not conserve — the
        conserved quantity is Blow & Wood's Eq. (A5) photon number
        ``Σ|Â(ω)|²/(ω₀+ω)dω`` (see :meth:`_photon_number`). Tracking (A5)
        instead, the same runs conserve to ≲5×10⁻⁵ relative even at
        τ·Ω_max = 0.88.
        """
        omega_max = float(self.grid.omega_max)
        omega0 = float(self.omega0)
        if omega_max >= omega0:
            ratio = omega_max / omega0
            raise ValueError(
                "self-steepening requires the grid to resolve only positive "
                "absolute frequencies, i.e. Ω_max < ω₀, but got "
                f"Ω_max = {omega_max:.4e} rad/s >= ω₀ = {omega0:.4e} rad/s "
                f"(Ω_max/ω₀ = {ratio:.3f}). The grid has N = {self.grid.N} "
                f"samples over Tmax = {self.grid.Tmax.as_s:.4e} s. Fix by "
                f"increasing the time window Tmax by at least {ratio:.3f}× "
                "(equivalently reducing N for the same Tmax), or by disabling "
                "self-steepening."
            )
        tau_omega = float(self.tau_shock) * omega_max
        if tau_omega > _SHOCK_TAYLOR_LIMIT:
            fix = tau_omega / _SHOCK_TAYLOR_LIMIT
            warnings.warn(
                "self-steepening is being run outside the GNLS derivation "
                f"regime: τ_shock·Ω_max = {tau_omega:.3f} puts the simulated "
                f"bandwidth above ω₀/3, the small-bandwidth assumption Blow & "
                "Wood (1989) derive the GNLS under. The shock symbol "
                "ω/ω₀ = 1 + Ω·τ_shock remains exact; what degrades is the "
                "equation's own validity. The conserved quantity in this "
                "regime is the photon number Σ|Â(ω)|²/(ω₀+ω)dω (Blow & Wood "
                "App. Eq. A5), not the unweighted energy. Fix by increasing "
                f"Tmax by at least {fix:.2f}× (equivalently reducing N), or "
                "by disabling self-steepening.",
                UserWarning,
                stacklevel=3,
            )

    def _linear_step(self, A: NDArray, dz: float) -> NDArray:
        """Apply dispersion via FFT: A(ω) ← A(ω) · exp(+i·Σ β_k(Ω)·Δz).

        (Sign audit 2026-09-21: the summary line previously read ``exp(−i·…)``
        while the code below applies ``exp(+1j*phi)`` — a stale docstring from
        before the dispersion-sign flip; the plus sign is the physically
        correct propagator and pairs with the ``exp(−iωt)`` forward FFT
        convention, as the note below describes.)
        Also applies scalar and per-bin field-amplitude loss as
        ``exp(-α·Δz/2)`` and ``exp(-alpha_w(Ω)·Δz/2)`` respectively.
        Dispersion Taylor expansion starts at k=2 (β₂, β₃, ...).
        β₁ (group velocity) is not included — pulse stays in group-velocity frame.

        The plus sign pairs with :meth:`~photonics_helper.pulse.TemporalGrid.fft`
        (numpy ``exp(−iωt)`` convention) for the standard GNLSE
        ``i∂A/∂z = (β_k/k!) ∂^k A/∂t^k − γ|A|²A`` form, i.e. a bright soliton
        forms for anomalous dispersion (β₂ < 0), matching ``Dispersion.get_betas``
        and ``mi_gain_spectrum``.
        """
        A_w = self.grid.fft(A)
        omega_ps = self.grid.w * 1e-12  # rad/s → rad/ps

        if self._full_phi is not None:
            # Static full-β(ω): the retarded-frame phase is precomputed in
            # rad/m on grid.w; only the dz scaling and loss remain.
            phi = self._full_phi * dz
            alpha = self.fiber.alpha
            if alpha > 0:
                A_w = A_w * np.exp(-alpha * dz / 2)
        elif self._is_z_dependent:
            # z-dependent path: compute phase from β(ω₀+Ω, z) - β(ω₀, z)
            # at the current propagation position.
            # grid.w is the offset Ω (centered at 0), so absolute frequency is ω₀ + grid.w.
            omega_abs = self.omega0 + self.grid.w  # absolute angular frequency (rad/s)
            # Clip to dispersion profile's valid omega range to avoid NaN from extrapolation
            if self._dispersion_profile is not None and hasattr(
                self._dispersion_profile, "omegas"
            ):
                omega_min, omega_max = (
                    float(min(self._dispersion_profile.omegas)),
                    float(max(self._dispersion_profile.omegas)),
                )
                clipped_mask = (omega_abs < omega_min) | (omega_abs > omega_max)
                fraction_clipped = clipped_mask.sum() / len(omega_abs)
                if fraction_clipped > 0:
                    warnings.warn(
                        f"{fraction_clipped * 100:.1f}% of simulation grid frequencies are clipped "
                        f"to dispersion bounds [{omega_min:.2e}, {omega_max:.2e}] rad/s. "
                        f"Grid range: [{omega_abs.min():.2e}, {omega_abs.max():.2e}] rad/s.",
                        UserWarning,
                        stacklevel=3,
                    )
                omega_abs = np.clip(omega_abs, omega_min, omega_max)
            # Evaluate β at current z for all frequencies
            beta_at_z = self._eval_dispersion(omega_abs, self._current_z)
            # Carrier phase at current z
            beta_carrier = self._eval_dispersion(
                np.array([self.omega0]), self._current_z
            )
            # Phase: [β(ω₀+Ω, z) - β(ω₀, z)] * Δz
            phi = (beta_at_z - beta_carrier[0]) * dz

            # z-dependent loss
            alpha = self._get_alpha(self._current_z)
            if alpha > 0:
                loss_factor = np.exp(-alpha * dz / 2)
                A_w = A_w * loss_factor
        else:
            # φ(Ω) = Σ_{k≥2} βₖ·Ωᵏ/k!  with Ω in rad/ps, β_k in ps^k/m.
            phi = np.zeros_like(omega_ps, dtype=float)
            for k, beta_k in enumerate(self.betas, start=2):
                phi += beta_k * omega_ps**k / factorial(k)
            phi *= dz

            # Apply loss: exp(-α·Δz/2)
            alpha = self.fiber.alpha
            if alpha > 0:
                loss_factor = np.exp(-alpha * dz / 2)
                A_w = A_w * loss_factor

        if self._alpha_w is not None:
            A_w = A_w * np.exp(-self._alpha_w * dz / 2)
        A_w = A_w * np.exp(1j * phi)
        return np.asarray(self.grid.ifft(A_w))

    def _eval_dispersion(self, omega: NDArray, z: float) -> NDArray:
        """Evaluate β(ω, z) from the dispersion profile.

        Parameters
        ----------
        omega : 1-D array — absolute angular frequencies (rad/s).
        z : float — propagation position (m).

        Returns
        -------
        β values at (omega, z).
        """
        if callable(self._dispersion_profile):
            return np.asarray(self._dispersion_profile(omega, z))
        else:
            # ZDependentDispersion case
            assert self._dispersion_profile is not None
            return np.asarray(self._dispersion_profile.fn(omega, z))

    def _get_gamma(self, z: float) -> float:
        """Compute γ(z) from gamma_fn/a_eff_fn(z) or the fiber's stored gamma."""
        if self.gamma_fn is not None:
            return self.gamma_fn(z)
        if self._a_eff_fn is not None:
            a_eff = self._a_eff_fn(z)
            if self.fiber.gamma is not None:
                # Stored gamma wins; a_eff_fn only matters for n2-derived fibers.
                return float(self.fiber.gamma)
            return _gamma(
                self.fiber.n2,
                self.omega0,
                Area(a_eff, "m^2"),
                self.fiber.confinement_factor,
            )
        else:
            return _fiber_gamma(self.fiber, self.omega0)

    def _get_alpha(self, z: float) -> float:
        """Get α(z) from alpha_fn(z) or fallback to fiber.alpha."""
        if self._alpha_fn is not None:
            return self._alpha_fn(z)
        else:
            return self.fiber.alpha

    def _nl_intensity(self, intensity: NDArray, h_R_fft: NDArray | None) -> NDArray:
        """Kerr + Raman nonlinear driving intensity P_NL(|A|²).

        Parameters
        ----------
        intensity : |A(t)|² on the simulation grid.
        h_R_fft : FFT of h_R(t) on ``self.grid.t``, or None when Raman is off.
        """
        if h_R_fft is not None:
            fR = _response_fR(self.raman_response)
            return _raman_polarization(intensity, fR, self.grid, h_R_fft)
        return intensity

    def _delayed_intensity(
        self, intensity: NDArray, h_R_fft: NDArray | None
    ) -> NDArray:
        """Delayed-Raman driving intensity ``f_R · (h_R ⊗ |A|²)``.

        Used by the opt-in conserving-shock (pcGNLSE) path to split the
        nonlinear amplitude into the instantaneous and delayed pieces
        (ISSUES.md #1); identical to the delayed part of
        :meth:`_nl_intensity`.
        """
        if h_R_fft is None:
            return np.zeros_like(intensity, dtype=float)
        fR = _response_fR(self.raman_response)
        result: NDArray = fR * np.real(
            self.grid.ifft(h_R_fft * self.grid.fft(intensity))
        )
        return result

    def _get_h_R_fft(self) -> NDArray:
        """FFT of the Raman response on ``self.grid.t`` (cached).

        Depends only on the temporal grid and the fiber's Raman response,
        both fixed for the engine's lifetime — recomputing it every
        nonlinear step wasted an FFT per step.
        """
        if self._h_R_fft_cache is None:
            h_R = _delayed_h_R(self.raman_response, self.grid.t)
            self._h_R_fft_cache = self.grid.fft(h_R)
        return self._h_R_fft_cache

    def _nonlinear_step(self, A: NDArray, dz: float) -> tuple[NDArray, float]:
        """Apply nonlinear effects with an interaction-picture shock step.

        The total nonlinear polarization is
        ``P_NL = (1-fR)|A|² + fR·(h_R ⊛ |A|²)`` (real for a real response),
        and the nonlinear GNLSE term is
        ``iγ (1 + (i/ω₀)∂_t)(A·P_NL)`` (Blow & Wood 1989;
        Dudley–Genty–Coen RMP 78, 1135 (2006), Eq. (3)).

        **Photon-number balance (Blow & Wood 1989, Appendix).** The
        conserved quantity of the GNLS with self-steepening is *not* the
        unweighted energy ``∫|A|²dt`` but the frequency-weighted photon number

        ``Q = ∫|Â(ω)|² / (ω₀ + ω) · dω``      (their Eq. (A5)),

        which with ``τ_shock = 1/ω₀`` is "the obvious classical analog of the
        total photon number in the optical wave" — the frequency-dependent
        prefactor ``n_eff/N`` may be taken outside the integral in a low-loss
        guide. Their abstract states it plainly: "Optical energy is not
        conserved by the equation, as indeed it is not conserved in stimulated
        Raman scattering because photons are preferentially emitted at a lower
        frequency." So a run may legitimately show a large change in
        ``energy_vs_z`` while ``photon_number_vs_z`` is constant — measured on
        a fissioning N=3 soliton: −14.4 % energy change with Q conserved to
        1×10⁻⁵ relative (this diagnosis replaced an earlier misreading of that
        drift as a truncation artifact; ``CHANGELOG.md`` 2026-10-27). With
        self-steepening *off* the unweighted energy is the invariant again.
        The shock factor ``(1 + Ω·τ_shock)`` is exactly ``ω/ω₀`` bin-by-bin in
        this realization — there is no Taylor expansion in the operator to
        truncate. The historical ``max(1 + Ω·τ, 0)`` clamp was removed both
        because it is unphysical (see :meth:`_validate_shock_grid`) and because
        it destroyed the operator whose invariant is (A5).

        **Integrator.** The stiff, exactly-integrable part of the step is the
        phase rotation ``exp(iγP_NL Δz)``. Following the RK4IP /
        exponential-integrator idea (Hult, *JLT* **25**, 3770 (2007);
        Hochbruck & Ostermann, *Acta Numerica* **19**, 209 (2010)), that phase
        is applied exactly in two symmetric half-steps (Strang splitting) and
        only the much smaller shock correction ``iγτ_shock ∂_t(A·P_NL)`` is
        advanced with frequency-domain RK4. No Runge–Kutta stage ever
        integrates the stiff phase, and no bin is clamped. The scheme is
        validated against a heavily-substepped reference of the same flow
        (``tests/test_gnlse_unitarity.py::test_integrator_fidelity``).

        The shock sign ``(1 + Ω·τ_shock)`` (rather than ``1 − Ω·τ_shock``) is
        the convention that reproduces the paper's femtosecond SCG, where
        self-steepening counteracts the Raman self-frequency red-shift. This
        is the same factor laserfun applies (``gamma/ω₀`` times ``Ω + ω₀`` in
        ``laserfun/laserfun/nlse.py``); the apparent cross-library
        disagreement reported in earlier revisions was step resolution, not
        a sign convention.

        The grid must satisfy ``Ω_max < ω₀`` when self-steepening is enabled;
        :meth:`_validate_shock_grid` enforces this at construction.

        With ``gamma_omega`` the Kerr/Raman action is not a time-domain phase
        any more (the symmetrised operator is not diagonal), so the step is the
        explicit midpoint (RK2) advance of :meth:`_gamma_omega_rhs`, which keeps
        the composition second order; the 3PA loss rides inside that remainder.
        """
        if self._gamma_omega is not None:
            gamma_omega = self._get_gamma(self._current_z)  # unused by the form

            def nl(state: NDArray) -> NDArray:
                return self._nonlinear_rhs(state, gamma_omega)

            A = A + dz * nl(A + 0.5 * dz * nl(A))
            A, U_new = tpa_step(
                A, self.fiber, self.grid, dz, self.include_tpa, self._U, self.omega0
            )
            if self.include_free_carriers:
                A, self._N = free_carrier_step(
                    A, self.fiber, self.grid, dz, self._N, self.omega0
                )
            return A, U_new

        gamma = self._get_gamma(self._current_z)
        intensity = np.abs(A) ** 2

        h_R_fft: NDArray | None = None
        if self.include_raman and self.raman_response is not None:
            # Cached FFT of h_R on the same centered grid (see _get_h_R_fft).
            h_R_fft = self._get_h_R_fft()
        elif self.include_raman and self.raman_response is None:
            raise ValueError(
                "include_raman=True but fiber.raman_response is None. "
                "Provide a RamanResponse or set include_raman=False."
            )

        P_NL = self._nl_intensity(intensity, h_R_fft)

        if self._three_photon_enabled:
            # First half of the symmetric (Strang) placement of the 3PA loss
            # around the exact Kerr phase; the second half follows tpa_step.
            A = self._three_photon_step(A, 0.5 * dz)
            P_NL = self._nl_intensity(np.abs(A) ** 2, h_R_fft)

        P_inst: NDArray | None
        P_del: NDArray | None
        if self.conserving_shock:
            # pcGNLSE (ISSUES.md #1): split the nonlinear amplitude into the
            # instantaneous (SPM, signed γ) and delayed-Raman (|γ|) pieces;
            # the shock stage applies signed γτ to the pure-SS (P_inst) arm
            # and |γ|τ to the SS–Raman cross (P_del) arm.
            if h_R_fft is None or self.raman_response is None:
                raise ValueError(
                    "conserving_shock=True requires include_raman=True and a "
                    "Raman response (the pcGNLSE modification is on the "
                    "Raman/SS–Raman arms)."
                )
            fR0 = float(_response_fR(self.raman_response))
            P_inst = (1.0 - fR0) * intensity
            P_del = fR0 * np.real(self.grid.ifft(h_R_fft * self.grid.fft(intensity)))
            g_r = abs(gamma)
        else:
            P_inst = None  # unused
            P_del = None  # unused
            g_r = gamma

        if self.include_self_steepening:
            tau = self.tau_shock
            # Shock correction operator in the frequency domain:
            #   shock_rhs(A) = iγτ ∂_t(A·P_NL)  →  iγτ·Ω·F{A·P_NL}
            # Purely imaginary on the diagonal, so its exact exponential is
            # unitary; RK4 only approximates the (small) P_NL variation.
            shock_kernel = 1j * gamma * tau * self.grid.w
            shock_kernel_pc = 1j * g_r * tau * self.grid.w

            def shock_rhs(A_state: NDArray) -> NDArray:
                src_w = self.grid.fft(A_state)
                if self.conserving_shock:
                    # signed γτ on the pure-SS (instantaneous) arm;
                    # |γ|τ on the SS–Raman (delayed) cross arm.
                    i_state = (1.0 - fR0) * np.abs(A_state) ** 2
                    d_state = self._delayed_intensity(np.abs(A_state) ** 2, h_R_fft)
                    rhs = shock_kernel * self.grid.fft(
                        i_state * A_state
                    ) + shock_kernel_pc * self.grid.fft(d_state * A_state)
                    return np.asarray(self.grid.ifft(rhs), dtype=complex)
                p_nl = self._nl_intensity(np.abs(A_state) ** 2, h_R_fft)
                src_w = self.grid.fft(p_nl * A_state)
                return np.asarray(self.grid.ifft(shock_kernel * src_w), dtype=complex)

            def _rk4_shock(A_state: NDArray, h: float) -> NDArray:
                k1 = shock_rhs(A_state)
                k2 = shock_rhs(A_state + 0.5 * h * k1)
                k3 = shock_rhs(A_state + 0.5 * h * k2)
                k4 = shock_rhs(A_state + h * k3)
                return np.asarray(
                    A_state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4),
                    dtype=complex,
                )

            # Symmetric (Strang) split: half the exact Kerr/Raman phase, the
            # full shock correction, then the trailing half phase.
            half_phase: NDArray
            if self.conserving_shock:
                assert P_inst is not None and P_del is not None  # narrowed above
                half_phase = (0.5j * gamma * P_inst + 0.5j * abs(gamma) * P_del) * dz
            else:
                half_phase = 0.5j * gamma * P_NL * dz
            A_state: NDArray = np.asarray(A * np.exp(half_phase), dtype=complex)

            # Substep so the shock-induced change per substep stays small.
            # Only the shock term is resolved; the stiff phase is analytic, so
            # this needs far fewer substeps than the old classical-RK4 scheme.
            shock_rate = float(np.max(np.abs(shock_kernel))) * float(np.max(P_NL))
            n_sub = int(min(200, max(1, np.ceil(shock_rate * dz / 0.05))))
            h_sub = dz / n_sub
            for _ in range(n_sub):
                A_state = _rk4_shock(A_state, h_sub)

            P_final_int = np.abs(A_state) ** 2
            if self.conserving_shock:
                assert P_inst is not None and P_del is not None  # narrowed above
                P_inst_f = (1.0 - fR0) * P_final_int
                P_del_f = self._delayed_intensity(P_final_int, h_R_fft)
                A = A_state * np.exp(
                    0.5j * gamma * P_inst_f * dz + 0.5j * abs(gamma) * P_del_f * dz
                )
            else:
                P_final = self._nl_intensity(P_final_int, h_R_fft)
                A = A_state * np.exp(0.5j * gamma * P_final * dz)
        else:
            # Exact (unitary) Kerr/Raman phase rotation.
            if self.conserving_shock:
                assert P_inst is not None and P_del is not None  # narrowed above
                A = A * np.exp(1j * gamma * P_inst * dz + 1j * abs(gamma) * P_del * dz)
            else:
                A = A * np.exp(1j * gamma * P_NL * dz)

        if self._three_photon_enabled:
            # Symmetric (Strang) placement of the loss around the exact Kerr
            # phase: exp(−α₃I²dz/4) · exp(iγP dz) · exp(−α₃I²dz/4). A single
            # trailing loss sub-step would compose with the phase at first
            # order (Lie), costing the field a scheme order — as the TPA path
            # does; this keeps the composition second order.
            A = self._three_photon_step(A, 0.5 * dz)

        A, U_new = tpa_step(
            A, self.fiber, self.grid, dz, self.include_tpa, self._U, self.omega0
        )
        if self.include_free_carriers:
            A, self._N = free_carrier_step(
                A, self.fiber, self.grid, dz, self._N, self.omega0
            )
        return A, U_new

    def _get_omega_bounds(self) -> tuple[float, float]:
        """Return (omega_min, omega_max) from the dispersion profile, or a wide default."""
        if self._dispersion_profile is not None and hasattr(
            self._dispersion_profile, "omegas"
        ):
            return float(min(self._dispersion_profile.omegas)), float(
                max(self._dispersion_profile.omegas)
            )
        # Fallback: wide range around carrier
        return self.omega0 - 5e15, self.omega0 + 5e15

    def _extract_beta_k(self, z: float, order: int = 2) -> float | NDArray:
        """Extract Taylor coefficients β₂…β_order from the dispersion profile at z.

        Returns β₂ (s²/m) when order=2, otherwise [β₂, …, β_order] in SI (s^k/m).
        """
        if order < 2:
            raise ValueError(f"order must be >= 2, got {order}")
        profile = self._dispersion_profile
        if profile is not None and hasattr(profile, "get_betas_at_z"):
            betas = profile.get_betas_at_z(z, order=order, omega0=self.omega0)
            if order == 2:
                return float(betas[0])
            return betas
        omega_min, omega_max = self._get_omega_bounds()
        halfwidth = min(0.1e15, (omega_max - omega_min) / 2)
        omega_low = max(omega_min, self.omega0 - halfwidth)
        omega_high = min(omega_max, self.omega0 + halfwidth)
        omega_near = np.linspace(omega_low, omega_high, 100)
        beta_at_z = self._eval_dispersion(omega_near, z)
        coeffs = np.polyfit(omega_near - self.omega0, beta_at_z, order)
        betas = np.array(
            [factorial(k) * coeffs[order - k] for k in range(2, order + 1)],
            dtype=float,
        )
        if order == 2:
            return float(betas[0])
        return betas

    def _extract_beta2(self, z: float) -> float:
        """Extract β₂ from the dispersion profile at position z (s²/m)."""
        result = self._extract_beta_k(z, order=2)
        return float(result)

    def _adaptive_step_size(self, A: NDArray) -> float:
        """Compute adaptive step size from local error estimates."""
        alpha = self._get_alpha(self._current_z)
        dz_loss = 1.0 / max(alpha, 1e-10) * 0.01

        T0 = self.pulse.envelope.pulse_width.as_s
        if not self._is_z_dependent:
            if self._full_beta2_si is not None:
                # Static full-β(ω): use the curve's β₂ at the carrier.
                dz_disp = T0**2 / max(abs(self._full_beta2_si), 1e-30) * 0.01
            elif len(self.betas) == 0:
                # No dispersion coefficients — skip dispersion length limit.
                dz_disp = float("inf")
            else:
                beta2 = abs(self.betas[0]) * 1e-24  # ps²/m → s²/m for dispersion length
                dz_disp = T0**2 / max(beta2, 1e-30) * 0.01
        else:
            # z-dependent: extract β₂ from the actual dispersion profile at current z.
            # Must use |β₂| — anomalous (β₂<0) would make max(β₂, ε)=ε and disable the
            # dispersion step-size limit, producing unstable/over-coarse steps.
            beta2 = abs(self._extract_beta2(self._current_z))
            dz_disp = T0**2 / max(beta2, 1e-30) * 0.01

        gamma = self._get_gamma(self._current_z)
        # Use input peak power for the nonlinear-length limit (same as
        # estimate_num_steps).  Instantaneous max(|A|²) can spike during
        # spectral broadening and shrink dz without bound, stalling propagation.
        # NOTE: peak_power() warns without A_eff, but this internal heuristic
        # works in the envelope convention (consistent with gamma-direct
        # fibers), so the warning would be unactionable noise here — direct
        # user calls to Wave.peak_power() still warn as before.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*normalized envelope units.*")
            I_ref = max(self.pulse.peak_power(), 1e-30)
        if gamma > 0.0:
            # Cap nonlinear phase per step: γ P₀ Δz < π/4 (stricter with steepening).
            omega_factor = 1.0
            if self.include_self_steepening:
                omega_factor = 1.0 + self.grid.omega_max / self.omega0
            dz_nl = np.pi / (4.0 * gamma * I_ref * omega_factor)
        else:
            dz_nl = float("inf")

        return float(min(dz_loss, dz_disp, dz_nl))

    def _gradient_shrink_factor(self, z: float, dz_base: float) -> float:
        """Estimate dispersion gradient at z and return a shrink factor.

        If |∂β/∂z| is large (taper neck), return a factor < 1 to reduce dz.
        Returns 1.0 if no z-dependent profile is set.
        """
        if not self._is_z_dependent:
            return 1.0

        dz_finite = dz_base * 0.1
        z_min = 0.0
        z_max = self.fiber.length.as_m

        z_lo = max(z_min, z - dz_finite / 2)
        z_hi = min(z_max, z + dz_finite / 2)

        dz_actual = z_hi - z_lo
        if dz_actual < 1e-15:
            return 1.0

        omega_carrier = np.array([self.omega0])
        beta_lo = self._eval_dispersion(omega_carrier, z_lo)
        beta_hi = self._eval_dispersion(omega_carrier, z_hi)

        abs_change = float(np.max(np.abs(beta_hi - beta_lo)))
        ref_beta = max(float(np.max(np.abs(beta_lo))), 1e-10)

        # Numerical-noise guard: for uniform (or nearly uniform) profiles,
        # skip shrink to avoid excessive step splitting.
        if abs_change < 1e-7 * ref_beta:
            return 1.0

        dbeta_dz = abs_change / dz_actual

        # Threshold: compare gradient against a relevant dispersion scale.
        # For z-dependent mode, use |β₂| at current z (sign must not flip the test);
        # for uniform mode, use the provided betas array.
        if self._is_z_dependent:
            beta2_ref = abs(self._extract_beta2(z))
        elif len(self.betas) > 0 and abs(self.betas[0]) > 1e-30:
            beta2_ref = abs(self.betas[0]) * 1e-24  # ps²/m → s²/m
        else:
            beta2_ref = ref_beta * dz_base * 0.1

        threshold = beta2_ref / (dz_base * 10)

        if dbeta_dz > threshold:
            factor = threshold / dbeta_dz
            return max(self._min_shrink_factor, min(factor, 1.0))
        else:
            return 1.0

    def _etdrk_linear(self, dz: float) -> NDArray:
        """Diagonal linear operator per bin, ``L(Ω) = i·φ(Ω) − α(z)/2`` (per m).

        The dispersion phase is the same ``φ(Ω) = Σ_{k≥2} βₖΩᵏ/k!`` that
        :meth:`_linear_step` applies — or, for a static ``full_beta`` curve,
        the precomputed retarded-frame phase — and the loss enters as a real
        part ``−α/2`` at the current ``z`` (matching ``exp(−α·dz/2)`` there) so
        that ``exp(L·dz)`` carries both.

        **Scope (narrowed 2026-10-25, Gap J).** ETDRK4 needs only a *diagonal*
        ``L`` and a nonlinear functional ``N(A)`` evaluable at arbitrary stage
        states. Both hold for a static ``full_beta`` curve, a per-bin
        ``gamma_omega``, ``gamma_fn``/``a_eff_fn`` (γ frozen at ``z``) and
        ``alpha_fn`` (α frozen at ``z``), so those are allowed. What is still
        refused, and why:

        * ``dispersion_profile`` — the φ-function weights assume ``L`` is
          constant along a step; this path changes β(ω,z) (with its own
          adaptive gradient shrink) every step, so the exponential-integrator
          error analysis does not apply. Use ``full_beta`` for a static curve.
        * ``include_tpa`` — the legacy TPA step carries a spatially averaged
          carrier density ``U`` from *previous* steps (see :func:`tpa_step`),
          so it is not a pointwise functional of the current state and cannot
          be evaluated at an RK stage. The pointwise intensity-nonlinear loss
          is the ``three_photon_alpha_3`` term, which *is* supported.
        * ``include_free_carriers`` — the carrier population ``N(t)`` is a
          history-dependent state advanced by the split-step stage.
        """
        reason: str | None = None
        if self._is_z_dependent:
            reason = (
                "dispersion_profile= (z-dependent β(ω,z)): the φ-function "
                "weights assume a step-constant linear operator"
            )
        elif self.include_tpa:
            reason = (
                "include_tpa=True: the legacy TPA step carries a spatially "
                "averaged carrier density from previous steps, so it is not "
                "a pointwise functional of the current state"
            )
        elif self.include_free_carriers:
            reason = (
                "include_free_carriers=True: the carrier population N(t) is a "
                "history-dependent state advanced by the split-step stage"
            )
        if reason is not None:
            raise NotImplementedError(
                f"scheme='etdrk4' cannot advance {reason}. Static full_beta, "
                "gamma_omega/gamma_fn, alpha_fn and three_photon_alpha_3 are "
                "supported; use scheme='strang' for the feature named above."
            )
        if self._full_phi is not None:
            phi = np.asarray(self._full_phi, dtype=float)
        else:
            omega_ps = self.grid.w * 1e-12  # rad/s → rad/ps
            phi = np.zeros_like(omega_ps, dtype=float)
            for k, beta_k in enumerate(self.betas, start=2):
                phi += beta_k * omega_ps**k / factorial(k)
        alpha = float(self._get_alpha(self._current_z))
        loss = alpha if self._alpha_w is None else alpha + self._alpha_w
        return np.asarray(1j * phi - 0.5 * loss, dtype=complex)

    def _nonlinear_rhs(self, A: NDArray, gamma: float) -> NDArray:
        """The full nonlinear remainder ``N(A)`` of the GNLSE RHS (time domain).

        ``N(A) = iγ·P_NL(|A|²)·A`` (Kerr + Raman phase) plus, with
        self-steepening, the same shock correction the RK4 path integrates:
        ``iγ·τ_shock·∂_t(P_NL·A)`` with symbol ``iγτΩ``. With
        ``conserving_shock`` (pcGNLSE) the instantaneous and delayed arms
        carry the signed ``γ`` and ``|γ|`` respectively, matching
        :meth:`_nonlinear_step`.

        With ``gamma_omega`` the scalar ``gamma`` is ignored and the
        symmetrised per-bin form is used instead (see
        :meth:`_gamma_omega_rhs`); with ``three_photon_alpha_3`` the pointwise
        loss ``−α₃|A|⁴A/2`` is added — both are legitimate ``N(A)``: a
        diagonal operator plus a functional of the current state.

        This is the exponential integrator's counterpart of the ``N`` of
        ``∂_zA = L·A + N(A)``; the stiff linear part is carried entirely by
        the φ-function weights, so no stage integrates the dispersion phase.
        """
        intensity = np.abs(A) ** 2
        h_R_fft: NDArray | None = None
        if self.include_raman:
            if self.raman_response is None:
                raise ValueError(
                    "include_raman=True but fiber.raman_response is None. "
                    "Provide a RamanResponse or set include_raman=False."
                )
            h_R_fft = self._get_h_R_fft()

        kernel: NDArray | None = None
        if self.include_self_steepening:
            tau = self.tau_shock
            kernel = 1j * tau * self.grid.w

        if self._gamma_omega is not None:
            rhs = self._gamma_omega_rhs(A, intensity, h_R_fft)
        elif self.conserving_shock:
            # pcGNLSE arms, mirroring _nonlinear_step: signed γ on the pure-SS
            # (instantaneous) arm, |γ| on the SS–Raman (delayed) arm.
            fR0 = float(_response_fR(self.raman_response))
            P_inst = (1.0 - fR0) * intensity
            P_del = self._delayed_intensity(intensity, h_R_fft)
            g_r = abs(gamma)
            rhs = (1j * gamma * P_inst + 1j * g_r * P_del) * A
            if kernel is not None:
                rhs = rhs + self.grid.ifft(
                    (1j * gamma * tau * self.grid.w) * self.grid.fft(P_inst * A)
                    + (1j * g_r * tau * self.grid.w) * self.grid.fft(P_del * A)
                )
        else:
            P_NL = self._nl_intensity(intensity, h_R_fft)
            rhs = 1j * gamma * P_NL * A
            if kernel is not None:
                rhs = rhs + self.grid.ifft(
                    (1j * gamma * tau * self.grid.w) * self.grid.fft(P_NL * A)
                )
        if self._three_photon_enabled:
            rhs = rhs + self._three_photon_rhs(A)
        return np.asarray(rhs, dtype=complex)

    @property
    def _three_photon_enabled(self) -> bool:
        """Whether the 3PA term is active (α₃ > 0)."""
        return self._alpha3_scalar is not None or self._alpha3_w is not None

    def _three_photon_rhs(self, A: NDArray) -> NDArray:
        """3PA source ``−α₃|A|⁴A/2`` (``I = |A|²``), pointwise in time.

        The scalar α₃ path is literally the field form ``−α₃I²/2·A`` (no
        transform). A callable α₃(Ω) is applied as the per-bin weight
        ``−½·F⁻¹[α₃(Ω)·F(I²A)]``; for a spectrally narrow field that is
        exactly the per-bin coefficient, and its energy loss is real
        (``2Re⟨A,N⟩ = −⟨|A|², α₃-weighted I²⟩``), so it cannot drift.

        Residual (documented, not modelled): 3PA-generated free carriers and
        their FCA/recombination. That is ``include_free_carriers``' separate
        TPA-based model.
        """
        src = np.abs(A) ** 2
        src = (src * src) * A
        if self._alpha3_scalar is not None:
            return np.asarray(-0.5 * self._alpha3_scalar * src, dtype=complex)
        assert self._alpha3_w is not None  # narrowed by _three_photon_enabled
        return np.asarray(
            -0.5 * self.grid.ifft(self._alpha3_w * self.grid.fft(src)),
            dtype=complex,
        )

    def _three_photon_energy_moment(self, A: NDArray) -> float:
        """Instantaneous 3PA energy-loss rate, including spectral weighting."""
        intensity = np.abs(A) ** 2
        if self._alpha3_scalar is not None:
            return float(
                self._alpha3_scalar * np.sum(intensity**3) * self.grid.dt
            )
        return float(
            -2.0
            * np.real(
                np.sum(np.conj(A) * self._three_photon_rhs(A)) * self.grid.dt
            )
        )

    def _gamma_omega_rhs(
        self, A: NDArray, intensity: NDArray, h_R_fft: NDArray | None
    ) -> NDArray:
        """Per-bin Kerr/Raman remainder ``N(A)`` in the symmetrised split form.

        ``N(A) = i√γ(Ω)·F{ P_NL(|A|²)·F⁻¹[√γ(Ω)·A] }`` — the
        energy-conserving operator of Gap I.2. ``√γ`` (diagonal and real in the
        Fourier basis) is applied to the field *inside* the polarisation and
        again outside it, which makes the operator Hermitian, so
        ``∫|A|²dt`` is conserved by the ODE for a real Raman response:
        ``⟨A, N⟩ = i⟨F⁻¹[√γÂ], P_NL·F⁻¹[√γÂ]⟩ = i∫P_NL|B|²dt`` is pure
        imaginary. The polarisation ``P_NL`` is evaluated at the **state's own**
        intensity ``|A|²``, not at the √γ-scaled ``|B|²``: with the constant-γ
        limit ``A`` bin-aligned this gives ``N = iγ|A|²A`` — the physical
        ``γ``, not ``γ²`` — and for a narrow field it reduces to the per-bin
        ``iγ(Ω)|A|²A``. That is also what the project's reference subclass
        (``uppe_probe.py::GammaOmega``) does.

        The single-sided ``iγ(Ω)·F{P_NL·A}`` form is *not* Hermitian (energy
        drifts by ~10 % on a 20 mm SCG deck, independent of ``dz``; the
        project measured 11 % on a 40 mm deck); it is reachable only through
        ``gamma_omega_naive=True`` for comparison.
        """
        g_w = self._gamma_omega
        assert g_w is not None  # only called when the per-bin γ is set
        if self._gamma_omega_naive:
            P_naive = self._nl_intensity(intensity, h_R_fft)
            return np.asarray(
                1j * self.grid.ifft(g_w * self.grid.fft(P_naive * A)),
                dtype=complex,
            )
        sq_g = np.sqrt(g_w)
        A_s = np.asarray(self.grid.ifft(sq_g * self.grid.fft(A)), dtype=complex)
        P_NL = self._nl_intensity(intensity, h_R_fft)
        return np.asarray(
            1j * self.grid.ifft(sq_g * self.grid.fft(P_NL * A_s)), dtype=complex
        )

    def _three_photon_step(self, A: NDArray, dz: float) -> NDArray:
        """Split-step 3PA sub-step: per-sample ODE ``dI/dz = −α₃I³``.

        The field form ``−α₃I²A/2`` with ``I = |A|²`` gives, per sample,
        ``dI/dz = 2Re(A* ∂_zA) = −α₃I³`` — i.e. the standard three-photon
        absorption equation, whose exact solution is

        ``I(z) = I₀ / sqrt(1 + 2α₃I₀²z)``,  ``A(z) = A₀/(1 + 2α₃I₀²z)^{1/4}``

        (the phase is untouched). A scalar α₃ is solved *exactly* per sample
        with that closed form, so a pure-3PA deck carries no step error — the
        Strang composition still limits the scheme to second order through the
        splitting. A callable α₃(Ω) uses midpoint RK2 on
        :meth:`_three_photon_rhs`, which is second-order accurate, so the
        composition stays second order.
        """
        if self._alpha3_scalar is not None:
            intensity = np.abs(A) ** 2
            denom = 1.0 + 2.0 * self._alpha3_scalar * intensity**2 * dz
            return np.asarray(A / np.sqrt(np.sqrt(denom)), dtype=complex)
        A_mid = A + 0.5 * dz * self._three_photon_rhs(A)
        return np.asarray(A + dz * self._three_photon_rhs(A_mid), dtype=complex)

    def _etdrk4_step(self, A: NDArray, dz: float) -> NDArray:
        """One full ETDRK4 step (Cox & Matthews 2002; Kassam–Trefethen 2005).

        ``∂_zA = L·A + N(A)`` with ``L`` diagonal in the Fourier basis: the
        linear propagator is carried *exactly* by the φ-function weights, and
        only the nonlinear remainder is advanced by RK4. This is the scheme
        Stage S3 evaluates; it is fourth order and, unlike the splittings,
        its accuracy is not limited by the linear phase rotation per bin —
        the stiff-regime motivation (``formal/PLAN.md`` §17.4(b)).

        Per bin ``x = L·dz`` the coefficients follow the Kassam–Trefethen
        diagonal case: each coefficient is evaluated as the mean of its
        defining function over a unit circle centred at ``x``,

        ``f(x) ≈ mean_j f(x + r·e^{iθ_j})``, θ_j = π(2j−1)/M,

        (their Eq. (2.7) specialised to scalars, radius 1, M = 48 — 32 in
        their paper). This is not a refinement but a correctness
        requirement: the GNLSE linear operator has eigenvalues approaching
        zero at the DC bin (Taylor dispersion vanishes, loss may be absent),
        and Cox & Matthews' original explicit-formula evaluation *of exactly
        these functions* there is catastrophically wrong — their own paper
        warns ``"care must be taken ... |λh| is small, to avoid rounding
        errors arising from the large amount of cancellation"``. All four
        integrands are entire functions with removable singularities, so the
        contour mean is well defined for every bin including DC.

        Stage update and final combination (K&T diagonal code, Fig. 7):

        ``a = E₂·A + Q·N₀``, ``b = E₂·A + Q·N_a``,
        ``c = E₂·a + Q·(2N_b − N₀)``,
        ``A' = E·A + f₁·N₀ + 2f₂·(N_a + N_b) + f₃·N_c``,

        with ``E = e^{x}``, ``E₂ = e^{x/2}`` per bin.

        Why the coefficients are evaluated exactly rather than by a contour:
        the failure both papers warn about is the catastrophic cancellation of
        the explicit formulas near ``x = 0`` (the DC bin — Cox & Matthews:
        "care must be taken in the evaluation of the coefficients for modes
        where |λh| is small, to avoid rounding errors arising from the large
        amount of cancellation in the coefficients"). Kassam–Trefethen repair
        that with a contour mean and, for real operators, discard the residue
        with ``real()``. Their published diagonal recipe samples a *half*
        circle (``exp(1i·π·((1:M)−.5)/M)``); its mean is exact for real
        operators but phase-destroying for complex ``L`` — the trapezoid
        residue is physics-sized there, of the order of the coefficient at
        the GNLSE's dispersion phases. The exact fix for a scalar-per-bin
        operator (each coefficient an entire function of ``x`` with a
        removable singularity) is the one Cox & Matthews themselves
        recommend: the Taylor series near zero, the defining formulas away
        from it. That is :func:`_etdrk4_scalar_coeffs`; a full-circle contour
        mean is verified equivalent (and the half-circle recipe's failure
        reproduced) in ``tests/test_gnlse_etdrk4.py``, so the failure mode
        stays demonstrated, not just asserted.
        """
        L_bin = self._etdrk_linear(dz)
        x = np.asarray(L_bin * dz, dtype=complex)
        E = np.exp(x)
        E2 = np.exp(0.5 * x)
        Q, f1, f2, f3 = _etdrk4_scalar_coeffs(x, dz)

        gamma = self._get_gamma(self._current_z)

        def rhs(state: NDArray) -> NDArray:
            return self._nonlinear_rhs(state, gamma)

        # Every nonlinear value enters the update in FOURIER space — the
        # per-bin coefficients are the frequency-domain ODE's
        # ``∂_z Â = L·Â + N̂``: N is computed in the time domain (where it
        # acts), FFT'd, then weighted bin-wise. Multiplying a time-domain
        # array by a frequency-domain coefficient array is the classic trap
        # for this scheme: it leaves a dz-independent residual that looks
        # exactly like a converged but wrong flow.
        A_w = np.asarray(self.grid.fft(A), dtype=complex)
        N0_w = np.asarray(self.grid.fft(rhs(A)), dtype=complex)
        a = np.asarray(self.grid.ifft(E2 * A_w + Q * N0_w), dtype=complex)
        Na_w = np.asarray(self.grid.fft(rhs(a)), dtype=complex)
        b = np.asarray(self.grid.ifft(E2 * A_w + Q * Na_w), dtype=complex)
        Nb_w = np.asarray(self.grid.fft(rhs(b)), dtype=complex)
        c = np.asarray(
            self.grid.ifft(E2 * (E2 * A_w + Q * N0_w) + Q * (2.0 * Nb_w - N0_w)),
            dtype=complex,
        )
        Nc_w = np.asarray(self.grid.fft(rhs(c)), dtype=complex)
        A_new_w = E * A_w + N0_w * f1 + 2.0 * (Na_w + Nb_w) * f2 + Nc_w * f3
        return np.asarray(self.grid.ifft(A_new_w), dtype=complex)

    def _rk4ip_step(self, A: NDArray, dz: float) -> NDArray:
        """One RK4IP step in the interaction picture (Hult 2007, Eq. (12)).

        The GNLSE step is ``∂_zA = L·A + N(A)`` with ``L`` diagonal in the
        Fourier basis. Hult's fourth-order Runge–Kutta *in the interaction
        picture* (J. Hult, *J. Lightwave Technol.* **25**(12), 3770 (2007),
        Eq. (12); Hochbruck & Ostermann, *Acta Numerica* **19**, 209 (2010))
        applies the interaction-picture transform at the **outer** level, so
        the linear operator is never *split* against the nonlinearity — it is
        carried exactly, and only the nonlinear remainder is advanced by
        classical RK4:

        ```
        A_I = D(A, +h/2)                     # transform to the picture
        k₁  = D(h·N̂(A), +h/2)               # N̂ evaluated in the normal picture
        k₂  = h·N̂(A_I + k₁/2)
        k₃  = h·N̂(A_I + k₂/2)
        k₄  = h·N̂(D(A_I + k₃, +h/2))        # one half transform back and forth
        A⁺  = D(A_I + (k₁ + 2k₂ + 2k₃)/6, +h/2) + k₄/6
        ```

        where :meth:`_linear_step` is the exact diagonal propagator ``D(A, h)``
        (dispersion + loss) and :meth:`_nonlinear_rhs` is the nonlinear
        remainder ``N̂(A)`` — the same vector field ETDRK4 advances, so
        Raman, self-steepening and the pcGNLSE shock variant ride along.

        The separation distance is the step *midpoint* (``+h/2`` transforms), which
        is the symmetric choice of the source; the four half transforms and four
        nonlinear evaluations are the published 8 FFTs per step of the method
        (each nonlinear evaluation adds its own Raman convolution / shock
        derivative transforms on top, exactly as in the splittings' ``N`` step).

        **No splitting error.** ``exp(h(L+N))`` is approximated by the RK4
        composition of the transformed flow rather than by a product of
        ``exp(hL)`` and ``exp(hN)``, so the ``[L, N]`` commutator error that
        the splittings carry is simply absent. (This does *not* by itself
        make every edge diagnostic vanish: the Blow & Wood (A5) photon
        number of example 45 is weighted by ``1/(ω₀+Ω)`` and dominated by
        the near-singular long-wavelength edge, so on that deck it is an
        edge-sensitive diagnostic and not a clean scheme ranking — see
        ``formal/CHANGELOG.md`` Stage S5.) For a purely linear flow
        (``N̂ = 0``) the scheme integrates exactly — pinned by
        ``tests/test_gnlse_rk4ip.py``, which also measures the fourth-order
        rate on a scalar problem with an exact reference.

        Stage S5 of ``formal/PLAN.md`` §18. The TPA and free-carrier operators
        are applied by the split-step nonlinear stage and are *not* part of
        ``N̂``, so they are rejected here rather than silently dropped.
        """
        if self.include_tpa or self.include_free_carriers:
            raise NotImplementedError(
                "scheme='rk4ip' advances only the nonlinear remainder N̂(A) "
                "(Kerr + Raman + shock); the TPA and free-carrier operators "
                "live in the split-step nonlinear stage and are not part of "
                "it. Use scheme='strang' (default) or 'yoshida4' for those "
                "decks."
            )

        gamma = self._get_gamma(self._current_z)

        def rhs(state: NDArray) -> NDArray:
            return self._nonlinear_rhs(state, gamma)

        half = 0.5 * dz
        # Transform into the interaction picture at the step midpoint.
        A_I = self._linear_step(A, half)
        # k₁ is a nonlinear evaluation in the *normal* picture, projected
        # forward by the same half transform.
        k1 = self._linear_step(dz * rhs(A), half)
        k2 = dz * rhs(A_I + 0.5 * k1)
        k3 = dz * rhs(A_I + 0.5 * k2)
        # k₄ is evaluated one half-step ahead: transform there, evaluate,
        # and stay in the normal picture for the final combination.
        k4 = dz * rhs(self._linear_step(A_I + k3, half))
        return (
            self._linear_step(A_I + (k1 + 2.0 * k2 + 2.0 * k3) / 6.0, half) + k4 / 6.0
        )

    def _strang_step(self, A: NDArray, dz: float) -> NDArray:
        """One symmetric Strang step ``exp(L dz/2) · exp(N dz) · exp(L dz/2)``.

        This is Blow & Wood (1989) Eq. (17), second order in ``dz``. The error
        comes from the noncommutativity of ``L`` and ``N``; Blow & Wood state
        it "is a series in odd powers of δz whose leading term is proportional
        to δz³ … and so the scheme is globally second-order accurate."
        """
        A = self._linear_step(A, dz / 2)
        A, self._U = self._nonlinear_step(A, dz)
        A = self._linear_step(A, dz / 2)
        return A

    def _advance(self, A: NDArray, dz: float) -> NDArray:
        """Advance the field by ``dz`` using the configured composition scheme.

        ``scheme="strang"`` (default) is a single second-order symmetric step
        and is bit-identical to the historical behaviour.

        ``scheme="yoshida4"`` is the Suzuki triple-jump composition of three
        Strang steps,

            S₂(w₀·dz) S₂(w₁·dz) S₂(w₀·dz),

        with ``w₀ = 1/(2 − 2^{1/3}) ≈ 1.35120719`` and
        ``w₁ = −2^{1/3}/(2 − 2^{1/3}) ≈ −1.70241438`` (Yoshida, *Phys. Lett. A*
        **150**(5–7), 262 (1990), Eqs. (4.1), (4.5), (4.6)). The weights
        satisfy ``2w₀ + w₁ = 1``, so the three sub-steps advance exactly ``dz``;
        and ``2w₀³ + w₁³ = 0`` cancels the leading ``δz³`` error term, giving a
        globally fourth-order method at 3 Strang steps per ``dz``.

        Note ``w₁`` is **negative**: one sub-step runs backwards. That is the
        error-extrapolation half of the construction, not an oversight, and it
        is why the scheme costs 3 Strang steps for the same distance.

        ``scheme="etdrk4"`` does not split at all (Cox & Matthews 2002;
        Kassam–Trefethen 2005): the whole step is the semilinear exponential
        RK4 on ``∂_zA = L·A + N(A)`` with ``L`` diagonal in the Fourier basis,
        so the stiff dispersion phase is carried exactly by the φ-function
        weights and only the nonlinear remainder is advanced by RK4.
        Fourth order at one RK4 stage set per ``dz`` (4 nonlinear evaluations,
        ~12 FFTs — vs. 3 × 2 FFTs plus the shock substeps for ``yoshida4``).
        See :meth:`_etdrk4_step` for the coefficient evaluation and the
        DC-bin failure mode it guards against (``formal/PLAN.md`` §17.4(b),
        Stage S3).

        ``scheme="rk4ip"`` applies Hult's fourth-order Runge–Kutta in the
        interaction picture (2007, Eq. (12)) at the *outer* level: the
        exact linear propagator is never split against the nonlinearity, so
        there is no ``[L, N]`` commutator error, and the scheme is fourth
        order at 4 nonlinear evaluations plus 4 half transforms per ``dz``
        (8 FFTs, the source's published cost). Stage S5 of
        ``formal/PLAN.md`` §18; see :meth:`_rk4ip_step`.
        """
        if self.scheme == "strang":
            return self._strang_step(A, dz)
        if self.scheme == "etdrk4":
            return self._etdrk4_step(A, dz)
        if self.scheme == "rk4ip":
            return self._rk4ip_step(A, dz)
        for weight in YOSHIDA4_WEIGHTS:
            A = self._strang_step(A, weight * dz)
        return A

    def propagate(
        self,
        num_steps: int,
        *,
        nsaves: int | None = 2,
        show_progress: bool = False,
        raman_noise: bool = False,
        noise_seed: int | None = None,
    ) -> None:
        """Run split-step simulation for num_steps steps.

        Parameters
        ----------
        num_steps : int
            Target number of split-steps along the fiber (actual integration
            steps may be higher if adaptive stepping shrinks ``dz`` below
            ``length/num_steps``).
        nsaves : int, optional
            Number of evenly spaced snapshots to retain along ``z`` (including
            ``z=0`` and ``z=L``); default 2 keeps peak memory flat in step count.
            ``None`` is the expensive diagnostic that stores every step (measured
            1016 MB at 2000 steps, 1777 MB at 4000, 3278 MB at 8000). Use ~200
            for contour plots, as in laserfun's ``NLSE(..., nsaves=200)``.
        show_progress : bool
            If True, show a ``tqdm`` progress bar over propagation distance.
            Requires ``tqdm`` (``pip install tqdm``).
        raman_noise : bool
            If True, inject spontaneous-Raman noise (Dudley Eq. 5 ``Γ_R``)
            once per step. Default False (bit-identical to legacy runs).
            Requires ``include_raman=True`` with a Raman response on the fiber.
        noise_seed : int, optional
            Seed for the per-step noise generator; same seed gives
            bit-identical output. ``None`` draws nondeterministically.
        """
        from photonics_helper.pulse import Wave

        noise_rng: np.random.Generator | None = None
        h_R_fft_noise = None
        if raman_noise:
            if not self.include_raman or self.raman_response is None:
                raise ValueError(
                    "raman_noise=True requires include_raman=True and a "
                    "Raman response on the fiber."
                )
            noise_rng = np.random.default_rng(noise_seed)
            h_R_fft_noise = self._get_h_R_fft()

        length = self.fiber.length.as_m
        dz_base = length / num_steps
        z = 0.0
        self.evolution = []
        self._energy_vs_z = None
        self._photon_vs_z = None
        self._z_positions = [0.0]
        self.w3pa = 0.0
        self.w3pa_energy = 0.0
        self.w3pa_spectral = 0.0
        self._w3pa_vs_z = [] if self._three_photon_enabled else None

        pbar = None
        if show_progress:
            try:
                from tqdm import tqdm
            except ImportError as exc:
                raise ImportError(
                    "show_progress=True requires tqdm. Install with: pip install tqdm"
                ) from exc
            pbar = tqdm(
                total=length,
                unit="m",
                unit_scale=True,
                desc="GNLSE propagation",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            )

        if nsaves is not None:
            if nsaves < 2:
                raise ValueError(f"nsaves must be >= 2, got {nsaves}")
            save_z = np.linspace(0.0, length, nsaves)
            next_save_idx = 1
        else:
            save_z = None
            next_save_idx = None

        def _append_snapshot() -> None:
            wave = Wave(
                grid=self.grid,
                envelope=self.pulse.envelope,
                central_wavelength=self.pulse.central_wavelength,
            )
            wave._pulse_train_field = self.A.copy()
            self.evolution.append(wave)
            if self._energy_vs_z is None:
                self._energy_vs_z = []
            self._energy_vs_z.append(float(np.sum(np.abs(self.A) ** 2) * self.grid.dt))
            if self.include_self_steepening:
                # With self-steepening the B&W (A5) photon number — not the
                # unweighted energy — is the invariant; record both so callers
                # can check the conserved one.
                if self._photon_vs_z is None:
                    self._photon_vs_z = []
                self._photon_vs_z.append(self._photon_number(self.A))
            if self._w3pa_vs_z is not None:
                self._w3pa_vs_z.append(self.w3pa)

        def _maybe_save(current_z: float) -> None:
            nonlocal next_save_idx
            if save_z is None:
                _append_snapshot()
                return
            assert next_save_idx is not None
            while (
                next_save_idx < len(save_z)
                and current_z >= save_z[next_save_idx] - 1e-15
            ):
                _append_snapshot()
                next_save_idx += 1

        # Store initial pulse (via _append_snapshot so a field supplied through
        # Wave.with_field / an arbitrary array is preserved in evolution[0]).
        _append_snapshot()

        # Termination tolerance: relative to fiber length with absolute floor.
        # Prevents infinite loop when shrink × remaining < ½ ulp(z).
        # For short fibers (<1 m), uses relative tolerance instead of fixed 1 nm.
        z_tol = max(1e-15, length * 1e-9)

        while z < length:
            remaining = length - z
            # Defense in depth: stop once within a negligible sliver of the end.
            # Prevents infinite loop when shrink × remaining < ½ ulp(z).
            if remaining <= z_tol:
                break

            # Determine step size
            if self.step_size is not None:
                dz = min(self.step_size.as_m, remaining)
            else:
                dz_adaptive = self._adaptive_step_size(self.A)
                dz = min(dz_adaptive, dz_base, remaining)
                if dz_base > 2.0 * dz_adaptive and z == 0.0:
                    recommended = max(
                        int(np.ceil(length / max(dz_adaptive, 1e-30))),
                        num_steps,
                    )
                    warnings.warn(
                        f"num_steps={num_steps} may be too coarse for this regime "
                        f"(dz={dz_base:.2e} m > 2× adaptive limit {dz_adaptive:.2e} m). "
                        f"Consider num_steps>={recommended} or call "
                        "GNLSESolver.estimate_num_steps().",
                        UserWarning,
                        stacklevel=3,
                    )

            # For z-dependent profile, apply gradient-based shrink
            if self._is_z_dependent:
                shrink = self._gradient_shrink_factor(z, dz_base)
                dz = dz * shrink

            # Update current z before steps
            self._current_z = z

            # Advance one step of length `dz` under the configured scheme.
            # 3PA ledger: trapezoid of ∫I²dt (and of ∫I³dt, the energy-closing
            # moment) over the step — second-order accurate and
            # scheme-independent, so every scheme reports the same quantity.
            if self._three_photon_enabled:
                absA = np.abs(self.A)
                f2_before = float(np.sum(absA**4) * self.grid.dt)
                f3_before = float(np.sum(absA**6) * self.grid.dt)
                spectral_before = self._three_photon_energy_moment(self.A)
            self.A = self._advance(A=self.A, dz=dz)
            if self._three_photon_enabled:
                absA = np.abs(self.A)
                self.w3pa += (
                    0.5 * dz * (f2_before + float(np.sum(absA**4) * self.grid.dt))
                )
                self.w3pa_energy += (
                    0.5 * dz * (f3_before + float(np.sum(absA**6) * self.grid.dt))
                )
                self.w3pa_spectral += 0.5 * dz * (
                    spectral_before + self._three_photon_energy_moment(self.A)
                )

            if noise_rng is not None:
                from .noise import raman_noise_field

                assert h_R_fft_noise is not None  # set whenever noise_rng is
                self.A = self.A + raman_noise_field(
                    self.grid,
                    h_R_fft_noise,
                    rng=noise_rng,
                    omega0=float(self.omega0),
                ) * np.sqrt(dz)

            z_new = z + dz
            if z_new <= z:
                raise RuntimeError(
                    f"SplitStepEngine.propagate: no forward progress at z={z!r} m "
                    f"(dz={dz!r} m, length={length!r} m). "
                    "Floating-point limit cycle — check step-size logic."
                )
            z = z_new
            self._current_z = z

            if save_z is None:
                self._z_positions.append(z)

            _maybe_save(z)

            if pbar is not None:
                pbar.update(dz)
                pbar.set_postfix(
                    steps=len(self._z_positions) - 1,
                    saved=len(self.evolution),
                    refresh=False,
                )

        if (
            save_z is not None
            and next_save_idx is not None
            and next_save_idx < len(save_z)
        ):
            _append_snapshot()

        if pbar is not None:
            # Cosmetic: fill the bar to 100% if we stopped early via z_tol.
            if z < length:
                pbar.update(length - z)
            pbar.close()

        if save_z is not None:
            self._z_positions = save_z[: len(self.evolution)].tolist()

        # Energy-conservation monitor (REPORT P2 item 5): with unitary loss
        # mechanisms disabled (α = 0, TPA off), Σ|A|²·dt must be conserved by
        # the split-step flow (Kerr/Raman phase + unitary shock shift). Warn
        # instead of raising so exploratory runs are never blocked.
        self._emit_energy_drift_warning()

        # Compute spectra vs z
        omega = self.grid.w
        spectra = np.zeros((len(self.evolution), len(omega)))
        spec_iter = enumerate(self.evolution)
        if show_progress and len(self.evolution) > 50:
            from tqdm import tqdm

            spec_iter = tqdm(
                spec_iter,
                total=len(self.evolution),
                desc="FFT spectra",
                unit="slice",
            )
        for i, wave in spec_iter:
            A_w = self.grid.fft(wave.envelope_field)
            spectra[i] = np.abs(A_w) ** 2
        self._spectra_vs_z = (omega, spectra)

    @property
    def z_array(self) -> NDArray:
        """Array of propagation distances (m) for each evolution entry."""
        return np.array(self._z_positions)

    def _emit_energy_drift_warning(self) -> None:
        """Warn if the conserved quantity drifted >5% with loss/TPA disabled.

        Which quantity is *conserved* depends on the operator set (Blow & Wood
        1989, Appendix): with self-steepening the GNLSE conserves the
        frequency-weighted photon number (A5) and the unweighted energy
        legitimately changes as the spectrum moves; without it, the unweighted
        energy is the invariant. Checking the wrong invariant produces false
        alarms on ordinary Raman+shock runs (a −14 % energy change with the
        photon number constant to 1e-5 was the motivating case).
        """
        if self.include_self_steepening:
            if not self._photon_vs_z:
                return
            p0 = self._photon_vs_z[0]
            p1 = self._photon_vs_z[-1]
            if p0 <= 0.0:
                return
            drift = abs(p1 / p0 - 1.0)
            what = "photon number"
        else:
            if not self._energy_vs_z:
                return
            e0 = self._energy_vs_z[0]
            e1 = self._energy_vs_z[-1]
            if e0 <= 0.0:
                return
            drift = abs(e1 / e0 - 1.0)
            what = "energy"
        lossless = (
            self._get_alpha(self._current_z) == 0.0
            and not self.include_tpa
            and not self._three_photon_enabled
        )
        if lossless and drift > 0.01:
            warnings.warn(
                f"GNLSE {what} drift {drift * 100:.2f}% exceeds 1% with "
                f"loss/TPA disabled. This is the coarse conservation guard; "
                "it can stay below 1% while the spectrum is still "
                "under-resolved. Reduce the step size (increase num_steps) "
                "and cross-check with GNLSESolver.verify_convergence().",
                UserWarning,
                stacklevel=3,
            )

    @property
    def photon_number_vs_z(self) -> NDArray:
        """Blow & Wood (1989) App. Eq. (A5) photon number at each saved step.

        Recorded only when ``include_self_steepening=True`` (the condition this
        invariant applies to, and the case where
        :meth:`_validate_shock_grid` guarantees ``ω₀ + Ω > 0``). Same length as
        ``z_array``. See :meth:`_photon_number`.
        """
        if self._photon_vs_z is None:
            raise RuntimeError(
                "photon_number_vs_z is only recorded when "
                "include_self_steepening=True; call propagate() first."
            )
        return np.asarray(self._photon_vs_z, dtype=float)

    @property
    def spectra_vs_z(self) -> tuple[NDArray, NDArray]:
        """Tuple of (frequency array, spectra at each step)."""
        if self._spectra_vs_z is None:
            raise RuntimeError("Call propagate() first.")
        return self._spectra_vs_z

    @property
    def energy_vs_z(self) -> NDArray:
        """Pulse energy ``Σ|A|²·dt`` at each saved step (same length as ``z_array``).

        **Not an invariant when self-steepening is on.** The GNLSE conserves
        the frequency-weighted photon number (Blow & Wood 1989, App. Eq.
        (A5)); the unweighted energy legitimately changes as the spectrum
        moves between frequencies. With ``include_self_steepening=True`` use
        :attr:`photon_number_vs_z` for the conserved quantity.
        """
        if self._energy_vs_z is None:
            raise RuntimeError("Call propagate() first.")
        return np.asarray(self._energy_vs_z, dtype=float)

    @property
    def w3pa_vs_z(self) -> NDArray:
        """Cumulative 3PA ledger ``∫∫|A|⁴ dt dz`` at each saved step.

        Same length as :attr:`z_array`; starts at 0 at ``z = 0``. Only
        recorded when ``three_photon_alpha_3`` is set (otherwise the term is
        not in the model and the ledger raises rather than reporting a
        meaningless zero). For the scalar α₃ operator the energy budget is
        closed by :attr:`w3pa_energy` (``E_out = E_in − α₃·w3pa_energy``).
        """
        if self._w3pa_vs_z is None:
            raise RuntimeError(
                "w3pa_vs_z is only recorded when three_photon_alpha_3 > 0; "
                "call propagate() first."
            )
        return np.asarray(self._w3pa_vs_z, dtype=float)


# ---------------------------------------------------------------------------
# Conservation + self-convergence diagnostics
# ---------------------------------------------------------------------------

#: Relative drift of the conserved invariant above which a run is flagged.
DRIFT_WARN_PCT = 1.0


def _max_drift_pct(series: NDArray | list[float]) -> float:
    """Maximum relative departure from the initial value, in percent."""
    arr = np.asarray(series, dtype=float)
    if arr.size < 2:
        return 0.0
    if not np.all(np.isfinite(arr)):
        # A blown-up / NaN run must never read as "0 % drift".
        return float("inf")
    if arr[0] == 0.0:
        return 0.0
    return float(np.max(np.abs(arr - arr[0])) / abs(arr[0]) * 100.0)


def _smooth_1d(y: NDArray, window: int) -> NDArray:
    """Moving-average smoothing (length-preserving, edge = zero pad)."""
    window = int(window)
    if window <= 1 or y.size <= window:
        return np.asarray(y, dtype=float)
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(np.asarray(y, dtype=float), kernel, mode="same")


def _sorted_spectrum(grid: Any, A: NDArray, omega0: float) -> tuple[NDArray, NDArray]:
    """Power spectrum on an ascending wavelength axis.

    Returns ``(wavelength_nm, |Â|²)`` both sorted by wavelength, so the pair
    is directly comparable and plottable. Bins whose absolute frequency
    ``ω₀ + Ω`` would be non-positive are dropped (the wavelength map is only
    physical for positive optical frequency).
    """
    S = np.abs(grid.fft(np.asarray(A, dtype=complex))) ** 2
    omega = omega0 + grid.w
    valid = omega > 0.0
    S = S[valid]
    wl = 2.0 * np.pi * C_MS / omega[valid] * 1e9
    order = np.argsort(wl)
    return wl[order], S[order]


def _self_convergence_metrics(
    wl: NDArray,
    S_coarse: NDArray,
    S_fine: NDArray,
    A_coarse: NDArray,
    A_fine: NDArray,
    *,
    floor_db: float = -40.0,
    smooth_window: int = 9,
) -> tuple[float, float, float]:
    """Phase-robust comparison of two solutions on the same wavelength axis.

    Returns ``(rms_db, peak_shift_nm, overlap)`` where ``rms_db`` is the RMS
    of the smoothed dB-spectrum difference over the band where the finer
    solution is above ``floor_db``, ``peak_shift_nm`` the shift of the
    spectral peak, and ``overlap`` the modal projection
    ``|⟨A_c, A_f⟩|² / (‖A_c‖²‖A_f‖²)`` (1 = identical mode).

    The raw field L2 distance is deliberately **not** used: on a
    multi-soliton state it stays O(1) even when the spectra agree, because a
    sub-pulse position/phase change is enough to decorrelate the fields.
    """
    sc = np.asarray(S_coarse, dtype=float)
    sf = np.asarray(S_fine, dtype=float)
    sc = sc / max(float(sc.max()), 1e-300)
    sf = sf / max(float(sf.max()), 1e-300)
    db_c = 10.0 * np.log10(sc + 1e-30)
    db_f = 10.0 * np.log10(sf + 1e-30)
    band = db_f > float(floor_db)
    if int(band.sum()) < 8:
        band = np.ones_like(db_f, dtype=bool)
    diff = np.abs(_smooth_1d(db_c, smooth_window) - _smooth_1d(db_f, smooth_window))[
        band
    ]
    rms_db = float(np.sqrt(np.mean(diff**2)))
    peak_shift = float(abs(wl[int(np.argmax(sc))] - wl[int(np.argmax(sf))]))
    a = np.asarray(A_coarse, dtype=complex).ravel()
    b = np.asarray(A_fine, dtype=complex).ravel()
    denom = float(np.vdot(a, a).real * np.vdot(b, b).real)
    overlap = float(abs(np.vdot(a, b)) ** 2 / denom) if denom > 0.0 else 1.0
    # A non-finite field (blow-up) must read as maximally unconverged, not NaN.
    if not np.isfinite(rms_db):
        rms_db = float("inf")
    if not np.isfinite(peak_shift):
        peak_shift = float("inf")
    if not np.isfinite(overlap):
        overlap = 0.0
    return rms_db, peak_shift, overlap


@dataclass
class ConvergenceDiagnostics:
    """Conservation + self-convergence diagnostics for one GNLSE run.

    Produced by :meth:`GNLSESolver.verify_convergence`. ``drift_pct`` is the
    maximum relative drift of the conserved invariant across the recorded
    steps (Blow & Wood (A5) photon number with self-steepening, unweighted
    energy otherwise). The convergence columns compare the run against a
    second run at ``probe_steps`` steps: ``rms_db`` (in-band, smoothed),
    ``peak_shift_nm`` and ``overlap``.
    """

    scheme: str
    num_steps: int
    dz_m: float
    probe_steps: int
    invariant: str
    drift_pct: float
    drift_tol_pct: float
    rms_db: float
    peak_shift_nm: float
    overlap: float
    rms_tol_db: float
    peak_tol_nm: float
    overlap_tol: float
    notes: list[str] = field(default_factory=list)

    @property
    def converged(self) -> bool:
        """True when conservation and every convergence metric are within tol."""
        drift_ok = self.invariant == "n/a" or self.drift_pct <= self.drift_tol_pct
        return bool(
            drift_ok
            and self.rms_db <= self.rms_tol_db
            and self.peak_shift_nm <= self.peak_tol_nm
            and self.overlap >= self.overlap_tol
        )

    def summary(self) -> str:
        """Human-readable multi-line table; ends with the verdict."""
        probe_factor = self.probe_steps / max(self.num_steps, 1)
        verdict = "CONVERGED" if self.converged else "NOT CONVERGED"
        lines = [
            "GNLSE conservation + convergence diagnostics",
            f"  scheme={self.scheme}  steps={self.num_steps}  "
            f"dz={self.dz_m:.3e} m  probe={self.probe_steps} "
            f"({probe_factor:g}x)",
            f"  invariant drift (max): {self.drift_pct:.4f} %  "
            f"[{self.invariant}, tol {self.drift_tol_pct:.2f} %]",
            f"  self-convergence: rms {self.rms_db:.3f} dB "
            f"(tol {self.rms_tol_db:g}) | peak shift {self.peak_shift_nm:.3f} nm "
            f"(tol {self.peak_tol_nm:g}) | overlap {self.overlap:.5f} "
            f"(tol {self.overlap_tol:g})",
        ]
        lines.extend(f"  note: {note}" for note in self.notes)
        lines.append(f"  verdict: {verdict}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------


class GNLSESolver:
    """High-level GNLSE solver using split-step Fourier method.

    Parameters
    ----------
    pulse : Wave
        Input pulse envelope.
    fiber : FiberProfile
        Fiber parameters.
    betas : array_like
        Dispersion coefficients [beta2, beta3, ...] in ``ps^k/m`` (with ``Ω``
        in ``rad/ps``), as returned by ``Dispersion.get_betas()``. Pass
        ``betas_unit="s^k/m"`` (or ``"SI"``) to supply SI coefficients
        instead; they are converted to ``ps^k/m`` internally.
    betas_unit : {"ps^k/m", "s^k/m", "SI"}
        Unit of *betas*. Default ``"ps^k/m"``.
    full_beta : array_like or callable, optional
        Static full ``β(ω)`` curve (rad/m) as in
        :class:`SplitStepEngine`; when supplied *betas* may be ``None``.
    raman_response : object, optional
        Raman response override; falls back to ``fiber.raman_response``.
    include_raman : bool
        Include Raman scattering. Default True.
    include_self_steepening : bool
        Include self-steepening (shock term). Default ``False``.

        Models intensity-dependent group velocity with the shock operator
        ``(1 + (i/ω₀)∂t)`` (see :class:`SplitStepEngine`; ``τ_shock``
        settable, default ``1/ω₀``). ``False`` by default; laserfun's
        ``NLSE`` uses ``shock=True`` by default — set ``True`` when comparing
        against laserfun or reproducing Dudley-style supercontinuum demos.
    include_tpa : bool
        Include two-photon absorption (legacy spatially-averaged model).
        Default False.
    include_free_carriers : bool
        Include the opt-in time-resolved TPA/free-carrier model (see
        :class:`SplitStepEngine`). Requires ``fiber.beta_tpa`` /
        ``fiber.sigma_fca``. Default False.
    check_phase_matching : bool
        Run the phase-matching preflight and emit warnings before propagating.
    scheme : {"strang", "yoshida4", "etdrk4", "rk4ip"}
        Composition/integration scheme passed to :class:`SplitStepEngine`
        (default ``"strang"``). ``"etdrk4"`` and ``"rk4ip"`` are the two
        fourth-order no-split options; see :meth:`SplitStepEngine._advance`.
    step_size : Length | None
        Fixed split-step size (m). If ``None`` (default), the engine uses its
        adaptive step heuristic. Supply an explicit value for deterministic,
        reproducible step counts, or when the adaptive heuristic is
        inappropriate (e.g. CW/finite-background fields where the pulse-width
        based dispersion limit is not meaningful).
    gamma_omega : array_like or callable, optional
        Per-bin nonlinear coefficient γ(Ω) (1/(W·m)), passed straight to
        :class:`SplitStepEngine` (see there for the symmetrised form). Default
        ``None`` → scalar γ. There is no ``gamma_fn``/``a_eff_fn`` here; the
        engine would ignore them anyway when ``gamma_omega`` is set (it warns).
    gamma_omega_naive : bool
        Diagnostic-only non-Hermitian form of ``gamma_omega``. Default False
        (warns when enabled).
    three_photon_alpha_3 : float or callable, optional
        Three-photon absorption coefficient α₃ (engine units ``α₃/A_eff²``,
        or physical m³/W² with ``Fiber(a_eff=...)``), passed straight to
        :class:`SplitStepEngine`. Default ``None``.
    alpha_w : float, array_like or callable, optional
        Per-bin field-amplitude loss α(Ω) in 1/m, passed straight to
        :class:`SplitStepEngine`. Default ``None``.
    """

    def __init__(
        self,
        pulse: Wave,
        fiber: FiberProfile,
        betas: NDArray | None = None,
        include_raman: bool = True,
        include_self_steepening: bool = False,
        include_tpa: bool = False,
        include_free_carriers: bool = False,
        check_phase_matching: bool = False,
        step_size: Length | None = None,
        betas_unit: BetasUnit = "ps^k/m",
        tau_shock: float | None = None,
        conserving_shock: bool = False,
        scheme: Scheme = "strang",
        full_beta=None,
        raman_response=None,
        gamma_omega: NDArray | Callable[[NDArray], NDArray] | None = None,
        gamma_omega_naive: bool = False,
        three_photon_alpha_3: float | Callable[[NDArray], NDArray] | None = None,
        alpha_w: (
            float | NDArray | Callable[[NDArray], NDArray] | None
        ) = None,
    ):
        self.pulse = pulse
        self.fiber = fiber
        if betas is None and full_beta is None:
            raise ValueError("pass either betas or full_beta")
        self.full_beta = full_beta
        # Normalised to ps^k/m (Ω in rad/ps) at the boundary — see _normalize_betas.
        self.betas = (
            _normalize_betas(betas, betas_unit)
            if full_beta is None
            else np.array([], dtype=float)
        )
        self.betas_unit = _validate_betas_unit(betas_unit)
        self.raman_response = (
            raman_response if raman_response is not None else fiber.raman_response
        )
        self.include_raman = include_raman
        self.include_self_steepening = include_self_steepening
        self.include_tpa = include_tpa
        self.include_free_carriers = include_free_carriers
        self.check_phase_matching = check_phase_matching
        self.step_size = step_size
        self.conserving_shock = bool(conserving_shock)
        if scheme not in _SCHEMES:
            raise ValueError(
                f"Unknown scheme {scheme!r}; expected one of {', '.join(_SCHEMES)}."
            )
        self.scheme: Scheme = scheme
        if tau_shock is not None and not tau_shock > 0.0:
            raise ValueError(
                f"tau_shock must be positive (seconds, SI), got {tau_shock!r}"
            )
        self.tau_shock = float(tau_shock) if tau_shock is not None else None
        # Gap N (§18.1): the two Gaps J/K keywords must be reachable from the
        # documented wrapper, not only from SplitStepEngine.
        self.gamma_omega = gamma_omega
        self.gamma_omega_naive = gamma_omega_naive
        self.three_photon_alpha_3 = three_photon_alpha_3
        self.alpha_w = alpha_w
        self._evolution: list[Wave] = []
        self._z_positions: NDArray | None = None
        self._spectra_vs_z: tuple[NDArray, NDArray] | None = None
        self._energy_vs_z: NDArray | None = None
        self._photon_vs_z: NDArray | None = None
        self._preflight_report: SimulationReadinessReport | None = None
        #: Records of the most recent :meth:`propagate` call, used by
        #: :meth:`verify_convergence` to reconstruct a matching probe run.
        self._last_num_steps: int = 100
        self._last_nsaves: int | None = 2
        self._last_raman_noise: bool = False

    def _make_engine(self, step_size: Length | None) -> SplitStepEngine:
        """Build a :class:`SplitStepEngine` from this solver's configuration."""
        return SplitStepEngine(
            pulse=self.pulse,
            fiber=self.fiber,
            betas=self.betas,
            include_raman=self.include_raman,
            include_self_steepening=self.include_self_steepening,
            include_tpa=self.include_tpa,
            include_free_carriers=self.include_free_carriers,
            conserving_shock=self.conserving_shock,
            tau_shock=self.tau_shock,
            step_size=step_size,
            scheme=self.scheme,
            full_beta=self.full_beta,
            raman_response=self.raman_response,
            gamma_omega=self.gamma_omega,
            gamma_omega_naive=self.gamma_omega_naive,
            three_photon_alpha_3=self.three_photon_alpha_3,
            alpha_w=self.alpha_w,
        )

    def propagate(
        self,
        num_steps: int = 100,
        *,
        nsaves: int | None = 2,
        show_progress: bool = False,
        raman_noise: bool = False,
        noise_seed: int | None = None,
        verify: bool = False,
        verify_probe_factor: int = 2,
        verify_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Run split-step simulation for num_steps steps.

        Parameters
        ----------
        verify : bool
            If True, run :meth:`verify_convergence` after propagating and
            print its table (a second, finer propagation; roughly doubles the
            runtime). Default False.
        verify_probe_factor : int
            Step-count multiplier for the verification probe (default 2).
        verify_kwargs : dict, optional
            Extra keyword arguments forwarded to :meth:`verify_convergence`.
        """
        if self.check_phase_matching:
            from .phase_matching import emit_readiness_warnings

            report = self.preflight_report
            if report is not None:
                emit_readiness_warnings(report)

        engine = self._make_engine(self.step_size)
        engine.propagate(
            num_steps,
            nsaves=nsaves,
            show_progress=show_progress,
            raman_noise=raman_noise,
            noise_seed=noise_seed,
        )
        self._evolution = engine.evolution
        self._z_positions = engine.z_array
        self._spectra_vs_z = engine.spectra_vs_z
        self._energy_vs_z = engine.energy_vs_z
        self._photon_vs_z = (
            engine.photon_number_vs_z if engine.include_self_steepening else None
        )
        self._last_num_steps = int(num_steps)
        self._last_nsaves = nsaves
        self._last_raman_noise = bool(raman_noise)

        if verify:
            probe_kwargs: dict[str, Any] = dict(verify_kwargs or {})
            probe_kwargs.setdefault("probe_factor", verify_probe_factor)
            self.verify_convergence(num_steps=int(num_steps), **probe_kwargs)

    def verify_convergence(
        self,
        *,
        num_steps: int | None = None,
        probe_factor: int = 2,
        drift_tol_pct: float = DRIFT_WARN_PCT,
        rms_tol_db: float = 0.5,
        peak_tol_nm: float = 2.0,
        overlap_tol: float = 0.99,
        floor_db: float = -40.0,
        smooth_window: int = 9,
        warn: bool = True,
    ) -> ConvergenceDiagnostics:
        """Conservation + self-convergence table for the last propagation.

        Compares the stored result against a fresh run at ``probe_factor``
        times the step count and reports, in one table, the maximum drift of
        the conserved invariant and three phase-robust convergence metrics.
        A run is only ``converged`` when **both** pass.

        The probe uses uniform steps: the coarse step is ``step_size`` when it
        was fixed, otherwise ``length / num_steps``. Verification therefore
        describes uniform-step refinement; on an adaptive run it is a proxy,
        not the exact adaptive schedule.

        Parameters
        ----------
        num_steps : int, optional
            Step count of the run being verified. Defaults to the value
            recorded by the last :meth:`propagate` call.
        probe_factor : int
            Step-count multiplier for the finer run (must be >= 2).
        warn : bool
            Emit a :class:`UserWarning` when the verdict is not converged.

        Returns
        -------
        ConvergenceDiagnostics
            Also printed via :meth:`ConvergenceDiagnostics.summary`.
        """
        if not self._evolution:
            raise RuntimeError("Call propagate() before verify_convergence().")
        if probe_factor < 2:
            raise ValueError(f"probe_factor must be >= 2, got {probe_factor!r}")
        if self._last_raman_noise:
            raise ValueError(
                "verify_convergence() needs a deterministic run; raman_noise=True "
                "requires an ensemble of trials, not a single field pair."
            )

        n_coarse = int(num_steps if num_steps is not None else self._last_num_steps)
        length = self.fiber.length.as_m
        if self.step_size is not None:
            dz_coarse = self.step_size.as_m
        else:
            dz_coarse = length / max(n_coarse, 1)
        n_fine = max(n_coarse * int(probe_factor), n_coarse + 1)
        dz_fine = dz_coarse / float(probe_factor)

        fine = self._make_engine(Length(dz_fine, "m"))
        fine.propagate(n_fine, nsaves=2, show_progress=False)
        A_fine = np.asarray(fine.evolution[-1].envelope_field, dtype=complex)
        A_coarse = np.asarray(self._evolution[-1].envelope_field, dtype=complex)

        grid = self.pulse.grid
        omega0 = float(self.pulse.central_frequency)
        wl, S_coarse = _sorted_spectrum(grid, A_coarse, omega0)
        _, S_fine = _sorted_spectrum(grid, A_fine, omega0)
        rms_db, peak_shift_nm, overlap = _self_convergence_metrics(
            wl,
            S_coarse,
            S_fine,
            A_coarse,
            A_fine,
            floor_db=floor_db,
            smooth_window=smooth_window,
        )

        lossless = (
            float(self.fiber.alpha) == 0.0
            and not self.include_tpa
            and not self.include_free_carriers
        )
        notes: list[str] = []
        if lossless:
            if self.include_self_steepening:
                invariant = "photon number (Blow & Wood A5)"
                drift = _max_drift_pct(self.photon_number_vs_z)
            else:
                invariant = "energy"
                drift = _max_drift_pct(self.energy_vs_z)
        else:
            invariant = "n/a (loss/gain/free-carrier run)"
            drift = 0.0
            notes.append(
                "drift is not an error proxy here: loss/gain/TPA/free "
                "carriers change the invariant by design"
            )
        if self.step_size is None:
            notes.append(
                "adaptive run: probe uses uniform steps at the nominal "
                "dz = length/num_steps"
            )

        diag = ConvergenceDiagnostics(
            scheme=str(self.scheme),
            num_steps=n_coarse,
            dz_m=dz_coarse,
            probe_steps=n_fine,
            invariant=invariant,
            drift_pct=drift,
            drift_tol_pct=float(drift_tol_pct),
            rms_db=rms_db,
            peak_shift_nm=peak_shift_nm,
            overlap=overlap,
            rms_tol_db=float(rms_tol_db),
            peak_tol_nm=float(peak_tol_nm),
            overlap_tol=float(overlap_tol),
            notes=notes,
        )
        print(diag.summary())
        if warn and not diag.converged:
            warnings.warn(
                f"GNLSE {self.scheme} run at {n_coarse} steps is not "
                f"self-converged: drift {diag.drift_pct:.3f}% "
                f"({diag.invariant}), rms {diag.rms_db:.3f} dB, peak shift "
                f"{diag.peak_shift_nm:.3f} nm, overlap {diag.overlap:.4f}. "
                "Refine the step count before trusting spectral detail.",
                UserWarning,
                stacklevel=2,
            )
        return diag

    def propagate_auto(
        self,
        *,
        energy_tol: float = 0.01,
        edge_tol: float = 1e-3,
        max_rounds: int = 4,
        max_steps: int = 50000,
        max_N: int = 2**17,
        nsaves: int = 200,
        start_steps: int | None = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Propagate, retrying with finer steps / wider grid until clean.

        Automates the two manual knobs that produce the classic “noisy”
        GNLSE plot: too few steps (energy drift) and too narrow a time
        window (fission products wrap around the periodic FFT boundary and
        interfere with themselves — speckly fringes + diagonal streaks).
        Each round propagates, then checks:

        - **drift**: ``|E_end − E_start| / E_start`` ≤ ``energy_tol``
          (skipped for loss/gain/TPA runs, where drift is physical).
          With ``include_self_steepening=True`` the unweighted energy is
          *not* the invariant — the spectrum legitimately moves between
          frequencies — so the check runs on the Blow & Wood App. Eq. (A5)
          photon number instead (``report["drift_metric"] == "photon"``
          and ``report["photon_drift"]`` carries the checked value, while
          ``report["drift"]`` keeps the raw energy drift for inspection);
        - **edges**: energy fraction in the outer 5% of the time window at
          the output ≤ ``edge_tol`` — a direct wrap-around detector.

        A drift failure doubles the steps; an edge failure doubles the
        window at fixed ``dt`` (``N`` doubles too). Before the first round,
        a grid violating the self-steepening positivity requirement
        (``Ω_max >= ω₀``) is widened at fixed ``N`` (``Tmax`` grows, ``dt``
        grows) so the shock operator stays defined; Raman ``h_R(t)`` is
        analytic and re-evaluated on the new grid, so this is exact.
        Returns a report dict
        with ``converged``, ``steps``, ``rounds``, ``drift``, ``edge_frac``
        and ``widened``. Warns instead of raising when ``max_rounds`` is
        exhausted, so a long run's data is never thrown away.
        """
        from photonics_helper.core.grids import TemporalGrid
        from photonics_helper.pulse import Wave

        widened = False
        if self.include_self_steepening:
            for _ in range(5):
                omega_max = float(self.pulse.grid.omega_max)
                omega0 = float(self.pulse.central_frequency)
                if omega_max < omega0:
                    break
                ratio = omega_max / omega0
                old = self.pulse.grid
                new_grid = TemporalGrid(
                    N=int(old.N), Tmax=Time(old.Tmax.as_s * ratio * 1.15, "s")
                )
                self.pulse = Wave.from_envelope(
                    self.pulse.envelope,
                    self.pulse.central_wavelength,
                    grid=new_grid,
                    refractive_index=self.pulse.refractive_index,
                    repetition_rate=self.pulse.repetition_rate,
                )
                widened = True
                warnings.warn(
                    f"propagate_auto: shock grid Ω_max/ω₀={ratio:.3f}; "
                    f"widened Tmax ×{ratio * 1.15:.3f} at fixed N={old.N}.",
                    UserWarning,
                    stacklevel=2,
                )
        steps = (
            int(start_steps)
            if start_steps is not None
            else max(
                10,
                type(self).estimate_num_steps(
                    self.pulse, self.fiber, np.asarray(self.betas)
                ),
            )
        )
        report: dict[str, Any] = {}
        for _round in range(max(1, max_rounds)):
            steps = min(steps, max_steps)
            self.propagate(
                num_steps=steps, nsaves=nsaves, show_progress=show_progress, **kwargs
            )
            energy = np.asarray(self.energy_vs_z, dtype=float)
            lossless = (
                self.fiber.alpha == 0.0
                and not self.include_tpa
                and not self.include_free_carriers
            )
            drift = (
                float(abs(energy[-1] - energy[0]) / max(energy[0], 1e-300))
                if lossless
                else 0.0
            )
            # With self-steepening the conserved quantity is the B&W (A5)
            # photon number, not the unweighted energy (which Raman
            # scattering legitimately changes): converge on the invariant.
            photon_drift: float | None = None
            if self.include_self_steepening:
                try:
                    phot = np.asarray(self.photon_number_vs_z, dtype=float)
                except RuntimeError:
                    phot = np.asarray([], dtype=float)
                if phot.size >= 2 and np.all(np.isfinite(phot)):
                    photon_drift = float(
                        abs(phot[-1] - phot[0]) / max(phot[0], 1e-300)
                    )
            metric = photon_drift if photon_drift is not None else drift
            final_A = np.asarray(self._evolution[-1].envelope_field)
            edge_n = max(1, int(0.05 * final_A.shape[0]))
            edge_frac = float(
                (
                    np.sum(np.abs(final_A[:edge_n]) ** 2)
                    + np.sum(np.abs(final_A[-edge_n:]) ** 2)
                )
                / max(np.sum(np.abs(final_A) ** 2), 1e-300)
            )
            report = {
                "converged": (metric <= energy_tol) and (edge_frac <= edge_tol),
                "steps": steps,
                "rounds": _round + 1,
                "drift": drift,
                "photon_drift": photon_drift,
                "drift_metric": "photon"
                if photon_drift is not None
                else "energy",
                "edge_frac": edge_frac,
                "widened": widened,
                "N": int(self.pulse.grid.N),
                "Tmax_s": float(self.pulse.grid.Tmax.as_s),
            }
            if report["converged"]:
                return report
            if _round + 1 >= max_rounds:
                break
            if edge_frac > edge_tol:
                old = self.pulse.grid
                new_N = min(int(old.N * 2), max_N)
                if new_N <= old.N:
                    warnings.warn(
                        f"propagate_auto: edges hot ({edge_frac:.2g}) but N is at max_N={max_N}; "
                        "keeping the grid.",
                        UserWarning,
                        stacklevel=2,
                    )
                else:
                    new_grid = TemporalGrid(
                        N=new_N, Tmax=Time(old.Tmax.as_s * 2.0, "s")
                    )
                    self.pulse = Wave.from_envelope(
                        self.pulse.envelope,
                        self.pulse.central_wavelength,
                        grid=new_grid,
                        refractive_index=self.pulse.refractive_index,
                        repetition_rate=self.pulse.repetition_rate,
                    )
                    widened = True
            if metric > energy_tol and lossless:
                steps = min(steps * 2, max_steps)
        warnings.warn(
            f"propagate_auto: not converged after {report.get('rounds', max_rounds)} round(s) "
            f"({report.get('drift_metric', 'energy')}_drift="
            f"{(report.get('photon_drift') if report.get('photon_drift') is not None else report.get('drift', float('nan'))):.3g}, "
            f"edge_frac={report.get('edge_frac', float('nan')):.2g}); "
            "results kept — refine manually.",
            UserWarning,
            stacklevel=2,
        )
        return report

    @classmethod
    def estimate_num_steps(
        cls,
        pulse: Wave,
        fiber: FiberProfile,
        betas: NDArray,
        *,
        include_self_steepening: bool = False,
        include_raman: bool = False,
        safety_factor: float = 2.0,
    ) -> int:
        """Estimate split-step count from dispersion and nonlinear length scales.

        Uses the same limits as :meth:`SplitStepEngine._adaptive_step_size`,
        divided by ``safety_factor`` (larger → more steps). Useful for
        supercontinuum / soliton regimes where coarse stepping gives wrong physics.
        """
        del include_raman  # Raman does not change the step heuristic here.
        length = fiber.length.as_m
        T0 = pulse.envelope.pulse_width.as_s
        omega0 = pulse.central_frequency
        gamma = _fiber_gamma(fiber, omega0)
        # NOTE: internal step heuristic — see _adaptive_step_size: the
        # normalized-units warning is suppressed here (envelope convention),
        # direct user calls to Wave.peak_power() still warn as before.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*normalized envelope units.*")
            I_max = max(pulse.peak_power(), 1e-30)

        betas_arr = np.asarray(betas, dtype=float)
        if len(betas_arr) > 0:
            beta2 = abs(betas_arr[0]) * 1e-24
            dz_disp = T0**2 / max(beta2, 1e-30) * 0.01
        else:
            dz_disp = float("inf")

        omega_factor = 1.0
        if include_self_steepening:
            omega_factor = 1.0 + pulse.grid.omega_max / omega0
        if gamma > 0.0 and I_max > 0.0:
            dz_nl = np.pi / (4.0 * gamma * I_max * omega_factor)
        else:
            dz_nl = float("inf")

        dz = min(dz_disp, dz_nl) / max(safety_factor, 1.0)
        return max(int(np.ceil(length / max(dz, 1e-30))), 10)

    def interpolated_spectrum_db(
        self,
        wl_min: float,
        wl_max: float,
        n_wl: int,
        *,
        step_index: int = -1,
    ) -> tuple[NDArray, NDArray]:
        """Return spectrum (dB) on a uniform wavelength grid (nm).

        Uses absolute angular frequency ``omega0 + grid.w`` before converting
        to wavelength, matching :func:`plot_spectrum_vs_distance`.

        Parameters
        ----------
        wl_min, wl_max : float
            Wavelength range in nm.
        n_wl : int
            Number of wavelength samples.
        step_index : int
            Evolution index (default -1 = final step).

        Returns
        -------
        wavelength_nm, spectrum_db : 1-D arrays
        """
        if self._spectra_vs_z is None:
            raise RuntimeError("Call propagate() first.")
        omega, spectra = self._spectra_vs_z
        spec = spectra[step_index]
        omega_abs = omega + self.omega0
        wavelength_nm = 2 * np.pi * C_MS / omega_abs * 1e9
        sort_idx = np.argsort(wavelength_nm)
        wavelength_nm = wavelength_nm[sort_idx]
        spec = spec[sort_idx]
        wl_grid = np.linspace(wl_min, wl_max, n_wl)
        spec_interp = np.interp(wl_grid, wavelength_nm, spec, left=0.0, right=0.0)
        spec_db = 10 * np.log10(spec_interp + 1e-30)
        spec_db -= spec_db.max()
        return wl_grid, spec_db

    @property
    def evolution(self) -> list[Wave]:
        """List of Wave objects at each propagation step."""
        return self._evolution

    @property
    def z_array(self) -> NDArray:
        """Array of propagation distances (m) for each evolution entry."""
        if self._z_positions is None:
            raise RuntimeError("Call propagate() first.")
        return self._z_positions

    @property
    def omega0(self) -> float:
        """Carrier angular frequency (rad/s)."""
        return self.pulse.central_frequency

    @property
    def spectra_vs_z(self) -> tuple[NDArray, NDArray]:
        """Tuple of (frequency array, spectra at each step)."""
        if self._spectra_vs_z is None:
            raise RuntimeError("Call propagate() first.")
        return self._spectra_vs_z

    @property
    def energy_vs_z(self) -> NDArray:
        """Pulse energy ``Σ|A|²·dt`` at each saved step (same length as ``z_array``)."""
        if self._energy_vs_z is None:
            raise RuntimeError("Call propagate() first.")
        return self._energy_vs_z

    # NOTE (Stage S3 session, 2026-10): this passthrough existed on
    # :class:`SplitStepEngine` and :class:`TaperedGNLSESolver` from the
    # photon-number work but was missing here — `test_gnlse_photon_number.py::
    # test_solver_passthrough` caught it.
    @property
    def photon_number_vs_z(self) -> NDArray:
        """Blow & Wood (1989) App. Eq. (A5) photon number at each saved step.

        Recorded only when ``include_self_steepening=True``. Same length as
        ``z_array``; raises otherwise.
        """
        if self._photon_vs_z is None:
            raise RuntimeError(
                "photon_number_vs_z is only recorded when "
                "include_self_steepening=True; call propagate() first."
            )
        return self._photon_vs_z

    @property
    def preflight_report(self) -> SimulationReadinessReport | None:
        """Lazy preflight report from phase-matching assessment.

        Builds and caches the report on first access when
        ``check_phase_matching=True``. Returns None when disabled.
        """
        if not self.check_phase_matching:
            return None
        if self._preflight_report is None:
            from .phase_matching import assess_simulation_readiness

            report = assess_simulation_readiness(
                pulse=self.pulse,
                fiber=self.fiber,
                dispersion=None,
                betas=self.betas,
            )
            self._preflight_report = report
        return self._preflight_report


# ---------------------------------------------------------------------------
# Visualization utilities
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TaperedGNLSESolver
# ---------------------------------------------------------------------------


class TaperedGNLSESolver:
    """High-level GNLSE solver for z-dependent (tapered/dispersion-managed) waveguides.

    Wraps a `SplitStepEngine` configured with z-dependent dispersion, nonlinearity,
    and loss. Exposes the same API as `GNLSESolver`.

    Parameters
    ----------
    pulse : Wave
        Input pulse envelope.
    fiber : FiberProfile
        Fiber parameters (n2, alpha, etc.).
    dispersion_profile : ZDependentDispersion or callable
        β(ω, z) profile. Either a ZDependentDispersion table or a callable
        beta(omega: NDArray, z: float) -> NDArray.
    a_eff_fn : callable, optional
        A_eff(z) in m². Falls back to fiber.A_eff if None.
    alpha_fn : callable, optional
        α(z) in 1/m. Falls back to fiber.alpha if None.
    include_raman : bool
        Include Raman scattering. Default True.
    include_self_steepening : bool
        Include self-steepening (shock term). Default ``False`` (laserfun
        ``NLSE`` defaults to ``shock=True``).
    include_tpa : bool
        Include two-photon absorption (legacy spatially-averaged model).
        Default False.
    include_free_carriers : bool
        Include the opt-in time-resolved TPA/free-carrier model (see
        :class:`SplitStepEngine`). Requires ``fiber.beta_tpa`` /
        ``fiber.sigma_fca``. Default False.
    min_shrink_factor : float
        Floor for the gradient-based step shrink factor in z-dependent mode.
        Must be in (0, 1]. Default 0.1. Raise it (e.g. 0.3) to trade a small
        amount of accuracy for speed on tapers with mild dispersion gradients.
    betas_unit : {"ps^k/m", "s^k/m", "SI"}
        Unit contract for dispersion coefficients. The tapered solver takes its
        dispersion from *dispersion_profile* (β(ω, z), SI), so this flag is
        validated for consistency with the other solvers but does not rescale a
        coefficient array. Default ``"ps^k/m"``.
    gamma_omega : array_like or callable, optional
        Per-bin nonlinear coefficient γ(Ω) (1/(W·m)), passed to
        :class:`SplitStepEngine`. Default ``None`` → scalar γ (from
        ``gamma_fn``/``a_eff_fn``/``fiber.gamma``). When set, ``gamma_fn`` and
        ``a_eff_fn`` are **ignored** (the per-bin operator takes precedence) and
        the engine warns instead of dropping them silently.
    gamma_omega_naive : bool
        Diagnostic-only non-Hermitian form of ``gamma_omega``. Default False.
    three_photon_alpha_3 : float or callable, optional
        Three-photon absorption coefficient α₃ (engine units ``α₃/A_eff²``, or
        physical m³/W² with ``Fiber(a_eff=...)``).
    alpha_w : float, array_like or callable, optional
        Per-bin field-amplitude loss α(Ω) in 1/m, combined with the scalar
        z-dependent loss and passed to :class:`SplitStepEngine`.
    """

    def __init__(
        self,
        pulse: Wave,
        fiber: FiberProfile,
        dispersion_profile: ZDependentDispersion | Callable[[NDArray, float], NDArray],
        a_eff_fn: Callable[[float], float] | None = None,
        alpha_fn: Callable[[float], float] | None = None,
        gamma_fn: Callable[[float], float] | None = None,
        include_raman: bool = True,
        include_self_steepening: bool = False,
        include_tpa: bool = False,
        include_free_carriers: bool = False,
        check_phase_matching: bool = False,
        min_shrink_factor: float = 0.1,
        step_size: Length | None = None,
        betas_unit: BetasUnit = "ps^k/m",
        tau_shock: float | None = None,
        raman_response=None,
        gamma_omega: NDArray | Callable[[NDArray], NDArray] | None = None,
        gamma_omega_naive: bool = False,
        three_photon_alpha_3: float | Callable[[NDArray], NDArray] | None = None,
        alpha_w: (
            float | NDArray | Callable[[NDArray], NDArray] | None
        ) = None,
    ):
        self.pulse = pulse
        self.fiber = fiber
        self.raman_response = (
            raman_response if raman_response is not None else fiber.raman_response
        )
        self.betas_unit = _validate_betas_unit(betas_unit)
        if betas_unit != "ps^k/m":
            warnings.warn(
                "TaperedGNLSESolver takes its dispersion from the"
                " dispersion_profile in SI: `betas_unit` is validated against "
                "the usual unit strings but does not rescale any "
                "coefficient array here (inert flag; see REVIEW.md).",
                UserWarning,
                stacklevel=2,
            )
        self.step_size = step_size
        self.dispersion_profile = dispersion_profile
        self.a_eff_fn = a_eff_fn
        self.alpha_fn = alpha_fn
        self.gamma_fn = gamma_fn
        self.include_raman = include_raman
        self.include_self_steepening = include_self_steepening
        self.include_tpa = include_tpa
        self.include_free_carriers = include_free_carriers
        self.check_phase_matching = check_phase_matching
        self.min_shrink_factor = min_shrink_factor
        if tau_shock is not None and not tau_shock > 0.0:
            raise ValueError(
                f"tau_shock must be positive (seconds, SI), got {tau_shock!r}"
            )
        self.tau_shock = float(tau_shock) if tau_shock is not None else None
        self.gamma_omega = gamma_omega
        self.gamma_omega_naive = gamma_omega_naive
        self.three_photon_alpha_3 = three_photon_alpha_3
        self.alpha_w = alpha_w
        self._engine: SplitStepEngine | None = None
        self._evolution: list[Wave] = []
        self._z_positions: NDArray | None = None
        self._spectra_vs_z: tuple[NDArray, NDArray] | None = None
        self._energy_vs_z: NDArray | None = None
        self._photon_vs_z: NDArray | None = None
        self._preflight_report: SimulationReadinessReport | None = None
        self._strict_mode = False

    def propagate(
        self,
        num_steps: int = 100,
        strict: bool = False,
        *,
        nsaves: int | None = 2,
        show_progress: bool = False,
        raman_noise: bool = False,
        noise_seed: int | None = None,
    ) -> None:
        """Run split-step simulation for num_steps steps.

        Parameters
        ----------
        num_steps : int — number of split-step steps.
        strict : bool — if True, raise ValueError on excessive β clipping (>5%).
        nsaves : int, optional — evenly spaced snapshots along z (see SplitStepEngine).
        show_progress : bool — if True, show ``tqdm`` progress bars (requires tqdm).
        """
        self._strict_mode = strict

        if strict:
            # Pre-flight check: verify dispersion coverage
            if hasattr(self.dispersion_profile, "omegas"):
                omega_min, omega_max = (
                    float(min(self.dispersion_profile.omegas)),
                    float(max(self.dispersion_profile.omegas)),
                )
                omega_abs = self.omega0 + self.pulse.grid.w
                clipped = (
                    (omega_abs < omega_min) | (omega_abs > omega_max)
                ).sum() / len(omega_abs)
                if clipped > 0.05:
                    raise ValueError(
                        f"{clipped * 100:.1f}% of grid frequencies clipped (>5% threshold in strict mode). "
                        f"Grid: [{omega_abs.min():.2e}, {omega_abs.max():.2e}] rad/s, "
                        f"Bounds: [{omega_min:.2e}, {omega_max:.2e}] rad/s. "
                        f"Reduce bandwidth or use broader dispersion table."
                    )

        if self.check_phase_matching:
            from .phase_matching import emit_readiness_warnings

            report = self.preflight_report
            if report is not None:
                emit_readiness_warnings(report)

        # Use a dummy betas array — the z-dependent path ignores it.
        dummy_betas = np.array([0.0])
        self._engine = SplitStepEngine(
            pulse=self.pulse,
            fiber=self.fiber,
            betas=dummy_betas,
            include_raman=self.include_raman,
            include_self_steepening=self.include_self_steepening,
            include_tpa=self.include_tpa,
            include_free_carriers=self.include_free_carriers,
            dispersion_profile=self.dispersion_profile,
            a_eff_fn=self.a_eff_fn,
            alpha_fn=self.alpha_fn,
            gamma_fn=self.gamma_fn,
            min_shrink_factor=self.min_shrink_factor,
            tau_shock=self.tau_shock,
            step_size=self.step_size,
            raman_response=self.raman_response,
            gamma_omega=self.gamma_omega,
            gamma_omega_naive=self.gamma_omega_naive,
            three_photon_alpha_3=self.three_photon_alpha_3,
            alpha_w=self.alpha_w,
        )
        self._engine.propagate(
            num_steps,
            nsaves=nsaves,
            show_progress=show_progress,
            raman_noise=raman_noise,
            noise_seed=noise_seed,
        )
        self._evolution = self._engine.evolution
        self._z_positions = self._engine.z_array
        self._spectra_vs_z = self._engine.spectra_vs_z
        self._energy_vs_z = self._engine.energy_vs_z
        self._photon_vs_z = (
            self._engine.photon_number_vs_z
            if self._engine.include_self_steepening
            else None
        )

    @property
    def evolution(self) -> list[Wave]:
        """List of Wave objects at each propagation step."""
        return self._evolution

    @property
    def z_array(self) -> NDArray:
        """Array of propagation distances (m) for each evolution entry."""
        if self._z_positions is None:
            raise RuntimeError("Call propagate() first.")
        return self._z_positions

    @property
    def omega0(self) -> float:
        """Carrier angular frequency (rad/s)."""
        return self.pulse.central_frequency

    @property
    def spectra_vs_z(self) -> tuple[NDArray, NDArray]:
        """Tuple of (frequency array, spectra at each step)."""
        if self._spectra_vs_z is None:
            raise RuntimeError("Call propagate() first.")
        return self._spectra_vs_z

    @property
    def energy_vs_z(self) -> NDArray:
        """Pulse energy ``Σ|A|²·dt`` at each saved step (same length as ``z_array``).

        Not an invariant when self-steepening is on — see
        :meth:`photon_number_vs_z`.
        """
        if self._energy_vs_z is None:
            raise RuntimeError("Call propagate() first.")
        return self._energy_vs_z

    @property
    def photon_number_vs_z(self) -> NDArray:
        """Blow & Wood (1989) App. Eq. (A5) photon number at each saved step.

        Recorded only when ``include_self_steepening=True``. Same length as
        ``z_array``; raises otherwise.
        """
        if self._photon_vs_z is None:
            raise RuntimeError(
                "photon_number_vs_z is only recorded when "
                "include_self_steepening=True; call propagate() first."
            )
        return self._photon_vs_z

    @property
    def preflight_report(self) -> SimulationReadinessReport | None:
        """Lazy preflight report from phase-matching assessment.

        Builds and caches the report on first access when
        ``check_phase_matching=True``. Returns None when disabled.
        """
        if not self.check_phase_matching:
            return None
        if self._preflight_report is None:
            from .phase_matching import assess_simulation_readiness

            report = assess_simulation_readiness(
                pulse=self.pulse,
                fiber=self.fiber,
                dispersion=self.dispersion_profile,
            )
            self._preflight_report = report
        return self._preflight_report


# ---------------------------------------------------------------------------
# Evolution arrays (wavelength / time grids for contour plots)
# ---------------------------------------------------------------------------


def spectral_evolution_on_wavelength_grid(
    solver: GNLSESolver,
    wl_min: float,
    wl_max: float,
    n_wl: int,
) -> tuple[NDArray, NDArray, NDArray]:
    """Interpolate stored spectra onto a uniform wavelength grid (nm).

    Uses absolute angular frequency ``omega0 + grid.w`` before converting to
    wavelength. Returns spectra in dB normalized to the global peak.

    Parameters
    ----------
    solver : GNLSESolver
        Solver with propagated results.
    wl_min, wl_max : float
        Wavelength range (nm).
    n_wl : int
        Number of wavelength samples.

    Returns
    -------
    z_m, wavelength_nm, spectra_dB
        ``z_m`` has shape (n_z,), ``wavelength_nm`` (n_wl,), ``spectra_dB`` (n_z, n_wl).
    """
    omega, spectra = solver.spectra_vs_z
    omega_abs = omega + solver.omega0
    wavelength_nm = 2 * np.pi * C_MS / omega_abs * 1e9
    sort_idx = np.argsort(wavelength_nm)
    wavelength_nm = wavelength_nm[sort_idx]
    spectra = spectra[:, sort_idx]

    wl_grid = np.linspace(wl_min, wl_max, n_wl)
    n_z = spectra.shape[0]
    interpolated = np.zeros((n_z, n_wl))
    for i in range(n_z):
        interpolated[i] = np.interp(
            wl_grid, wavelength_nm, spectra[i], left=0.0, right=0.0
        )

    global_max = interpolated.max()
    spectra_dB = 10 * np.log10(interpolated / global_max + 1e-30)
    return solver.z_array, wl_grid, spectra_dB


def spectral_evolution_on_frequency_grid(
    solver: GNLSESolver,
    f_min_THz: float,
    f_max_THz: float,
    n_f: int,
) -> tuple[NDArray, NDArray, NDArray]:
    """Interpolate stored spectra onto a uniform absolute-frequency grid (THz).

    Returns
    -------
    z_m, f_THz, spectra_dB
        ``spectra_dB`` has shape (n_z, n_f), normalized to global peak = 0 dB.
    """
    omega, spectra = solver.spectra_vs_z
    f_THz = (omega + solver.omega0) / (2 * np.pi) / 1e12
    sort_idx = np.argsort(f_THz)
    f_THz = f_THz[sort_idx]
    spectra = spectra[:, sort_idx]

    f_grid = np.linspace(f_min_THz, f_max_THz, n_f)
    n_z = spectra.shape[0]
    interpolated = np.zeros((n_z, n_f))
    for i in range(n_z):
        interpolated[i] = np.interp(f_grid, f_THz, spectra[i], left=0.0, right=0.0)

    global_max = interpolated.max()
    spectra_dB = 10 * np.log10(interpolated / global_max + 1e-30)
    return solver.z_array, f_grid, spectra_dB


def temporal_evolution_intensity(
    solver: GNLSESolver,
) -> tuple[NDArray, NDArray, NDArray]:
    """Build |A(t)|² at each stored propagation step.

    Returns
    -------
    z_m, t_ps, intensity
        ``intensity`` has shape (n_z, N_time).
    """
    z_m = solver.z_array
    t_ps = solver.pulse.grid.t * 1e12
    intensity = np.array(
        [np.abs(w.envelope_field) ** 2 for w in solver.evolution], dtype=float
    )
    return z_m, t_ps, intensity


def plot_waterfall(
    solver: GNLSESolver, ax=None, dB: bool = True, offset_scale: float = 1.0
) -> plt.Figure:
    """Pulse envelope waterfall plot (envelope vs propagation distance).

    Each saved trace ``i`` is drawn as ``y_norm + i * offset_scale`` where
    ``y_norm`` is the trace normalized to unit range (0..1 linear, or
    -1..0 for dB scaled by 90 dB). Y-tick labels show the propagation
    distance of each trace, so offsets are documented and interpretable.

    Parameters
    ----------
    solver : GNLSESolver
        Solver with propagated results.
    ax : matplotlib Axes, optional
        Axis to plot on. Creates new figure if None.
    dB : bool
        Plot in dB scale. Default True.
    offset_scale : float
        Vertical spacing between consecutive traces in normalized units.
        Must be > 0. Default 1.0.

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.transforms import ScaledTranslation

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    if not offset_scale > 0.0:
        raise ValueError(f"offset_scale must be positive, got {offset_scale!r}")
    z_steps = solver.z_array
    t = solver.pulse.grid.t * 1e12  # ps

    # True ridge/waterfall rendering: each trace is plotted at its own data
    # values (not pre-offsetted numbers) and shifted vertically through
    # per-artist transforms. The y axis therefore maps to *trace offset
    # index* units consistently, and the z labels at the trace baselines
    # come from the same offsets (ISSUES.md #4: ridge redraw).
    z_mm = z_steps * 1e3
    norm = plt.Normalize(float(z_mm.min()), float(z_mm.max())) if len(z_mm) else None
    cmap = plt.get_cmap("viridis")

    for i, wave in enumerate(solver.evolution):
        envelope = wave.envelope_field
        if dB:
            intensity_dB = 10 * np.log10(np.abs(envelope) ** 2 + 1e-30)
            y = intensity_dB - intensity_dB.max()  # 0 down to ~-90 dB
            y = y / 90.0  # scale into roughly [-1, 0]
        else:
            y = np.abs(envelope)
            y = y / max(y.max(), 1e-30)  # 0..1
        color = cmap(norm(z_mm[i])) if norm is not None else "b"
        # Per-trace vertical offset through an artist transform: the data
        # values stay untouched, so y-axis interpretation stays exact.
        offset = ScaledTranslation(0.0, i * offset_scale, fig.dpi_scale_trans)
        ax.plot(t, y, color=color, linewidth=0.5, transform=ax.transData + offset)

    if norm is not None:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Propagation distance (mm)")

    # Label y ticks with the z distance of each trace so the offsets are
    # documented on the axis itself (one label per trace).
    n_traces = len(solver.evolution)
    ax.set_yticks([i * offset_scale for i in range(n_traces)])
    ax.set_yticklabels([f"{z * 1e3:.2f} mm" for z in z_steps])
    ax.set_ylim(-0.1, (n_traces - 1) * offset_scale + 1.1)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Propagation distance (z)")
    ax.set_title("Pulse Evolution Waterfall Plot")
    return fig


def plot_spectrum_vs_distance(
    solver: GNLSESolver, ax=None, dB: bool = True
) -> plt.Figure:
    """Spectrum vs propagation distance contour (legacy wrapper).

    See :func:`plot_spectral_evolution` for wavelength range and dynamic-range
    control. Uses millimetres on the distance axis and the ``hot`` colormap
    for backward compatibility.
    """
    return plot_spectral_evolution(solver, ax=ax, dB=dB, z_scale="mm", cmap="hot")


def plot_spectral_evolution(
    solver: GNLSESolver,
    ax=None,
    *,
    wl_min: float | None = None,
    wl_max: float | None = None,
    f_min_THz: float | None = None,
    f_max_THz: float | None = None,
    n_points: int = 400,
    dynamic_range_db: float = 40.0,
    dB: bool = True,
    cmap: str = "viridis",
    z_scale: str = "m",
    use_imshow: bool = True,
) -> plt.Figure:
    """Spectral evolution contour: wavelength or frequency vs propagation distance.

    Parameters
    ----------
    solver : GNLSESolver
        Solver with propagated results.
    ax : matplotlib Axes, optional
        Axis to plot on. Creates new figure if None.
    wl_min, wl_max : float, optional
        Wavelength limits (nm) when plotting vs wavelength.
    f_min_THz, f_max_THz : float, optional
        Absolute frequency limits (THz) when plotting vs frequency.
        If both frequency limits are set they take precedence over wavelength.
    n_points : int
        Number of samples along the spectral axis.
    dynamic_range_db : float
        Color scale spans [global_max - dynamic_range_db, global_max] in dB.
    dB : bool
        Plot log-scale intensity (recommended). If False, plots linear power.
    cmap : str
        Matplotlib colormap name.
    z_scale : str
        ``"m"`` for metres on the distance axis, ``"mm"`` for millimetres.
    use_imshow : bool
        If True, use ``imshow`` with ``origin='lower'`` (laserfun style).

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    use_frequency = f_min_THz is not None and f_max_THz is not None
    if use_frequency:
        assert f_min_THz is not None and f_max_THz is not None
        z_m, x_axis, spectra_dB = spectral_evolution_on_frequency_grid(
            solver, f_min_THz, f_max_THz, n_points
        )
        x_label = "Frequency (THz)"
        x_min, x_max = f_min_THz, f_max_THz
    else:
        center_wl_nm = solver.pulse.central_wavelength.as_nm
        if wl_min is None:
            wl_min = 0.25 * center_wl_nm
        if wl_max is None:
            wl_max = 4.0 * center_wl_nm
        z_m, x_axis, spectra_dB = spectral_evolution_on_wavelength_grid(
            solver, wl_min, wl_max, n_points
        )
        x_label = "Wavelength (nm)"
        x_min, x_max = wl_min, wl_max

    z_plot = z_m * 1e3 if z_scale == "mm" else z_m
    z_label = (
        "Propagation distance (mm)" if z_scale == "mm" else "Propagation distance (m)"
    )

    if dB:
        vmin, vmax = -dynamic_range_db, 0.0
        data = spectra_dB
    else:
        vmin, vmax = None, None
        data = 10 ** (spectra_dB / 10.0)

    if use_imshow:
        ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            extent=(x_min, x_max, z_plot[0], z_plot[-1]),
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
    else:
        ax.pcolormesh(
            x_axis, z_plot, data, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(z_label)
    ax.set_title("Spectral evolution")
    return fig


def plot_temporal_evolution(
    solver: GNLSESolver,
    ax=None,
    *,
    t_min: float | None = None,
    t_max: float | None = None,
    dynamic_range_db: float = 40.0,
    cmap: str = "viridis",
    z_scale: str = "m",
    use_imshow: bool = True,
    time_reversal: bool = False,
) -> plt.Figure:
    """Temporal evolution contour: time vs propagation distance.

    Parameters
    ----------
    solver : GNLSESolver
        Solver with propagated results.
    ax : matplotlib Axes, optional
        Axis to plot on. Creates new figure if None.
    t_min, t_max : float, optional
        Time window in ps (in the plotted convention). Default: full grid.
    dynamic_range_db : float
        Color scale spans [global_max - dynamic_range_db, global_max] in dB.
    cmap : str
        Matplotlib colormap name.
    z_scale : str
        ``"m"`` or ``"mm"`` for the distance axis.
    time_reversal : bool
        Kept for back-compatibility with pre-#0 notebooks; default False.
        Post-ISSUES.md #0 the internal time grid ALREADY uses the standard
        literature (Agrawal) comoving direction -- Raman-shifted solitons
        appear at positive delay without any flip (chirp and spectra are
        unaffected). Set True only to reproduce the historical (pre-#0)
        mirrored axis.

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    z_m, t_ps, intensity = temporal_evolution_intensity(solver)
    if time_reversal:
        t_ps = -t_ps
        order = np.argsort(t_ps)
        t_ps = t_ps[order]
        intensity = intensity[:, order]
    global_max = intensity.max()
    intensity_dB = 10 * np.log10(intensity / global_max + 1e-30)

    if t_min is None:
        t_min = float(t_ps[0])
    if t_max is None:
        t_max = float(t_ps[-1])

    z_plot = z_m * 1e3 if z_scale == "mm" else z_m
    z_label = (
        "Propagation distance (mm)" if z_scale == "mm" else "Propagation distance (m)"
    )

    if use_imshow:
        ax.imshow(
            intensity_dB,
            aspect="auto",
            origin="lower",
            extent=(t_ps[0], t_ps[-1], z_plot[0], z_plot[-1]),
            vmin=-dynamic_range_db,
            vmax=0.0,
            cmap=cmap,
        )
    else:
        ax.pcolormesh(
            t_ps,
            z_plot,
            intensity_dB,
            shading="auto",
            cmap=cmap,
            vmin=-dynamic_range_db,
            vmax=0.0,
        )
    ax.set_xlim(t_min, t_max)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel(z_label)
    ax.set_title("Temporal evolution")
    return fig


def _default_wl_bounds(
    solver: GNLSESolver, fraction: float = 0.5
) -> tuple[float, float]:
    """Default wavelength window (nm) around the carrier: ``(1-f)·λ0, (1+f)·λ0``."""
    pump_nm = float(solver.pulse.central_wavelength.as_nm)
    return (pump_nm * (1.0 - fraction), pump_nm * (1.0 + fraction))


def _classify_features(wavelength_nm: NDArray, pump_nm: float) -> NDArray:
    """Label each wavelength bin as a SC feature (used for Plotly hover text).

    The classification is intentionally simple and is based on the offset from
    the pump: normal-GVD / blue components are labelled *dispersive wave*, the
    region around the carrier is *SPM / pump*, and the anomalous-GVD side is a
    *Raman soliton*.  It is a guide for interactive exploration, not a
    substitute for checking the dispersion.
    """
    offset = np.asarray(wavelength_nm, dtype=float) - pump_nm
    labels = np.full(wavelength_nm.shape, "SPM / pump", dtype=object)
    labels[offset < -80.0] = "Dispersive wave (blue)"
    labels[offset > 80.0] = "Raman soliton (red)"
    return labels


def _field_feature_labels(
    fields,
    omega: NDArray,
    omega0: float,
    pump_nm: float,
    *,
    time_reversal: bool = False,
    n_bands: int = 24,
    wl_range: tuple[float, float] = (400.0, 1400.0),
    floor_db: float = 40.0,
) -> NDArray:
    """Coarse filter-bank feature labels for each ``(z, time)`` cell.

    A single delay can carry several spectrally distinct components (a far-red
    Raman soliton and the blue dispersive wave can overlap), so each cell is
    labelled by the spectral band with the most local energy rather than by
    its time offset.  Cells more than ``floor_db`` below the global peak are
    labelled ``"low-level background"``.

    Parameters
    ----------
    fields : iterable of complex NDArray
        Stored field snapshots, one per propagation step.
    omega : NDArray
        Angular-frequency offset grid (rad/s), as in ``pulse.grid.w``.
    omega0 : float
        Carrier angular frequency (rad/s).
    pump_nm : float
        Carrier wavelength (nm), used by :func:`_classify_features`.
    time_reversal : bool
        Mirror the returned time axis to the literature comoving convention.
    n_bands : int
        Number of logarithmically spaced wavelength bands.
    wl_range : (float, float)
        Wavelength span (nm) covered by the filter bank.
    floor_db : float
        Dynamic range (dB) below which a cell is called background.

    Returns
    -------
    labels : NDArray of object, shape (n_z, n_time)
    """
    omega = np.asarray(omega, dtype=float)
    lam = 2.0 * np.pi * C_MS / (omega0 + omega) * 1e9
    edges = np.geomspace(float(wl_range[0]), float(wl_range[1]), n_bands + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    band_of_bin = np.digitize(lam, edges) - 1
    band_masks = [(band_of_bin == b) for b in range(int(n_bands))]

    snapshots = list(fields)
    n_z = len(snapshots)
    n_t = np.asarray(snapshots[0], dtype=complex).shape[0]
    labels: NDArray[np.object_] = np.empty((n_z, n_t), dtype=object)
    for iz, snapshot in enumerate(snapshots):
        spectrum = np.fft.fftshift(np.fft.fft(np.asarray(snapshot, dtype=complex)))
        best = np.full(n_t, -np.inf)
        best_band = np.zeros(n_t, dtype=int)
        for b, mask in enumerate(band_masks):
            if not mask.any():
                continue
            profile = np.abs(np.fft.ifft(np.fft.ifftshift(spectrum * mask))) ** 2
            update = profile > best
            best[update] = profile[update]
            best_band[update] = b
        row = np.asarray(_classify_features(centers[best_band], pump_nm), dtype=object)
        peak = float(best.max()) if np.isfinite(best).any() else 0.0
        if peak > 0.0:
            row[best < peak * 10.0 ** (-abs(floor_db) / 10.0)] = "low-level background"
        labels[iz] = row

    if time_reversal:
        labels = labels[:, ::-1]
    return labels


def plot_scg_dashboard(
    *,
    z_axis: NDArray,
    wl_nm: NDArray,
    spec_db: NDArray,
    t_ps: NDArray,
    int_db: NDArray,
    out_wl_nm: NDArray,
    out_spec_db: NDArray,
    out_t_ps: NDArray,
    out_int_db: NDArray,
    spec_labels: NDArray,
    t_labels: NDArray,
    title: str = "",
    z_label: str = "Distance (m)",
    wl_bounds: tuple[float, float] | None = None,
    t_bounds: tuple[float, float] | None = None,
    dynamic_range_db: float = 40.0,
    colorscale: str = "Jet",
    annotate_features: bool = True,
    height: int = 850,
) -> FigureLike:
    """Interactive four-panel Plotly SCG dashboard with feature hover labels.

    Layout: output line profiles ``(a) intensity (dB) vs wavelength`` and
    ``(b) intensity (dB) vs time`` on top, with the taller ``(c)`` spectral and
    ``(d)`` temporal evolution heatmaps below.  Hovering a heatmap cell
    reports wavelength/time, distance, power and the local feature class.

    This is the low-level array consumer used by
    :func:`plot_spectral_temporal_summary` with ``plotly=True`` and by the
    Dudley Fig. 3 reproduction (which stores an ``Evolution`` rather than a
    solver).  Requires the optional ``plotly`` dependency.

    Parameters
    ----------
    z_axis : NDArray
        Propagation-distance values for the heatmap y axis.
    wl_nm : NDArray
        Uniform wavelength grid (nm) for the spectral heatmap columns.
    spec_db : NDArray, shape (n_z, n_wl)
        Spectral density in dB (peak 0, clipped to the dynamic range).
    t_ps : NDArray
        Time grid (ps, plotted convention) for the temporal heatmap columns.
    int_db : NDArray, shape (n_z, n_time)
        Temporal intensity in dB (peak 0, clipped to the dynamic range).
    out_wl_nm, out_spec_db : NDArray
        Output spectrum line for panel (a).
    out_t_ps, out_int_db : NDArray
        Output intensity line for panel (b).
    spec_labels : NDArray
        Per-wavelength feature labels, shape ``(n_wl,)`` or ``(n_z, n_wl)``.
    t_labels : NDArray
        Per-time feature labels, shape ``(n_time,)`` or ``(n_z, n_time)``.
    title : str
        Figure title.
    z_label : str
        Label for the heatmap distance axis.
    wl_bounds, t_bounds : (float, float), optional
        Axis ranges; default to the extent of the supplied grids.
    dynamic_range_db : float
        Colour-axis floor (dB).
    colorscale : str
        Plotly colorscale name (e.g. ``"Jet"``, ``"Viridis"``).
    annotate_features : bool
        Mark the strongest cell of each feature class.
    height : int
        Figure height in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "plotly is required for the interactive dashboard; install with "
            "`pip install plotly` or `pip install photonics-helper[plotting]`"
        ) from exc

    z_axis = np.asarray(z_axis, dtype=float)
    wl_nm = np.asarray(wl_nm, dtype=float)
    t_ps = np.asarray(t_ps, dtype=float)
    spec_db = np.asarray(spec_db, dtype=float)
    int_db = np.asarray(int_db, dtype=float)
    zones = -abs(dynamic_range_db)

    def _custom(labels, n_rows: int) -> NDArray:
        arr = np.asarray(labels, dtype=object)
        return np.tile(arr, (n_rows, 1)) if arr.ndim == 1 else arr

    spec_custom = _custom(spec_labels, spec_db.shape[0])
    t_custom = _custom(t_labels, int_db.shape[0])

    wl_lo, wl_hi = (
        (float(wl_bounds[0]), float(wl_bounds[1]))
        if wl_bounds is not None
        else (float(wl_nm[0]), float(wl_nm[-1]))
    )
    t_lo, t_hi = (
        (float(t_bounds[0]), float(t_bounds[1]))
        if t_bounds is not None
        else (float(t_ps[0]), float(t_ps[-1]))
    )

    fig = make_subplots(
        rows=2,
        cols=2,
        row_heights=[0.38, 0.62],
        subplot_titles=(
            "(a) Intensity (dB) vs wavelength",
            "(b) Intensity (dB) vs time",
            "(c) Spectral evolution",
            "(d) Temporal evolution",
        ),
        horizontal_spacing=0.09,
        vertical_spacing=0.10,
    )

    fig.add_trace(
        go.Scatter(
            x=np.asarray(out_wl_nm, dtype=float),
            y=np.asarray(out_spec_db, dtype=float),
            mode="lines",
            line=dict(color="royalblue"),
            name="output spectrum",
            hovertemplate=(
                "Wavelength: %{x:.0f} nm<br>Intensity: %{y:.1f} dB<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=np.asarray(out_t_ps, dtype=float),
            y=np.asarray(out_int_db, dtype=float),
            mode="lines",
            line=dict(color="royalblue"),
            name="output intensity",
            hovertemplate=(
                "Time: %{x:.2f} ps<br>Intensity: %{y:.1f} dB<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Heatmap(
            x=wl_nm,
            y=z_axis,
            z=spec_db,
            customdata=spec_custom,
            colorscale=colorscale,
            coloraxis="coloraxis",
            hovertemplate=(
                "Wavelength: %{x:.0f} nm<br>" + z_label + ": %{y:.3g}"
                "<br>Power: %{z:.1f} dB<br>Feature: %{customdata}<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=t_ps,
            y=z_axis,
            z=int_db,
            customdata=t_custom,
            colorscale=colorscale,
            coloraxis="coloraxis",
            hovertemplate=(
                "Time: %{x:.2f} ps<br>" + z_label + ": %{y:.3g}"
                "<br>Power: %{z:.1f} dB<br>Feature: %{customdata}<extra></extra>"
            ),
        ),
        row=2,
        col=2,
    )

    if annotate_features:
        for heat_z, xs, ys, custom, col in (
            (spec_db, wl_nm, z_axis, spec_custom, 1),
            (int_db, t_ps, z_axis, t_custom, 2),
        ):
            for name in np.unique(custom):
                mask = custom == name
                if not mask.any():
                    continue
                masked = np.where(mask, heat_z, -np.inf)
                iy, ix = np.unravel_index(int(np.argmax(masked)), masked.shape)
                fig.add_trace(
                    go.Scatter(
                        x=[xs[ix]],
                        y=[ys[iy]],
                        mode="markers+text",
                        text=[str(name).split(" (")[0]],
                        textposition="middle right",
                        marker=dict(
                            size=9, color="white", line=dict(color="black", width=1)
                        ),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=2,
                    col=col,
                )

    fig.update_xaxes(range=[wl_lo, wl_hi], row=1, col=1)
    fig.update_xaxes(range=[t_lo, t_hi], row=1, col=2)
    fig.update_xaxes(range=[wl_lo, wl_hi], row=2, col=1)
    fig.update_xaxes(range=[t_lo, t_hi], row=2, col=2)
    fig.update_xaxes(title_text="Wavelength (nm)", row=1, col=1)
    fig.update_xaxes(title_text="Time (ps)", row=1, col=2)
    fig.update_xaxes(title_text="Wavelength (nm)", row=2, col=1)
    fig.update_xaxes(title_text="Time (ps)", row=2, col=2)
    fig.update_yaxes(title_text="Intensity (dB)", range=[zones, 2.0], row=1, col=1)
    fig.update_yaxes(title_text="Intensity (dB)", range=[zones, 2.0], row=1, col=2)
    fig.update_yaxes(title_text=z_label, row=2, col=1)
    fig.update_yaxes(title_text=z_label, row=2, col=2)

    fig.update_layout(
        coloraxis=dict(
            colorscale=colorscale,
            cmin=zones,
            cmax=0.0,
            colorbar=dict(title="dB", x=1.02, y=0.28, len=0.55),
        ),
        title=title,
        template="plotly_white",
        height=height,
        showlegend=False,
    )
    return fig


def plot_spectral_temporal_summary(
    solver: GNLSESolver,
    *,
    wl_bounds: tuple[float, float] | None = None,
    wl_min: float | None = None,
    wl_max: float | None = None,
    t_bounds: tuple[float, float] | None = None,
    t_min: float | None = None,
    t_max: float | None = None,
    dynamic_range_db: float = 40.0,
    cmap: str = "jet",
    z_scale: str = "m",
    time_reversal: bool = False,
    height_ratios: tuple[float, float] = (1.0, 1.7),
    figsize: tuple[float, float] = (11.0, 9.0),
    n_points: int = 500,
    plotly: bool = False,
) -> FigureLike:
    """Four-panel GNLSE summary with line profiles on top and dense plots below.

    Layout
    ------
    (a) output intensity (dB) vs wavelength   — line
    (b) output intensity (dB) vs time         — line
    (c) spectral evolution                    — density plot (taller)
    (d) temporal evolution                    — density plot (taller)

    The two contour panels occupy a larger fraction of the figure
    (``height_ratios``) because their dynamic range carries the SC dynamics;
    the line profiles give a quantitative, easily read complement.

    Parameters
    ----------
    solver : GNLSESolver
        Solver with stored fields (call :meth:`propagate` first).
    wl_bounds : (float, float), optional
        Wavelength window in nm.  Defaults to the carrier ± 50 %.
    wl_min, wl_max : float, optional
        Individual overrides used only when *wl_bounds* is None.
    t_bounds : (float, float), optional
        Time window in ps (plotted convention).  Defaults to the full grid.
    t_min, t_max : float, optional
        Individual overrides used only when *t_bounds* is None.
    dynamic_range_db : float
        Colour/floor range below the peak for the density plots and the line
        profiles.
    cmap : str
        Matplotlib colormap for the density plots.
    z_scale : {"m", "cm", "mm"}
        Propagation-distance unit on the density-plot y axes.
    time_reversal : bool
        Use the standard literature (Agrawal/Dudley) comoving time (default).
    height_ratios : (float, float)
        Relative heights of the line row and the (taller) contour row.
    figsize : (float, float)
        Figure size in inches.
    n_points : int
        Spectral samples used by the Plotly path.
    plotly : bool
        If True, return an interactive ``plotly.graph_objects.Figure`` whose
        heatmap hover labels each cell as DW / SPM / Raman soliton (see
        :func:`plot_scg_dashboard`).  ``write_html`` it for a standalone page,
        or use :func:`save_summary_html`.

    Returns
    -------
    fig : matplotlib Figure or plotly Figure
        The assembled figure (not closed), so the caller can adjust it.
    """
    if wl_bounds is not None:
        wl_lo, wl_hi = float(wl_bounds[0]), float(wl_bounds[1])
    else:
        default_lo, default_hi = _default_wl_bounds(solver)
        wl_lo = default_lo if wl_min is None else float(wl_min)
        wl_hi = default_hi if wl_max is None else float(wl_max)
    if wl_hi <= wl_lo:
        raise ValueError(f"wl_bounds must be increasing, got ({wl_lo}, {wl_hi})")

    # Output spectrum (a): PSD on the absolute wavelength grid, in dB.
    grid = solver.pulse.grid
    field = np.asarray(solver.evolution[-1].envelope_field, dtype=complex)
    spectrum = np.abs(grid.fft(field)) ** 2
    wavelength = 2.0 * np.pi * C_MS / (solver.omega0 + grid.w) * 1e9
    order = np.argsort(wavelength)
    wavelength, spectrum = wavelength[order], spectrum[order]
    spectrum_db = 10.0 * np.log10(spectrum / spectrum.max() + 1e-30)

    # Output temporal intensity (b), in the same time convention as panel (d).
    t_ps = np.asarray(grid.t, dtype=float) * 1e12
    intensity = np.abs(field) ** 2
    if time_reversal:
        t_ps = -t_ps
        order_t = np.argsort(t_ps)
        t_ps, intensity = t_ps[order_t], intensity[order_t]
    intensity_db = 10.0 * np.log10(intensity / intensity.max() + 1e-30)

    if t_bounds is not None:
        t_lo, t_hi = float(t_bounds[0]), float(t_bounds[1])
    else:
        t_lo = float(t_ps[0]) if t_min is None else float(t_min)
        t_hi = float(t_ps[-1]) if t_max is None else float(t_max)
    if t_hi <= t_lo:
        raise ValueError(f"t_bounds must be increasing, got ({t_lo}, {t_hi})")

    if plotly:
        z_evo, wl_grid, spectra_db = spectral_evolution_on_wavelength_grid(
            solver, wl_lo, wl_hi, n_points
        )
        pump_nm = float(solver.pulse.central_wavelength.as_nm)
        _, t_grid, intensity_evo = temporal_evolution_intensity(solver)
        if time_reversal:
            t_grid = -t_grid
            order_evo = np.argsort(t_grid)
            t_grid = t_grid[order_evo]
            intensity_evo = intensity_evo[:, order_evo]
        evo_peak = max(float(intensity_evo.max()), 1e-30)
        int_db = np.clip(
            10.0 * np.log10(intensity_evo / evo_peak + 1e-30),
            -abs(dynamic_range_db),
            0.0,
        )
        spec_db = np.clip(spectra_db, -abs(dynamic_range_db), 0.0)
        t_labels = _field_feature_labels(
            [wave.envelope_field for wave in solver.evolution],
            solver.pulse.grid.w,
            solver.omega0,
            pump_nm,
            time_reversal=time_reversal,
        )
        return plot_scg_dashboard(
            z_axis=z_evo,
            wl_nm=wl_grid,
            spec_db=spec_db,
            t_ps=t_grid,
            int_db=int_db,
            out_wl_nm=wavelength,
            out_spec_db=spectrum_db,
            out_t_ps=t_ps,
            out_int_db=intensity_db,
            spec_labels=_classify_features(wl_grid, pump_nm),
            t_labels=t_labels,
            title=(
                f"GNLSE summary — {solver.pulse.central_wavelength.as_nm:.0f} nm, "
                f"{solver.fiber.length.as_m * 100:.1f} cm"
            ),
            z_label="Distance (m)",
            wl_bounds=(wl_lo, wl_hi),
            t_bounds=(t_lo, t_hi),
            dynamic_range_db=dynamic_range_db,
        )

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=list(height_ratios))
    ax_spec = fig.add_subplot(gs[0, 0])
    ax_time = fig.add_subplot(gs[0, 1])
    ax_evo_spec = fig.add_subplot(gs[1, 0])
    ax_evo_time = fig.add_subplot(gs[1, 1])

    ax_spec.plot(wavelength, spectrum_db, color="C0", lw=1.2)
    ax_spec.set_xlim(wl_lo, wl_hi)
    ax_spec.set_ylim(-dynamic_range_db, 2.0)
    ax_spec.set_xlabel("Wavelength (nm)")
    ax_spec.set_ylabel("Intensity (dB)")
    ax_spec.set_title("(a) Intensity (dB) vs wavelength")
    ax_spec.grid(True, which="both", alpha=0.3)

    ax_time.plot(t_ps, intensity_db, color="C0", lw=1.2)
    ax_time.set_xlim(t_lo, t_hi)
    ax_time.set_ylim(-dynamic_range_db, 2.0)
    ax_time.set_xlabel("Time (ps)")
    ax_time.set_ylabel("Intensity (dB)")
    ax_time.set_title("(b) Intensity (dB) vs time")
    ax_time.grid(True, which="both", alpha=0.3)

    plot_spectral_evolution(
        solver,
        ax=ax_evo_spec,
        wl_min=wl_lo,
        wl_max=wl_hi,
        dynamic_range_db=dynamic_range_db,
        cmap=cmap,
        z_scale=z_scale,
    )
    ax_evo_spec.set_title("(c) Spectral evolution")

    plot_temporal_evolution(
        solver,
        ax=ax_evo_time,
        t_min=t_lo,
        t_max=t_hi,
        dynamic_range_db=dynamic_range_db,
        cmap=cmap,
        z_scale=z_scale,
        time_reversal=time_reversal,
    )
    ax_evo_time.set_title("(d) Temporal evolution")

    return fig


def save_summary_html(
    solver: GNLSESolver,
    path,
    **kwargs,
) -> Path:
    """Render the interactive summary and write a standalone HTML file.

    Builds :func:`plot_spectral_temporal_summary` with ``plotly=True`` and
    writes a self-contained ``.html`` document (Plotly.js loaded from the
    CDN).  This is the automated path that turns a propagated solver into the
    interactive Fig. 3-style dashboard described in the reproduction docs.

    Parameters
    ----------
    solver : GNLSESolver
        Solver with stored fields.
    path : str or pathlib.Path
        Output HTML path; parent directories are created.
    **kwargs
        Forwarded to :func:`plot_spectral_temporal_summary` (for example
        ``wl_bounds``, ``t_bounds``, ``dynamic_range_db``).

    Returns
    -------
    pathlib.Path
        The written path.
    """
    from pathlib import Path

    out = Path(path)
    kwargs.pop("plotly", None)
    fig: Any = plot_spectral_temporal_summary(solver, plotly=True, **kwargs)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)
    return out


def gnlse_spectrogram(
    solver: GNLSESolver,
    *,
    gate: NDArray | None = None,
    n_delays: int = 161,
    delay_span_ps: float | None = None,
    snapshot: int = -1,
    wl_bounds: tuple[float, float] | None = None,
    n_wavelength: int = 500,
    time_reversal: bool = False,
) -> tuple[NDArray, NDArray, NDArray]:
    """Cross-correlation spectrogram of a propagated field (GNLSE Eq. 4).

    Computes

    ``Σ(Ω, τ) = |∫ E(t) g(t − τ) e^{−iΩt} dt|²``

    with the complex envelope ``E(t)`` taken from the last stored field (or
    ``snapshot``) and a real gate ``g`` (by default the input pulse envelope,
    as in Dudley et al., *Rev. Mod. Phys.* **78**, 1135 (2006), Fig. 10).  This
    is the cross-correlation FROG trace, not a conventional STFT: the gate is
    the actual pulse rather than a fixed window, and the trace preserves the
    relative phase information of the pulse.

    Parameters
    ----------
    solver : GNLSESolver
        Solver with stored fields (call :meth:`propagate` first).
    gate : NDArray, optional
        Real gate sampled on ``solver.pulse.grid.t``.  Defaults to the input
        pulse envelope ``real(E_in(t))``.
    n_delays : int
        Number of delay samples.
    delay_span_ps : float, optional
        Half-range of the delay axis in ps.  Defaults to 40 % of the full
        temporal window.
    snapshot : int
        Index into ``solver.evolution`` of the field to analyse (default last).
    wl_bounds : (float, float), optional
        Wavelength window ``(wl_min, wl_max)`` in nm.  Defaults to the carrier
        wavelength ± 50 %, i.e. ``(0.5·λ0, 1.5·λ0)``; pass an explicit tuple to
        zoom or widen.
    n_wavelength : int
        Number of samples on the uniform wavelength grid.
    time_reversal : bool
        Kept for back-compatibility with pre-#0 notebooks; default False.
        Post-ISSUES.md #0 the internal time grid already uses the standard
        literature (Agrawal) comoving time ``T`` (Raman red-shift -> positive
        delay); spectra and the Raman red-shift are unaffected either way.
        Set True only to reproduce the historical (pre-#0) mirrored axis.

    Returns
    -------
    delay_ps : NDArray, shape (n_delays,)
    wavelength_nm : NDArray, shape (n_wavelength,)
    spectrogram : NDArray, shape (n_delays, n_wavelength)
        Power spectrogram (arb. units; normalise with its maximum).
    """
    grid = solver.pulse.grid
    t = np.asarray(grid.t, dtype=float)
    omega = np.asarray(grid.w, dtype=float)

    if not solver.evolution:
        raise ValueError("solver has no stored fields; call propagate() first")
    field = np.asarray(solver.evolution[snapshot].envelope_field, dtype=complex)

    if gate is None:
        gate = np.real(solver.pulse.envelope.field(t))
    gate = np.asarray(gate, dtype=float)
    if gate.shape != t.shape:
        raise ValueError(
            f"gate shape {gate.shape} does not match the temporal grid {t.shape}"
        )

    if delay_span_ps is None:
        delay_span_ps = 0.4 * (t.max() - t.min()) * 1e12
    delays = np.linspace(-delay_span_ps, delay_span_ps, n_delays) * 1e-12

    trace: NDArray[np.float64] = np.empty((n_delays, t.size), dtype=float)
    for i, tau in enumerate(delays):
        gated = field * np.interp(t - tau, t, gate)
        trace[i] = np.abs(np.fft.fftshift(np.fft.fft(gated))) ** 2

    wl_native = 2 * np.pi * C_MS / (solver.omega0 + omega) * 1e9
    order = np.argsort(wl_native)
    wl_native = wl_native[order]
    trace = trace[:, order]

    if wl_bounds is None:
        wl_bounds = _default_wl_bounds(solver)
    wl_lo, wl_hi = float(wl_bounds[0]), float(wl_bounds[1])
    if wl_hi <= wl_lo:
        raise ValueError(f"wl_bounds must be increasing, got {wl_bounds!r}")
    wl_grid = np.linspace(wl_lo, wl_hi, int(n_wavelength))

    resampled: NDArray[np.float64] = np.empty(
        (n_delays, int(n_wavelength)), dtype=float
    )
    for i in range(n_delays):
        resampled[i] = np.interp(wl_grid, wl_native, trace[i], left=0.0, right=0.0)

    delay_ps = delays * 1e12
    if time_reversal:
        delay_ps = -delay_ps[::-1]
        resampled = resampled[::-1]

    return delay_ps, wl_grid, resampled


def _plotly_spectrogram(
    solver: GNLSESolver,
    delay_ps: NDArray,
    wavelength_nm: NDArray,
    trace_dB: NDArray,
    dynamic_range_db: float,
    t_min: float | None,
    t_max: float | None,
    annotate: bool,
) -> PlotlyFigure:
    """Interactive Plotly heatmap with feature-labelled hover text."""
    import plotly.graph_objects as go

    pump_nm = float(solver.pulse.central_wavelength.as_nm)
    labels = _classify_features(wavelength_nm, pump_nm)
    custom = np.tile(labels, (trace_dB.shape[0], 1)).T

    fig = go.Figure(
        go.Heatmap(
            x=delay_ps,
            y=wavelength_nm,
            z=trace_dB.T,
            customdata=custom,
            colorscale="Jet",
            zmin=-abs(dynamic_range_db),
            zmax=0.0,
            colorbar=dict(title="Power (dB)"),
            hovertemplate=(
                "Delay: %{x:.2f} ps"
                "<br>Wavelength: %{y:.0f} nm"
                "<br>Power: %{z:.1f} dB"
                "<br>Feature: %{customdata}"
                "<extra></extra>"
            ),
        )
    )

    if annotate:
        # Label the strongest (delay, wavelength) cell of each class.
        classes = ["Dispersive wave (blue)", "SPM / pump", "Raman soliton (red)"]
        for name in classes:
            mask = custom == name
            if not mask.any():
                continue
            idx = np.unravel_index(
                np.argmax(np.where(mask, trace_dB, -np.inf)), trace_dB.shape
            )
            fig.add_trace(
                go.Scatter(
                    x=[delay_ps[idx[1]]],
                    y=[wavelength_nm[idx[0]]],
                    mode="markers+text",
                    text=[name.split(" (")[0]],
                    textposition="middle right",
                    marker=dict(
                        size=9, color="white", line=dict(color="black", width=1)
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        title="Supercontinuum spectrogram (cross-correlation FROG)",
        xaxis_title="Delay (ps)",
        yaxis_title="Wavelength (nm)",
        template="plotly_white",
    )
    if t_min is not None or t_max is not None:
        fig.update_xaxes(range=[t_min, t_max])
    return fig


def plot_spectrogram(
    solver: GNLSESolver,
    ax=None,
    *,
    gate: NDArray | None = None,
    n_delays: int = 161,
    delay_span_ps: float | None = None,
    snapshot: int = -1,
    wl_bounds: tuple[float, float] | None = None,
    dynamic_range_db: float = 40.0,
    cmap: str = "jet",
    with_projections: bool = False,
    t_min: float | None = None,
    t_max: float | None = None,
    time_reversal: bool = False,
    plotly: bool = False,
    annotate_features: bool = True,
) -> FigureLike:
    """Plot the supercontinuum spectrogram (Dudley et al. Fig. 10 style).

    Parameters
    ----------
    solver : GNLSESolver
        Solver with stored fields.
    ax : matplotlib Axes, optional
        Axis for the spectrogram when ``with_projections=False`` and
        ``plotly=False``.
    wl_bounds : (float, float), optional
        Wavelength window ``(wl_min, wl_max)`` in nm.  Defaults to the carrier
        ± 50 %.
    with_projections : bool
        If True, build a matplotlib figure with the spectrogram plus its
        temporal (bottom) and spectral (right) projections (ignores ``ax``).
    t_min, t_max : float, optional
        Delay limits in ps (defaults to the full delay span).
    time_reversal : bool
        Use the standard literature (Agrawal) comoving time (default True).
    plotly : bool
        If True, return an interactive ``plotly.graph_objects.Figure`` instead
        of a matplotlib figure.  The hover text labels each ``(delay, λ)`` cell
        as a dispersive wave, SPM/pump, or Raman soliton, and (with
        ``annotate_features=True``) the strongest cell of each class is marked.
    annotate_features : bool
        Add feature labels in the Plotly path.  Ignored for matplotlib.

    Returns
    -------
    fig : matplotlib Figure or plotly Figure
        The figure, so the caller can adjust axes/labels afterwards.
    """
    import matplotlib.pyplot as plt

    delay_ps, wavelength_nm, trace = gnlse_spectrogram(
        solver,
        gate=gate,
        n_delays=n_delays,
        delay_span_ps=delay_span_ps,
        snapshot=snapshot,
        wl_bounds=wl_bounds,
        time_reversal=time_reversal,
    )
    peak = max(float(trace.max()), 1e-30)
    trace_dB = 10 * np.log10(trace / peak + 1e-30)

    if plotly:
        return _plotly_spectrogram(
            solver,
            delay_ps,
            wavelength_nm,
            trace_dB,
            dynamic_range_db,
            t_min,
            t_max,
            annotate_features,
        )

    wl_lo, wl_hi = float(wavelength_nm[0]), float(wavelength_nm[-1])

    def _draw(ax_main):
        return ax_main.imshow(
            trace_dB.T,
            origin="lower",
            aspect="auto",
            extent=(delay_ps[0], delay_ps[-1], wl_lo, wl_hi),
            cmap=cmap,
            vmin=-abs(dynamic_range_db),
            vmax=0.0,
        )

    if not with_projections:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure
        im = _draw(ax)
        fig.colorbar(im, ax=ax, label="Power (dB)")
        ax.set_xlabel("Delay (ps)")
        ax.set_ylabel("Wavelength (nm)")
        ax.set_title("Supercontinuum spectrogram")
        if t_min is not None or t_max is not None:
            ax.set_xlim(t_min, t_max)
        return fig

    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[4, 1, 0.18],
        height_ratios=[4, 1],
        hspace=0.05,
        wspace=0.05,
    )
    ax_main = fig.add_subplot(gs[0, 0])
    ax_spec = fig.add_subplot(gs[0, 1], sharey=ax_main)
    ax_cbar = fig.add_subplot(gs[0, 2])
    ax_time = fig.add_subplot(gs[1, 0], sharex=ax_main)

    im = _draw(ax_main)
    fig.colorbar(im, cax=ax_cbar, label="Power (dB)")
    ax_main.set_ylabel("Wavelength (nm)")
    ax_main.tick_params(labelbottom=False)

    spectral_marginal = trace.sum(axis=0)
    marginal_peak = max(float(spectral_marginal.max()), 1e-30)
    ax_spec.plot(spectral_marginal / marginal_peak, wavelength_nm, "k-", lw=0.8)
    ax_spec.tick_params(labelleft=False)
    ax_spec.set_xlabel("Spectrum")

    grid = solver.pulse.grid
    intensity = np.abs(solver.evolution[snapshot].envelope_field) ** 2
    t_plot = grid.t * 1e12
    if time_reversal:
        t_plot = -t_plot
    order = np.argsort(t_plot)
    ax_time.plot(
        delay_ps,
        np.interp(delay_ps, t_plot[order], intensity[order]),
        "k-",
        lw=0.8,
    )
    ax_time.set_xlabel("Delay (ps)")
    ax_time.set_ylabel("Intensity")
    if t_min is not None or t_max is not None:
        ax_main.set_xlim(t_min, t_max)
    fig.suptitle("Supercontinuum spectrogram (cross-correlation FROG)")
    return fig


def plot_intensity_metrics(solver: GNLSESolver, ax=None) -> plt.Figure:
    """Peak power and pulse width vs propagation distance.

    Parameters
    ----------
    solver : GNLSESolver
        Solver with propagated results.
    ax : matplotlib Axes, optional
        Axis to plot on. Creates new figure with two subplots if None.

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    else:
        # ax must be a Figure (has subplots), not a single Axes
        if hasattr(ax, "subplots"):
            fig = ax
            ax1, ax2 = ax.subplots(2, 1)
        else:
            raise ValueError(
                "ax must be None or a matplotlib Figure, not a single Axes. "
                "Pass ax=None to let the function create its own figure."
            )

    z_steps = solver.z_array
    z_mm = z_steps * 1e3

    peak_power = np.array([w.peak_power() for w in solver.evolution])
    t = solver.pulse.grid.t
    pulse_width = np.array(
        [
            np.sqrt(
                np.sum(t**2 * np.abs(w.envelope_field) ** 2)
                / np.sum(np.abs(w.envelope_field) ** 2)
            )
            for w in solver.evolution
        ]
    )

    ax1.plot(z_mm, peak_power)
    ax1.set_ylabel("Peak Power (W)")
    ax1.set_title("Peak Power vs Distance")
    ax1.grid(True)

    ax2.plot(z_mm, pulse_width * 1e12)  # ps
    ax2.set_xlabel("Propagation distance (mm)")
    ax2.set_ylabel("Pulse Width (ps)")
    ax2.set_title("Pulse Width vs Distance")
    ax2.grid(True)

    fig.suptitle("Intensity Metrics vs Propagation Distance")
    fig.tight_layout()
    return fig
