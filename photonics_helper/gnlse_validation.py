"""GNLSE validation and convergence harness.

Two related tools for *demonstrating numerical credibility on the user's own
configuration* (the "research-first" validation stance of this library):

1. :func:`convergence_study` — run a caller-supplied GNLSE factory at a
   sequence of refined temporal grids / step sizes and report, per
   caller-selected observable, the value at each resolution, the relative
   change between successive refinements, and a converged verdict at a
   caller tolerance (Sinkin, Holzlöhner, Zweck & Menyuk, *J. Lightwave
   Technol.* **21**, 61 (2003); Agrawal, *Nonlinear Fiber Optics*, 5th ed.,
   §2.4).

2. Cited analytical checks — :func:`check_spm`, :func:`check_mi`,
   :func:`check_soliton`, :func:`check_gordon_ssfs` — compare a *propagation
   result* against a closed-form reference for the canonical limits:

   - SPM:   Stolen & Lin, *Phys. Rev. A* **17**, 1448 (1978);
            Agrawal §4.1 (``N_peaks = floor(φ_max/π) + 1``).
   - MI:    Agrawal §5.1 (``g(Ω) = |β₂Ω|√(Ω_c²−Ω²)``, ``Ω_c² = 4γP/|β₂|``).
   - Soliton: Agrawal §5.2 (fundamental soliton, ``z_sol = (π/2) L_D``).
   - SSFS:  Gordon, *Opt. Lett.* **11**, 662 (1986);
            ``dΩ/dz = −8|β₂|T_R/(15T₀⁴)``, ``T_R = f_R ∫t·h_R(t) dt``.

The checks raise :class:`ValidationFailure` when the documented tolerance is
exceeded, so they can be wired directly into application-level test suites.

Importing this module pulls the solver stack only — no plotting backends.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_OBSERVABLES",
    "ConvergenceReport",
    "ObservableReport",
    "ObservableFn",
    "ValidationFailure",
    "convergence_study",
    "check_spm",
    "check_mi",
    "check_soliton",
    "check_gordon_ssfs",
    "gordon_ssfs_rate",
]


class ValidationFailure(AssertionError):
    """Raised by the ``check_*`` helpers when a closed-form check fails."""

    def __init__(
        self,
        case: str,
        message: str,
        metrics: dict[str, Any] | None = None,
    ):
        self.case = case
        self.metrics = metrics or {}
        suffix = f" | metrics: {self.metrics}" if self.metrics else ""
        super().__init__(f"[{case}] {message}{suffix}")


# ---------------------------------------------------------------------------
# Convergence harness
# ---------------------------------------------------------------------------

ObservableFn = Callable[[Any], float]


def _field_of(result: Any) -> np.ndarray:
    """Extract the complex envelope from a solver / evolution object."""
    if hasattr(result, "envelope_field"):
        return np.asarray(result.envelope_field)
    if hasattr(result, "A"):
        return np.asarray(result.A)
    if hasattr(result, "field"):
        return np.asarray(result.field)
    # Evolved solvers (GNLSESolver etc.) keep a list of saved states.
    evolution = getattr(result, "evolution", None)
    if evolution is not None and len(evolution) > 0:
        return np.asarray(evolution[-1].envelope_field)
    raise TypeError(
        "observables expect a Wave, evolved solver or any object exposing "
        "envelope_field / A / field / evolution; got "
        f"{type(result).__name__}"
    )


def _grid_of(result: Any) -> Any:
    grid = getattr(result, "grid", None)
    if grid is not None:
        return grid
    evolution = getattr(result, "evolution", None)
    if evolution is not None and len(evolution) > 0:
        return evolution[-1].grid
    raise TypeError(f"no grid attribute on {type(result).__name__}")


def _obs_peak_intensity(result: Any) -> float:
    return float(np.max(np.abs(_field_of(result)) ** 2))


def _obs_pulse_energy(result: Any) -> float:
    A = _field_of(result)
    return float(np.sum(np.abs(A) ** 2) * float(_grid_of(result).dt))


def _obs_rms_bandwidth(result: Any) -> float:
    A = _field_of(result)
    grid = _grid_of(result)
    spec = np.abs(grid.fft(A)) ** 2
    return float(np.sqrt(np.sum(spec * grid.w**2) / np.sum(spec)))


def _obs_rms_width(result: Any) -> float:
    A = _field_of(result)
    grid = _grid_of(result)
    t = grid.t
    i = np.abs(A) ** 2
    t0 = float(np.sum(i * t) / np.sum(i))
    var = float(np.sum(i * (t - t0) ** 2) / np.sum(i))
    return float(np.sqrt(max(var, 0.0)))


#: Built-in caller-selectable observables (name → extractor).
DEFAULT_OBSERVABLES: dict[str, ObservableFn] = {
    "peak_intensity": _obs_peak_intensity,
    "pulse_energy": _obs_pulse_energy,
    "rms_bandwidth": _obs_rms_bandwidth,
    "rms_width": _obs_rms_width,
}


@dataclass
class ObservableReport:
    """Per-observable convergence data (Sinkin et al. 2003)."""

    name: str
    values: list[float]
    changes: list[float | None]  # relative change vs previous refinement
    change_last: float | None
    converged: bool


@dataclass
class ConvergenceReport:
    """Result of :func:`convergence_study`."""

    refinements: list[dict[str, Any]] = field(default_factory=list)
    observables: list[ObservableReport] = field(default_factory=list)

    @property
    def converged(self) -> bool:
        """True when every selected observable converged at the finest pair."""
        return bool(self.observables) and all(o.converged for o in self.observables)

    def summary(self) -> str:
        lines = [f"convergence study over {len(self.refinements)} refinements:"]
        for o in self.observables:
            lines.append(
                f"  {o.name}: last change {o.change_last} → "
                + ("converged" if o.converged else "NOT converged")
            )
        return "\n".join(lines)


def convergence_study(
    build_solver: Callable[..., Any],
    refinements: Sequence[Mapping[str, Any]] | None = None,
    *,
    observables: Sequence[str] | Mapping[str, ObservableFn] | None = None,
    tolerance: float = 1e-3,
    shared: Mapping[str, Any] | None = None,
) -> ConvergenceReport:
    """Run a GNLSE factory at successive refinements and report convergence.

    Parameters
    ----------
    build_solver : callable
        ``build_solver(**refinement) -> result`` — the factory receives the
        refinement's keyword arguments (plus every entry of ``shared``) and
        must return a **propagated** result exposing the final field (a
        ``Wave``, an evolved solver with ``.A``, or anything with
        ``.envelope_field``).
    refinements : sequence of mappings
        One kwargs dict per resolution, ordered coarse → fine.
    observables : sequence[str] | mapping[str, callable], optional
        Caller-selected observables: names into :data:`DEFAULT_OBSERVABLES`
        or a custom ``{name: callable(result) -> float}`` mapping.
        Default: all defaults.
    tolerance : float
        Relative-change threshold applied between the two finest resolutions.
    shared : mapping, optional
        Factory kwargs forwarded to *every* refinement.

    Returns
    -------
    ConvergenceReport
    """
    if refinements is None or len(refinements) < 2:
        raise ValueError(
            "convergence_study needs at least two refinements (coarse → fine); "
            f"got {refinements!r}"
        )
    tol = float(tolerance)
    if tol <= 0:
        raise ValueError(f"tolerance must be positive, got {tol}")

    if observables is None:
        selected: dict[str, ObservableFn] = dict(DEFAULT_OBSERVABLES)
    elif isinstance(observables, Mapping):
        selected = dict(observables)
    else:
        selected = {}
        for name in observables:
            if name not in DEFAULT_OBSERVABLES:
                raise KeyError(
                    f"unknown observable {name!r}; available: "
                    f"{sorted(DEFAULT_OBSERVABLES)}"
                )
            selected[name] = DEFAULT_OBSERVABLES[name]

    values: dict[str, list[float]] = {name: [] for name in selected}
    for refinement in refinements:
        kwargs: dict[str, Any] = dict(shared or {})
        kwargs.update(refinement)
        result = build_solver(**kwargs)
        for name, fn in selected.items():
            values[name].append(float(fn(result)))

    reports: list[ObservableReport] = []
    for name in selected:
        vs = values[name]
        changes: list[float | None] = [None]
        for prev, cur in zip(vs, vs[1:]):
            changes.append(abs(cur - prev) / max(abs(prev), 1e-300))
        last = changes[-1]
        reports.append(
            ObservableReport(
                name=name,
                values=vs,
                changes=changes,
                change_last=last,
                converged=last is not None and last < tol,
            )
        )
    return ConvergenceReport(
        refinements=[dict(r) for r in refinements], observables=reports
    )


# ---------------------------------------------------------------------------
# Analytical checks
# ---------------------------------------------------------------------------


def check_spm(
    gamma: float,
    peak_power: float,
    t0: float,
    wavelength_m: float,
    phi_max: float,
    grid: Any,
    tolerance: float = 5e-3,
) -> dict[str, Any]:
    """Kerr-only SPM: compare the propagated spectrum with the closed form.

    For a Gaussian input with zero dispersion the exact output field is
    ``A(L,t) = √P₀ e^{−t²/2T₀²} e^{i φ_max e^{−t²/T₀²}}`` with
    ``φ_max = γP₀L`` (Stolen & Lin 1978; Agrawal §4.1). Checks (i) the
    simulated spectrum against the closed-form Fourier integral and (ii) the
    fringe rule ``N_peaks = floor(φ_max/π) + 1``.

    Parameters
    ----------
    grid : TemporalGrid
    tolerance : float
        Max absolute deviation between normalized spectra.

    Returns
    -------
    dict — the validation metrics (also returned on success).

    Raises
    ------
    ValidationFailure — when either check fails.
    """
    from scipy.signal import find_peaks

    from .base import C_MS, Length, Time, Wavelength
    from .gnlse import FiberProfile, GNLSESolver
    from .pulse import Envelope, Wave

    P0 = float(peak_power)
    T0 = float(t0)
    L = float(phi_max) / (abs(gamma) * P0)
    env = Envelope(shape="gaussian", peak_amplitude=np.sqrt(P0), pulse_width=Time(T0, "s"))
    pulse = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(wavelength_m, "m"))
    fiber = FiberProfile.from_gamma(
        gamma=gamma,
        n2=2.6e-20,
        omega0=2 * np.pi * C_MS / wavelength_m,
        length=Length(L, "m"),
    )
    solver = GNLSESolver(
        pulse=pulse,
        fiber=fiber,
        betas=np.array([0.0]),
        include_raman=False,
        include_self_steepening=False,
        include_tpa=False,
    )
    solver.propagate(num_steps=GNLSESolver.estimate_num_steps(pulse, fiber, np.array([0.0])))
    W_num = np.abs(grid.fft(solver.evolution[-1].envelope_field)) ** 2
    W_num = W_num / W_num.max()

    t = grid.t
    A0 = np.sqrt(P0) * np.exp(-(t**2) / (2 * T0**2))
    A_an = A0 * np.exp(1j * phi_max * np.exp(-(t**2) / T0**2))
    W_an = np.abs(grid.fft(A_an)) ** 2
    W_an = W_an / W_an.max()

    max_diff = float(np.max(np.abs(W_num - W_an)))
    peaks, _ = find_peaks(W_num, prominence=0.05)
    n_peaks = int(len(peaks))
    n_expected = int(np.floor(phi_max / np.pi)) + 1

    metrics = {
        "max_abs_spectrum_diff": max_diff,
        "n_peaks": n_peaks,
        "n_peaks_expected": n_expected,
        "phi_max": float(phi_max),
    }
    if max_diff >= tolerance:
        raise ValidationFailure(
            "SPM",
            f"spectrum deviation {max_diff:.2e} ≥ tolerance {tolerance:g}",
            metrics,
        )
    if n_peaks != n_expected:
        raise ValidationFailure(
            "SPM",
            f"fringe peak count {n_peaks} ≠ floor(φ_max/π)+1 = {n_expected}",
            metrics,
        )
    return metrics


def mi_gain_of(beta2: float, gamma: float, P: float, omega: float) -> float:
    """Exact linear-stability MI gain g(Ω) at a single offset (Agrawal 5.1.9).

    ``g(Ω) = |β₂Ω|√(Ω_c²−Ω²)``, ``Ω_c² = 4γP/|β₂|``, in the **power**-gain
    convention (sideband intensity grows as ``e^{g z}``) used throughout this
    library — validated by :func:`check_mi` and the Dudley Fig. 23
    reproduction (``g_max = 2γP``).
    """
    if beta2 >= 0:
        return 0.0
    omega = float(omega)
    omega_c_sq = -4.0 * gamma * P / beta2
    if omega**2 >= omega_c_sq:
        return 0.0
    return float(abs(beta2 * omega) * np.sqrt(omega_c_sq - omega**2))


def gordon_ssfs_rate(
    beta2: float, gamma: float, peak_power: float, t0: float, t_raman: float
) -> float:
    """Gordon's linear SSFS rate ``dΩ/dz`` (rad/s/m, negative = red shift).

    Gordon, *Opt. Lett.* **11**, 662 (1986), for a *fundamental soliton*:

    ``dΩ/dz = −(8/15)·|β₂|·T_R/T₀⁴``, ``T_R = f_R ∫t·h_R(t)dt``.

    The ``peak_power``/``gamma`` inputs are used to assert the soliton-order
    condition ``γP₀T₀²/|β₂| = 1`` (within 1 %); pass the soliton's peak power.
    Dimensions: ``β₂`` (s²/m) × ``T_R`` (s) / ``T₀⁴`` (s⁴) → 1/(m·s³).
    """
    b2 = float(beta2)
    if b2 >= 0:
        raise ValueError("Gordon's law requires anomalous dispersion (beta2 < 0)")
    if t0 <= 0 or t_raman < 0:
        raise ValueError(f"need t0 > 0 and T_R ≥ 0, got t0={t0}, T_R={t_raman}")
    order = float(gamma) * float(peak_power) * float(t0) ** 2 / abs(b2)
    if abs(order - 1.0) > 0.01:
        raise ValueError(
            "check_gordon_ssfs applies to a fundamental soliton: "
            f"γP₀T₀²/|β₂| = {order:.4f} ≠ 1. Rescale peak_power = |β₂|/(γT₀²)."
        )
    return -8.0 * abs(b2) * float(t_raman) / (15.0 * float(t0) ** 4)


def check_mi(
    beta2: float,
    gamma: float,
    pump_power: float,
    wavelength_m: float,
    probe_omega: float,
    grid: Any,
    length: float | None = None,
    rel_tolerance: float = 0.15,
) -> dict[str, Any]:
    """Modulation-instability gain vs the exact linear-stability result.

    Seeds a CW pump with a weak real probe perturbation
    ``ε·cos(probe_omega·t)`` (one sideband pair on the periodic grid) and
    measures the exponential **power**-gain coefficient of the sideband pair
    from the late-time slope of ``ln(P_side(z))``; compares against the closed
    form ``g(Ω) = |β₂Ω|√(Ω_c²−Ω²)``, ``Ω_c² = 4γP/|β₂|`` (Agrawal Eq. 5.1.9,
    in the power-gain convention validated against Dudley Fig. 23 by the
    ``fix-mi-gain-convention`` change: sideband *intensity* grows as
    ``e^{g z}``, so the field amplitude grows as ``e^{g z/2}``). Requires
    anomalous dispersion (``β₂ < 0``) and ``Ω_probe < Ω_c``.

    Parameters
    ----------
    grid : TemporalGrid
    length : float | None — propagation length (m). When ``None``, chosen so
        the analytic sideband power gain reaches a factor ≈ e⁴.
    rel_tolerance : float — allowed relative deviation of the measured gain.

    Raises
    ------
    ValidationFailure — when the measured gain deviates beyond tolerance.
    """
    from .base import C_MS, Length, Time, Wavelength
    from .gnlse import FiberProfile, GNLSESolver
    from .pulse import Envelope, Wave

    if beta2 >= 0:
        raise ValueError("check_mi requires anomalous dispersion (beta2 < 0)")

    P = float(pump_power)
    amp = np.sqrt(P)
    w0 = 2 * np.pi * C_MS / wavelength_m
    # Snap the probe onto the nearest resolved frequency bin (the MI gain is
    # extracted per bin, so a discrete-grid probe is required).
    pos_w = np.unique(np.abs(grid.w)[np.abs(grid.w) > 0.0])
    probe_eff = float(pos_w[np.argmin(np.abs(pos_w - probe_omega))])
    if abs(probe_eff - float(probe_omega)) / max(float(probe_omega), 1e-300) > 0.5:
        raise ValidationFailure(
            "MI",
            "requested probe_omega is not resolvable on the supplied temporal "
            "grid (increase Tmax for finer frequency bins)",
        )
    g_ref = mi_gain_of(beta2, gamma, P, probe_eff)  # power-gain coefficient
    if g_ref <= 0:
        raise ValueError(
            f"probe_omega {probe_eff:.3e} rad/s must be below the MI cutoff "
            "for the given (beta2, gamma, P)"
        )

    # CW pump (DC on the periodic grid) plus a weak real probe perturbation
    # that seeds both sidebands of the ±Ω pair. ε = 1 % keeps the dynamics
    # in the linear-stability regime while staying far above the solver's
    # numerical sideband floor.
    env = Envelope(shape="custom", peak_amplitude=amp, pulse_width=Time(1.0, "s"),
                   func=lambda t, T0, A0: np.full_like(t, A0))
    pulse = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(wavelength_m, "m"))
    eps = 1e-2
    perturb = eps * amp * np.cos(probe_eff * grid.t)
    A_in = np.asarray(pulse.envelope_field) + perturb
    pulse = pulse.with_field(A_in)

    if length is None:
        length = 4.0 / g_ref  # analytic power gain ≈ e⁴ over the run
    fiber = FiberProfile.from_gamma(
        gamma=gamma, n2=2.6e-20, omega0=w0, length=Length(length, "m")
    )
    solver = GNLSESolver(
        pulse=pulse,
        fiber=fiber,
        betas=np.array([beta2 * 1e24]),  # s²/m → ps²/m
        include_raman=False,
        include_self_steepening=False,
        include_tpa=False,
        step_size=Length(length / 4000.0, "m"),
    )
    solver.propagate(num_steps=4000, nsaves=100)
    n_saves = len(solver.evolution)
    zs = np.linspace(0.0, float(length), n_saves)

    iw = int(np.argmin(np.abs(np.abs(grid.w) - probe_eff)))
    im = int(len(grid.w) - iw)
    P_side = np.empty(len(solver.evolution))
    for k, state in enumerate(solver.evolution):
        W = np.abs(np.asarray(grid.fft(np.asarray(state.envelope_field)))) ** 2
        P_side[k] = float(W[iw] + W[im])
    # Late-time log-slope: the MI eigenmode is asymptotically exponential;
    # the cosh-like lead-in and the eigenmode interference beat contaminate
    # the first half of the trace.
    half = len(zs) // 2
    if P_side[-1] <= 0 or P_side[half] <= 0:
        raise ValidationFailure(
            "MI", "sideband power vanished (numerical noise floor)"
        )
    g_meas = float(np.polyfit(zs[half:], np.log(P_side[half:]), 1)[0])
    rel_err = abs(g_meas - g_ref) / g_ref
    metrics = {
        "g_measured": g_meas,
        "g_reference": g_ref,
        "omega_probe": float(probe_omega),
        "length": float(length),
    }
    if rel_err > rel_tolerance:
        raise ValidationFailure(
            "MI",
            f"measured power gain {g_meas:.4g} deviates from the "
            f"linear-stability reference {g_ref:.4g} by {rel_err:.1%}",
            metrics,
        )
    return metrics


def check_soliton(
    beta2: float,
    gamma: float,
    t0: float,
    wavelength_m: float,
    grid: Any,
    periods: float = 1.0,
    tolerance: float = 0.02,
) -> dict[str, float]:
    """Fundamental soliton: after ``z_sol = (π/2)·L_D`` the shape returns.

    Propagates an ``N = 1`` soliton over ``periods × z_sol`` and checks the
    field-weighted shape overlap with the input (Agrawal §5.2).

    Parameters
    ----------
    grid : TemporalGrid
    periods : float — number of soliton periods to propagate (default 1).
    tolerance : float — maximum allowed overlap deficit (1 − overlap).

    Raises
    ------
    ValidationFailure — when the overlap drops below ``1 − tolerance``.
    """
    from .base import C_MS, Length, Time, Wavelength
    from .gnlse import FiberProfile, GNLSESolver
    from .pulse import Envelope, Wave

    if beta2 >= 0:
        raise ValueError("check_soliton requires anomalous dispersion (beta2 < 0)")
    P0 = abs(float(beta2)) / (float(gamma) ** 2 * float(t0) ** 2)  # N = 1
    L_d = t0**2 / abs(beta2)
    z_sol = (np.pi / 2) * L_d
    L = periods * z_sol

    sech_fwhm_factor = 2.0 * np.arccosh(np.sqrt(2.0)) * float(t0)
    env = Envelope.from_fwhm(
        "sech", peak_amplitude=np.sqrt(P0), fwhm=Time(sech_fwhm_factor, "s")
    )
    pulse = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(wavelength_m, "m"))
    fiber = FiberProfile.from_gamma(
        gamma=gamma,
        n2=2.6e-20,
        omega0=2 * np.pi * C_MS / wavelength_m,
        length=Length(L, "m"),
    )
    solver = GNLSESolver(
        pulse=pulse,
        fiber=fiber,
        betas=np.array([beta2 * 1e24]),  # s²/m → ps²/m
        include_raman=False,
        include_self_steepening=False,
        include_tpa=False,
        step_size=Length(L / 2000.0, "m"),
    )
    solver.propagate(2000, nsaves=3)
    A_out = np.asarray(solver.evolution[-1].envelope_field)
    A_in = np.asarray(pulse.envelope_field)
    overlap = float(
        abs(np.vdot(A_in, A_out))
        / np.sqrt(abs(np.vdot(A_in, A_in)) * abs(np.vdot(A_out, A_out)))
    )
    metrics = {
        "shape_overlap": overlap,
        "energy_ratio": float(np.sum(np.abs(A_out) ** 2) / np.sum(np.abs(A_in) ** 2)),
        "periods": float(periods),
        "z_sol": float(z_sol),
    }
    if overlap < 1.0 - tolerance:
        raise ValidationFailure(
            "soliton",
            f"after {periods:g} soliton period(s) the shape overlap is only {overlap:.5f}",
            metrics,
        )
    return metrics


def check_gordon_ssfs(
    beta2: float,
    gamma: float,
    peak_power: float,
    t0: float,
    wavelength_m: float,
    raman_response: Any,
    grid: Any,
    length: float,
    ratio_tolerance: float = 0.25,
    num_steps: int = 4000,
) -> dict[str, float]:
    """Raman soliton self-frequency shift vs Gordon's analytic law.

    Propagates a soliton with the fiber's Raman response and compares the
    measured red shift of the spectral peak against Gordon's analytic rate
    ``dΩ/dz = −8|β₂|T_R/(15 T₀⁴)`` with ``T_R = f_R·∫t·h_R(t)dt`` (Gordon
    1986). The measured/analytic ratio must be within ``ratio_tolerance``
    of unity; see ``reproductions/gordon_1986_ssfs`` (ratio 1.19 on the
    published benchmark configuration).

    Parameters
    ----------
    grid : TemporalGrid
    length : float — fibre length (m).
    raman_response : RamanResponse (or any object with ``fR`` and a public
        ``h_R(t)`` / private ``_h_R(t)``).

    Raises
    ------
    ValidationFailure — when the shift direction or rate is wrong.
    """
    from .base import C_MS, Length, Time, Wavelength
    from .gnlse import FiberProfile, GNLSESolver
    from .pulse import Envelope, Wave

    if beta2 >= 0:
        raise ValueError("check_gordon_ssfs requires anomalous dispersion (beta2 < 0)")
    if raman_response is None:
        raise ValueError("check_gordon_ssfs needs a Raman response")

    f_r = float(getattr(raman_response, "fR") or 0.0)
    h_r_fn = getattr(raman_response, "h_R", None) or getattr(
        raman_response, "_h_R", None
    )
    if h_r_fn is None:
        raise TypeError("raman_response must expose h_R(t) (or _h_R(t))")
    t_raman = f_r * float(np.trapezoid(grid.t * h_r_fn(grid.t), grid.t))

    rate = gordon_ssfs_rate(beta2, gamma, peak_power, float(t0), t_raman)  # rad/s/m
    analytic_dnu = rate * float(length)  # rad/s (positive → blue; signed)
    analytic_dlam = -(wavelength_m**2) / (2 * np.pi * C_MS) * analytic_dnu * 1e9

    env = Envelope(
        shape="sech", peak_amplitude=np.sqrt(peak_power), pulse_width=Time(float(t0), "s")
    )
    pulse = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(wavelength_m, "m"))
    fiber = FiberProfile.from_gamma(
        gamma=gamma,
        n2=2.6e-20,
        omega0=float(pulse.central_frequency),
        length=Length(float(length), "m"),
        raman_response=raman_response,
    )
    solver = GNLSESolver(
        pulse=pulse,
        fiber=fiber,
        betas=np.array([beta2 * 1e24]),  # s²/m → ps²/m
        include_raman=True,
    )
    solver.propagate(num_steps=num_steps)
    nm0 = _peak_wavelength(grid, np.asarray(pulse.envelope_field), wavelength_m)
    nm1 = _peak_wavelength(
        grid, np.asarray(solver.evolution[-1].envelope_field), wavelength_m
    )
    measured = nm1 - nm0
    ratio = measured / analytic_dlam if analytic_dlam else float("nan")

    metrics = {
        "measured_shift_nm": measured,
        "analytic_shift_nm": analytic_dlam,
        "ratio": ratio,
        "T_R": t_raman,
    }
    if measured <= 0:
        raise ValidationFailure(
            "SSFS", f"Raman SSFS must red-shift, measured {measured:.4f} nm", metrics
        )
    if not (1.0 - ratio_tolerance <= ratio <= 1.0 + ratio_tolerance):
        raise ValidationFailure(
            "SSFS",
            f"measured shift {measured:.4f} nm vs Gordon {analytic_dlam:.4f} nm "
            f"(ratio {ratio:.3f} outside 1 ± {ratio_tolerance:g})",
            metrics,
        )
    return metrics


def _peak_wavelength(grid: Any, A: np.ndarray, wl0: float) -> float:
    """Spectral-peak wavelength (nm, parabolic refinement) of a field.

    The parabolic vertex is fitted in ``Ω`` space, where the DFT grid is
    *uniform* (the mapped wavelength grid is not), then converted to a
    wavelength.
    """
    from .base import C_MS

    W = np.abs(grid.fft(A)) ** 2
    w = np.asarray(grid.w, dtype=float)
    i = int(np.argmax(W))
    if 0 < i < len(W) - 1:
        y0, y1, y2 = W[i - 1], W[i], W[i + 1]
        den = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        w_peak = float(w[i] + delta * (w[i + 1] - w[i]))
    else:
        w_peak = float(w[i])
    return float((2 * np.pi * C_MS / (2 * np.pi * C_MS / wl0 + w_peak)) * 1e9)
