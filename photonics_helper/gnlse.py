"""Generalized nonlinear Schrödinger equation (GNLSE) solver.

Split-step Fourier solver for pulse propagation through nonlinear media,
supporting Kerr, Raman, self-steepening, and two-photon absorption effects.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from math import factorial
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from photonics_helper.base import C_MS, Area, Length, Time

# Accepted unit strings for dispersion coefficients passed to the solvers.
BetasUnit = Literal["ps^k/m", "s^k/m", "SI"]
_BETAS_UNITS: tuple[str, ...] = ("ps^k/m", "s^k/m", "SI")


def _validate_betas_unit(betas_unit: str) -> str:
    """Return *betas_unit* if it is an accepted unit string, else raise."""
    if betas_unit not in _BETAS_UNITS:
        raise ValueError(
            f"Unknown betas_unit {betas_unit!r}; accepted values are "
            f"{', '.join(repr(u) for u in _BETAS_UNITS)}."
        )
    return betas_unit


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
# FiberProfile
# ---------------------------------------------------------------------------


@dataclass
class FiberProfile:
    """Optical fiber/waveguide parameters for GNLSE propagation.

    Attributes:
        n2: Nonlinear refractive index n₂ (m²/W).
        alpha: Fiber loss coefficient α (1/m).
        A_eff: Effective mode area (m²).
        confinement_factor: Waveguide confinement factor Γ (1.0 for fibers, <1.0 for waveguides).
        sigma_tpa: Two-photon absorption cross-section (m²·W⁻¹). Default 0.
        carrier_lifetime: Carrier recombination lifetime (s). Default None.
        length: Fiber length (m).
        raman_response: Raman response function. Default None.
    """

    n2: float
    alpha: float
    A_eff: Area
    length: Length
    confinement_factor: float = 1.0
    sigma_tpa: float = 0.0
    carrier_lifetime: Time | None = None
    raman_response: object | None = None

    @classmethod
    def from_gamma(
        cls,
        gamma: float,
        n2: float,
        omega0: float,
        alpha: float = 0.0,
        length: Length = Length(1.0, "m"),
        confinement_factor: float = 1.0,
        sigma_tpa: float = 0.0,
        carrier_lifetime: Time | None = None,
        raman_response: object | None = None,
    ) -> FiberProfile:
        """Create a FiberProfile from a target nonlinear coefficient γ.

        Convenience constructor that mirrors laserfun's API where γ is
        specified directly (e.g. ``gamma_W_m``) rather than deriving it
        from ``n₂`` and ``A_eff``.

        Parameters
        ----------
        gamma : float — target nonlinear coefficient γ in 1/(W·m).
        n2 : float — nonlinear refractive index n₂ in m²/W.
        omega0 : float — carrier angular frequency in rad/s.
        alpha : float — fiber loss coefficient α in 1/m. Default 0.
        length : Length — fiber length. Default 1 m.
        confinement_factor : float — waveguide confinement factor Γ. Default 1.0.
        sigma_tpa : float — two-photon absorption cross-section. Default 0.
        carrier_lifetime : Time — carrier recombination lifetime. Default None.
        raman_response : object — Raman response function. Default None.

        Returns
        -------
        FiberProfile — configured with the computed A_eff.

        Notes
        -----
        Computes ``A_eff = n₂·ω₀·Γ / (c·γ)`` from the relation
        ``γ = n₂·ω₀·Γ / (c·A_eff)``.
        """
        A_eff_m2 = n2 * omega0 * confinement_factor / (C_MS * gamma)
        return cls(
            n2=n2,
            alpha=alpha,
            A_eff=Area(A_eff_m2, "m^2"),
            length=length,
            confinement_factor=confinement_factor,
            sigma_tpa=sigma_tpa,
            carrier_lifetime=carrier_lifetime,
            raman_response=raman_response,
        )


# ---------------------------------------------------------------------------
# Nonlinear effect functions
# ---------------------------------------------------------------------------


def _gamma(
    n2: float, omega0: float, A_eff: Area, confinement_factor: float = 1.0
) -> float:
    """Nonlinear coefficient γ = n₂·ω₀·Γ / (c·A_eff).

    Parameters
    ----------
    n2 : float — Nonlinear refractive index n₂ (m²/W).
    omega0 : float — Carrier angular frequency (rad/s).
    A_eff : Area — Effective mode area.
    confinement_factor : float — Waveguide confinement factor Γ (1.0 for fibers).
    """
    return n2 * omega0 * confinement_factor / (C_MS * A_eff.as_m2)


def kerr_step(
    A: NDArray,
    fiber: FiberProfile,
    grid: TemporalGrid,
    dz: float,
    omega0: float,
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
    gamma = _gamma(fiber.n2, omega0, fiber.A_eff, fiber.confinement_factor)
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
        # (red) sideband.  On this FFT grid the required direction is the
        # correlation with h_R, i.e. conj(fft(h_R)) = fft(h_R(-t)); using the
        # plain convolution instead amplifies the anti-Stokes band and makes
        # solitons blue-shift.  The DC component is unchanged (H(0) = 1).
        P_delayed = fR * np.real(grid.ifft(np.conj(h_R_fft) * I_fft))
    else:
        P_delayed = np.zeros_like(intensity)
    return np.asarray(P_inst + P_delayed)


def raman_step(
    A: NDArray,
    fiber: FiberProfile,
    grid: TemporalGrid,
    dz: float,
    include_raman: bool,
    omega0: float = 0.0,
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
    if not hasattr(fiber.raman_response, "fR"):
        raise ValueError("fiber.raman_response must define fR when include_raman=True")
    fR = fiber.raman_response.fR

    # Compute h_R_fft on the fly (no cache available for standalone function)
    h_R_fft = None
    if hasattr(fiber.raman_response, "_h_R"):
        h_R = fiber.raman_response._h_R(grid.t)  # causal: h_R[t<0] = 0
        h_R_fft = grid.fft(h_R)

    P_Raman = _raman_polarization(intensity, fR, grid, h_R_fft)

    gamma = _gamma(fiber.n2, omega0, fiber.A_eff, fiber.confinement_factor)
    raman_phase = np.exp(1j * gamma * P_Raman * dz)
    return np.asarray(A * raman_phase)


def tpa_step(
    A: NDArray,
    fiber: FiberProfile,
    grid: TemporalGrid,
    dz: float,
    include_tpa: bool,
    U: float = 0.0,
    omega0: float = 0.0,
) -> tuple[NDArray, float]:
    """Apply two-photon absorption with carrier dynamics.

    dA/dz = -σ·U·A
    dU/dz = σ·|A|²/(2ħω) - U/τ_c

    Parameters
    ----------
    A : complex array — pulse envelope.
    fiber : FiberProfile — fiber parameters.
    grid : TemporalGrid — time grid.
    dz : float — step size (m).
    include_tpa : bool — whether to include TPA.
    U : float — current carrier density (W⁻¹·m⁻³). Default 0.
    omega0 : float — carrier angular frequency (rad/s). Required when include_tpa=True.

    Returns
    -------
    A : updated complex array.
    U_new : updated carrier density.

    Raises
    ------
    ValueError : if include_tpa=True and omega0 <= 0.
    """
    if not include_tpa:
        return A, U

    if omega0 <= 0:
        raise ValueError(
            "tpa_step: omega0 (carrier angular frequency in rad/s) is required when include_tpa=True"
        )

    sigma = fiber.sigma_tpa
    tau_c = fiber.carrier_lifetime.as_s if fiber.carrier_lifetime is not None else 1e-9

    if sigma <= 0:
        return A, U

    hbar = 1.0545718e-34  # J·s
    hbar_omega = hbar * omega0

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
    include_raman : bool
        Include Raman. Default False.
    include_self_steepening : bool
        Include self-steepening (shock term). Default ``False``.

        When enabled, the shock operator ``(1 + (i/ω₀)∂t)`` of Blow & Wood
        (1989) and Dudley–Genty–Coen RMP 78, 1135 (2006), Eq. (3), is
        integrated with frequency-domain RK4 (``τ_shock`` settable, default
        ``1/ω₀``). Enable explicitly for cross-library comparisons; laserfun
        defaults to shock on (but its shock term has the opposite
        linear-in-Ω asymmetry).

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
        (iii) energy is conserved. laserfun's opposite raw-bin asymmetry
        follows from its own (unshifted-FFT) convention, not from different
        physics.
    include_tpa : bool
        Include TPA. Default False.
    tau_shock : float | None
        Shock (self-steepening) timescale in seconds (SI). ``None``
        (default) uses ``1/ω₀`` at the pulse carrier frequency, matching
        Agrawal and the uncorrected Dudley Eq. (3) value (0.443 fs at
        835 nm). Pass an explicit value for the effective-area-corrected
        timescale (Dudley–Genty–Coen RMP 78, 1135 (2006), Sec. V.B:
        ``tau_shock = 0.56e-15`` for the Fig. 3 PCF config). Must be > 0.
    step_size : Length | None
        Fixed step size (m). If None, adaptive stepping is used.
    min_shrink_factor : float
        Floor for the gradient-based step shrink factor in z-dependent mode.
        Must be in (0, 1]. Default 0.1.
    """

    def __init__(
        self,
        pulse: Wave,
        fiber: FiberProfile,
        betas: NDArray,
        include_raman: bool = False,
        include_self_steepening: bool = False,
        include_tpa: bool = False,
        tau_shock: float | None = None,
        step_size: Length | None = None,
        dispersion_profile: ZDependentDispersion | Callable[[NDArray, float], NDArray] | None = None,
        a_eff_fn: Callable[[float], float] | None = None,
        alpha_fn: Callable[[float], float] | None = None,
        gamma_fn: Callable[[float], float] | None = None,
        min_shrink_factor: float = 0.1,
        betas_unit: BetasUnit = "ps^k/m",
    ):
        self.pulse = pulse
        self.fiber = fiber
        # get_betas() returns betas in ps^(k)/m with omega in rad/ps.
        # We keep them in native units and convert omega to rad/ps in _linear_step.
        self.betas = _normalize_betas(betas, betas_unit)
        self.include_raman = include_raman
        self.include_self_steepening = include_self_steepening
        self.include_tpa = include_tpa
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

        # z-dependent hooks (all optional; None → uniform behavior)
        self._dispersion_profile = dispersion_profile
        self._a_eff_fn = a_eff_fn
        self._alpha_fn = alpha_fn
        self.gamma_fn = gamma_fn
        self._is_z_dependent = dispersion_profile is not None

        # Set up temporal grid from pulse
        self.grid: TemporalGrid = pulse.grid
        self.omega0 = pulse.central_frequency
        self.A = np.array(pulse.envelope_field, dtype=complex)
        self.evolution: list[Wave] = []
        self._z_positions: list[float] = [0.0]
        self._current_z: float = 0.0
        self._spectra_vs_z: tuple[NDArray, NDArray] | None = None
        self._energy_vs_z: list[float] | None = None
        self._U = 0.0  # carrier density for TPA

    @property
    def tau_shock(self) -> float:
        """Shock timescale in seconds (SI): explicit override or ``1/ω₀``."""
        if self._tau_shock_override is not None:
            return self._tau_shock_override
        return 1.0 / float(self.omega0)

    def _linear_step(self, A: NDArray, dz: float) -> NDArray:
        """Apply dispersion via FFT: A(ω) ← A(ω) · exp(−i·Σ β_k(Ω)·Δz).

        Also applies loss: multiply by exp(-α·Δz/2).
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

        if self._is_z_dependent:
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
        """Compute γ(z) from a_eff_fn(z) or fallback to fiber.A_eff."""
        if self.gamma_fn is not None:
            return self.gamma_fn(z)
        if self._a_eff_fn is not None:
            a_eff = self._a_eff_fn(z)
            return _gamma(
                self.fiber.n2,
                self.omega0,
                Area(a_eff, "m^2"),
                self.fiber.confinement_factor,
            )
        else:
            return _gamma(
                self.fiber.n2,
                self.omega0,
                self.fiber.A_eff,
                self.fiber.confinement_factor,
            )

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
            fR = self.fiber.raman_response.fR  # type: ignore[union-attr,attr-defined]
            return _raman_polarization(intensity, fR, self.grid, h_R_fft)
        return intensity

    def _get_h_R_fft(self) -> NDArray:
        """FFT of the Raman response on ``self.grid.t`` (cached).

        Depends only on the temporal grid and the fiber's Raman response,
        both fixed for the engine's lifetime — recomputing it every
        nonlinear step wasted an FFT per step.
        """
        if self._h_R_fft_cache is None:
            h_R = self.fiber.raman_response._h_R(self.grid.t)  # type: ignore[union-attr,attr-defined]
            self._h_R_fft_cache = self.grid.fft(h_R)
        return self._h_R_fft_cache

    def _nonlinear_step(self, A: NDArray, dz: float) -> tuple[NDArray, float]:
        """Apply nonlinear effects.

        Kerr/Raman phase rotation is an exact exponential (unitary). With
        self-steepening, the shock operator ``(1 + (i/ω₀)∂t)`` (Blow & Wood
        1989; Dudley–Genty–Coen RMP 78, 1135 (2006), Eq. (3)) is integrated
        with classical RK4 in the frequency domain: the nonlinear source
        spectrum is multiplied by ``(1 + Ω·τ_shock)`` with ``τ_shock``
        settable (default ``1/ω₀``).  This is the sign that reproduces the
        paper's femtosecond SCG, where self-steepening counteracts the Raman
        self-frequency red-shift (adding the shock term *reduces* the red
        edge).  laserfun applies the opposite linear-in-Ω asymmetry, so a
        cross-library shock comparison disagrees (tracked in
        ``tests/test_gnlse_regression.py``).  The factor is clamped at zero:
        FFT bins with ``1 + Ω·τ_shock < 0`` correspond to non-positive
        absolute frequencies (aliased band) where the multiplier would
        unphysically flip the sign of the nonlinear drive — the source of the
        historic energy drift.

        The total nonlinear polarization is:
          P_NL = (1-fR)|A|² + fR·(h_R ⊗ |A|²)
        """
        gamma = self._get_gamma(self._current_z)
        intensity = np.abs(A) ** 2

        h_R_fft: NDArray | None = None
        if self.include_raman and self.fiber.raman_response is not None:
            # Cached FFT of h_R on the same centered grid (see _get_h_R_fft).
            h_R_fft = self._get_h_R_fft()
        elif self.include_raman and self.fiber.raman_response is None:
            raise ValueError(
                "include_raman=True but fiber.raman_response is None. "
                "Provide a RamanResponse or set include_raman=False."
            )

        P_NL = self._nl_intensity(intensity, h_R_fft)

        # Exact (unitary) Kerr/Raman phase rotation.
        A_noshock = A * np.exp(1j * gamma * P_NL * dz)

        if self.include_self_steepening:
            # Self-steepening: RK4 in the frequency domain with the shock
            # factor (ω/ω₀ = 1 + Ω·τ_shock).  This sign reproduces the Fig. 3
            # SCG (self-steepening reduces the Raman red edge); laserfun's
            # Ω-asymmetry is opposite.  ``τ_shock = 1/ω₀`` matches the paper's
            # uncorrected value; ``0.56 fs`` gives Dudley's
            # effective-area-corrected value. The factor is clamped at zero
            # (see docstring).
            shock_factor = np.clip(1.0 + self.grid.w * self.tau_shock, 0.0, None)
            A_w = self.grid.fft(A)

            def shock_rhs_freq(A_w_stage):
                a = self.grid.ifft(A_w_stage)
                p_nl = self._nl_intensity(np.abs(a) ** 2, h_R_fft)
                nl_src = p_nl * a
                nl_w = self.grid.fft(nl_src)
                nl_w *= shock_factor
                return 1j * gamma * nl_w

            def _rk4_shock_step(A_w_in, h):
                k1 = shock_rhs_freq(A_w_in)
                k2 = shock_rhs_freq(A_w_in + 0.5 * h * k1)
                k3 = shock_rhs_freq(A_w_in + 0.5 * h * k2)
                k4 = shock_rhs_freq(A_w_in + h * k3)
                return A_w_in + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            # The shock factor (up to ~1 + Ωmax·τ_shock ≈ 4–5 at the band
            # edge) amplifies the effective nonlinear phase per step; at
            # fission spikes γ·P·factor·dz can approach O(1), where a single
            # RK4 step loses accuracy and leaks energy. Substep the shock
            # update so the effective phase per substep stays ≤ 0.05 rad
            # (RK4 local error ~ (0.05)⁵ — negligible). Deterministic: M is
            # fixed by the step-opening state, not by iteration.
            phase_scale = gamma * float(np.max(P_NL)) * float(shock_factor.max()) * dz
            n_sub = int(min(50, max(1, np.ceil(phase_scale / 0.05))))
            h_sub = dz / n_sub
            for _ in range(n_sub):
                A_w = _rk4_shock_step(A_w, h_sub)
            A = self.grid.ifft(A_w)
        else:
            A = A_noshock

        A, U_new = tpa_step(
            A, self.fiber, self.grid, dz, self.include_tpa, self._U, self.omega0
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
            if len(self.betas) == 0:
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

    def propagate(
        self,
        num_steps: int,
        *,
        nsaves: int | None = None,
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
            ``z=0`` and ``z=L``). If None, every integration step is stored
            (can use many GB for long runs). Use ~200 for contour plots, as in
            laserfun's ``NLSE(..., nsaves=200)``.
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
            if not self.include_raman or self.fiber.raman_response is None:
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
        self._z_positions = [0.0]

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

            # Half linear step
            self.A = self._linear_step(self.A, dz / 2)
            # Full nonlinear step
            self.A, self._U = self._nonlinear_step(self.A, dz)
            if noise_rng is not None:
                from .noise import raman_noise_field

                assert h_R_fft_noise is not None  # set whenever noise_rng is
                self.A = self.A + raman_noise_field(
                    self.grid,
                    h_R_fft_noise,
                    rng=noise_rng,
                    omega0=float(self.omega0),
                ) * np.sqrt(dz)
            # Half linear step
            self.A = self._linear_step(self.A, dz / 2)

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
        """Warn if recorded energy drifted >5% with loss/TPA disabled."""
        if not self._energy_vs_z:
            return
        e0 = self._energy_vs_z[0]
        e1 = self._energy_vs_z[-1]
        if e0 <= 0.0:
            return
        drift = abs(e1 / e0 - 1.0)
        lossless = (self._get_alpha(self._current_z) == 0.0) and not self.include_tpa
        if lossless and drift > 0.05:
            warnings.warn(
                f"GNLSE energy drift {drift * 100:.2f}% exceeds 5% with "
                f"loss/TPA disabled (E₀={e0:.3e}, E={e1:.3e}). "
                f"Reduce step size (num_steps) or check shock/Raman settings.",
                UserWarning,
                stacklevel=3,
            )

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
        return np.asarray(self._energy_vs_z, dtype=float)


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
        Include two-photon absorption. Default False.
    check_phase_matching : bool
        Run the phase-matching preflight and emit warnings before propagating.
    step_size : Length | None
        Fixed split-step size (m). If ``None`` (default), the engine uses its
        adaptive step heuristic. Supply an explicit value for deterministic,
        reproducible step counts, or when the adaptive heuristic is
        inappropriate (e.g. CW/finite-background fields where the pulse-width
        based dispersion limit is not meaningful).
    """

    def __init__(
        self,
        pulse: Wave,
        fiber: FiberProfile,
        betas: NDArray,
        include_raman: bool = True,
        include_self_steepening: bool = False,
        include_tpa: bool = False,
        check_phase_matching: bool = False,
        step_size: Length | None = None,
        betas_unit: BetasUnit = "ps^k/m",
        tau_shock: float | None = None,
    ):
        self.pulse = pulse
        self.fiber = fiber
        # Normalised to ps^k/m (Ω in rad/ps) at the boundary — see _normalize_betas.
        self.betas = _normalize_betas(betas, betas_unit)
        self.betas_unit = _validate_betas_unit(betas_unit)
        self.include_raman = include_raman
        self.include_self_steepening = include_self_steepening
        self.include_tpa = include_tpa
        self.check_phase_matching = check_phase_matching
        self.step_size = step_size
        if tau_shock is not None and not tau_shock > 0.0:
            raise ValueError(
                f"tau_shock must be positive (seconds, SI), got {tau_shock!r}"
            )
        self.tau_shock = float(tau_shock) if tau_shock is not None else None
        self._evolution: list[Wave] = []
        self._z_positions: NDArray | None = None
        self._spectra_vs_z: tuple[NDArray, NDArray] | None = None
        self._energy_vs_z: NDArray | None = None
        self._preflight_report: SimulationReadinessReport | None = None

    def propagate(
        self,
        num_steps: int = 100,
        *,
        nsaves: int | None = None,
        show_progress: bool = False,
        raman_noise: bool = False,
        noise_seed: int | None = None,
    ) -> None:
        """Run split-step simulation for num_steps steps."""
        if self.check_phase_matching:
            from .phase_matching import emit_readiness_warnings

            report = self.preflight_report
            if report is not None:
                emit_readiness_warnings(report)

        engine = SplitStepEngine(
            pulse=self.pulse,
            fiber=self.fiber,
            betas=self.betas,
            include_raman=self.include_raman,
            include_self_steepening=self.include_self_steepening,
            include_tpa=self.include_tpa,
            tau_shock=self.tau_shock,
            step_size=self.step_size,
        )
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
        gamma = _gamma(fiber.n2, omega0, fiber.A_eff, fiber.confinement_factor)
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
        Include two-photon absorption. Default False.
    min_shrink_factor : float
        Floor for the gradient-based step shrink factor in z-dependent mode.
        Must be in (0, 1]. Default 0.1. Raise it (e.g. 0.3) to trade a small
        amount of accuracy for speed on tapers with mild dispersion gradients.
    betas_unit : {"ps^k/m", "s^k/m", "SI"}
        Unit contract for dispersion coefficients. The tapered solver takes its
        dispersion from *dispersion_profile* (β(ω, z), SI), so this flag is
        validated for consistency with the other solvers but does not rescale a
        coefficient array. Default ``"ps^k/m"``.
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
        check_phase_matching: bool = False,
        min_shrink_factor: float = 0.1,
        step_size: Length | None = None,
        betas_unit: BetasUnit = "ps^k/m",
        tau_shock: float | None = None,
    ):
        self.pulse = pulse
        self.fiber = fiber
        self.betas_unit = _validate_betas_unit(betas_unit)
        self.step_size = step_size
        self.dispersion_profile = dispersion_profile
        self.a_eff_fn = a_eff_fn
        self.alpha_fn = alpha_fn
        self.gamma_fn = gamma_fn
        self.include_raman = include_raman
        self.include_self_steepening = include_self_steepening
        self.include_tpa = include_tpa
        self.check_phase_matching = check_phase_matching
        self.min_shrink_factor = min_shrink_factor
        if tau_shock is not None and not tau_shock > 0.0:
            raise ValueError(
                f"tau_shock must be positive (seconds, SI), got {tau_shock!r}"
            )
        self.tau_shock = float(tau_shock) if tau_shock is not None else None
        self._engine: SplitStepEngine | None = None
        self._evolution: list[Wave] = []
        self._z_positions: NDArray | None = None
        self._spectra_vs_z: tuple[NDArray, NDArray] | None = None
        self._energy_vs_z: NDArray | None = None
        self._preflight_report: SimulationReadinessReport | None = None
        self._strict_mode = False

    def propagate(
        self,
        num_steps: int = 100,
        strict: bool = False,
        *,
        nsaves: int | None = None,
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
            dispersion_profile=self.dispersion_profile,
            a_eff_fn=self.a_eff_fn,
            alpha_fn=self.alpha_fn,
            gamma_fn=self.gamma_fn,
            min_shrink_factor=self.min_shrink_factor,
            tau_shock=self.tau_shock,
            step_size=self.step_size,
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

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    if not offset_scale > 0.0:
        raise ValueError(f"offset_scale must be positive, got {offset_scale!r}")
    z_steps = solver.z_array
    t = solver.pulse.grid.t * 1e12  # ps

    # Offset each trace by a uniform amount and colour it by propagation
    # distance, instead of adding the absolute z value to the intensity
    # (which made the y-axis physically meaningless and overlapping).
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
        ax.plot(t, y + i * offset_scale, color=color, linewidth=0.5)

    if norm is not None:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Propagation distance (mm)")

    # Label y ticks with the z distance of each trace so the offsets are
    # documented on the axis itself (one label per trace).
    n_traces = len(solver.evolution)
    ax.set_yticks([i * offset_scale for i in range(n_traces)])
    ax.set_yticklabels([f"{z * 1e3:.2f} mm" for z in z_steps])
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
    time_reversal: bool = True,
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
        If True (default) plot against the standard literature (Agrawal)
        comoving time, where Raman-shifted solitons appear at positive delay.
        The library's internal time grid has the opposite sign (chirp and
        spectra are unaffected); set False for the raw internal time.

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
    time_reversal: bool = True,
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
    labels = np.empty((n_z, n_t), dtype=object)
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
    time_reversal: bool = True,
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
    time_reversal: bool = True,
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
        If True (default) the returned delay axis uses the standard
        literature (Agrawal) comoving time ``T``.  The library's internal time
        grid has the opposite sign (spectra and the Raman red-shift are
        unaffected, only the temporal direction is mirrored); set False to get
        the raw internal time.

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

    trace = np.empty((n_delays, t.size), dtype=float)
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

    resampled = np.empty((n_delays, int(n_wavelength)), dtype=float)
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
    time_reversal: bool = True,
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
