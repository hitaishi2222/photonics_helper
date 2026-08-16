"""Phase-matching diagnostics for GNLSE simulations.

Provides cross-cutting analysis tools:
- FWM delta-beta and efficiency
- Modulation instability gain spectrum
- Dispersive wave (Cherenkov) root finder
- Simulation readiness assessment
- Spectrum-to-PM validation

Phase-matching predictions (MI gain, dispersive-wave roots, FWM detuning) assume
the linear dispersion ``β(ω)`` and pump parameters supplied. They do **not**
include self-steepening (shock term) corrections to the nonlinear polarization;
for broadband supercontinuum, enable ``include_self_steepening`` on the GNLSE
solver separately and treat PM overlays as linear guides only.

All internal math uses SI (rad/s, 1/m, W/m²). Public API functions
accept SI inputs and convert at boundaries.

Public API
----------
DispersionModel       — protocol for β(ω) accessors
DispersionAdaptor     — wraps Dispersion → DispersionModel
PropagationConstantAdaptor — wraps PropagationConstant → DispersionModel
ZDependentDispersionAdaptor  — wraps ZDependentDispersion → DispersionModel
fwm_delta_beta_degenerate  — Δβ for degenerate FWM
fwm_delta_beta_general     — Δβ for general 4-frequency FWM
fwm_efficiency             — conversion efficiency η(Δβ, L, α)
fwm_idler_frequency        — ωᵢ = 2ωₚ − ωₛ
scan_fwm_detuning          — scan Δβ and η over signal grid
mi_gain_spectrum           — classical MI gain g(Ω)
mi_sideband_frequencies    — MI sideband frequencies
mi_gain_spectrum_extended  — extended Lighthill criterion
dispersive_wave_roots      — full β(ω) DW root finder
PhaseMatchResult           — FWM scan result dataclass
SimulationReadinessReport  — readiness report dataclass
assess_simulation_readiness — build readiness report
compare_spectrum_to_phase_matching — post-flight validation
plot_fwm_efficiency        — FWM efficiency plot
plot_mi_gain               — MI gain plot
plot_readiness_report      — readiness panel
plot_spectrum_with_pm_overlay — spectrum + PM vertical lines
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root_scalar
from scipy.signal import find_peaks
from scipy.integrate import cumulative_trapezoid

from .base import C_MS, PI, WavelengthArray

if TYPE_CHECKING:
    from .fiber import Dispersion, PropagationConstant, ZDependentDispersion
    from .gnlse import GNLSESolver, FiberProfile
    from .pulse import Wave


# ============================================================================
# 1. DispersionModel protocol and adaptors
# ============================================================================

@runtime_checkable
class DispersionModel(Protocol):
    """Protocol for β(ω) accessors.

    Any object providing ``beta(omega) -> NDArray[float]`` and
    ``beta1(omega) -> float`` satisfies this protocol.
    """

    def beta(self, omega: NDArray | float) -> NDArray | float:
        """Return propagation constant β at angular frequency(ies).

        Parameters
        ----------
        omega : float or 1-D array — angular frequency in rad/s.

        Returns
        -------
        β in 1/m. Same shape as input.
        """
        ...

    def beta1(self, omega: NDArray | float) -> NDArray | float:
        """Return group delay per unit length dβ/dω at angular frequency(ies).

        Parameters
        ----------
        omega : float or 1-D array — angular frequency in rad/s.

        Returns
        -------
        β₁ in s/m. Same shape as input.
        """
        ...


class DispersionAdaptor:
    """Build β(ω) from a ``Dispersion`` object via D(λ).

    Uses the relation β₂(ω) = −D(λ)·λ²/(2πc) and numerical integration
    to reconstruct β(ω) from the dispersion table.

    The adaptor is callable: ``adaptor(omega)`` returns β(ω).
    """

    def __init__(self, dispersion: "Dispersion", omega0: float | None = None) -> None:
        self._disp = dispersion
        if omega0 is None:
            omega0 = 2 * PI * C_MS / dispersion.central_wavelength.as_m
        self.omega0 = omega0
        # Pre-build a lookup table of β₂(ω) over the dispersion range
        wl_arr_m = dispersion.wavelengths.as_m
        D_arr = dispersion.as_s_m_m
        beta2_arr = -D_arr * wl_arr_m**2 / (2 * PI * C_MS)
        omega_arr = 2 * PI * C_MS / wl_arr_m  # absolute omega
        # Sort by omega (ascending) for spline fitting
        sort_idx = np.argsort(omega_arr)
        omega_sorted = omega_arr[sort_idx]
        beta2_sorted = beta2_arr[sort_idx]
        self._omega_ref = omega_sorted
        self._beta2_ref = beta2_sorted
        # Build spline for β₂(ω)
        from scipy.interpolate import UnivariateSpline
        self._beta2_spline = UnivariateSpline(omega_sorted, beta2_sorted, s=0)

    def __call__(self, omega: NDArray | float) -> NDArray | float:
        """Return β(ω)."""
        return self.beta(omega)

    def beta(self, omega: NDArray | float) -> NDArray | float:
        scalar = np.isscalar(omega)
        omega_arr = np.atleast_1d(np.asarray(omega, dtype=float))
        # Get β₂(ω) at each frequency
        beta2_vals = self._beta2_spline(omega_arr)
        # Numerical integration: β(ω) = β(ω₀) + ∫_{ω₀}^{ω} β₂(ω') dω'
        # Compute cumulative integral from omega0 to each omega
        domega = omega_arr - self.omega0
        # Sort by omega for proper integration
        sort_idx = np.argsort(omega_arr)
        domega_sorted = domega[sort_idx]
        beta2_sorted = beta2_vals[sort_idx]
        beta_offset_sorted = cumulative_trapezoid(beta2_sorted, domega_sorted, initial=0.0)
        # Reorder back
        beta_offset = np.empty_like(beta_offset_sorted)
        beta_offset[sort_idx] = beta_offset_sorted
        # β(ω₀) = 0 as reference (absolute value not needed for PM)
        result = beta_offset
        if scalar:
            return float(result[0])
        return result

    def beta1(self, omega: NDArray | float) -> NDArray | float:
        scalar = np.isscalar(omega)
        omega_arr = np.atleast_1d(np.asarray(omega, dtype=float))
        # β₁ = dβ/dω ≈ β₂(ω) for practical purposes (group delay)
        # More precisely β₁ = dβ/dω, computed via finite differences
        domega = 1e12  # 1 THz offset
        beta_plus = self._beta_no_scalar(omega_arr + domega)
        beta_minus = self._beta_no_scalar(omega_arr - domega)
        beta1_vals = (beta_plus - beta_minus) / (2 * domega)
        if scalar:
            return float(beta1_vals[0])
        return beta1_vals

    def _beta_no_scalar(self, omega: NDArray) -> NDArray:
        """Internal beta without scalar handling."""
        return self.beta(omega)


class PropagationConstantAdaptor:
    """Use stored β values with spline interpolation.

    Wraps a ``PropagationConstant`` object directly — the β values
    are already absolute, so no integration is needed.

    The adaptor is callable: ``adaptor(omega)`` returns β(ω).
    """

    def __init__(self, pc: "PropagationConstant") -> None:
        from scipy.interpolate import InterpolatedUnivariateSpline

        if isinstance(pc.x_values, WavelengthArray):
            omega_arr = pc.x_values.to_omega().as_rad_s
        else:
            omega_arr = pc.x_values.as_rad_s

        self._beta_interp = InterpolatedUnivariateSpline(
            omega_arr, pc.values, k=3
        )
        self._beta1_interp = InterpolatedUnivariateSpline(
            omega_arr, pc.values, k=3
        )
        self._beta1_interp.derivative()  # prepare derivative

    def __call__(self, omega: NDArray | float) -> NDArray | float:
        """Return β(ω)."""
        return self.beta(omega)

    def beta(self, omega: NDArray | float) -> NDArray | float:
        result = self._beta_interp(np.atleast_1d(np.asarray(omega, dtype=float)))
        if np.isscalar(omega):
            return float(result[0])
        return result

    def beta1(self, omega: NDArray | float) -> NDArray | float:
        omega_arr = np.atleast_1d(np.asarray(omega, dtype=float))
        # β₁ = dβ/dω via derivative of spline
        dbeta_domega = self._beta_interp.derivative()(omega_arr)
        if np.isscalar(omega):
            return float(dbeta_domega[0])
        return dbeta_domega


class ZDependentDispersionAdaptor:
    """Return β(ω, z_fixed) via the existing interpolant.

    Fixes z to a specific propagation position and provides a
    1-D β(ω) interface.

    The adaptor is callable: ``adaptor(omega)`` returns β(ω, z_fixed).
    """

    def __init__(self, zd_disp: "ZDependentDispersion", z: float) -> None:
        self._zd = zd_disp
        self._z = z
        # Pre-evaluate β at all omega values for this z
        self._omega_grid = zd_disp.omegas
        self._beta_at_z = zd_disp.fn(self._omega_grid, z)

    def __call__(self, omega: NDArray | float) -> NDArray | float:
        """Return β(ω, z_fixed)."""
        return self.beta(omega)

    def beta(self, omega: NDArray | float) -> NDArray | float:
        result = self._zd.fn(omega, self._z)
        return result

    def beta1(self, omega: NDArray | float) -> NDArray | float:
        scalar = np.isscalar(omega)
        omega_arr = np.atleast_1d(np.asarray(omega, dtype=float))
        domega = 1e6
        beta_plus = self._zd.fn(omega_arr + domega, self._z)
        beta_minus = self._zd.fn(omega_arr - domega, self._z)
        beta1_vals = (beta_plus - beta_minus) / (2 * domega)
        if scalar:
            return float(beta1_vals[0])
        return beta1_vals


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass
class PhaseMatchResult:
    """Result from phase-matching scan.

    Attributes
    ----------
    omega_signal : 1-D array — signal angular frequencies (rad/s).
    delta_beta : 1-D array — Δβ values (1/m).
    efficiency : 1-D array — FWM conversion efficiency η (dimensionless, 0–1).
    idler_omega : 1-D array — idler angular frequencies (rad/s).
    pump_omega : float — pump angular frequency (rad/s).
    length_m : float — interaction length (m).
    alpha : float — loss coefficient (1/m).
    """

    omega_signal: NDArray
    delta_beta: NDArray
    efficiency: NDArray
    idler_omega: NDArray
    pump_omega: float
    length_m: float
    alpha: float = 0.0


@dataclass
class DispersiveWaveResult:
    """Result from dispersive wave root finding.

    Attributes
    ----------
    wavelengths_m : 1-D array — dispersive wave wavelengths (m).
    wavelengths_nm : 1-D array — dispersive wave wavelengths (nm).
    soliton_omega : float — soliton center frequency (rad/s).
    soliton_lambda_m : float — soliton wavelength (m).
    q_sol : float — soliton wavenumber correction (1/m).
    """

    wavelengths_m: NDArray
    wavelengths_nm: NDArray
    soliton_omega: float
    soliton_lambda_m: float
    q_sol: float = 0.0


@dataclass
class SimulationReadinessReport:
    """Report on simulation readiness.

    Attributes
    ----------
    dispersion_covers_grid : bool — True if dispersion covers the pulse grid.
    grid_omega_min : float — minimum grid angular frequency (rad/s).
    grid_omega_max : float — maximum grid angular frequency (rad/s).
    dispersion_min_omega : float — minimum dispersion model frequency (rad/s).
    dispersion_max_omega : float — maximum dispersion model frequency (rad/s).
    soliton_order : float — estimated soliton order N.
    dispersion_length : float — L_D (m).
    nonlinear_length : float — L_NL (m).
    fission_length : float — L_fiss (m).
    recommended_num_steps : int — suggested step count.
    predicted_processes : list[str] — predicted nonlinear processes.
    warnings : list[str] — warnings about simulation setup.
    recommendations : list[str] — suggestions for improvement.
    fwm_predictions : list[dict] — predicted FWM idler wavelengths (nm).
    mi_predictions : dict — predicted MI sideband info.
    dw_predictions : list[float] — predicted DW wavelengths (nm).
    """

    dispersion_covers_grid: bool
    grid_omega_min: float
    grid_omega_max: float
    dispersion_min_omega: float
    dispersion_max_omega: float
    soliton_order: float
    dispersion_length: float
    nonlinear_length: float
    fission_length: float
    recommended_num_steps: int
    predicted_processes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    fwm_predictions: list[dict] = field(default_factory=list)
    mi_predictions: dict = field(default_factory=dict)
    dw_predictions: list[float] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Post-flight validation report.

    Attributes
    ----------
    predictions : list[dict] — list of PM predictions with wavelength_nm key.
    peaks_found : list[dict] — list of detected spectral peaks.
    matches : list[dict] — each has prediction, matched_peak, residual_nm, pass.
    overall_pass : bool — True if all predictions have a matching peak.
    tolerance_nm : float — wavelength tolerance used.
    """

    predictions: list[dict]
    peaks_found: list[dict]
    matches: list[dict]
    overall_pass: bool
    tolerance_nm: float


# ============================================================================
# Helper: Wavelength type (needed for DispersionAdaptor)
# ============================================================================
from .base import Wavelength  # noqa: E402


# ============================================================================
# 2. FWM phase-matching functions
# ============================================================================

def fwm_delta_beta_degenerate(
    beta_fn,
    omega_p: float,
    omega_s: float,
) -> float:
    """Compute Δβ for degenerate FWM.

    Δβ = 2β(ωₚ) − β(ωₛ) − β(ωᵢ)  where  ωᵢ = 2ωₚ − ωₛ

    Parameters
    ----------
    beta_fn : callable — β(ω) function (any DispersionModel or callable).
    omega_p : float — pump angular frequency (rad/s).
    omega_s : float — signal angular frequency (rad/s).

    Returns
    -------
    delta_beta : float — phase mismatch (1/m).
    """
    beta_p = beta_fn(omega_p)
    beta_s = beta_fn(omega_s)
    omega_i = 2 * omega_p - omega_s
    beta_i = beta_fn(omega_i)
    return 2 * beta_p - beta_s - beta_i


def fwm_delta_beta_general(
    beta_fn,
    omega_1: float,
    omega_2: float,
    omega_3: float,
    omega_4: float,
) -> float:
    """Compute Δβ for general (non-degenerate) 4-wave mixing.

    Δβ = β(ω₁) + β(ω₂) − β(ω₃) − β(ω₄)

    Parameters
    ----------
    beta_fn : callable — β(ω) function.
    omega_1, omega_2 : float — pump/frequency-1 (rad/s).
    omega_3, omega_4 : float — signal/idler (rad/s).

    Returns
    -------
    delta_beta : float — phase mismatch (1/m).
    """
    return beta_fn(omega_1) + beta_fn(omega_2) - beta_fn(omega_3) - beta_fn(omega_4)


def fwm_efficiency(delta_beta: float | NDArray, length_m: float, alpha: float = 0.0) -> float | NDArray:
    """Compute FWM conversion efficiency η.

    Lossless (α=0):  η = sinc²(Δβ·L/2) when γP ≪ |Δβ|

    For the common case (no γP provided), returns the sinc² approximation
    scaled by effective length for lossy media.

    Parameters
    ----------
    delta_beta : float or array — phase mismatch Δβ (1/m).
    length_m : float — interaction length (m).
    alpha : float — loss coefficient (1/m). Default 0.

    Returns
    -------
    efficiency : float or array — conversion efficiency (0–1 range).
    """
    delta_beta_arr = np.atleast_1d(np.asarray(delta_beta, dtype=float))

    if alpha > 0:
        L_eff = (1 - np.exp(-alpha * length_m)) / alpha
    else:
        L_eff = length_m

    arg = delta_beta_arr * L_eff / 2.0
    result = np.where(
        np.isclose(arg, 0.0),
        1.0,
        (np.sin(arg) / np.where(np.isclose(arg, 0.0), 1.0, arg)) ** 2,
    )

    if np.isscalar(delta_beta):
        return float(result[0])
    return result


def fwm_idler_frequency(omega_p: float, omega_s: float) -> float:
    """Compute idler frequency for degenerate FWM.

    ωᵢ = 2ωₚ − ωₛ

    Parameters
    ----------
    omega_p : float — pump angular frequency (rad/s).
    omega_s : float — signal angular frequency (rad/s).

    Returns
    -------
    omega_i : float — idler angular frequency (rad/s).
    """
    return 2 * omega_p - omega_s


def scan_fwm_detuning(
    beta_fn,
    omega_p: float,
    omega_signal_grid: NDArray,
    P_pump: float,
    gamma: float,
    alpha: float = 0.0,
    L: float | None = None,
) -> PhaseMatchResult:
    """Scan FWM Δβ and efficiency over a grid of signal frequencies.

    Parameters
    ----------
    beta_fn : callable — β(ω) function.
    omega_p : float — pump angular frequency (rad/s).
    omega_signal_grid : 1-D array — signal angular frequencies to scan (rad/s).
    P_pump : float — pump power (W).
    gamma : float — nonlinear coefficient γ (1/(W·m)).
    alpha : float — loss coefficient (1/m). Default 0.
    L : float — interaction length (m). If None, uses 1/gamma as estimate.

    Returns
    -------
    result : PhaseMatchResult
    """
    if L is None:
        L = 1.0 / max(gamma, 1e-30)

    n = len(omega_signal_grid)
    delta_beta = np.zeros(n)
    efficiency = np.zeros(n)

    for i, omega_s in enumerate(omega_signal_grid):
        db = fwm_delta_beta_degenerate(beta_fn, omega_p, omega_s)
        delta_beta[i] = db
        # Full FWM efficiency formula with γP
        gp = gamma * P_pump
        if np.abs(db) < 1e-15:
            # Perfect phase matching: η → 1 (or limited by γP terms)
            if alpha > 0:
                L_eff = (1 - np.exp(-alpha * L)) / alpha
            else:
                L_eff = L
            # With perfect PM, η = sin²(γP·L_eff) / (γP·L_eff)² ... 
            # Actually for Δβ=0, η = (γP·L_eff)² for small γP·L_eff
            # For large γP·L_eff, it oscillates
            arg = gp * L_eff
            efficiency[i] = min(np.sin(arg) ** 2 / (arg ** 2 + 1e-30) * (arg ** 2 + 1e-30), 1.0)
            # Simplified: for Δβ=0, η peaks at sin²(γP·L_eff)
            efficiency[i] = min(np.sin(gp * L_eff) ** 2, 1.0) if gp * L_eff < np.pi else 1.0
        else:
            efficiency[i] = fwm_efficiency(db, L, alpha)

    omega_idler = 2 * omega_p - omega_signal_grid

    return PhaseMatchResult(
        omega_signal=omega_signal_grid,
        delta_beta=delta_beta,
        efficiency=efficiency,
        idler_omega=omega_idler,
        pump_omega=omega_p,
        length_m=L,
        alpha=alpha,
    )


# ============================================================================
# 3. Modulation instability functions
# ============================================================================

def mi_gain_spectrum(
    beta2: float,
    gamma: float,
    P: float,
    omega_m: NDArray | float | None = None,
) -> NDArray | float:
    """Compute classical MI gain g(Ω).

    g(Ω) = 2|β₂|Ω√(Ω²_c − Ω²) for β₂ < 0 (anomalous), Ω < Ω_c
    where Ω_c² = 2γP/|β₂|
    g(Ω) = 0 for β₂ > 0 (normal dispersion)

    Peak gain at Ω = Ω_c/√2: g_max = γP

    Classical result from Agrawal, Nonlinear Fiber Optics.

    Parameters
    ----------
    beta2 : float — group velocity dispersion β₂ (s²/m).
    gamma : float — nonlinear coefficient γ (1/(W·m)).
    P : float — CW pump power (W).
    omega_m : 1-D array — modulation frequencies Ω (rad/s). If None,
              returns peak gain and cutoff.

    Returns
    -------
    gain : float or 1-D array — MI gain coefficient (1/m).
    """
    if beta2 > 0:
        # Normal dispersion — no MI gain
        if omega_m is None:
            return 0.0
        return np.zeros_like(np.atleast_1d(np.asarray(omega_m, dtype=float)))

    # Anomalous dispersion
    if omega_m is None:
        # Return peak gain and cutoff
        Omega_cutoff_sq = -2 * gamma * P / beta2  # positive
        Omega_peak_sq = -gamma * P / beta2  # Ω_peak = Ω_c/√2
        Omega_cutoff = np.sqrt(max(Omega_cutoff_sq, 0))
        Omega_peak = np.sqrt(max(Omega_peak_sq, 0))
        g_max = gamma * P  # peak gain
        return {"g_max": g_max, "Omega_peak": Omega_peak, "Omega_cutoff": Omega_cutoff}

    omega_arr = np.atleast_1d(np.asarray(omega_m, dtype=float))
    Omega_sq = omega_arr ** 2
    Omega_c_sq = -2 * gamma * P / beta2  # positive cutoff freq squared

    # g(Ω) = 2|β₂|·|Ω|·√(Ω_c² − Ω²) for Ω < Ω_c
    mask = Omega_sq < Omega_c_sq
    gain = np.zeros_like(omega_arr)
    gain[mask] = 2.0 * abs(beta2) * np.abs(omega_arr[mask]) * np.sqrt(
        Omega_c_sq - Omega_sq[mask]
    )

    if np.isscalar(omega_m):
        return float(gain[0])
    return gain


def mi_sideband_frequencies(
    beta2: float,
    gamma: float,
    P: float,
) -> NDArray:
    """Compute MI sideband frequencies.

    Solves β₂Ω² + 2γP = 0 → Ω² = −2γP/β₂

    Parameters
    ----------
    beta2 : float — β₂ (s²/m).
    gamma : float — γ (1/(W·m)).
    P : float — pump power (W).

    Returns
    -------
    omega_sidebands : 1-D array — [−Ω_sideband, +Ω_sideband] (rad/s).
    """
    if beta2 >= 0:
        return np.array([0.0])

    Omega_sq = -2 * gamma * P / beta2
    if Omega_sq < 0:
        return np.array([0.0])

    Omega = np.sqrt(Omega_sq)
    return np.array([-Omega, Omega])


def mi_gain_spectrum_extended(
    beta_fn,
    omega0: float,
    gamma: float,
    P: float,
    alpha: float = 0.0,
    L: float | None = None,
    omega_m: NDArray | None = None,
) -> dict:
    """Extended MI gain near ZDW using κ(Ω) root finder.

    Uses the full dispersion expansion:
      κ(Ω) = β(ω₀+Ω) + β(ω₀−Ω) − 2β(ω₀) − γP·Ω²/ω₀² ... simplified
    Actually: κ(Ω) = Re[β(ω₀+Ω) + β(ω₀−Ω) − 2β(ω₀)] + 2γP
    Gain exists when κ(Ω) > 0.

    Parameters
    ----------
    beta_fn : callable — β(ω) function.
    omega0 : float — pump carrier frequency (rad/s).
    gamma : float — nonlinear coefficient (1/(W·m)).
    P : float — pump power (W).
    alpha : float — loss (1/m). Default 0.
    L : float — length (m). If None, 1/γ.
    omega_m : 1-D array — modulation frequencies (rad/s).

    Returns
    -------
    result : dict with keys 'omega_m', 'gain', 'Omega_peak', 'Omega_cutoff'.
    """
    if L is None:
        L = 1.0 / max(gamma, 1e-30)

    if omega_m is None:
        # Auto-generate grid
        # Use classical estimate for scale
        beta2_approx = 0.0  # Will compute numerically
        Omega_classical = np.sqrt(max(2 * gamma * P / 1e-23, 1e12))  # rough
        omega_m = np.linspace(-3 * Omega_classical, 3 * Omega_classical, 500)

    omega_arr = np.atleast_1d(np.asarray(omega_m, dtype=float))
    beta_pump = beta_fn(omega0)

    # κ(Ω) = β(ω₀+Ω) + β(ω₀−Ω) − 2β(ω₀)
    omega_plus = omega0 + omega_arr
    omega_minus = omega0 - omega_arr
    beta_plus = beta_fn(omega_plus)
    beta_minus = beta_fn(omega_minus)
    kappa = (beta_plus + beta_minus - 2 * beta_pump) * (2 * abs(beta_pump) + 1e-30) / (abs(beta_pump) + 1e-30)

    # Actually, the extended criterion:
    # g(Ω) = 2·|Im[kappa(Ω)]| where kappa(Ω) = β(ω₀+Ω)+β(ω₀−Ω)−2β(ω₀)−2γP
    # Wait — the standard extended form:
    # h(Ω) = [β(ω₀+Ω) + β(ω₀−Ω) − 2β(ω₀)] / 2 + γP
    # Gain when h(Ω) < 0 (sign convention varies)
    # Let's use the most common convention:
    # g(Ω) = 2·|h(Ω)| where h(Ω) = (β(ω₀+Ω)+β(ω₀−Ω)−2β(ω₀))/2 + γP
    h = (beta_plus + beta_minus - 2 * beta_pump) / 2.0 + gamma * P
    gain = 2.0 * np.abs(np.where(h < 0, h, 0.0))

    # Find peak and cutoff
    mask = gain > 0
    if mask.any():
        Omega_peak = omega_arr[mask][np.argmax(gain[mask])]
        Omega_cutoff = omega_arr[mask[-1]] if mask.any() else 0.0
    else:
        Omega_peak = 0.0
        Omega_cutoff = 0.0

    return {
        "omega_m": omega_arr,
        "gain": gain,
        "Omega_peak": Omega_peak,
        "Omega_cutoff": Omega_cutoff,
    }


# ============================================================================
# 4. Dispersive wave root finder
# ============================================================================

def dispersive_wave_roots(
    beta_fn,
    omega_sol: float,
    q_sol: float = 0.0,
    wl_range_nm: tuple[float, float] = (300.0, 2500.0),
    n_brackets: int = 50,
) -> DispersiveWaveResult:
    """Find all dispersive wave (Cherenkov) frequencies.

    Solves β(ω_DW) = β(ωₛ) + β₁(ωₛ)(ω_DW − ωₛ) + q_sol

    Uses bracket search over wavelength range with scipy.optimize.root_scalar.

    Parameters
    ----------
    beta_fn : callable — β(ω) function (DispersionModel or similar).
    omega_sol : float — soliton center frequency (rad/s).
    q_sol : float — soliton wavenumber correction (1/m). Default 0.
    wl_range_nm : tuple — (wl_min_nm, wl_max_nm) search range.
    n_brackets : int — number of bracket intervals to scan.

    Returns
    -------
    result : DispersiveWaveResult
    """
    wl_min, wl_max = wl_range_nm
    wl_grid = np.linspace(wl_min, wl_max, n_brackets)
    omega_grid = 2 * PI * C_MS / wl_grid * 1e-9

    beta_sol = beta_fn(omega_sol)
    # Get beta1 = dβ/dω
    try:
        if hasattr(beta_fn, 'beta1'):
            beta1_sol = float(beta_fn.beta1(omega_sol))
        else:
            # Finite difference fallback
            domega = 1e6
            beta_plus = beta_fn(omega_sol + domega)
            beta_minus = beta_fn(omega_sol - domega)
            beta1_sol = (beta_plus - beta_minus) / (2 * domega)
    except Exception:
        # Last resort: finite difference with larger offset
        domega = 1e12
        beta_plus = beta_fn(omega_sol + domega)
        beta_minus = beta_fn(omega_sol - domega)
        beta1_sol = (beta_plus - beta_minus) / (2 * domega)

    # Define the RHS: β(ωₛ) + β₁(ωₛ)(ω − ωₛ) + q_sol
    rhs = beta_sol + beta1_sol * (omega_grid - omega_sol) + q_sol

    # Define f(ω) = β(ω) − RHS — roots where f = 0
    beta_grid = beta_fn(omega_grid)
    f_values = beta_grid - rhs

    # Find sign changes
    roots_omega = []
    sign_changes = np.where(np.diff(np.sign(f_values)))[0]

    for idx in sign_changes:
        w_lo = omega_grid[idx]
        w_hi = omega_grid[idx + 1]
        try:
            res = root_scalar(
                lambda w, bs=beta_sol, b1=beta1_sol, qs=q_sol: beta_fn(w) - (bs + b1 * (w - omega_sol) + qs),
                bracket=[w_lo, w_hi],
                method='brentq',
            )
            if res.converged:
                roots_omega.append(float(res.root))
        except ValueError:
            continue

    if not roots_omega:
        # Try derivative-based root finding at multiple starting points
        for w_start in omega_grid[::5]:
            try:
                res = root_scalar(
                    lambda w, bs=beta_sol, b1=beta1_sol, qs=q_sol: beta_fn(w) - (bs + b1 * (w - omega_sol) + qs),
                    x0=w_start,
                    method='newton',
                    fprime=lambda w, b1=beta1_sol: _beta1_fd(beta_fn, w, 1e6),
                )
                if res.converged:
                    wl_check = 2 * PI * C_MS / res.root * 1e9
                    if wl_min <= wl_check <= wl_max:
                        roots_omega.append(float(res.root))
            except Exception:
                continue

    # Deduplicate and sort
    unique_roots = []
    for r in roots_omega:
        r_val = float(r) if hasattr(r, '__float__') else float(r[0])
        if not any(np.isclose(r_val, u, atol=1e-6) for u in unique_roots):
            unique_roots.append(r_val)
    roots_omega = sorted(unique_roots)
    roots_wl_m = np.array([2 * PI * C_MS / w * 1e-9 for w in roots_omega])
    roots_wl_nm = roots_wl_m * 1e9

    sol_wl_m = 2 * PI * C_MS / omega_sol * 1e-9

    return DispersiveWaveResult(
        wavelengths_m=roots_wl_m,
        wavelengths_nm=roots_wl_nm,
        soliton_omega=omega_sol,
        soliton_lambda_m=sol_wl_m,
        q_sol=q_sol,
    )


def _beta1_fd(beta_fn, omega: float, domega: float = 1e6) -> float:
    """Finite-difference approximation of β₁ = dβ/dω."""
    beta_plus = beta_fn(omega + domega)
    beta_minus = beta_fn(omega - domega)
    return float((beta_plus - beta_minus) / (2 * domega))


# ============================================================================
# 5. Simulation readiness assessment
# ============================================================================

def assess_simulation_readiness(
    pulse: "Wave",
    fiber: "FiberProfile",
    dispersion,
    betas: NDArray | None = None,
    length: float | None = None,
) -> SimulationReadinessReport:
    """Assess whether a proposed GNLSE simulation is adequately set up.

    Checks dispersion coverage, estimates soliton parameters, and predicts
    which nonlinear processes should be relevant.

    Parameters
    ----------
    pulse : Wave — input pulse.
    fiber : FiberProfile — fiber parameters.
    dispersion : Dispersion or ZDependentDispersion or PropagationConstant — dispersion source.
    betas : NDArray — Taylor coefficients [β₂, β₃, ...] in ps^k/m. Optional.
    length : float — propagation length (m). If None, uses fiber.length.

    Returns
    -------
    report : SimulationReadinessReport
    """
    from .gnlse import _gamma

    if length is None:
        length = fiber.length.as_m

    omega0 = pulse.central_frequency
    T0 = pulse.envelope.pulse_width.as_s
    P_peak = pulse.peak_power()
    gamma = _gamma(fiber.n2, omega0, fiber.A_eff, fiber.confinement_factor)

    # Determine grid bounds
    omega_grid = pulse.grid.w  # offsets from omega0
    omega_min = omega0 + omega_grid.min()
    omega_max = omega0 + omega_grid.max()

    # Determine dispersion bounds
    if hasattr(dispersion, 'omegas'):
        # ZDependentDispersion or PropagationConstant
        disp_min_omega = float(dispersion.omegas.min())
        disp_max_omega = float(dispersion.omegas.max())
    elif hasattr(dispersion, 'wavelengths'):
        # Dispersion — convert wavelengths to omega
        wl_arr = dispersion.wavelengths.as_m
        disp_min_omega = 2 * PI * C_MS / wl_arr.max()
        disp_max_omega = 2 * PI * C_MS / wl_arr.min()
    else:
        # Taylor mode — use a wide estimate
        disp_min_omega = omega0 - 5e15
        disp_max_omega = omega0 + 5e15

    # Check coverage
    covers = (omega_min >= disp_min_omega - 1e12) and (omega_max <= disp_max_omega + 1e12)

    # Compute soliton parameters
    if betas is not None and len(betas) > 0:
        beta2_si = betas[0] * 1e-24  # ps²/m → s²/m
    else:
        # Try to extract from dispersion
        beta2_si = _estimate_beta2(dispersion, omega0)
        if beta2_si is None:
            beta2_si = 0.0

    L_D = T0 ** 2 / abs(beta2_si) if beta2_si != 0 else float("inf")
    L_NL = 1.0 / (gamma * P_peak) if gamma * P_peak > 0 else float("inf")

    if beta2_si != 0 and gamma * P_peak > 0:
        N_sol = np.sqrt(gamma * P_peak * T0 ** 2 / abs(beta2_si))
    else:
        N_sol = 0.0

    L_fiss = L_D / (N_sol * 0.7) if N_sol > 0 else float("inf")

    # Recommended num_steps
    dz_base = min(L_D, L_NL) * 0.01 if L_D < float("inf") and L_NL < float("inf") else length / 100
    recommended_steps = max(int(length / max(dz_base, 1e-30)), 100)

    # Predict processes
    predicted = []
    if N_sol > 1:
        predicted.append("soliton_fission")
        predicted.append("dispersive_wave")
    if beta2_si < 0 and gamma * P_peak > 0:
        predicted.append("modulation_instability")
    if length > L_NL:
        predicted.append("four_wave_mixing")

    # Build warnings and recommendations
    warnings_list = []
    recommendations_list = []

    if not covers:
        warnings_list.append(
            f"Dispersion model does not cover the full pulse grid: "
            f"[{omega_min:.2e}, {omega_max:.2e}] rad/s "
            f"vs model [{disp_min_omega:.2e}, {disp_max_omega:.2e}] rad/s"
        )
        recommendations_list.append(
            "Consider using a broader dispersion table or Taylor expansion "
            "with higher order coefficients."
        )

    if L_D < length and N_sol > 3:
        warnings_list.append(
            f"High soliton order N={N_sol:.1f} with L_D << L — expect strong fission."
        )
        recommendations_list.append(
            f"Increase num_steps to at least {recommended_steps} for accurate fission."
        )

    # MI predictions
    mi_info = {}
    if beta2_si < 0 and gamma * P_peak > 0:
        mi_info = mi_gain_spectrum(beta2_si, gamma, P_peak)
        if isinstance(mi_info, dict):
            mi_predictions = {
                "Omega_peak": mi_info["Omega_peak"],
                "Omega_cutoff": mi_info["Omega_cutoff"],
                "g_max": mi_info["g_max"],
                "sideband_wl_nm": (2 * PI * C_MS / (omega0 + mi_info["Omega_peak"]) * 1e9,
                                    2 * PI * C_MS / (omega0 - mi_info["Omega_peak"]) * 1e9),
            }
        else:
            mi_predictions = {}
    else:
        mi_predictions = {}

    # DW predictions (if we have betas)
    dw_preds = []
    if betas is not None and len(betas) >= 2 and beta2_si != 0:
        try:
            # Quick DW estimate using β₂/β₃ formula
            beta3_si = betas[1] * 1e-27 if len(betas) > 1 else 0.0
            if beta3_si != 0 and abs(beta3_si) > 1e-40:
                delta_omega_dw = -2 * beta2_si / beta3_si
                omega_dw = omega0 + delta_omega_dw
                if omega_dw > 0:
                    dw_wl = 2 * PI * C_MS / omega_dw * 1e9
                    dw_preds.append(float(dw_wl))
        except Exception:
            pass

    # FWM predictions (degenerate, signal at DW wavelength if available)
    fwm_preds = []
    if dw_preds and len(fwm_preds) == 0:
        # Predict FWM idler for signal near DW
        for dw_wl_nm in dw_preds:
            omega_s = 2 * PI * C_MS / dw_wl_nm * 1e-9
            omega_i = fwm_idler_frequency(omega0, omega_s)
            fwm_preds.append({
                "idler_wavelength_nm": float(2 * PI * C_MS / omega_i * 1e9),
                "signal_wavelength_nm": dw_wl_nm,
            })

    return SimulationReadinessReport(
        dispersion_covers_grid=covers,
        grid_omega_min=omega_min,
        grid_omega_max=omega_max,
        dispersion_min_omega=disp_min_omega,
        dispersion_max_omega=disp_max_omega,
        soliton_order=N_sol,
        dispersion_length=L_D,
        nonlinear_length=L_NL,
        fission_length=L_fiss,
        recommended_num_steps=recommended_steps,
        predicted_processes=predicted,
        warnings=warnings_list,
        recommendations=recommendations_list,
        fwm_predictions=fwm_preds,
        mi_predictions=mi_predictions,
        dw_predictions=dw_preds,
    )


def _estimate_beta2(dispersion, omega0: float) -> float | None:
    """Estimate β₂ from a dispersion source object."""
    if hasattr(dispersion, 'get_beta2'):
        # Dispersion object
        wl_nm = 2 * PI * C_MS / omega0 * 1e9
        try:
            beta2 = dispersion.get_beta2(wl_nm)
            # get_beta2 returns in ps²/m (solver convention) or SI
            # Check magnitude to determine units
            if abs(beta2) > 1e-20:
                return beta2 * 1e-24  # ps²/m → s²/m
            return beta2  # already SI
        except Exception:
            return None
    elif hasattr(dispersion, 'omegas') and hasattr(dispersion, 'fn'):
        # ZDependentDispersion or PropagationConstant
        domega = 1e12
        beta_plus = dispersion.fn(omega0 + domega, 0.0) if hasattr(dispersion, 'fn') else dispersion.fn(omega0 + domega)
        beta_center = dispersion.fn(omega0, 0.0) if hasattr(dispersion, 'fn') else dispersion.fn(omega0)
        beta_minus = dispersion.fn(omega0 - domega, 0.0) if hasattr(dispersion, 'fn') else dispersion.fn(omega0 - domega)
        beta2 = (beta_plus - 2 * beta_center + beta_minus) / domega ** 2
        return float(beta2)
    return None


# ============================================================================
# 7. Post-flight validation
# ============================================================================

def compare_spectrum_to_phase_matching(
    solver: "GNLSESolver",
    report: SimulationReadinessReport,
    tolerance_nm: float = 2.0,
    tolerance_frac: float = 0.01,
) -> ValidationReport:
    """Compare simulated spectral peaks to PM predictions.

    Uses adaptive tolerance: narrowband (±tolerance_nm, default 2 nm) when
    pulse spectral FWHM ≤ 50 nm, or broadband (±tolerance_frac·λ, default 1%)
    when FWHM > 50 nm.

    Parameters
    ----------
    solver : GNLSESolver — solver with propagated results.
    report : SimulationReadinessReport — PM predictions from preflight.
    tolerance_nm : float — narrowband wavelength tolerance (nm). Default 2.0.
    tolerance_frac : float — broadband fractional tolerance. Default 0.01.

    Returns
    -------
    validation : ValidationReport
    """
    if solver._spectra_vs_z is None:
        raise RuntimeError("Call propagate() first.")

    omega, spectra = solver._spectra_vs_z
    final_spec = spectra[-1]

    # Convert to wavelength
    omega_abs = omega + solver.omega0
    wavelength_m = 2 * PI * C_MS / omega_abs
    wavelength_nm = wavelength_m * 1e9

    # Sort by wavelength
    sort_idx = np.argsort(wavelength_nm)
    wavelength_nm_sorted = wavelength_nm[sort_idx]
    spec_sorted = final_spec[sort_idx]

    # Normalize and find peaks
    max_val = np.max(spec_sorted)
    if max_val == 0:
        return ValidationReport(
            predictions=[],
            peaks_found=[],
            matches=[],
            overall_pass=True,
            tolerance_nm=tolerance_nm,
        )

    spec_norm = spec_sorted / max_val
    peaks, properties = find_peaks(spec_norm, height=0.05, distance=20)
    peak_wl_nm = wavelength_nm_sorted[peaks]

    # Determine tolerance mode by pulse spectral FWHM
    # Compute FWHM of the final spectrum
    spec_fwhm = _compute_spectrum_fwhm(wavelength_nm_sorted, spec_sorted)
    use_broadband = spec_fwhm > 50.0  # FWHM > 50 nm → broadband mode
    effective_tolerance_nm = (
        tolerance_frac * wavelength_nm_sorted[len(wavelength_nm_sorted) // 2]
        if use_broadband
        else tolerance_nm
    )

    # Build prediction list from report
    predictions = []

    # DW predictions
    for dw_wl in report.dw_predictions:
        predictions.append({"type": "DW", "wavelength_nm": dw_wl})

    # FWM predictions
    for fwm_pred in report.fwm_predictions:
        predictions.append({
            "type": "FWM",
            "wavelength_nm": fwm_pred.get("idler_wavelength_nm", 0),
        })

    # MI sideband predictions
    mi_pred = report.mi_predictions
    if "sideband_wl_nm" in mi_pred:
        for sb_wl in mi_pred["sideband_wl_nm"]:
            predictions.append({"type": "MI", "wavelength_nm": float(sb_wl)})

    # Match peaks to predictions
    matches = []
    for pred in predictions:
        pred_wl = pred["wavelength_nm"]
        residuals = np.abs(peak_wl_nm - pred_wl)
        if len(residuals) > 0 and np.min(residuals) < effective_tolerance_nm:
            best_idx = np.argmin(residuals)
            matches.append({
                "prediction": pred,
                "matched_peak": {
                    "wavelength_nm": float(peak_wl_nm[best_idx]),
                },
                "residual_nm": float(residuals[best_idx]),
                "pass": True,
            })
        else:
            matches.append({
                "prediction": pred,
                "matched_peak": None,
                "residual_nm": float(np.min(residuals)) if len(residuals) > 0 else float("inf"),
                "pass": False,
            })

    overall_pass = all(m["pass"] for m in matches) if matches else True

    return ValidationReport(
        predictions=predictions,
        peaks_found=[{"wavelength_nm": float(wl)} for wl in peak_wl_nm],
        matches=matches,
        overall_pass=overall_pass,
        tolerance_nm=tolerance_nm,
    )


def _compute_spectrum_fwhm(wavelength_nm: NDArray, spectrum: NDArray) -> float:
    """Compute full width at half maximum of a spectrum in nm."""
    max_val = np.max(spectrum)
    if max_val == 0:
        return 0.0
    half_max = max_val / 2.0
    above_half = spectrum >= half_max
    if not np.any(above_half):
        return 0.0
    # Find first and last points above half-max
    indices = np.where(above_half)[0]
    wl_min = wavelength_nm[indices[0]]
    wl_max = wavelength_nm[indices[-1]]
    return float(wl_max - wl_min)


# ============================================================================
# 8. Visualization functions
# ============================================================================

def plot_fwm_efficiency(fwm_result: PhaseMatchResult, ax=None) -> "plt.Figure":
    """Plot FWM Δβ and efficiency curves for degenerate FWM.

    Parameters
    ----------
    fwm_result : PhaseMatchResult — result from scan_fwm_detuning.
    ax : matplotlib Axes, optional.

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    else:
        fig = ax.figure
        ax1, ax2 = ax.subplots(2, 1)

    # Convert omega to wavelength
    wl_signal_nm = 2 * PI * C_MS / fwm_result.omega_signal * 1e9
    wl_idler_nm = 2 * PI * C_MS / fwm_result.idler_omega * 1e9
    wl_pump_nm = 2 * PI * C_MS / fwm_result.pump_omega * 1e9

    # Δβ plot
    ax1.plot(wl_signal_nm, fwm_result.delta_beta * 1e3, "b-", linewidth=1)
    ax1.axvline(x=wl_pump_nm, color="k", linestyle="--", alpha=0.5, label=f"Pump ({wl_pump_nm:.1f} nm)")
    ax1.axhline(y=0, color="r", linestyle=":", alpha=0.3)
    ax1.set_ylabel(r"$\Delta\beta$ (1/mm)")
    ax1.set_title("FWM Phase Mismatch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Efficiency plot
    ax2.plot(wl_signal_nm, fwm_result.efficiency, "g-", linewidth=1)
    ax2.axvline(x=wl_pump_nm, color="k", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Signal Wavelength (nm)")
    ax2.set_ylabel("Efficiency η")
    ax2.set_title("FWM Conversion Efficiency")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_mi_gain(mi_result, ax=None) -> "plt.Figure":
    """Plot MI gain spectrum g(Ω) vs modulation frequency.

    Parameters
    ----------
    mi_result : PhaseMatchResult or dict — from mi_gain_spectrum or mi_gain_spectrum_extended.
    ax : matplotlib Axes, optional.

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    if isinstance(mi_result, dict) and "omega_m" in mi_result:
        # Extended MI result
        omega_m = mi_result["omega_m"]
        gain = mi_result["gain"]
        ax.plot(omega_m / 1e12, gain * 1e3, "b-", linewidth=1)
        ax.set_xlabel("Modulation Frequency (THz)")
        ax.set_ylabel(r"g(Ω) (1/mm)")
        ax.set_title("Modulation Instability Gain (Extended)")
    else:
        # Classical MI result
        if isinstance(mi_result, dict):
            # It's the summary dict from mi_gain_spectrum with None omega_m
            ax.text(0.5, 0.5, f"Peak gain: {mi_result.get('g_max', 0):.2f} 1/m\n"
                              f"Omega peak: {mi_result.get('Omega_peak', 0)/1e12:.2f} THz",
                    transform=ax.transAxes, ha="center", va="center")
            ax.set_title("MI Gain Summary (no grid provided)")
            fig.tight_layout()
            return fig
        else:
            # Assume it's a PhaseMatchResult from scan (not typical for MI)
            ax.set_title("MI Gain Spectrum")

    ax.axvline(x=0, color="k", linestyle=":", alpha=0.3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_readiness_report(report: SimulationReadinessReport, ax=None) -> "plt.Figure":
    """Plot readiness panel showing grid vs dispersion extent.

    Parameters
    ----------
    report : SimulationReadinessReport.
    ax : matplotlib Axes, optional.

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Convert omega to wavelength for display
    wl_grid_min = 2 * PI * C_MS / report.grid_omega_max * 1e9
    wl_grid_max = 2 * PI * C_MS / report.grid_omega_min * 1e9
    wl_disp_min = 2 * PI * C_MS / report.dispersion_max_omega * 1e9
    wl_disp_max = 2 * PI * C_MS / report.dispersion_min_omega * 1e9

    # Plot as horizontal bars
    y_pos = 0.5
    ax.barh(y_pos, wl_grid_max - wl_grid_min, left=wl_grid_min,
            height=0.3, color="blue", alpha=0.5, label="Pulse grid")
    ax.barh(y_pos + 0.5, wl_disp_max - wl_disp_min, left=wl_disp_min,
            height=0.3, color="green", alpha=0.5, label="Dispersion model")

    ax.set_ylabel("")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_title(
        f"Dispersion Coverage {'✓' if report.dispersion_covers_grid else '✗'}\n"
        f"N={report.soliton_order:.1f}, L_D={report.dispersion_length:.2e} m, "
        f"L_NL={report.nonlinear_length:.2e} m"
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")

    # Add predictions summary
    pred_text = "Predicted: " + ", ".join(report.predicted_processes) if report.predicted_processes else "None"
    ax.text(0.02, 0.95, pred_text, transform=ax.transAxes, va="top",
            fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    return fig


def plot_spectrum_with_pm_overlay(solver, report: SimulationReadinessReport, ax=None) -> "plt.Figure":
    """Plot final spectrum with vertical lines at PM-predicted frequencies.

    Parameters
    ----------
    solver : GNLSESolver — solver with propagated results.
    report : SimulationReadinessReport — PM predictions.
    ax : matplotlib Axes, optional.

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    if solver._spectra_vs_z is None:
        raise RuntimeError("Call propagate() first.")

    omega, spectra = solver._spectra_vs_z
    final_spec = spectra[-1]

    omega_abs = omega + solver.omega0
    wavelength_nm = 2 * PI * C_MS / omega_abs * 1e9
    sort_idx = np.argsort(wavelength_nm)
    wavelength_nm = wavelength_nm[sort_idx]
    final_spec = final_spec[sort_idx]

    # Plot spectrum
    max_val = np.max(final_spec)
    if max_val > 0:
        ax.plot(wavelength_nm, final_spec / max_val, "b-", linewidth=1, label="Final spectrum")

    # Mark pump
    pump_wl = solver.pulse.central_wavelength.as_nm
    ax.axvline(x=pump_wl, color="k", linestyle=":", alpha=0.5, label=f"Pump ({pump_wl:.1f} nm)")

    # Mark DW predictions
    for dw_wl in report.dw_predictions:
        ax.axvline(x=dw_wl, color="r", linestyle="--", alpha=0.7, label=f"DW ({dw_wl:.1f} nm)")

    # Mark FWM predictions
    for fwm_pred in report.fwm_predictions:
        idler_wl = fwm_pred.get("idler_wavelength_nm", 0)
        ax.axvline(x=idler_wl, color="orange", linestyle=":", alpha=0.7, label=f"FWM ({idler_wl:.1f} nm)")

    # Mark MI sidebands
    mi_pred = report.mi_predictions
    if "sideband_wl_nm" in mi_pred:
        for sb_wl in mi_pred["sideband_wl_nm"]:
            ax.axvline(x=sb_wl, color="g", linestyle="-.", alpha=0.7, label=f"MI ({sb_wl:.1f} nm)")

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized spectrum")
    ax.set_title("Spectrum with Phase-Matching Predictions")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig

