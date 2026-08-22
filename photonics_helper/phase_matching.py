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

from .base import C_MS, PI, Wavelength, WavelengthArray, AngularFrequency, AngularFrequencyArray

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
        # Pre-compute β₁(ω) and β(ω) on the reference grid via double integration
        # Reference: β₁(ω₀) = 0, β(ω₀) = 0 (absolute values not needed for PM)
        domega_ref = omega_sorted - self.omega0
        # First integration: β₁(ω) = ∫ β₂ dω (from ω₀ to ω)
        beta1_sorted = cumulative_trapezoid(beta2_sorted, domega_ref, initial=0.0)
        # Second integration: β(ω) = ∫ β₁ dω (from ω₀ to ω)
        beta_sorted = cumulative_trapezoid(beta1_sorted, domega_ref, initial=0.0)
        # Build splines for β₁(ω) and β(ω)
        self._beta1_spline = UnivariateSpline(omega_sorted, beta1_sorted, s=0)
        self._beta_spline = UnivariateSpline(omega_sorted, beta_sorted, s=0)

    def __call__(self, omega: NDArray | float) -> NDArray | float:
        """Return β(ω)."""
        return self.beta(omega)

    def beta(self, omega: NDArray | float) -> NDArray | float:
        """Return β(ω) via spline interpolation of pre-integrated values."""
        scalar = np.isscalar(omega)
        omega_arr = np.atleast_1d(np.asarray(omega, dtype=float))
        result = self._beta_spline(omega_arr)
        if scalar:
            return float(result[0])
        return result

    def beta1(self, omega: NDArray | float) -> NDArray | float:
        """Return β₁(ω) = dβ/dω via spline interpolation of pre-integrated values."""
        scalar = np.isscalar(omega)
        omega_arr = np.atleast_1d(np.asarray(omega, dtype=float))
        result = self._beta1_spline(omega_arr)
        if scalar:
            return float(result[0])
        return result


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
    wavelengths : WavelengthArray — dispersive wave wavelengths.
    soliton_omega : AngularFrequency — soliton center angular frequency.
    soliton_wavelength : Wavelength — soliton wavelength.
    q_sol : float — soliton wavenumber correction (1/m).
    """

    wavelengths: WavelengthArray
    soliton_omega: AngularFrequency
    soliton_wavelength: Wavelength
    q_sol: float = 0.0


@dataclass
class SimulationReadinessReport:
    """Report on simulation readiness.

    Attributes
    ----------
    dispersion_covers_grid : bool — True if dispersion covers the pulse grid.
    grid_omega_min : AngularFrequency — minimum grid angular frequency.
    grid_omega_max : AngularFrequency — maximum grid angular frequency.
    dispersion_min_omega : AngularFrequency — minimum dispersion model frequency.
    dispersion_max_omega : AngularFrequency — maximum dispersion model frequency.
    soliton_order : float — estimated soliton order N.
    dispersion_length : float — L_D (m).
    nonlinear_length : float — L_NL (m).
    fission_length : float — L_fiss (m).
    recommended_num_steps : int — suggested step count.
    predicted_processes : list[str] — predicted nonlinear processes.
    warnings : list[str] — warnings about simulation setup.
    recommendations : list[str] — suggestions for improvement.
    fwm_predictions : list[dict] — predicted FWM idler wavelengths (contains Wavelength).
    mi_predictions : dict — predicted MI sideband info (contains WavelengthArray).
    dw_predictions : WavelengthArray — predicted DW wavelengths.
    """

    dispersion_covers_grid: bool
    grid_omega_min: AngularFrequency
    grid_omega_max: AngularFrequency
    dispersion_min_omega: AngularFrequency
    dispersion_max_omega: AngularFrequency
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
    dw_predictions: WavelengthArray = field(default_factory=lambda: WavelengthArray(np.array([]), "nm"))


@dataclass
class ValidationReport:
    """Post-flight validation report.

    Attributes
    ----------
    predictions : list[dict] — list of PM predictions with Wavelength key.
    peaks_found : list[dict] — list of detected spectral peaks with Wavelength.
    matches : list[dict] — each has prediction, matched_peak, residual (Wavelength), pass.
    overall_pass : bool — True if all predictions have a matching peak.
    tolerance : Wavelength — wavelength tolerance used.
    """

    predictions: list[dict]
    peaks_found: list[dict]
    matches: list[dict]
    overall_pass: bool
    tolerance: Wavelength


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

    Lossless (α=0): η = sinc²(Δβ·L/2)

    Lossy (α>0): Full formula from plan §2.3:
        η = [α²/(α²+Δβ²)] · { 1 + [4 e^{-αL} sin²(Δβ·L/2)] / [(1−e^{-αL})² (α²+Δβ²)] }

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

    if alpha == 0:
        # Lossless case: η = sinc²(Δβ·L/2)
        arg = delta_beta_arr * length_m / 2.0
        # Use safe division to avoid warning at arg=0
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(
                np.isclose(arg, 0.0),
                1.0,
                (np.sin(arg) / arg) ** 2,
            )
    else:
        # Lossy case: full formula from plan §2.3
        # η = [α²/(α²+Δβ²)] · { 1 + [4 e^{-αL} sin²(Δβ·L/2)] / [(1−e^{-αL})² (α²+Δβ²)] }
        alpha_sq = alpha * alpha
        db_sq = delta_beta_arr * delta_beta_arr
        denom = alpha_sq + db_sq
        
        # sin²(Δβ·L/2) uses actual length L (not L_eff) for phase accumulation
        arg = delta_beta_arr * length_m / 2.0
        sin_term = np.sin(arg) ** 2
        
        exp_term = np.exp(-alpha * length_m)
        one_minus_exp = 1.0 - exp_term
        
        # First factor: α²/(α²+Δβ²)
        factor1 = alpha_sq / denom
        
        # Second factor: 1 + [4 e^{-αL} sin²(Δβ·L/2)] / [(1−e^{-αL})² (α²+Δβ²)]
        factor2 = 1.0 + (4.0 * exp_term * sin_term) / (one_minus_exp * one_minus_exp * denom)
        
        result = factor1 * factor2
        
        # Handle Δβ = 0 case: the formula limit gives η = 1
        # This is the parametric gain limit at perfect phase matching
        zero_mask = np.isclose(delta_beta_arr, 0.0)
        if np.any(zero_mask):
            result[zero_mask] = 1.0

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
        gp = gamma * P_pump
        if np.abs(db) < 1e-15:
            # Perfect phase matching (Δβ = 0): degenerate FWM efficiency
            # η = sin²(γP·L_eff) where L_eff = (1 - e^{-αL})/α for α>0, else L
            if alpha > 0:
                L_eff = (1 - np.exp(-alpha * L)) / alpha
            else:
                L_eff = L
            arg = gp * L_eff
            efficiency[i] = np.sin(arg) ** 2
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

    Peak gain at Ω = Ω_c/√2: g_max = 2γP

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
        g_max = 2.0 * gamma * P  # peak gain
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
    """Extended MI gain using full dispersion relation.

    Computes the MI gain spectrum using the exact dispersion β(ω) instead of
    a Taylor expansion. Based on the linear stability analysis of the NLSE,
    the gain is consistent with the classical formula in this module:

        g(Ω) = 2√[−D(Ω)·(D(Ω) + 4γP)/4]  for -4γP < D(Ω) < 0
        g(Ω) = 0  otherwise

    where D(Ω) = 2[β(ω₀+Ω) + β(ω₀−Ω) − 2β(ω₀)] is twice the even part of
    the linear dispersion. This scaling ensures the extended formula reduces
    exactly to the classical result g(Ω) = 2|β₂|Ω√(Ω_c² − Ω²) with
    Ω_c² = 2γP/|β₂| when β(ω) ≈ β₀ + β₁Ω + ½β₂Ω².

    Parameters
    ----------
    beta_fn : callable — β(ω) function.
    omega0 : float — pump carrier frequency (rad/s).
    gamma : float — nonlinear coefficient (1/(W·m)).
    P : float — pump power (W).
    alpha : float — loss (1/m). Default 0 (not used in gain formula).
    L : float — length (m). If None, 1/γ (not used in gain formula).
    omega_m : 1-D array — modulation frequencies (rad/s). If None,
              auto-generates a grid based on classical estimate.

    Returns
    -------
    result : dict with keys 'omega_m', 'gain', 'Omega_peak', 'Omega_cutoff'.
    """
    if L is None:
        L = 1.0 / max(gamma, 1e-30)

    if omega_m is None:
        # Auto-generate grid using classical estimate for scale
        # Classical cutoff: Ω_c² = 2γP/|β₂|
        domega = 1e12
        beta2_est = (beta_fn(omega0 + domega) - 2 * beta_fn(omega0) + beta_fn(omega0 - domega)) / domega**2
        if beta2_est < 0:
            Omega_classical = np.sqrt(max(2 * gamma * P / abs(beta2_est), 1e12))
        else:
            Omega_classical = 1e12
        omega_m = np.linspace(-3 * Omega_classical, 3 * Omega_classical, 500)

    omega_arr = np.atleast_1d(np.asarray(omega_m, dtype=float))
    beta_pump = beta_fn(omega0)

    # D(Ω) = 2[β(ω₀+Ω) + β(ω₀−Ω) − 2β(ω₀)]
    # This scaling matches the classical formula in mi_gain_spectrum
    omega_plus = omega0 + omega_arr
    omega_minus = omega0 - omega_arr
    beta_plus = beta_fn(omega_plus)
    beta_minus = beta_fn(omega_minus)
    D = 2.0 * (beta_plus + beta_minus - 2 * beta_pump)

    # Extended MI gain: g(Ω) = √[−D(Ω)·(D(Ω) + 4γP)]
    # Gain exists when -4γP < D < 0
    term = -D * (D + 4 * gamma * P)
    gain = np.zeros_like(term)
    mask = term > 0
    gain[mask] = np.sqrt(term[mask])

    # Find peak and cutoff
    if mask.any():
        # Peak gain: occurs at D = -2γP
        Omega_peak = omega_arr[mask][np.argmax(gain[mask])]
        # Cutoff: gain goes to zero at D = 0 and D = -4γP
        # We want the largest |Ω| where gain > 0
        Omega_cutoff = np.max(np.abs(omega_arr[mask]))
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
    omega_sol: float | AngularFrequency,
    q_sol: float = 0.0,
    wl_range: tuple[Wavelength, Wavelength] | None = None,
    n_brackets: int = 50,
) -> DispersiveWaveResult:
    """Find all dispersive wave (Cherenkov) frequencies.

    Solves β(ω_DW) = β(ωₛ) + β₁(ωₛ)(ω_DW − ωₛ) + q_sol

    Uses bracket search over wavelength range with scipy.optimize.root_scalar.

    Parameters
    ----------
    beta_fn : callable — β(ω) function (DispersionModel or similar).
    omega_sol : float or AngularFrequency — soliton center angular frequency.
    q_sol : float — soliton wavenumber correction (1/m). Default 0.
    wl_range : tuple[Wavelength, Wavelength] — (wl_min, wl_max) search range.
                    Defaults to 300 nm – 2500 nm.
    n_brackets : int — number of bracket intervals to scan.

    Returns
    -------
    result : DispersiveWaveResult with wavelengths as WavelengthArray
    """
    if wl_range is None:
        wl_range = (Wavelength(300, "nm"), Wavelength(2500, "nm"))
    
    wl_min, wl_max = wl_range
    # Create wavelength grid
    wl_values_nm = np.linspace(wl_min.as_nm, wl_max.as_nm, n_brackets)
    wl_grid = WavelengthArray(wl_values_nm, "nm")
    omega_grid = wl_grid.to_omega().as_rad_s

    omega_sol_val = omega_sol.as_rad_s if isinstance(omega_sol, AngularFrequency) else omega_sol

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
                    wl_check = AngularFrequency(res.root).to_wl().as_nm
                    if wl_min.as_nm <= wl_check <= wl_max.as_nm:
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
    
    # Convert roots to WavelengthArray
    if roots_omega:
        wl_values = np.array([AngularFrequency(w, "rad/s").to_wl().as_m for w in roots_omega])
        roots_wavelengths = WavelengthArray(wl_values, "m")
    else:
        roots_wavelengths = WavelengthArray(np.array([]), "nm")
    soliton_wavelength = AngularFrequency(omega_sol_val, "rad/s").to_wl()

    return DispersiveWaveResult(
        wavelengths=roots_wavelengths,
        soliton_omega=AngularFrequency(omega_sol_val, "rad/s"),
        soliton_wavelength=soliton_wavelength,
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
        disp_min_omega = AngularFrequency(dispersion.omegas.min())
        disp_max_omega = AngularFrequency(dispersion.omegas.max())
    elif hasattr(dispersion, 'wavelengths'):
        # Dispersion — convert wavelengths to omega
        wl_arr = dispersion.wavelengths
        disp_min_omega = wl_arr.to_omega().min()
        disp_max_omega = wl_arr.to_omega().max()
    else:
        # Taylor mode — use a wide estimate
        disp_min_omega = AngularFrequency(omega0 - 5e15, "rad/s")
        disp_max_omega = AngularFrequency(omega0 + 5e15, "rad/s")

    # Check coverage
    covers = (omega_min >= disp_min_omega.as_rad_s - 1e12) and (omega_max <= disp_max_omega.as_rad_s + 1e12)

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
            omega_peak = AngularFrequency(mi_info["Omega_peak"], "rad/s")
            omega_cutoff = AngularFrequency(mi_info["Omega_cutoff"], "rad/s")
            sideband_plus = AngularFrequency(omega0 + mi_info["Omega_peak"], "rad/s").to_wl()
            sideband_minus = AngularFrequency(omega0 - mi_info["Omega_peak"], "rad/s").to_wl()
            # Create WavelengthArray from raw meter values
            sb_values = np.array([sideband_plus.as_m, sideband_minus.as_m])
            mi_predictions = {
                "Omega_peak": omega_peak,
                "Omega_cutoff": omega_cutoff,
                "g_max": mi_info["g_max"],
                "sideband_wavelengths": WavelengthArray(sb_values, "m"),
            }
        else:
            mi_predictions = {}
    else:
        mi_predictions = {}

    # DW predictions (if we have betas)
    dw_preds = WavelengthArray(np.array([]), "nm")
    if betas is not None and len(betas) >= 2 and beta2_si != 0:
        try:
            # Quick DW estimate using β₂/β₃ formula
            beta3_si = betas[1] * 1e-27 if len(betas) > 1 else 0.0
            if beta3_si != 0 and abs(beta3_si) > 1e-40:
                delta_omega_dw = -2 * beta2_si / beta3_si
                omega_dw = omega0 + delta_omega_dw
                if omega_dw > 0:
                    dw_wavelength = AngularFrequency(omega_dw).to_wl()
                    dw_preds = WavelengthArray([dw_wavelength])
        except Exception:
            pass

    # FWM predictions (degenerate, signal at DW wavelength if available)
    fwm_preds = []
    if dw_preds.as_m.shape[0] > 0 and len(fwm_preds) == 0:
        # Predict FWM idler for signal near DW
        for dw_wavelength in dw_preds:
            omega_s = dw_wavelength.to_omega().as_rad_s
            omega_i = fwm_idler_frequency(omega0, omega_s)
            fwm_preds.append({
                "idler_wavelength": AngularFrequency(omega_i).to_wl(),
                "signal_wavelength": dw_wavelength,
            })

    return SimulationReadinessReport(
        dispersion_covers_grid=covers,
        grid_omega_min=AngularFrequency(omega_min, "rad/s"),
        grid_omega_max=AngularFrequency(omega_max, "rad/s"),
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


def _estimate_beta2(dispersion, omega0: float | AngularFrequency) -> float | None:
    """Estimate β₂ from a dispersion source object."""
    omega0_val = omega0.as_rad_s if isinstance(omega0, AngularFrequency) else omega0
    
    if hasattr(dispersion, 'get_beta2'):
        # Dispersion object
        wl = AngularFrequency(omega0_val).to_wl()
        try:
            beta2 = dispersion.get_beta2(wl)
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
        beta_plus = dispersion.fn(omega0_val + domega, 0.0) if hasattr(dispersion, 'fn') else dispersion.fn(omega0_val + domega)
        beta_center = dispersion.fn(omega0_val, 0.0) if hasattr(dispersion, 'fn') else dispersion.fn(omega0_val)
        beta_minus = dispersion.fn(omega0_val - domega, 0.0) if hasattr(dispersion, 'fn') else dispersion.fn(omega0_val - domega)
        beta2 = (beta_plus - 2 * beta_center + beta_minus) / domega ** 2
        return float(beta2)
    return None


# ============================================================================
# 7. Post-flight validation
# ============================================================================

def compare_spectrum_to_phase_matching(
    solver: "GNLSESolver",
    report: SimulationReadinessReport,
    tolerance: Wavelength | float = 2.0,
    tolerance_frac: float = 0.01,
) -> ValidationReport:
    """Compare simulated spectral peaks to PM predictions.

    Uses adaptive tolerance: narrowband (±tolerance, default 2 nm) when
    pulse spectral FWHM ≤ 50 nm, or broadband (±tolerance_frac·λ, default 1%)
    when FWHM > 50 nm.

    Parameters
    ----------
    solver : GNLSESolver — solver with propagated results.
    report : SimulationReadinessReport — PM predictions from preflight.
    tolerance : Wavelength or float — narrowband wavelength tolerance.
                If float, interpreted as nm. Default 2.0 nm.
    tolerance_frac : float — broadband fractional tolerance. Default 0.01.

    Returns
    -------
    validation : ValidationReport
    """
    if solver._spectra_vs_z is None:
        raise RuntimeError("Call propagate() first.")

    # Handle tolerance input
    if isinstance(tolerance, (int, float)):
        tolerance = Wavelength(tolerance, "nm")
    
    omega, spectra = solver._spectra_vs_z
    final_spec = spectra[-1]

    # Convert to wavelength using base classes
    omega_abs = omega + solver.omega0
    wavelength_array = AngularFrequencyArray(omega_abs, "rad/s").to_wl()
    
    # Sort by wavelength
    sort_idx = np.argsort(wavelength_array.as_nm)
    wavelength_sorted = WavelengthArray(wavelength_array.as_nm[sort_idx], "nm")
    spec_sorted = final_spec[sort_idx]

    # Normalize and find peaks
    max_val = np.max(spec_sorted)
    if max_val == 0:
        return ValidationReport(
            predictions=[],
            peaks_found=[],
            matches=[],
            overall_pass=True,
            tolerance=tolerance,
        )

    spec_norm = spec_sorted / max_val
    peaks, properties = find_peaks(spec_norm, height=0.05, distance=20)
    peak_wavelengths = WavelengthArray(wavelength_sorted.as_nm[peaks], "nm")

    # Determine tolerance mode by pulse spectral FWHM
    # Compute FWHM of the final spectrum
    spec_fwhm = _compute_spectrum_fwhm(wavelength_sorted.as_nm, spec_sorted)
    use_broadband = spec_fwhm > 50.0  # FWHM > 50 nm → broadband mode
    
    if use_broadband:
        effective_tolerance = Wavelength(tolerance_frac * wavelength_sorted.as_nm[wavelength_sorted.as_m.shape[0] // 2], "nm")
    else:
        effective_tolerance = tolerance

    # Build prediction list from report
    predictions = []

    # DW predictions
    for dw_wl in report.dw_predictions.as_nm:
        if isinstance(dw_wl, Wavelength):
            predictions.append({"type": "DW", "wavelength": dw_wl})
        else:
            # Handle legacy float nm values
            predictions.append({"type": "DW", "wavelength": Wavelength(dw_wl, "nm")})

    # FWM predictions
    for fwm_pred in report.fwm_predictions:
        idler_wl = fwm_pred.get("idler_wavelength")
        if idler_wl is not None:
            if isinstance(idler_wl, Wavelength):
                predictions.append({"type": "FWM", "wavelength": idler_wl})
            else:
                predictions.append({"type": "FWM", "wavelength": Wavelength(idler_wl, "nm")})

    # MI sideband predictions
    mi_pred = report.mi_predictions
    if "sideband_wavelengths" in mi_pred:
        for sb_wl in mi_pred["sideband_wavelengths"].as_nm:
            if isinstance(sb_wl, Wavelength):
                predictions.append({"type": "MI", "wavelength": sb_wl})
            else:
                predictions.append({"type": "MI", "wavelength": Wavelength(sb_wl, "nm")})

    # Match peaks to predictions
    matches = []
    for pred in predictions:
        pred_wl = pred["wavelength"]
        if pred_wl is None:
            continue
        # Convert to nm for comparison
        pred_wl_nm = pred_wl.as_nm if isinstance(pred_wl, Wavelength) else pred_wl
        residuals = np.abs(peak_wavelengths.as_nm - pred_wl_nm)
        if len(residuals) > 0 and np.min(residuals) < effective_tolerance.as_nm:
            best_idx = np.argmin(residuals)
            matches.append({
                "prediction": pred,
                "matched_peak": {
                    "wavelength": Wavelength(peak_wavelengths.as_nm[best_idx], "nm"),
                },
                "residual": Wavelength(residuals[best_idx], "nm"),
                "pass": True,
            })
        else:
            matches.append({
                "prediction": pred,
                "matched_peak": None,
                "residual": Wavelength(np.min(residuals) if len(residuals) > 0 else float("inf"), "nm"),
                "pass": False,
            })

    overall_pass = all(m["pass"] for m in matches) if matches else True

    return ValidationReport(
        predictions=predictions,
        peaks_found=[{"wavelength": Wavelength(wl, "nm")} for wl in peak_wavelengths.as_nm],
        matches=matches,
        overall_pass=overall_pass,
        tolerance=tolerance,
    )


def _compute_spectrum_fwhm(wavelength_nm: NDArray, spectrum: NDArray) -> float:
    """Compute full width at half maximum of a spectrum in nm.
    
    Args:
        wavelength_nm: Wavelength values in nm (can be WavelengthArray.as_nm)
        spectrum: Spectrum values
    """
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

    # Convert omega to wavelength using base classes
    wl_signal = AngularFrequencyArray(fwm_result.omega_signal, "rad/s").to_wl()
    wl_idler = AngularFrequencyArray(fwm_result.idler_omega, "rad/s").to_wl()
    wl_pump = AngularFrequency(fwm_result.pump_omega, "rad/s").to_wl()

    wl_signal_nm = wl_signal.as_nm
    wl_idler_nm = wl_idler.as_nm
    wl_pump_nm = wl_pump.as_nm

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
        # Handle AngularFrequencyArray or raw array
        if hasattr(omega_m, 'as_rad_s'):
            omega_m_thz = omega_m.as_rad_s / 1e12
        else:
            omega_m_thz = np.asarray(omega_m) / 1e12
        ax.plot(omega_m_thz, gain * 1e3, "b-", linewidth=1)
        ax.set_xlabel("Modulation Frequency (THz)")
        ax.set_ylabel(r"g(Ω) (1/mm)")
        ax.set_title("Modulation Instability Gain (Extended)")
    else:
        # Classical MI result
        if isinstance(mi_result, dict):
            # It's the summary dict from mi_gain_spectrum with None omega_m
            omega_peak = mi_result.get('Omega_peak', 0)
            if hasattr(omega_peak, 'as_rad_s'):
                omega_peak_thz = omega_peak.as_rad_s / 1e12
            else:
                omega_peak_thz = omega_peak / 1e12
            ax.text(0.5, 0.5, f"Peak gain: {mi_result.get('g_max', 0):.2f} 1/m\n"
                              f"Omega peak: {omega_peak_thz:.2f} THz",
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

    # Convert omega to wavelength for display using base classes
    wl_grid_min = report.grid_omega_max.to_wl().as_nm
    wl_grid_max = report.grid_omega_min.to_wl().as_nm
    wl_disp_min = report.dispersion_max_omega.to_wl().as_nm
    wl_disp_max = report.dispersion_min_omega.to_wl().as_nm

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
    wavelength_array = AngularFrequencyArray(omega_abs, "rad/s").to_wl()
    sort_idx = np.argsort(wavelength_array.as_nm)
    wavelength_nm = wavelength_array.as_nm[sort_idx]
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
    for dw_wl in report.dw_predictions.as_nm:
        ax.axvline(x=dw_wl, color="r", linestyle="--", alpha=0.7, label=f"DW ({dw_wl:.1f} nm)")

    # Mark FWM predictions
    for fwm_pred in report.fwm_predictions:
        idler_wl = fwm_pred.get("idler_wavelength")
        if idler_wl is not None:
            ax.axvline(x=idler_wl.as_nm, color="orange", linestyle=":", alpha=0.7, label=f"FWM ({idler_wl.as_nm:.1f} nm)")

    # Mark MI sidebands
    mi_pred = report.mi_predictions
    if "sideband_wavelengths" in mi_pred:
        for sb_wl in mi_pred["sideband_wavelengths"]:
            ax.axvline(x=sb_wl.as_nm, color="g", linestyle="-.", alpha=0.7, label=f"MI ({sb_wl.as_nm:.1f} nm)")

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized spectrum")
    ax.set_title("Spectrum with Phase-Matching Predictions")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig

