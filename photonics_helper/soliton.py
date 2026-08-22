"""Soliton dynamics analysis module.

Provides diagnostics for soliton propagation simulations:
- Soliton order N
- Dispersion/nonlinear/fission lengths
- Dispersive wave wavelength
- Soliton trajectory extraction
- Soliton counting
- Raman self-frequency shift rate
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

from .base import C_MS, PI, AngularFrequency, Wavelength, AngularFrequencyArray

if TYPE_CHECKING:
    from photonics_helper.gnlse import FiberProfile, GNLSESolver
    from photonics_helper.pulse import Wave

__all__ = ["SolitonAnalyzer", "plot_soliton_trajectories", "plot_fission_dynamics",
           "plot_raman_shift", "plot_dispersion_wave"]


class SolitonAnalyzer:
    """Extract soliton dynamics metrics from GNLSE simulation results.

    Parameters
    ----------
    pulse : Wave
        Input pulse (provides T0, peak power).
    fiber : FiberProfile
        Fiber/waveguide parameters (provides gamma via _gamma).
    betas : array_like
        Dispersion coefficients [beta2, beta3, ...] in ps^2/m.
    z_array : array_like
        Propagation distances (m) for each saved evolution step.
    spectra_vs_z : tuple
        (omega_array, spectra_matrix) from solver.spectra_vs_z.

    Notes
    -----
    The ``dispersive_wave_wavelength(dispersion=None)`` method accepts an
    optional ``dispersion`` source for the full β(ω) root finder. When a
    dispersion source is provided, it uses ``dispersive_wave_roots`` from
    the phase_matching module. When ``dispersion=None`` (default), it falls
    back to the β₂/β₃ analytic formula (Δω = −2β₂/β₃). This is backward
    compatible with existing code that relied on the implicit
    ``pulse._dispersion`` attribute.
    """

    def __init__(
        self,
        pulse: "Wave",
        fiber: "FiberProfile",
        betas: NDArray,
        z_array: NDArray,
        spectra_vs_z: tuple,
    ) -> None:
        self.pulse = pulse
        self.fiber = fiber
        self.betas = np.asarray(betas, dtype=float)
        self.z_array = np.asarray(z_array, dtype=float)
        self.omega, self.spectra = spectra_vs_z

        # Pre-compute gamma
        from .gnlse import _gamma

        self.gamma = _gamma(
            fiber.n2, pulse.central_frequency, fiber.A_eff, fiber.confinement_factor
        )

        # Input pulse parameters
        self.T0 = pulse.envelope.pulse_width.as_s  # s
        self.P_peak = pulse.peak_power()  # W

        # beta2, beta3 in SI (convert from ps^2/m, ps^3/m)
        self.beta2_si = self.betas[0] * 1e-24 if len(self.betas) > 0 else 0.0
        self.beta3_si = self.betas[1] * 1e-27 if len(self.betas) > 1 else 0.0

    def soliton_order(self) -> float:
        """Compute soliton order N = sqrt(gamma * P_peak * T0^2 / |beta2|).

        Returns
        -------
        float
            Soliton order (dimensionless).
        """
        if self.beta2_si == 0:
            raise ValueError("beta2 is zero; soliton order undefined.")
        return np.sqrt(self.gamma * self.P_peak * self.T0**2 / abs(self.beta2_si))

    def dispersion_length(self) -> float:
        """Compute dispersion length L_D = T0^2 / |beta2|.

        Returns
        -------
        float
            Dispersion length (m).
        """
        if self.beta2_si == 0:
            raise ValueError("beta2 is zero; dispersion length undefined.")
        return self.T0**2 / abs(self.beta2_si)

    def nonlinear_length(self) -> float:
        """Compute nonlinear length L_NL = 1 / (gamma * P_peak).

        Returns
        -------
        float
            Nonlinear length (m).
        """
        if self.gamma * self.P_peak == 0:
            raise ValueError("gamma * P_peak is zero; nonlinear length undefined.")
        return 1.0 / (self.gamma * self.P_peak)

    def fission_length(self, eta: float = 0.7) -> float:
        """Compute fission length L_fiss = L_D / (N * eta).

        Parameters
        ----------
        eta : float
            Raman correction factor (default 0.7).

        Returns
        -------
        float
            Fission length (m).
        """
        L_D = self.dispersion_length()
        N = self.soliton_order()
        if N == 0:
            raise ValueError("Soliton order is zero; fission length undefined.")
        return L_D / (N * eta)

    def dispersive_wave_wavelength(
        self,
        dispersion=None,
        wl_range: tuple[Wavelength, Wavelength] | None = None,
        n_brackets: int = 100,
    ) -> Wavelength:
        """Compute dispersive wave (Cherenkov) wavelength from phase-matching.

        Accepts an optional ``dispersion`` source for the full β(ω) root finder.
        When no dispersion source is supplied, falls back to the β₂/β₃ formula
        (Δω = −2β₂/β₃) for the common case.

        Parameters
        ----------
        dispersion : Dispersion or PropagationConstant or ZDependentDispersion, optional
            Dispersion source for full β(ω) root finding. When None, uses the
            β₂/β₃ analytic fallback.
        wl_range : tuple[Wavelength, Wavelength], optional
            Search range as Wavelength objects. Defaults to 0.5× to 3× the pump wavelength.
        n_brackets : int, optional
            Number of bracket intervals for the root finder. Default 100.

        Returns
        -------
        Wavelength
            DW wavelength.

        Raises
        ------
        ValueError
            If no valid root is found and β₂/β₃ data unavailable.
        """
        from photonics_helper.base import Wavelength
        
        if wl_range is None:
            pump_wl = self.pulse.central_wavelength
            wl_range = (
                Wavelength(pump_wl.as_nm * 0.5, "nm"),
                Wavelength(pump_wl.as_nm * 3.0, "nm"),
            )

        # Try full root finder if a dispersion source is provided
        if dispersion is not None:
            try:
                from .phase_matching import (
                    DispersionAdaptor,
                    PropagationConstantAdaptor,
                    ZDependentDispersionAdaptor,
                    dispersive_wave_roots,
                )
                # Build an appropriate adaptor
                if hasattr(dispersion, 'get_betas'):
                    adaptor = DispersionAdaptor(dispersion, self.pulse.central_frequency)
                elif hasattr(dispersion, 'omegas') and hasattr(dispersion, 'fn'):
                    # ZDependentDispersion — fix at z=0
                    adaptor = ZDependentDispersionAdaptor(dispersion, z=0.0)
                else:
                    # PropagationConstant or similar
                    adaptor = PropagationConstantAdaptor(dispersion)

                result = dispersive_wave_roots(
                    adaptor,
                    self.pulse.central_frequency,
                    wl_range=wl_range,
                    n_brackets=n_brackets,
                )
                if result.wavelengths.as_m.shape[0] > 0:
                    return result.wavelengths[0]
            except Exception:
                pass  # Fall through to β₂/β₃

        # β₂/β₃ fast fallback
        if self.beta3_si == 0:
            raise ValueError("beta3 is zero or not available; DW wavelength undefined.")
        delta_omega = -2 * self.beta2_si / self.beta3_si
        omega_dw = self.pulse.central_frequency + delta_omega
        if omega_dw <= 0:
            raise ValueError("DW frequency would be non-positive.")
        return AngularFrequency(omega_dw, "rad/s").to_wl()

    def count_solitons(self, spectrum: Optional[NDArray] = None) -> int:
        """Count solitons in output spectrum using peak detection.

        Parameters
        ----------
        spectrum : NDArray, optional
            Spectrum to analyze. If None, uses last propagation step.

        Returns
        -------
        int
            Number of soliton peaks detected.
        """
        if spectrum is None:
            spectrum = self.spectra[-1]

        # Convert frequency spectrum to wavelength for peak detection
        omega_abs = self.omega + self.pulse.central_frequency
        wavelength = 2 * np.pi * C_MS / omega_abs  # m

        # Sort by wavelength
        sort_idx = np.argsort(wavelength)
        wavelength_sorted = wavelength[sort_idx]
        spectrum_sorted = spectrum[sort_idx]

        # Normalize
        max_val = np.max(spectrum_sorted)
        if max_val == 0:
            return 0
        spectrum_norm = spectrum_sorted / max_val

        # Find peaks: require height > 5% of max, minimum distance of 10 points
        peaks, _ = find_peaks(spectrum_norm, height=0.05, distance=10)

        return len(peaks)

    def soliton_trajectories(self) -> list[tuple[float, float]]:
        """Extract soliton trajectories (peak wavelength vs propagation distance).

        Returns
        -------
        list of (z, lambda_peak_nm) tuples for each identified soliton peak.
        """
        trajectories = []
        n_steps = self.spectra.shape[0]

        for i in range(n_steps):
            spectrum = self.spectra[i]
            omega_abs = self.omega + self.pulse.central_frequency
            wavelength = 2 * np.pi * C_MS / omega_abs  # m

            # Sort by wavelength
            sort_idx = np.argsort(wavelength)
            wavelength_sorted = wavelength[sort_idx]
            spectrum_sorted = spectrum[sort_idx]

            # Normalize
            max_val = np.max(spectrum_sorted)
            if max_val == 0:
                continue
            spectrum_norm = spectrum_sorted / max_val

            # Find peaks
            peaks, _ = find_peaks(spectrum_norm, height=0.05, distance=10)

            for peak_idx in peaks:
                z = self.z_array[i]
                lam_nm = wavelength_sorted[peak_idx] * 1e9  # nm
                trajectories.append((z, lam_nm))

        return trajectories

    def raman_shift_rate(self) -> float:
        """Compute Raman self-frequency shift rate (nm/mm).

        Fits a line to the soliton trajectory in the linear regime
        and returns the slope.

        Returns
        -------
        float
            RSFS rate in nm/mm.
        """
        trajectories = self.soliton_trajectories()
        if len(trajectories) < 2:
            return 0.0

        z_vals = np.array([t[0] for t in trajectories])
        lam_vals = np.array([t[1] for t in trajectories])

        # Sort by z
        sort_idx = np.argsort(z_vals)
        z_vals = z_vals[sort_idx]
        lam_vals = lam_vals[sort_idx]

        # Fit linear trend (nm vs mm)
        z_mm = z_vals * 1e3  # m to mm
        if len(z_mm) > 1:
            coeffs = np.polyfit(z_mm, lam_vals, 1)
            return float(coeffs[0])  # nm/mm
        return 0.0


# ---------------------------------------------------------------------------
# Visualization utilities
# ---------------------------------------------------------------------------

def plot_soliton_trajectories(solver: "GNLSESolver", ax=None) -> "plt.Figure":
    """Plot individual soliton peak wavelengths vs propagation distance.

    Parameters
    ----------
    solver : GNLSESolver
        Solver with propagated results.
    ax : matplotlib Axes, optional
        Axis to plot on.

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt

    analyzer = SolitonAnalyzer(
        pulse=solver.pulse,
        fiber=solver.fiber,
        betas=solver.betas,
        z_array=solver.z_array,
        spectra_vs_z=solver.spectra_vs_z,
    )
    trajectories = analyzer.soliton_trajectories()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    if trajectories:
        z_vals = np.array([t[0] for t in trajectories]) * 1e3  # mm
        lam_vals = np.array([t[1] for t in trajectories])  # nm
        ax.scatter(z_vals, lam_vals, s=5, alpha=0.5, label="Soliton peaks")

    ax.set_xlabel("Propagation distance (mm)")
    ax.set_ylabel("Peak wavelength (nm)")
    ax.set_title("Soliton Trajectories")
    ax.grid(True, alpha=0.3)
    return fig


def plot_fission_dynamics(solver: "GNLSESolver", N: float, L_D: float,
                          ax=None) -> "plt.Figure":
    """Plot soliton fission process: spectrum evolution with fission length marker.

    Parameters
    ----------
    solver : GNLSESolver
        Solver with propagated results.
    N : float
        Soliton order.
    L_D : float
        Dispersion length (m).
    ax : matplotlib Axes, optional
        Axis to plot on.

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt

    omega, spectra = solver.spectra_vs_z
    omega_abs = omega + solver.omega0
    wavelength_array = AngularFrequencyArray(omega_abs, "rad/s").to_wl()
    sort_idx = np.argsort(wavelength_array.as_nm)
    wavelength_nm = wavelength_array.as_nm[sort_idx]
    spectra = spectra[:, sort_idx]

    z_steps = solver.z_array * 1e3  # mm
    fission_length_mm = L_D / N * 0.7 * 1e3 if N > 0 else float("inf")  # mm

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Plot spectrum at several z positions
    n_plot = min(10, spectra.shape[0])
    z_plot = np.linspace(0, z_steps[-1], n_plot)
    for z in z_plot:
        idx = int(z / z_steps[-1] * (spectra.shape[0] - 1)) if z_steps[-1] > 0 else 0
        spec = spectra[idx]
        max_val = np.max(spec)
        if max_val > 0:
            ax.plot(wavelength_nm, spec / max_val, alpha=0.5, linewidth=0.5)

    # Mark fission length
    if fission_length_mm < z_steps[-1]:
        ax.axvline(x=fission_length_mm, color="r", linestyle="--", alpha=0.5,
                   label=f"Fission length (L_D/{N}·0.7)")

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized spectrum")
    ax.set_title(f"Soliton Fission Dynamics (N={N:.1f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_raman_shift(solver: "GNLSESolver", ax=None) -> "plt.Figure":
    """Plot soliton peak wavelength drift due to Raman self-frequency shift.

    Parameters
    ----------
    solver : GNLSESolver
        Solver with propagated results.
    ax : matplotlib Axes, optional
        Axis to plot on.

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt

    analyzer = SolitonAnalyzer(
        pulse=solver.pulse,
        fiber=solver.fiber,
        betas=solver.betas,
        z_array=solver.z_array,
        spectra_vs_z=solver.spectra_vs_z,
    )
    trajectories = analyzer.soliton_trajectories()
    rate = analyzer.raman_shift_rate()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    if trajectories:
        z_vals = np.array([t[0] for t in trajectories]) * 1e3  # mm
        lam_vals = np.array([t[1] for t in trajectories])  # nm
        ax.plot(z_vals, lam_vals, "o-", markersize=3, alpha=0.7)

        # Fit and plot trend line
        if len(z_vals) > 1:
            coeffs = np.polyfit(z_vals, lam_vals, 1)
            z_fit = np.array([z_vals.min(), z_vals.max()])
            ax.plot(z_fit, coeffs[0] * z_fit + coeffs[1], "r--", alpha=0.5,
                    label=f"Rate: {coeffs[0]:.3f} nm/mm")

    ax.set_xlabel("Propagation distance (mm)")
    ax.set_ylabel("Peak wavelength (nm)")
    ax.set_title("Raman Self-Frequency Shift")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_dispersion_wave(solver: "GNLSESolver", ax=None) -> "plt.Figure":
    """Plot final spectrum with dispersive wave wavelength marked.

    Parameters
    ----------
    solver : GNLSESolver
        Solver with propagated results.
    ax : matplotlib Axes, optional
        Axis to plot on.

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt

    omega, spectra = solver.spectra_vs_z
    omega_abs = omega + solver.omega0
    wavelength_array = AngularFrequencyArray(omega_abs, "rad/s").to_wl()
    sort_idx = np.argsort(wavelength_array.as_nm)
    wavelength_nm = wavelength_array.as_nm[sort_idx]
    spectra = spectra[:, sort_idx]

    # Plot final spectrum
    final_spec = spectra[-1]
    max_val = np.max(final_spec)
    if max_val > 0:
        final_spec = final_spec / max_val

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    ax.plot(wavelength_nm, final_spec, "b-", linewidth=1, label="Final spectrum")

    # Mark pump wavelength
    pump_wl = solver.pulse.central_wavelength
    ax.axvline(x=pump_wl.as_nm, color="k", linestyle=":", alpha=0.5, label=f"Pump ({pump_wl.as_nm:.1f} nm)")

    # Mark DW wavelength if available
    try:
        analyzer = SolitonAnalyzer(
            pulse=solver.pulse,
            fiber=solver.fiber,
            betas=solver.betas,
            z_array=solver.z_array,
            spectra_vs_z=solver.spectra_vs_z,
        )
        dw_wl = analyzer.dispersive_wave_wavelength()
        ax.axvline(x=dw_wl.as_nm, color="r", linestyle="--", alpha=0.5, label=f"DW ({dw_wl.as_nm:.1f} nm)")
    except (ValueError, IndexError):
        pass

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized spectrum")
    ax.set_title("Spectrum with Dispersive Wave")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig
