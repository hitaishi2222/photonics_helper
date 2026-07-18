"""Generalized nonlinear Schrödinger equation (GNLSE) solver.

Split-step Fourier solver for pulse propagation through nonlinear media,
supporting Kerr, Raman, self-steepening, and two-photon absorption effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from photonics_helper.base import C_MS, Area, Length, Time

if TYPE_CHECKING:
    from photonics_helper.pulse import Wave, TemporalGrid

__all__ = ["FiberProfile", "GNLSESolver", "SplitStepEngine"]


# ---------------------------------------------------------------------------
# FiberProfile
# ---------------------------------------------------------------------------

@dataclass
class FiberProfile:
    """Optical fiber parameters for GNLSE propagation.

    Attributes:
        n2: Nonlinear refractive index n₂ (m²/W).
        alpha: Fiber loss coefficient α (1/m).
        A_eff: Effective mode area (m²).
        sigma_tpa: Two-photon absorption cross-section (m²·W⁻¹). Default 0.
        carrier_lifetime: Carrier recombination lifetime (s). Default None.
        length: Fiber length (m).
        raman_response: Raman response function. Default None.
    """

    n2: float
    alpha: float
    A_eff: Area
    length: Length
    sigma_tpa: float = 0.0
    carrier_lifetime: Optional[Time] = None
    raman_response: Optional[object] = None


# ---------------------------------------------------------------------------
# Nonlinear effect functions
# ---------------------------------------------------------------------------

def _gamma(n2: float, omega0: float, A_eff: Area) -> float:
    """Nonlinear coefficient γ = n₂·ω₀ / (c·A_eff)."""
    return n2 * omega0 / (C_MS * A_eff.as_m2)


def kerr_step(
    A: NDArray,
    fiber: "FiberProfile",
    grid: "TemporalGrid",
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
    gamma = _gamma(fiber.n2, omega0, fiber.A_eff)
    phase = np.exp(1j * gamma * np.abs(A) ** 2 * dz)
    return A * phase


def raman_step(
    A: NDArray,
    fiber: "FiberProfile",
    grid: "TemporalGrid",
    dz: float,
    include_raman: bool,
    omega0: float = 0.0,
) -> NDArray:
    """Apply Raman convolution: P_Raman = (1-fR)|A|² + fR·(h_R ⊗ |A|²).

    Uses FFT-based circular convolution with proper dt normalization.
    h_R is evaluated on a causal delay grid [0, dt, 2dt, ...].

    Parameters
    ----------
    A : complex array — pulse envelope.
    fiber : FiberProfile — fiber parameters.
    grid : TemporalGrid — time grid.
    dz : float — step size (m).
    include_raman : bool — whether to include Raman.

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

    N = grid.N
    intensity = np.abs(A) ** 2

    fR = fiber.raman_response.fR if hasattr(fiber.raman_response, 'fR') else 0.18

    P_inst = (1.0 - fR) * intensity

    if hasattr(fiber.raman_response, '_h_R'):
        delays = np.arange(N) * grid.dt
        h_R_kernel = fiber.raman_response._h_R(delays)
        h_R_fft = np.fft.fft(h_R_kernel)
        I_fft = np.fft.fft(intensity)
        P_delayed = fR * grid.dt * np.real(np.fft.ifft(h_R_fft * I_fft))
    else:
        P_delayed = np.zeros_like(intensity)

    P_Raman = P_inst + P_delayed

    gamma = _gamma(fiber.n2, omega0, fiber.A_eff)
    raman_phase = np.exp(1j * gamma * P_Raman * dz)
    return A * raman_phase


def self_steepening_step(
    A: NDArray,
    fiber: "FiberProfile",
    grid: "TemporalGrid",
    dz: float,
    omega0: float,
    include_self_steepening: bool,
    P_NL: NDArray | None = None,
) -> NDArray:
    """Apply self-steepening via scipy solve_ivp with frequency-domain operator.

    The full GNLSE self-steepening is:
        ∂A/∂z = iγ(1 + i/ω₀·∂/∂T)[P_NL·A]

    In the frequency domain (ω = offset frequency, centered at 0):
        ∂Ã/∂z = iγ(1 - ω/ω₀)·FFT[P_NL·A]

    Parameters
    ----------
    A : complex array — pulse envelope.
    fiber : FiberProfile — fiber parameters.
    grid : TemporalGrid — time grid.
    dz : float — step size (m).
    omega0 : float — carrier frequency (rad/s).
    include_self_steepening : bool — whether to include self-steepening.
    P_NL : real array, optional — nonlinear polarization. If None, uses |A|².

    Returns
    -------
    A : updated complex array.
    """
    if not include_self_steepening:
        return A

    gamma = _gamma(fiber.n2, omega0, fiber.A_eff)

    if P_NL is None:
        P_NL = np.abs(A) ** 2

    omega_cut = 2 * omega0
    H = np.exp(-(grid.w / omega_cut) ** 8)
    P_NL_w = grid.fft(P_NL)
    P_NL_w *= H
    P_NL = grid.ifft(P_NL_w).real

    omega_ratio = grid.w / omega0
    factor = 1j * gamma * (1 - omega_ratio)

    def _rhs(z: float, A_flat: NDArray) -> NDArray:
        return grid.ifft(factor * grid.fft(P_NL * A_flat))

    from scipy.integrate import solve_ivp
    sol = solve_ivp(
        _rhs,
        t_span=(0.0, dz),
        y0=A.astype(complex),
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
        dense_output=False,
    )
    return sol.y[:, -1]


def tpa_step(
    A: NDArray,
    fiber: "FiberProfile",
    grid: "TemporalGrid",
    dz: float,
    include_tpa: bool,
    U: float = 0.0,
    omega0: float = 0.0,
) -> Tuple[NDArray, float]:
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

    Returns
    -------
    A : updated complex array.
    U_new : updated carrier density.
    """
    if not include_tpa:
        return A, U

    sigma = fiber.sigma_tpa
    tau_c = fiber.carrier_lifetime.as_s if fiber.carrier_lifetime is not None else 1e-9

    if sigma <= 0:
        return A, U

    # Use actual carrier frequency if provided, else default 1064 nm
    hbar = 1.0545718e-34  # J·s
    if omega0 > 0:
        hbar_omega = hbar * omega0
    else:
        omega0_placeholder = 2 * np.pi * C_MS / (1064e-9)  # ~1064 nm default
        hbar_omega = hbar * omega0_placeholder

    # Intensity |A|^2 (W/m² for proper TPA)
    I = np.abs(A) ** 2

    # Carrier density update (spatially-averaged, explicit Euler)
    U_avg = np.mean(I) * sigma / (2 * hbar_omega) * dz - U / tau_c * dz
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
        Dispersion coefficients [beta2, beta3, ...].
    include_raman : bool
        Include Raman. Default False.
    include_self_steepening : bool
        Include self-steepening. Default False.
    include_tpa : bool
        Include TPA. Default False.
    step_size : Length | None
        Fixed step size (m). If None, adaptive stepping is used.
    """

    def __init__(
        self,
        pulse: "Wave",
        fiber: "FiberProfile",
        betas: NDArray,
        include_raman: bool = False,
        include_self_steepening: bool = False,
        include_tpa: bool = False,
        step_size: Length | None = None,
    ):
        self.pulse = pulse
        self.fiber = fiber
        # get_betas() returns betas in ps^(k)/m with omega in rad/ps.
        # We keep them in native units and convert omega to rad/ps in _linear_step.
        self.betas = np.asarray(betas, dtype=float)
        self.include_raman = include_raman
        self.include_self_steepening = include_self_steepening
        self.include_tpa = include_tpa
        self.step_size = step_size

        # Set up temporal grid from pulse
        self.grid: TemporalGrid = pulse.grid
        self.omega0 = pulse.central_frequency
        self.A = np.array(pulse.envelope_field, dtype=complex)
        self.evolution: list["Wave"] = []
        self._z_positions: list[float] = [0.0]
        self._spectra_vs_z: Tuple[NDArray, NDArray] | None = None
        self._U = 0.0  # carrier density for TPA

    def _linear_step(self, A: NDArray, dz: float) -> NDArray:
        """Apply dispersion via FFT: A(ω) ← A(ω) · exp(Σ D_k(ω) · Δz).

        Also applies loss: multiply by exp(-α·Δz/2).
        Dispersion Taylor expansion starts at k=2 (β₂, β₃, ...).
        β₁ (group velocity) is not included — pulse stays in group-velocity frame.

        Note: get_betas() returns β_k in ps^k/m with ω in rad/ps. We convert
        the frequency axis to rad/ps here (matching get_betas() convention)
        rather than converting betas to SI, because each β_k has different
        power-of-ps units (ps²/m for β₂, ps³/m for β₃, etc.).
        """
        A_w = self.grid.fft(A)
        omega_ps = self.grid.w * 1e-12  # rad/s → rad/ps

        # Build real phase function: φ(Ω) = Σ_{k≥2} βₖ·Ωᵏ/k!
        # where Ω is in rad/ps and β_k in ps^k/m.
        # Propagation: A(ω) ← A(ω) · exp(i·φ(ω)·Δz) — pure phase shift (unitary).
        phi = np.zeros_like(omega_ps, dtype=float)
        for k, beta_k in enumerate(self.betas, start=2):
            phi += beta_k * omega_ps ** k / factorial(k)

        # Apply loss: exp(-α·Δz/2)
        alpha = self.fiber.alpha
        if alpha > 0:
            loss_factor = np.exp(-alpha * dz / 2)
            A_w = A_w * loss_factor

        A_w = A_w * np.exp(1j * phi * dz)
        return self.grid.ifft(A_w)

    def _nonlinear_step(self, A: NDArray, dz: float) -> Tuple[NDArray, float]:
        """Apply nonlinear effects.

        Without self-steepening: uses exponential step for exact Kerr+Raman.
        With self-steepening: uses scipy solve_ivp for the full GNLSE nonlinear
        term iγ(1+i/ω₀·∂/∂T)[P_NL·A].

        The total nonlinear polarization is:
          P_NL = (1-fR)|A|² + fR·(h_R ⊗ |A|²)
        """
        gamma = _gamma(self.fiber.n2, self.omega0, self.fiber.A_eff)
        intensity = np.abs(A) ** 2

        if self.include_raman and self.fiber.raman_response is not None:
            fR = self.fiber.raman_response.fR
            P_inst = (1.0 - fR) * intensity

            N = self.grid.N
            delays = np.arange(N) * self.grid.dt
            h_R_kernel = self.fiber.raman_response._h_R(delays)
            h_R_fft = np.fft.fft(h_R_kernel)
            I_fft = np.fft.fft(intensity)
            P_delayed = fR * self.grid.dt * np.real(np.fft.ifft(h_R_fft * I_fft))

            P_NL = P_inst + P_delayed
        elif self.include_raman and self.fiber.raman_response is None:
            raise ValueError(
                "include_raman=True but fiber.raman_response is None. "
                "Provide a RamanResponse or set include_raman=False."
            )
        else:
            P_NL = intensity

        if self.include_self_steepening:
            omega_cut = 2 * self.omega0
            H = np.exp(-(self.grid.w / omega_cut) ** 8)
            P_NL_w = self.grid.fft(P_NL)
            P_NL_w *= H
            P_NL = self.grid.ifft(P_NL_w).real

            omega_ratio = self.grid.w / self.omega0
            factor = 1j * gamma * (1 - omega_ratio)

            def _ss_rhs(z: float, A_flat: NDArray) -> NDArray:
                A_local = A_flat
                NL_A = P_NL * A_local
                return self.grid.ifft(factor * self.grid.fft(NL_A))

            from scipy.integrate import solve_ivp
            sol = solve_ivp(
                _ss_rhs,
                t_span=(0.0, dz),
                y0=A.astype(complex),
                method="RK45",
                rtol=1e-8,
                atol=1e-10,
                dense_output=False,
            )
            A = sol.y[:, -1]
        else:
            A = A * np.exp(1j * gamma * P_NL * dz)

        A, U_new = tpa_step(A, self.fiber, self.grid, dz, self.include_tpa, self._U, self.omega0)
        return A, U_new

    def _adaptive_step_size(self, A: NDArray) -> float:
        """Compute adaptive step size from local error estimates."""
        alpha = self.fiber.alpha
        dz_loss = 1.0 / max(alpha, 1e-10) * 0.01

        T0 = self.pulse.envelope.pulse_width.as_s
        if len(self.betas) > 0:
            beta2 = abs(self.betas[0]) * 1e-24  # ps²/m → s²/m for dispersion length
            dz_disp = T0**2 / max(beta2, 1e-30) * 0.01
        else:
            dz_disp = float("inf")

        gamma = self.fiber.n2 * self.omega0 / (299792458.0 * self.fiber.A_eff.as_m2) if self.fiber.A_eff.as_m2 > 0 else 1e10
        I_max = np.max(np.abs(A) ** 2) if np.any(A) else 0.0
        dz_nl = 1.0 / max(gamma * I_max, 1e-30) * 0.01

        return min(dz_loss, dz_disp, dz_nl)

    def propagate(self, num_steps: int) -> None:
        """Run split-step simulation for num_steps steps."""
        from photonics_helper.pulse import Envelope, Wave

        length = self.fiber.length.as_m
        dz_base = length / num_steps
        z = 0.0

        # Store initial pulse
        self.evolution.append(
            Wave(grid=self.grid, envelope=self.pulse.envelope, central_wavelength=self.pulse.central_wavelength)
        )

        while z < length:
            if self.step_size is not None:
                dz = min(self.step_size.as_m, length - z)
            else:
                dz = min(self._adaptive_step_size(self.A), dz_base, length - z)

            # Half linear step
            self.A = self._linear_step(self.A, dz / 2)
            # Full nonlinear step
            self.A, self._U = self._nonlinear_step(self.A, dz)
            # Half linear step
            self.A = self._linear_step(self.A, dz / 2)

            z += dz

            self._z_positions.append(z)

            # Store evolution — keep reference to original envelope but
            # override envelope_field with actual propagated field.
            wave = Wave(
                grid=self.grid,
                envelope=self.pulse.envelope,
                central_wavelength=self.pulse.central_wavelength,
            )
            wave._pulse_train_field = self.A.copy()
            self.evolution.append(wave)

        # Compute spectra vs z
        omega = self.grid.w
        spectra = np.zeros((len(self.evolution), len(omega)))
        for i, wave in enumerate(self.evolution):
            A_w = self.grid.fft(wave.envelope_field)
            spectra[i] = np.abs(A_w) ** 2
        self._spectra_vs_z = (omega, spectra)

    @property
    def z_array(self) -> NDArray:
        """Array of propagation distances (m) for each evolution entry."""
        return np.array(self._z_positions)

    @property
    def spectra_vs_z(self) -> Tuple[NDArray, NDArray]:
        """Tuple of (frequency array, spectra at each step)."""
        if self._spectra_vs_z is None:
            raise RuntimeError("Call propagate() first.")
        return self._spectra_vs_z


# ---------------------------------------------------------------------------
# GNLSESolver
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
        Dispersion coefficients [beta2, beta3, ...] from Dispersion.get_betas().
    include_raman : bool
        Include Raman scattering. Default True.
    include_self_steepening : bool
        Include self-steepening. Default False.
    include_tpa : bool
        Include two-photon absorption. Default False.
    """

    def __init__(
        self,
        pulse: "Wave",
        fiber: "FiberProfile",
        betas: NDArray,
        include_raman: bool = True,
        include_self_steepening: bool = False,
        include_tpa: bool = False,
    ):
        self.pulse = pulse
        self.fiber = fiber
        # Store betas in ps²/m — SplitStepEngine handles SI conversion.
        self.betas = np.asarray(betas, dtype=float)
        self.include_raman = include_raman
        self.include_self_steepening = include_self_steepening
        self.include_tpa = include_tpa
        self._evolution: list["Wave"] = []
        self._z_positions: NDArray | None = None
        self._spectra_vs_z: Tuple[NDArray, NDArray] | None = None

    def propagate(self, num_steps: int = 100) -> None:
        """Run split-step simulation for num_steps steps."""
        engine = SplitStepEngine(
            pulse=self.pulse,
            fiber=self.fiber,
            betas=self.betas,
            include_raman=self.include_raman,
            include_self_steepening=self.include_self_steepening,
            include_tpa=self.include_tpa,
        )
        engine.propagate(num_steps)
        self._evolution = engine.evolution
        self._z_positions = engine.z_array
        self._spectra_vs_z = engine.spectra_vs_z

    @property
    def evolution(self) -> list["Wave"]:
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
    def spectra_vs_z(self) -> Tuple[NDArray, NDArray]:
        """Tuple of (frequency array, spectra at each step)."""
        if self._spectra_vs_z is None:
            raise RuntimeError("Call propagate() first.")
        return self._spectra_vs_z


# ---------------------------------------------------------------------------
# Visualization utilities
# ---------------------------------------------------------------------------

def plot_waterfall(solver: "GNLSESolver", ax=None, dB: bool = True) -> "plt.Figure":
    """Pulse envelope waterfall plot (envelope vs propagation distance).

    Parameters
    ----------
    solver : GNLSESolver
        Solver with propagated results.
    ax : matplotlib Axes, optional
        Axis to plot on. Creates new figure if None.
    dB : bool
        Plot in dB scale. Default True.

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    z_steps = solver.z_array
    t = solver.pulse.grid.t * 1e12  # ps

    for i, wave in enumerate(solver.evolution):
        envelope = wave.envelope_field
        if dB:
            intensity_dB = 10 * np.log10(np.abs(envelope) ** 2 + 1e-30)
            offset = intensity_dB.max()
            envelope = intensity_dB - offset + i * 0.5
        else:
            envelope = np.abs(envelope)
            envelope = envelope / envelope.max() + i * 0.5

        ax.plot(t, envelope + z_steps[i] * 1e3, "b-", linewidth=0.5)

    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Propagation distance (mm)")
    ax.set_title("Pulse Evolution Waterfall Plot")
    return fig


def plot_spectrum_vs_distance(solver: "GNLSESolver", ax=None, dB: bool = True) -> "plt.Figure":
    """Spectrum vs propagation distance plot.

    Plots spectral evolution using wavelength (nm) to avoid the positive/negative
    frequency folding issue that occurs when using np.abs(omega).

    Parameters
    ----------
    solver : GNLSESolver
        Solver with propagated results.
    ax : matplotlib Axes, optional
        Axis to plot on. Creates new figure if None.
    dB : bool
        Plot in dB scale. Default True.

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    omega, spectra = solver.spectra_vs_z

    # Convert offset frequency to wavelength (nm)
    omega_abs = omega + solver.omega0
    wavelength_nm = 2 * np.pi * C_MS / omega_abs * 1e9
    sort_idx = np.argsort(wavelength_nm)
    wavelength_nm = wavelength_nm[sort_idx]
    spectra = spectra[:, sort_idx]

    z_steps = np.linspace(0, solver.fiber.length.as_m, spectra.shape[0])

    if dB:
        spectra_dB = 10 * np.log10(spectra + 1e-30)
        max_dB = spectra_dB.max()
        spectra_dB = spectra_dB - max_dB + 50
        ax.pcolormesh(wavelength_nm, z_steps * 1e3, spectra_dB, shading="auto", cmap="hot")
    else:
        ax.pcolormesh(wavelength_nm, z_steps * 1e3, spectra, shading="auto", cmap="hot")

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Propagation distance (mm)")
    ax.set_title("Spectrum vs Distance")
    return fig


def plot_intensity_metrics(solver: "GNLSESolver", ax=None) -> "plt.Figure":
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
        fig = ax.figure
        ax1, ax2 = ax.subplots(2, 1)

    z_steps = solver.z_array
    z_mm = z_steps * 1e3

    peak_power = np.array([w.peak_power() for w in solver.evolution])
    t = solver.pulse.grid.t
    pulse_width = np.array([
        np.sqrt(np.sum(t**2 * np.abs(w.envelope_field)**2) / np.sum(np.abs(w.envelope_field)**2))
        for w in solver.evolution
    ])

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
