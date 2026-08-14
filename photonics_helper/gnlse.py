"""Generalized nonlinear Schrödinger equation (GNLSE) solver.

Split-step Fourier solver for pulse propagation through nonlinear media,
supporting Kerr, Raman, self-steepening, and two-photon absorption effects.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from math import factorial
from typing import TYPE_CHECKING, Optional, Tuple, Callable, Union

import numpy as np
from numpy.typing import NDArray

from photonics_helper.base import C_MS, Area, Length, Time

if TYPE_CHECKING:
    from photonics_helper.pulse import Wave, TemporalGrid
    from photonics_helper.fiber import ZDependentDispersion

__all__ = [
    "FiberProfile",
    "GNLSESolver",
    "SplitStepEngine",
    "TaperedGNLSESolver",
    "spectral_evolution_on_wavelength_grid",
    "spectral_evolution_on_frequency_grid",
    "temporal_evolution_intensity",
    "plot_waterfall",
    "plot_spectrum_vs_distance",
    "plot_spectral_evolution",
    "plot_temporal_evolution",
    "plot_intensity_metrics",
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
    carrier_lifetime: Optional[Time] = None
    raman_response: Optional[object] = None

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
        carrier_lifetime: Optional[Time] = None,
        raman_response: Optional[object] = None,
    ) -> "FiberProfile":
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

def _gamma(n2: float, omega0: float, A_eff: Area, confinement_factor: float = 1.0) -> float:
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
    gamma = _gamma(fiber.n2, omega0, fiber.A_eff, fiber.confinement_factor)
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

    gamma = _gamma(fiber.n2, omega0, fiber.A_eff, fiber.confinement_factor)
    raman_phase = np.exp(1j * gamma * P_Raman * dz)
    return A * raman_phase


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
        dispersion_profile: "ZDependentDispersion | Callable[[float, float], NDArray] | None" = None,
        a_eff_fn: Callable[[float], float] | None = None,
        alpha_fn: Callable[[float], float] | None = None,
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

        # z-dependent hooks (all optional; None → uniform behavior)
        self._dispersion_profile = dispersion_profile
        self._a_eff_fn = a_eff_fn
        self._alpha_fn = alpha_fn
        self._is_z_dependent = dispersion_profile is not None

        # Set up temporal grid from pulse
        self.grid: TemporalGrid = pulse.grid
        self.omega0 = pulse.central_frequency
        self.A = np.array(pulse.envelope_field, dtype=complex)
        self.evolution: list["Wave"] = []
        self._z_positions: list[float] = [0.0]
        self._current_z: float = 0.0
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

        if self._is_z_dependent:
            # z-dependent path: compute phase from β(ω₀+Ω, z) - β(ω₀, z)
            # at the current propagation position.
            # grid.w is the offset Ω (centered at 0), so absolute frequency is ω₀ + grid.w.
            omega_abs = self.omega0 + self.grid.w  # absolute angular frequency (rad/s)
            # Clip to dispersion profile's valid omega range to avoid NaN from extrapolation
            if hasattr(self._dispersion_profile, 'omegas'):
                omega_min, omega_max = float(min(self._dispersion_profile.omegas)), float(max(self._dispersion_profile.omegas))
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
            # Uniform path: Taylor expansion (unchanged behavior)
            # Build real phase function: φ(Ω) = Σ_{k≥2} βₖ·Ωᵏ/k!
            # where Ω is in rad/ps and β_k in ps^k/m.
            # Propagation: A(ω) ← A(ω) · exp(i·φ(ω)·Δz) — pure phase shift (unitary).
            phi = np.zeros_like(omega_ps, dtype=float)
            for k, beta_k in enumerate(self.betas, start=2):
                phi += beta_k * omega_ps ** k / factorial(k)
            phi *= dz

            # Apply loss: exp(-α·Δz/2)
            alpha = self.fiber.alpha
            if alpha > 0:
                loss_factor = np.exp(-alpha * dz / 2)
                A_w = A_w * loss_factor

        A_w = A_w * np.exp(1j * phi)
        return self.grid.ifft(A_w)

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
            return self._dispersion_profile(omega, z)
        else:
            # ZDependentDispersion case
            return self._dispersion_profile.fn(omega, z)

    def _get_gamma(self, z: float) -> float:
        """Compute γ(z) from a_eff_fn(z) or fallback to fiber.A_eff."""
        if self._a_eff_fn is not None:
            a_eff = self._a_eff_fn(z)
            return _gamma(self.fiber.n2, self.omega0, Area(a_eff, "m^2"), self.fiber.confinement_factor)
        else:
            return _gamma(
                self.fiber.n2, self.omega0, self.fiber.A_eff, self.fiber.confinement_factor
            )

    def _get_alpha(self, z: float) -> float:
        """Get α(z) from alpha_fn(z) or fallback to fiber.alpha."""
        if self._alpha_fn is not None:
            return self._alpha_fn(z)
        else:
            return self.fiber.alpha

    def _nonlinear_step(self, A: NDArray, dz: float) -> Tuple[NDArray, float]:
        """Apply nonlinear effects.

        Without self-steepening: uses exponential step for exact Kerr+Raman.
        With self-steepening: uses direct analytical FFT-based computation of
        the linear self-steepening operator.

        The total nonlinear polarization is:
          P_NL = (1-fR)|A|² + fR·(h_R ⊗ |A|²)
        """
        gamma = self._get_gamma(self._current_z)
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
            # Self-steepening: RK4 in frequency domain (Dudley / laserfun form).
            # dÃ/dz = iγ · (1 - Ω/ω₀) · FFT(NL_src) with NL_src = P_NL·A (Raman)
            # or |A|²·A (Kerr). Field-dependent NL_src at each RK4 stage avoids
            # energy drift when combined with the dispersion split-step.
            shock_factor = 1.0 - self.grid.w / self.omega0
            A_w = self.grid.fft(A)
            use_frozen_pnl = self.include_raman and self.fiber.raman_response is not None

            def shock_rhs_freq(A_w_stage: NDArray) -> NDArray:
                a = self.grid.ifft(A_w_stage)
                if use_frozen_pnl:
                    nl_src = P_NL * a
                else:
                    nl_src = np.abs(a) ** 2 * a
                nl_w = self.grid.fft(nl_src)
                nl_w *= shock_factor
                return 1j * gamma * nl_w

            k1 = shock_rhs_freq(A_w)
            k2 = shock_rhs_freq(A_w + 0.5 * dz * k1)
            k3 = shock_rhs_freq(A_w + 0.5 * dz * k2)
            k4 = shock_rhs_freq(A_w + dz * k3)
            A = self.grid.ifft(A_w + (dz / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))
        else:
            A = A * np.exp(1j * gamma * P_NL * dz)

        A, U_new = tpa_step(A, self.fiber, self.grid, dz, self.include_tpa, self._U, self.omega0)
        return A, U_new

    def _get_omega_bounds(self) -> tuple[float, float]:
        """Return (omega_min, omega_max) from the dispersion profile, or a wide default."""
        if self._dispersion_profile is not None and hasattr(self._dispersion_profile, 'omegas'):
            return float(min(self._dispersion_profile.omegas)), float(max(self._dispersion_profile.omegas))
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
        I_max = np.max(np.abs(A) ** 2) if np.any(A) else 0.0
        if gamma > 0.0 and I_max > 0.0:
            dz_nl = 1.0 / (gamma * I_max)
            # Cap nonlinear phase per step: γ P₀ Δz < π/4 (stricter with steepening).
            omega_factor = 1.0
            if self.include_self_steepening:
                omega_factor = 1.0 + self.grid.omega_max / self.omega0
            dz_nl = min(dz_nl, np.pi / (4.0 * gamma * I_max * omega_factor))
        else:
            dz_nl = float("inf")

        return min(dz_loss, dz_disp, dz_nl)

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
            return max(0.1, min(factor, 1.0))
        else:
            return 1.0

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
            # Determine step size
            if self.step_size is not None:
                dz = min(self.step_size.as_m, length - z)
            else:
                dz_adaptive = self._adaptive_step_size(self.A)
                dz = min(dz_adaptive, dz_base, length - z)
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
            # Half linear step
            self.A = self._linear_step(self.A, dz / 2)

            z += dz
            self._current_z = z

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

    @classmethod
    def estimate_num_steps(
        cls,
        pulse: "Wave",
        fiber: "FiberProfile",
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
    ) -> Tuple[NDArray, NDArray]:
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
        Include self-steepening. Default False.
    include_tpa : bool
        Include two-photon absorption. Default False.
    """

    def __init__(
        self,
        pulse: "Wave",
        fiber: "FiberProfile",
        dispersion_profile: "ZDependentDispersion | Callable[[float, float], NDArray]",
        a_eff_fn: Callable[[float], float] | None = None,
        alpha_fn: Callable[[float], float] | None = None,
        include_raman: bool = True,
        include_self_steepening: bool = False,
        include_tpa: bool = False,
    ):
        self.pulse = pulse
        self.fiber = fiber
        self.dispersion_profile = dispersion_profile
        self.a_eff_fn = a_eff_fn
        self.alpha_fn = alpha_fn
        self.include_raman = include_raman
        self.include_self_steepening = include_self_steepening
        self.include_tpa = include_tpa
        self._engine: SplitStepEngine | None = None
        self._evolution: list["Wave"] = []
        self._z_positions: NDArray | None = None
        self._spectra_vs_z: Tuple[NDArray, NDArray] | None = None

    def propagate(self, num_steps: int = 100) -> None:
        """Run split-step simulation for num_steps steps."""
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
        )
        self._engine.propagate(num_steps)
        self._evolution = self._engine.evolution
        self._z_positions = self._engine.z_array
        self._spectra_vs_z = self._engine.spectra_vs_z

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
# Evolution arrays (wavelength / time grids for contour plots)
# ---------------------------------------------------------------------------

def spectral_evolution_on_wavelength_grid(
    solver: "GNLSESolver",
    wl_min: float,
    wl_max: float,
    n_wl: int,
) -> Tuple[NDArray, NDArray, NDArray]:
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
    solver: "GNLSESolver",
    f_min_THz: float,
    f_max_THz: float,
    n_f: int,
) -> Tuple[NDArray, NDArray, NDArray]:
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
        interpolated[i] = np.interp(
            f_grid, f_THz, spectra[i], left=0.0, right=0.0
        )

    global_max = interpolated.max()
    spectra_dB = 10 * np.log10(interpolated / global_max + 1e-30)
    return solver.z_array, f_grid, spectra_dB


def temporal_evolution_intensity(solver: "GNLSESolver") -> Tuple[NDArray, NDArray, NDArray]:
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
    """Spectrum vs propagation distance contour (legacy wrapper).

    See :func:`plot_spectral_evolution` for wavelength range and dynamic-range
    control. Uses millimetres on the distance axis and the ``hot`` colormap
    for backward compatibility.
    """
    return plot_spectral_evolution(solver, ax=ax, dB=dB, z_scale="mm", cmap="hot")


def plot_spectral_evolution(
    solver: "GNLSESolver",
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
) -> "plt.Figure":
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
    z_label = "Propagation distance (mm)" if z_scale == "mm" else "Propagation distance (m)"

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
        ax.pcolormesh(x_axis, z_plot, data, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xlabel(x_label)
    ax.set_ylabel(z_label)
    ax.set_title("Spectral evolution")
    return fig


def plot_temporal_evolution(
    solver: "GNLSESolver",
    ax=None,
    *,
    t_min: float | None = None,
    t_max: float | None = None,
    dynamic_range_db: float = 40.0,
    cmap: str = "viridis",
    z_scale: str = "m",
    use_imshow: bool = True,
) -> "plt.Figure":
    """Temporal evolution contour: time vs propagation distance.

    Parameters
    ----------
    solver : GNLSESolver
        Solver with propagated results.
    ax : matplotlib Axes, optional
        Axis to plot on. Creates new figure if None.
    t_min, t_max : float, optional
        Time window in ps. Default: full grid.
    dynamic_range_db : float
        Color scale spans [global_max - dynamic_range_db, global_max] in dB.
    cmap : str
        Matplotlib colormap name.
    z_scale : str
        ``"m"`` or ``"mm"`` for the distance axis.

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
    global_max = intensity.max()
    intensity_dB = 10 * np.log10(intensity / global_max + 1e-30)

    if t_min is None:
        t_min = float(t_ps[0])
    if t_max is None:
        t_max = float(t_ps[-1])

    z_plot = z_m * 1e3 if z_scale == "mm" else z_m
    z_label = "Propagation distance (mm)" if z_scale == "mm" else "Propagation distance (m)"

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
