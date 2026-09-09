"""Generalized nonlinear Schrödinger equation (GNLSE) solver.

Split-step Fourier solver for pulse propagation through nonlinear media,
supporting Kerr, Raman, self-steepening, and two-photon absorption effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

from numpy.typing import NDArray

if TYPE_CHECKING:
    from photonics_helper.pulse import Wave, TemporalGrid

__all__ = ["FiberProfile", "GNLSESolver", "SplitStepEngine", "TaperedGNLSESolver"]


@dataclass
class FiberProfile:
    """Optical fiber parameters for GNLSE propagation.

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
    A_eff: float
    length: float
    confinement_factor: float = 1.0
    sigma_tpa: float = 0.0
    carrier_lifetime: Optional[float] = None
    raman_response: Optional[object] = None

    @classmethod
    def from_gamma(
        cls,
        gamma: float,
        n2: float,
        omega0: float,
        alpha: float = ...,
        length: float = ...,
        confinement_factor: float = ...,
        sigma_tpa: float = ...,
        carrier_lifetime: Optional[float] = ...,
        raman_response: Optional[object] = ...,
    ) -> "FiberProfile":
        """Create a FiberProfile from a target nonlinear coefficient γ.

        Computes A_eff = n₂·ω₀·Γ / (c·γ) from the relation
        γ = n₂·ω₀·Γ / (c·A_eff).
        """
        ...


def _gamma(n2: float, omega0: float, A_eff: float, confinement_factor: float = ...) -> float:
    """Nonlinear coefficient γ = n₂·ω₀·Γ / (c·A_eff)."""
    ...


def kerr_step(
    A: NDArray,
    fiber: "FiberProfile",
    grid: "TemporalGrid",
    dz: float,
    omega0: float,
) -> NDArray:
    """Apply instantaneous Kerr effect: A ← A · exp(i·γ·|A|²·Δz)."""
    ...


def raman_step(
    A: NDArray,
    fiber: "FiberProfile",
    grid: "TemporalGrid",
    dz: float,
    include_raman: bool,
) -> NDArray:
    """Apply Raman convolution: P_Raman = n₂ · FFT⁻¹[Ṙ(ω) · FFT[|A|²]]."""
    ...


def self_steepening_step(
    A: NDArray,
    fiber: "FiberProfile",
    grid: "TemporalGrid",
    dz: float,
    omega0: float,
    include_self_steepening: bool,
) -> NDArray:
    """Apply self-steepening: A ← A · exp(-(i/ω₀)·∂/∂T(n₂·|A|²))·Δz."""
    ...


def tpa_step(
    A: NDArray,
    fiber: "FiberProfile",
    grid: "TemporalGrid",
    dz: float,
    include_tpa: bool,
    U: float = 0.0,
) -> Tuple[NDArray, float]:
    """Apply two-photon absorption with carrier dynamics."""
    ...


class SplitStepEngine:
    """Split-step Fourier engine for GNLSE propagation."""

    pulse: "Wave"
    fiber: "FiberProfile"
    betas: NDArray
    include_raman: bool
    include_self_steepening: bool
    include_tpa: bool
    step_size: Optional[float]
    grid: "TemporalGrid"
    omega0: float
    A: NDArray
    evolution: list["Wave"]
    _spectra_vs_z: Optional[Tuple[NDArray, NDArray]]
    _U: float

    def __init__(
        self,
        pulse: "Wave",
        fiber: "FiberProfile",
        betas: NDArray,
        include_raman: bool = False,
        include_self_steepening: bool = False,
        include_tpa: bool = False,
        step_size: Optional[float] = None,
        dispersion_profile: Optional[Any] = None,
        a_eff_fn: Optional[Callable[[float], float]] = None,
        alpha_fn: Optional[Callable[[float], float]] = None,
        gamma_fn: Optional[Callable[[float], float]] = None,
    ) -> None: ...

    def _linear_step(self, A: NDArray, dz: float) -> NDArray: ...
    def _nonlinear_step(self, A: NDArray, dz: float) -> Tuple[NDArray, float]: ...
    def _adaptive_step_size(self, A: NDArray) -> float: ...
    def propagate(
        self, num_steps: int, *, nsaves: int | None = None, show_progress: bool = False
    ) -> None: ...

    @property
    def spectra_vs_z(self) -> Tuple[NDArray, NDArray]: ...


class GNLSESolver:
    """High-level GNLSE solver using split-step Fourier method."""

    pulse: "Wave"
    fiber: "FiberProfile"
    betas: NDArray
    include_raman: bool
    include_self_steepening: bool
    include_tpa: bool
    _evolution: list["Wave"]
    _spectra_vs_z: Optional[Tuple[NDArray, NDArray]]

    def __init__(
        self,
        pulse: "Wave",
        fiber: "FiberProfile",
        betas: NDArray,
        include_raman: bool = True,
        include_self_steepening: bool = False,
        include_tpa: bool = False,
    ) -> None: ...

    def propagate(
        self, num_steps: int = 100, *, nsaves: int | None = None, show_progress: bool = False
    ) -> None: ...

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
    ) -> int: ...

    def interpolated_spectrum_db(
        self,
        wl_min: float,
        wl_max: float,
        n_wl: int,
        *,
        step_index: int = -1,
    ) -> Tuple[NDArray, NDArray]: ...

    @property
    def evolution(self) -> list["Wave"]: ...

    @property
    def spectra_vs_z(self) -> Tuple[NDArray, NDArray]: ...


class TaperedGNLSESolver:
    """High-level GNLSE solver for z-dependent (tapered/dispersion-managed) waveguides."""

    pulse: "Wave"
    fiber: "FiberProfile"
    dispersion_profile: object
    a_eff_fn: Optional[Callable[[float], float]]
    alpha_fn: Optional[Callable[[float], float]]
    include_raman: bool
    include_self_steepening: bool
    include_tpa: bool
    _evolution: list["Wave"]
    _z_positions: Optional[NDArray]
    _spectra_vs_z: Optional[Tuple[NDArray, NDArray]]

    def __init__(
        self,
        pulse: "Wave",
        fiber: "FiberProfile",
        dispersion_profile: object,
        a_eff_fn: Optional[Callable[[float], float]] = None,
        alpha_fn: Optional[Callable[[float], float]] = None,
        include_raman: bool = True,
        include_self_steepening: bool = False,
        include_tpa: bool = False,
    ) -> None: ...

    def propagate(
        self,
        num_steps: int = 100,
        strict: bool = False,
        *,
        nsaves: int | None = None,
        show_progress: bool = False,
    ) -> None: ...

    @property
    def evolution(self) -> list["Wave"]: ...

    @property
    def z_array(self) -> NDArray: ...

    @property
    def omega0(self) -> float: ...

    @property
    def spectra_vs_z(self) -> Tuple[NDArray, NDArray]: ...


def plot_waterfall(
    solver: "GNLSESolver",
    ax: Optional["plt.Axes"] = None,
    dB: bool = True,
) -> "plt.Figure": ...


def plot_spectral_evolution(
    solver: "GNLSESolver",
    ax: Optional["plt.Axes"] = None,
    *,
    wl_min: Optional[float] = None,
    wl_max: Optional[float] = None,
    n_wl: int = 400,
    dynamic_range_db: float = 40.0,
    dB: bool = True,
    cmap: str = "jet",
    z_scale: str = "m",
) -> "plt.Figure": ...


def plot_temporal_evolution(
    solver: "GNLSESolver",
    ax: Optional["plt.Axes"] = None,
    *,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
    dynamic_range_db: float = 40.0,
    cmap: str = "jet",
    z_scale: str = "m",
) -> "plt.Figure": ...


def spectral_evolution_on_wavelength_grid(
    solver: "GNLSESolver",
    wl_min: float,
    wl_max: float,
    n_wl: int,
) -> Tuple[NDArray, NDArray, NDArray]: ...


def temporal_evolution_intensity(
    solver: "GNLSESolver",
) -> Tuple[NDArray, NDArray, NDArray]: ...


def plot_intensity_metrics(
    solver: "GNLSESolver",
    ax: Optional["plt.Axes"] = None,
) -> "plt.Figure": ...
