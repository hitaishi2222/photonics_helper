"""Generalized nonlinear Schrödinger equation (GNLSE) solver.

Split-step Fourier solver for pulse propagation through nonlinear media,
supporting Kerr, Raman, self-steepening, and two-photon absorption effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from photonics_helper.pulse import Wave, TemporalGrid

__all__ = ["FiberProfile", "GNLSESolver", "SplitStepEngine"]


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
    A_eff: float
    length: float
    sigma_tpa: float = 0.0
    carrier_lifetime: Optional[float] = None
    raman_response: Optional[object] = None


def _gamma(n2: float, omega0: float, A_eff: float) -> float:
    """Nonlinear coefficient γ = n₂·ω₀ / (c·A_eff)."""
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
    ) -> None: ...

    def _linear_step(self, A: NDArray, dz: float) -> NDArray: ...
    def _nonlinear_step(self, A: NDArray, dz: float) -> Tuple[NDArray, float]: ...
    def _adaptive_step_size(self, A: NDArray) -> float: ...
    def propagate(self, num_steps: int) -> None: ...

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

    def propagate(self, num_steps: int = 100) -> None: ...

    @property
    def evolution(self) -> list["Wave"]: ...

    @property
    def spectra_vs_z(self) -> Tuple[NDArray, NDArray]: ...


def plot_waterfall(
    solver: "GNLSESolver",
    ax: Optional["plt.Axes"] = None,
    dB: bool = True,
) -> "plt.Figure": ...


def plot_spectrum_vs_distance(
    solver: "GNLSESolver",
    ax: Optional["plt.Axes"] = None,
    dB: bool = True,
) -> "plt.Figure": ...


def plot_intensity_metrics(
    solver: "GNLSESolver",
    ax: Optional["plt.Axes"] = None,
) -> "plt.Figure": ...
