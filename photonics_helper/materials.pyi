from __future__ import annotations

from numpy.typing import NDArray
from typing import List, Self, Tuple

from .base import WavelengthArray

class RefractiveIndex:
    """Represents refractive index data with real (n) and imaginary (k) components."""

    def __init__(self, n: NDArray, k: NDArray, wl: WavelengthArray) -> None:
        """Initialize refractive index data.

        Args:
            n: Array of real refractive index values
            k: Array of extinction coefficient values
            wl: Array of wavelengths at which n,k are defined
        """
        ...

    @property
    def n(self) -> NDArray:
        """Real part of the refractive index."""
        ...

    @property
    def k(self) -> NDArray:
        """Imaginary part (extinction coefficient) of the refractive index."""
        ...

    @property
    def wl(self) -> WavelengthArray:
        """Wavelength points at which n,k are defined."""
        ...

    @property
    def nk(self) -> NDArray:
        """Complex refractive index (n + ik)."""
        ...

    @classmethod
    def from_complex(cls, nk: NDArray, wl: WavelengthArray) -> Self:
        """Create RefractiveIndex from complex values.

        Args:
            nk: Complex refractive index values (n + ik)
            wl: Wavelength points

        Returns:
            New RefractiveIndex instance
        """
        ...

    def n_func(self, wavelength: float) -> float:
        """Interpolate real refractive index at a specific wavelength.

        Args:
            wavelength: Target wavelength

        Returns:
            Interpolated n value
        """
        ...

    def k_func(self, wavelength: float) -> float:
        """Interpolate extinction coefficient at a specific wavelength.

        Args:
            wavelength: Target wavelength

        Returns:
            Interpolated k value
        """
        ...

    def nk_func(self, wavelength: float) -> complex:
        """Interpolate complex refractive index at a specific wavelength.

        Args:
            wavelength: Target wavelength

        Returns:
            Interpolated complex n+ik value
        """
        ...

    def plot(self, include_k: bool = True) -> None:
        """Plot refractive index data.

        Args:
            include_k: Whether to include extinction coefficient plot
        """
        ...

    @classmethod
    def from_sellmeier(
        cls,
        A0: int | float,
        A: List[float],
        B: List[float],
        wl_from_to_in_m: Tuple[float, float],
        n_points: int = 200,
    ) -> Self:
        """Create RefractiveIndex using Sellmeier equation.

        Args:
            A0: Offset coefficient
            A: List of amplitude coefficients
            B: List of wavelength coefficients
            wl_from_to_in_m: Tuple of (min, max) wavelength in meters
            n_points: Number of points to generate

        Returns:
            New RefractiveIndex instance
        """
        ...

    @classmethod
    def from_alt_sellmeier(
        cls,
        A0: int | float,
        A: List[float],
        B: List[float],
        wl_from_to_in_m: Tuple[float, float],
        n_points: int = 200,
    ) -> Self:
        """Create RefractiveIndex using alternative Sellmeier equation.

        Args:
            A0: Offset coefficient
            A: List of amplitude coefficients
            B: List of wavelength coefficients
            wl_from_to_in_m: Tuple of (min, max) wavelength in meters
            n_points: Number of points to generate

        Returns:
            New RefractiveIndex instance
        """
        ...
