from __future__ import annotations

from numpy.typing import NDArray
from typing import List, Optional, Self, Tuple
from functools import cached_property
from pydantic.dataclasses import dataclass

from .base import WavelengthArray

@dataclass(config={"arbitrary_types_allowed": True})
class RefractiveIndex:
    """Wavelength-dependent complex refractive index (n + ik).

    Stores tabulated n and k values and interpolates via cubic splines.

    Attributes
    ----------
    n : real part of refractive index.
    k : imaginary part (extinction coefficient).
    wl : wavelength array (um).
    """

    n: NDArray
    k: NDArray
    wl: WavelengthArray

    @cached_property
    def n(self) -> NDArray:
        """Real part of the refractive index."""
        ...

    @cached_property
    def k(self) -> NDArray:
        """Imaginary part (extinction coefficient) of the refractive index."""
        ...

    @cached_property
    def wl(self) -> WavelengthArray:
        """Wavelength points at which n,k are defined."""
        ...

    @cached_property
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

        Raises:
            AttributeError: If wavelength is outside valid range
        """
        ...

    def k_func(self, wavelength: float) -> float:
        """Interpolate extinction coefficient at a specific wavelength.

        Args:
            wavelength: Target wavelength

        Returns:
            Interpolated k value

        Raises:
            AttributeError: If wavelength is outside valid range
        """
        ...

    def nk_func(self, wavelength: float) -> float:
        """Interpolate complex refractive index at a specific wavelength.

        Args:
            wavelength: Target wavelength

        Returns:
            Interpolated complex n+ik value

        Raises:
            AttributeError: If wavelength is outside valid range
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
        wl_from_to_in_um: Tuple[float, float],
        n_points: int = 200,
    ) -> Self:
        """Create RefractiveIndex using Sellmeier equation.

        Args:
            A0: Offset coefficient
            A: List of amplitude coefficients
            B: List of wavelength coefficients
            wl_from_to_in_um: Tuple of (min, max) wavelength in micrometers
            n_points: Number of points to generate

        Returns:
            New RefractiveIndex instance

        Raises:
            ValueError: If A and B lists have different lengths
        """
        ...

    @classmethod
    def from_alt_sellmeier(
        cls,
        A0: int | float,
        A: List[float],
        B: List[float],
        wl_from_to_in_um: Tuple[float, float],
        n_points: int = 200,
    ) -> Self:
        """Create RefractiveIndex using alternative Sellmeier equation.

        Args:
            A0: Offset coefficient
            A: List of amplitude coefficients
            B: List of wavelength coefficients
            wl_from_to_in_um: Tuple of (min, max) wavelength in micrometers
            n_points: Number of points to generate

        Returns:
            New RefractiveIndex instance

        Raises:
            ValueError: If A and B lists have different lengths
        """
        ...

    def propagation_loss(self):
        """Compute propagation loss (dB/m) from the extinction coefficient k."""
        ...
