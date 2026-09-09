from __future__ import annotations

from numpy.typing import NDArray
from typing import Any, List, Literal, Self, Tuple
from functools import cached_property
from scipy.interpolate import BSpline
from pydantic.dataclasses import dataclass

from .base import WavelengthArray

# Canonical material names stored in materials.db (see materials.py).
NK_MATERIALS: tuple[str, ...]
# A material name known to the database.
NKMaterial = Literal[...]


def validate_nk_dataset(entry: Any) -> list[str]:
    """Validate a tabulated n/k dataset record; returns a list of error strings
    (empty when the dataset is valid)."""
    ...


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

    @cached_property
    def _n_spline(self) -> BSpline:
        ...

    @cached_property
    def _k_spline(self) -> BSpline:
        ...

    _wl_min: float
    _wl_max: float

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

    def nk_func(self, wavelength: float) -> complex:
        """Interpolate complex refractive index at a specific wavelength.

        Args:
            wavelength: Target wavelength

        Returns:
            Interpolated complex n+ik value

        Raises:
            ValueError: If wavelength is outside valid range
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

    def propagation_loss(self) -> NDArray:
        """Compute propagation loss (dB/m) from the extinction coefficient k."""
        ...
