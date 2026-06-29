"""Refractive index data with spline interpolation and Sellmeier models."""

import warnings
from .base import PI, WavelengthArray

from typing import List, Self, Tuple
from numpy.typing import NDArray
from functools import cached_property

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_splrep
from rich.traceback import install

install()


class RefractiveIndex:
    """Wavelength-dependent complex refractive index (n + ik).

    Stores tabulated n and k values and interpolates via cubic splines.

    Attributes
    ----------
    n : real part of refractive index.
    k : imaginary part (extinction coefficient).
    wl : wavelength array (um).
    """

    def __init__(self, n: NDArray, k: NDArray, wl: WavelengthArray) -> None:
        self._n = n
        self._k = k
        self._wl = wl

        self._wl_min = float(self._wl.as_um.min())
        self._wl_max = float(self._wl.as_um.max())
        self._n_spline = make_splrep(self._wl.as_um, self._n)
        self._k_spline = make_splrep(self._wl.as_um, self._k)

    @cached_property
    def n(self) -> NDArray:
        return self._n

    @cached_property
    def k(self) -> NDArray:
        return self._k

    @cached_property
    def wl(self) -> WavelengthArray:
        return self._wl

    @cached_property
    def nk(self) -> NDArray:
        return self._n + 1j * self._k

    @classmethod
    def from_complex(cls, nk: NDArray, wl: WavelengthArray) -> Self:
        """Construct from a complex refractive index array."""
        return cls(n=np.real(nk), k=np.imag(nk), wl=wl)

    def _validate_range(self, wavelength: float):
        if not (self._wl_min <= wavelength <= self._wl_max):
            raise ValueError(
                f"Index valid only between {self._wl_min} μm and {self._wl_max} μm"
            )

    def n_func(self, wavelength: float) -> float:
        """Interpolated real refractive index n at wavelength (μm)."""
        self._validate_range(wavelength)
        return self._n_spline(wavelength).item()

    def k_func(self, wavelength: float) -> float:
        """Interpolated extinction coefficient k at wavelength (μm)."""
        self._validate_range(wavelength)
        return self._k_spline(wavelength).item()

    def nk_func(self, wavelength: float) -> complex:
        """Interpolated complex refractive index n+ik at wavelength (μm)."""
        self._validate_range(wavelength)
        return complex(
            self._n_spline(wavelength).item(), self._k_spline(wavelength).item()
        )

    def plot(self, include_k: bool = True):
        """Plot n (and optionally k) versus wavelength."""

        plt.plot(self._wl.as_um, self.n, label="n")
        plt.xlabel("wavelength [m]")
        plt.ylabel("n")

        if include_k:
            plt.plot(self._wl.as_um, self._k, label="k")
            plt.ylabel("n,k")
            plt.legend()

        plt.show()

    @classmethod
    def from_sellmeier(
        cls,
        A0: int | float,
        A: List[float],
        B: List[float],
        wl_from_to_in_um: Tuple[float, float],
        n_points=200,
    ) -> Self:
        """Construct from a Sellmeier equation: n² = A₀ + Σ Aᵢλ²/(λ² - Bᵢ)."""
        if len(A) != len(B):
            raise ValueError("Length of A and B should be same")
        else:
            n = []
            wl = np.linspace(wl_from_to_in_um[0], wl_from_to_in_um[1], n_points)
            wls = WavelengthArray(wl, "um")
            for wl in wls.as_um:
                sum = 0.0
                for i in range(len(A)):
                    sum += A[i] * wl**2 / (wl**2 - B[i])
                n.append(np.sqrt(A0 + sum))
            k = np.zeros(len(wls.value))

        return cls(n=np.array(n), k=k, wl=wls)

    @classmethod
    def from_alt_sellmeier(
        cls,
        A0: int | float,
        A: List[float],
        B: List[float],
        wl_from_to_in_um: Tuple[float, float],
        n_points=200,
    ) -> Self:
        """Construct from an alternative Sellmeier form: n² = A₀ + Σ Aᵢ/(λ² - Bᵢ²)."""
        if len(A) != len(B):
            raise ValueError("Length of A and B should be same")
        else:
            n = []
            wl = np.linspace(wl_from_to_in_um[0], wl_from_to_in_um[1], n_points)
            wls = WavelengthArray(wl, "m")
            for wl in wls.as_um:
                sum = 0.0
                for i in range(len(A)):
                    sum += A[i] / (wl**2 - B[i] ** 2)
                n.append(np.sqrt(A0 + sum))
            k = np.zeros(len(wls.value))

        return cls(n=np.array(n), k=k, wl=wls)

    def propagation_loss(self):
        """Compute propagation loss (dB/m) from the extinction coefficient k."""
        if not np.all(self.k):
            warnings.warn(
                "RefractiveIndex doesn't have imaginary index values. please provide it before loss calculation"
            )
        else:
            return -20 * np.log10(np.exp(-2 * PI * self.k / self.wl.as_m))
