"""Refractive index data with spline interpolation and Sellmeier models."""

from scipy.interpolate import BSpline
import warnings
from .base import PI, WavelengthArray

from typing import List, Self, Tuple
from numpy.typing import ArrayLike, NDArray
from functools import cached_property
from pydantic.dataclasses import dataclass
from pydantic import model_validator

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_splrep
from rich.traceback import install

install()


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

    @model_validator(mode="after")
    def _setup_splines(self) -> "RefractiveIndex":
        n_arr = np.asarray(self.n)
        k_arr = np.asarray(self.k)
        wl_arr = np.asarray(self.wl.as_um)

        if not (len(n_arr) == len(k_arr) == len(wl_arr)):
            raise ValueError(
                f"n, k and wl must have equal lengths, got "
                f"n={len(n_arr)}, k={len(k_arr)}, wl={len(wl_arr)}"
            )
        if len(wl_arr) < 4:
            raise ValueError(
                "cubic spline interpolation requires at least 4 wavelength "
                f"points, got {len(wl_arr)}"
            )
        if not np.all(np.diff(wl_arr) > 0):
            raise ValueError(
                "wl must be strictly increasing (sorted, no duplicate values)"
            )

        self._wl_min = float(wl_arr.min())
        self._wl_max = float(wl_arr.max())
        return self

    @cached_property
    def nk(self) -> NDArray:
        return self.n + 1j * self.k

    @cached_property
    def _n_spline(self) -> BSpline:
        try:
            return make_splrep(self.wl.as_um, self.n)
        except Exception as e:
            raise ValueError(f"failed to build n-spline: {e}") from e

    @cached_property
    def _k_spline(self) -> BSpline:
        try:
            return make_splrep(self.wl.as_um, self.k)
        except Exception as e:
            raise ValueError(f"failed to build k-spline: {e}") from e

    @classmethod
    def from_complex(cls, nk: ArrayLike, wl: WavelengthArray) -> Self:
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

    def dn_dlambda(self, wavelength: float) -> float:
        """Derivative dn/dλ at a scalar wavelength (μm). Returns value in μm⁻¹."""
        self._validate_range(wavelength)
        return self._dn_spline(wavelength).item()

    def group_index(self, wavelength: float) -> float:
        """Group index n_g = n - λ·(dn/dλ) at a scalar wavelength (μm). Dimensionless."""
        self._validate_range(wavelength)
        n_val = self.n_func(wavelength)
        return n_val - wavelength * self.dn_dlambda(wavelength)

    def group_index_array(self) -> NDArray:
        """Group index n_g across the full wavelength grid. Dimensionless."""
        wl_um = self.wl.as_um
        n_arr = self._n_spline(wl_um)
        dn_dl_arr = self._dn_spline(wl_um)
        return n_arr - wl_um * dn_dl_arr

    def group_velocity(self, wavelength: float) -> float:
        """Group velocity v_g = c / n_g at a scalar wavelength (μm). Returns m/s.

        Warns if n_g approaches zero (division-by-zero guard).
        """
        self._validate_range(wavelength)
        n_g = self.group_index(wavelength)
        if abs(n_g) < 1e-10:
            warnings.warn(
                f"group_index ≈ 0 at {wavelength} μm — group_velocity will diverge",
                stacklevel=2,
            )
        return self._C / n_g

    def group_velocity_array(self) -> NDArray:
        """Group velocity v_g across the full wavelength grid. Returns m/s."""
        n_g_arr = self.group_index_array()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return self._C / n_g_arr

    @cached_property
    def _dn_spline(self) -> BSpline:
        """Cached derivative of the n-spline."""
        return self._n_spline.derivative()

    _C = 2.99792458e8  # speed of light in vacuum (m/s)

    def plot(self, include_k: bool = True):
        """Plot n (and optionally k) versus wavelength."""
        plt.plot(self.wl.as_um, self.n, label="n")
        if include_k:
            plt.plot(self.wl.as_um, self.k, label="k")

        plt.xlabel("wavelength [μm]")
        plt.ylabel("n, k" if include_k else "n")
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
        """Construct from a Sellmeier equation: n² = A₀ + Σ Aᵢλ²/(λ² - Bᵢ).

        Note: ``B`` entries must be **squared** resonance wavelengths
        (Bᵢ = λ_res²) in μm², matching the convention in materials.db.
        """
        if len(A) != len(B):
            raise ValueError("Length of A and B should be same")
        else:
            wl = np.linspace(wl_from_to_in_um[0], wl_from_to_in_um[1], n_points)
            wls = WavelengthArray(wl, "um")
            wl_arr = np.array(wls.as_um)
            A_arr = np.array(A)
            B_arr = np.array(B)
            with np.errstate(divide="ignore", invalid="ignore"):
                terms = A_arr * wl_arr[:, None] ** 2 / (wl_arr[:, None] ** 2 - B_arr)
                n_squared = A0 + terms.sum(axis=1)
            invalid = ~np.isfinite(n_squared) | (n_squared < 0)
            if np.any(invalid):
                bad = wl_arr[invalid]
                raise ValueError(
                    "Sellmeier sum is negative/undefined for some wavelengths "
                    f"(λ² approaches a resonance pole Bᵢ); offending range: "
                    f"{bad.min():.4g}–{bad.max():.4g} μm. Restrict "
                    "wl_from_to_in_um to the valid transparency window."
                )
            n = np.sqrt(n_squared)
            k = np.zeros(len(wls.value))  # type: ignore[arg-type]

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
        """Construct from an alternative Sellmeier form: n² = A₀ + Σ Aᵢ/(λ² - Bᵢ²).

        Note: unlike :meth:`from_sellmeier`, ``B`` entries here are the
        **unsquared** resonance wavelengths in μm (they are squared internally).
        """
        if len(A) != len(B):
            raise ValueError("Length of A and B should be same")
        else:
            n = []
            wl = np.linspace(wl_from_to_in_um[0], wl_from_to_in_um[1], n_points)
            wls = WavelengthArray(wl, "um")
            for wl in wls.as_um:
                sum = 0.0
                for i in range(len(A)):
                    sum += A[i] / (wl**2 - B[i] ** 2)
                n.append(np.sqrt(A0 + sum))
            k = np.zeros(len(wls.value))  # type: ignore[arg-type]

        return cls(n=np.array(n), k=k, wl=wls)

    @classmethod
    def from_material_database(cls, material: str, n_points: int = 200) -> Self:
        """Build a tabulated RefractiveIndex from Sellmeier data in materials.db."""
        from .raman import RamanDatabase

        db = RamanDatabase()
        sellmeier = db.get_sellmeier(material)
        if sellmeier is None:
            raise ValueError(f"No Sellmeier data for {material} in materials.db")

        wl_range = (sellmeier["valid_from_um"], sellmeier["valid_to_um"])
        kwargs = dict(
            A0=sellmeier["a0"],
            A=sellmeier["coefficients"],
            B=sellmeier["wavelengths"],
            wl_from_to_in_um=wl_range,
            n_points=n_points,
        )
        form = sellmeier["form"]
        if form == "standard":
            return cls.from_sellmeier(**kwargs)
        elif form == "alt":
            return cls.from_alt_sellmeier(**kwargs)
        raise ValueError(
            f"Unknown Sellmeier form {form!r} for {material} in materials.db"
        )

    def propagation_loss(self) -> NDArray:
        """Compute propagation loss (dB/m) from the extinction coefficient k.

        Uses intensity attenuation α(λ) = 4πk(λ)/λ, then converts to dB/m:
            loss = 10·log₁₀(e)·α ≈ 4.343·α
        """
        k_arr = np.asarray(self.k)
        if np.any(k_arr < 0):
            bad = self.wl.as_um[k_arr < 0]
            raise ValueError(
                f"k must be non-negative for loss calculation; found k < 0 "
                f"at wavelengths {bad.min():.4g}–{bad.max():.4g} μm"
            )
        if not np.any(k_arr):
            warnings.warn(
                "RefractiveIndex has no imaginary index values (k = 0); "
                "computed propagation loss will be zero.",
                stacklevel=2,
            )
        alpha = 4 * PI * self.k / self.wl.as_m  # 1/m (intensity attenuation)
        return 10 * np.log10(np.e) * alpha  # dB/m
