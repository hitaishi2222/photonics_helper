"""Fiber dispersion and propagation constant calculations."""

from photonics_helper.base import (
    C_MS,
    PI,
    AngularFrequencyArray,
    Wavelength,
    WavelengthArray,
)
from photonics_helper.looks import c_info

from functools import cached_property
from numpy.typing import NDArray
from typing import Literal, Self
from pydantic.dataclasses import dataclass
from pydantic import model_validator, Field

import warnings
import numpy as np
import matplotlib.pyplot as plt
from math import factorial
from scipy.interpolate import make_splrep
from rich.traceback import install

install()


@dataclass(config={"arbitrary_types_allowed": True})
class Dispersion:
    """Wavelength-dependent dispersion D(λ) with spline interpolation.

    Values are stored internally in s/m² and can be queried in ps/(nm·km).

    Attributes
    ----------
    wavelengths : wavelength array.
    values : dispersion values in s/m².
    central_wavelength : design central wavelength.
    """

    central_wavelength: Wavelength
    _wavelengths: WavelengthArray = Field(alias="wavelengths")
    _values: NDArray = Field(alias="values")
    _unit: Literal["ps/nm.km", "s/m^2"] = Field(default="s/m^2", alias="unit")

    @model_validator(mode="after")
    def _setup(self) -> "Dispersion":
        # unit conversion handled externally; _unit is always "s/m^2" internally
        return self

    def __repr__(self):
        return f"Dispersion: from wl: {self._wavelengths.as_m.min()} to {self._wavelengths.as_m.max()}"

    def __str__(self):
        return f"Dispersion D(λ) in ps/(nm·km), range {self._wavelengths.as_nm.min():.0f}–{self._wavelengths.as_nm.max():.0f} nm"

    @cached_property
    def as_ps_nm_km(self) -> NDArray:
        """Dispersion values in ps/(nm·km)."""
        return self._values * 1e6

    @cached_property
    def as_s_m_m(self) -> NDArray:
        """Dispersion values in s/m² (internal units)."""
        return self._values

    def get_wls(self) -> WavelengthArray:
        """Return the wavelength array."""
        return self._wavelengths

    def _disp_fn(self):
        """Return a cubic spline interpolant for dispersion vs wavelength."""
        return make_splrep(self._wavelengths.as_m, self.as_s_m_m)

    def _check_wl(self, wavelength_m: float):
        """Raise ValueError if wavelength is outside the interpolation range."""
        if wavelength_m < self._wavelengths.as_m.min() or wavelength_m > self._wavelengths.as_m.max():
            raise ValueError(
                f"values of dispersion available between "
                f"{self._wavelengths.as_m.min():.3e} and {self._wavelengths.as_m.max():.3e} m"
            )

    def fn(self, wavelength: float) -> float:
        """Interpolated dispersion at wavelength (m), returned in s/m²."""
        self._check_wl(wavelength)
        c_info("Dispersion unit: s/m^2")
        spl = self._disp_fn()
        return spl(wavelength).item()

    def fn_s_m_m(self, wavelength_nm: float) -> float:
        """Interpolated dispersion at wavelength (nm), returned in s/m²."""
        return self.fn(wavelength_nm * 1e-9)

    def fn_ps_nm_km(self, wavelength_nm: float) -> float:
        """Interpolated dispersion at wavelength (nm), returned in ps/(nm·km)."""
        return self.fn(wavelength_nm * 1e-9) * 1e6

    @classmethod
    def from_neff(
        cls,
        neff: NDArray,
        wavelengths: WavelengthArray,
        central_wavelength_nm: float,
        ignore_fit_error: bool = False,
    ) -> Self:
        """Compute dispersion D(λ) from effective index neff(λ).

        D = -λ/c × d²neff/dλ²
        """

        if not isinstance(wavelengths, WavelengthArray):
            raise TypeError("wavelengths should be an instance of `WavelengthArray`")
        # D = -lambda / C_MS * (d^2 neff/ d lambda^2)

        if len(neff) != len(wavelengths.value):  # type: ignore[arg-type]
            raise ValueError("Length of both neff and wavelengths should be same")
        wl = wavelengths.to_equally_spaced()
        interp = make_splrep(wavelengths.as_m, neff)(wl)

        spline = make_splrep(wl, interp)
        diff_2 = spline.derivative(2)

        dispersion: NDArray = -wavelengths.as_m / C_MS * diff_2(wavelengths.as_m)

        smooth_fit = np.all(np.diff(dispersion * 1e6) < 50)
        if not smooth_fit:
            warnings.warn(
                "Bad fitting of neff values. Consider building Disperison in other ways..."
            )
            if not ignore_fit_error:
                raise ChildProcessError(
                    "Can't perform numerical differentiation with small error..."
                )
        return cls(
            wavelengths=wavelengths,
            values=dispersion,
            unit="s/m^2",
            central_wavelength=Wavelength(central_wavelength_nm, "nm"),
        )

    @classmethod
    def from_propagation_constant(
        cls,
        beta: NDArray,
        wavelengths: WavelengthArray,
        central_wavelength_nm: float,
        ignore_fit_error: bool = False,
    ) -> Self:
        """Compute dispersion D(λ) from propagation constant β(ω).

        D = -(2πc)/λ² × d²β/dω²
        """
        # -(2*PI*C_MS) / lambda^2 * (d^2 beta/ d omega^2)

        if len(beta) != len(wavelengths.value):  # type: ignore[arg-type]
            raise ValueError("Length of both beta and wavelengths should be same")
        if not isinstance(wavelengths, WavelengthArray):
            raise TypeError(
                f"wavelengths cannot process the type: {type(wavelengths)}, required WavelengthArray"
            )

        omega = wavelengths.to_omega().to_equally_spaced()
        interp = make_splrep(wavelengths.to_omega().as_rad_s[::-1], beta[::-1])(omega)

        spline = make_splrep(omega[::-1], interp[::-1])
        diff_2 = spline.derivative(2)

        dispersion: NDArray = (
            -(2 * PI * C_MS)
            / wavelengths.as_m**2
            * diff_2(wavelengths.to_omega().as_rad_s)
        )

        smooth_fit = np.all(np.diff(dispersion * 1e6) < 50)
        if not smooth_fit:
            warnings.warn(
                "Bad fitting of neff values. Consider building Disperison in other ways..."
            )
            if not ignore_fit_error:
                raise ChildProcessError(
                    "Can't perform numerical differentiation with small error..."
                )
        return cls(
            wavelengths=wavelengths,
            values=dispersion,
            unit="s/m^2",
            central_wavelength=Wavelength(central_wavelength_nm, "nm"),
        )

    def get_beta2(self, wavelength_nm: float):
        """Return β₂ at wavelength_nm (s²/m) via spline interpolation."""
        _min_wl = self._wavelengths.as_nm.min()
        _max_wl = self._wavelengths.as_nm.max()
        if wavelength_nm > _max_wl or wavelength_nm < _min_wl:
            raise ValueError(
                f"values of disersion available between {_min_wl} and {_max_wl} nm."
            )
        beta2 = -self._wavelengths.as_m**2 / (2 * PI * C_MS) * self.as_s_m_m
        spline = make_splrep(self._wavelengths.as_nm, beta2)
        return float(spline(wavelength_nm))

    def get_betas(
        self,
        polyOrder,
        wavelength: Wavelength | None = None,
        make_plot=False,
        return_diagnostics=False,
    ):
        """Fit dispersion data to a polynomial in ω and return beta coefficients.

        Returns betas in array [beta2, beta3, ...].
        If return_diagnostics is True, returns (betas, fit_x_axis, data, fit).
        """

        if wavelength is None:
            wavelength = self.central_wavelength

        if wavelength is None:
            raise ValueError("No central wavelength available")

        min_nm = self.get_wls().as_nm
        maximum = max(min_nm)
        minimum = min(min_nm)
        if wavelength.as_nm < minimum or wavelength.as_nm > maximum:
            warnings.warn(
                f"wavelength given is not in the range of dispersion: \nit should be between {minimum} and {maximum}"
            )
            return

        # omega - omega_0
        omegaAxis = 2 * np.pi * C_MS / (self.get_wls().as_m) - 2 * np.pi * C_MS / (
            wavelength.as_m
        )

        # Convert from D to beta via beta2 = -D * lambda^2 / (2*pi*c)
        betaTwo = -self.as_s_m_m * (self.get_wls().as_m) ** 2 / (2 * np.pi * C_MS)
        # The units of beta2 for the GNLSE solver are ps^2/m; convert
        betaTwo = betaTwo * 1e24
        # Also convert angular frequency to rad/ps
        omegaAxis = omegaAxis * 1e-12  #  s/ps

        # Fit beta2 with high-order polynomial
        polyFitCo = np.polyfit(omegaAxis, betaTwo, polyOrder)

        Betas = polyFitCo[::-1]

        polyFit = np.zeros((len(omegaAxis),))

        for i in range(len(Betas)):
            Betas[i] = Betas[i] * factorial(i)
            polyFit = polyFit + Betas[i] / factorial(i) * omegaAxis**i

        if make_plot:
            plt.plot(omegaAxis, betaTwo, "o")
            plt.plot(omegaAxis, polyFit)
            plt.show()
        if return_diagnostics:
            return Betas, omegaAxis, betaTwo, polyFit
        else:
            return Betas


@dataclass(config={"arbitrary_types_allowed": True})
class PropagationConstant:
    """Propagation constant β as a function of wavelength or angular frequency.

    Attributes
    ----------
    values : β values.
    x_values : wavelength or angular frequency array.
    """

    _values: NDArray = Field(alias="values")
    _x_values: WavelengthArray | AngularFrequencyArray = Field(alias="x_values")
    _wavelengths: WavelengthArray | None = None
    _omegas: AngularFrequencyArray | None = None

    @model_validator(mode="after")
    def _setup(self) -> "PropagationConstant":
        if isinstance(self._x_values, WavelengthArray):
            self._wavelengths = self._x_values
        elif isinstance(self._x_values, AngularFrequencyArray):
            self._omegas = self._x_values
        return self

    @classmethod
    def beta2_from_neff(
        cls, neff: NDArray, x_values: WavelengthArray | AngularFrequencyArray
    ):
        """Compute β₂ = neff × ω / c from effective index."""
        if not isinstance(x_values, (WavelengthArray, AngularFrequencyArray)):
            raise TypeError("x_values should be a type of either `WavelengthArray` or `AngularFrequencyArray`")
        if len(neff) != len(x_values.value):  # type: ignore[arg-type]
            raise ValueError("both neff and x_values must be of same length.")

        if isinstance(x_values, WavelengthArray):
            omegas = x_values.to_omega()
        elif isinstance(x_values, AngularFrequencyArray):
            omegas = x_values

        beta2 = neff * omegas.as_rad_s / C_MS
        return beta2

    @classmethod
    def from_neff_omega(cls, neff: NDArray, omega: AngularFrequencyArray) -> Self:
        """Construct from effective index neff and angular frequency array."""
        if not isinstance(omega, AngularFrequencyArray):
            raise TypeError(
                f"omega should be a type of 'AngularFrequencyArray' : got {type(omega)}"
            )
        if len(neff) != len(omega.value):  # type: ignore[arg-type]
            raise ValueError(
                "both neff and angular frequency array must have same length"
            )

        betas = omega.as_rad_s * neff / C_MS
        return cls(values=betas, x_values=omega)
