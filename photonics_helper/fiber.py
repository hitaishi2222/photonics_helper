"""Fiber dispersion and propagation constant calculations."""

from scipy.interpolate import BSpline
from photonics_helper.base import (
    C_MS,
    PI,
    AngularFrequencyArray,
    Wavelength,
    WavelengthArray,
)

from functools import cached_property
from numpy.typing import NDArray
from typing import Literal, Self
from pydantic.dataclasses import dataclass
from pydantic import model_validator

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

    values: NDArray
    unit: Literal["ps/nm.km", "s/m^2"]
    wavelengths: WavelengthArray
    central_wavelength: Wavelength

    def __repr__(self):
        return f"Dispersion: from wl: {self.wavelengths.as_m.min()} to {self.wavelengths.as_m.max()}"

    def __str__(self):
        return f"Dispersion D(λ) in ps/(nm·km), range {self.wavelengths.as_nm.min():.0f}–{self.wavelengths.as_nm.max():.0f} nm"

    @cached_property
    def as_ps_nm_km(self) -> NDArray:
        """Dispersion values in ps/(nm·km)."""
        return self.values

    @cached_property
    def as_s_m_m(self) -> NDArray:
        """Dispersion values in s/m² (internal units)."""
        return self.values

    def get_wls(self) -> WavelengthArray:
        """Return the wavelength array."""
        return self.wavelengths

    @cached_property
    def _disp_fn(self) -> BSpline:
        """Return a cubic spline interpolant for dispersion vs wavelength."""
        try:
            return make_splrep(self.wavelengths.as_m, self.as_s_m_m)
        except Exception as e:
            raise Exception(f"{e}")

    def _check_wl(self, wavelength: Wavelength):
        """Raise ValueError if wavelength is outside the interpolation range."""
        if (
            wavelength.as_m < self.wavelengths.as_m.min()
            or wavelength.as_m > self.wavelengths.as_m.max()
        ):
            raise ValueError(
                f"values of dispersion available between "
                f"{self.wavelengths.as_m.min():.3e} and {self.wavelengths.as_m.max():.3e} m"
            )

    def fn(self, wavelength: Wavelength) -> float:
        """Interpolated dispersion at wavelength (m)"""
        self._check_wl(wavelength)
        return self._disp_fn(wavelength.as_m).item()

    @classmethod
    def from_neff(
        cls,
        neff: NDArray,
        wavelengths: WavelengthArray,
        central_wavelength: Wavelength,
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
            central_wavelength=central_wavelength,
        )

    @classmethod
    def from_propagation_constant(
        cls,
        beta: NDArray,
        wavelengths: WavelengthArray,
        central_wavelength: Wavelength,
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
            central_wavelength=central_wavelength,
        )

    def get_beta2(self, wavelength_nm: float):
        """Return β₂ at wavelength_nm (s²/m) via spline interpolation."""
        _min_wl = self.wavelengths.as_nm.min()
        _max_wl = self.wavelengths.as_nm.max()
        if wavelength_nm > _max_wl or wavelength_nm < _min_wl:
            raise ValueError(
                f"values of disersion available between {_min_wl} and {_max_wl} nm."
            )
        beta2 = -self.wavelengths.as_m**2 / (2 * PI * C_MS) * self.as_s_m_m
        spline = make_splrep(self.wavelengths.as_nm, beta2)
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

    values: NDArray
    x_values: WavelengthArray | AngularFrequencyArray

    @model_validator(mode="after")
    def _setup(self) -> "PropagationConstant":
        if isinstance(self.x_values, WavelengthArray):
            self.wavelengths = self.x_values
            self.omegas = self.x_values.to_omega()
        elif isinstance(self.x_values, AngularFrequencyArray):
            self.omegas = self.x_values
            self.wavelengths = self.x_values.to_wl()
        return self

    @classmethod
    def beta2_from_neff(
        cls, neff: NDArray, x_values: WavelengthArray | AngularFrequencyArray
    ):
        """Compute β₂ = neff × ω / c from effective index."""
        if not isinstance(x_values, (WavelengthArray, AngularFrequencyArray)):
            raise TypeError(
                "x_values should be a type of either `WavelengthArray` or `AngularFrequencyArray`"
            )
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
