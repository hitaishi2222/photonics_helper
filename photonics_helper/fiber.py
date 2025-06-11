from photonics_helper.base import (
    C_MS,
    PI,
    AngularFrequencyArray,
    Wavelength,
    WavelengthArray,
)
from photonics_helper.looks import c_info

from numpy.typing import NDArray
from typing import Literal, Self

import warnings
import numpy as np
from scipy.interpolate import make_splrep
from rich.traceback import install

install()


class Dispersion:
    def __init__(
        self,
        wavelengths: WavelengthArray,
        values: NDArray,
        unit: Literal["ps/nm.km", "s/m^2"],
        central_wavelength: Wavelength,
    ):
        if unit == "ps/nm.km":
            values = values * 1e6  # (12-9+3)
        elif unit == "s/m^2":
            pass
        self._values = values
        self._wavelengths = wavelengths
        self._unit = "s/m^2"

    def __repr__(self):
        return f"Dispersion: from wl: {self._wavelengths.min()} to {self._wavelengths.max()}"

    @property
    def as_ps_nm_km(self) -> NDArray:
        return self._values * 1e6

    @property
    def as_s_m_m(self) -> NDArray:
        return self._values

    def get_wl(self) -> WavelengthArray:
        return self._wavelengths

    def check_wavelength_limit(self, wavelength: float, unit: Literal["nm", "m", "um"]):
        min = 0
        max = 0

        if unit == "m":
            min = self._wavelengths.as_m.min()
            max = self._wavelengths.as_m.max()
        elif unit == "um":
            min = self._wavelengths.as_um.min()
            max = self._wavelengths.as_um.max()
        else:
            min = self._wavelengths.as_nm.min()
            max = self._wavelengths.as_nm.max()

        if wavelength > max or wavelength < min:
            raise ValueError(f"values of disersion available between {min} and {max}")

    def fn(self, wavelength: float) -> float:
        self.check_wavelength_limit(wavelength, "m")
        c_info("Dispersion unit: s/m^2")
        spline = make_splrep(self._wavelengths.as_m, self.as_s_m_m)
        return float(spline(wavelength))

    def fn_s_m_m(self, wavelength_nm: float) -> float:
        self.check_wavelength_limit(wavelength_nm, "nm")
        c_info("Dispersion unit: s/m^2")
        spline = make_splrep(self._wavelengths.as_nm, self.as_s_m_m)
        return float(spline(wavelength_nm))

    def fn_ps_nm_km(self, wavelength_nm: float) -> float:
        self.check_wavelength_limit(wavelength_nm, "nm")
        c_info("Dispersion unit: ps/nm.km")
        spline = make_splrep(self._wavelengths.as_nm, self.as_ps_nm_km)
        return float(spline(wavelength_nm))

    @classmethod
    def from_neff(
        cls, neff: NDArray, wavelengths: WavelengthArray, central_wavelength_nm: float
    ) -> Self:
        # D = -lambda / C_MS * (d^2 neff/ d lambda^2)

        if len(neff) != len(wavelengths):
            raise ValueError("Length of both neff and wavelengths should be same")
        if not isinstance(wavelengths, WavelengthArray):
            raise TypeError(
                f"wavelengths cannot process the type: {type(wavelengths)}, required WavelengthArray"
            )
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
    def from_propagation_constanant(
        cls, beta: NDArray, wavelengths: WavelengthArray, central_wavelength_nm: float
    ) -> Self:
        # -(2*PI*C_MS) / lambda^2 * (d^2 beta/ d omega^2)

        if len(beta) != len(wavelengths):
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
        min = self._wavelengths.as_nm.min()
        max = self._wavelengths.as_nm.max()
        if wavelength_nm > max or wavelength_nm < min:
            raise ValueError(
                f"values of disersion available between {min} and {max} nm."
            )
        beta2 = -self._wavelengths.as_m**2 / (2 * PI * C_MS) * self.as_s_m_m
        spline = make_splrep(self._wavelengths.as_nm, beta2)
        return float(spline(wavelength_nm))


class PropagationConstant:
    def __init__(
        self, values: NDArray, x_values: WavelengthArray | AngularFrequencyArray
    ):
        if isinstance(x_values, WavelengthArray):
            self._wavelengths = x_values
        elif isinstance(x_values, AngularFrequencyArray):
            self._omegas = x_values
        self._values = values

    @classmethod
    def beta2_from_neff(
        cls, neff: NDArray, x_values: WavelengthArray | AngularFrequencyArray
    ):
        if len(neff) != len(x_values):
            raise ValueError("both neff and x_values must be of same length.")

        if isinstance(x_values, WavelengthArray):
            omegas = x_values.to_omega()
        elif isinstance(x_values, AngularFrequencyArray):
            omegas = x_values

        beta2 = neff * omegas / C_MS
        return beta2

    @classmethod
    def from_neff_omega(cls, neff: NDArray, omega: AngularFrequencyArray) -> Self:
        if len(neff) != len(omega):
            raise ValueError(
                "both neff and angular frequency array must have same length"
            )
        if not isinstance(omega, AngularFrequencyArray):
            raise TypeError(
                f"omega should be a type of 'AngularFrequencyArray' : got {type(omega)}"
            )

        betas = omega.as_rad_s * neff / C_MS
        return cls(values=betas, x_values=omega)
