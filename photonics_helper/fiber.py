from typing import Literal, Self
from numpy.typing import NDArray
from scipy.interpolate import make_splrep

from photonics_helper.base import C_MS, AngularFrequencyArray, WavelengthArray


class Dispersion:
    def __init__(
        self,
        wavelengths: WavelengthArray,
        values: NDArray,
        unit: Literal["ps/nm.km", "s/m^2"],
    ):
        if unit == "ps/nm.km":
            values = values * 1e6  # (12-9+3)
        elif unit == "s/m^2":
            pass
        self._values = values
        self._wavelengths = wavelengths
        self._unit = "s/m^2"

    @property
    def as_ps_nm_km(self) -> NDArray:
        return self._values * 1e-6

    @property
    def as_s_m_m(self) -> NDArray:
        return self._values

    def get_wl(self) -> WavelengthArray:
        return self._wavelengths

    @classmethod
    def from_neff(cls, neff: NDArray, wavelengths: WavelengthArray):
        # -lambda / C_MS * (d^2 neff/ d lambda^2)
        spline = make_splrep(wavelengths, neff)
        diff_2 = spline.derivative(2)
        dispersion = -wavelengths.as_m / C_MS * diff_2(wavelengths)
        return cls(values=dispersion, wavelengths=wavelengths, unit="s/m^2")


class Betas:
    def __init__(
        self, values: NDArray, x_values: WavelengthArray | AngularFrequencyArray
    ):
        if isinstance(x_values, WavelengthArray):
            self._wavelengths = x_values
        elif isinstance(x_values, AngularFrequencyArray):
            self._omegas = x_values
        self._values = values

    @classmethod
    def from_neff(
        cls, neff: NDArray, x_values: WavelengthArray | AngularFrequencyArray
    ) -> Self:
        if len(neff) != len(x_values):
            raise ValueError("both neff and x_values must be of same length.")

        if isinstance(x_values, WavelengthArray):
            omegas = x_values.to_omega()
        elif isinstance(x_values, AngularFrequencyArray):
            omegas = x_values
        else:
            raise ValueError(
                "x_values only take type of: WavelengthArray and AngularFrequencyArray only."
            )

        betas = neff * omegas / C_MS
        return cls(values=betas, x_values=omegas)
