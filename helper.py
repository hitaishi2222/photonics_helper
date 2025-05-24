from __future__ import annotations
from typing import Literal, Self

import numpy as np
import scipy as sp
from numpy.typing import NDArray

# Constants
PI: float = sp.constants.pi
C_MS: float = sp.constants.c


class Wavelength(float):
    def __new__(cls, value: float, unit: Literal["nm", "um", "m"] = "nm") -> Self:
        if unit == "nm":
            value *= 1e-9
        elif unit == "um":
            value *= 1e-6
        else:
            raise ValueError(f"Unsupported unit: {unit} use 'nm', 'um', or 'm'")
        return super().__new__(cls, value)

    def __repr__(self) -> str:
        return f"Wavelength -> {self.as_m} m"

    @property
    def as_m(self) -> float:
        return self

    @property
    def as_nm(self) -> float:
        return self * 1e9

    def to_freq(self) -> Frequency:
        return Frequency(C_MS / self)

    def to_omega(self) -> Omega:
        return Omega(2 * PI * C_MS / self, "rad/s")


class Frequency(float):
    def __new__(
        cls, value: float, unit: Literal["THz", "GHz", "MHz", "Hz"] = "Hz"
    ) -> Self:
        if unit == "THz":
            value *= 1e12
        elif unit == "GHz":
            value *= 1e9
        elif unit == "MHz":
            value *= 1e6
        else:
            raise ValueError(
                f"Unsupported unit: {unit} use 'THz', 'GHz', 'MHz' or 'Hz'"
            )
        return super().__new__(cls, value)

    def __repr__(self) -> str:
        return f"Frequency -> {self.as_Hz} Hz"

    @property
    def as_Hz(self):
        return self

    @property
    def as_THz(self):
        return self * 1e-12

    @property
    def as_GHz(self):
        return self * 1e-9

    @property
    def as_MHz(self):
        return self * 1e-6

    def to_wl(self) -> Wavelength:
        return Wavelength(C_MS / self, "m")

    def to_omega(self) -> Omega:
        return Omega(2 * PI * self.as_Hz, "rad/s")


class Omega(float):
    def __new__(cls, value: float, unit: Literal["rad/s", "rad/ps"] = "rad/s") -> Self:
        if unit == "rad/ps":
            value *= 1e-12
        else:
            raise ValueError(f"Unsupported unit: {unit} use 'rad/s' or 'rad/ps'")
        return super().__new__(cls, value)

    def __repr__(self) -> str:
        return f"Angular Frequency -> {self.as_rad_s} rad/s"

    @property
    def as_rad_s(self) -> float:
        return self

    @property
    def as_rad_ps(self) -> float:
        return self * 1e-12

    def to_wl(self) -> Wavelength:
        return Wavelength((2 * PI) * C_MS / self, "m")

    def to_freq(self) -> Frequency:
        return Frequency(self / (2 * PI), "Hz")


class WavelengthArray(np.ndarray):
    def __new__(cls, value: NDArray, unit: Literal["nm", "um", "m"] = "nm") -> Self:
        if unit == "nm":
            value *= 1e-9
        elif unit == "um":
            value *= 1e-6
        else:
            raise ValueError(f"Unsupported unit: {unit} use 'nm', 'um', or 'm'")
        return super().__new__(cls, value)

    def __repr__(self) -> str:
        return f"Wavelength -> {self.as_m} m"

    @property
    def as_m(self) -> NDArray:
        return self

    @property
    def as_nm(self) -> NDArray:
        return self * 1e9

    def to_freq(self) -> FrequencyArray:
        return FrequencyArray(C_MS / self)

    def to_omega(self) -> OmegaArray:
        return OmegaArray(2 * PI * C_MS / self, "rad/s")


class FrequencyArray(np.ndarray):
    def __new__(
        cls, value: NDArray, unit: Literal["THz", "GHz", "MHz", "Hz"] = "Hz"
    ) -> Self:
        if unit == "THz":
            value *= 1e12
        elif unit == "GHz":
            value *= 1e9
        elif unit == "MHz":
            value *= 1e6
        else:
            raise ValueError(
                f"Unsupported unit: {unit} use 'THz', 'GHz', 'MHz' or 'Hz'"
            )
        return super().__new__(cls, value)

    def __repr__(self) -> str:
        return f"Frequency -> {self.as_Hz} Hz"

    @property
    def as_Hz(self) -> NDArray:
        return self

    @property
    def as_THz(self) -> NDArray:
        return self * 1e-12

    @property
    def as_GHz(self) -> NDArray:
        return self * 1e-9

    @property
    def as_MHz(self) -> NDArray:
        return self * 1e-6

    def to_wl(self) -> WavelengthArray:
        return WavelengthArray(C_MS / self, "m")

    def to_omega(self) -> OmegaArray:
        return OmegaArray(2 * PI * self.as_Hz, "rad/s")


class OmegaArray(np.ndarray):
    def __new__(
        cls, value: NDArray, unit: Literal["rad/s", "rad/ps"] = "rad/s"
    ) -> Self:
        if unit == "rad/ps":
            value *= 1e-12
        else:
            raise ValueError(f"Unsupported unit: {unit} use 'rad/s' or 'rad/ps'")
        return super().__new__(cls, value)

    def __repr__(self) -> str:
        return f"Angular Frequency -> {self.as_rad_s} rad/s"

    @property
    def as_rad_s(self) -> NDArray:
        return self

    @property
    def as_rad_ps(self) -> NDArray:
        return self * 1e-12

    def to_wl(self) -> WavelengthArray:
        return WavelengthArray((2 * PI) * C_MS / self, "m")

    def to_freq(self) -> FrequencyArray:
        return FrequencyArray(self / (2 * PI), "Hz")
