from __future__ import annotations
from typing import Literal, Self

import numpy as np
import scipy as sp
from numpy.typing import NDArray

# Constants
PI: float = sp.constants.pi
C_MS: float = sp.constants.c


class Wavelength(float):
    def __new__(cls, value: float, unit: Literal["nm", "um", "m"]) -> Self:
        if unit == "nm":
            value *= 1e-9
        elif unit == "um":
            value *= 1e-6
        elif unit == "m":
            pass  # Already in meters, no conversion needed
        else:
            raise ValueError(f"Unsupported unit: {unit} use 'nm', 'um', or 'm'")
        return super().__new__(cls, value)

    @property
    def as_m(self) -> float:
        return float(self)

    @property
    def as_um(self) -> float:
        return self * 1e6

    @property
    def as_nm(self) -> float:
        return self * 1e9

    def to_freq(self) -> Frequency:
        return Frequency(C_MS / self, "Hz")

    def to_omega(self) -> AngularFrequency:
        return AngularFrequency(2 * PI * C_MS / self, "rad/s")


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
        elif unit == "Hz":
            pass  # Already in Hz, no conversion needed
        else:
            raise ValueError(
                f"Unsupported unit: {unit} use 'THz', 'GHz', 'MHz' or 'Hz'"
            )
        return super().__new__(cls, value)

    @property
    def as_Hz(self) -> float:
        return float(self)

    @property
    def as_THz(self) -> float:
        return self * 1e-12

    @property
    def as_GHz(self) -> float:
        return self * 1e-9

    @property
    def as_MHz(self) -> float:
        return self * 1e-6

    def to_wl(self) -> Wavelength:
        return Wavelength(C_MS / self, "m")

    def to_omega(self) -> AngularFrequency:
        return AngularFrequency(2 * PI * self.as_Hz, "rad/s")


class AngularFrequency(float):
    def __new__(cls, value: float, unit: Literal["rad/s", "rad/ps"] = "rad/s") -> Self:
        if unit == "rad/ps":
            value *= 1e-12  # Convert from rad/ps to rad/s
        elif unit == "rad/s":
            pass  # Already in rad/s, no conversion needed
        else:
            raise ValueError(f"Unsupported unit: {unit} use 'rad/s' or 'rad/ps'")
        return super().__new__(cls, value)

    def __repr__(self) -> str:
        # Use float() to avoid recursion when converting self to string
        return f"Angular Frequency -> {float(self)} rad/s"

    @property
    def as_rad_s(self) -> float:
        # Use float() to avoid recursion when accessing value
        return float(self)

    @property
    def as_rad_ps(self) -> float:
        # Use float() to avoid recursion in multiplication
        return float(self) * 1e12  # Convert from rad/s to rad/ps

    def to_wl(self) -> Wavelength:
        return Wavelength((2 * PI) * C_MS / self, "m")

    def to_freq(self) -> Frequency:
        return Frequency(self / (2 * PI), "Hz")


class WavelengthArray(np.ndarray):
    def __new__(cls, value: NDArray, unit: Literal["nm", "um", "m"]) -> Self:
        # Convert input array to float type
        value = np.array(value, dtype=float)
        if unit == "nm":
            value *= 1e-9
        elif unit == "um":
            value *= 1e-6
        elif unit == "m":
            pass  # Already in meters, no conversion needed
        else:
            raise ValueError(f"Unsupported unit: {unit} use 'nm', 'um', or 'm'")
        return super().__new__(cls, value.shape, dtype=float, buffer=value)

    @property
    def as_m(self) -> NDArray:
        return np.array(self)

    @property
    def as_um(self) -> NDArray:
        return np.array(self) * 1e6

    @property
    def as_nm(self) -> NDArray:
        return np.array(self) * 1e9

    def to_freq(self) -> FrequencyArray:
        return FrequencyArray(C_MS / self.as_m, "Hz")

    def to_omega(self) -> AngularFrequencyArray:
        return AngularFrequencyArray(2 * PI * C_MS / self.as_m, "rad/s")

    def to_equally_spaced(self, points=51) -> NDArray:
        min = self.as_m.min()
        max = self.as_m.max()
        return np.linspace(min, max, points)


class FrequencyArray(np.ndarray):
    def __new__(
        cls, value: NDArray, unit: Literal["THz", "GHz", "MHz", "Hz"] = "Hz"
    ) -> Self:
        # Convert input array to float type
        value = np.array(value, dtype=float)
        if unit == "THz":
            value *= 1e12
        elif unit == "GHz":
            value *= 1e9
        elif unit == "MHz":
            value *= 1e6
        elif unit == "Hz":
            pass  # Already in Hz, no conversion needed
        else:
            raise ValueError(
                f"Unsupported unit: {unit} use 'THz', 'GHz', 'MHz' or 'Hz'"
            )
        return super().__new__(cls, value.shape, dtype=float, buffer=value)

    @property
    def as_Hz(self) -> NDArray:
        return np.array(self)

    @property
    def as_THz(self) -> NDArray:
        return np.array(self) * 1e-12

    @property
    def as_GHz(self) -> NDArray:
        return np.array(self) * 1e-9

    @property
    def as_MHz(self) -> NDArray:
        return np.array(self) * 1e-6

    def to_wl(self) -> WavelengthArray:
        return WavelengthArray(C_MS / self, "m")

    def to_omega(self) -> AngularFrequencyArray:
        return AngularFrequencyArray(2 * PI * self.as_Hz, "rad/s")

    def to_equally_spaced(self, points=51) -> NDArray:
        min = self.as_Hz.min()
        max = self.as_Hz.max()
        return np.linspace(min, max, points)


class AngularFrequencyArray(np.ndarray):
    def __new__(
        cls, value: NDArray, unit: Literal["rad/s", "rad/ps"] = "rad/s"
    ) -> Self:
        # Convert input array to float type
        value = np.array(value, dtype=float)
        if unit == "rad/ps":
            value *= 1e-12  # Convert from rad/ps to rad/s
        elif unit == "rad/s":
            pass  # Already in rad/s, no conversion needed
        else:
            raise ValueError(f"Unsupported unit: {unit} use 'rad/s' or 'rad/ps'")
        return super().__new__(cls, value.shape, dtype=float, buffer=value)

    @property
    def as_rad_s(self) -> NDArray:
        return np.array(self)

    @property
    def as_rad_ps(self) -> NDArray:
        return np.array(self) * 1e-12

    def to_wl(self) -> WavelengthArray:
        return WavelengthArray((2 * PI) * C_MS / self, "m")

    def to_freq(self) -> FrequencyArray:
        return FrequencyArray(self / (2 * PI), "Hz")

    def to_equally_spaced(self, points=51) -> NDArray:
        min = self.as_rad_s.min()
        max = self.as_rad_s.max()
        return np.linspace(min, max, points)
