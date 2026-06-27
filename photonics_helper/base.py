from __future__ import annotations
from typing import Literal, Self
from numpy.typing import NDArray, ArrayLike
from functools import cached_property


import numpy as np
import scipy as sp
from rich.traceback import install
from pydantic.dataclasses import dataclass

install()

# Constants
PI: float = sp.constants.pi
C_MS: float = sp.constants.c
EPS_0: float = sp.constants.epsilon_0
MU_0: float = sp.constants.mu_0


@dataclass(config={"arbitrary_types_allowed": True})
class Wavelength:
    value: float
    unit: Literal["nm", "um", "m"]

    def __post_init__(self):
        if self.unit == "nm":
            self.value *= 1e-9
        elif self.unit == "um":
            self.value *= 1e-6
        elif self.unit == "m":
            pass  # Already in meters, no conversion needed
        else:
            raise ValueError(f"Unsupported unit: {self.unit} use 'nm', 'um', or 'm'")

    def __repr__(self) -> str:
        return f"Wavelength -> {self.as_m} m"

    @cached_property
    def as_m(self) -> float:
        return self.value

    @cached_property
    def as_um(self) -> float:
        return self.value * 1e6

    @cached_property
    def as_nm(self) -> float:
        return self.value * 1e9

    def to_freq(self) -> Frequency:
        return Frequency(C_MS / self.value, "Hz")

    def to_omega(self) -> AngularFrequency:
        return AngularFrequency(2 * PI * C_MS / self.value, "rad/s")

    def to_wn(self) -> Wavenumber:
        return Wavenumber(value=1 / self.as_m, unit="1/m")


@dataclass(config={"arbitrary_types_allowed": True})
class Frequency:
    value: float
    unit: Literal["THz", "GHz", "MHz", "Hz"]

    def __post_init__(self):
        if self.unit == "THz":
            self.value *= 1e12
        elif self.unit == "GHz":
            self.value *= 1e9
        elif self.unit == "MHz":
            self.value *= 1e6
        elif self.unit == "Hz":
            pass  # Already in Hz, no conversion needed
        else:
            raise ValueError(
                f"Unsupported unit: {self.unit} use 'THz', 'GHz', 'MHz' or 'Hz'"
            )

    def __repr__(self) -> str:
        return f"Frequency -> {self.as_Hz} Hz"

    @cached_property
    def as_Hz(self) -> float:
        return self.value

    @cached_property
    def as_THz(self) -> float:
        return self.value * 1e-12

    @cached_property
    def as_GHz(self) -> float:
        return self.value * 1e-9

    @cached_property
    def as_MHz(self) -> float:
        return self.value * 1e-6

    def to_wl(self) -> Wavelength:
        return Wavelength(C_MS / self.value, "m")

    def to_omega(self) -> AngularFrequency:
        return AngularFrequency(2 * PI * self.as_Hz, "rad/s")

    def to_wn(self) -> Wavenumber:
        return Wavenumber(value=self.as_Hz / C_MS, unit="1/m")


@dataclass(config={"arbitrary_types_allowed": True})
class AngularFrequency:
    value: float
    unit: Literal["rad/s", "rad/ps"]

    def __post_init__(self):
        if self.unit == "rad/ps":
            self.value *= 1e12  # Convert from rad/ps to rad/s
        elif self.unit == "rad/s":
            pass  # Already in rad/s, no conversion needed
        else:
            raise ValueError(f"Unsupported unit: {self.unit} use 'rad/s' or 'rad/ps'")

    def __repr__(self) -> str:
        return f"Angular Frequency -> {self.value} rad/s"

    @cached_property
    def as_rad_s(self) -> float:
        return self.value

    @cached_property
    def as_rad_ps(self) -> float:
        return self.value * 1e-12

    def to_wl(self) -> Wavelength:
        return Wavelength((2 * PI * C_MS) / self.value, "m")

    def to_freq(self) -> Frequency:
        return Frequency(self.value / (2 * PI), "Hz")

    def to_wn(self) -> Wavenumber:
        return Wavenumber(value=self.as_rad_s / (2 * PI * C_MS), unit="1/m")


@dataclass(config={"arbitrary_types_allowed": True})
class Wavenumber:
    value: float
    unit: Literal["1/cm", "1/m"]

    def __post_init__(self):
        if self.unit == "1/cm":
            self.value *= 1e2  # Convert from 1/cm to 1/m
        elif self.unit == "1/m":
            pass  # Already in 1/m, no conversion needed
        else:
            raise ValueError(f"Unsupported unit: {self.unit} use '1/cm' or '1/m'")

    def __repr__(self) -> str:
        return f"Wavenumber -> {self.as_1_m} 1/m"

    @cached_property
    def as_1_m(self) -> float:
        return self.value

    @cached_property
    def as_1_cm(self) -> float:
        return self.value * 1e-2

    @cached_property
    def as_angular(self) -> float:
        return self.value * 2 * PI

    def to_wl(self) -> Wavelength:
        return Wavelength(value=1 / self.as_1_m, unit="m")

    def to_freq(self) -> Frequency:
        return Frequency(value=C_MS * self.as_1_m, unit="Hz")

    def to_omega(self) -> AngularFrequency:
        return AngularFrequency(value=C_MS * 2 * PI * self.as_1_m, unit="rad/s")


@dataclass(config={"arbitrary_types_allowed": True})
class WavelengthArray:
    value: ArrayLike
    unit: Literal["nm", "um", "m"]

    def __post_init__(self):
        self.value = np.array(self.value, dtype=float)
        if self.unit == "nm":
            self.value *= 1e-9
        elif self.unit == "um":
            self.value *= 1e-6
        elif self.unit == "m":
            pass  # Already in meters, no conversion needed
        else:
            raise ValueError(f"Unsupported unit: {self.unit} use 'nm', 'um', or 'm'")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} -> from:{min(self.as_m)} m to: {max(self.as_m)} m"

    @cached_property
    def as_m(self) -> NDArray:
        return self.value

    @cached_property
    def as_um(self) -> NDArray:
        return self.value * 1e6

    @cached_property
    def as_nm(self) -> NDArray:
        return self.value * 1e9

    def to_freq(self) -> FrequencyArray:
        return FrequencyArray(C_MS / self.as_m, "Hz")

    def to_omega(self) -> AngularFrequencyArray:
        return AngularFrequencyArray(2 * PI * C_MS / self.as_m, "rad/s")

    def to_wn(self) -> WavenumberArray:
        return WavenumberArray(value=1 / self.as_m, unit="1/m")

    def to_equally_spaced(self, points=51) -> NDArray:
        min = self.as_m.min()
        max = self.as_m.max()
        return np.linspace(min, max, points)


@dataclass(config={"arbitrary_types_allowed": True})
class FrequencyArray:
    value: ArrayLike
    unit: Literal["THz", "GHz", "MHz", "Hz"]

    def __post_init__(self):
        self.value = np.array(self.value, dtype=float)
        if self.unit == "THz":
            self.value *= 1e12
        elif self.unit == "GHz":
            self.value *= 1e9
        elif self.unit == "MHz":
            self.value *= 1e6
        elif self.unit == "Hz":
            pass  # Already in Hz, no conversion needed
        else:
            raise ValueError(
                f"Unsupported unit: {self.unit} use 'THz', 'GHz', 'MHz' or 'Hz'"
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} -> from:{min(self.as_Hz)} Hz to: {max(self.as_Hz)} Hz"

    @cached_property
    def as_Hz(self) -> NDArray:
        return self.value

    @cached_property
    def as_THz(self) -> NDArray:
        return self.value * 1e-12

    @cached_property
    def as_GHz(self) -> NDArray:
        return self.value * 1e-9

    @cached_property
    def as_MHz(self) -> NDArray:
        return self.value * 1e-6

    def to_wl(self) -> WavelengthArray:
        return WavelengthArray(C_MS / self.value, "m")

    def to_omega(self) -> AngularFrequencyArray:
        return AngularFrequencyArray(2 * PI * self.as_Hz, "rad/s")

    def to_wn(self) -> WavenumberArray:
        return WavenumberArray(value=self.as_Hz / C_MS, unit="1/m")

    def to_equally_spaced(self, points=51) -> NDArray:
        _min = self.as_Hz.min()
        _max = self.as_Hz.max()
        return np.linspace(_max, _min, points)


@dataclass(config={"arbitrary_types_allowed": True})
class AngularFrequencyArray:
    value: ArrayLike
    unit: Literal["rad/s", "rad/ps"]

    def __post_init__(self) -> Self:
        self.value = np.array(self.value, dtype=float)
        if self.unit == "rad/ps":
            self.value *= 1e12  # Convert from rad/ps to rad/s
        elif self.unit == "rad/s":
            pass  # Already in rad/s, no conversion needed
        else:
            raise ValueError(f"Unsupported unit: {self.unit} use 'rad/s' or 'rad/ps'")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} -> from:{min(self.as_rad_s)} rad/s to: {max(self.as_rad_s)} rad/s"

    @cached_property
    def as_rad_s(self) -> NDArray:
        return self.value

    @cached_property
    def as_rad_ps(self) -> NDArray:
        return self.value * 1e-12

    def to_wl(self) -> WavelengthArray:
        return WavelengthArray((2 * PI * C_MS) / self.value, "m")

    def to_freq(self) -> FrequencyArray:
        return FrequencyArray(self.value / (2 * PI), "Hz")

    def to_wn(self) -> WavenumberArray:
        return WavenumberArray(value=self.as_rad_s / (2 * PI * C_MS), unit="1/m")

    def to_equally_spaced(self, points=51) -> NDArray:
        _min = self.as_rad_s.min()
        _max = self.as_rad_s.max()
        return np.linspace(_max, _min, points)


@dataclass(config={"arbitrary_types_allowed": True})
class WavenumberArray:
    value: ArrayLike
    unit: Literal["1/cm", "1/m"]

    def __post_init__(self):
        self.value = np.array(self.value, dtype=float)
        if self.unit == "1/cm":
            self.value *= 1e2  # Convert from 1/cm to 1/m
        elif self.unit == "1/m":
            pass  # Already in 1/m, no conversion needed
        else:
            raise ValueError(f"Unsupported unit: {self.unit} use '1/cm' or '1/m'")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} -> from:{min(self.as_1_m)} 1/m to: {max(self.as_1_m)} 1/m"

    @cached_property
    def as_1_m(self) -> NDArray:
        return self.value

    @cached_property
    def as_1_cm(self) -> NDArray:
        return self.value * 1e-2

    @cached_property
    def as_angular(self) -> NDArray:
        return self.value * 2 * PI

    def to_wl(self) -> WavelengthArray:
        return WavelengthArray(value=1 / self.as_1_m, unit="m")

    def to_freq(self) -> FrequencyArray:
        return FrequencyArray(value=C_MS * self.as_1_m, unit="Hz")

    def to_omega(self) -> AngularFrequencyArray:
        return AngularFrequencyArray(value=C_MS * 2 * PI * self.as_1_m, unit="rad/s")

    def to_equally_spaced(self, points=51) -> NDArray:
        _min = self.as_1_m.min()
        _max = self.as_1_m.max()
        return np.linspace(_max, _min, points)


class Permittivity(float):
    @classmethod
    def from_relative(cls, relative_value: float):
        return cls(EPS_0 * relative_value)

    def __repr__(self):
        return f"{super().__repr__()} F/m"


class Permiability(float):
    @classmethod
    def from_relative(cls, relative_value: float):
        return cls(MU_0 * relative_value)

    def __repr__(self):
        return f"{super().__repr__()} H/m"
