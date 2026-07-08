"""Physical units and constants for photonics calculations."""

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
    """Wavelength stored internally in meters.

    Accepts nm, um, or m on construction; converts to meters in ``__post_init__``.
    """

    value: float
    unit: Literal["nm", "um", "m"]

    def __post_init__(self):
        """Convert input value to meters."""
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

    def __str__(self) -> str:
        return f"{self.as_nm:.2f} nm"

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
        """Convert to Frequency (Hz)."""
        return Frequency(C_MS / self.value, "Hz")

    def to_omega(self) -> AngularFrequency:
        """Convert to AngularFrequency (rad/s)."""
        return AngularFrequency(2 * PI * C_MS / self.value, "rad/s")

    def to_wn(self) -> Wavenumber:
        """Convert to Wavenumber (1/m)."""
        return Wavenumber(value=1 / self.as_m, unit="1/m")

    @classmethod
    def from_meep(cls, value: float, base_length: Wavelength | None = None) -> Self:
        """Create from MEEP frequency units.

        In MEEP, c = 1, so f_Meep = a/λ where a is the base_length.
        """
        if base_length is None:
            base_length = Wavelength(1.0, "um")
        return cls(base_length.as_m / value, "m")

    @cached_property
    def as_meep(self) -> float:
        """Convert to MEEP frequency units: f_Meep = a/λ."""
        return 1e-6 / self.as_m


@dataclass(config={"arbitrary_types_allowed": True})
class Frequency:
    """Optical frequency stored internally in Hz.

    Accepts THz, GHz, MHz, or Hz on construction; converts to Hz in ``__post_init__``.
    """

    value: float
    unit: Literal["THz", "GHz", "MHz", "Hz"]

    def __post_init__(self):
        """Convert input value to Hz."""
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

    def __str__(self) -> str:
        return f"{self.as_THz:.4f} THz"

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
        """Convert to Wavelength (m)."""
        return Wavelength(C_MS / self.value, "m")

    def to_omega(self) -> AngularFrequency:
        """Convert to AngularFrequency (rad/s)."""
        return AngularFrequency(2 * PI * self.as_Hz, "rad/s")

    def to_wn(self) -> Wavenumber:
        """Convert to Wavenumber (1/m)."""
        return Wavenumber(value=self.as_Hz / C_MS, unit="1/m")

    @classmethod
    def from_meep(cls, value: float, base_length: Wavelength | None = None) -> Self:
        """Create from MEEP frequency units.

        In MEEP, c = 1, so f_Meep = ν·a.
        """
        if base_length is None:
            base_length = Wavelength(1.0, "um")
        return cls(value / base_length.as_m, "Hz")

    @cached_property
    def as_meep(self) -> float:
        """Convert to MEEP units (λ₀ = 1 μm)."""
        return self.as_Hz * 1e-6 / C_MS


@dataclass(config={"arbitrary_types_allowed": True})
class AngularFrequency:
    """Angular optical frequency stored internally in rad/s.

    Accepts rad/s or rad/ps on construction; converts to rad/s in ``__post_init__``.
    """

    value: float
    unit: Literal["rad/s", "rad/ps"]

    def __post_init__(self):
        """Convert input value to rad/s."""
        if self.unit == "rad/ps":
            self.value *= 1e12  # Convert from rad/ps to rad/s
        elif self.unit == "rad/s":
            pass  # Already in rad/s, no conversion needed
        else:
            raise ValueError(f"Unsupported unit: {self.unit} use 'rad/s' or 'rad/ps'")

    def __repr__(self) -> str:
        return f"Angular Frequency -> {self.value} rad/s"

    def __str__(self) -> str:
        return f"{self.as_rad_ps:.4f} rad/ps"

    @cached_property
    def as_rad_s(self) -> float:
        return self.value

    @cached_property
    def as_rad_ps(self) -> float:
        return self.value * 1e-12

    def to_wl(self) -> Wavelength:
        """Convert to Wavelength (m)."""
        return Wavelength((2 * PI * C_MS) / self.value, "m")

    def to_freq(self) -> Frequency:
        """Convert to Frequency (Hz)."""
        return Frequency(self.value / (2 * PI), "Hz")

    def to_wn(self) -> Wavenumber:
        """Convert to Wavenumber (1/m)."""
        return Wavenumber(value=self.as_rad_s / (2 * PI * C_MS), unit="1/m")

    @classmethod
    def from_meep(cls, value: float, base_length: Wavelength | None = None) -> Self:
        """Create from MEEP frequency units.

        In MEEP, c = 1, so f_Meep = ω·a/(2π).
        """
        if base_length is None:
            base_length = Wavelength(1.0, "um")
        return cls(2 * PI * value / base_length.as_m, "rad/s")

    @cached_property
    def as_meep(self) -> float:
        """Convert to MEEP frequency units: f_Meep = aω/(2πc)."""
        return self.as_rad_s * 1e-6 / (2 * PI * C_MS)


@dataclass(config={"arbitrary_types_allowed": True})
class Wavenumber:
    """Wavenumber stored internally in 1/m.

    Accepts 1/cm or 1/m on construction; converts to 1/m in ``__post_init__``.
    """

    value: float
    unit: Literal["1/cm", "1/m"]

    def __post_init__(self):
        """Convert input value to 1/m."""
        if self.unit == "1/cm":
            self.value *= 1e2  # Convert from 1/cm to 1/m
        elif self.unit == "1/m":
            pass  # Already in 1/m, no conversion needed
        else:
            raise ValueError(f"Unsupported unit: {self.unit} use '1/cm' or '1/m'")

    def __repr__(self) -> str:
        return f"Wavenumber -> {self.as_1_m} 1/m"

    def __str__(self) -> str:
        return f"{self.as_1_cm:.4f} 1/cm"

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
        """Convert to Wavelength (m)."""
        return Wavelength(value=1 / self.as_1_m, unit="m")

    def to_freq(self) -> Frequency:
        """Convert to Frequency (Hz)."""
        return Frequency(value=C_MS * self.as_1_m, unit="Hz")

    def to_omega(self) -> AngularFrequency:
        """Convert to AngularFrequency (rad/s)."""
        return AngularFrequency(value=C_MS * 2 * PI * self.as_1_m, unit="rad/s")

    @classmethod
    def from_meep(cls, value: float, base_length: Wavelength | None = None) -> Self:
        """Create from MEEP frequency units.

        In MEEP, c = 1, so f_Meep = k·a/(2π).
        """
        if base_length is None:
            base_length = Wavelength(1.0, "um")
        return cls(2 * PI * value / base_length.as_m, "1/m")

    @cached_property
    def as_meep(self) -> float:
        """Convert to MEEP frequency units: f_Meep = ak/(2π)."""
        return self.as_1_m * 1e-6 / (2 * PI)


@dataclass(config={"arbitrary_types_allowed": True})
class WavelengthArray:
    """Array of wavelengths stored internally in meters."""

    value: ArrayLike
    unit: Literal["nm", "um", "m"]

    def __post_init__(self):
        """Convert input array to meters."""
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
        """Convert to FrequencyArray (Hz)."""
        return FrequencyArray(C_MS / self.as_m, "Hz")

    def to_omega(self) -> AngularFrequencyArray:
        """Convert to AngularFrequencyArray (rad/s)."""
        return AngularFrequencyArray(2 * PI * C_MS / self.as_m, "rad/s")

    def to_wn(self) -> WavenumberArray:
        """Convert to WavenumberArray (1/m)."""
        return WavenumberArray(value=1 / self.as_m, unit="1/m")

    def to_equally_spaced(self, points=51) -> NDArray:
        """Return equally-spaced wavelength values between min and max."""
        _min = self.as_m.min()
        _max = self.as_m.max()
        return np.linspace(_min, _max, points)

    @classmethod
    def from_meep(cls, value: float | NDArray, base_length: Wavelength | None = None) -> Self:
        """Create from MEEP frequency units.

        In MEEP, c = 1, so f_Meep = a/λ where a is the base_length.
        """
        if base_length is None:
            base_length = Wavelength(1.0, "um")
        return cls(base_length.as_m / value, "m")

    @cached_property
    def as_meep(self) -> NDArray:
        """Convert to MEEP frequency units: f_Meep = a/λ."""
        return 1e-6 / self.as_m


@dataclass(config={"arbitrary_types_allowed": True})
class FrequencyArray:
    """Array of frequencies stored internally in Hz."""

    value: ArrayLike
    unit: Literal["THz", "GHz", "MHz", "Hz"]

    def __post_init__(self):
        """Convert input array to Hz."""
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
        """Convert to WavelengthArray (m)."""
        return WavelengthArray(C_MS / self.value, "m")

    def to_omega(self) -> AngularFrequencyArray:
        """Convert to AngularFrequencyArray (rad/s)."""
        return AngularFrequencyArray(2 * PI * self.as_Hz, "rad/s")

    def to_wn(self) -> WavenumberArray:
        """Convert to WavenumberArray (1/m)."""
        return WavenumberArray(value=self.as_Hz / C_MS, unit="1/m")

    def to_equally_spaced(self, points=51) -> NDArray:
        """Return equally-spaced frequency values between min and max."""
        _min = self.as_Hz.min()
        _max = self.as_Hz.max()
        return np.linspace(_min, _max, points)

    @classmethod
    def from_meep(cls, value: float | NDArray, base_length: Wavelength | None = None) -> Self:
        """Create from MEEP frequency units.

        In MEEP, c = 1, so f_Meep = ν·a.
        """
        if base_length is None:
            base_length = Wavelength(1.0, "um")
        return cls(value / base_length.as_m, "Hz")

    @cached_property
    def as_meep(self) -> NDArray:
        """Convert to MEEP units (λ₀ = 1 μm)."""
        return self.as_Hz * 1e-6 / C_MS


@dataclass(config={"arbitrary_types_allowed": True})
class AngularFrequencyArray:
    """Array of angular frequencies stored internally in rad/s."""

    value: ArrayLike
    unit: Literal["rad/s", "rad/ps"]

    def __post_init__(self) -> Self:
        """Convert input array to rad/s."""
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
        """Convert to WavelengthArray (m)."""
        return WavelengthArray((2 * PI * C_MS) / self.value, "m")

    def to_freq(self) -> FrequencyArray:
        """Convert to FrequencyArray (Hz)."""
        return FrequencyArray(self.value / (2 * PI), "Hz")

    def to_wn(self) -> WavenumberArray:
        """Convert to WavenumberArray (1/m)."""
        return WavenumberArray(value=self.as_rad_s / (2 * PI * C_MS), unit="1/m")

    def to_equally_spaced(self, points=51) -> NDArray:
        """Return equally-spaced angular frequency values between min and max."""
        _min = self.as_rad_s.min()
        _max = self.as_rad_s.max()
        return np.linspace(_min, _max, points)

    @classmethod
    def from_meep(cls, value: float | NDArray, base_length: Wavelength | None = None) -> Self:
        """Create from MEEP frequency units.

        In MEEP, c = 1, so f_Meep = ω·a/(2π).
        """
        if base_length is None:
            base_length = Wavelength(1.0, "um")
        return cls(2 * PI * value / base_length.as_m, "rad/s")

    @cached_property
    def as_meep(self) -> NDArray:
        """Convert to MEEP frequency units: f_Meep = aω/(2πc)."""
        return self.as_rad_s * 1e-6 / (2 * PI * C_MS)


@dataclass(config={"arbitrary_types_allowed": True})
class WavenumberArray:
    """Array of wavenumbers stored internally in 1/m."""

    value: ArrayLike
    unit: Literal["1/cm", "1/m"]

    def __post_init__(self):
        """Convert input array to 1/m."""
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
        """Convert to WavelengthArray (m)."""
        return WavelengthArray(value=1 / self.as_1_m, unit="m")

    def to_freq(self) -> FrequencyArray:
        """Convert to FrequencyArray (Hz)."""
        return FrequencyArray(value=C_MS * self.as_1_m, unit="Hz")

    def to_omega(self) -> AngularFrequencyArray:
        """Convert to AngularFrequencyArray (rad/s)."""
        return AngularFrequencyArray(value=C_MS * 2 * PI * self.as_1_m, unit="rad/s")

    def to_equally_spaced(self, points=51) -> NDArray:
        """Return equally-spaced wavenumber values between min and max."""
        _min = self.as_1_m.min()
        _max = self.as_1_m.max()
        return np.linspace(_min, _max, points)

    @classmethod
    def from_meep(cls, value: float | NDArray, base_length: Wavelength | None = None) -> Self:
        """Create from MEEP frequency units.

        In MEEP, c = 1, so f_Meep = k·a/(2π).
        """
        if base_length is None:
            base_length = Wavelength(1.0, "um")
        return cls(2 * PI * value / base_length.as_m, "1/m")

    @cached_property
    def as_meep(self) -> NDArray:
        """Convert to MEEP frequency units: f_Meep = ak/(2π)."""
        return self.as_1_m * 1e-6 / (2 * PI)


class Permittivity(float):
    """Electric permittivity (F/m). Wraps ``float`` with a convenience constructor."""

    @classmethod
    def from_relative(cls, relative_value: float):
        """Create from relative permittivity (ε_r): ε = ε₀ × ε_r."""
        return cls(EPS_0 * relative_value)

    def __repr__(self):
        return f"{super().__repr__()} F/m"


class Permiability(float):
    """Magnetic permeability (H/m). Wraps ``float`` with a convenience constructor."""

    @classmethod
    def from_relative(cls, relative_value: float):
        """Create from relative permeability (μ_r): μ = μ₀ × μ_r."""
        return cls(MU_0 * relative_value)

    def __repr__(self):
        return f"{super().__repr__()} H/m"
