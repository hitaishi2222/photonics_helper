"""Physical units and constants for photonics calculations."""

from __future__ import annotations
from typing import Literal, Self
from numpy.typing import NDArray
from functools import cached_property

import numpy as np
import scipy.constants as const
from rich.traceback import install
from pydantic.dataclasses import dataclass

install()

# Constants
PI: float = const.pi
C_MS: float = const.c
EPS_0: float = const.epsilon_0
MU_0: float = const.mu_0
H_PLANCK: float = const.h
HBAR: float = const.hbar


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

    def to_energy(self) -> Energy:
        return Energy(H_PLANCK * C_MS / self.value, "J")

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

    def to_time(self) -> Time:
        return Time(1.0 / self.as_Hz, "s")

    def to_energy(self) -> Energy:
        return Energy(H_PLANCK * self.as_Hz, "J")

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

    def to_time(self) -> Time:
        return Time(2 * PI / self.value, "s")

    def to_energy(self) -> Energy:
        return Energy(HBAR * self.value, "J")

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

    def to_energy(self) -> Energy:
        return Energy(H_PLANCK * C_MS * self.as_1_m, "J")

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
class Length:
    """Length stored internally in meters.

    Accepts km, m, cm, mm, um, nm, or pm on construction; converts to meters in ``__post_init__``.
    """

    value: float
    unit: Literal["km", "m", "cm", "mm", "um", "nm", "pm"]

    def __post_init__(self):
        """Convert input value to meters."""
        if self.unit == "km":
            self.value *= 1e3
        elif self.unit == "m":
            pass
        elif self.unit == "cm":
            self.value *= 1e-2
        elif self.unit == "mm":
            self.value *= 1e-3
        elif self.unit == "um":
            self.value *= 1e-6
        elif self.unit == "nm":
            self.value *= 1e-9
        elif self.unit == "pm":
            self.value *= 1e-12
        else:
            raise ValueError(f"Unsupported unit: {self.unit}")

    def __repr__(self) -> str:
        return f"Length -> {self.as_m} m"

    def __str__(self) -> str:
        if self.as_km >= 0.1:
            return f"{self.as_km:.3f} km"
        if self.as_m >= 0.1:
            return f"{self.as_m:.3f} m"
        if self.as_um >= 0.1:
            return f"{self.as_um:.3f} um"
        return f"{self.as_nm:.2f} nm"

    @cached_property
    def as_km(self) -> float:
        return self.value * 1e-3

    @cached_property
    def as_m(self) -> float:
        return self.value

    @cached_property
    def as_cm(self) -> float:
        return self.value * 1e2

    @cached_property
    def as_mm(self) -> float:
        return self.value * 1e3

    @cached_property
    def as_um(self) -> float:
        return self.value * 1e6

    @cached_property
    def as_nm(self) -> float:
        return self.value * 1e9

    @cached_property
    def as_pm(self) -> float:
        return self.value * 1e12

    def to_wl(self) -> Wavelength:
        return Wavelength(self.value, "m")

    @classmethod
    def from_wl(cls, wl: Wavelength) -> Self:
        return cls(wl.as_m, "m")

    @classmethod
    def from_meep(cls, value: float, base_length: Wavelength | None = None) -> Self:
        if base_length is None:
            base_length = Wavelength(1.0, "um")
        return cls(value * base_length.as_m, "m")

    @cached_property
    def as_meep(self) -> float:
        return self.as_m / 1e-6


@dataclass(config={"arbitrary_types_allowed": True})
class Time:
    """Time stored internally in seconds.

    Accepts s, ms, us, ns, ps, fs, or as on construction; converts to seconds in ``__post_init__``.
    """

    value: float
    unit: Literal["s", "ms", "us", "ns", "ps", "fs", "as"]

    def __post_init__(self):
        """Convert input value to seconds."""
        if self.unit == "s":
            pass
        elif self.unit == "ms":
            self.value *= 1e-3
        elif self.unit == "us":
            self.value *= 1e-6
        elif self.unit == "ns":
            self.value *= 1e-9
        elif self.unit == "ps":
            self.value *= 1e-12
        elif self.unit == "fs":
            self.value *= 1e-15
        elif self.unit == "as":
            self.value *= 1e-18
        else:
            raise ValueError(f"Unsupported unit: {self.unit}")

    def __repr__(self) -> str:
        return f"Time -> {self.as_s} s"

    def __str__(self) -> str:
        if self.as_ns >= 1:
            return f"{self.as_ns:.3f} ns"
        if self.as_ps >= 1:
            return f"{self.as_ps:.3f} ps"
        if self.as_fs >= 1:
            return f"{self.as_fs:.2f} fs"
        return f"{self.as_fs:.2f} fs"

    @cached_property
    def as_s(self) -> float:
        return self.value

    @cached_property
    def as_ms(self) -> float:
        return self.value * 1e3

    @cached_property
    def as_us(self) -> float:
        return self.value * 1e6

    @cached_property
    def as_ns(self) -> float:
        return self.value * 1e9

    @cached_property
    def as_ps(self) -> float:
        return self.value * 1e12

    @cached_property
    def as_fs(self) -> float:
        return self.value * 1e15

    @cached_property
    def as_as(self) -> float:
        return self.value * 1e18

    def to_freq(self) -> Frequency:
        return Frequency(1.0 / self.value, "Hz")

    def to_omega(self) -> AngularFrequency:
        return AngularFrequency(2 * PI / self.value, "rad/s")

    @classmethod
    def from_freq(cls, freq: Frequency) -> Self:
        return cls(1.0 / freq.as_Hz, "s")

    @classmethod
    def from_omega(cls, omega: AngularFrequency) -> Self:
        return cls(2 * PI / omega.as_rad_s, "s")

    @classmethod
    def from_meep(cls, value: float, base_length: Wavelength | None = None) -> Self:
        if base_length is None:
            base_length = Wavelength(1.0, "um")
        return cls(value * base_length.as_m / C_MS, "s")

    @cached_property
    def as_meep(self) -> float:
        return self.as_s * C_MS / 1e-6


@dataclass(config={"arbitrary_types_allowed": True})
class Energy:
    """Energy stored internally in Joules.

    Accepts J, mJ, uJ, nJ, pJ, eV, or meV on construction; converts to Joules in ``__post_init__``.
    """

    value: float
    unit: Literal["J", "mJ", "uJ", "nJ", "pJ", "eV", "meV"]

    def __post_init__(self):
        """Convert input value to Joules."""
        if self.unit == "J":
            pass
        elif self.unit == "mJ":
            self.value *= 1e-3
        elif self.unit == "uJ":
            self.value *= 1e-6
        elif self.unit == "nJ":
            self.value *= 1e-9
        elif self.unit == "pJ":
            self.value *= 1e-12
        elif self.unit == "eV":
            self.value *= const.eV
        elif self.unit == "meV":
            self.value *= const.eV * 1e-3
        else:
            raise ValueError(f"Unsupported unit: {self.unit}")

    def __repr__(self) -> str:
        return f"Energy -> {self.as_J} J"

    def __str__(self) -> str:
        if self.as_eV >= 0.01:
            return f"{self.as_eV:.4f} eV"
        return f"{self.as_nJ:.4f} nJ"

    @cached_property
    def as_J(self) -> float:
        return self.value

    @cached_property
    def as_mJ(self) -> float:
        return self.value * 1e3

    @cached_property
    def as_uJ(self) -> float:
        return self.value * 1e6

    @cached_property
    def as_nJ(self) -> float:
        return self.value * 1e9

    @cached_property
    def as_pJ(self) -> float:
        return self.value * 1e12

    @cached_property
    def as_eV(self) -> float:
        return self.value / const.eV

    @cached_property
    def as_meV(self) -> float:
        return self.value / const.eV * 1e3

    def to_freq(self) -> Frequency:
        return Frequency(self.value / H_PLANCK, "Hz")

    def to_omega(self) -> AngularFrequency:
        return AngularFrequency(self.value / HBAR, "rad/s")

    def to_wl(self) -> Wavelength:
        return Wavelength(H_PLANCK * C_MS / self.value, "m")

    def to_wn(self) -> Wavenumber:
        return Wavenumber(value=self.value / (H_PLANCK * C_MS), unit="1/m")

    @classmethod
    def from_freq(cls, freq: Frequency) -> Self:
        return cls(H_PLANCK * freq.as_Hz, "J")

    @classmethod
    def from_omega(cls, omega: AngularFrequency) -> Self:
        return cls(HBAR * omega.as_rad_s, "J")

    @classmethod
    def from_wl(cls, wl: Wavelength) -> Self:
        return cls(H_PLANCK * C_MS / wl.as_m, "J")

    @classmethod
    def from_wn(cls, wn: Wavenumber) -> Self:
        return cls(H_PLANCK * C_MS * wn.as_1_m, "J")

    @classmethod
    def from_meep(cls, value: float, base_length: Wavelength | None = None) -> Self:
        if base_length is None:
            base_length = Wavelength(1.0, "um")
        return cls(value * H_PLANCK * C_MS / base_length.as_m, "J")

    @cached_property
    def as_meep(self) -> float:
        return self.as_J * 1e-6 / (H_PLANCK * C_MS)


@dataclass(config={"arbitrary_types_allowed": True})
class Power:
    """Power stored internally in Watts.

    Accepts kW, W, mW, uW, or nW on construction; converts to Watts in ``__post_init__``.
    """

    value: float
    unit: Literal["kW", "W", "mW", "uW", "nW"]

    def __post_init__(self):
        """Convert input value to Watts."""
        if self.unit == "kW":
            self.value *= 1e3
        elif self.unit == "W":
            pass
        elif self.unit == "mW":
            self.value *= 1e-3
        elif self.unit == "uW":
            self.value *= 1e-6
        elif self.unit == "nW":
            self.value *= 1e-9
        else:
            raise ValueError(f"Unsupported unit: {self.unit}")

    def __repr__(self) -> str:
        return f"Power -> {self.as_W} W"

    def __str__(self) -> str:
        if self.as_kW >= 0.1:
            return f"{self.as_kW:.3f} kW"
        if self.as_W >= 0.1:
            return f"{self.as_W:.3f} W"
        return f"{self.as_mW:.3f} mW"

    @cached_property
    def as_kW(self) -> float:
        return self.value * 1e-3

    @cached_property
    def as_W(self) -> float:
        return self.value

    @cached_property
    def as_mW(self) -> float:
        return self.value * 1e3

    @cached_property
    def as_uW(self) -> float:
        return self.value * 1e6

    @cached_property
    def as_nW(self) -> float:
        return self.value * 1e9

    @classmethod
    def from_meep(cls, value: float, base_length: Wavelength | None = None) -> Self:
        if base_length is None:
            base_length = Wavelength(1.0, "um")
        return cls(value * C_MS ** 3 * MU_0 / base_length.as_m ** 2, "W")

    @cached_property
    def as_meep(self) -> float:
        return self.as_W * 1e-6 / (C_MS ** 3 * MU_0)


@dataclass(config={"arbitrary_types_allowed": True})
class Area:
    """Area stored internally in square meters (m²).

    Accepts m^2, cm^2, mm^2, um^2, or nm^2 on construction; converts to m² in ``__post_init__``.
    """

    value: float
    unit: Literal["m^2", "cm^2", "mm^2", "um^2", "nm^2"]

    def __post_init__(self):
        """Convert input value to m²."""
        if self.unit == "m^2":
            pass
        elif self.unit == "cm^2":
            self.value *= 1e-4
        elif self.unit == "mm^2":
            self.value *= 1e-6
        elif self.unit == "um^2":
            self.value *= 1e-12
        elif self.unit == "nm^2":
            self.value *= 1e-18
        else:
            raise ValueError(f"Unsupported unit: {self.unit}")

    def __repr__(self) -> str:
        return f"Area -> {self.as_m2} m²"

    def __str__(self) -> str:
        if self.as_cm2 >= 0.01:
            return f"{self.as_cm2:.3f} cm²"
        if self.as_mm2 >= 0.01:
            return f"{self.as_mm2:.3f} mm²"
        return f"{self.as_um2:.2f} um²"

    @cached_property
    def as_m2(self) -> float:
        return self.value

    @cached_property
    def as_cm2(self) -> float:
        return self.value * 1e4

    @cached_property
    def as_mm2(self) -> float:
        return self.value * 1e6

    @cached_property
    def as_um2(self) -> float:
        return self.value * 1e12

    @cached_property
    def as_nm2(self) -> float:
        return self.value * 1e18

    @classmethod
    def from_meep(cls, value: float, base_length: Wavelength | None = None) -> Self:
        if base_length is None:
            base_length = Wavelength(1.0, "um")
        return cls(value * base_length.as_m ** 2, "m^2")

    @cached_property
    def as_meep(self) -> float:
        return self.as_m2 / 1e-12


@dataclass(config={"arbitrary_types_allowed": True})
class WavelengthArray:
    """Array of wavelengths stored internally in meters."""

    value: NDArray
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
    def from_meep(
        cls, value: float | NDArray, base_length: Wavelength | None = None
    ) -> Self:
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

    value: NDArray
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
    def from_meep(
        cls, value: float | NDArray, base_length: Wavelength | None = None
    ) -> Self:
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

    value: NDArray
    unit: Literal["rad/s", "rad/ps"]

    def __post_init__(self):
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
    def from_meep(
        cls, value: float | NDArray, base_length: Wavelength | None = None
    ) -> Self:
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

    value: NDArray
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
    def from_meep(
        cls, value: float | NDArray, base_length: Wavelength | None = None
    ) -> Self:
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


@dataclass(config={"arbitrary_types_allowed": True})
class Permittivity(float):
    """Electric permittivity (F/m). Wraps ``float`` with a convenience constructor."""

    value: float

    def __post_init__(self):
        object.__setattr__(self, "value", float(self.value))

    @classmethod
    def from_relative(cls, relative_value: float):
        """Create from relative permittivity (ε_r): ε = ε₀ × ε_r."""
        return cls(EPS_0 * relative_value)

    def __repr__(self):
        return f"{super().__repr__()} F/m"


@dataclass(config={"arbitrary_types_allowed": True})
class Permiability(float):
    """Magnetic permeability (H/m). Wraps ``float`` with a convenience constructor."""

    value: float

    def __post_init__(self):
        object.__setattr__(self, "value", float(self.value))

    @classmethod
    def from_relative(cls, relative_value: float):
        """Create from relative permeability (μ_r): μ = μ₀ × μ_r."""
        return cls(MU_0 * relative_value)

    def __repr__(self):
        return f"{super().__repr__()} H/m"
