from __future__ import annotations
from typing import Literal, Self

import numpy as np
from numpy.typing import NDArray

PI: float
C_MS: float

class Wavelength(float):
    """Represents a scalar wavelength value with unit conversion methods."""

    def __new__(cls, value: float, unit: Literal["nm", "um", "m"]) -> Self:
        """Create a new wavelength instance.

        Args:
            value: The wavelength value.
            unit: The unit of the wavelength ('nm', 'um', or 'm').

        Returns:
            A normalized Wavelength object in meters.
        """
        ...

    @property
    def as_m(self) -> float:
        """Return the wavelength in meters."""
        ...

    @property
    def as_um(self) -> float:
        """Return the wavelength in micrometers."""
        ...

    @property
    def as_nm(self) -> float:
        """Return the wavelength in nanometers."""
        ...

    def to_freq(self) -> Frequency:
        """Convert wavelength to frequency."""
        ...

    def to_omega(self) -> AngularFrequency:
        """Convert wavelength to angular frequency."""
        ...

    def to_wn(self) -> Wavenumber:
        """Convert wavelength to wavenumber."""
        ...

class Frequency(float):
    """Represents a scalar frequency value with unit conversion methods."""

    def __new__(
        cls, value: float, unit: Literal["THz", "GHz", "MHz", "Hz"]
    ) -> Self:
        """Create a new frequency instance.

        Args:
            value: The frequency value.
            unit: The unit of the frequency ('THz', 'GHz', 'MHz', or 'Hz').

        Returns:
            A normalized Frequency object in Hz.
        """
        ...

    @property
    def as_Hz(self) -> float:
        """Return the frequency in Hertz."""
        ...

    @property
    def as_THz(self) -> float:
        """Return the frequency in Terahertz."""
        ...

    @property
    def as_GHz(self) -> float:
        """Return the frequency in Gigahertz."""
        ...

    @property
    def as_MHz(self) -> float:
        """Return the frequency in Megahertz."""
        ...

    def to_wl(self) -> Wavelength:
        """Convert frequency to wavelength."""
        ...

    def to_omega(self) -> AngularFrequency:
        """Convert frequency to angular frequency."""
        ...

    def to_wn(self) -> Wavenumber:
        """Convert frequency to wavenumber."""
        ...

class AngularFrequency(float):
    """Represents a scalar angular frequency value with unit conversion methods."""

    def __new__(
        cls, value: float, unit: Literal["rad/s", "rad/ps"]
    ) -> Self:
        """Create a new angular frequency instance.

        Args:
            value: The angular frequency value.
            unit: The unit of angular frequency ('rad/s' or 'rad/ps').

        Returns:
            A normalized AngularFrequency object in rad/s.
        """
        ...

    def __repr__(self) -> str: ...

    @property
    def as_rad_s(self) -> float:
        """Return the angular frequency in rad/s."""
        ...

    @property
    def as_rad_ps(self) -> float:
        """Return the angular frequency in rad/ps."""
        ...

    def to_wl(self) -> Wavelength:
        """Convert angular frequency to wavelength."""
        ...

    def to_freq(self) -> Frequency:
        """Convert angular frequency to frequency."""
        ...

    def to_wn(self) -> Wavenumber:
        """Convert angular frequency to wavenumber."""
        ...

class Wavenumber(float):
    """Represents a scalar wavenumber value with unit conversion methods."""

    def __new__(cls, value: float, unit: Literal["1/cm", "1/m"]) -> Self:
        """Create a new wavenumber instance.

        Args:
            value: The wavenumber value.
            unit: The unit of the wavenumber ('1/cm' or '1/m').

        Returns:
            A normalized Wavenumber object in 1/m.
        """
        ...

    @property
    def as_1_m(self) -> float:
        """Return the wavenumber in 1/m."""
        ...

    @property
    def as_1_cm(self) -> float:
        """Return the wavenumber in 1/cm."""
        ...

    @property
    def as_angular(self) -> float:
        """Return the angular wavenumber (k = 2π/λ)."""
        ...

    def to_wl(self) -> Wavelength:
        """Convert wavenumber to wavelength."""
        ...

    def to_freq(self) -> Frequency:
        """Convert wavenumber to frequency."""
        ...

    def to_omega(self) -> AngularFrequency:
        """Convert wavenumber to angular frequency."""
        ...

class WavelengthArray(np.ndarray):
    """Numpy array wrapper for multiple wavelength values with unit conversions."""

    def __new__(cls, value: NDArray, unit: Literal["nm", "um", "m"]) -> Self:
        """Create a new WavelengthArray instance.

        Args:
            value: An array of wavelength values.
            unit: The unit of each wavelength ('nm', 'um', or 'm').

        Returns:
            A WavelengthArray object with values in meters.
        """
        ...

    def __array_finalize__(self, obj) -> None: ...

    @property
    def as_m(self) -> NDArray:
        """Return the wavelengths in meters."""
        ...

    @property
    def as_um(self) -> NDArray:
        """Return the wavelengths in micrometers."""
        ...

    @property
    def as_nm(self) -> NDArray:
        """Return the wavelengths in nanometers."""
        ...

    def to_freq(self) -> FrequencyArray:
        """Convert wavelengths to frequency array."""
        ...

    def to_omega(self) -> AngularFrequencyArray:
        """Convert wavelengths to angular frequency array."""
        ...

    def to_wn(self) -> WavenumberArray:
        """Convert wavelengths to wavenumber array."""
        ...

    def to_equally_spaced(self, points: int = 51) -> NDArray:
        """Convert to equally spaced array."""
        ...

class FrequencyArray(np.ndarray):
    """Numpy array wrapper for multiple frequency values with unit conversions."""

    def __new__(
        cls, value: NDArray, unit: Literal["THz", "GHz", "MHz", "Hz"]
    ) -> Self:
        """Create a new FrequencyArray instance.

        Args:
            value: An array of frequency values.
            unit: The unit of each frequency ('THz', 'GHz', 'MHz', or 'Hz').

        Returns:
            A FrequencyArray object with values in Hz.
        """
        ...

    def __array_finalize__(self, obj) -> None: ...

    @property
    def as_Hz(self) -> NDArray:
        """Return the frequencies in Hz."""
        ...

    @property
    def as_THz(self) -> NDArray:
        """Return the frequencies in THz."""
        ...

    @property
    def as_GHz(self) -> NDArray:
        """Return the frequencies in GHz."""
        ...

    @property
    def as_MHz(self) -> NDArray:
        """Return the frequencies in MHz."""
        ...

    def to_wl(self) -> WavelengthArray:
        """Convert frequencies to wavelength array."""
        ...

    def to_omega(self) -> AngularFrequencyArray:
        """Convert frequencies to angular frequency array."""
        ...

    def to_wn(self) -> WavenumberArray:
        """Convert frequencies to wavenumber array."""
        ...

    def to_equally_spaced(self, points: int = 51) -> NDArray:
        """Convert to equally spaced array."""
        ...

class AngularFrequencyArray(np.ndarray):
    """Numpy array wrapper for multiple angular frequency values with unit conversions."""

    def __new__(
        cls, value: NDArray, unit: Literal["rad/s", "rad/ps"]
    ) -> Self:
        """Create a new AngularFrequencyArray instance.

        Args:
            value: An array of angular frequency values.
            unit: The unit of each value ('rad/s' or 'rad/ps').

        Returns:
            An AngularFrequencyArray object with values in rad/s.
        """
        ...

    def __array_finalize__(self, obj) -> None: ...

    @property
    def as_rad_s(self) -> NDArray:
        """Return the angular frequencies in rad/s."""
        ...

    @property
    def as_rad_ps(self) -> NDArray:
        """Return the angular frequencies in rad/ps."""
        ...

    def to_wl(self) -> WavelengthArray:
        """Convert angular frequencies to wavelength array."""
        ...

    def to_freq(self) -> FrequencyArray:
        """Convert angular frequencies to frequency array."""
        ...

    def to_wn(self) -> WavenumberArray:
        """Convert angular frequencies to wavenumber array."""
        ...

    def to_equally_spaced(self, points: int = 51) -> NDArray:
        """Convert to equally spaced array."""
        ...

class WavenumberArray(np.ndarray):
    """Numpy array wrapper for multiple wavenumber values with unit conversions."""

    def __new__(cls, value: NDArray, unit: Literal["1/cm", "1/m"]) -> Self:
        """Create a new WavenumberArray instance.

        Args:
            value: An array of wavenumber values.
            unit: The unit of each wavenumber ('1/cm' or '1/m').

        Returns:
            A WavenumberArray object with values in 1/m.
        """
        ...

    @property
    def as_1_m(self) -> NDArray:
        """Return the wavenumbers in 1/m."""
        ...

    @property
    def as_1_cm(self) -> NDArray:
        """Return the wavenumbers in 1/cm."""
        ...

    @property
    def as_angular(self) -> NDArray:
        """Return the angular wavenumbers (k = 2π/λ)."""
        ...

    def to_wl(self) -> WavelengthArray:
        """Convert wavenumbers to wavelength array."""
        ...

    def to_freq(self) -> FrequencyArray:
        """Convert wavenumbers to frequency array."""
        ...

    def to_omega(self) -> AngularFrequencyArray:
        """Convert wavenumbers to angular frequency array."""
        ...

    def to_equally_spaced(self, points: int = 51) -> NDArray:
        """Convert to equally spaced array."""
        ...