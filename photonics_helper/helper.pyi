from __future__ import annotations
from typing import Literal

import numpy as np
from numpy.typing import NDArray

PI: float
C_MS: float

class Wavelength(float):
    """Represents a scalar wavelength value with unit conversion methods."""

    def __new__(cls, value: float, unit: Literal["nm", "um", "m"] = "nm") -> Wavelength:
        """Create a new wavelength instance.

        Args:
            value: The wavelength value.
            unit: The unit of the wavelength ('nm', 'um', or 'm').

        Returns:
            A normalized Wavelength object in meters.
        """
        ...

    def __repr__(self) -> str: ...
    @property
    def as_m(self) -> float:
        """Return the wavelength in meters."""
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

class Frequency(float):
    """Represents a scalar frequency value with unit conversion methods."""

    def __new__(
        cls, value: float, unit: Literal["THz", "GHz", "MHz", "Hz"] = "Hz"
    ) -> Frequency:
        """Create a new frequency instance.

        Args:
            value: The frequency value.
            unit: The unit of the frequency ('THz', 'GHz', 'MHz', or 'Hz').

        Returns:
            A normalized Frequency object in Hz.
        """
        ...

    def __repr__(self) -> str: ...
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

class AngularFrequency(float):
    """Represents a scalar angular frequency value with unit conversion methods."""

    def __new__(
        cls, value: float, unit: Literal["rad/s", "rad/ps"] = "rad/s"
    ) -> AngularFrequency:
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

class WavelengthArray(np.ndarray):
    """Numpy array wrapper for multiple wavelength values with unit conversions."""

    def __new__(
        cls, value: NDArray, unit: Literal["nm", "um", "m"] = "nm"
    ) -> WavelengthArray:
        """Create a new WavelengthArray instance.

        Args:
            value: An array of wavelength values.
            unit: The unit of each wavelength ('nm', 'um', or 'm').

        Returns:
            A WavelengthArray object with values in meters.
        """
        ...

    def __repr__(self) -> str: ...
    @property
    def as_m(self) -> NDArray:
        """Return the wavelengths in meters."""
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

class FrequencyArray(np.ndarray):
    """Numpy array wrapper for multiple frequency values with unit conversions."""

    def __new__(
        cls, value: NDArray, unit: Literal["THz", "GHz", "MHz", "Hz"] = "Hz"
    ) -> FrequencyArray:
        """Create a new FrequencyArray instance.

        Args:
            value: An array of frequency values.
            unit: The unit of each frequency ('THz', 'GHz', 'MHz', or 'Hz').

        Returns:
            A FrequencyArray object with values in Hz.
        """
        ...

    def __repr__(self) -> str: ...
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

class AngularFrequencyArray(np.ndarray):
    """Numpy array wrapper for multiple angular frequency values with unit conversions."""

    def __new__(
        cls, value: NDArray, unit: Literal["rad/s", "rad/ps"] = "rad/s"
    ) -> AngularFrequencyArray:
        """Create a new AngularFrequencyArray instance.

        Args:
            value: An array of angular frequency values.
            unit: The unit of each value ('rad/s' or 'rad/ps').

        Returns:
            An AngularFrequencyArray object with values in rad/s.
        """
        ...

    def __repr__(self) -> str: ...
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
