from photonics_helper.looks import c_help, c_info
from .base import Wavelength
from typing import Literal, Optional


class Pulse:
    def __init__(
        self,
        duration: float,
        duration_unit: Literal["fs", "ps", "ns", "s", "os"],
        peak_power: float,
        central_wavelength: Wavelength | None = None,
        energy: float | None = None,
    ) -> None:
        self._duration = duration
        match duration_unit:
            case "s":
                self._duration = duration
            case "ns":
                self._duration = duration * 1e-9
            case "ps":
                self._duration = duration * 1e-12
            case "fs":
                self._duration = duration * 1e-15
            case "os":
                self._duration = duration * 1e-18
            case _:
                raise AttributeError(
                    "Pulse duration units should be only from ['fs', 'ps', 'ns', 's', 'os']"
                )
        self._duration_unit = duration_unit
        self._peak_power = peak_power
        self._central_wavelength = central_wavelength
        self._energy = energy
        self._period: float = 2 * self._duration
        self._rate: float = 1 / (2 * self._duration)

    @property
    def duration(self) -> float:
        return self._duration

    @duration.setter
    def duration(self, value):
        self._duration = value

    @property
    def peak_power(self) -> float:
        return self._peak_power

    @peak_power.setter
    def peak_power(self, value):
        self._peak_power = value

    @property
    def central_wavelength(self) -> Wavelength:
        if not isinstance(self._central_wavelength, Wavelength):
            raise AttributeError(
                "Central wavelength is not defined yet.. Please define.."
            )
        return self._central_wavelength

    @property
    def period(self) -> float:
        if self._period == 2 * self._duration:
            c_info("Pulse period of pules is considered as twice as its duration.")
            c_help("To change pulse rate and period use setters...")
        return self._period

    def set_period(self, value: float, unit: Literal["fs", "ps", "ns", "s", "os"]):
        match unit:
            case "s":
                self._period = value
            case "ns":
                self._period = value * 1e-9
            case "ps":
                self._period = value * 1e-12
            case "fs":
                self._period = value * 1e-15
            case "os":
                self._period = value * 1e-18
            case _:
                raise AttributeError(
                    "Pulse duration units should be only from ['fs', 'ps', 'ns', 's', 'os']"
                )
        self._rate = 1 / self._period

    @property
    def rate(self):
        if self._period == 2 * self._duration:
            c_info("Pulse rate of pules is considered as 1/(2* duration)")
            c_help("To change pulse rate and period use setters...")
        return self._rate

    def set_rate(self, value: float, unit: Literal["THz", "GHz", "MHz", "Hz"]):
        match unit:
            case "Hz":
                self._rate = value
            case "MHz":
                self._rate = value * 1e6
            case "GHz":
                self._rate = value * 1e9
            case "THz":
                self._rate = value * 1e12
            case _:
                raise AttributeError(
                    "Pulse rate units should be only from ['THz', 'GHz', 'MHz', 'Hz']"
                )
        self._period = 1 / self._rate


class RectangularPulse(Pulse):
    def __init__(
        self,
        duration: float,
        duration_unit: Literal["fs", "ps", "ns", "s"],
        peak_power: float,
        central_wavelength: Wavelength | None = None,
        energy: float | None = None,
        amplitude: float = 1,
    ) -> None:
        super().__init__(
            duration, duration_unit, peak_power, central_wavelength, energy
        )
        self.amplitude = amplitude
