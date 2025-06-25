from photonics_helper.utils import c_help, c_info, convert_frequency, convert_time
from .base import Wavelength
from typing import Literal


class Pulse:
    def __init__(
        self,
        duration: float,
        duration_unit: Literal["fs", "ps", "ns", "s"],
        peak_power: float,
        central_wavelength: Wavelength | None = None,
        energy: float | None = None,
    ) -> None:
        self._duration = convert_time(duration, duration_unit, "s")
        self._peak_power = peak_power
        self._central_wavelength = central_wavelength
        self._energy = energy
        self._period: float = 2 * self._duration
        self._rate: float = 1 / (2 * self._duration)

    @property
    def duration(self) -> float:
        return self._duration

    @duration.setter
    def duration(self, value: float) -> None:
        self._duration = value

    @property
    def peak_power(self) -> float:
        return self._peak_power

    @peak_power.setter
    def peak_power(self, value: float) -> None:
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

    def set_period(self, value: float, unit: Literal["fs", "ps", "ns", "s"]) -> None:
        self._period = convert_time(value, unit, "s")
        self._rate = 1 / self._period

    @property
    def rate(self) -> float:
        if self._period == 2 * self._duration:
            c_info("Pulse rate of pules is considered as 1/(2* duration)")
            c_help("To change pulse rate and period use setters...")
        return self._rate

    def set_rate(self, value: float, unit: Literal["THz", "GHz", "MHz", "Hz"]) -> None:
        self._rate = convert_frequency(value, unit, "Hz")
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
