from typing import Literal, TypeVar, Union, overload
from rich.console import Console
from numpy.typing import NDArray
import numpy as np
import scipy.constants as const

E_CHARGE: float = const.e

console = Console()


def c_error(msg: str):
    console.print(f"[bold red]:x: {msg}[/bold red]")


def c_info(msg: str):
    console.print(f"[bold blue]{msg}[/bold blue]")


def c_help(msg: str):
    console.print(f"[bold green]{msg}[/bold green]")


@overload
def convert_length(
    value: float,
    from_units: Literal["m", "um", "nm"],
    to_units: Literal["m", "um", "nm"],
) -> float: ...
@overload
def convert_length(
    value: NDArray,
    from_units: Literal["m", "um", "nm"],
    to_units: Literal["m", "um", "nm"],
) -> NDArray: ...
def convert_length(
    value: Union[float, NDArray],
    from_units: Literal["m", "um", "nm"],
    to_units: Literal["m", "um", "nm"],
) -> Union[float, NDArray]:
    if not (
        isinstance(value, float)
        or isinstance(value, int)
        or isinstance(value, np.ndarray)
    ):
        raise TypeError("value should be a type of either: float or NDArray")

    length: Union[float, NDArray] = value
    match from_units:
        case "m":
            match to_units:
                case "m":
                    length = value
                case "um":
                    length = value * 1e6
                case "nm":
                    length = value * 1e9
        case "um":
            match to_units:
                case "m":
                    length = value * 1e-6
                case "um":
                    length = value
                case "nm":
                    length = value * 1e3
        case "nm":
            match to_units:
                case "m":
                    length = value * 1e-9
                case "um":
                    length = value * 1e-3
                case "nm":
                    length = value
        case _:
            raise ValueError(
                f"Unsupported unit: {from_units} -> {to_units} use 'nm', 'um', or 'm'"
            )

    if isinstance(length, float) or isinstance(length, int):
        return float(length)
    elif isinstance(length, np.ndarray):
        return np.asarray(length)
    else:
        raise TypeError("value should be a type of either: float or NDArray")


@overload
def convert_time(
    value: float,
    from_units: Literal["s", "ns", "ps", "fs"],
    to_units: Literal["s", "ns", "ps", "fs"],
) -> float: ...
@overload
def convert_time(
    value: NDArray,
    from_units: Literal["s", "ns", "ps", "fs"],
    to_units: Literal["s", "ns", "ps", "fs"],
) -> NDArray: ...
def convert_time(
    value: Union[float, NDArray],
    from_units: Literal["s", "ns", "ps", "fs"],
    to_units: Literal["s", "ns", "ps", "fs"],
) -> Union[float, NDArray]:
    if not (
        isinstance(value, float)
        or isinstance(value, int)
        or isinstance(value, np.ndarray)
    ):
        raise TypeError("value should be a type of either: float or NDArray")

    time: Union[float, NDArray] = value
    match from_units:
        case "s":
            match to_units:
                case "s":
                    time = value
                case "ns":
                    time = value * 1e9
                case "ps":
                    time = value * 1e12
                case "fs":
                    time = value * 1e15
        case "ns":
            match to_units:
                case "s":
                    time = value * 1e-9
                case "ns":
                    time = value
                case "ps":
                    time = value * 1e3
                case "fs":
                    time = value * 1e6
        case "ps":
            match to_units:
                case "s":
                    time = value * 1e-12
                case "ns":
                    time = value * 1e-3
                case "ps":
                    time = value
                case "fs":
                    time = value * 1e3
        case "fs":
            match to_units:
                case "s":
                    time = value * 1e-15
                case "ns":
                    time = value * 1e-6
                case "ps":
                    time = value * 1e-3
                case "fs":
                    time = value
        case _:
            raise ValueError(
                f"Unsupported unit: {from_units} -> {to_units} use 's', 'ns', 'ps' or 'fs'"
            )

    if isinstance(time, float) or isinstance(time, int):
        return float(time)
    elif isinstance(time, np.ndarray):
        return np.asarray(time, dtype=np.float64)
    else:
        raise TypeError("value should be a type of either: float or NDArray")


@overload
def convert_frequency(
    value: float,
    from_units: Literal["Hz", "MHz", "GHz", "THz"],
    to_units: Literal["Hz", "MHz", "GHz", "THz"],
) -> float: ...
@overload
def convert_frequency(
    value: NDArray,
    from_units: Literal["Hz", "MHz", "GHz", "THz"],
    to_units: Literal["Hz", "MHz", "GHz", "THz"],
) -> NDArray: ...
def convert_frequency(
    value: Union[float, NDArray],
    from_units: Literal["Hz", "MHz", "GHz", "THz"],
    to_units: Literal["Hz", "MHz", "GHz", "THz"],
) -> Union[float, NDArray]:
    if not (
        isinstance(value, float)
        or isinstance(value, int)
        or isinstance(value, np.ndarray)
    ):
        raise TypeError("value should be a type of either: float or NDArray")

    frequency: Union[float, NDArray] = value
    match from_units:
        case "Hz":
            match to_units:
                case "Hz":
                    frequency = value
                case "MHz":
                    frequency = value * 1e-6
                case "GHz":
                    frequency = value * 1e-9
                case "THz":
                    frequency = value * 1e-12
        case "MHz":
            match to_units:
                case "Hz":
                    frequency = value * 1e6
                case "MHz":
                    frequency = value
                case "GHz":
                    frequency = value * 1e-3
                case "THz":
                    frequency = value * 1e-6
        case "GHz":
            match to_units:
                case "Hz":
                    frequency = value * 1e9
                case "MHz":
                    frequency = value * 1e3
                case "GHz":
                    frequency = value
                case "THz":
                    frequency = value * 1e-3
        case "THz":
            match to_units:
                case "Hz":
                    frequency = value * 1e12
                case "MHz":
                    frequency = value * 1e6
                case "GHz":
                    frequency = value * 1e3
                case "THz":
                    frequency = value
        case _:
            raise ValueError(
                f"Unsupported unit: {from_units} -> {to_units} use 'Hz', 'MHz', 'GHz' or 'THz'"
            )

    if isinstance(frequency, float) or isinstance(frequency, int):
        return float(frequency)
    elif isinstance(frequency, np.ndarray):
        return np.asarray(frequency)
    else:
        raise TypeError("value should be a type of either: float or NDArray")


@overload
def convert_energy(
    value: float, from_units: Literal["J", "eV"], to_units: Literal["J", "eV"]
) -> float: ...
@overload
def convert_energy(
    value: NDArray, from_units: Literal["J", "eV"], to_units: Literal["J", "eV"]
) -> NDArray: ...
def convert_energy(
    value: Union[float, NDArray],
    from_units: Literal["J", "eV"],
    to_units: Literal["J", "eV"],
) -> Union[float, NDArray]:
    if not (isinstance(value, float) or isinstance(value, np.ndarray)):
        raise TypeError(
            f"value should be only a type of float or NDArray: got {type(value)}."
        )
    energy = value
    match from_units:
        case "J":
            match to_units:
                case "J":
                    energy = value
                case "eV":
                    energy = value / E_CHARGE
        case "eV":
            match to_units:
                case "J":
                    energy = value * E_CHARGE
                case "eV":
                    energy = value
        case _:
            raise ValueError(
                f"Unsupported unit: {from_units} -> {to_units} use 'J', or 'eV'"
            )

    if isinstance(energy, float) or isinstance(energy, int):
        return float(energy)
    elif isinstance(energy, np.ndarray):
        return np.asarray(energy, dtype=np.float64)
    else:
        raise TypeError("value should be a type of either: float or NDArray")


@overload
def convert_angular_frequency(
    value: float,
    from_units: Literal["rad/s", "rad/ps"],
    to_units: Literal["rad/s", "rad/ps"],
) -> float: ...
@overload
def convert_angular_frequency(
    value: NDArray,
    from_units: Literal["rad/s", "rad/ps"],
    to_units: Literal["rad/s", "rad/ps"],
) -> NDArray: ...
def convert_angular_frequency(
    value: Union[float, NDArray],
    from_units: Literal["rad/s", "rad/ps"],
    to_units: Literal["rad/s", "rad/ps"],
) -> Union[float, NDArray]:
    if not (
        isinstance(value, float)
        or isinstance(value, int)
        or isinstance(value, np.ndarray)
    ):
        raise TypeError("value should be a type of either: float or NDArray")

    angular_frequency: Union[float, NDArray] = value
    match from_units:
        case "rad/s":
            match to_units:
                case "rad/s":
                    angular_frequency = value
                case "rad/ps":
                    angular_frequency = value * 1e-12
        case "rad/ps":
            match to_units:
                case "rad/s":
                    angular_frequency = value * 1e12
                case "rad/ps":
                    angular_frequency = value
        case _:
            raise ValueError(
                f"Unsupported unit: {from_units} -> {to_units} use 'rad/s' or 'rad/ps'"
            )

    if isinstance(angular_frequency, float) or isinstance(angular_frequency, int):
        return float(angular_frequency)
    elif isinstance(angular_frequency, np.ndarray):
        return np.asarray(angular_frequency)
    else:
        raise TypeError("value should be a type of either: float or NDArray")


@overload
def convert_wavenumber(
    value: float,
    from_units: Literal["1/m", "1/cm"],
    to_units: Literal["1/m", "1/cm"],
) -> float: ...
@overload
def convert_wavenumber(
    value: NDArray,
    from_units: Literal["1/m", "1/cm"],
    to_units: Literal["1/m", "1/cm"],
) -> NDArray: ...
def convert_wavenumber(
    value: Union[float, NDArray],
    from_units: Literal["1/m", "1/cm"],
    to_units: Literal["1/m", "1/cm"],
) -> Union[float, NDArray]:
    if not (
        isinstance(value, float)
        or isinstance(value, int)
        or isinstance(value, np.ndarray)
    ):
        raise TypeError("value should be a type of either: float or NDArray")

    wavenumber: Union[float, NDArray] = value
    match from_units:
        case "1/m":
            match to_units:
                case "1/m":
                    wavenumber = value
                case "1/cm":
                    wavenumber = value * 1e-2
        case "1/cm":
            match to_units:
                case "1/m":
                    wavenumber = value * 1e2
                case "1/cm":
                    wavenumber = value
        case _:
            raise ValueError(
                f"Unsupported unit: {from_units} -> {to_units} use '1/m' or '1/cm'"
            )

    if isinstance(wavenumber, float) or isinstance(wavenumber, int):
        return float(wavenumber)
    elif isinstance(wavenumber, np.ndarray):
        return np.asarray(wavenumber)
    else:
        raise TypeError("value should be a type of either: float or NDArray")


def main():
    t1 = convert_time(100, "fs", "s")
    print(t1, type(t1))

    t2 = convert_time(np.array([50e-12, 100e-12]), "s", "ps")
    print(t2, type(t2))


if __name__ == "__main__":
    main()
