from typing import Literal, TypeVar, Union, overload
from rich.console import Console
from numpy.typing import NDArray
import numpy as np

console = Console()


def c_error(msg: str):
    console.print(f"[bold red]:x: {msg}[/bold red]")


def c_info(msg: str):
    console.print(f"[bold blue]{msg}[/bold blue]")


def c_help(msg: str):
    console.print(f"[bold green]{msg}[/bold green]")


T = TypeVar("T", float, NDArray)


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

    if isinstance(length, float):
        return float(length)
    elif isinstance(length, np.ndarray):
        return np.asarray(length)
    else:
        raise TypeError("value should be a type of either: float or NDArray")


@overload
def convert_time(
    value: float,
    from_units: Literal["s", "ps", "fs"],
    to_units: Literal["s", "ps", "fs"],
) -> float: ...
@overload
def convert_time(
    value: NDArray,
    from_units: Literal["s", "ps", "fs"],
    to_units: Literal["s", "ps", "fs"],
) -> NDArray: ...
def convert_time(
    value: Union[float, NDArray],
    from_units: Literal["s", "ps", "fs"],
    to_units: Literal["s", "ps", "fs"],
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
                case "ps":
                    time = value * 1e12
                case "fs":
                    time = value * 1e15
        case "ps":
            match to_units:
                case "s":
                    time = value * 1e-12
                case "ps":
                    time = value
                case "fs":
                    time = value * 1e3
        case "fs":
            match to_units:
                case "s":
                    time = value * 1e-15
                case "ps":
                    time = value * 1e-3
                case "fs":
                    time = value

    if isinstance(time, float):
        return float(time)
    elif isinstance(time, np.ndarray):
        return np.asarray(time)
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

    if isinstance(frequency, float):
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
    energy = value
    match from_units:
        case "J":
            match to_units:
                case "J":
                    energy = value
                case "eV":
                    energy = value / 1.602176634e-19
        case "eV":
            match to_units:
                case "J":
                    energy = value * 1.602176634e-19
                case "eV":
                    energy = value

    if isinstance(energy, float):
        return float(energy)
    elif isinstance(energy, np.ndarray):
        return np.asarray(energy)
    else:
        raise TypeError("value should be a type of either: float or NDArray")


def main():
    t1 = convert_time(100, "fs", "s")
    print(t1, type(t1))

    t2 = convert_time(np.array([50e-12, 100e-12]), "s", "ps")
    print(t2, type(t2))


if __name__ == "__main__":
    main()
