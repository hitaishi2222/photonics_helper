from typing import Literal, Union
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


def convert_length(
    value: Union[float, NDArray],
    from_units: Literal["m", "um", "nm"],
    to_units: Literal["m", "um", "nm"],
) -> Union[float, NDArray]:
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


def convert_time(
    value: float | NDArray,
    from_units: Literal["s", "ps", "fs", "os"],
    to_units: Literal["s", "ps", "fs", "os"],
):
    time: Union[float, NDArray] = value
    match from_units:
        case "s":
            match "s":



def main():
    wl = convert_length(200, "nm", "m")
    print(wl, type(wl))

    wl2 = convert_length(np.array([0.2, 0.3, 0.4]), "um", "nm")
    print(wl2, type(wl2))


if __name__ == "__main__":
    main()
