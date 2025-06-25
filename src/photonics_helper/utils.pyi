from typing import Literal, TypeVar, Union, overload
from rich.console import Console
from numpy.typing import NDArray

console: Console

def c_error(msg: str) -> None:
    """Rich color output: Red (error)"""
    ...

def c_info(msg: str) -> None:
    """Rich color output: Blue (info)"""
    ...

def c_help(msg: str) -> None:
    """Rich color output: Green (help)"""
    ...

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
    """Converts lengths of one units to other

    Args:
        value: length value
        from_units: units which user wants to change from
        to_units: units which user wants to change to

    Return:
        Returns a float or NDArray based on the input type.

    """
    ...

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
    """Converts time from one unit to another

    Args:
        value: time value
        from_units: units which user wants to change from
        to_units: units which user wants to change to

    Return:
        Returns a float or NDArray based on the input type.

    """
    ...

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
    """Converts frequency from one unit to another

    Args:
        value: frequency value
        from_units: units which user wants to change from
        to_units: units which user wants to change to

    Return:
        Returns a float or NDArray based on the input type.

    """
    ...

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
    """Converts energy from one unit to another

    Args:
        value: energy value
        from_units: units which user wants to change from
        to_units: units which user wants to change to

    Return:
        Returns a float or NDArray based on the input type.

    """
    ...

def main() -> None:
    """Main function for testing the conversion functions"""
    ...
