from typing import Literal, TypeVar, Union, overload
from rich.console import Console
from numpy.typing import NDArray

console = Console()

def c_error(msg: str):
    """Rich color output: Red (error)"""
    ...

def c_info(msg: str):
    """Rich color output: Blue (info)"""
    ...

def c_help(msg: str):
    """Rich color output: Green (green)"""
    ...

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
    from_units: Literal["s", "ps", "fs", "os"],
    to_units: Literal["s", "ps", "fs", "os"],
) -> float: ...
@overload
def convert_time(
    value: NDArray,
    from_units: Literal["s", "ps", "fs", "os"],
    to_units: Literal["s", "ps", "fs", "os"],
) -> NDArray: ...
def convert_time(
    value: Union[float, NDArray],
    from_units: Literal["s", "ps", "fs", "os"],
    to_units: Literal["s", "ps", "fs", "os"],
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
