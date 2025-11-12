from rich.console import Console

console: Console

def c_error(msg: str) -> None:
    """
    Prints an error message to the console in red.

    Args:
        msg: The error message to print.
    """
    ...

def c_info(msg: str) -> None:
    """
    Prints an informational message to the console in blue.

    Args:
        msg: The informational message to print.
    """
    ...
