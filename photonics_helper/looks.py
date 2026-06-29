"""Console logging helpers using Rich."""

from rich.console import Console

console = Console()


def c_error(msg: str):
    """Print an error message in red."""
    console.print(f"[bold red]:x: {msg}[/bold red]")


def c_info(msg: str):
    """Print an info message in blue."""
    console.print(f"[bold blue]{msg}[/bold blue]")
