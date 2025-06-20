from rich.console import Console

console = Console()


def c_error(msg: str):
    console.print(f"[bold red]:x: {msg}[/bold red]")


def c_info(msg: str):
    console.print(f"[bold blue]{msg}[/bold blue]")


def c_help(msg: str):
    console.print(f"[bold green]{msg}[/bold green]")
