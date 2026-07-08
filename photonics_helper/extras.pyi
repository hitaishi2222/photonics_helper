from pathlib import Path
from typing import List

def create_mode_animation(
    export_name: str, export_dir: str | Path, image_list: List[str | Path]
) -> None:
    """
    Create a GIF animation from a list of images.

    Args:
        export_name: The name of the output GIF file. If it doesn't
                     end with '.gif', the extension will be added.
        export_dir: The directory to save the GIF into.
        image_list: A list of image file names (paths relative to export_dir or absolute).
    """
    ...
