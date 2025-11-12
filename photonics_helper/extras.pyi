from os import PathLike
from typing import List

def create_mode_animation(
    export_name: str, dir: str | PathLike, image_list: List[str | PathLike]
) -> None:
    """
    Create a GIF animation from a list of images.

    Args:
        export_name: The name of the output GIF file. If it doesn't
                     end with '.gif', the extension will be added.
        dir: The directory where the images are located.
        image_list: A list of image file names.
    """
    ...
