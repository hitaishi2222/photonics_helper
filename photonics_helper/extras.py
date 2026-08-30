"""Extra utility functions for photonics_helper."""

from pathlib import Path
from typing import List
import imageio.v2 as io


def create_mode_animation(
    export_name: str, export_dir: str | Path, image_list: List[str | Path]
):
    """Save a list of images as an animated GIF.

    Parameters
    ----------
    export_name : output filename (appends ``.gif`` if not already present).
    export_dir : directory to save the GIF into.
    image_list : list of image filenames to include.
    """

    export_dir = Path(export_dir).resolve()
    filepath = []
    for img in image_list:
        p = Path(img)
        if not p.is_absolute():
            p = export_dir / p
        p = p.resolve()
        if not str(p).startswith(str(export_dir)):
            raise ValueError(f"Image path {img} escapes export directory")
        filepath.append(p)

    img = [io.imread(file) for file in filepath]
    if ".gif" not in export_name:
        export_name += ".gif"

    io.mimsave(str(export_dir / export_name), img, format="gif", duration=0.2)  # type: ignore[call-overload]

