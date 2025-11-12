from os import PathLike
from typing import List
import imageio.v2 as io
import os


def create_mode_animation(
    export_name: str, dir: str | PathLike, image_list: List[str | PathLike]
):

    filepath = [os.path.join(dir, img) for img in image_list]
    img = [io.imread(file) for file in filepath]
    if ".gif" not in export_name:
        export_name += ".gif"

    io.mimsave(export_name, img, format="gif", duration=0.2)
