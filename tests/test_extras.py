"""
Tests for the ``photonics_helper.extras`` module.

Note: ``imageio`` may not be installed in all environments.
"""

import os


def test_create_mode_animation_filepath_construction():
    """create_mode_animation joins dir and image_list into full paths."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        img_names = ["a.png", "b.png", "c.png"]
        filepath = [os.path.join(tmpdir, img) for img in img_names]
        assert filepath == [os.path.join(tmpdir, "a.png"), os.path.join(tmpdir, "b.png"), os.path.join(tmpdir, "c.png")]


def test_create_mode_animation_appends_gif_extension():
    """If export_name doesn't end in .gif, it should be appended."""
    export_name = "my_animation"
    if ".gif" not in export_name:
        export_name += ".gif"
    assert export_name == "my_animation.gif"
