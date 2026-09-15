"""Import-surface regression guard for the split ``photonics_helper.raman`` subpackage.

See openspec change `split-raman-subpackage`. The refactor moved the monolithic
``raman.py`` into ``raman/`` but must keep every public import path working.
"""

import subprocess
import sys
import textwrap

import photonics_helper
import photonics_helper.raman as raman


def test_all_names_resolve():
    """Every name in ``raman.__all__`` exists on the module."""
    assert raman.__all__
    for name in raman.__all__:
        assert hasattr(raman, name), f"photonics_helper.raman.{name} is missing"


def test_legacy_import_paths():
    """The documented import paths still resolve."""
    from photonics_helper.raman import (  # noqa: F401
        COMMON_COMPARISONS,
        MaterialComparison,
        PumpWavelengthExplorer,
        RAMAN_MATERIALS,
        RamanDatabase,
        RamanFrequencyResponse,
        RamanPulseInteraction,
        RamanResponse,
        RamanSpec,
        THORLABS_SUBSTRATE_MATERIALS,
        app,
    )


def test_top_level_reexports_are_identical():
    """Top-level re-exports are the same objects as the subpackage ones."""
    # Names that were exported from the top-level package before the split.
    previously_exported = {
        "RamanSpec",
        "RamanDatabase",
        "RamanResponse",
        "RamanFrequencyResponse",
        "RamanPulseInteraction",
        "PumpWavelengthExplorer",
        "MaterialComparison",
        "RAMAN_MATERIALS",
        "COMMON_COMPARISONS",
        "app",
    }
    for name in previously_exported:
        assert hasattr(photonics_helper, name), name
        assert getattr(photonics_helper, name) is getattr(raman, name)

    # Any other subpackage name that is also at the top level must be identical.
    for name in raman.__all__:
        if hasattr(photonics_helper, name):
            assert getattr(photonics_helper, name) is getattr(raman, name)


def test_submodules_are_importable_individually():
    """Responsibility modules exist and are directly importable."""
    import importlib

    for mod in ("reference", "spec", "db", "response", "explorer", "dashboard"):
        importlib.import_module(f"photonics_helper.raman.{mod}")


def test_optional_backends_are_not_required():
    """Importing the package works even if dash/plotly cannot be imported."""
    code = textwrap.dedent(
        """
        import builtins

        _real_import = builtins.__import__

        def _blocking_import(name, *args, **kwargs):
            if name.split(".")[0] in {"dash", "plotly"}:
                raise ImportError(f"blocked {name}")
            return _real_import(name, *args, **kwargs)

        builtins.__import__ = _blocking_import
        import photonics_helper.raman as raman
        assert raman.RamanSpec is not None
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_no_sqlite_logic_in_package_init():
    """``__init__`` is re-export wiring only."""
    import inspect

    source = inspect.getsource(raman)
    assert "CREATE TABLE" not in source
    assert "sqlite3.connect" not in source
