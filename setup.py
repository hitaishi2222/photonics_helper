"""Legacy shim.

All packaging metadata lives in ``pyproject.toml`` (PEP 621). This file only
exists so tools that expect a ``setup.py`` still find one; it intentionally
passes no arguments, so setuptools reads the authoritative configuration from
``pyproject.toml``. Do not add metadata here.
"""

from setuptools import setup

setup()
