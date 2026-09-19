"""Foundation primitives: units, constants, grids and materials.

This is the documented, dependency-light core of ``photonics_helper`` — the
layer other projects are meant to build on. Importing it pulls in only numpy,
scipy and pydantic; the plotting/web/simulation stack (matplotlib, plotly,
dash) and the solver satellites are not loaded.

The public surface is deliberately small and stable::

    from photonics_helper.core import units, constants, grids, materials

Existing import paths (``photonics_helper.base``, ``photonics_helper.pulse``,
…) remain fully supported; this namespace names the foundation explicitly
rather than replacing anything.

See ``REPORT.md`` ("Next Roadmap — Foundation Backbone") for the boundary
between this core and the satellite modules.
"""

from __future__ import annotations

from . import constants, grids, materials, units

__all__ = ["constants", "grids", "materials", "units"]
