"""Compat facade for the historical ``RamanDatabase`` import path.

The SQLite materials-database backend moved to the foundation layer as
:mod:`~photonics_helper.core.data` (class ``MaterialsDatabase``) — the
bundled ``materials.db`` is a *materials* database, not a Raman one.
This module keeps every existing import path working:

    from photonics_helper.raman.db import RamanDatabase   # still valid
    from photonics_helper.core.data import MaterialsDatabase  # canonical
"""

from __future__ import annotations

from ..core.data import MaterialsDatabase

# Historical name kept as an alias (identical class object — not a
# deprecated shim, since no rename is involved for public use).
RamanDatabase = MaterialsDatabase

__all__ = ["RamanDatabase", "MaterialsDatabase"]
