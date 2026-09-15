"""Raman scattering subpackage.

Public re-export surface preserving the historical
``photonics_helper.raman.<Name>`` import paths. The implementation is split by
responsibility:

- :mod:`~photonics_helper.raman.reference` — hardcoded material tables
- :mod:`~photonics_helper.raman.spec` — :class:`RamanSpec`
- :mod:`~photonics_helper.raman.db` — :class:`RamanDatabase`
- :mod:`~photonics_helper.raman.response` — time/frequency response physics
- :mod:`~photonics_helper.raman.explorer` — explorer and comparison helpers
- :mod:`~photonics_helper.raman.dashboard` — Dash app factory
"""

from __future__ import annotations

from .dashboard import app
from .db import RamanDatabase
from .explorer import (
    COMMON_COMPARISONS,
    MaterialComparison,
    PumpWavelengthExplorer,
)
from .reference import RAMAN_MATERIALS, THORLABS_SUBSTRATE_MATERIALS
from .response import (
    RamanFrequencyResponse,
    RamanPulseInteraction,
    RamanResponse,
)
from .spec import RamanSpec

__all__ = [
    "RamanSpec",
    "RamanDatabase",
    "RamanResponse",
    "RamanFrequencyResponse",
    "RamanPulseInteraction",
    "PumpWavelengthExplorer",
    "MaterialComparison",
    "COMMON_COMPARISONS",
    "RAMAN_MATERIALS",
    "THORLABS_SUBSTRATE_MATERIALS",
    "app",
]
