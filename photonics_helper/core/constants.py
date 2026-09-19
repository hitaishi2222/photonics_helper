"""Physical constants — a foundation primitive.

Re-exports the constants from :mod:`photonics_helper.base` under a stable
namespace. Values come from :mod:`scipy.constants` (CODATA).
"""

from __future__ import annotations

from ..base import C_MS, EPS_0, H_PLANCK, HBAR, MU_0, PI, Z0

__all__ = [
    "C_MS",
    "EPS_0",
    "H_PLANCK",
    "HBAR",
    "MU_0",
    "PI",
    "Z0",
]
