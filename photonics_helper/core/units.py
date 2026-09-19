"""Typed unit classes — a foundation primitive.

Re-exports the unit-safe scalar/array types from
:mod:`photonics_helper.base`. This module exists so that downstream code can
type against ``photonics_helper.core.units`` as the stable foundation surface;
the implementation currently lives in ``base.py`` (moving it is deferred).

Importing this module is dependency-light: numpy, scipy and pydantic only.
"""

from __future__ import annotations

from ..base import (
    AngularFrequency,
    AngularFrequencyArray,
    Area,
    Energy,
    Frequency,
    FrequencyArray,
    Length,
    PeakPower,
    Permeability,
    Permittivity,
    Power,
    Time,
    Wavelength,
    WavelengthArray,
    Wavenumber,
    WavenumberArray,
)

__all__ = [
    "AngularFrequency",
    "AngularFrequencyArray",
    "Area",
    "Energy",
    "Frequency",
    "FrequencyArray",
    "Length",
    "PeakPower",
    "Permeability",
    "Permittivity",
    "Power",
    "Time",
    "Wavelength",
    "WavelengthArray",
    "Wavenumber",
    "WavenumberArray",
]
