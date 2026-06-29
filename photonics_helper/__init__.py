"""Photonics helper library — units, materials, fibers, pulses, FROG."""

from .base import (
    Wavelength,
    Frequency,
    AngularFrequency,
    Wavenumber,
    WavelengthArray,
    FrequencyArray,
    AngularFrequencyArray,
    WavenumberArray,
    PI,
    C_MS,
    EPS_0,
    MU_0,
)

from .materials import RefractiveIndex
from .fiber import Dispersion, PropagationConstant
from .frog import FROGTrace, generate_trace, retrieve, fidelity

__all__ = [
    "Wavelength",
    "Frequency",
    "AngularFrequency",
    "Wavenumber",
    "WavelengthArray",
    "FrequencyArray",
    "AngularFrequencyArray",
    "WavenumberArray",
    "PI",
    "C_MS",
    "RefractiveIndex",
    "Dispersion",
    "PropagationConstant",
    "FROGTrace",
    "generate_trace",
    "retrieve",
    "fidelity",
    "EPS_0",
    "MU_0",
]
