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
from .pulse import FROGTrace, generate_trace, retrieve, fidelity
from .raman import (
    RamanSpec,
    RamanDatabase,
    RamanResponse,
    RamanFrequencyResponse,
    RamanPulseInteraction,
    PumpWavelengthExplorer,
    MaterialComparison,
    RAMAN_MATERIALS,
    COMMON_COMPARISONS,
    app,
)

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
]
