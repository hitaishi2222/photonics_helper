"""Photonics helper library — units, materials, fibers, pulses, FROG.

This package uses **lazy imports** (PEP 562). ``import photonics_helper`` binds
almost nothing; each public name is imported from its defining module the first
time it is accessed::

    from photonics_helper import Wavelength   # imports .base only

The goal is that the foundation layer — units, constants, grids and materials —
can be used without pulling in the plotting/simulation stack (matplotlib,
plotly, dash). See ``photonics_helper.core`` for the documented foundation
surface.

The public surface is the static ``__all__`` below; it must not shrink. Names
are resolved from ``_LAZY_MODULES`` (and ``_LAZY_ATTR`` for aliased exports) by
``__getattr__`` and cached in the module globals after first access.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - static analyzers/IDEs only
    from .base import (
        C_MS,
        EPS_0,
        H_PLANCK,
        HBAR,
        MU_0,
        PI,
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
    from .breathers import (
        SolitonOnBackground,
        akhmediev_breather,
        general_sfb,
        kuznetsov_ma,
        peregrine_soliton,
        sfb_peak_ratio,
        sfb_spatial_period,
        sfb_temporal_period,
    )
    from .chi2 import (
        Chi2Result,
        Lambda_qpm,
        delta_k_shg,
        qpm_grating,
        shg_coupling,
        solve_dfg,
        solve_sfg,
        solve_shg,
        solve_three_wave,
    )
    from .dashboard import app as dashboard_app
    from .dbr import TMM, Block, Material, Pattern, plot_2d, plot_index
    from .fiber import (
        Dispersion,
        PropagationConstant,
        WaveguideMode,
        ZDependentDispersion,
    )
    from .gnlse import FiberProfile, GNLSESolver, SplitStepEngine, TaperedGNLSESolver
    from .gnlse_validation import (
        DEFAULT_OBSERVABLES,
        ConvergenceReport,
        ObservableReport,
        ValidationFailure,
        check_gordon_ssfs,
        check_mi,
        check_soliton,
        check_spm,
        convergence_study,
        gordon_ssfs_rate,
        mi_gain_of,
    )
    from .materials import RefractiveIndex
    from .vector_gnlse import (
        Coupling,
        RandomBirefringenceEngine,
        VectorSplitStepEngine,
    )
    from .multimode_gnlse import (
        CoeffModel,
        MultimodeSplitStepEngine,
    )
    from .noise import (
        add_ase_noise,
        add_noise,
        ase_noise_field,
        complex_gaussian_noise,
    )
    from .phase_matching import (
        DispersionAdaptor,
        DispersionModel,
        DispersiveWaveResult,
        PhaseMatchResult,
        PropagationConstantAdaptor,
        SimulationReadinessReport,
        ValidationReport,
        ZDependentDispersionAdaptor,
        assess_simulation_readiness,
        compare_spectrum_to_phase_matching,
        dispersive_wave_roots,
        fwm_delta_beta_degenerate,
        fwm_delta_beta_general,
        fwm_efficiency,
        fwm_idler_frequency,
        mi_gain_spectrum,
        mi_gain_spectrum_extended,
        mi_sideband_frequencies,
        plot_fwm_efficiency,
        plot_mi_gain,
        plot_readiness_report,
        plot_spectrum_with_pm_overlay,
        scan_fwm_detuning,
    )
    from .phonon import PHONON_MATERIALS, PhononMode, PhononResponse
    from .pulse import (
        Envelope,
        FROGTrace,
        TemporalGrid,
        Wave,
        fidelity,
        generate_trace,
        retrieve,
    )
    from .raman import (
        COMMON_COMPARISONS,
        RAMAN_MATERIALS,
        MaterialComparison,
        PumpWavelengthExplorer,
        RamanDatabase,
        RamanFrequencyResponse,
        RamanPulseInteraction,
        RamanResponse,
        RamanSpec,
        app,
    )
    from .soliton import SolitonAnalyzer
    from .structured import (
        LaguerreGaussianMode,
        StructuredField,
        beam_waist,
        gouy_phase,
        overlap,
        plot_transverse_profile,
        radius_of_curvature,
        rayleigh_range,
    )
    from .wave_breaking import (
        WaveBreaking,
        detect_oscillation_onset,
        detect_steepening_onset,
        dispersion_length,
        edge_steepness,
        gaussian_edge_steepness,
        nonlinear_length,
        wave_breaking_distance,
    )
    from .inverse_design import FitResult, design_efficiency, fit_two_wave


__all__ = [
    "COMMON_COMPARISONS",
    "C_MS",
    "EPS_0",
    "HBAR",
    "H_PLANCK",
    "MU_0",
    "PHONON_MATERIALS",
    "PI",
    "RAMAN_MATERIALS",
    "TMM",
    "AngularFrequency",
    "AngularFrequencyArray",
    "Area",
    "Block",
    "Chi2Result",
    "ConvergenceReport",
    "DEFAULT_OBSERVABLES",
    "Dispersion",
    "DispersionAdaptor",
    "DispersionModel",
    "DispersiveWaveResult",
    "Energy",
    "Envelope",
    "FROGTrace",
    "FiberProfile",
    "FitResult",
    "Frequency",
    "FrequencyArray",
    "GNLSESolver",
    "ObservableReport",
    "ValidationFailure",
    "LaguerreGaussianMode",
    "Lambda_qpm",
    "Length",
    "Material",
    "MaterialComparison",
    "MaterialDataset",
    "Pattern",
    "PeakPower",
    "Permeability",
    "Permittivity",
    "PhaseMatchResult",
    "PhononMode",
    "PhononResponse",
    "Power",
    "PropagationConstant",
    "PropagationConstantAdaptor",
    "PumpWavelengthExplorer",
    "RamanDatabase",
    "RamanFrequencyResponse",
    "RamanPulseInteraction",
    "RamanResponse",
    "RamanSpec",
    "RefractiveIndex",
    "SimulationReadinessReport",
    "SolitonAnalyzer",
    "SolitonOnBackground",
    "SplitStepEngine",
    "StructuredField",
    "CoeffModel",
    "Coupling",
    "MultimodeSplitStepEngine",
    "RandomBirefringenceEngine",
    "TaperedGNLSESolver",
    "VectorSplitStepEngine",
    "TemporalGrid",
    "Time",
    "ValidationReport",
    "Wave",
    "WaveBreaking",
    "WaveguideMode",
    "Wavelength",
    "WavelengthArray",
    "Wavenumber",
    "WavenumberArray",
    "ZDependentDispersion",
    "ZDependentDispersionAdaptor",
    "add_ase_noise",
    "add_noise",
    "akhmediev_breather",
    "app",
    "ase_noise_field",
    "assess_simulation_readiness",
    "beam_waist",
    "check_gordon_ssfs",
    "check_mi",
    "check_soliton",
    "check_spm",
    "compare_spectrum_to_phase_matching",
    "complex_gaussian_noise",
    "convergence_study",
    "dashboard_app",
    "delta_k_shg",
    "design_efficiency",
    "detect_oscillation_onset",
    "detect_steepening_onset",
    "dispersion_length",
    "dispersive_wave_roots",
    "edge_steepness",
    "fidelity",
    "fit_two_wave",
    "fwm_delta_beta_degenerate",
    "fwm_delta_beta_general",
    "fwm_efficiency",
    "fwm_idler_frequency",
    "gaussian_edge_steepness",
    "general_sfb",
    "generate_trace",
    "gordon_ssfs_rate",
    "gouy_phase",
    "kuznetsov_ma",
    "material_catalog",
    "mi_gain_of",
    "mi_gain_spectrum",
    "mi_gain_spectrum_extended",
    "mi_sideband_frequencies",
    "nonlinear_length",
    "overlap",
    "peregrine_soliton",
    "plot_2d",
    "plot_fwm_efficiency",
    "plot_index",
    "plot_mi_gain",
    "plot_readiness_report",
    "plot_spectrum_with_pm_overlay",
    "plot_transverse_profile",
    "print_material_catalog",
    "qpm_grating",
    "radius_of_curvature",
    "rayleigh_range",
    "retrieve",
    "scan_fwm_detuning",
    "sfb_peak_ratio",
    "sfb_spatial_period",
    "sfb_temporal_period",
    "shg_coupling",
    "solve_cascaded_shg",
    "solve_dfg",
    "solve_sfg",
    "solve_shg",
    "solve_three_wave",
    "wave_breaking_distance",
]

# Public name -> defining module. Kept in one place so __getattr__ is a lookup,
# not a chain of ifs. Every name in __all__ must appear here (tested).
_LAZY_MODULES: dict[str, str] = {
    # .base — constants + unit types
    **{n: ".base" for n in (
        "C_MS", "EPS_0", "H_PLANCK", "HBAR", "MU_0", "PI",
        "AngularFrequency", "AngularFrequencyArray", "Area", "Energy",
        "Frequency", "FrequencyArray", "Length", "PeakPower", "Permeability",
        "Permittivity", "Power", "Time", "Wavelength", "WavelengthArray",
        "Wavenumber", "WavenumberArray",
    )},
    # .breathers
    **{n: ".breathers" for n in (
        "SolitonOnBackground", "akhmediev_breather", "general_sfb",
        "kuznetsov_ma", "peregrine_soliton", "sfb_peak_ratio",
        "sfb_spatial_period", "sfb_temporal_period",
    )},
    # .chi2
    **{n: ".chi2" for n in (
        "Chi2Result", "Lambda_qpm", "delta_k_shg", "qpm_grating",
        "shg_coupling", "solve_cascaded_shg", "solve_dfg", "solve_sfg",
        "solve_shg", "solve_three_wave",
    )},
    # .inverse_design — Phase 4 item 5 (inverse-design layer)
    **{n: ".inverse_design" for n in (
        "FitResult", "design_efficiency", "fit_two_wave",
    )},
    "dashboard_app": ".dashboard",
    # .dbr
    **{n: ".dbr" for n in (
        "TMM", "Block", "Material", "Pattern", "plot_2d", "plot_index",
    )},
    # .fiber
    **{n: ".fiber" for n in (
        "Dispersion", "PropagationConstant", "WaveguideMode", "ZDependentDispersion",
    )},
    # .gnlse
    **{n: ".gnlse" for n in (
        "FiberProfile", "GNLSESolver", "SplitStepEngine", "TaperedGNLSESolver",
    )},
    # .gnlse_validation — convergence + analytical checks
    **{n: ".gnlse_validation" for n in (
        "DEFAULT_OBSERVABLES", "ConvergenceReport", "ObservableReport",
        "ValidationFailure", "convergence_study", "check_spm", "check_mi",
        "check_soliton", "check_gordon_ssfs", "gordon_ssfs_rate", "mi_gain_of",
    )},
    # .multimode_gnlse — few-mode coupled GNLSE
    **{n: ".multimode_gnlse" for n in (
        "CoeffModel", "MultimodeSplitStepEngine",
    )},
    # .vector_gnlse — polarization-coupled GNLSE
    **{n: ".vector_gnlse" for n in (
        "Coupling", "RandomBirefringenceEngine", "VectorSplitStepEngine",
    )},
    "RefractiveIndex": ".materials",
    "MaterialDataset": ".materials",
    "material_catalog": ".materials",
    "print_material_catalog": ".materials",
    # .noise
    **{n: ".noise" for n in (
        "add_ase_noise", "add_noise", "ase_noise_field", "complex_gaussian_noise",
    )},
    # .phase_matching
    **{n: ".phase_matching" for n in (
        "DispersionAdaptor", "DispersionModel", "DispersiveWaveResult",
        "PhaseMatchResult", "PropagationConstantAdaptor",
        "SimulationReadinessReport", "ValidationReport",
        "ZDependentDispersionAdaptor", "assess_simulation_readiness",
        "compare_spectrum_to_phase_matching", "dispersive_wave_roots",
        "fwm_delta_beta_degenerate", "fwm_delta_beta_general", "fwm_efficiency",
        "fwm_idler_frequency", "mi_gain_spectrum", "mi_gain_spectrum_extended",
        "mi_sideband_frequencies", "plot_fwm_efficiency", "plot_mi_gain",
        "plot_readiness_report", "plot_spectrum_with_pm_overlay", "scan_fwm_detuning",
    )},
    # .phonon
    **{n: ".phonon" for n in ("PHONON_MATERIALS", "PhononMode", "PhononResponse")},
    # .pulse
    **{n: ".pulse" for n in (
        "Envelope", "FROGTrace", "TemporalGrid", "Wave", "fidelity",
        "generate_trace", "retrieve",
    )},
    # .raman
    **{n: ".raman" for n in (
        "COMMON_COMPARISONS", "RAMAN_MATERIALS", "MaterialComparison",
        "PumpWavelengthExplorer", "RamanDatabase", "RamanFrequencyResponse",
        "RamanPulseInteraction", "RamanResponse", "RamanSpec", "app",
    )},
    "SolitonAnalyzer": ".soliton",
    # .structured
    **{n: ".structured" for n in (
        "LaguerreGaussianMode", "StructuredField", "beam_waist", "gouy_phase",
        "overlap", "plot_transverse_profile", "radius_of_curvature", "rayleigh_range",
    )},
    # .wave_breaking
    **{n: ".wave_breaking" for n in (
        "WaveBreaking", "detect_oscillation_onset", "detect_steepening_onset",
        "dispersion_length", "edge_steepness", "gaussian_edge_steepness",
        "nonlinear_length", "wave_breaking_distance",
    )},
}

# Exports whose public name differs from the attribute in the defining module.
_LAZY_ATTR: dict[str, str] = {"dashboard_app": "app"}


def __getattr__(name: str) -> Any:
    """Import and cache a public name on first access (PEP 562)."""
    try:
        module_name = _LAZY_MODULES[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    module = importlib.import_module(module_name, __name__)
    value = getattr(module, _LAZY_ATTR.get(name, name))
    globals()[name] = value  # cache so __getattr__ runs once
    return value


def __dir__() -> list[str]:
    """Expose the public surface without importing it."""
    return sorted(__all__)
