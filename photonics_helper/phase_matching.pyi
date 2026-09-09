"""Type stubs for photonics_helper.phase_matching."""

from __future__ import annotations

from typing import NDArray, Protocol, runtime_checkable



@runtime_checkable
class DispersionModel(Protocol):
    def beta(self, omega: NDArray | float) -> NDArray | float: ...
    def beta1(self, omega: NDArray | float) -> NDArray | float: ...


class DispersionAdaptor:
    def __init__(self, dispersion: object, omega0: float | None = None) -> None: ...
    def __call__(self, omega: NDArray | float) -> NDArray | float: ...
    def beta(self, omega: NDArray | float) -> NDArray | float: ...
    def beta1(self, omega: NDArray | float) -> NDArray | float: ...


class PropagationConstantAdaptor:
    def __init__(self, pc: object) -> None: ...
    def __call__(self, omega: NDArray | float) -> NDArray | float: ...
    def beta(self, omega: NDArray | float) -> NDArray | float: ...
    def beta1(self, omega: NDArray | float) -> NDArray | float: ...


class ZDependentDispersionAdaptor:
    def __init__(self, zd_disp: object, z: float) -> None: ...
    def __call__(self, omega: NDArray | float) -> NDArray | float: ...
    def beta(self, omega: NDArray | float) -> NDArray | float: ...
    def beta1(self, omega: NDArray | float) -> NDArray | float: ...


class PhaseMatchResult:
    omega_signal: NDArray
    delta_beta: NDArray
    efficiency: NDArray
    idler_omega: NDArray
    pump_omega: float
    length_m: float
    alpha: float


class DispersiveWaveResult:
    wavelengths_m: NDArray
    wavelengths_nm: NDArray
    soliton_omega: float
    soliton_lambda_m: float
    q_sol: float


class SimulationReadinessReport:
    dispersion_covers_grid: bool
    grid_omega_min: float
    grid_omega_max: float
    dispersion_min_omega: float
    dispersion_max_omega: float
    soliton_order: float
    dispersion_length: float
    nonlinear_length: float
    fission_length: float
    recommended_num_steps: int
    predicted_processes: list[str]
    warnings: list[str]
    recommendations: list[str]
    fwm_predictions: list[dict]
    mi_predictions: dict
    dw_predictions: list[float]


class ValidationReport:
    predictions: list[dict]
    peaks_found: list[dict]
    matches: list[dict]
    overall_pass: bool
    tolerance_nm: float


def fwm_delta_beta_degenerate(
    beta_fn: object,
    omega_p: float,
    omega_s: float,
) -> float: ...


def fwm_delta_beta_general(
    beta_fn: object,
    omega_1: float,
    omega_2: float,
    omega_3: float,
    omega_4: float,
) -> float: ...


def fwm_efficiency(
    delta_beta: float | NDArray,
    length_m: float,
    alpha: float = 0.0,
) -> float | NDArray: ...


def fwm_idler_frequency(
    omega_p: float,
    omega_s: float,
) -> float: ...


def scan_fwm_detuning(
    beta_fn: object,
    omega_p: float,
    omega_signal_grid: NDArray,
    P_pump: float,
    gamma: float,
    alpha: float = 0.0,
    L: float | None = None,
) -> PhaseMatchResult: ...


def mi_gain_spectrum(
    beta2: float,
    gamma: float,
    P: float,
    omega_m: NDArray | float | None = None,
) -> NDArray | float | dict: ...


def mi_sideband_frequencies(
    beta2: float,
    gamma: float,
    P: float,
) -> NDArray: ...


def mi_gain_spectrum_extended(
    beta_fn: object,
    omega0: float,
    gamma: float,
    P: float,
    alpha: float = 0.0,
    L: float | None = None,
    omega_m: NDArray | None = None,
) -> dict: ...


def dispersive_wave_roots(
    beta_fn: object,
    omega_sol: float,
    q_sol: float = 0.0,
    wl_range_nm: tuple[float, float] = (300.0, 2500.0),
    n_brackets: int = 50,
) -> DispersiveWaveResult: ...


def assess_simulation_readiness(
    pulse: object,
    fiber: object,
    dispersion: object,
    betas: NDArray | None = None,
    length: float | None = None,
) -> SimulationReadinessReport: ...


def compare_spectrum_to_phase_matching(
    solver: object,
    report: SimulationReadinessReport,
    tolerance_nm: float = 2.0,
) -> ValidationReport: ...


def plot_fwm_efficiency(
    fwm_result: PhaseMatchResult,
    ax: object = None,
) -> object: ...


def plot_mi_gain(
    mi_result: object,
    ax: object = None,
) -> object: ...


def plot_readiness_report(
    report: SimulationReadinessReport,
    ax: object = None,
) -> object: ...


def plot_spectrum_with_pm_overlay(
    solver: object,
    report: SimulationReadinessReport,
    ax: object = None,
) -> object: ...
