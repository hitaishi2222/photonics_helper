"""Task 4.3 — grid-convergence harness and cited analytical validation checks.

Covers the ``gnlse-validation`` delta specification:

- :func:`convergence_study` reports converged for an adequately resolved
  configuration and not converged for an under-resolved one (Sinkin,
  Holzlöhner, Zweck & Menyuk, *J. Lightwave Technol.* **21**, 61 (2003));
  observables are caller-selected.
- The cited analytical checks all pass on their canonical configurations:
  SPM (Stolen & Lin 1978; Agrawal §4.1), MI (Agrawal §5.1.9 in the validated
  power-gain convention), the fundamental soliton (Agrawal §5.2), and the
  Raman SSFS (Gordon 1986).

The soliton-fission configuration used for the convergence scenarios is an
``N = 3`` fundamental-family soliton over two soliton periods — a regime where
window under-resolution and step-size error measurably move the peak intensity
and the RMS bandwidth.
"""

import numpy as np
import pytest

from photonics_helper.base import Length, Time, Wavelength, C_MS
from photonics_helper.gnlse import FiberProfile, GNLSESolver
from photonics_helper.gnlse_validation import (
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
from photonics_helper.pulse import Envelope, TemporalGrid, Wave
from photonics_helper.raman import RamanResponse, RamanSpec

WL0 = 1550e-9
BETA2 = -21e-27  # s²/m — SMF-28-ish at 1550 nm
OMEGA0 = 2 * np.pi * C_MS / WL0


def _fission_build(N: int, Tmax_s: float, num_steps: int) -> GNLSESolver:
    """N = 3 soliton propagated over two soliton periods."""
    grid = TemporalGrid(N=N, Tmax=Time(Tmax_s, "s"))
    t0 = 50e-15 / 1.763
    gamma = 2.0
    peak = 9.0 * abs(BETA2) / (gamma * t0**2)  # soliton order N = 3
    length = 2.0 * (np.pi / 2) * t0**2 / abs(BETA2)
    envelope = Envelope(shape="sech", peak_amplitude=np.sqrt(peak), pulse_width=Time(t0, "s"))
    pulse = Wave(grid=grid, envelope=envelope, central_wavelength=Wavelength(WL0, "m"))
    fiber = FiberProfile.from_gamma(gamma=gamma, n2=2.6e-20, omega0=OMEGA0, length=Length(length, "m"))
    solver = GNLSESolver(
        pulse=pulse,
        fiber=fiber,
        betas=np.array([BETA2 * 1e24]),  # s²/m → ps²/m
        include_raman=False,
        include_self_steepening=False,
        include_tpa=False,
        step_size=Length(length / num_steps, "m"),
    )
    solver.propagate(num_steps)
    return solver


# ── Convergence harness ─────────────────────────────────────────────────────


def test_converged_config_reports_converged():
    """A properly refined fission run converges in both observables."""
    report = convergence_study(
        _fission_build,
        refinements=[
            {"N": 8192, "Tmax_s": 8e-12, "num_steps": 1000},
            {"N": 16384, "Tmax_s": 8e-12, "num_steps": 2000},
            {"N": 32768, "Tmax_s": 8e-12, "num_steps": 2000},
        ],
        observables=["peak_intensity", "rms_bandwidth"],
        tolerance=2e-3,
    )
    assert isinstance(report, ConvergenceReport)
    assert report.converged
    for observable in report.observables:
        assert isinstance(observable, ObservableReport)
        assert 0.0 <= observable.change_last < 2e-3
        # three refinements → three values, first change is None
        assert len(observable.values) == 3
        assert observable.changes[0] is None


def test_coarse_config_reports_not_converged():
    """A heavily under-resolved grid moves the observables materially."""
    report = convergence_study(
        _fission_build,
        refinements=[
            {"N": 128, "Tmax_s": 8e-12, "num_steps": 16},
            {"N": 256, "Tmax_s": 8e-12, "num_steps": 32},
            {"N": 256, "Tmax_s": 8e-12, "num_steps": 64},
        ],
        observables=["peak_intensity", "rms_bandwidth"],
        tolerance=2e-3,
    )
    assert not report.converged
    assert any(o.change_last > 5e-3 for o in report.observables)
    assert report.summary()  # summary text renders


def test_observables_are_caller_selected():
    """A named subset and a custom mapping both drive the report."""
    shared = dict(
        refinements=[
            {"N": 1024, "Tmax_s": 8e-12, "num_steps": 64},
            {"N": 2048, "Tmax_s": 8e-12, "num_steps": 128},
        ],
        tolerance=1e-3,
    )
    report = convergence_study(_fission_build, observables=["rms_bandwidth"], **shared)
    assert [o.name for o in report.observables] == ["rms_bandwidth"]

    custom = convergence_study(
        _fission_build,
        observables={"max_minus_two": lambda r: float(np.max(np.abs(
            (r.evolution[-1].envelope_field if hasattr(r, "evolution") else r.envelope_field)
        )) ** 2) - 2.0},
        **shared,
    )
    assert [o.name for o in custom.observables] == ["max_minus_two"]

    with pytest.raises(KeyError):
        convergence_study(_fission_build, observables=["nope"], **shared)


def test_convergence_study_requires_two_refinements():
    with pytest.raises(ValueError, match="at least two refinements"):
        convergence_study(_fission_build, refinements=[{"N": 1024, "Tmax_s": 8e-12, "num_steps": 64}])
    with pytest.raises(ValueError, match="tolerance"):
        convergence_study(
            _fission_build,
            refinements=[
                {"N": 1024, "Tmax_s": 8e-12, "num_steps": 64},
                {"N": 2048, "Tmax_s": 8e-12, "num_steps": 128},
            ],
            tolerance=0.0,
        )


def test_default_observables_registry():
    assert set(DEFAULT_OBSERVABLES) >= {
        "peak_intensity",
        "pulse_energy",
        "rms_bandwidth",
        "rms_width",
    }


# ── Cited analytical checks ────────────────────────────────────────────────


def test_check_spm_matches_closed_form():
    """SPM spectrum ∥ the Fourier integral + fringe rule (Stolen & Lin 1978)."""
    grid = TemporalGrid(N=16384, Tmax=Time(8e-12, "s"))
    metrics = check_spm(
        gamma=11.0,
        peak_power=1e-3,
        t0=100e-15,
        wavelength_m=WL0,
        phi_max=4 * np.pi,
        grid=grid,
    )
    assert metrics["max_abs_spectrum_diff"] < 5e-3
    assert metrics["n_peaks"] == metrics["n_peaks_expected"] == 5


def test_check_spm_detects_discrepancies():
    """A zero tolerance must flag even the tiny numerical residual — proving
    both checks inside ``check_spm`` are actually evaluated."""
    grid = TemporalGrid(N=16384, Tmax=Time(8e-12, "s"))
    with pytest.raises(ValidationFailure, match="SPM"):
        check_spm(
            gamma=11.0,
            peak_power=1e-3,
            t0=100e-15,
            wavelength_m=WL0,
            phi_max=4 * np.pi,
            grid=grid,
            tolerance=0.0,
        )


def test_check_mi_matches_linear_stability():
    """Sideband power gain vs the exact linear-stability g(Ω)."""
    grid = TemporalGrid(N=16384, Tmax=Time(8e-12, "s"))
    metrics = check_mi(
        beta2=BETA2,
        gamma=1.0,
        pump_power=1.0,
        wavelength_m=WL0,
        probe_omega=2 * np.pi * 1.0e12,
        grid=grid,
    )
    assert abs(metrics["g_measured"] - metrics["g_reference"]) / metrics["g_reference"] < 0.15
    # the validated power-gain convention: g_max = 2γP
    assert mi_gain_of(BETA2, 1.0, 1.0, 9759000729485.332) == pytest.approx(2.0, rel=1e-3)


def test_check_mi_rejects_normal_dispersion():
    grid = TemporalGrid(N=1024, Tmax=Time(8e-12, "s"))
    with pytest.raises(ValueError, match="anomalous"):
        check_mi(
            beta2=+21e-27,
            gamma=1.0,
            pump_power=1.0,
            wavelength_m=WL0,
            probe_omega=2 * np.pi * 1.0e12,
            grid=grid,
        )


def test_check_soliton_recovers_shape():
    """Fundamental soliton returns after z_sol = (π/2)L_D (Agrawal §5.2)."""
    grid = TemporalGrid(N=16384, Tmax=Time(8e-12, "s"))
    metrics = check_soliton(
        beta2=BETA2, gamma=1.0, t0=1e-12, wavelength_m=WL0, grid=grid
    )
    assert metrics["shape_overlap"] > 0.98
    assert metrics["energy_ratio"] == pytest.approx(1.0, abs=1e-6)


def test_check_soliton_detects_slight_drift():
    """A tolerance of zero must flag even legitimate numerical drift —
    proving the check bites rather than vacuously passing."""
    grid = TemporalGrid(N=16384, Tmax=Time(8e-12, "s"))
    with pytest.raises(ValidationFailure):
        check_soliton(
            beta2=BETA2, gamma=1.0, t0=1e-12, wavelength_m=WL0, grid=grid,
            periods=1.0, tolerance=0.0,
        )


def test_check_gordon_ssfs_matches_gordon():
    """Raman SSFS rate vs Gordon's analytic law (Gordon 1986)."""
    grid = TemporalGrid(N=8192, Tmax=Time(14e-12, "s"))
    spec = RamanSpec(name="Silica", raman_shift_cm=440.0, raman_linewidth_cm=45.0, fR=0.18)
    response = RamanResponse(spec=spec, fR=0.18, tau1=12.2e-15, tau2=32e-15, grid=grid)
    beta2 = -7.0e-27
    gamma = 0.11
    t0 = 50e-15 / 1.763  # soliton width from 50 fs FWHM sech
    peak = abs(beta2) / (gamma * t0**2)  # fundamental soliton peak power
    metrics = check_gordon_ssfs(
        beta2=beta2,
        gamma=gamma,
        peak_power=peak,
        t0=t0,
        wavelength_m=835e-9,
        raman_response=response,
        grid=grid,
        length=0.5,
    )
    assert 0.75 <= metrics["ratio"] <= 1.25
    assert metrics["measured_shift_nm"] > 0  # Stokes red shift


def test_gordon_ssfs_rate_guards():
    with pytest.raises(ValueError, match="anomalous"):
        gordon_ssfs_rate(beta2=1e-27, gamma=1.0, peak_power=1.0, t0=1e-12, t_raman=1e-15)
    # soliton-order assertion: γP₀T₀²/|β₂| = 1 required
    with pytest.raises(ValueError, match="fundamental soliton"):
        gordon_ssfs_rate(beta2=-21e-27, gamma=1.0, peak_power=1.0, t0=1e-12, t_raman=1e-15)
