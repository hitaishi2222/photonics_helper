"""Tests for the post-run conservation + self-convergence diagnostics."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from photonics_helper.base import Length, Time, Wavelength
from photonics_helper.gnlse import (
    ConvergenceDiagnostics,
    DRIFT_WARN_PCT,
    Fiber,
    GNLSESolver,
    _max_drift_pct,
    _self_convergence_metrics,
)
from photonics_helper.pulse import Envelope, TemporalGrid, Wave
from photonics_helper.raman import RamanResponse, RamanSpec

WL0 = 835.0
OMEGA0 = Wavelength(WL0, "nm").to_omega().as_rad_s
BETAS = np.array(
    [
        -11.830e-3,
        8.1038e-5,
        -9.5205e-8,
        2.0737e-10,
        -5.3943e-13,
        1.3486e-15,
        -2.5495e-18,
        3.0524e-21,
        -1.7140e-24,
    ]
)


def _wave(n: int = 2048) -> Wave:
    grid = TemporalGrid(N=n, Tmax=Time(12.5e-12, "s"))
    env = Envelope.from_fwhm(
        "sech", peak_amplitude=np.sqrt(10000.0), fwhm=Time(0.0284 * 1.76, "ps")
    )
    return Wave(grid=grid, envelope=env, central_wavelength=Wavelength(WL0, "nm"))


def _dudley_solver(step_size: Length, *, alpha: float = 0.0) -> GNLSESolver:
    pulse = _wave()
    grid = pulse.grid
    raman = RamanResponse(
        spec=RamanSpec(
            name="Silica", raman_shift_cm=440, raman_linewidth_cm=45, fR=0.18
        ),
        fR=0.18,
        tau1=0.0122e-12,
        tau2=0.032e-12,
        grid=grid,
    )
    fiber = Fiber.from_gamma(
        gamma=0.11,
        alpha=alpha,
        length=Length(0.15, "m"),
        raman_response=raman,
    )
    return GNLSESolver(
        pulse=pulse,
        fiber=fiber,
        betas=BETAS,
        include_raman=True,
        include_self_steepening=True,
        include_tpa=False,
        step_size=step_size,
    )


def _quiet(solver: GNLSESolver, *args, **kwargs) -> ConvergenceDiagnostics:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return solver.verify_convergence(*args, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_max_drift_pct_uses_max_not_endpoint():
    # endpoint-only drift is 10 %, but the transient peak is 50 %
    assert _max_drift_pct([1.0, 1.5, 0.9]) == pytest.approx(50.0)
    assert _max_drift_pct([1.0]) == 0.0
    assert _max_drift_pct([0.0, 1.0]) == 0.0


def test_max_drift_pct_flags_nonfinite_series():
    # A blown-up run must never read as a clean 0 % drift.
    assert _max_drift_pct([1.0, float("nan")]) == float("inf")
    assert _max_drift_pct([1.0, float("inf")]) == float("inf")


def test_self_convergence_metrics_identical_solution():
    rng = np.random.default_rng(0)
    A = rng.standard_normal(256) + 1j * rng.standard_normal(256)
    S = np.abs(A) ** 2
    wl = np.linspace(400.0, 1350.0, 256)
    rms, peak, overlap = _self_convergence_metrics(wl, S, S, A, A)
    assert rms == pytest.approx(0.0, abs=1e-12)
    assert peak == pytest.approx(0.0)
    assert overlap == pytest.approx(1.0)


def test_self_convergence_overlap_is_phase_robust():
    # A global phase must not spoil the modal overlap (the overlap is a
    # projection, so |⟨A, e^{iφ}A⟩|² is phase-invariant).
    rng = np.random.default_rng(1)
    A = rng.standard_normal(256) + 1j * rng.standard_normal(256)
    S = np.abs(A) ** 2
    wl = np.linspace(400.0, 1350.0, 256)
    _, _, overlap_phase = _self_convergence_metrics(
        wl, S, S, A, A * np.exp(1j * 0.7)
    )
    assert overlap_phase == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# verify_convergence
# ---------------------------------------------------------------------------


def test_verify_convergence_requires_a_completed_run():
    solver = _dudley_solver(Length(0.15 / 50, "m"))
    with pytest.raises(RuntimeError):
        solver.verify_convergence()


def test_verify_convergence_rejects_probe_factor_one():
    solver = _dudley_solver(Length(0.15 / 50, "m"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver.propagate(num_steps=50, nsaves=2)
    with pytest.raises(ValueError):
        solver.verify_convergence(probe_factor=1)


def test_verify_convergence_flags_underresolved_run():
    solver = _dudley_solver(Length(0.15 / 40, "m"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver.propagate(num_steps=40, nsaves=2)
    diag = _quiet(solver, probe_factor=2, warn=False)
    assert isinstance(diag, ConvergenceDiagnostics)
    assert diag.num_steps == 40
    assert diag.probe_steps == 80
    assert not diag.converged
    assert diag.drift_pct > DRIFT_WARN_PCT
    # the table always ends with an explicit verdict
    assert "verdict:" in diag.summary()
    assert "NOT CONVERGED" in diag.summary()


def test_verify_convergence_warns_when_not_converged():
    solver = _dudley_solver(Length(0.15 / 40, "m"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver.propagate(num_steps=40, nsaves=2)
        with pytest.warns(UserWarning, match="not self-converged"):
            solver.verify_convergence(probe_factor=2)


def test_verify_convergence_converges_on_dispersionless_run():
    # With no dispersion the Strang split is exact, so any step count is
    # converged and the drift is zero.
    pulse = _wave(1024)
    fiber = Fiber.from_gamma(
        gamma=0.11,
        length=Length(0.01, "m"),
    )
    solver = GNLSESolver(
        pulse=pulse,
        fiber=fiber,
        betas=np.array([0.0]),
        include_raman=False,
        include_self_steepening=False,
        step_size=Length(0.01 / 20, "m"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver.propagate(num_steps=20, nsaves=2)
    diag = _quiet(solver, probe_factor=2, warn=False)
    assert diag.converged
    assert diag.rms_db < 1e-6
    assert diag.overlap > 0.999999


def test_verify_convergence_marks_drift_not_applicable_with_loss():
    solver = _dudley_solver(Length(0.15 / 40, "m"), alpha=2.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver.propagate(num_steps=40, nsaves=2)
    diag = _quiet(solver, probe_factor=2, warn=False)
    assert "n/a" in diag.invariant
    assert any("loss/gain" in note for note in diag.notes)


def test_propagate_verify_kwarg_runs_and_returns_table(capsys):
    solver = _dudley_solver(Length(0.15 / 40, "m"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver.propagate(num_steps=40, nsaves=2, verify=True, verify_probe_factor=2)
    out = capsys.readouterr().out
    assert "GNLSE conservation + convergence diagnostics" in out
    assert "verdict:" in out


def test_verify_convergence_rejects_raman_noise():
    solver = _dudley_solver(Length(0.15 / 40, "m"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver.propagate(num_steps=40, nsaves=2, raman_noise=True, noise_seed=0)
    with pytest.raises(ValueError):
        solver.verify_convergence()


def test_convergence_summary_contains_all_columns():
    diag = ConvergenceDiagnostics(
        scheme="strang",
        num_steps=100,
        dz_m=1e-4,
        probe_steps=200,
        invariant="energy",
        drift_pct=0.2,
        drift_tol_pct=1.0,
        rms_db=0.1,
        peak_shift_nm=0.5,
        overlap=0.999,
        rms_tol_db=0.5,
        peak_tol_nm=2.0,
        overlap_tol=0.99,
    )
    text = diag.summary()
    assert diag.converged
    for token in ("drift", "rms", "peak shift", "overlap", "CONVERGED"):
        assert token in text
