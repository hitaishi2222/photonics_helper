"""Gap E: h_R supplied-grid normalization residual is exposed and warned.

Ground truth: the legacy pure-Lorentzian h_R normalization divides by the
trapezoid integral on the *supplied* grid, so the continuous integral of the
returned response deviates from 1 by an Euler–Maclaurin residual that grows
~ dt² (the trapezoid rule misses dt²·f'(0)/12 with f'(0) ≠ 0 at the causal
onset). The residual is exposed as ``PhononResponse.integral_error`` and a
:class:`UserWarning` fires above 1e-5. The h_R *values* are unchanged — the
consumer's model is calibrated on them — and the Voigt path keeps its
closed-form, grid-independent normalization (Gap D).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

import photonics_helper.phonon as phonon_module
from photonics_helper.phonon import PHONON_MATERIALS, PhononMode, PhononResponse

C_CM_S = 2.99792458e10
TWO_PI_C = 2.0 * np.pi * C_CM_S

# Measured table: fixed 40 ps window (tail truncation negligible for the
# shipped LiNbO3 linewidths), dt swept 0.5–20 fs. The residual grows ~ dt².
LNB_DT_FS = [0.5, 1.0, 2.0, 3.66, 5.0, 10.0, 20.0]
LNB_TMAX_S = 40e-12


def _grid_t(dt_fs: float, tmax_s: float = LNB_TMAX_S) -> np.ndarray:
    n = int(round(tmax_s / (dt_fs * 1e-15)))
    return np.arange(n) * dt_fs * 1e-15


def _raw_lorentzian_components(response: PhononResponse, t_pos: np.ndarray):
    """Independent recomputation of the raw sum and the closed-form integral."""
    total = np.zeros_like(t_pos)
    analytic = 0.0
    for mode in response.modes:
        omega = TWO_PI_C * float(mode.shift_cm.as_1_cm)
        gamma = TWO_PI_C * float(mode.linewidth_cm.as_1_cm)
        tau = 2.0 / gamma
        weight = float(mode.relative_strength)
        total += weight * np.sin(omega * t_pos) * np.exp(-t_pos / tau)
        analytic += weight * omega * tau**2 / (1.0 + (omega * tau) ** 2)
    return total, analytic


def _lnb_response() -> PhononResponse:
    return PhononResponse(PHONON_MATERIALS["LiNbO3"])


def test_h_R_values_unchanged_legacy_grid_normalization() -> None:
    """Pure-Lorentzian h_R is still the raw sum divided by the grid integral."""
    response = _lnb_response()
    t = _grid_t(3.66)
    h = response.h_R(t)
    t_pos = t[t >= 0]
    total, _ = _raw_lorentzian_components(response, t_pos)
    expected = total / np.trapezoid(total, t_pos)
    assert np.array_equal(h[t >= 0], expected)
    assert np.all(h[t < 0] == 0.0)


def test_integral_error_matches_independent_recomputation() -> None:
    """integral_error equals |grid/analytic − 1|, monotone and ~ dt² at fine dt."""
    response = _lnb_response()
    residuals = []
    for dt_fs in LNB_DT_FS:
        t = _grid_t(dt_fs)
        response.h_R(t)
        t_pos = t[t >= 0]
        total, analytic = _raw_lorentzian_components(response, t_pos)
        expected = abs(np.trapezoid(total, t_pos) / analytic - 1.0)
        assert response.integral_error == pytest.approx(expected, rel=1e-12)
        residuals.append(expected)

    assert residuals == sorted(residuals), "residual must grow with dt"
    # Euler–Maclaurin: residual ∝ dt² while higher-order terms are negligible.
    assert residuals[2] / residuals[1] == pytest.approx(4.0, rel=0.15)
    assert residuals[1] / residuals[0] == pytest.approx(4.0, rel=0.15)

    # The grid trapezoid of the returned response is still exactly 1.
    t = _grid_t(3.66)
    h = response.h_R(t)
    assert np.trapezoid(h[t >= 0], t[t >= 0]) == pytest.approx(1.0, abs=1e-12)


def test_user_warning_fires_above_threshold() -> None:
    response = _lnb_response()
    with pytest.warns(UserWarning, match="integral"):
        response.h_R(_grid_t(3.66))
    assert response.integral_error > 1e-5


def test_no_warning_when_residual_below_threshold() -> None:
    # A broad, low-shift mode (tau = 200 fs >> dt = 1 fs) keeps the residual
    # ~ (dt/tau)^2/12 ≈ 2e-6, below the threshold.
    response = PhononResponse(
        [PhononMode(shift_cm=10.0, linewidth_cm=53.0)]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        response.h_R(_grid_t(1.0, tmax_s=20e-12))
    assert response.integral_error < 1e-5


def test_voigt_path_reports_zero_residual_and_no_warning() -> None:
    response = PhononResponse(
        [PhononMode(shift_cm=254.0, linewidth_cm=14.0, fwhm_gauss_cm=42.0)]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        h = response.h_R(_grid_t(3.66))
    assert response.integral_error == 0.0
    # Closed-form normalization: the continuous integral is exactly 1, so the
    # grid trapezoid sits at 1 only to discretisation accuracy.
    t_pos = _grid_t(3.66)[_grid_t(3.66) >= 0]
    assert np.trapezoid(h[t_pos >= 0], t_pos) == pytest.approx(1.0, abs=5e-3)


def test_empty_response_reports_zero_residual() -> None:
    response = PhononResponse(modes=[])
    response.h_R(np.linspace(0.0, 1e-12, 8))
    assert response.integral_error == 0.0


def test_phonon_module_is_torch_free() -> None:
    source = Path(phonon_module.__file__).read_text()
    assert "torch" not in source
