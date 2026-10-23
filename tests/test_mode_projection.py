"""Launch-field projection onto a solved mode set (GAP A7)."""

from __future__ import annotations

import numpy as np
import pytest

from photonics_helper.mode_analysis import (
    LoadedMode,
    ModalProjection,
    project_field_onto_modes,
)

N = 101
X = np.linspace(-2.0, 2.0, N)
Y = np.linspace(-2.0, 2.0, N)
LAM = 1.55


def _gaussian(x, y, w=0.5, ell=0):
    r = np.hypot(x, y)
    phi = np.arctan2(y, x)
    return np.exp(-(r**2) / w**2) * np.exp(1j * ell * phi)


def _mode(label, ey, *, ex=None):
    ex = np.zeros_like(ey) if ex is None else ex
    return LoadedMode(
        label=label,
        wavelengths_um=np.array([LAM]),
        neff=np.array([2.0]),
        ng=None,
        Ex=np.asarray(ex)[None, :, :],
        Ey=np.asarray(ey)[None, :, :],
        x_um=X.copy(),
        y_um=Y.copy(),
    )


def _grid():
    return np.meshgrid(X, Y, indexing="xy")


def test_exact_mode_match_captures_all_power():
    xx, yy = _grid()
    f0 = _gaussian(xx, yy)
    launch = 2.5 * f0
    res = project_field_onto_modes(launch, X, Y, [_mode(0, f0)], LAM)
    assert isinstance(res, ModalProjection)
    # amplitudes are sqrt(power): |a_0|² is the captured power
    assert abs(res.amplitudes[0]) ** 2 == pytest.approx(res.launched_power, rel=1e-12)
    assert res.coupling_efficiency == pytest.approx(1.0, rel=1e-12)
    assert res.residual_power == pytest.approx(0.0, abs=1e-12)
    assert res.labels == [0]


def test_orthogonal_mode_carries_no_amplitude():
    xx, yy = _grid()
    even = _gaussian(xx, yy)
    odd = xx * _gaussian(xx, yy)  # odd in x: orthogonal on a symmetric grid
    res = project_field_onto_modes(even, X, Y, [_mode(0, even), _mode(1, odd)], LAM)
    assert res.amplitudes[1] == pytest.approx(0.0, abs=1e-12)
    assert res.coupling_efficiency == pytest.approx(1.0, rel=1e-12)


def test_partial_capture_is_reported():
    xx, yy = _grid()
    f0 = _gaussian(xx, yy)
    narrow = _gaussian(xx, yy, w=0.2)
    res = project_field_onto_modes(narrow, X, Y, [_mode(0, f0)], LAM)
    assert 0.0 < res.coupling_efficiency < 1.0
    assert res.residual_power > 0.0
    # reconstruction closes the budget
    captured = float(np.sum(np.abs(res.amplitudes) ** 2))
    assert res.launched_power == pytest.approx(captured + res.residual_power)


def test_component_selection_uses_the_right_transverse_field():
    xx, yy = _grid()
    f0 = _gaussian(xx, yy)
    mode = _mode(0, f0, ex=3.0 * f0)
    res_y = project_field_onto_modes(f0, X, Y, [mode], LAM, component="y")
    res_x = project_field_onto_modes(3.0 * f0, X, Y, [mode], LAM, component="x")
    # both amplitudes scale with the launched field, so the ratio is 3
    assert res_y.coupling_efficiency == pytest.approx(1.0, rel=1e-12)
    assert res_x.coupling_efficiency == pytest.approx(1.0, rel=1e-12)
    assert np.isclose(res_x.amplitudes[0] / res_y.amplitudes[0], 3.0)


def test_shape_mismatch_raises():
    xx, yy = _grid()
    with pytest.raises(ValueError, match="does not match"):
        project_field_onto_modes(
            np.zeros((N, N - 1), dtype=complex),
            X,
            Y,
            [_mode(0, _gaussian(xx, yy))],
            LAM,
        )


def test_grid_mismatch_raises():
    xx, yy = _grid()
    with pytest.raises(ValueError, match="export grid"):
        project_field_onto_modes(
            _gaussian(xx, yy), X + 1e-3, Y, [_mode(0, _gaussian(xx, yy))], LAM
        )


def test_zero_launch_raises():
    xx, yy = _grid()
    with pytest.raises(ValueError, match="zero"):
        project_field_onto_modes(
            np.zeros((N, N), dtype=complex), X, Y, [_mode(0, _gaussian(xx, yy))], LAM
        )
