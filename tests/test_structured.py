"""Tests for :mod:`photonics_helper.structured`.

Covers the Laguerre-Gaussian / OAM transverse modes and helpers:

- orthonormality of the normalized LG modes,
- OAM phase winding ``2 pi l`` around the optical axis,
- waist / Gouy / radius-of-curvature propagation against the analytic laws,
- the :class:`StructuredField` container and its discrete overlap integral,
- transverse-profile plotting (matplotlib + optional plotly).
"""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose

from photonics_helper.structured import (
    LaguerreGaussianMode,
    StructuredField,
    beam_waist,
    gouy_phase,
    overlap,
    plot_transverse_profile,
    radius_of_curvature,
    rayleigh_range,
)

W0 = 1e-3  # 1 mm waist
WAVELENGTH = 1064e-9  # 1064 nm


def lg(p: int, l: int, **kwargs) -> LaguerreGaussianMode:  # noqa: E741
    """Shorthand for a mode at the default 1 mm / 1064 nm test geometry."""
    return LaguerreGaussianMode(p, l, W0, WAVELENGTH, **kwargs)


# ---------------------------------------------------------------------------
# Orthonormality
# ---------------------------------------------------------------------------
def test_orthonormality_same_mode():
    """``overlap(LG(0,1), LG(0,1))`` is 1 for normalized modes."""
    result = overlap(lg(0, 1), lg(0, 1))
    assert abs(result - 1.0) < 1e-9


def test_orthogonality_different_oam():
    """``overlap(LG(0,1), LG(0,2))`` vanishes (orthogonal OAM)."""
    result = overlap(lg(0, 1), lg(0, 2))
    assert abs(result) < 1e-6


@pytest.mark.parametrize(
    "a,b",
    [
        ((0, 0), (1, 0)),
        ((0, 0), (2, 0)),
        ((0, 1), (1, 1)),
        ((1, 0), (0, 2)),
        ((0, 3), (2, 3)),
    ],
)
def test_orthogonality_general(a, b):
    """Distinct LG modes are orthogonal to numerical precision."""
    assert abs(overlap(lg(*a), lg(*b))) < 1e-6


def test_overlap_accepts_fields():
    """``overlap`` works on pre-sampled :class:`StructuredField` inputs."""
    fa = lg(0, 1).structured()
    fb = lg(0, 2).structured()
    assert abs(overlap(fa, fa) - 1.0) < 1e-9
    assert abs(overlap(fa, fb)) < 1e-6


def test_overlap_rejects_mixed_types():
    with pytest.raises(TypeError, match="two LaguerreGaussianMode"):
        overlap(lg(0, 1), lg(0, 1).structured())


def test_overlap_rejects_grid_mismatch():
    a = lg(0, 1).structured(n=129)
    b = lg(0, 1).structured(n=257)
    with pytest.raises(ValueError, match="same grid shape"):
        overlap(a, b)


# ---------------------------------------------------------------------------
# OAM phase winding
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("l", [1, 2, 3, -1, -2])
def test_oam_phase_winding(l):  # noqa: E741
    """Phase accumulated around a closed loop enclosing the axis is ``2 pi l``."""
    mode = lg(0, l)  # p=0 keeps the radial factor strictly positive
    phi = np.linspace(0.0, 2.0 * np.pi, 2881)
    radius = 2.0 * W0
    values = mode.evaluate(radius * np.cos(phi), radius * np.sin(phi))
    unwrapped = np.unwrap(np.angle(values))
    winding = float(unwrapped[-1] - unwrapped[0])
    assert abs(winding - 2.0 * np.pi * l) < 1e-3


def test_oam_phase_winding_higher_radial_order():
    """The winding is ``2 pi l`` for p > 0 too, sampling outside the last zero."""
    mode = lg(1, 1)
    phi = np.linspace(0.0, 2.0 * np.pi, 2881)
    radius = 3.0 * W0
    values = mode.evaluate(radius * np.cos(phi), radius * np.sin(phi))
    unwrapped = np.unwrap(np.angle(values))
    winding = float(unwrapped[-1] - unwrapped[0])
    assert abs(winding - 2.0 * np.pi) < 1e-3


# ---------------------------------------------------------------------------
# Propagation helpers
# ---------------------------------------------------------------------------
def test_rayleigh_range_and_waist():
    z_r = rayleigh_range(W0, WAVELENGTH)
    assert_allclose(z_r, np.pi * W0**2 / WAVELENGTH, rtol=1e-12)
    assert_allclose(beam_waist(W0, 0.0, WAVELENGTH), W0, rtol=1e-12)
    assert_allclose(beam_waist(W0, z_r, WAVELENGTH), np.sqrt(2.0) * W0, rtol=1e-12)
    assert_allclose(beam_waist(W0, -z_r, WAVELENGTH), np.sqrt(2.0) * W0, rtol=1e-12)


def test_radius_of_curvature_and_gouy():
    z_r = rayleigh_range(W0, WAVELENGTH)
    assert np.isinf(radius_of_curvature(0.0, z_r))
    assert_allclose(radius_of_curvature(z_r, z_r), 2.0 * z_r, rtol=1e-12)
    assert_allclose(gouy_phase(3, z_r, z_r), 3.0 * np.pi / 4.0, rtol=1e-12)


def test_mode_waist_matches_analytic_law():
    """``mode.waist(z)`` follows ``w0 sqrt(1 + (z/z_R)^2)``."""
    mode = lg(1, 2)
    z_r = mode.z_r
    for z in (0.0, 0.5 * z_r, z_r, 2.0 * z_r, -z_r, -3.0 * z_r):
        expected = W0 * np.sqrt(1.0 + (z / z_r) ** 2)
        assert_allclose(mode.waist(z), expected, rtol=1e-12)


def test_second_moment_radius_tracks_waist():
    """The sampled field's RMS radius scales exactly as ``w(z) / w0``."""
    mode = lg(1, 2)
    z_r = mode.z_r
    rms0 = mode.structured().second_moment_radius()
    order = 2 * 1 + abs(2) + 1
    assert_allclose(rms0, W0 * np.sqrt(order / 2.0), rtol=1e-4)

    for z in (0.0, 0.5 * z_r, z_r, 2.0 * z_r, -z_r, 5.0 * z_r):
        field = mode.structured(z=z)
        ratio = field.second_moment_radius() / rms0
        analytic = beam_waist(W0, z, WAVELENGTH) / W0
        assert_allclose(ratio, analytic, rtol=1e-3)


def test_mode_at_waist_needs_no_wavelength():
    """A mode without a wavelength can still be sampled at ``z = 0``."""
    mode = LaguerreGaussianMode(0, 1, W0)
    field = mode.structured()
    assert field.power == pytest.approx(1.0, rel=1e-9)
    with pytest.raises(ValueError, match="wavelength is required"):
        mode.structured(z=W0)
    with pytest.raises(ValueError, match="wavelength is required"):
        _ = mode.z_r


# ---------------------------------------------------------------------------
# StructuredField container
# ---------------------------------------------------------------------------
def test_structured_field_power_and_normalization():
    field = lg(0, 1).structured()
    assert field.shape == (257, 257)
    assert field.intensity.shape == field.shape
    assert field.power == pytest.approx(1.0, rel=1e-9)
    assert field.normalize().power == pytest.approx(1.0, rel=1e-12)


def test_structured_field_geometry_and_centroid():
    field = lg(0, 0).structured(half_width=5 * W0, n=201)
    assert_allclose(field.spacing, (field.dx, field.dx))
    xc, yc = field.centroid()
    assert abs(xc) < 1e-12
    assert abs(yc) < 1e-12
    xmin, xmax, ymin, ymax = field.extent
    assert xmin == pytest.approx(-5 * W0 - field.dx / 2.0)
    assert xmax == pytest.approx(5 * W0 + field.dx / 2.0)
    assert (xmax - xmin) == pytest.approx((ymax - ymin))


def test_structured_field_defaults_dy_to_dx():
    field = StructuredField(np.ones((4, 5), dtype=complex), dx=0.1)
    assert field.spacing == (0.1, 0.1)
    assert field.nx == 5
    assert field.ny == 4


def test_structured_field_validation():
    with pytest.raises(ValueError, match="2-D"):
        StructuredField(np.ones(4, dtype=complex), dx=0.1)
    with pytest.raises(ValueError, match="positive"):
        StructuredField(np.ones((4, 4), dtype=complex), dx=0.0)
    zero = StructuredField(np.zeros((4, 4), dtype=complex), dx=0.1)
    with pytest.raises(ValueError, match="zero-power"):
        zero.normalize()
    with pytest.raises(ValueError, match="zero-power"):
        zero.centroid()


def test_mode_validation():
    with pytest.raises(ValueError, match="p must be"):
        LaguerreGaussianMode(-1, 0, W0)
    with pytest.raises(ValueError, match="l must be"):
        LaguerreGaussianMode(0, 1.5, W0)
    with pytest.raises(ValueError, match="w0 must be"):
        LaguerreGaussianMode(0, 0, 0.0)
    with pytest.raises(ValueError, match="wavelength must be"):
        LaguerreGaussianMode(0, 0, W0, -1.0)


def test_field_grid_shape():
    mode = lg(0, 1)
    x = np.linspace(-W0, W0, 11)
    y = np.linspace(-2 * W0, 2 * W0, 7)
    field = mode.field(x, y)
    assert field.shape == (7, 11)
    with pytest.raises(ValueError, match="1-D"):
        mode.field(np.zeros((2, 2)), x)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def test_plot_matplotlib_returns_figure():
    fig = plot_transverse_profile(lg(1, 2))
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_from_field_and_styles():
    field = lg(0, 3).structured()
    fig = field.plot(theme="dark", title="custom", figsize=(6, 3))
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_plotly_returns_figure():
    pytest.importorskip("plotly")
    fig = plot_transverse_profile(lg(0, 1), backend="plotly")
    assert hasattr(fig, "to_html")
    assert len(fig.data) == 2


def test_plot_invalid_arguments():
    with pytest.raises(ValueError, match="backend"):
        plot_transverse_profile(lg(0, 1), backend="bokeh")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="theme"):
        plot_transverse_profile(lg(0, 1), theme="neon")  # type: ignore[arg-type]
