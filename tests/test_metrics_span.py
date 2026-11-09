"""``span_at_level``: the bandwidth at a level, with everything it needs.

The metric exists because "N octaves at −M dB" is meaningless without the
numeraire, the level and the interpolation rule. These tests pin the three, and
pin that the answer is *resolution-visible* rather than resolution-free — a
number that hides its grid is the failure mode the module was written against.
"""

from __future__ import annotations

import numpy as np
import pytest

from photonics_helper.metrics import span_at_level, to_wavelength_density

C_MS = 299792458.0


def _log_gaussian(lam0_nm=2000.0, sigma_ln=0.25, n=4001, lo=500.0, hi=9000.0):
    """A spectrum whose −20 dB edges have a closed form on both numeraires."""
    wl = np.linspace(lo, hi, n)
    x = np.log(wl / lam0_nm)
    return wl, np.exp(-0.5 * (x / sigma_ln) ** 2)


def test_one_octave_is_one_octave():
    """1.0 → 2.0 µm must read 1.000 octaves at the level its own edges sit."""
    wl = np.linspace(500.0, 4000.0, 4001)
    density = np.where((wl >= 1000.0) & (wl <= 2000.0), 1.0, 0.0)
    span = span_at_level(wl, density, level_db=-0.5)
    assert span.octaves == pytest.approx(1.0, abs=5e-3)
    assert span.short_nm == pytest.approx(1000.0, abs=2.0)
    assert span.long_nm == pytest.approx(2000.0, abs=2.0)


def test_level_dependence_is_reported_not_hidden():
    wl, spec = _log_gaussian()
    loose = span_at_level(wl, spec, level_db=-10.0)
    tight = span_at_level(wl, spec, level_db=-40.0)
    assert tight.octaves > loose.octaves  # a lower floor is a wider band
    assert tight.level_db == -40.0 and loose.level_db == -10.0


def test_missing_crossing_is_nan_not_the_axis_edge():
    """A spectrum that never reaches the level must not report the window edge."""
    wl = np.linspace(1000.0, 2000.0, 401)
    peak = np.exp(-0.5 * (np.log(wl / 1500.0) / 0.05) ** 2)
    span = span_at_level(wl, peak, level_db=-80.0)  # tails are far above this
    assert np.isnan(span.short_nm) or np.isnan(span.long_nm)
    assert np.isnan(span.octaves)


def test_numeraire_moves_the_crossings_by_the_predicted_offset():
    """A per-frequency source read with and without the Jacobian.

    The Jacobian is ``|dω/dλ| = 2πc/λ²``, so the same array read as a per-λ
    density (no Jacobian, the historical mistake) and as a per-ν one (Jacobian
    applied) gives different crossings — worth up to ~13 dB between 2 µm and
    9 µm — and therefore different octave counts. A *log*-Gaussian is the
    degenerate case: the Jacobian is a power law, so it shifts that shape without
    distorting it and the octave count is unchanged. A Gaussian in frequency, the
    physical synthetic source, is the shape that exposes the difference.
    """
    wl = np.linspace(500.0, 9000.0, 8501)
    nu = C_MS / (wl * 1e-9)
    nu0 = C_MS / 2000e-9
    per_frequency_density = np.exp(-((nu - nu0) / (0.25 * nu0)) ** 2)

    # The helper is the Jacobian, exactly.
    jacobian = to_wavelength_density(wl, per_frequency_density)
    assert np.allclose(
        jacobian / per_frequency_density, C_MS * 2 * np.pi / (wl * 1e-9) ** 2
    )

    with_jacobian = span_at_level(
        wl, per_frequency_density, level_db=-20.0, numeraire="frequency"
    )
    without = span_at_level(wl, per_frequency_density, level_db=-20.0)
    assert with_jacobian.octaves != pytest.approx(without.octaves, abs=0.05)
    # The move at the long crossing is the Jacobian between the two readings.
    offset_db = 10.0 * np.log10((with_jacobian.long_nm / without.long_nm) ** 2)
    assert abs(offset_db) > 0.5


def test_the_numeraire_travels_with_the_number():
    wl, spec = _log_gaussian()
    span = span_at_level(wl, spec, numeraire="frequency")
    assert span.numeraire == "frequency"
    assert span.interpolation == "linear-in-db"
    assert span.level_db == -50.0  # the default is stated, not implied


def test_resolution_is_visible_in_the_answer():
    """One field, two grids: the spans differ and both say what their bin is."""
    wl_coarse = np.linspace(500.0, 9000.0, 851)  # 10 nm
    wl_fine = np.linspace(500.0, 9000.0, 8501)  # 1 nm
    x_c = np.log(wl_coarse / 2000.0)
    x_f = np.log(wl_fine / 2000.0)
    coarse = span_at_level(wl_coarse, np.exp(-0.5 * (x_c / 0.25) ** 2), level_db=-50.0)
    fine = span_at_level(wl_fine, np.exp(-0.5 * (x_f / 0.25) ** 2), level_db=-50.0)
    assert coarse.bin_nm == pytest.approx(10.0, rel=0.05)
    assert fine.bin_nm == pytest.approx(1.0, rel=0.05)
    assert coarse.octaves != fine.octaves
    assert abs(coarse.octaves - fine.octaves) < 0.1  # resolution, not disagreement


@pytest.mark.parametrize(
    "axis,spectrum,match",
    [
        (np.array([1.0, 2.0, 3.0]), np.ones(3), "1-D axis"),
        (np.linspace(1.0, 9.0, 20)[::-1], np.ones(20), "increasing"),
        (np.linspace(1.0, 9.0, 20), np.ones(19), "same length"),
    ],
)
def test_degenerate_inputs_raise(axis, spectrum, match):
    with pytest.raises(ValueError, match=match):
        span_at_level(axis, spectrum)
