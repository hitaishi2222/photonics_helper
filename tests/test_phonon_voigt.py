"""Voigt phonon lines with analytic area and Lorentzian compatibility."""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from photonics_helper.base import Time
from photonics_helper.phonon import PHONON_MATERIALS, PhononMode, PhononResponse
from photonics_helper.pulse import TemporalGrid
from photonics_helper.raman import RamanDatabase, RamanSpec

C_CM = 2.99792458e10
TWO_PI_C_CM = 2.0 * np.pi * C_CM


def _legacy_h_r(modes, t):
    """The pre-Voigt implementation, retained as a bit-compatibility oracle."""
    t = np.asarray(t, dtype=float)
    result = np.zeros_like(t)
    mask = t >= 0
    t_pos = t[mask]
    total = np.zeros_like(t_pos)
    normalizer = 0.0
    for mode in modes:
        shift_cm = float(mode.shift_cm.as_1_cm)
        gamma_cm = float(mode.linewidth_cm.as_1_cm)
        omega = TWO_PI_C_CM * shift_cm
        gamma = TWO_PI_C_CM * gamma_cm
        tau = 2.0 / gamma
        weight = float(mode.relative_strength)
        mode_integral = omega * tau**2 / (1.0 + (omega * tau) ** 2)
        total += weight * np.sin(omega * t_pos) * np.exp(-t_pos / tau)
        normalizer += weight * mode_integral
    grid_integral = np.trapezoid(total, t_pos)
    scale = grid_integral if grid_integral > 0.0 else normalizer
    if scale > 0.0:
        total /= scale
    result[mask] = total
    return result


def test_every_shipped_material_without_gaussian_width_is_bit_identical():
    grid = TemporalGrid(N=4096, Tmax=Time(20e-12, "s"))
    for name, modes in PHONON_MATERIALS.items():
        assert all(mode.fwhm_gauss_cm is None for mode in modes)
        old = _legacy_h_r(modes, grid.t)
        new = PhononResponse(modes).h_R(grid.t)
        assert np.array_equal(new, old), name
        assert np.array_equal(np.imag(grid.fft(new)), np.imag(grid.fft(old))), name


def test_zero_gaussian_width_is_continuous_with_lorentzian():
    # Fine sampling makes the legacy Lorentzian grid integral agree with its
    # analytic integral to much better than the 1e-6 comparison tolerance.
    t = np.linspace(0.0, 10e-12, 400001)
    lorentzian = PhononResponse(
        [PhononMode(shift_cm=440.0, linewidth_cm=45.0)]
    ).h_R(t)
    limit = PhononResponse(
        [PhononMode(shift_cm=440.0, linewidth_cm=45.0, fwhm_gauss_cm=1e-9)]
    ).h_R(t)
    assert np.max(np.abs(limit - lorentzian)) / np.max(np.abs(lorentzian)) <= 1e-6


def test_voigt_unit_area_has_analytic_cross_check():
    response = PhononResponse(
        [PhononMode(shift_cm=440.0, linewidth_cm=45.0, fwhm_gauss_cm=135.0)]
    )
    integral = quad(
        lambda t: response.h_R(np.array([t]))[0],
        0.0,
        2e-12,
        epsabs=1e-12,
        epsrel=1e-11,
        limit=300,
    )[0]
    assert abs(integral - 1.0) <= 1e-9

    # Independent quadrature of Lorentzian ⊗ Gaussian at the line centre.
    gamma, gauss = 45.0, 135.0
    sigma = gauss / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    centre = quad(
        lambda x: (0.5 * gamma / np.pi)
        / (x**2 + (0.5 * gamma) ** 2)
        * np.exp(-(x**2) / (2.0 * sigma**2))
        / (sigma * np.sqrt(2.0 * np.pi)),
        -np.inf,
        np.inf,
        epsabs=1e-12,
        epsrel=1e-11,
    )[0]
    np.testing.assert_approx_equal(
        response._lineshape(response.modes[0], np.array([440.0]))[0],
        centre,
        significant=10,
    )


def test_threefold_gaussian_width_lowers_imaginary_peak_to_four_tenths():
    grid = TemporalGrid(N=65536, Tmax=Time(20e-12, "s"))
    lor_modes = PHONON_MATERIALS["GeSbSe"]
    vgt_modes = [
        PhononMode(
            shift_cm=mode.shift_cm,
            linewidth_cm=mode.linewidth_cm,
            symmetry=mode.symmetry,
            relative_strength=mode.relative_strength,
            lo_phonon_cm=mode.lo_phonon_cm,
            to_phonon_cm=mode.to_phonon_cm,
            note=mode.note,
            fwhm_gauss_cm=3.0 * mode.linewidth_cm.as_1_cm,
        )
        for mode in lor_modes
    ]
    peak_lor = np.max(
        np.imag(grid.fft(PhononResponse(lor_modes).h_R(grid.t)))
    )
    peak_vgt = np.max(
        np.imag(grid.fft(PhononResponse(vgt_modes).h_R(grid.t)))
    )
    np.testing.assert_approx_equal(peak_vgt / peak_lor, 0.40, significant=2)


def test_gaussian_width_survives_raman_spec_and_database(tmp_path):
    mode = PhononMode(
        shift_cm=440.0, linewidth_cm=45.0, fwhm_gauss_cm=135.0
    )
    spec = RamanSpec(name="Voigt test", phonon_modes=[mode])
    response = spec.phonon_response
    assert response is not None
    assert response.modes[0].fwhm_gauss_cm.as_1_cm == 135.0

    db = RamanDatabase(db_path=tmp_path / "voigt.db")
    db.add_phonon_mode("Voigt test", mode)
    loaded = db.get_phonon_modes("Voigt test")
    assert loaded[0].fwhm_gauss_cm.as_1_cm == 135.0
