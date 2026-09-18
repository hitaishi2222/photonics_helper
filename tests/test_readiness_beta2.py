"""Regression tests for phase-matching readiness β₂ extraction (review N3).

The historical bug passed a ``Wavelength`` object to ``Dispersion.get_beta2``
(which expects nm), the resulting ``TypeError`` was swallowed, and the readiness
report silently reported ``N_sol = 0`` / ``L_D = inf``.
"""

import warnings

import numpy as np

from photonics_helper.base import Area, Length, Time, Wavelength, WavelengthArray
from photonics_helper.fiber import Dispersion, PropagationConstant
from photonics_helper.gnlse import FiberProfile
from photonics_helper.phase_matching import (
    _estimate_beta2,
    assess_simulation_readiness,
)
from photonics_helper.pulse import Envelope, TemporalGrid, Wave

import photonics_helper.phase_matching as pm


LAM = 1550e-9
BETA2_EXPECTED = 2.5508963989899877e-26  # -D λ²/(2πc) for D = -20 ps/(nm·km)


def _setup():
    grid = TemporalGrid(N=256, Tmax=Time(20e-12, "s"))
    pulse = Wave(
        grid=grid,
        envelope=Envelope(
            shape="sech", peak_amplitude=1.0, pulse_width=Time(100e-15, "s")
        ),
        central_wavelength=Wavelength(1550, "nm"),
    )
    fiber = FiberProfile(
        n2=6e-18,
        alpha=0.0,
        A_eff=Area(0.2e-12, "m^2"),
        length=Length(5e-3, "m"),
        confinement_factor=0.8,
    )
    return pulse, fiber


def _dispersion():
    wl = WavelengthArray(np.linspace(800, 2500, 200), "nm")
    return Dispersion(
        values=np.full(200, -20.0),
        unit="ps/nm.km",
        wavelengths=wl,
        central_wavelength=Wavelength(1550, "nm"),
    )


def _propagation_constant(pulse):
    wl = WavelengthArray(np.linspace(800, 2500, 200), "nm")
    om = wl.to_omega()
    om0 = float(pulse.central_frequency)
    beta = (
        1e7
        + 1e-29 * (om.as_rad_s - om0)
        + 0.5 * BETA2_EXPECTED * (om.as_rad_s - om0) ** 2
    )
    return PropagationConstant(values=beta, x_values=om)


def test_estimate_beta2_from_dispersion():
    pulse, _ = _setup()
    beta2 = _estimate_beta2(_dispersion(), pulse.central_frequency)
    assert beta2 is not None
    assert np.isclose(beta2, BETA2_EXPECTED, rtol=1e-3)


def test_readiness_dispersion_without_betas_is_populated():
    pulse, fiber = _setup()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = assess_simulation_readiness(pulse, fiber, _dispersion())
    assert np.isfinite(report.dispersion_length)
    assert report.soliton_order > 0
    assert report.predicted_processes


def test_estimate_beta2_from_propagation_constant():
    pulse, _ = _setup()
    beta2 = _estimate_beta2(_propagation_constant(pulse), pulse.central_frequency)
    assert beta2 is not None
    assert np.isclose(beta2, BETA2_EXPECTED, rtol=1e-2)


def test_readiness_propagation_constant_populated():
    pulse, fiber = _setup()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = assess_simulation_readiness(pulse, fiber, _propagation_constant(pulse))
    assert np.isfinite(report.dispersion_length)
    assert report.soliton_order > 0


def test_dw_root_finder_failure_warns(monkeypatch):
    pulse, fiber = _setup()
    monkeypatch.setattr(
        pm,
        "dispersive_wave_roots",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("forced")),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pm.assess_simulation_readiness(pulse, fiber, _dispersion())
    messages = [str(w.message) for w in caught]
    assert any("root finder failed" in m for m in messages)
