"""Regression tests for MEEP unit conversions (review N4).

MEEP uses dimensionless units with ``c = 1`` and length unit ``a``, so
``f_meep = a/λ = νa/c = ωa/(2πc)``.  The historical ``from_meep`` helpers for
frequency/angular frequency omitted the factor ``c`` and ``Power.from_meep`` was
not the inverse of ``Power.as_meep``.
"""

import numpy as np
import pytest

from photonics_helper.base import (
    C_MS,
    Area,
    AngularFrequency,
    AngularFrequencyArray,
    Energy,
    Frequency,
    FrequencyArray,
    Length,
    Power,
    Time,
    Wavelength,
    WavelengthArray,
    Wavenumber,
    WavenumberArray,
)


SCALARS = [
    Wavelength(1550, "nm"),
    Frequency(193.4, "THz"),
    AngularFrequency(1.215e15, "rad/s"),
    Wavenumber(6450.0, "1/cm"),
    Energy(1.0, "eV"),
    Power(1.0, "W"),
    Time(1.0, "ps"),
    Length(1.0, "um"),
    Area(1.0, "um^2"),
]


@pytest.mark.parametrize("obj", SCALARS, ids=[type(o).__name__ for o in SCALARS])
def test_meep_round_trip_scalar(obj):
    meep = float(obj.as_meep)
    recovered = type(obj).from_meep(meep)
    assert np.isclose(float(recovered.value), float(obj.value), rtol=1e-10, atol=0.0)


def test_frequency_from_meep_uses_speed_of_light():
    """1 MEEP frequency unit at a = 1 µm is c/1µm = 3×10¹⁴ Hz."""
    f = Frequency.from_meep(1.0)
    assert np.isclose(f.as_Hz, C_MS / 1e-6, rtol=1e-12)


def test_wavelength_frequency_meep_agree():
    """At a = 1 µm the MEEP value equals a/λ = 1/1.55 for wavelength and frequency."""
    lam = 1.55e-6
    f = Frequency(C_MS / lam, "Hz")
    assert np.isclose(Wavelength(lam, "m").as_meep, 1.0 / 1.55, rtol=1e-10)
    assert np.isclose(Wavelength(lam, "m").as_meep, f.as_meep, rtol=1e-10)


def test_energy_time_power_consistency():
    """P_meep == E_meep / t_meep for P = E / t."""
    E = Energy(1.0, "J")
    t = Time(1.0, "s")
    P = Power(1.0, "W")
    assert np.isclose(P.as_meep, E.as_meep / t.as_meep, rtol=1e-10)


def test_power_from_meep_round_trip():
    p = Power(2.5, "mW")
    assert np.isclose(Power.from_meep(p.as_meep).as_W, p.as_W, rtol=1e-10)


def test_array_round_trips():
    wl = WavelengthArray(np.array([400.0, 500.0, 1550.0]), "nm")
    fa = FrequencyArray(np.array([300.0, 400.0, 500.0]), "THz")
    oma = AngularFrequencyArray(np.array([1e15, 1.5e15, 2e15]), "rad/s")
    wna = WavenumberArray(np.array([1000.0, 5000.0, 10000.0]), "1/cm")

    np.testing.assert_allclose(
        WavelengthArray.from_meep(wl.as_meep).as_m, wl.as_m, rtol=1e-10
    )
    np.testing.assert_allclose(
        FrequencyArray.from_meep(fa.as_meep).as_Hz, fa.as_Hz, rtol=1e-10
    )
    np.testing.assert_allclose(
        AngularFrequencyArray.from_meep(oma.as_meep).as_rad_s,
        oma.as_rad_s,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        WavenumberArray.from_meep(wna.as_meep).as_1_m, wna.as_1_m, rtol=1e-10
    )
