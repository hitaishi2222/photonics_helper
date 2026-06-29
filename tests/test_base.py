"""
Additional unit tests for the ``photonics_helper.base`` module.

These tests focus on the array‑type classes and their conversion methods,
ensuring that the returned objects have the correct type, shape and numerical
behaviour.
"""

import numpy as np
import pytest

from pydantic import ValidationError

from photonics_helper.base import (
    C_MS,
    EPS_0,
    MU_0,
    PI,
    Permiability,
    Permittivity,
    AngularFrequency,
    AngularFrequencyArray,
    Frequency,
    FrequencyArray,
    Wavelength,
    WavelengthArray,
    Wavenumber,
    WavenumberArray,
    MeepUnit,
    MeepUnitArray,
)


def test_wavelength_array_conversions():
    # Create a wavelength array (nanometres) and test conversions to other domains
    wl_vals_nm = np.array([800.0, 1550.0, 2000.0])
    wl_arr = WavelengthArray(wl_vals_nm, "nm")

    # Direct properties
    assert np.allclose(wl_arr.as_m, wl_vals_nm * 1e-9)
    assert np.allclose(wl_arr.as_um, wl_vals_nm * 1e-3)
    assert np.allclose(wl_arr.as_nm, wl_vals_nm)

    # Conversions to frequency, angular frequency and wavenumber
    freq_arr = wl_arr.to_freq()
    omega_arr = wl_arr.to_omega()
    wn_arr = wl_arr.to_wn()

    assert isinstance(freq_arr, FrequencyArray)
    assert isinstance(omega_arr, AngularFrequencyArray)
    assert isinstance(wn_arr, WavenumberArray)

    # Verify the numerical relationships
    expected_freq = C_MS / wl_arr.as_m
    expected_omega = 2 * PI * expected_freq
    expected_wn = 1.0 / wl_arr.as_m

    np.testing.assert_allclose(freq_arr.as_Hz, expected_freq, rtol=1e-12)
    np.testing.assert_allclose(omega_arr.as_rad_s, expected_omega, rtol=1e-12)
    np.testing.assert_allclose(wn_arr.as_1_m, expected_wn, rtol=1e-12)


def test_frequency_array_conversions():
    # Frequency array (THz) → wavelength and angular frequency
    f_vals_THz = np.array([200.0, 300.0])  # THz
    f_arr = FrequencyArray(f_vals_THz, "THz")

    wl_arr = f_arr.to_wl()
    omega_arr = f_arr.to_omega()
    wn_arr = f_arr.to_wn()

    assert isinstance(wl_arr, WavelengthArray)
    assert isinstance(omega_arr, AngularFrequencyArray)
    assert isinstance(wn_arr, WavenumberArray)

    expected_wl = C_MS / (f_arr.as_Hz)
    expected_omega = 2 * PI * f_arr.as_Hz
    expected_wn = f_arr.as_Hz / C_MS

    np.testing.assert_allclose(wl_arr.as_m, expected_wl, rtol=1e-12)
    np.testing.assert_allclose(omega_arr.as_rad_s, expected_omega, rtol=1e-12)
    np.testing.assert_allclose(wn_arr.as_1_m, expected_wn, rtol=1e-12)


def test_angular_frequency_array_conversions():
    # Angular frequency array (rad/ps) → wavelength, frequency, wavenumber
    omega_vals = np.array([2.0, 4.0])  # rad/ps
    omega_arr = AngularFrequencyArray(omega_vals, "rad/ps")

    wl_arr = omega_arr.to_wl()
    freq_arr = omega_arr.to_freq()
    wn_arr = omega_arr.to_wn()

    assert isinstance(wl_arr, WavelengthArray)
    assert isinstance(freq_arr, FrequencyArray)
    assert isinstance(wn_arr, WavenumberArray)

    # Convert rad/ps → rad/s for reference calculations
    omega_rad_s = omega_vals * 1e12

    expected_wl = (2 * PI * C_MS) / omega_rad_s
    expected_freq = omega_rad_s / (2 * PI)
    expected_wn = omega_rad_s / (2 * PI * C_MS)

    np.testing.assert_allclose(wl_arr.as_m, expected_wl, rtol=1e-12)
    np.testing.assert_allclose(freq_arr.as_Hz, expected_freq, rtol=1e-12)
    np.testing.assert_allclose(wn_arr.as_1_m, expected_wn, rtol=1e-12)


def test_wavenumber_array_conversions():
    # Wavenumber array (1/cm) → wavelength, frequency, angular frequency
    wn_vals = np.array([1e4, 2e4])  # 1/cm
    wn_arr = WavenumberArray(wn_vals, "1/cm")

    wl_arr = wn_arr.to_wl()
    freq_arr = wn_arr.to_freq()
    omega_arr = wn_arr.to_omega()

    assert isinstance(wl_arr, WavelengthArray)
    assert isinstance(freq_arr, FrequencyArray)
    assert isinstance(omega_arr, AngularFrequencyArray)

    # Convert to 1/m for reference
    wn_m = wn_vals * 1e2

    expected_wl = 1.0 / wn_m
    expected_freq = C_MS * wn_m
    expected_omega = 2 * PI * C_MS * wn_m

    np.testing.assert_allclose(wl_arr.as_m, expected_wl, rtol=1e-12)
    np.testing.assert_allclose(freq_arr.as_Hz, expected_freq, rtol=1e-12)
    np.testing.assert_allclose(omega_arr.as_rad_s, expected_omega, rtol=1e-12)


def test_scalar_units_consistency():
    # Verify that scalar classes round‑trip correctly via conversion methods.
    wl = Wavelength(1550, "nm")
    freq = wl.to_freq()
    omega = wl.to_omega()
    wn = wl.to_wn()

    # Back‑conversion should recover the original wavelength (within tolerance)
    np.testing.assert_allclose(freq.to_wl().as_m, wl.as_m, rtol=1e-12)
    np.testing.assert_allclose(omega.to_wl().as_m, wl.as_m, rtol=1e-12)
    np.testing.assert_allclose(wn.to_wl().as_m, wl.as_m, rtol=1e-12)

    # Frequency → angular frequency → back to frequency
    np.testing.assert_allclose(freq.to_omega().to_freq().as_Hz, freq.as_Hz, rtol=1e-12)

    # Angular frequency → wavenumber → back to angular frequency
    np.testing.assert_allclose(
        omega.to_wn().to_omega().as_rad_s, omega.as_rad_s, rtol=1e-12
    )


# ─── to_equally_spaced ───────────────────────────────────────────────


def test_wavelength_array_equally_spaced():
    wl = WavelengthArray(np.array([1500.0, 1600.0]), "nm")
    eq = wl.to_equally_spaced(points=51)
    assert len(eq) == 51
    assert pytest.approx(eq[0]) == wl.as_m.min()
    assert pytest.approx(eq[-1]) == wl.as_m.max()
    # Should be linearly spaced
    np.testing.assert_allclose(np.diff(eq), eq[1] - eq[0], rtol=1e-10, atol=1e-30)


def test_frequency_array_equally_spaced():
    f = FrequencyArray(np.array([200.0, 300.0]), "THz")
    eq = f.to_equally_spaced(points=51)
    assert len(eq) == 51
    assert eq[0] == f.as_Hz.max()
    assert eq[-1] == f.as_Hz.min()


def test_angular_frequency_array_equally_spaced():
    omega = AngularFrequencyArray(np.array([2.0, 4.0]), "rad/ps")
    eq = omega.to_equally_spaced(points=51)
    assert len(eq) == 51
    assert eq[0] == omega.as_rad_s.max()
    assert eq[-1] == omega.as_rad_s.min()


def test_wavenumber_array_equally_spaced():
    wn = WavenumberArray(np.array([1e4, 2e4]), "1/cm")
    eq = wn.to_equally_spaced(points=51)
    assert len(eq) == 51
    assert eq[0] == wn.as_1_m.max()
    assert eq[-1] == wn.as_1_m.min()


# ─── Permittivity / Permiability ────────────────────────────────────


def test_permittivity_from_relative():
    from photonics_helper.base import Permittivity

    p = Permittivity.from_relative(1.0)
    assert pytest.approx(p) == EPS_0


def test_permiability_from_relative():
    from photonics_helper.base import Permiability

    p = Permiability.from_relative(1.0)
    assert pytest.approx(p) == MU_0


# ─── repr methods ────────────────────────────────────────────────────


def test_wavelength_array_repr():
    wl = WavelengthArray(np.array([1500.0, 1600.0]), "nm")
    r = repr(wl)
    assert "WavelengthArray" in r
    assert "from:" in r
    assert "to:" in r


def test_frequency_array_repr():
    f = FrequencyArray(np.array([100.0, 200.0]), "THz")
    r = repr(f)
    assert "FrequencyArray" in r


def test_angular_frequency_array_repr():
    omega = AngularFrequencyArray(np.array([2.0, 4.0]), "rad/ps")
    r = repr(omega)
    assert "AngularFrequencyArray" in r


def test_wavenumber_array_repr():
    wn = WavenumberArray(np.array([1e4, 2e4]), "1/cm")
    r = repr(wn)
    assert "WavenumberArray" in r


# ─── additional scalar tests ────────────────────────────────────────


def test_wavelength_as_um():
    wl = Wavelength(1550, "nm")
    assert pytest.approx(wl.as_um) == 1.55


def test_wavelength_as_nm():
    wl = Wavelength(1.55e-6, "m")
    assert pytest.approx(wl.as_nm) == 1550.0


def test_frequency_as_THz():
    f = Frequency(193.414e12, "Hz")
    assert pytest.approx(f.as_THz) == 193.414


def test_frequency_as_GHz():
    f = Frequency(1e9, "Hz")
    assert pytest.approx(f.as_GHz) == 1.0


def test_frequency_as_MHz():
    f = Frequency(1e6, "Hz")
    assert pytest.approx(f.as_MHz) == 1.0


def test_angular_frequency_as_rad_ps():
    omega = AngularFrequency(1e12, "rad/s")
    assert pytest.approx(omega.as_rad_ps) == 1.0


def test_wavenumber_as_1_cm():
    # 10000 1/m = 100 1/cm (1 m = 100 cm)
    wn = Wavenumber(10000, "1/m")
    assert pytest.approx(wn.as_1_cm) == 100.0


def test_wavenumber_as_angular():
    wn = Wavenumber(1.0, "1/m")
    assert pytest.approx(wn.as_angular) == 2 * PI


def test_wavelength_round_trip_via_wavenumber():
    wl = Wavelength(800, "nm")
    wn = wl.to_wn()
    wl_back = wn.to_wl()
    assert pytest.approx(wl_back.as_nm) == 800.0


def test_frequency_round_trip_via_wavenumber():
    f = Frequency(200, "THz")
    wn = f.to_wn()
    f_back = wn.to_freq()
    assert pytest.approx(f_back.as_THz) == 200.0


def test_angular_frequency_round_trip_via_wavenumber():
    omega = AngularFrequency(2 * PI * 193e12, "rad/s")
    wn = omega.to_wn()
    omega_back = wn.to_omega()
    assert pytest.approx(omega_back.as_rad_s, rel=1e-10) == omega.as_rad_s


# ─── Invalid unit handling ─────────────────────────────────────────


def test_wavelength_invalid_unit():
    with pytest.raises(ValidationError):
        Wavelength(500, "pm")


def test_frequency_invalid_unit():
    with pytest.raises(ValidationError):
        Frequency(1.0, "kHz")


def test_angular_frequency_invalid_unit():
    with pytest.raises(ValidationError):
        AngularFrequency(1.0, "rad/ns")


def test_wavenumber_invalid_unit():
    with pytest.raises(ValidationError):
        Wavenumber(1.0, "1/nm")


def test_wavelength_array_invalid_unit():
    with pytest.raises(ValidationError):
        WavelengthArray([500.0], "pm")


def test_frequency_array_invalid_unit():
    with pytest.raises(ValidationError):
        FrequencyArray([1.0], "kHz")


def test_angular_frequency_array_invalid_unit():
    with pytest.raises(ValidationError):
        AngularFrequencyArray([1.0], "rad/ns")


def test_wavenumber_array_invalid_unit():
    with pytest.raises(ValidationError):
        WavenumberArray([1.0], "1/nm")


# ─── as_meep (MEEP units, λ₀ = 1 μm) ────────────────────────────────


def test_wavelength_as_meep():
    # 500 nm → 0.5 in MEEP units (λ₀ = 1 μm)
    wl = Wavelength(500, "nm")
    assert pytest.approx(wl.as_meep) == 0.5

    # 1550 nm → 1.55
    wl = Wavelength(1550, "nm")
    assert pytest.approx(wl.as_meep) == 1.55

    # 1 μm → 1.0
    wl = Wavelength(1.0, "um")
    assert pytest.approx(wl.as_meep) == 1.0


def test_frequency_as_meep():
    # f = c / λ₀ ⇒ c / 1e-6 ≈ 2.998e14 Hz
    f = Frequency(C_MS / 1e-6, "Hz")
    assert pytest.approx(f.as_meep, rel=1e-10) == 1.0

    # 300 THz → ~1.0
    f = Frequency(300e12, "Hz")
    assert pytest.approx(f.as_meep, rel=1e-10) == 300e12 * 1e-6 / C_MS


def test_angular_frequency_as_meep():
    # ω = 2πc/λ₀ ⇒ ω * λ₀ / c = 2π
    omega = AngularFrequency(2 * PI * C_MS / 1e-6, "rad/s")
    assert pytest.approx(omega.as_meep, rel=1e-10) == 2 * PI


def test_wavelength_array_as_meep():
    wl = WavelengthArray(np.array([400.0, 500.0, 600.0]), "nm")
    np.testing.assert_allclose(wl.as_meep, [0.4, 0.5, 0.6])


def test_frequency_array_as_meep():
    f = FrequencyArray(np.array([300.0, 400.0, 500.0]), "THz")
    expected = np.array([300e12, 400e12, 500e12]) * 1e-6 / C_MS
    np.testing.assert_allclose(f.as_meep, expected, rtol=1e-10)


def test_angular_frequency_array_as_meep():
    omega = AngularFrequencyArray(
        np.array([2 * PI * C_MS / 1e-6, 2 * PI * C_MS / 0.5e-6]),
        "rad/s",
    )
    np.testing.assert_allclose(omega.as_meep, [2 * PI, 4 * PI], rtol=1e-10)


def test_meep_round_trip_wavelength():
    # wavelength → meep → back via as_meep * λ₀
    wl = Wavelength(800, "nm")
    meep = wl.as_meep
    assert pytest.approx(meep * 1e-6) == wl.as_m


# ─── MeepUnit (scalar) ───────────────────────────────────────────────


def test_meepunit_basic():
    m = MeepUnit(1.0)
    assert pytest.approx(m.as_meep) == 1.0


def test_meepunit_repr():
    m = MeepUnit(1.55)
    r = repr(m)
    assert "MeepUnit" in r
    assert "1.55" in r


def test_meepunit_str():
    m = MeepUnit(1.55)
    s = str(m)
    assert "1.55" in s
    assert "MEEP" in s


def test_meepunit_to_wl():
    m = MeepUnit(1.55)
    wl = m.to_wl()
    assert isinstance(wl, Wavelength)
    assert pytest.approx(wl.as_um) == 1.55
    assert pytest.approx(wl.as_nm) == 1550.0


def test_meepunit_to_freq():
    m = MeepUnit(1.0)
    freq = m.to_freq()
    assert isinstance(freq, Frequency)
    # λ = 1 μm → f = c / 1e-6 ≈ 2.998e14 Hz
    expected = C_MS / 1e-6
    assert pytest.approx(freq.as_Hz, rel=1e-10) == expected


def test_meepunit_to_omega():
    m = MeepUnit(1.0)
    omega = m.to_omega()
    assert isinstance(omega, AngularFrequency)
    expected = 2 * PI * C_MS / 1e-6
    assert pytest.approx(omega.as_rad_s, rel=1e-10) == expected


def test_meepunit_to_wn():
    m = MeepUnit(1.0)
    wn = m.to_wn()
    assert isinstance(wn, Wavenumber)
    assert pytest.approx(wn.as_1_m, rel=1e-10) == 1e6


# ─── MeepUnitArray ──────────────────────────────────────────────────


def test_meepunitarray_basic():
    ma = MeepUnitArray([0.5, 1.0, 1.5])
    assert np.allclose(ma.as_meep, [0.5, 1.0, 1.5])


def test_meepunitarray_repr():
    ma = MeepUnitArray([0.5, 1.0, 1.5])
    r = repr(ma)
    assert "MeepUnitArray" in r
    assert "from:" in r
    assert "to:" in r


def test_meepunitarray_to_wl():
    ma = MeepUnitArray([0.8, 1.0, 1.2])
    wl = ma.to_wl()
    assert isinstance(wl, WavelengthArray)
    assert np.allclose(wl.as_um, [0.8, 1.0, 1.2])
    assert np.allclose(wl.as_nm, [800.0, 1000.0, 1200.0])


def test_meepunitarray_to_freq():
    ma = MeepUnitArray([1.0, 2.0])
    freq = ma.to_freq()
    assert isinstance(freq, FrequencyArray)
    expected = C_MS / np.array([1e-6, 2e-6])
    np.testing.assert_allclose(freq.as_Hz, expected, rtol=1e-10)


def test_meepunitarray_to_omega():
    ma = MeepUnitArray([1.0, 2.0])
    omega = ma.to_omega()
    assert isinstance(omega, AngularFrequencyArray)
    expected = 2 * PI * C_MS / np.array([1e-6, 2e-6])
    np.testing.assert_allclose(omega.as_rad_s, expected, rtol=1e-10)


def test_meepunitarray_to_wn():
    ma = MeepUnitArray([1.0, 2.0])
    wn = ma.to_wn()
    assert isinstance(wn, WavenumberArray)
    expected = 1.0 / np.array([1e-6, 2e-6])
    np.testing.assert_allclose(wn.as_1_m, expected, rtol=1e-10)


# ─── MeepUnit ↔ Wavelength round-trip ──────────────────────────────


def test_meepunit_round_trip_wavelength():
    # MeepUnit → Wavelength → MeepUnit
    m = MeepUnit(1.55)
    wl = m.to_wl()
    m_back = Wavelength(wl.as_um, "um")
    assert pytest.approx(m_back.as_meep) == m.as_meep


def test_meepunit_round_trip_frequency():
    m = MeepUnit(1.0)
    freq = m.to_freq()
    freq_back = Frequency(freq.as_Hz, "Hz")
    assert pytest.approx(freq_back.as_meep, rel=1e-10) == m.as_meep


def test_meepunit_round_trip_angular_frequency():
    """MeepUnit → AngularFrequency → Wavelength recovers original μm value."""
    for val in [0.5, 1.0, 1.55]:
        m = MeepUnit(val)
        omega = m.to_omega()
        wl = omega.to_wl()
        assert pytest.approx(wl.as_um) == val


def test_meepunit_round_trip_wavenumber():
    """MeepUnit → Wavenumber → Wavelength recovers original μm value."""
    for val in [0.5, 1.0, 1.55]:
        m = MeepUnit(val)
        wn = m.to_wn()
        wl = wn.to_wl()
        assert pytest.approx(wl.as_um) == val


# ─── MeepUnitArray round-trip ──────────────────────────────────────


def test_meepunitarray_round_trip_wavelength():
    ma = MeepUnitArray([0.8, 1.0, 1.2])
    wl = ma.to_wl()
    wl_back = WavelengthArray(wl.as_um, "um")
    np.testing.assert_allclose(wl_back.as_meep, ma.as_meep, rtol=1e-12)


def test_meepunitarray_round_trip_frequency():
    """MeepUnitArray → FrequencyArray → WavelengthArray recovers μm values."""
    for val in [0.5, 1.0, 1.5]:
        ma = MeepUnitArray([val])
        freq = ma.to_freq()
        wl = freq.to_wl()
        assert pytest.approx(wl.as_um[0]) == val


def test_meepunitarray_round_trip_angular_frequency():
    """MeepUnitArray → AngularFrequencyArray → WavelengthArray recovers μm values."""
    for val in [0.5, 1.0, 1.5]:
        ma = MeepUnitArray([val])
        omega = ma.to_omega()
        wl = omega.to_wl()
        assert pytest.approx(wl.as_um[0]) == val


def test_meepunitarray_round_trip_wavenumber():
    """MeepUnitArray → WavenumberArray → WavelengthArray recovers μm values."""
    for val in [0.5, 1.0, 1.5]:
        ma = MeepUnitArray([val])
        wn = ma.to_wn()
        wl = wn.to_wl()
        assert pytest.approx(wl.as_um[0]) == val


# ─── MeepUnit ↔ as_meep consistency ────────────────────────────────


def test_meepunit_as_meep_matches_wavelength_as_meep():
    """MeepUnit(x).as_meep should equal Wavelength(x um).as_meep."""
    for val in [0.5, 1.0, 1.55, 2.0]:
        m = MeepUnit(val)
        wl = Wavelength(val, "um")
        assert pytest.approx(m.as_meep) == wl.as_meep


def test_meepunit_as_meep_matches_frequency_as_meep():
    """MeepUnit(x) and Frequency(c/(x·1e-6) Hz) describe the same mode.

    Frequency.as_meep = f·1e-6/C_MS = 1/x, MeepUnit(x).as_meep = x.
    Their product should be 1 (reciprocal relationship in MEEP units).
    """
    for val in [0.5, 1.0, 1.55]:
        m = MeepUnit(val)
        f = Frequency(C_MS / (val * 1e-6), "Hz")
        assert pytest.approx(m.as_meep * f.as_meep) == 1.0


def test_meepunit_as_meep_matches_angular_frequency_as_meep():
    """MeepUnit(x) and AngularFrequency(2πc/(x·1e-6) rad/s): product of as_meep = 2π."""
    for val in [0.5, 1.0, 1.55]:
        m = MeepUnit(val)
        omega = AngularFrequency(2 * PI * C_MS / (val * 1e-6), "rad/s")
        assert pytest.approx(m.as_meep * omega.as_meep) == 2 * PI


def test_meepunit_as_meep_matches_wavenumber_as_meep():
    """MeepUnit(x) and Wavenumber(1/(x·1e-6) 1/m): reciprocal relationship."""
    for val in [0.5, 1.0, 1.55]:
        m = MeepUnit(val)
        wn = Wavenumber(1 / (val * 1e-6), "1/m")
        assert pytest.approx(wn.as_1_m * m.as_meep * 1e-6) == 1.0



