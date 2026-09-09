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


def test_wavelength_array_sequence_protocol():
    wl = WavelengthArray(np.array([1500.0, 1600.0, 1700.0]), "nm")
    assert len(wl) == 3
    assert wl[0].as_nm == pytest.approx(1500.0)
    assert wl[1].as_nm == pytest.approx(1600.0)
    indexed = wl[1:3]
    assert len(indexed) == 2
    assert indexed[0].as_nm == pytest.approx(1600.0)
    from_wls = WavelengthArray.from_wavelengths([Wavelength(1300, "nm"), Wavelength(1400, "nm")])
    assert len(from_wls) == 2
    assert from_wls[1].as_nm == pytest.approx(1400.0)


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
    assert eq[0] == f.as_Hz.min()
    assert eq[-1] == f.as_Hz.max()


def test_angular_frequency_array_equally_spaced():
    omega = AngularFrequencyArray(np.array([2.0, 4.0]), "rad/ps")
    eq = omega.to_equally_spaced(points=51)
    assert len(eq) == 51
    assert eq[0] == omega.as_rad_s.min()
    assert eq[-1] == omega.as_rad_s.max()


def test_wavenumber_array_equally_spaced():
    wn = WavenumberArray(np.array([1e4, 2e4]), "1/cm")
    eq = wn.to_equally_spaced(points=51)
    assert len(eq) == 51
    assert eq[0] == wn.as_1_m.min()
    assert eq[-1] == wn.as_1_m.max()


# ─── Permittivity / Permiability ────────────────────────────────────


def test_permittivity_from_relative():

    p = Permittivity.from_relative(1.0)
    assert pytest.approx(p) == EPS_0


def test_permiability_from_relative():

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
        Wavelength(500, "pm" + "")  # type: ignore[arg-type]


def test_frequency_invalid_unit():
    with pytest.raises(ValidationError):
        Frequency(1.0, "kHz" + "")  # type: ignore[arg-type]


def test_angular_frequency_invalid_unit():
    with pytest.raises(ValidationError):
        AngularFrequency(1.0, "rad/ns" + "")  # type: ignore[arg-type]


def test_wavenumber_invalid_unit():
    with pytest.raises(ValidationError):
        Wavenumber(1.0, "1/nm" + "")  # type: ignore[arg-type]


def test_wavelength_array_invalid_unit():
    with pytest.raises(ValidationError):
        WavelengthArray([500.0], "pm" + "")  # type: ignore[arg-type]


def test_frequency_array_invalid_unit():
    with pytest.raises(ValidationError):
        FrequencyArray([1.0], "kHz" + "")  # type: ignore[arg-type]


def test_angular_frequency_array_invalid_unit():
    with pytest.raises(ValidationError):
        AngularFrequencyArray([1.0], "rad/ns" + "")  # type: ignore[arg-type]


def test_wavenumber_array_invalid_unit():
    with pytest.raises(ValidationError):
        WavenumberArray([1.0], "1/nm" + "")  # type: ignore[arg-type]


# ─── as_meep (MEEP units, λ₀ = 1 μm) ────────────────────────────────


def test_wavelength_as_meep():
    # f_Meep = a/λ: 500 nm → 1e-6/500e-9 = 2.0
    wl = Wavelength(500, "nm")
    assert pytest.approx(wl.as_meep) == 2.0

    # 1550 nm → 1e-6/1550e-9 ≈ 0.64516
    wl = Wavelength(1550, "nm")
    assert pytest.approx(wl.as_meep, rel=1e-5) == 1e-6 / 1550e-9

    # 1 μm → 1.0 (a/a = 1)
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
    # f_Meep = aω/(2πc): ω = 2πc/a ⇒ f_Meep = 1.0
    omega = AngularFrequency(2 * PI * C_MS / 1e-6, "rad/s")
    assert pytest.approx(omega.as_meep, rel=1e-10) == 1.0


def test_wavelength_array_as_meep():
    wl = WavelengthArray(np.array([400.0, 500.0, 600.0]), "nm")
    expected = 1e-6 / np.array([400.0, 500.0, 600.0]) * 1e9  # a/λ
    np.testing.assert_allclose(wl.as_meep, expected, rtol=1e-10)


def test_frequency_array_as_meep():
    f = FrequencyArray(np.array([300.0, 400.0, 500.0]), "THz")
    expected = np.array([300e12, 400e12, 500e12]) * 1e-6 / C_MS
    np.testing.assert_allclose(f.as_meep, expected, rtol=1e-10)


def test_angular_frequency_array_as_meep():
    omega = AngularFrequencyArray(
        np.array([2 * PI * C_MS / 1e-6, 2 * PI * C_MS / 0.5e-6]),
        "rad/s",
    )
    np.testing.assert_allclose(omega.as_meep, [1.0, 2.0], rtol=1e-10)


def test_meep_round_trip_wavelength():
    # wavelength → meep → back via a / f_Meep
    wl = Wavelength(800, "nm")
    meep = wl.as_meep
    assert pytest.approx(1e-6 / meep) == wl.as_m


def test_wavenumber_as_meep():
    # f_Meep = ak/(2π): k = 2π/1μm ⇒ f_Meep = 1.0
    wn = Wavenumber(2 * PI / 1e-6, "1/m")
    assert pytest.approx(wn.as_meep, rel=1e-10) == 1.0


def test_wavenumber_array_as_meep():
    wn = WavenumberArray(np.array([2 * PI / 1e-6, 2 * PI / 0.5e-6]), "1/m")
    np.testing.assert_allclose(wn.as_meep, [1.0, 2.0], rtol=1e-10)


# ─── from_meep tests ─────────────────────────────────────────────────


def test_wavelength_from_meep_default_base():
    w = Wavelength.from_meep(1.0)
    assert pytest.approx(w.as_um) == 1.0


def test_wavelength_from_meep_custom_base():
    w = Wavelength.from_meep(2.0, base_length=Wavelength(0.5, "um"))
    assert pytest.approx(w.as_um) == 0.25


def test_frequency_from_meep_default_base():
    f = Frequency.from_meep(1.0)
    assert pytest.approx(f.as_Hz) == 1.0 / 1e-6


def test_frequency_from_meep_custom_base():
    f = Frequency.from_meep(1.0, base_length=Wavelength(2.0, "um"))
    assert pytest.approx(f.as_Hz) == 1.0 / 2e-6


def test_angular_frequency_from_meep_default_base():
    om = AngularFrequency.from_meep(1.0)
    assert pytest.approx(om.as_rad_s) == 2 * PI / 1e-6


def test_angular_frequency_from_meep_custom_base():
    om = AngularFrequency.from_meep(1.0, base_length=Wavelength(0.5, "um"))
    assert pytest.approx(om.as_rad_s) == 2 * PI / 0.5e-6


def test_wavenumber_from_meep_default_base():
    wn = Wavenumber.from_meep(1.0)  # type: ignore[attr-defined]
    assert pytest.approx(wn.as_1_m) == 2 * PI / 1e-6


def test_wavenumber_from_meep_custom_base():
    wn = Wavenumber.from_meep(1.0, base_length=Wavelength(2.0, "um"))  # type: ignore[attr-defined]
    assert pytest.approx(wn.as_1_m) == 2 * PI / 2e-6


def test_wavelength_array_from_meep_default_base():
    wa = WavelengthArray.from_meep(np.array([1.0, 2.0]))
    np.testing.assert_allclose(wa.as_um, [1.0, 0.5], rtol=1e-10)


def test_wavelength_array_from_meep_custom_base():
    wa = WavelengthArray.from_meep(
        np.array([1.0, 2.0]), base_length=Wavelength(3.0, "um")
    )
    np.testing.assert_allclose(wa.as_um, [3.0, 1.5], rtol=1e-10)


def test_frequency_array_from_meep_default_base():
    fa = FrequencyArray.from_meep(np.array([1.0, 2.0]))
    np.testing.assert_allclose(fa.as_Hz, np.array([1.0, 2.0]) / 1e-6, rtol=1e-10)


def test_frequency_array_from_meep_custom_base():
    fa = FrequencyArray.from_meep(
        np.array([1.0, 2.0]), base_length=Wavelength(0.5, "um")
    )
    np.testing.assert_allclose(fa.as_Hz, np.array([1.0, 2.0]) / 0.5e-6, rtol=1e-10)


def test_angular_frequency_array_from_meep_default_base():
    oma = AngularFrequencyArray.from_meep(np.array([1.0, 2.0]))
    np.testing.assert_allclose(
        oma.as_rad_s, 2 * PI * np.array([1.0, 2.0]) / 1e-6, rtol=1e-10
    )


def test_angular_frequency_array_from_meep_custom_base():
    oma = AngularFrequencyArray.from_meep(
        np.array([1.0, 2.0]), base_length=Wavelength(0.5, "um")
    )
    np.testing.assert_allclose(
        oma.as_rad_s, 2 * PI * np.array([1.0, 2.0]) / 0.5e-6, rtol=1e-10
    )


def test_wavenumber_array_from_meep_default_base():
    wna = WavenumberArray.from_meep(np.array([1.0, 2.0]))
    np.testing.assert_allclose(
        wna.as_1_m, 2 * PI * np.array([1.0, 2.0]) / 1e-6, rtol=1e-10
    )


def test_wavenumber_array_from_meep_custom_base():
    wna = WavenumberArray.from_meep(
        np.array([1.0, 2.0]), base_length=Wavelength(2.0, "um")
    )
    np.testing.assert_allclose(
        wna.as_1_m, 2 * PI * np.array([1.0, 2.0]) / 2e-6, rtol=1e-10
    )





