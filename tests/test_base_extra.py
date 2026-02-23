"""
Additional unit tests for the ``photonics_helper.base`` module.

These tests focus on the array‑type classes and their conversion methods,
ensuring that the returned objects have the correct type, shape and numerical
behaviour.
"""

import numpy as np
import pytest

from photonics_helper.base import (
    C_MS,
    PI,
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
