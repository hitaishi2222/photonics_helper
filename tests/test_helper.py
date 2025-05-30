import pytest
import numpy as np
from photonics_helper import (
    Wavelength,
    Frequency,
    AngularFrequency,
    WavelengthArray,
    FrequencyArray,
    AngularFrequencyArray,
    C_MS,
    PI,
)


def test_wavelength_scalar():
    wl = Wavelength(1000, "nm")
    assert pytest.approx(wl.as_m) == 1e-6
    assert pytest.approx(wl.as_nm) == 1000


def test_wavelength_invalid_unit():
    with pytest.raises(ValueError):
        Wavelength(1000, "cm")


def test_wavelength_to_freq():
    wl = Wavelength(1550, "nm")
    freq = wl.to_freq()
    assert isinstance(freq, Frequency)
    assert pytest.approx(freq.as_Hz) == C_MS / 1.55e-6


def test_wavelength_to_angular_frequency():
    wl = Wavelength(1, "um")
    omega = wl.to_omega()
    expected = 2 * PI * C_MS / 1e-6
    assert isinstance(omega, AngularFrequency)
    assert pytest.approx(omega.as_rad_s) == expected


def test_frequency_scalar():
    f = Frequency(100, "GHz")
    assert pytest.approx(f.as_Hz) == 1e11
    assert pytest.approx(f.as_GHz) == 100


def test_frequency_invalid_unit():
    with pytest.raises(ValueError):
        Frequency(10, "kHz")


def test_frequency_to_wl():
    f = Frequency(193.414, "THz")
    wl = f.to_wl()
    expected = C_MS / (193.414e12)
    assert isinstance(wl, Wavelength)
    assert pytest.approx(wl.as_m) == expected


def test_omega_scalar():
    omega = AngularFrequency(628, "rad/ps")
    assert pytest.approx(omega.as_rad_s) == 628e-12


def test_omega_invalid_unit():
    with pytest.raises(ValueError):
        AngularFrequency(1, "deg/s")


def test_omega_to_freq():
    omega = AngularFrequency(2 * PI * 1e12, "rad/s")
    freq = omega.to_freq()
    assert isinstance(freq, Frequency)
    assert pytest.approx(freq.as_Hz) == 1e12


def test_omega_to_wl():
    omega = AngularFrequency(2 * PI * C_MS / 1.55e-6, "rad/s")
    wl = omega.to_wl()
    assert isinstance(wl, Wavelength)
    assert pytest.approx(wl.as_m) == 1.55e-6


def test_array_wavelength():
    data = np.array([1550, 1310])
    wl_arr = WavelengthArray(data, "nm")
    assert isinstance(wl_arr.to_freq(), FrequencyArray)
    assert isinstance(wl_arr.to_omega(), AngularFrequencyArray)


def test_array_frequency():
    data = np.array([100, 200])
    f_arr = FrequencyArray(data, "GHz")
    assert isinstance(f_arr.to_wl(), WavelengthArray)
    assert isinstance(f_arr.to_omega(), AngularFrequencyArray)


def test_array_omega():
    data = np.array([628e12, 314e12])
    omega_arr = AngularFrequencyArray(data, "rad/s")
    assert isinstance(omega_arr.to_freq(), FrequencyArray)
    assert isinstance(omega_arr.to_wl(), WavelengthArray)


def test_array_invalid_unit():
    with pytest.raises(ValueError):
        WavelengthArray(np.array([500]), "cm")
