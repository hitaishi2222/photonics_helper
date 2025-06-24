import pytest
import numpy as np
from photonics_helper import (
    Wavelength,
    Frequency,
    AngularFrequency,
    Wavenumber,
    WavelengthArray,
    FrequencyArray,
    AngularFrequencyArray,
    WavenumberArray,
    C_MS,
    PI,
)

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal


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


class TestWavenumber:
    """Test suite for Wavenumber class"""

    def test_wavenumber_init_1_m(self):
        """Test Wavenumber initialization with 1/m units"""
        wn = Wavenumber(1e6, "1/m")
        assert pytest.approx(wn.as_1_m) == 1e6
        assert pytest.approx(wn.as_1_cm) == 1e4

    def test_wavenumber_init_1_cm(self):
        """Test Wavenumber initialization with 1/cm units"""
        wn = Wavenumber(1e4, "1/cm")
        assert pytest.approx(wn.as_1_m) == 1e6
        assert pytest.approx(wn.as_1_cm) == 1e4

    def test_wavenumber_invalid_unit(self):
        """Test error handling for invalid units"""
        with pytest.raises(ValueError, match="Unsupported unit"):
            Wavenumber(1000, "1/mm")

    def test_wavenumber_angular_property(self):
        """Test angular wavenumber property"""
        wn = Wavenumber(1e6, "1/m")
        expected = 1e6 * 2 * PI
        assert pytest.approx(wn.as_angular) == expected

    def test_wavenumber_to_wavelength(self):
        """Test conversion to wavelength"""
        wn = Wavenumber(1e6, "1/m")
        wl = wn.to_wl()
        assert isinstance(wl, Wavelength)
        assert pytest.approx(wl.as_m) == 1e-6

    def test_wavenumber_to_frequency(self):
        """Test conversion to frequency"""
        wn = Wavenumber(1e6, "1/m")
        freq = wn.to_freq()
        assert isinstance(freq, Frequency)
        expected = C_MS * 1e6
        assert pytest.approx(freq.as_Hz) == expected

    def test_wavenumber_to_omega(self):
        """Test conversion to angular frequency"""
        wn = Wavenumber(1e6, "1/m")
        omega = wn.to_omega()
        assert isinstance(omega, AngularFrequency)
        expected = C_MS * 2 * PI * 1e6
        assert pytest.approx(omega.as_rad_s) == expected


class TestWavenumberArray:
    """Test suite for WavenumberArray class"""

    def test_wavenumber_array_init_1_m(self):
        """Test WavenumberArray initialization with 1/m units"""
        data = np.array([1e6, 2e6, 3e6])
        wn_arr = WavenumberArray(data, "1/m")
        assert_array_almost_equal(wn_arr.as_1_m, data)
        assert_array_almost_equal(wn_arr.as_1_cm, data * 1e-2)

    def test_wavenumber_array_init_1_cm(self):
        """Test WavenumberArray initialization with 1/cm units"""
        data = np.array([1e4, 2e4, 3e4])
        wn_arr = WavenumberArray(data, "1/cm")
        expected_1_m = data * 1e2
        assert_array_almost_equal(wn_arr.as_1_m, expected_1_m)
        assert_array_almost_equal(wn_arr.as_1_cm, data)

    def test_wavenumber_array_invalid_unit(self):
        """Test error handling for invalid units"""
        data = np.array([1e4, 2e4])
        with pytest.raises(ValueError, match="Unsupported unit"):
            WavenumberArray(data, "1/mm")

    def test_wavenumber_array_angular_property(self):
        """Test angular wavenumber property"""
        data = np.array([1e6, 2e6])
        wn_arr = WavenumberArray(data, "1/m")
        expected = data * 2 * PI
        assert_array_almost_equal(wn_arr.as_angular, expected)

    def test_wavenumber_array_to_wavelength(self):
        """Test conversion to wavelength array"""
        data = np.array([1e6, 2e6])
        wn_arr = WavenumberArray(data, "1/m")
        wl_arr = wn_arr.to_wl()
        assert isinstance(wl_arr, WavelengthArray)
        expected = 1 / data
        assert_array_almost_equal(wl_arr.as_m, expected)

    def test_wavenumber_array_to_frequency(self):
        """Test conversion to frequency array"""
        data = np.array([1e6, 2e6])
        wn_arr = WavenumberArray(data, "1/m")
        freq_arr = wn_arr.to_freq()
        assert isinstance(freq_arr, FrequencyArray)
        expected = C_MS * data
        assert_array_almost_equal(freq_arr.as_Hz, expected)

    def test_wavenumber_array_to_omega(self):
        """Test conversion to angular frequency array"""
        data = np.array([1e6, 2e6])
        wn_arr = WavenumberArray(data, "1/m")
        omega_arr = wn_arr.to_omega()
        assert isinstance(omega_arr, AngularFrequencyArray)
        expected = C_MS * 2 * PI * data
        assert_array_almost_equal(omega_arr.as_rad_s, expected)

    def test_wavenumber_array_to_equally_spaced(self):
        """Test equally spaced array generation"""
        data = np.array([1e6, 5e6, 10e6])
        wn_arr = WavenumberArray(data, "1/m")
        equally_spaced = wn_arr.to_equally_spaced(points=11)
        assert len(equally_spaced) == 11
        assert equally_spaced[0] == pytest.approx(10e6)  # max first
        assert equally_spaced[-1] == pytest.approx(1e6)  # min last


class TestBaseConversions:
    """Test conversion consistency between different base classes"""

    def test_wavelength_wavenumber_round_trip(self):
        """Test round-trip conversion between wavelength and wavenumber"""
        wl = Wavelength(1550, "nm")
        wn = wl.to_wn()
        wl_back = wn.to_wl()
        assert pytest.approx(wl.as_nm, rel=1e-10) == wl_back.as_nm

    def test_frequency_wavenumber_round_trip(self):
        """Test round-trip conversion between frequency and wavenumber"""
        freq = Frequency(193.414, "THz")
        wn = freq.to_wn()
        freq_back = wn.to_freq()
        assert pytest.approx(freq.as_Hz, rel=1e-10) == freq_back.as_Hz

    def test_omega_wavenumber_round_trip(self):
        """Test round-trip conversion between angular frequency and wavenumber"""
        omega = AngularFrequency(2 * PI * 193.414e12, "rad/s")
        wn = omega.to_wn()
        omega_back = wn.to_omega()
        assert pytest.approx(omega.as_rad_s, rel=1e-10) == omega_back.as_rad_s

    def test_array_conversions_consistency(self):
        """Test that array conversions are consistent with scalar conversions"""
        # Test wavelength array
        wl_data = np.array([1550, 1310, 850])
        wl_arr = WavelengthArray(wl_data, "nm")
        wn_arr = wl_arr.to_wn()

        # Compare with individual scalar conversions
        for i, wl_val in enumerate(wl_data):
            wl_scalar = Wavelength(wl_val, "nm")
            wn_scalar = wl_scalar.to_wn()
            assert pytest.approx(wn_arr.as_1_m[i]) == wn_scalar.as_1_m

    def test_physical_constants_consistency(self):
        """Test that physical relationships are maintained"""
        # Test c = λf relationship
        wl = Wavelength(1550, "nm")
        freq = wl.to_freq()
        assert pytest.approx(wl.as_m * freq.as_Hz) == C_MS

        # Test ω = 2πf relationship
        omega = freq.to_omega()
        assert pytest.approx(omega.as_rad_s) == 2 * PI * freq.as_Hz

        # Test k = 2π/λ relationship
        wn = wl.to_wn()
        assert pytest.approx(wn.as_angular) == 2 * PI / wl.as_m


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_wavelength_to_wavenumber_units(self):
        """Test that wavelength to wavenumber conversion handles units correctly"""
        # Test with different wavelength units
        wl_nm = Wavelength(1550, "nm")
        wl_um = Wavelength(1.55, "um")
        wl_m = Wavelength(1.55e-6, "m")

        wn_nm = wl_nm.to_wn()
        wn_um = wl_um.to_wn()
        wn_m = wl_m.to_wn()

        # All should give the same wavenumber
        assert pytest.approx(wn_nm.as_1_m) == wn_um.as_1_m
        assert pytest.approx(wn_um.as_1_m) == wn_m.as_1_m

    def test_zero_values_handling(self):
        """Test handling of zero and very small values"""
        # Test very small wavelength (high energy)
        wl_small = Wavelength(1, "nm")
        wn_small = wl_small.to_wn()
        assert wn_small.as_1_m > 0

        # Test very large wavelength (low energy)
        wl_large = Wavelength(100, "um")
        wn_large = wl_large.to_wn()
        assert wn_large.as_1_m > 0

    def test_array_dtype_consistency(self):
        """Test that arrays maintain proper dtype"""
        data = np.array([1e6, 2e6, 3e6], dtype=np.float32)
        wn_arr = WavenumberArray(data, "1/m")
        assert wn_arr.dtype == np.float64  # Should be converted to float64

    def test_angular_frequency_repr(self):
        """Test the __repr__ method of AngularFrequency"""
        omega = AngularFrequency(2 * PI * 1e12, "rad/s")
        repr_str = repr(omega)
        assert "Angular Frequency" in repr_str
        assert "rad/s" in repr_str
        assert str(float(omega)) in repr_str
