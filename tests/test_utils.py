import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from photonics_helper.utils import (
    convert_length,
    convert_time,
    convert_frequency,
    convert_energy,
    c_error,
    c_info,
    c_help,
)
from rich.console import Console
from io import StringIO


class TestConvertLength:
    """Test suite for length conversion functions"""

    def test_convert_length_m_to_um_float(self):
        """Test meter to micrometer conversion with float"""
        result = convert_length(1e-6, "m", "um")
        assert pytest.approx(result) == 1.0
        assert isinstance(result, float)

    def test_convert_length_m_to_nm_float(self):
        """Test meter to nanometer conversion with float"""
        result = convert_length(1e-9, "m", "nm")
        assert pytest.approx(result) == 1.0
        assert isinstance(result, float)

    def test_convert_length_um_to_m_float(self):
        """Test micrometer to meter conversion with float"""
        result = convert_length(1550.0, "um", "m")
        assert pytest.approx(result) == 1.55e-3
        assert isinstance(result, float)

    def test_convert_length_um_to_nm_float(self):
        """Test micrometer to nanometer conversion with float"""
        result = convert_length(1.55, "um", "nm")
        assert pytest.approx(result) == 1550.0
        assert isinstance(result, float)

    def test_convert_length_nm_to_m_float(self):
        """Test nanometer to meter conversion with float"""
        result = convert_length(1550.0, "nm", "m")
        assert pytest.approx(result) == 1.55e-6
        assert isinstance(result, float)

    def test_convert_length_nm_to_um_float(self):
        """Test nanometer to micrometer conversion with float"""
        result = convert_length(1550.0, "nm", "um")
        assert pytest.approx(result) == 1.55
        assert isinstance(result, float)

    def test_convert_length_same_unit_float(self):
        """Test conversion with same units (no conversion)"""
        result = convert_length(1550.0, "nm", "nm")
        assert pytest.approx(result) == 1550.0
        assert isinstance(result, float)

    def test_convert_length_m_to_um_array(self):
        """Test meter to micrometer conversion with numpy array"""
        input_array = np.array([1e-6, 2e-6, 3e-6])
        result = convert_length(input_array, "m", "um")
        expected = np.array([1.0, 2.0, 3.0])
        assert_array_almost_equal(result, expected)
        assert isinstance(result, np.ndarray)

    def test_convert_length_nm_to_m_array(self):
        """Test nanometer to meter conversion with numpy array"""
        input_array = np.array([1550, 1310, 850])
        result = convert_length(input_array, "nm", "m")
        expected = np.array([1.55e-6, 1.31e-6, 0.85e-6])
        assert_array_almost_equal(result, expected)
        assert isinstance(result, np.ndarray)

    def test_convert_length_int_input(self):
        """Test conversion with integer input"""
        result = convert_length(1550, "nm", "um")
        assert pytest.approx(result) == 1.55
        assert isinstance(result, float)

    def test_convert_length_invalid_type(self):
        """Test error handling for invalid input type"""
        with pytest.raises(
            TypeError, match="value should be a type of either: float or NDArray"
        ):
            convert_length("1550", "nm", "um")

    def test_convert_length_all_combinations(self):
        """Test all possible unit combinations"""
        test_value = 1.55e-6  # 1.55 micrometers in meters

        # Test all possible conversions
        combinations = [
            ("m", "m", 1.55e-6),
            ("m", "um", 1.55),
            ("m", "nm", 1550),
            ("um", "m", 1.55e-6),
            ("um", "um", 1.55),
            ("um", "nm", 1550),
            ("nm", "m", 1.55e-6),
            ("nm", "um", 1.55),
            ("nm", "nm", 1550),
        ]

        for from_unit, to_unit, expected in combinations:
            if from_unit == "m":
                input_val = 1.55e-6
            elif from_unit == "um":
                input_val = 1.55
            else:  # nm
                input_val = 1550

            result = convert_length(input_val, from_unit, to_unit)
            assert pytest.approx(result, rel=1e-10) == expected


class TestConvertTime:
    """Test suite for time conversion functions"""

    def test_convert_time_s_to_ps_float(self):
        """Test second to picosecond conversion with float"""
        result = convert_time(1e-12, "s", "ps")
        assert pytest.approx(result) == 1.0
        assert isinstance(result, float)

    def test_convert_time_s_to_fs_float(self):
        """Test second to femtosecond conversion with float"""
        result = convert_time(1e-15, "s", "fs")
        assert pytest.approx(result) == 1.0
        assert isinstance(result, float)

    def test_convert_time_ps_to_s_float(self):
        """Test picosecond to second conversion with float"""
        result = convert_time(100.0, "ps", "s")
        assert pytest.approx(result) == 100e-12
        assert isinstance(result, float)

    def test_convert_time_ps_to_fs_float(self):
        """Test picosecond to femtosecond conversion with float"""
        result = convert_time(1.0, "ps", "fs")
        assert pytest.approx(result) == 1000.0
        assert isinstance(result, float)

    def test_convert_time_fs_to_s_float(self):
        """Test femtosecond to second conversion with float"""
        result = convert_time(100.0, "fs", "s")
        assert pytest.approx(result) == 100e-15
        assert isinstance(result, float)

    def test_convert_time_fs_to_ps_float(self):
        """Test femtosecond to picosecond conversion with float"""
        result = convert_time(1000.0, "fs", "ps")
        assert pytest.approx(result) == 1.0
        assert isinstance(result, float)

    def test_convert_time_same_unit_float(self):
        """Test conversion with same units"""
        result = convert_time(100.0, "fs", "fs")
        assert pytest.approx(result) == 100.0
        assert isinstance(result, float)

    def test_convert_time_array(self):
        """Test time conversion with numpy array"""
        input_array = np.array([100, 200, 300])
        result = convert_time(input_array, "fs", "ps")
        expected = np.array([0.1, 0.2, 0.3])
        assert_array_almost_equal(result, expected)
        assert isinstance(result, np.ndarray)

    def test_convert_time_int_input(self):
        """Test conversion with integer input"""
        result = convert_time(100, "fs", "ps")
        assert pytest.approx(result) == 0.1
        assert isinstance(result, float)

    def test_convert_time_invalid_type(self):
        """Test error handling for invalid input type"""
        with pytest.raises(
            TypeError, match="value should be a type of either: float or NDArray"
        ):
            convert_time("100", "fs", "ps")


class TestConvertFrequency:
    """Test suite for frequency conversion functions"""

    def test_convert_frequency_Hz_to_MHz_float(self):
        """Test Hz to MHz conversion with float"""
        result = convert_frequency(1e6, "Hz", "MHz")
        assert pytest.approx(result) == 1.0
        assert isinstance(result, float)

    def test_convert_frequency_Hz_to_GHz_float(self):
        """Test Hz to GHz conversion with float"""
        result = convert_frequency(1e9, "Hz", "GHz")
        assert pytest.approx(result) == 1.0
        assert isinstance(result, float)

    def test_convert_frequency_Hz_to_THz_float(self):
        """Test Hz to THz conversion with float"""
        result = convert_frequency(1e12, "Hz", "THz")
        assert pytest.approx(result) == 1.0
        assert isinstance(result, float)

    def test_convert_frequency_THz_to_Hz_float(self):
        """Test THz to Hz conversion with float"""
        result = convert_frequency(193.414, "THz", "Hz")
        assert pytest.approx(result) == 193.414e12
        assert isinstance(result, float)

    def test_convert_frequency_GHz_to_MHz_float(self):
        """Test GHz to MHz conversion with float"""
        result = convert_frequency(1.0, "GHz", "MHz")
        assert pytest.approx(result) == 1000.0
        assert isinstance(result, float)

    def test_convert_frequency_MHz_to_GHz_float(self):
        """Test MHz to GHz conversion with float"""
        result = convert_frequency(1000.0, "MHz", "GHz")
        assert pytest.approx(result) == 1.0
        assert isinstance(result, float)

    def test_convert_frequency_same_unit_float(self):
        """Test conversion with same units"""
        result = convert_frequency(100.0, "GHz", "GHz")
        assert pytest.approx(result) == 100.0
        assert isinstance(result, float)

    def test_convert_frequency_array(self):
        """Test frequency conversion with numpy array"""
        input_array = np.array([1, 2, 3])
        result = convert_frequency(input_array, "THz", "GHz")
        expected = np.array([1000, 2000, 3000])
        assert_array_almost_equal(result, expected)
        assert isinstance(result, np.ndarray)

    def test_convert_frequency_int_input(self):
        """Test conversion with integer input"""
        result = convert_frequency(193, "THz", "Hz")
        assert pytest.approx(result) == 193e12
        assert isinstance(result, float)

    def test_convert_frequency_invalid_type(self):
        """Test error handling for invalid input type"""
        with pytest.raises(
            TypeError, match="value should be a type of either: float or NDArray"
        ):
            convert_frequency("100", "GHz", "MHz")


class TestConvertEnergy:
    """Test suite for energy conversion functions"""

    def test_convert_energy_J_to_eV_float(self):
        """Test Joule to eV conversion with float"""
        result = convert_energy(1.602176634e-19, "J", "eV")
        assert pytest.approx(result, rel=1e-10) == 1.0
        assert isinstance(result, float)

    def test_convert_energy_eV_to_J_float(self):
        """Test eV to Joule conversion with float"""
        result = convert_energy(1.0, "eV", "J")
        assert pytest.approx(result, rel=1e-10) == 1.602176634e-19
        assert isinstance(result, float)

    def test_convert_energy_same_unit_J(self):
        """Test conversion with same units (J)"""
        result = convert_energy(1e-19, "J", "J")
        assert pytest.approx(result) == 1e-19
        assert isinstance(result, float)

    def test_convert_energy_same_unit_eV(self):
        """Test conversion with same units (eV)"""
        result = convert_energy(1.5, "eV", "eV")
        assert pytest.approx(result) == 1.5
        assert isinstance(result, float)

    def test_convert_energy_array(self):
        """Test energy conversion with numpy array"""
        input_array = np.array([1.0, 2.0, 3.0])
        result = convert_energy(input_array, "eV", "J")
        expected = input_array * 1.602176634e-19
        assert_array_almost_equal(result, expected, decimal=25)
        assert isinstance(result, np.ndarray)

    def test_convert_energy_round_trip(self):
        """Test round-trip conversion consistency"""
        original = 2.5  # eV
        converted = convert_energy(original, "eV", "J")
        back_converted = convert_energy(converted, "J", "eV")
        assert pytest.approx(back_converted, rel=1e-15) == original

    def test_convert_energy_invalid_type(self):
        """Test error handling for invalid input type"""
        with pytest.raises(
            TypeError,
            match="value should be only a type of float or NDArray: got <class 'str'>.",
        ):
            convert_energy("1.5", "eV", "J")


class TestConsoleOutputFunctions:
    """Test suite for console output functions"""

    def test_c_error_function_exists(self):
        """Test that c_error function exists and is callable"""
        assert callable(c_error)
        # Just test that it doesn't raise an exception
        c_error("Test error message")

    def test_c_info_function_exists(self):
        """Test that c_info function exists and is callable"""
        assert callable(c_info)
        # Just test that it doesn't raise an exception
        c_info("Test info message")

    def test_c_help_function_exists(self):
        """Test that c_help function exists and is callable"""
        assert callable(c_help)
        # Just test that it doesn't raise an exception
        c_help("Test help message")

    def test_console_functions_with_various_messages(self):
        """Test console functions with various message types"""
        # Test with empty string
        c_error("")
        c_info("")
        c_help("")

        # Test with special characters
        c_error("Error: [Special] characters & symbols!")
        c_info("Info: Unicode characters 中文")
        c_help("Help: Numbers 123 and symbols @#$%")

        # Test with multiline messages
        c_error("Line 1\nLine 2")
        c_info("Multi\nline\ninfo")
        c_help("Help\nwith\nmultiple\nlines")


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_convert_length_zero_value(self):
        """Test conversion with zero value"""
        result = convert_length(0.0, "m", "nm")
        assert result == 0.0

    def test_convert_time_zero_value(self):
        """Test time conversion with zero value"""
        result = convert_time(0.0, "s", "fs")
        assert result == 0.0

    def test_convert_frequency_zero_value(self):
        """Test frequency conversion with zero value"""
        result = convert_frequency(0.0, "Hz", "THz")
        assert result == 0.0

    def test_convert_energy_zero_value(self):
        """Test energy conversion with zero value"""
        result = convert_energy(0.0, "J", "eV")
        assert result == 0.0

    def test_convert_length_negative_value(self):
        """Test conversion with negative value"""
        result = convert_length(-1550.0, "nm", "um")
        assert pytest.approx(result) == -1.55

    def test_convert_time_negative_value(self):
        """Test time conversion with negative value"""
        result = convert_time(-100.0, "fs", "ps")
        assert pytest.approx(result) == -0.1

    def test_convert_frequency_negative_value(self):
        """Test frequency conversion with negative value"""
        result = convert_frequency(-1.0, "THz", "GHz")
        assert pytest.approx(result) == -1000.0

    def test_convert_energy_negative_value(self):
        """Test energy conversion with negative value"""
        result = convert_energy(-1.0, "eV", "J")
        assert pytest.approx(result) == -1.602176634e-19

    def test_convert_length_very_large_value(self):
        """Test conversion with very large value"""
        result = convert_length(1e20, "nm", "m")
        assert pytest.approx(result) == 1e11

    def test_convert_time_very_small_value(self):
        """Test time conversion with very small value"""
        result = convert_time(1e-30, "s", "fs")
        assert pytest.approx(result) == 1e-15

    def test_empty_array_conversion(self):
        """Test conversion with empty numpy array"""
        empty_array = np.array([])
        result = convert_length(empty_array, "nm", "um")
        assert len(result) == 0
        assert isinstance(result, np.ndarray)

    def test_single_element_array_conversion(self):
        """Test conversion with single-element array"""
        single_array = np.array([1550])
        result = convert_length(single_array, "nm", "um")
        expected = np.array([1.55])
        assert_array_almost_equal(result, expected)

    def test_mixed_type_array_conversion(self):
        """Test conversion with mixed integer/float array"""
        mixed_array = np.array([1550, 1310.5, 850])
        result = convert_length(mixed_array, "nm", "um")
        expected = np.array([1.55, 1.3105, 0.85])
        assert_array_almost_equal(result, expected)


class TestTypeConsistency:
    """Test type consistency across different input types"""

    def test_float_input_float_output(self):
        """Test that float input gives float output"""
        result = convert_length(1550.0, "nm", "um")
        assert isinstance(result, float)

    def test_int_input_float_output(self):
        """Test that int input gives float output"""
        result = convert_length(1550, "nm", "um")
        assert isinstance(result, float)

    def test_array_input_array_output(self):
        """Test that array input gives array output"""
        input_array = np.array([1550, 1310])
        result = convert_length(input_array, "nm", "um")
        assert isinstance(result, np.ndarray)

    def test_array_dtypes_preserved(self):
        """Test that array conversions maintain appropriate dtypes"""
        # Test with different input dtypes
        int_array = np.array([1550, 1310], dtype=np.int32)
        float_array = np.array([1550.0, 1310.0], dtype=np.float64)

        result_int = convert_length(int_array, "nm", "um")
        result_float = convert_length(float_array, "nm", "um")

        # Both should be numeric types
        assert np.issubdtype(result_int.dtype, np.number)
        assert np.issubdtype(result_float.dtype, np.number)


class TestPhysicalConsistency:
    """Test that conversions maintain physical consistency"""

    def test_length_conversion_consistency(self):
        """Test that length conversions are physically consistent"""
        # 1 meter = 1e6 micrometers = 1e9 nanometers
        meter_val = 1.0
        um_val = convert_length(meter_val, "m", "um")
        nm_val = convert_length(meter_val, "m", "nm")

        assert pytest.approx(um_val) == 1e6
        assert pytest.approx(nm_val) == 1e9

        # Check inverse relationship
        um_to_nm = convert_length(um_val, "um", "nm")
        assert pytest.approx(um_to_nm) == nm_val

    def test_time_conversion_consistency(self):
        """Test that time conversions are physically consistent"""
        # 1 second = 1e12 picoseconds = 1e15 femtoseconds
        second_val = 1.0
        ps_val = convert_time(second_val, "s", "ps")
        fs_val = convert_time(second_val, "s", "fs")

        assert pytest.approx(ps_val) == 1e12
        assert pytest.approx(fs_val) == 1e15

        # Check inverse relationship
        ps_to_fs = convert_time(ps_val, "ps", "fs")
        assert pytest.approx(ps_to_fs) == fs_val

    def test_frequency_conversion_consistency(self):
        """Test that frequency conversions are physically consistent"""
        # 1 THz = 1e3 GHz = 1e6 MHz = 1e12 Hz
        thz_val = 1.0
        ghz_val = convert_frequency(thz_val, "THz", "GHz")
        mhz_val = convert_frequency(thz_val, "THz", "MHz")
        hz_val = convert_frequency(thz_val, "THz", "Hz")

        assert pytest.approx(ghz_val) == 1e3
        assert pytest.approx(mhz_val) == 1e6
        assert pytest.approx(hz_val) == 1e12

    def test_energy_conversion_physical_constant(self):
        """Test that energy conversion uses correct physical constant"""
        # Elementary charge: 1.602176634e-19 C
        # 1 eV = 1.602176634e-19 J
        eV_val = 1.0
        J_val = convert_energy(eV_val, "eV", "J")

        # Check against known physical constant
        assert pytest.approx(J_val, rel=1e-15) == 1.602176634e-19
