import pytest
import numpy as np
from photonics_helper import Wavelength
from photonics_helper.pulse import Pulse, RectangularPulse


class TestPulse:
    """Test suite for Pulse class"""

    def test_pulse_init_fs(self):
        """Test Pulse initialization with femtosecond units"""
        pulse = Pulse(duration=100, duration_unit="fs", peak_power=1000)
        assert pytest.approx(pulse.duration) == 100e-15
        assert pulse.peak_power == 1000

    def test_pulse_init_ps(self):
        """Test Pulse initialization with picosecond units"""
        pulse = Pulse(duration=1, duration_unit="ps", peak_power=500)
        assert pytest.approx(pulse.duration) == 1e-12
        assert pulse.peak_power == 500

    def test_pulse_init_ns(self):
        """Test Pulse initialization with nanosecond units"""
        pulse = Pulse(duration=10, duration_unit="ns", peak_power=100)
        assert pytest.approx(pulse.duration) == 10e-9
        assert pulse.peak_power == 100

    def test_pulse_init_s(self):
        """Test Pulse initialization with second units"""
        pulse = Pulse(duration=1e-6, duration_unit="s", peak_power=50)
        assert pytest.approx(pulse.duration) == 1e-6
        assert pulse.peak_power == 50

    def test_pulse_invalid_duration_unit(self):
        """Test error handling for invalid duration units"""
        with pytest.raises(
            ValueError, match="Unsupported unit: ms -> s use 's', 'ns', 'ps' or 'fs'"
        ):
            Pulse(duration=100, duration_unit="ms", peak_power=1000)

    def test_pulse_with_central_wavelength(self):
        """Test Pulse initialization with central wavelength"""
        wl = Wavelength(1550, "nm")
        pulse = Pulse(
            duration=100, duration_unit="fs", peak_power=1000, central_wavelength=wl
        )
        assert pulse.central_wavelength == wl
        assert pytest.approx(pulse.central_wavelength.as_nm) == 1550

    def test_pulse_with_energy(self):
        """Test Pulse initialization with energy parameter"""
        pulse = Pulse(duration=100, duration_unit="fs", peak_power=1000, energy=1e-9)
        assert pulse._energy == 1e-9

    def test_pulse_duration_setter(self):
        """Test duration setter"""
        pulse = Pulse(duration=100, duration_unit="fs", peak_power=1000)
        pulse.duration = 200e-15
        assert pytest.approx(pulse.duration) == 200e-15

    def test_pulse_peak_power_setter(self):
        """Test peak power setter"""
        pulse = Pulse(duration=100, duration_unit="fs", peak_power=1000)
        pulse.peak_power = 2000
        assert pulse.peak_power == 2000

    def test_pulse_central_wavelength_not_defined(self):
        """Test error when accessing undefined central wavelength"""
        pulse = Pulse(duration=100, duration_unit="fs", peak_power=1000)
        with pytest.raises(AttributeError, match="Central wavelength is not defined"):
            _ = pulse.central_wavelength

    def test_pulse_period_default(self):
        """Test default period calculation (2 * duration)"""
        pulse = Pulse(duration=100, duration_unit="fs", peak_power=1000)
        expected_period = 2 * 100e-15
        assert pytest.approx(pulse.period) == expected_period

    def test_pulse_rate_default(self):
        """Test default rate calculation (1 / (2 * duration))"""
        pulse = Pulse(duration=100, duration_unit="fs", peak_power=1000)
        expected_rate = 1 / (2 * 100e-15)
        assert pytest.approx(pulse.rate) == expected_rate

    def test_set_period_fs(self):
        """Test setting period with femtosecond units"""
        pulse = Pulse(duration=100, duration_unit="fs", peak_power=1000)
        pulse.set_period(500, "fs")
        assert pytest.approx(pulse.period) == 500e-15
        assert pytest.approx(pulse.rate) == 1 / (500e-15)

    def test_set_period_ps(self):
        """Test setting period with picosecond units"""
        pulse = Pulse(duration=100, duration_unit="fs", peak_power=1000)
        pulse.set_period(1, "ps")
        assert pytest.approx(pulse.period) == 1e-12
        assert pytest.approx(pulse.rate) == 1e12

    def test_set_period_ns(self):
        """Test setting period with nanosecond units"""
        pulse = Pulse(duration=100, duration_unit="fs", peak_power=1000)
        pulse.set_period(10, "ns")
        assert pytest.approx(pulse.period) == 10e-9
        assert pytest.approx(pulse.rate) == 1 / (10e-9)

    def test_set_period_s(self):
        """Test setting period with second units"""
        pulse = Pulse(duration=100, duration_unit="fs", peak_power=1000)
        pulse.set_period(1e-6, "s")
        assert pytest.approx(pulse.period) == 1e-6
        assert pytest.approx(pulse.rate) == 1e6

    def test_set_period_invalid_unit(self):
        """Test error handling for invalid period units"""
        pulse = Pulse(duration=100, duration_unit="fs", peak_power=1000)
        with pytest.raises(
            ValueError, match="Unsupported unit: ms -> s use 's', 'ns', 'ps' or 'fs'"
        ):
            pulse.set_period(1, "ms")

    def test_set_rate_Hz(self):
        """Test setting rate with Hz units"""
        pulse = Pulse(duration=100, duration_unit="fs", peak_power=1000)
        pulse.set_rate(1e12, "Hz")
        assert pytest.approx(pulse.rate) == 1e12
        assert pytest.approx(pulse.period) == 1e-12

    def test_set_rate_MHz(self):
        """Test setting rate with MHz units"""
        pulse = Pulse(duration=100, duration_unit="fs", peak_power=1000)
        pulse.set_rate(1000, "MHz")
        assert pytest.approx(pulse.rate) == 1000e6
        assert pytest.approx(pulse.period) == 1 / (1000e6)

    def test_set_rate_GHz(self):
        """Test setting rate with GHz units"""
        pulse = Pulse(duration=100, duration_unit="fs", peak_power=1000)
        pulse.set_rate(10, "GHz")
        assert pytest.approx(pulse.rate) == 10e9
        assert pytest.approx(pulse.period) == 1 / (10e9)

    def test_set_rate_THz(self):
        """Test setting rate with THz units"""
        pulse = Pulse(duration=100, duration_unit="fs", peak_power=1000)
        pulse.set_rate(0.1, "THz")
        assert pytest.approx(pulse.rate) == 0.1e12
        assert pytest.approx(pulse.period) == 1 / (0.1e12)


class TestRectangularPulse:
    """Test suite for RectangularPulse class"""

    def test_rectangular_pulse_init(self):
        """Test RectangularPulse initialization"""
        rect_pulse = RectangularPulse(
            duration=100, duration_unit="fs", peak_power=1000, amplitude=0.8
        )
        assert pytest.approx(rect_pulse.duration) == 100e-15
        assert rect_pulse.peak_power == 1000
        assert rect_pulse.amplitude == 0.8

    def test_rectangular_pulse_with_wavelength(self):
        """Test RectangularPulse with central wavelength"""
        wl = Wavelength(800, "nm")
        rect_pulse = RectangularPulse(
            duration=50,
            duration_unit="fs",
            peak_power=2000,
            central_wavelength=wl,
            amplitude=1.0,
        )
        assert rect_pulse.central_wavelength == wl
        assert pytest.approx(rect_pulse.central_wavelength.as_nm) == 800
        assert rect_pulse.amplitude == 1.0

    def test_rectangular_pulse_with_energy(self):
        """Test RectangularPulse with energy parameter"""
        rect_pulse = RectangularPulse(
            duration=100,
            duration_unit="fs",
            peak_power=1000,
            energy=5e-10,
            amplitude=0.5,
        )
        assert rect_pulse._energy == 5e-10
        assert rect_pulse.amplitude == 0.5

    def test_rectangular_pulse_inheritance(self):
        """Test that RectangularPulse inherits Pulse methods"""
        rect_pulse = RectangularPulse(
            duration=100, duration_unit="fs", peak_power=1000, amplitude=1.0
        )

        # Test inherited methods
        rect_pulse.set_period(1, "ps")
        assert pytest.approx(rect_pulse.period) == 1e-12

        rect_pulse.set_rate(1, "GHz")
        assert pytest.approx(rect_pulse.rate) == 1e9

    def test_rectangular_pulse_default_amplitude(self):
        """Test RectangularPulse with default amplitude"""
        rect_pulse = RectangularPulse(duration=100, duration_unit="fs", peak_power=1000)
        assert rect_pulse.amplitude == 1  # default value

    def test_rectangular_pulse_invalid_duration_unit(self):
        """Test error handling for invalid duration units in RectangularPulse"""
        with pytest.raises(
            ValueError, match="Unsupported unit: ms -> s use 's', 'ns', 'ps' or 'fs'"
        ):
            RectangularPulse(duration=100, duration_unit="ms", peak_power=1000)


class TestPulseEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_very_long_pulse(self):
        """Test very long pulse durations"""
        pulse = Pulse(duration=1, duration_unit="s", peak_power=1)
        assert pytest.approx(pulse.duration) == 1
        assert pulse.rate == 0.5  # Low rate

    def test_zero_peak_power(self):
        """Test pulse with zero peak power"""
        pulse = Pulse(duration=100, duration_unit="fs", peak_power=0)
        assert pulse.peak_power == 0
        assert pytest.approx(pulse.duration) == 100e-15

    def test_negative_peak_power(self):
        """Test pulse with negative peak power (physically allowed for some contexts)"""
        pulse = Pulse(duration=100, duration_unit="fs", peak_power=-1000)
        assert pulse.peak_power == -1000

    def test_period_rate_consistency(self):
        """Test that period and rate remain consistent after modifications"""
        pulse = Pulse(duration=100, duration_unit="fs", peak_power=1000)

        # Set period and check rate
        pulse.set_period(1, "ps")
        expected_rate = 1 / (1e-12)
        assert pytest.approx(pulse.rate) == expected_rate

        # Set rate and check period
        pulse.set_rate(2, "GHz")
        expected_period = 1 / (2e9)
        assert pytest.approx(pulse.period) == expected_period

    def test_pulse_properties_immutable_after_init(self):
        """Test that certain properties maintain their relationships"""
        pulse = Pulse(duration=100, duration_unit="fs", peak_power=1000)
        original_duration = pulse.duration

        # Modifying period should not affect duration
        pulse.set_period(1, "ps")
        assert pytest.approx(pulse.duration) == original_duration

        # But period and rate should be updated
        assert pytest.approx(pulse.period) == 1e-12
        assert pytest.approx(pulse.rate) == 1e12
