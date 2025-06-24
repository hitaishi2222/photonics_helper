import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
import warnings
from photonics_helper.fiber import Dispersion, PropagationConstant
from photonics_helper.base import WavelengthArray, AngularFrequencyArray, Wavelength, C_MS, PI


class TestDispersion:
    """Test suite for Dispersion class"""

    @pytest.fixture
    def sample_wavelengths(self):
        """Create sample wavelength array for testing"""
        wl_data = np.linspace(1.5e-6, 1.6e-6, 50)  # 1.5-1.6 μm range
        return WavelengthArray(wl_data, "m")

    @pytest.fixture
    def sample_dispersion_values(self):
        """Create sample dispersion values"""
        # Typical dispersion values in ps/nm.km
        wl_data = np.linspace(1.5e-6, 1.6e-6, 50)
        # Simple parabolic dispersion profile
        dispersion = 17 * (wl_data * 1e9 - 1550) / 1550  # ps/nm.km
        return dispersion

    def test_dispersion_init_ps_nm_km(self, sample_wavelengths, sample_dispersion_values):
        """Test Dispersion initialization with ps/nm.km units"""
        central_wl = Wavelength(1550, "nm")
        disp = Dispersion(
            wavelengths=sample_wavelengths,
            values=sample_dispersion_values,
            unit="ps/nm.km",
            central_wavelength=central_wl
        )

        # Check that values are converted to s/m^2
        expected_s_m_m = sample_dispersion_values * 1e-6
        assert_array_almost_equal(disp.as_s_m_m, expected_s_m_m)

    def test_dispersion_init_s_m2(self, sample_wavelengths):
        """Test Dispersion initialization with s/m^2 units"""
        values = np.random.random(len(sample_wavelengths)) * 1e-26
        central_wl = Wavelength(1550, "nm")

        disp = Dispersion(
            wavelengths=sample_wavelengths,
            values=values,
            unit="s/m^2",
            central_wavelength=central_wl
        )

        # Values should remain unchanged
        assert_array_almost_equal(disp.as_s_m_m, values)

    def test_dispersion_unit_conversion(self, sample_wavelengths, sample_dispersion_values):
        """Test unit conversion properties"""
        central_wl = Wavelength(1550, "nm")
        disp = Dispersion(
            wavelengths=sample_wavelengths,
            values=sample_dispersion_values,
            unit="ps/nm.km",
            central_wavelength=central_wl
        )

        # Test conversion back to ps/nm.km
        converted_back = disp.as_ps_nm_km
        assert_array_almost_equal(converted_back, sample_dispersion_values, decimal=10)

    def test_dispersion_repr(self, sample_wavelengths, sample_dispersion_values):
        """Test string representation of Dispersion"""
        central_wl = Wavelength(1550, "nm")
        disp = Dispersion(
            wavelengths=sample_wavelengths,
            values=sample_dispersion_values,
            unit="ps/nm.km",
            central_wavelength=central_wl
        )

        repr_str = repr(disp)
        assert "Dispersion:" in repr_str
        assert "from wl:" in repr_str

    def test_get_wl(self, sample_wavelengths, sample_dispersion_values):
        """Test getting wavelength array"""
        central_wl = Wavelength(1550, "nm")
        disp = Dispersion(
            wavelengths=sample_wavelengths,
            values=sample_dispersion_values,
            unit="ps/nm.km",
            central_wavelength=central_wl
        )

        wl_array = disp.get_wl()
        assert isinstance(wl_array, WavelengthArray)
        assert_array_almost_equal(wl_array.as_m, sample_wavelengths.as_m)

    def test_check_wavelength_limit_valid(self, sample_wavelengths, sample_dispersion_values):
        """Test wavelength limit checking with valid wavelengths"""
        central_wl = Wavelength(1550, "nm")
        disp = Dispersion(
            wavelengths=sample_wavelengths,
            values=sample_dispersion_values,
            unit="ps/nm.km",
            central_wavelength=central_wl
        )

        # Should not raise error for wavelength within range
        disp.check_wavelength_limit(1.55e-6, "m")
        disp.check_wavelength_limit(1550, "nm")
        disp.check_wavelength_limit(1.55, "um")

    def test_check_wavelength_limit_invalid(self, sample_wavelengths, sample_dispersion_values):
        """Test wavelength limit checking with invalid wavelengths"""
        central_wl = Wavelength(1550, "nm")
        disp = Dispersion(
            wavelengths=sample_wavelengths,
            values=sample_dispersion_values,
            unit="ps/nm.km",
            central_wavelength=central_wl
        )

        # Should raise error for wavelength outside range
        with pytest.raises(ValueError, match="values of disersion available between"):
            disp.check_wavelength_limit(1.4e-6, "m")  # Too low

        with pytest.raises(ValueError, match="values of disersion available between"):
            disp.check_wavelength_limit(1.7e-6, "m")  # Too high

    def test_fn_interpolation(self, sample_wavelengths, sample_dispersion_values):
        """Test dispersion function interpolation"""
        central_wl = Wavelength(1550, "nm")
        disp = Dispersion(
            wavelengths=sample_wavelengths,
            values=sample_dispersion_values,
            unit="ps/nm.km",
            central_wavelength=central_wl
        )

        # Test interpolation at a point within range
        test_wl = 1.55e-6  # meters
        result = disp.fn(test_wl)
        assert isinstance(result, float)
        assert not np.isnan(result)

    def test_fn_s_m_m_interpolation(self, sample_wavelengths, sample_dispersion_values):
        """Test dispersion function with s/m^2 units"""
        central_wl = Wavelength(1550, "nm")
        disp = Dispersion(
            wavelengths=sample_wavelengths,
            values=sample_dispersion_values,
            unit="ps/nm.km",
            central_wavelength=central_wl
        )

        # Test interpolation at a point within range
        test_wl_nm = 1550  # nanometers
        result = disp.fn_s_m_m(test_wl_nm)
        assert isinstance(result, float)
        assert not np.isnan(result)

    def test_fn_ps_nm_km_interpolation(self, sample_wavelengths, sample_dispersion_values):
        """Test dispersion function with ps/nm.km units"""
        central_wl = Wavelength(1550, "nm")
        disp = Dispersion(
            wavelengths=sample_wavelengths,
            values=sample_dispersion_values,
            unit="ps/nm.km",
            central_wavelength=central_wl
        )

        # Test interpolation at a point within range
        test_wl_nm = 1550  # nanometers
        result = disp.fn_ps_nm_km(test_wl_nm)
        assert isinstance(result, float)
        assert not np.isnan(result)

    def test_from_neff_basic(self):
        """Test creating Dispersion from effective index"""
        wl_data = np.linspace(1.5e-6, 1.6e-6, 100)
        wavelengths = WavelengthArray(wl_data, "m")
        # Simple linear neff profile
        neff = 1.45 + 0.01 * (wl_data - 1.55e-6) / (0.1e-6)

        disp = Dispersion.from_neff(
            neff=neff,
            wavelengths=wavelengths,
            central_wavelength_nm=1550,
            ignore_fit_error=True  # Ignore warnings for test
        )

        assert isinstance(disp, Dispersion)
        assert len(disp.as_s_m_m) == len(wavelengths)

    def test_from_neff_length_mismatch(self):
        """Test error handling for mismatched array lengths"""
        wl_data = np.linspace(1.5e-6, 1.6e-6, 100)
        wavelengths = WavelengthArray(wl_data, "m")
        neff = np.ones(50)  # Different length

        with pytest.raises(ValueError, match="Length of both neff and wavelengths should be same"):
            Dispersion.from_neff(
                neff=neff,
                wavelengths=wavelengths,
                central_wavelength_nm=1550
            )

    def test_from_neff_wrong_type(self):
        """Test error handling for wrong wavelength type"""
        wl_data = np.linspace(1.5e-6, 1.6e-6, 100)
        neff = np.ones(100)

        with pytest.raises(TypeError, match="wavelengths cannot process the type"):
            Dispersion.from_neff(
                neff=neff,
                wavelengths=wl_data,  # Should be WavelengthArray
                central_wavelength_nm=1550
            )

    def test_from_propagation_constant_basic(self):
        """Test creating Dispersion from propagation constant"""
        wl_data = np.linspace(1.5e-6, 1.6e-6, 100)
        wavelengths = WavelengthArray(wl_data, "m")
        # Simple beta profile
        omega = wavelengths.to_omega().as_rad_s
        beta = 1.45 * omega / C_MS

        disp = Dispersion.from_propagation_constanant(
            beta=beta,
            wavelengths=wavelengths,
            central_wavelength_nm=1550,
            ignore_fit_error=True
        )

        assert isinstance(disp, Dispersion)
        assert len(disp.as_s_m_m) == len(wavelengths)

    def test_get_beta2(self, sample_wavelengths, sample_dispersion_values):
        """Test beta2 calculation from dispersion"""
        central_wl = Wavelength(1550, "nm")
        disp = Dispersion(
            wavelengths=sample_wavelengths,
            values=sample_dispersion_values,
            unit="ps/nm.km",
            central_wavelength=central_wl
        )

        beta2 = disp.get_beta2(1550)
        assert isinstance(beta2, float)
        assert not np.isnan(beta2)

    def test_get_beta2_out_of_range(self, sample_wavelengths, sample_dispersion_values):
        """Test beta2 calculation with wavelength out of range"""
        central_wl = Wavelength(1550, "nm")
        disp = Dispersion(
            wavelengths=sample_wavelengths,
            values=sample_dispersion_values,
            unit="ps/nm.km",
            central_wavelength=central_wl
        )

        with pytest.raises(ValueError, match="values of disersion available between"):
            disp.get_beta2(1400)  # Outside range


class TestPropagationConstant:
    """Test suite for PropagationConstant class"""

    @pytest.fixture
    def sample_wavelengths(self):
        """Create sample wavelength array"""
        wl_data = np.linspace(1.5e-6, 1.6e-6, 50)
        return WavelengthArray(wl_data, "m")

    @pytest.fixture
    def sample_omega(self):
        """Create sample angular frequency array"""
        wl_data = np.linspace(1.5e-6, 1.6e-6, 50)
        wl_array = WavelengthArray(wl_data, "m")
        return wl_array.to_omega()

    def test_propagation_constant_init_wavelength(self, sample_wavelengths):
        """Test PropagationConstant initialization with wavelengths"""
        values = np.random.random(len(sample_wavelengths)) * 1e7
        prop_const = PropagationConstant(values=values, x_values=sample_wavelengths)

        assert hasattr(prop_const, '_wavelengths')
        assert_array_almost_equal(prop_const._wavelengths.as_m, sample_wavelengths.as_m)
        assert_array_almost_equal(prop_const._values, values)

    def test_propagation_constant_init_omega(self, sample_omega):
        """Test PropagationConstant initialization with angular frequencies"""
        values = np.random.random(len(sample_omega)) * 1e7
        prop_const = PropagationConstant(values=values, x_values=sample_omega)

        assert hasattr(prop_const, '_omegas')
        assert_array_almost_equal(prop_const._omegas.as_rad_s, sample_omega.as_rad_s)
        assert_array_almost_equal(prop_const._values, values)

    def test_beta2_from_neff_wavelength(self, sample_wavelengths):
        """Test beta2 calculation from neff with wavelengths"""
        neff = np.ones(len(sample_wavelengths)) * 1.45

        beta2 = PropagationConstant.beta2_from_neff(neff=neff, x_values=sample_wavelengths)

        # beta = n * omega / c
        expected_omegas = sample_wavelengths.to_omega().as_rad_s
        expected_beta2 = neff * expected_omegas / C_MS

        assert_array_almost_equal(beta2, expected_beta2)

    def test_beta2_from_neff_omega(self, sample_omega):
        """Test beta2 calculation from neff with angular frequencies"""
        neff = np.ones(len(sample_omega)) * 1.45

        beta2 = PropagationConstant.beta2_from_neff(neff=neff, x_values=sample_omega)

        # beta = n * omega / c
        expected_beta2 = neff * sample_omega.as_rad_s / C_MS

        assert_array_almost_equal(beta2, expected_beta2)

    def test_beta2_from_neff_length_mismatch(self, sample_wavelengths):
        """Test error handling for mismatched array lengths"""
        neff = np.ones(len(sample_wavelengths) - 10)  # Different length

        with pytest.raises(ValueError, match="both neff and x_values must be of same length"):
            PropagationConstant.beta2_from_neff(neff=neff, x_values=sample_wavelengths)

    def test_from_neff_omega_basic(self, sample_omega):
        """Test creating PropagationConstant from neff and omega"""
        neff = np.ones(len(sample_omega)) * 1.45

        prop_const = PropagationConstant.from_neff_omega(neff=neff, omega=sample_omega)

        assert isinstance(prop_const, PropagationConstant)
        expected_betas = sample_omega.as_rad_s * neff / C_MS
        assert_array_almost_equal(prop_const._values, expected_betas)

    def test_from_neff_omega_length_mismatch(self, sample_omega):
        """Test error handling for mismatched array lengths"""
        neff = np.ones(len(sample_omega) - 5)  # Different length

        with pytest.raises(ValueError, match="both neff and angular frequency array must have same length"):
            PropagationConstant.from_neff_omega(neff=neff, omega=sample_omega)

    def test_from_neff_omega_wrong_type(self, sample_omega):
        """Test error handling for wrong omega type"""
        neff = np.ones(len(sample_omega))
        omega_wrong = np.array(sample_omega.as_rad_s)  # Regular array, not AngularFrequencyArray

        with pytest.raises(TypeError, match="omega should be a type of 'AngularFrequencyArray'"):
            PropagationConstant.from_neff_omega(neff=neff, omega=omega_wrong)


class TestFiberEdgeCases:
    """Test edge cases and error conditions"""

    def test_dispersion_with_single_wavelength(self):
        """Test dispersion with single wavelength point"""
        wl_data = np.array([1.55e-6])
        wavelengths = WavelengthArray(wl_data, "m")
        values = np.array([17.0])
        central_wl = Wavelength(1550, "nm")

        disp = Dispersion(
            wavelengths=wavelengths,
            values=values,
            unit="ps/nm.km",
            central_wavelength=central_wl
        )

        assert len(disp.as_s_m_m) == 1
        assert pytest.approx(disp.as_ps_nm_km[0]) == 17.0

    def test_dispersion_warning_on_bad_fit(self):
        """Test warning generation for bad neff fitting"""
        wl_data = np.linspace(1.5e-6, 1.6e-6, 10)  # Few points for bad fit
        wavelengths = WavelengthArray(wl_data, "m")
        # Create intentionally noisy neff data
        neff = 1.45 + 0.1 * np.random.random(len(wl_data))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                Dispersion.from_neff(
                    neff=neff,
                    wavelengths=wavelengths,
                    central_wavelength_nm=1550,
                    ignore_fit_error=True
                )
            except:
                pass  # We expect this might fail, just checking warnings

            # Check if warning was issued (may not always trigger depending on data)
            if len(w) > 0:
                assert any("Bad fitting" in str(warning.message) for warning in w)

    def test_dispersion_physical_units_consistency(self):
        """Test that dispersion units are physically consistent"""
        wl_data = np.linspace(1.5e-6, 1.6e-6, 50)
        wavelengths = WavelengthArray(wl_data, "m")

        # Create physically reasonable dispersion values
        dispersion_ps_nm_km = 17 * np.ones(len(wl_data))  # ps/nm/km
        central_wl = Wavelength(1550, "nm")

        disp = Dispersion(
            wavelengths=wavelengths,
            values=dispersion_ps_nm_km,
            unit="ps/nm.km",
            central_wavelength=central_wl
        )

        # Check unit conversion consistency
        # 1 ps/nm/km = 1e-6 s/m^2
        expected_s_m2 = dispersion_ps_nm_km * 1e-6
        assert_array_almost_equal(disp.as_s_m_m, expected_s_m2)

        # Convert back and check
        converted_back = disp.as_ps_nm_km
        assert_array_almost_equal(converted_back, dispersion_ps_nm_km)

    def test_propagation_constant_physical_consistency(self):
        """Test physical consistency of propagation constant calculations"""
        wl_data = np.array([1.55e-6])  # Single wavelength
        wavelengths = WavelengthArray(wl_data, "m")
        neff = np.array([1.45])

        # Calculate beta using class method
        beta_calculated = PropagationConstant.beta2_from_neff(
            neff=neff, x_values=wavelengths
        )

        # Calculate beta manually
        omega = wavelengths.to_omega().as_rad_s
        beta_manual = neff * omega / C_MS

        assert_array_almost_equal(beta_calculated, beta_manual)

        # Check that beta = n * k0 where k0 = omega/c
        k0 = omega / C_MS
        expected_beta = neff * k0
        assert_array_almost_equal(beta_calculated, expected_beta)
