"""Tests for ZDependentDispersion dataclass."""
import math
import numpy as np
import pytest
import tempfile
import os

from photonics_helper.fiber import ZDependentDispersion


def _make_simple_profile():
    """Create a simple β(ω, z) profile for testing.

    β(ω, z) = β0 + β2(z) * ω²/2, where β2(z) varies linearly from -1e-26 to 1e-26.
    """
    omegas = np.linspace(1e15, 3e15, 50)  # rad/s
    z_positions = np.linspace(0, 1e-3, 20)  # m
    beta2_start = -1e-26  # s²/m
    beta2_end = 1e-26  # s²/m
    beta2_profile = np.linspace(beta2_start, beta2_end, len(z_positions))
    beta = np.zeros((len(omegas), len(z_positions)))
    for j, beta2_j in enumerate(beta2_profile):
        beta[:, j] = 1e8 + beta2_j * omegas ** 2 / 2  # β0 = 1e8 rad/m
    return omegas, z_positions, beta


class TestZDependentDispersion:
    """Tests for the ZDependentDispersion dataclass."""

    def test_basic_creation(self):
        """Test basic creation from arrays."""
        omegas, z_positions, beta = _make_simple_profile()
        zdd = ZDependentDispersion.from_arrays(
            omegas=omegas,
            z_positions=z_positions,
            beta=beta,
            central_wavelength=1550e-9,
        )
        assert zdd.n_omega == 50
        assert zdd.n_z == 20
        assert zdd.central_wavelength == 1550e-9

    def test_interpolation_at_grid_points(self):
        """Test that interpolation at grid points returns exact values."""
        omegas, z_positions, beta = _make_simple_profile()
        zdd = ZDependentDispersion.from_arrays(
            omegas=omegas,
            z_positions=z_positions,
            beta=beta,
            central_wavelength=1550e-9,
        )
        # Test at a grid point
        omega_idx = 25
        z_idx = 10
        omega = omegas[omega_idx]
        z = z_positions[z_idx]
        result = zdd.fn(omega, z)
        expected = beta[omega_idx, z_idx]
        assert np.isclose(result, expected, rtol=1e-10)

    def test_interpolation_between_grid_points(self):
        """Test interpolation between grid points (linear interpolation)."""
        omegas, z_positions, beta = _make_simple_profile()
        zdd = ZDependentDispersion.from_arrays(
            omegas=omegas,
            z_positions=z_positions,
            beta=beta,
            central_wavelength=1550e-9,
        )
        # Test at a point between grid points
        omega_mid = (omegas[10] + omegas[11]) / 2
        z_mid = (z_positions[5] + z_positions[6]) / 2
        result = zdd.fn(omega_mid, z_mid)
        # Should be close to the interpolated value (within linear interpolation tolerance)
        expected = zdd.fn(omegas[10], z_positions[5]) * 0.5 + zdd.fn(omegas[11], z_positions[6]) * 0.5
        assert np.isclose(result, expected, rtol=0.1)  # Loose tolerance for linear interp

    def test_out_of_range_returns_nan(self):
        """Test that out-of-range queries return NaN."""
        omegas, z_positions, beta = _make_simple_profile()
        zdd = ZDependentDispersion.from_arrays(
            omegas=omegas,
            z_positions=z_positions,
            beta=beta,
            central_wavelength=1550e-9,
        )
        # Query outside the range
        omega_out = omegas[-1] * 2  # Beyond the max omega
        result = zdd.fn(omega_out, z_positions[0])
        assert np.isnan(result)

    def test_scalar_vs_array_input(self):
        """Test that scalar and array inputs work correctly."""
        omegas, z_positions, beta = _make_simple_profile()
        zdd = ZDependentDispersion.from_arrays(
            omegas=omegas,
            z_positions=z_positions,
            beta=beta,
            central_wavelength=1550e-9,
        )
        # Scalar input
        omega_scalar = omegas[10]
        z = z_positions[5]
        result_scalar = zdd.fn(omega_scalar, z)
        assert isinstance(result_scalar, float)

        # Array input
        omega_array = np.array([omegas[10], omegas[11]])
        result_array = zdd.fn(omega_array, z)
        assert isinstance(result_array, np.ndarray)
        assert len(result_array) == 2

    def test_from_npz(self):
        """Test loading from NPZ file."""
        omegas, z_positions, beta = _make_simple_profile()
        central_wavelength = 1550e-9

        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = os.path.join(tmpdir, "test_dispersion.npz")
            np.savez(
                npz_path,
                omegas=omegas,
                z_positions=z_positions,
                beta=beta,
                central_wavelength=central_wavelength,
            )
            zdd = ZDependentDispersion.from_npz(npz_path)
            assert zdd.n_omega == 50
            assert zdd.n_z == 20
            assert zdd.central_wavelength == central_wavelength

    def test_from_npz_without_central_wavelength(self):
        """Test loading from NPZ without central_wavelength in file."""
        omegas, z_positions, beta = _make_simple_profile()
        central_wavelength = 1550e-9

        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = os.path.join(tmpdir, "test_dispersion_no_cw.npz")
            np.savez(
                npz_path,
                omegas=omegas,
                z_positions=z_positions,
                beta=beta,
            )
            # Should require central_wavelength as argument
            zdd = ZDependentDispersion.from_npz(npz_path, central_wavelength=central_wavelength)
            assert zdd.central_wavelength == central_wavelength

    def test_from_npz_missing_central_wavelength_raises(self):
        """Test that missing central_wavelength raises ValueError."""
        omegas, z_positions, beta = _make_simple_profile()

        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = os.path.join(tmpdir, "test_dispersion_bad.npz")
            np.savez(
                npz_path,
                omegas=omegas,
                z_positions=z_positions,
                beta=beta,
            )
            with pytest.raises(ValueError, match="central_wavelength"):
                ZDependentDispersion.from_npz(npz_path)

    def test_shape_validation(self):
        """Test that invalid shapes raise ValueError."""
        omegas = np.linspace(1e15, 3e15, 50)
        z_positions = np.linspace(0, 1e-3, 20)
        beta_wrong_shape = np.zeros((49, 20))  # Wrong n_omega
        with pytest.raises(ValueError, match="must match"):
            ZDependentDispersion.from_arrays(
                omegas=omegas,
                z_positions=z_positions,
                beta=beta_wrong_shape,
                central_wavelength=1550e-9,
            )

    def test_2d_validation(self):
        """Test that non-2D beta raises ValueError."""
        omegas = np.linspace(1e15, 3e15, 50)
        z_positions = np.linspace(0, 1e-3, 20)
        beta_1d = np.zeros(50)  # 1D instead of 2D
        with pytest.raises(ValueError, match="2-D"):
            ZDependentDispersion.from_arrays(
                omegas=omegas,
                z_positions=z_positions,
                beta=beta_1d,
                central_wavelength=1550e-9,
            )

    def test_sorting_non_monotonic(self):
        """Test that non-monotonic axes are sorted."""
        omegas = np.array([2e15, 1e15, 3e15])  # Not sorted
        z_positions = np.array([0.5e-3, 0.0, 1e-3])  # Not sorted
        beta = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        zdd = ZDependentDispersion.from_arrays(
            omegas=omegas,
            z_positions=z_positions,
            beta=beta,
            central_wavelength=1550e-9,
        )
        # After sorting, omegas should be [1e15, 2e15, 3e15]
        assert np.allclose(zdd.omegas, [1e15, 2e15, 3e15])
        assert np.allclose(zdd.z_positions, [0.0, 0.5e-3, 1e-3])

    def test_repr(self):
        """Test string representation."""
        omegas, z_positions, beta = _make_simple_profile()
        zdd = ZDependentDispersion.from_arrays(
            omegas=omegas,
            z_positions=z_positions,
            beta=beta,
            central_wavelength=1550e-9,
        )
        repr_str = repr(zdd)
        assert "ZDependentDispersion" in repr_str
        assert "rad/s" in repr_str
        assert "m" in repr_str

    def test_get_betas_at_z_quadratic_profile(self):
        """β₂ from Taylor fit should match the analytic quadratic model."""
        omegas, z_positions, beta = _make_simple_profile()
        zdd = ZDependentDispersion.from_arrays(
            omegas=omegas,
            z_positions=z_positions,
            beta=beta,
            central_wavelength=1550e-9,
        )
        z_idx = 10
        z = z_positions[z_idx]
        beta2_expected = np.linspace(-1e-26, 1e-26, len(z_positions))[z_idx]
        betas = zdd.get_betas_at_z(z, order=7)
        assert np.isclose(betas[0], beta2_expected, rtol=0.05)
        assert np.allclose(betas[1:], 0, atol=1e-28)

    def test_get_betas_vs_z_shape(self):
        """get_betas_vs_z returns (n_z, order-1)."""
        omegas, z_positions, beta = _make_simple_profile()
        zdd = ZDependentDispersion.from_arrays(
            omegas=omegas,
            z_positions=z_positions,
            beta=beta,
            central_wavelength=1550e-9,
        )
        betas_vs_z = zdd.get_betas_vs_z(order=5)
        assert betas_vs_z.shape == (len(z_positions), 4)

    def test_taylor_round_trip(self):
        """Taylor(β₂…β₇) should approximate the β(ω) table inside the fit window."""
        omegas, z_positions, beta = _make_simple_profile()
        central_wavelength = 1550e-9
        zdd = ZDependentDispersion.from_arrays(
            omegas=omegas,
            z_positions=z_positions,
            beta=beta,
            central_wavelength=central_wavelength,
        )
        omega0 = 2 * np.pi * 3e8 / central_wavelength
        z = z_positions[10]
        betas = zdd.get_betas_at_z(z, order=7, omega0=omega0)
        omega_fit = np.linspace(omega0 - 0.3e15, omega0 + 0.3e15, 200)
        beta_table = zdd.fn(omega_fit, z)
        beta0 = float(zdd.fn(omega0, z))
        beta_taylor = beta0
        for k in range(2, 8):
            beta_taylor = beta_taylor + betas[k - 2] / math.factorial(k) * (omega_fit - omega0) ** k
        valid = ~np.isnan(beta_table)
        rel_err = np.abs(beta_taylor[valid] - beta_table[valid]) / np.maximum(
            np.abs(beta_table[valid]), 1e-30
        )
        assert np.max(rel_err) < 0.01
