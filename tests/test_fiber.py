"""
Tests for the ``photonics_helper.fiber`` module.

The suite covers:
* ``Dispersion`` creation from refractive index data (linear case → zero
  dispersion) and error handling for mismatched input lengths.
* The spline‐based accessor methods ``fn``, ``fn_ps_nm_km`` and ``fn_s_m_m``.
* ``PropagationConstant`` utilities – conversion from effective index to
  beta2 and validation of input shapes/types.
"""

import numpy as np
import pytest

from photonics_helper.base import AngularFrequencyArray, Wavelength, WavelengthArray
from photonics_helper.fiber import Dispersion, PropagationConstant


def _linear_neff(wl_m):
    """Simple linear effective index: n(λ) = 1.5 + 0.1·(λ - λ0) / λ0."""
    λ0 = 1550e-9
    return 1.5 + 0.1 * (wl_m - λ0) / λ0


def test_dispersion_from_neff_zero_dispersion():
    """
    A perfectly linear ``neff(λ)`` has zero second derivative, therefore the
    dispersion returned by ``Dispersion.from_neff`` should be (close to) zero.
    """
    # Generate a wavelength grid (nm) and its corresponding neff values.
    wl_vals_nm = np.linspace(1500, 1600, 10)
    wl_arr = WavelengthArray(wl_vals_nm, "nm")
    neff = _linear_neff(wl_arr.as_m)

    # Build Dispersion; the central wavelength is set to the midpoint.
    disp = Dispersion.from_neff(
        neff=neff,
        wavelengths=wl_arr,
        central_wavelength=Wavelength(1550, "nm"),
        ignore_fit_error=True,  # we expect a smooth fit
    )

    # The dispersion values should be essentially zero (within numerical tolerance)
    assert np.allclose(disp.as_s_m_m, 0.0, atol=1e-10)

    # Verify the unit conversion helpers (both should also be zero).
    assert np.allclose(disp.as_ps_nm_km, 0.0, atol=1e-5)
    assert np.allclose(disp.as_s_m_m, 0.0, atol=1e-10)


def test_dispersion_accessor_functions():
    """
    The spline accessors should return values consistent with the underlying
    dispersion data.  We test ``fn`` (SI units) and the two convenience
    wrappers ``fn_ps_nm_km`` and ``fn_s_m_m``.
    """
    # Simple constant dispersion for which the spline is trivial.
    wl_vals_nm = np.linspace(1500.0, 1600.0, 51)
    disp_vals = np.ones(51) * 1e-6  # s/m^2
    wl_arr = WavelengthArray(wl_vals_nm, "nm")
    disp = Dispersion(
        wavelengths=wl_arr,
        values=disp_vals,
        unit="s/m^2",
        central_wavelength=Wavelength(1550, "nm"),
    )

    # Test at an intermediate wavelength (in meters for ``fn``).
    test_wl_m = 1550e-9
    assert pytest.approx(disp.fn(Wavelength(test_wl_m, "m")), rel=1e-12) == 1e-6

    # Both should give identical results via fn with different Wavelength units.
    assert pytest.approx(disp.fn(Wavelength(test_wl_m, "m")), rel=1e-12) == 1e-6
    assert pytest.approx(disp.fn(Wavelength(1550, "nm")), rel=1e-12) == 1e-6


def test_dispersion_unit_conversion_ps_nm_km():
    """
    ps/(nm.km) inputs must be normalised to SI (s/m^2) internally so that
    get_betas/get_beta2 are correct (1 ps/(nm.km) = 1e-6 s/m^2).
    Regression test for the raw-storage bug flagged in revision.md item 1.
    """
    wl_arr = WavelengthArray(np.linspace(1500.0, 1600.0, 51), "nm")
    cw = Wavelength(1550, "nm")

    disp_ps = Dispersion(
        values=np.full(51, 17.0), unit="ps/nm.km",
        wavelengths=wl_arr, central_wavelength=cw,
    )
    disp_si = Dispersion(
        values=np.full(51, 17.0e-6), unit="s/m^2",
        wavelengths=wl_arr, central_wavelength=cw,
    )

    # Accessors reflect correct, converted units on both sides.
    assert np.allclose(disp_ps.as_s_m_m, 17.0e-6)
    assert np.allclose(disp_ps.as_ps_nm_km, 17.0)
    # Equivalent values in either unit must agree internally.
    assert np.allclose(disp_ps.as_s_m_m, disp_si.as_s_m_m)
    # get_betas feeds the GNLSE solver: silica D=17 -> beta2 ~= -21.7 ps^2/km
    b = disp_ps.get_betas(2)
    assert -3e-2 < b[0] < -1e-2  # ps^2/m, negative & in the silica ballpark


def test_dispersion_from_neff_error_conditions():
    """
    Verify that ``Dispersion.from_neff`` raises the appropriate exceptions when
    the input lengths are mismatched or when the wavelength container is not a
    ``WavelengthArray``.
    """
    wl_arr = WavelengthArray(np.array([1500.0, 1550.0]), "nm")
    neff = np.array([1.5, 1.51, 1.52])  # mismatched length

    # Length mismatch should raise a ``ValueError``.
    with pytest.raises(ValueError):
        Dispersion.from_neff(
            neff=neff,
            wavelengths=wl_arr,
            central_wavelength=Wavelength(1550, "nm"),
        )

    # Passing a plain list instead of a ``WavelengthArray`` should raise ``TypeError``.
    with pytest.raises(TypeError):
        Dispersion.from_neff(
            neff=np.array([1.5, 1.51]),
            wavelengths=[1500.0, 1550.0],  # type: ignore[arg-type]  # not a WavelengthArray
            central_wavelength=Wavelength(1550, "nm"),
        )


def test_propagation_constant_beta_from_neff():
    """
    ``PropagationConstant.beta_from_neff`` should compute β = n·ω / c.
    We verify the result against a manual calculation for a simple case.
    """
    neff = np.array([1.0, 1.2, 1.5])
    # Use a wavelength grid (nm) and obtain the corresponding angular frequencies.
    wl_arr = WavelengthArray(np.array([1500.0, 1550.0, 1600.0]), "nm")
    omega_arr = wl_arr.to_omega()

    # Expected β values: β = n·ω / c (c = C_MS)
    from photonics_helper.base import C_MS

    expected_beta = neff * omega_arr.as_rad_s / C_MS

    beta = PropagationConstant.beta_from_neff(neff=neff, x_values=wl_arr)
    np.testing.assert_allclose(beta, expected_beta, rtol=1e-12)


def test_propagation_constant_from_neff_omega_error_handling():
    """
    The ``from_neff_omega`` constructor must enforce matching lengths and the
    correct type for the angular frequency array.
    """
    neff = np.linspace(1.0, 1.1, 31)
    omega = AngularFrequencyArray(np.linspace(2.0, 4.0, 31), "rad/s")

    # Mismatched lengths raise ``ValueError``.
    with pytest.raises(ValueError):
        PropagationConstant.from_neff_omega(
            neff=np.array([1.0, 1.1, 1.2]),
            omega=omega,
        )

    # Wrong type for ``omega`` raises ``TypeError``.
    with pytest.raises(TypeError):
        PropagationConstant.from_neff_omega(
            neff=neff,
            omega=np.array([2.0, 4.0]),  # type: ignore[arg-type]  # not an AngularFrequencyArray
        )

    # Valid construction should succeed.
    pc = PropagationConstant.from_neff_omega(neff=neff, omega=omega)
    assert isinstance(pc, PropagationConstant)
    # The stored values should be exactly the computed β values.
    from photonics_helper.base import C_MS

    expected = omega.as_rad_s * neff / C_MS
    np.testing.assert_allclose(getattr(pc, "values"), expected, rtol=1e-12)
