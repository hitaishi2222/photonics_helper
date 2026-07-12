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
        central_wavelength_nm=1550.0,
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
    assert pytest.approx(disp.fn(test_wl_m), rel=1e-12) == 1e-6

    # The two convenience wrappers should give identical results.
    test_wl_nm = 1550.0
    assert pytest.approx(disp.fn_ps_nm_km(test_wl_nm), rel=1e-12) == 1e-6 * 1e6
    assert pytest.approx(disp.fn_s_m_m(test_wl_nm), rel=1e-12) == 1e-6


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
            central_wavelength_nm=1550.0,
        )

    # Passing a plain list instead of a ``WavelengthArray`` should raise ``TypeError``.
    with pytest.raises(TypeError):
        Dispersion.from_neff(
            neff=np.array([1.5, 1.51]),
            wavelengths=[1500.0, 1550.0],  # type: ignore[arg-type]  # not a WavelengthArray
            central_wavelength_nm=1550.0,
        )


def test_propagation_constant_beta2_from_neff():
    """
    ``PropagationConstant.beta2_from_neff`` should compute β₂ = n·ω / c.
    We verify the result against a manual calculation for a simple case.
    """
    neff = np.array([1.0, 1.2, 1.5])
    # Use a wavelength grid (nm) and obtain the corresponding angular frequencies.
    wl_arr = WavelengthArray(np.array([1500.0, 1550.0, 1600.0]), "nm")
    omega_arr = wl_arr.to_omega()

    # Expected β₂ values: β₂ = n·ω / c (c = C_MS)
    from photonics_helper.base import C_MS

    expected_beta2 = neff * omega_arr.as_rad_s / C_MS

    beta2 = PropagationConstant.beta2_from_neff(neff=neff, x_values=wl_arr)
    np.testing.assert_allclose(beta2, expected_beta2, rtol=1e-12)


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
    np.testing.assert_allclose(getattr(pc, "_values"), expected, rtol=1e-12)
