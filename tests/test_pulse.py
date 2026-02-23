"""
Unit tests for the :mod:`photonics_helper.pulse` module.

The tests focus on the core numerical properties of the ``Envelope``,
``TemporalGrid`` and ``Wave`` classes – ensuring that conversions between
FWHM and pulse width, intensity calculations, energy/power evaluation and the
time‑bandwidth product for a transform‑limited Gaussian pulse behave as
expected.
"""

import numpy as np
import pytest

from photonics_helper.base import Wavelength
from photonics_helper.pulse import SHAPE_FACTORS, Envelope, TemporalGrid, Wave


@pytest.fixture
def gaussian_wave():
    """Create a transform‑limited Gaussian pulse for use in several tests.

    The pulse has a peak amplitude of 1.0 and a full‑width‑half‑maximum of
    100 fs.  A temporal window ten times larger than the FWHM is chosen to avoid
    aliasing.
    """
    fwhm = 100e-15  # 100 fs
    envelope = Envelope.from_fwhm(shape="gaussian", peak_amplitude=1.0, fwhm=fwhm)
    # Temporal grid – use a power‑of‑two number of points for FFT efficiency.
    N = 2**12
    Tmax = 10 * fwhm
    grid = TemporalGrid(N=N, Tmax=Tmax)
    wave = Wave(
        grid=grid,
        envelope=envelope,
        central_wavelength=Wavelength(800, "nm"),
    )
    return wave, fwhm


def test_envelope_fwhm_consistency():
    fwhm = 200e-15
    env = Envelope.from_fwhm(shape="gaussian", peak_amplitude=2.0, fwhm=fwhm)
    # The ``fwhm`` property should return the original value.
    assert pytest.approx(env.fwhm, rel=1e-12) == fwhm
    # Verify internal pulse width matches the analytical conversion.
    expected_T0 = fwhm / SHAPE_FACTORS["gaussian"]
    assert pytest.approx(env.pulse_width, rel=1e-12) == expected_T0


def test_wave_energy_and_power(gaussian_wave):
    wave, _ = gaussian_wave
    # Numerical integration of the intensity should match the ``pulse_energy``
    # convenience method within a small tolerance.
    numerical_energy = np.sum(wave.envelope_intensity) * wave.grid.dt
    assert pytest.approx(wave.pulse_energy(), rel=1e-6) == numerical_energy
    # Peak power – maximum of the intensity array.
    assert pytest.approx(wave.peak_power(), rel=1e-12) == np.max(
        wave.envelope_intensity
    )


def test_time_bandwidth_product_gaussian(gaussian_wave):
    wave, _ = gaussian_wave
    tbp = wave.time_bandwidth_product()
    # Theoretical value for a transform‑limited Gaussian pulse is approximately
    # 0.44 (time·angular‑frequency).  Allow a modest tolerance due to discretisation.
    assert 0.48 < tbp < 0.52


def test_wave_spectrum_shape(gaussian_wave):
    wave, _ = gaussian_wave
    # The spectrum is computed via FFT; its length must match the frequency grid.
    assert wave.spectrum.shape == wave.grid.w.shape
