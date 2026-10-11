"""Contract tests for the :class:`photonics_helper.pulse.ElectricField` class.

Covers:
- construction validation (1-D, length matches grid)
- ``from_envelope`` carrier reconstruction convention (single source of truth)
- ``to_envelope`` roundtrip amplitude + phase preservation
- intensity scale ``½·n·c·ε₀·|E|²`` (absolute SI check)
- power / peak_power / pulse_energy unit discipline and the required ``A_eff``
- Parseval identity between time and spectral domain under the grid FFT
"""

import numpy as np
import pytest

from photonics_helper import (
    AngularFrequency,
    Area,
    ElectricField,
    TemporalGrid,
    Time,
    Wavelength,
)
from photonics_helper.base import C_MS, EPS_0

N = 2**13
WINDOW = 2e-12  # s — 2 ps window, dt ≈ 0.244 fs
LAMBDA0 = Wavelength(1550, "nm")
N_AIR = 1.0


@pytest.fixture()
def grid():
    return TemporalGrid(N=N, Tmax=Time(WINDOW, "s"))


@pytest.fixture()
def envelope(grid):
    """Gaussian Envelope-like analytic field centered at t = 0."""
    t = grid.t
    A = np.exp(-((t / 50e-15) ** 2)).astype(complex)
    A *= np.exp(-0.2j * (t / 50e-15) ** 2)  # mild chirp
    return A


@pytest.fixture()
def efield(grid, envelope):
    return ElectricField.from_envelope(envelope, grid, LAMBDA0, N_AIR)


# -- construction ----------------------------------------------------------


class TestConstruction:
    def test_length_mismatch_raises(self, grid):
        with pytest.raises(ValueError, match="match grid time samples"):
            ElectricField(
                field=np.ones(N - 1, dtype=complex), grid=grid,
                central_frequency=LAMBDA0.to_omega(),
            )

    def test_non_1d_raises(self, grid):
        with pytest.raises(ValueError, match="1-D array"):
            ElectricField(
                field=np.ones((4, 4), dtype=complex), grid=grid,
                central_frequency=LAMBDA0.to_omega(),
            )

    def test_stored_as_complex(self, grid):
        ef = ElectricField(
            field=np.ones(N), grid=grid, central_frequency=LAMBDA0.to_omega()
        )
        assert np.iscomplexobj(ef.field)


# -- carrier convention ------------------------------------------------------


class TestCarrierConvention:
    def test_from_envelope_uses_exp_minus_i_omega0_t(self, grid, envelope):
        ef = ElectricField.from_envelope(envelope, grid, LAMBDA0, N_AIR)
        expected = envelope * np.exp(-1j * LAMBDA0.to_omega().as_rad_s * grid.t)
        np.testing.assert_allclose(ef.field, expected)
        assert isinstance(ef.central_frequency, AngularFrequency)
        assert ef.central_frequency.as_rad_s == pytest.approx(
            LAMBDA0.to_omega().as_rad_s
        )

    def test_real_field_matches_carried_carrier(self, efield, envelope, grid):
        """Re[E(t)] must be the standard A·cos reconstruction."""
        expected = np.real(envelope * np.exp(-1j * LAMBDA0.to_omega().as_rad_s * grid.t))
        np.testing.assert_allclose(efield.real_field, expected)


# -- roundtrip ----------------------------------------------------------------


class TestEnvelopeRoundtrip:
    def test_roundtrip_preserves_amplitude(self, efield, envelope):
        recovered = efield.to_envelope()
        # Focus on the pulse bulk (convolution edge smearing lives in the tails).
        mask = np.abs(envelope) > 1e-2
        np.testing.assert_allclose(np.abs(recovered[mask]), np.abs(envelope[mask]),
                                   rtol=1e-2)

    def test_roundtrip_preserves_phase(self, efield, envelope):
        recovered = efield.to_envelope()
        mask = np.abs(envelope) > 1e-2
        phase_err = np.angle(recovered[mask] * np.conj(envelope[mask]))
        # Global phase is free (definitions may differ by a constant); variation
        # across the bulk is what chirp fidelity means.
        assert np.std(phase_err) < 5e-2
        assert np.abs(np.mean(phase_err)) < 5e-2


# -- SI intensity scale -------------------------------------------------------


class TestIntensityScale:
    def test_vacuum_scale_matches_sconstants(self, efield):
        assert efield.intensity_scale == pytest.approx(0.5 * C_MS * EPS_0)

    def test_absolute_intensity_of_known_field(self, grid):
        """A plane wave with E₀ = 1e8 V/m in vacuum → I = ½cε₀E₀² ≈ 1.33e13 W/m²."""
        ef = ElectricField(
            field=np.full(N, 1e8, dtype=complex), grid=grid,
            central_frequency=LAMBDA0.to_omega(), refractive_index=1.0,
        )
        expected = 0.5 * C_MS * EPS_0 * 1e16  # ≈ 1.327e13 W/m²
        assert ef.intensity[0] == pytest.approx(expected)


# -- power & energy unit discipline --------------------------------------------


class TestPowerEnergy:
    A_EFF = Area(80, "um^2")

    def test_power_is_intensity_times_area(self, efield):
        expected = self.A_EFF.as_m2 * efield.intensity
        np.testing.assert_allclose(efield.power(self.A_EFF), expected)

    def test_peak_power_is_max_of_power(self, efield):
        assert efield.peak_power(self.A_EFF) == pytest.approx(
            float(np.max(efield.power(self.A_EFF)))
        )

    def test_pulse_energy_units_joule(self, efield, grid):
        e = efield.pulse_energy(self.A_EFF)
        manual = float(np.trapezoid(efield.power(self.A_EFF), grid.t))
        assert e == pytest.approx(manual)
        assert 0 < e < np.max(efield.power(self.A_EFF)) * grid.time_window

    def test_energy_scales_linearly_with_area(self, efield):
        e1 = efield.pulse_energy(Area(80, "um^2"))
        e2 = efield.pulse_energy(Area(160, "um^2"))
        assert e2 == pytest.approx(2 * e1)


# -- Parseval / spectral domain -------------------------------------------------


class TestParseval:
    def test_time_and_frequency_domain_energy_densities_recorded(self, efield, grid):
        """∫|E(t)|²dt == ∫|Ê(Ω)|²dΩ under grid.fft / grid.ifft conventions
        (exact for the periodic transform pair on this grid)."""
        e_t = float(np.sum(np.abs(efield.field) ** 2) * grid.dt)
        e_w = float(np.sum(efield.spectral_intensity) * grid.dw)
        # The convention pair is unitary for periodic grids; compare within 1e-6.
        assert e_w == pytest.approx(e_t, rel=1e-6)

    def test_iffft_inverts_fft(self, efield):
        np.testing.assert_allclose(efield.grid.ifft(efield.spectrum), efield.field,
                                   atol=1e-9)


# -- phase helper ------------------------------------------------------------------


class TestPhase:
    def test_instantaneous_phase_unwrapped_length(self, efield):
        phase = efield.instantaneous_phase()
        assert phase.shape == efield.field.shape
        # Differential of an unwrapped phase cannot jump by ~2π.
        assert np.max(np.abs(np.diff(phase))) < np.pi
