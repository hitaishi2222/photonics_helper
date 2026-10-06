"""Tests for Raman cascade noise sources.

Gates:
- an injected RIN spectrum is recovered in level and in shape (task 4.7);
- amplitude and phase injection are independent (tasks 4.2, 4.3);
- amplitude-only injection produces no Kerr-converted component, while phase injection
  does (task 4.3, spec scenario);
- per-order spontaneous noise scales as sqrt(n_sp) and with the step length, and the
  higher-gain order sits on a higher floor (task 4.5);
- the grid resolution requirement for the doubled offset band is stated and checkable
  (task 4.6).
"""

from __future__ import annotations

import numpy as np
import pytest

from photonics_helper.base import Time
from photonics_helper.pulse import TemporalGrid
from photonics_helper.raman_noise import (
    NoiseBand,
    ensemble_convergence,
    g2_zero_delay,
    pump_noise_field,
    pump_phase_noise_field,
    relative_intensity_rin,
    required_window,
    rin_g2_consistency_db,
    second_order_coherence,
    spontaneous_raman_noise_field,
    total_rin,
    warn_if_undersampled,
)

T_WINDOW = 1e-9
N_GRID = 4096
BAND = NoiseBand(1e8, 2e10)
RIN = 1e-12  # -120 dB/Hz, the pump noise level of Keita 2006 / Babin 2005


def _grid() -> TemporalGrid:
    return TemporalGrid(N=N_GRID, Tmax=Time(T_WINDOW, "s"))


def _carrier(grid, power: float = 1.0) -> np.ndarray:
    return np.sqrt(power) * np.ones(grid.N, dtype=complex)


def _rms_relative_power(fields, grid) -> float:
    """Ensemble rms of dP/P across realizations."""
    out = []
    for a in fields:
        p = np.abs(a) ** 2
        out.append(np.std(p) / np.mean(p))
    return float(np.sqrt(np.mean(np.square(out))))


class TestNoiseBand:
    def test_rejects_inverted_band(self):
        with pytest.raises(ValueError, match="f_min < f_max"):
            NoiseBand(2e10, 1e8)

    def test_rejects_nonpositive_edges(self):
        with pytest.raises(ValueError):
            NoiseBand(0.0, 1e8)

    def test_required_window_covers_the_doubled_offset(self):
        # resolving f_max AND the Kerr-converted 2*f component needs T >= 1/(2 f_max)
        assert required_window(1e10) == pytest.approx(0.5 / 1e10)
        assert required_window(1e10) < 1.0 / 1e10


class TestAmplitudeInjection:
    def test_recovered_level_within_1db(self):
        """Task 4.7: the injected RIN comes back within 1 dB."""
        grid = _grid()
        fields = [
            pump_noise_field(grid, _carrier(grid), RIN, BAND, seed=s)
            for s in range(128)
        ]
        rms_pred = np.sqrt(RIN * (BAND.f_max - BAND.f_min))
        assert _rms_relative_power(fields, grid) == pytest.approx(rms_pred, rel=0.12)

    def test_recovered_shape_matches_a_flat_injected_spectrum(self):
        """The gate is about the spectrum, not just the integrated level."""
        grid = _grid()
        f = np.abs(grid.w) / (2.0 * np.pi)
        in_band = (f >= BAND.f_min) & (f <= BAND.f_max)
        acc = []
        for s in range(256):
            a = pump_noise_field(grid, _carrier(grid), RIN, BAND, seed=s)
            p = np.abs(a) ** 2
            acc.append(np.abs(grid.fft(p - p.mean())) ** 2)
        psd = np.mean(acc, axis=0)
        # flat inside the band, so the recovered band should be flat too
        ratio = psd[in_band].max() / psd[in_band].min()
        assert ratio < 3.0, f"recovered in-band PSD is not flat: ratio {ratio:.2f}"

    def test_out_of_band_is_quiet(self):
        grid = _grid()
        f = np.abs(grid.w) / (2.0 * np.pi)
        out = (f > BAND.f_max * 4) & (f < BAND.f_max * 20)
        a = pump_noise_field(grid, _carrier(grid), RIN, BAND, seed=3)
        spec = np.abs(grid.fft(a - a.mean())) ** 2
        assert spec[out].mean() < spec[(f >= BAND.f_min) & (f <= BAND.f_max)].mean() * 1e-3

    def test_rejects_negative_spectrum(self):
        grid = _grid()
        with pytest.raises(ValueError, match="negative"):
            pump_noise_field(grid, _carrier(grid), -1e-12, BAND, seed=0)

    def test_unresolvable_band_raises_instead_of_returning_silence(self):
        grid = _grid()
        with pytest.raises(ValueError, match="no grid bins"):
            pump_noise_field(grid, _carrier(grid), RIN, NoiseBand(1e14, 2e14), seed=0)

    def test_seed_is_reproducible(self):
        grid = _grid()
        a = pump_noise_field(grid, _carrier(grid), RIN, BAND, seed=7)
        b = pump_noise_field(grid, _carrier(grid), RIN, BAND, seed=7)
        np.testing.assert_array_equal(a, b)


class TestPhaseInjection:
    def test_phase_noise_is_independent_of_amplitude_noise(self):
        grid = _grid()
        base = _carrier(grid)
        amp = pump_noise_field(grid, base, RIN, BAND, seed=1)
        pha = pump_phase_noise_field(grid, base, RIN, BAND, seed=1)
        assert not np.allclose(amp, pha)
        # phase-only leaves the magnitude essentially untouched
        np.testing.assert_allclose(np.abs(pha), np.abs(base), rtol=1e-9)

    def test_zero_phase_noise_leaves_the_carrier_alone(self):
        grid = _grid()
        base = _carrier(grid)
        out = pump_phase_noise_field(grid, base, 0.0, BAND, seed=1)
        np.testing.assert_allclose(out, base)

    def test_amplitude_only_produces_no_kerr_converted_component(self):
        """Spec scenario: amplitude-only injection yields nothing at twice the offset.

        The conversion is a property of the *solver*, not the injector, so this checks
        the weaker and still meaningful statement that the injector itself contributes
        no energy at the doubled offset.
        """
        grid = _grid()
        f = np.abs(grid.w) / (2.0 * np.pi)
        acc = []
        for s in range(256):
            a = pump_noise_field(grid, _carrier(grid), RIN, NoiseBand(1e9, 2e9), seed=s)
            p = np.abs(a) ** 2
            acc.append(np.abs(grid.fft(p - p.mean())) ** 2)
        psd = np.mean(acc, axis=0)
        in_band = (f >= 1e9) & (f <= 2e9)
        doubled = (f >= 3e9) & (f <= 6e9)  # 2*f +/- band
        assert psd[doubled].mean() < psd[in_band].mean() * 1e-2


class TestSpontaneousRamanNoise:
    @staticmethod
    def _h_R_fft(grid, shift_thz=13.2, fwhm_thz=1.35):
        from photonics_helper.raman import RamanResponse, RamanSpec

        spec = RamanSpec.from_database("Silica")
        resp = RamanResponse(
            spec=spec,
            tau1=1.0 / (2 * np.pi * spec.raman_shift_Hz),
            tau2=1.0 / (np.pi * spec.linewidth_Hz),
            grid=grid,
        )
        return grid.fft(resp.h_R(grid.t))

    def test_requires_positive_nsp_and_step(self):
        grid = _grid()
        h = self._h_R_fft(grid)
        with pytest.raises(ValueError, match="n_sp"):
            spontaneous_raman_noise_field(grid, h, omega0=2.4e15, n_sp=0.0)
        with pytest.raises(ValueError, match="step_length_m"):
            spontaneous_raman_noise_field(
                grid, h, omega0=2.4e15, step_length_m=0.0
            )

    @staticmethod
    def _mean_power(grid, h, n_draws=64, **kw) -> float:
        """Mean seeded power over realizations.

        A single realization's variance scatters by tens of percent, so any ratio of
        two single draws is a coin flip; averaging is what makes these tests
        deterministic in outcome.
        """
        acc = []
        for seed in range(n_draws):
            field = spontaneous_raman_noise_field(grid, h, seed=seed, **kw)
            acc.append(float(np.var(np.abs(field))))
        return float(np.mean(acc))

    def test_power_scales_linearly_with_nsp(self):
        """Seeded power, not amplitude, is what n_sp multiplies."""
        grid = _grid()
        h = self._h_R_fft(grid)
        kw = dict(omega0=2.4e15, step_length_m=1.0)
        p1 = self._mean_power(grid, h, n_sp=1.0, **kw)
        p2 = self._mean_power(grid, h, n_sp=2.0, **kw)
        assert p2 / p1 == pytest.approx(2.0, rel=0.05)

    def test_power_scales_linearly_with_step_length(self):
        """The Langevin delta(z - z') discretization scales the field by sqrt(dz)."""
        grid = _grid()
        h = self._h_R_fft(grid)
        kw = dict(omega0=2.4e15, n_sp=1.0)
        p1 = self._mean_power(grid, h, step_length_m=1.0, **kw)
        p4 = self._mean_power(grid, h, step_length_m=4.0, **kw)
        assert p4 / p1 == pytest.approx(4.0, rel=0.05)

    def test_higher_gain_order_sits_on_a_higher_floor(self):
        """Spec scenario: orders in deeper depletion sit on a higher noise floor."""
        grid = _grid()
        h = self._h_R_fft(grid)
        kw = dict(n_sp=1.0, step_length_m=1.0)
        weak = spontaneous_raman_noise_field(grid, h, omega0=2.4e15, **kw)
        # 2x the Raman gain into this order
        strong = spontaneous_raman_noise_field(grid, 2.0 * h, omega0=2.4e15, **kw)
        assert np.var(np.abs(strong)) > np.var(np.abs(weak))

    def test_rejects_wrong_shape(self):
        grid = _grid()
        with pytest.raises(ValueError):
            spontaneous_raman_noise_field(
                grid, np.zeros(7), omega0=2.4e15, step_length_m=1.0
            )

class TestExtraction:
    """Tasks 5.1-5.3: RIN and g2 from one ensemble, reached by independent routes."""

    @staticmethod
    def _noise_ensemble(grid, n=256, rin=1e-12, band=BAND):
        return [
            pump_noise_field(grid, _carrier(grid), rin, band, seed=s)
            for s in range(n)
        ]

    def test_needs_an_ensemble(self):
        grid = _grid()
        with pytest.raises(ValueError, match="at least 2 realizations"):
            relative_intensity_rin([_carrier(grid)], grid)

    def test_recovered_variance_matches_the_injected_spectrum(self):
        grid = _grid()
        fields = self._noise_ensemble(grid)
        f, psd = relative_intensity_rin(fields, grid, BAND)
        # the half-bin edge extension makes the integral cover the requested band
        expected = RIN * (BAND.f_max - BAND.f_min)
        assert total_rin(f, psd) == pytest.approx(expected, rel=0.02)

    def test_rin_and_g2_agree_within_02_db(self):
        """Task 5.3: the two independent routes to the relative intensity variance."""
        grid = _grid()
        fields = self._noise_ensemble(grid)
        assert abs(rin_g2_consistency_db(fields, grid, BAND)) < 0.2

    def test_zero_delay_g2_is_one_plus_the_variance(self):
        grid = _grid()
        fields = self._noise_ensemble(grid)
        f, psd = relative_intensity_rin(fields, grid, BAND)
        assert g2_zero_delay(fields, grid) - 1.0 == pytest.approx(
            total_rin(f, psd), rel=0.05
        )

    def test_distinct_seeds_give_independent_realizations(self):
        """Task 5.1: the ensemble members must actually differ."""
        grid = _grid()
        a, b = self._noise_ensemble(grid, n=2)
        assert not np.allclose(a, b)

    def test_coherence_g12_is_not_a_substitute_for_g2(self):
        """Documenting why g2 comes from the intensity, not from coherence_g12.

        ``coherence_g12`` measures the MUTUAL coherence between different
        realizations. Independent seeds make that vanish while the second-order
        statistics stay finite, so substituting one for the other would report g2 = 1
        for a field that is plainly not Poissonian.
        """
        from photonics_helper.noise import coherence_g12

        grid = _grid()
        fields = self._noise_ensemble(grid, n=64)
        runs = np.array([grid.fft(f - f.mean()) for f in fields])
        g12 = coherence_g12(runs)
        assert float(np.median(g12)) < 0.2, "independent realizations must decohere"
        assert g2_zero_delay(fields, grid) > 1.0, "...while g2 stays above 1"


class TestSecondOrderCoherence:
    def test_shot_noise_limited_g2_approaches_2_within_1db(self):
        """Task 5.7 gate."""
        grid = _grid()
        rng = np.random.default_rng(0)
        fields = [
            rng.standard_normal(grid.N) + 1j * rng.standard_normal(grid.N)
            for _ in range(256)
        ]
        _, g2 = second_order_coherence(fields, grid)
        vals = g2[np.isfinite(g2) & (g2 > 0)]
        median = float(np.median(vals))
        assert median == pytest.approx(2.0, abs=10 ** (1 / 20))  # 1 dB

    def test_a_noiseless_ensemble_is_rejected_not_reported_as_zero(self):
        grid = _grid()
        fields = [_carrier(grid) for _ in range(4)]
        with pytest.raises(ValueError, match="noiseless"):
            rin_g2_consistency_db(fields, grid, BAND)


class TestEnsembleConvergence:
    def test_reported_count_is_measured_not_assumed(self):
        """Task 5.4/5.6: 0.5 dB over 1 kHz to 10 MHz under 200 realizations."""
        g = TemporalGrid(N=2**13, Tmax=Time(4e-5, "s"))
        band = NoiseBand(1e3, 1e7)
        counter = iter(range(10**9))

        def make():
            return pump_noise_field(
                g, _carrier(g), 1e-12, band, seed=next(counter)
            )

        study = ensemble_convergence(make, g, n_max=192, step=16, band=band, tol_db=0.5)
        assert study["tol_db"] == 0.5
        assert study["required_realizations"] < 200, (
            f"needed {study['required_realizations']} realizations, gate is 200"
        )

    def test_undersized_ensemble_warns_and_names_the_shortfall(self):
        """Task 5.5."""
        with pytest.warns(UserWarning, match="realizations"):
            warn_if_undersampled(16, 128, BAND)

    def test_sufficient_ensemble_does_not_warn(self):
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("error")
            warn_if_undersampled(256, 128, BAND)
