"""Energy-based launch and amplitude-convention guard (Gaps F/G)."""

from __future__ import annotations

import numpy as np
import pytest

from photonics_helper.base import Area, Length, Power, Time, Wavelength
from photonics_helper.core.grids import TemporalGrid
from photonics_helper.gnlse import Fiber, SplitStepEngine
from photonics_helper.multimode_gnlse import MultimodeSplitStepEngine
from photonics_helper.pulse import Envelope, Wave

SHAPES = [
    "gaussian",
    "sech",
    "lorentzian",
    "rectangular",
    "triangular",
    "parabolic",
    "cosine",
    "exponential",
    "super-gaussian",
]


def _dense_energy(env: Envelope) -> float:
    T0 = env.pulse_width.as_s
    grid = TemporalGrid(N=2**17, Tmax=Time(max(200 * T0, 1e-12), "s"))
    return float(np.trapezoid(np.abs(env.field(grid.t)) ** 2, grid.t))


class TestFromEnergy:
    @pytest.mark.parametrize("shape", SHAPES)
    def test_energy_and_fwhm_roundtrip(self, shape):
        energy = 5.6e-12
        kw = {"super_gaussian_order": 2} if shape == "super-gaussian" else {}
        env = Envelope.from_energy(
            shape, energy, Time(200e-15, "s"), **kw
        )
        assert np.isclose(_dense_energy(env) / energy, 1.0, atol=2e-3)
        assert env.fwhm.as_fs == pytest.approx(200.0, rel=1e-9)
        assert env.measured_width().as_fs == pytest.approx(200.0, rel=1e-9)
        assert env.amplitude_convention == "power"

    def test_airy_has_no_whole_line_energy(self):
        with pytest.raises(ValueError, match="not in L"):
            Envelope.from_energy("airy", 1e-12, Time(200e-15, "s"))

    def test_super_gaussian_order_is_honoured(self):
        e = 1e-12
        env2 = Envelope.from_energy(
            "super-gaussian", e, Time(200e-15, "s"), super_gaussian_order=2
        )
        env4 = Envelope.from_energy(
            "super-gaussian", e, Time(200e-15, "s"), super_gaussian_order=4
        )
        assert env2.pulse_width.as_s != env4.pulse_width.as_s
        assert np.isclose(_dense_energy(env2) / e, 1.0, atol=3e-3)
        assert np.isclose(_dense_energy(env4) / e, 1.0, atol=3e-3)


class TestAmplitudeConvention:
    def test_field_factory_is_marked(self):
        env = Envelope.from_fwhm_peak_power(
            "gaussian", Power(1.0, "W"), Time(50e-15, "s"), Area(80, "um^2")
        )
        assert env.amplitude_convention == "field"
        env2 = Envelope.from_peak_power(
            "sech", Power(1.0, "W"), Time(50e-15, "s"), Area(80, "um^2")
        )
        assert env2.amplitude_convention == "field"

    def test_engine_rejects_field_units(self):
        env = Envelope.from_fwhm_peak_power(
            "gaussian", Power(1.0, "W"), Time(50e-15, "s"), Area(80, "um^2")
        )
        wave = Wave.from_envelope(env, Wavelength(1550, "nm"))
        fiber = Fiber(
            n2=2.6e-20,
            alpha=0.0,
            A_eff=Area(80e-12, "m^2"),
            length=Length(1.0, "m"),
        )
        with pytest.raises(ValueError, match="field units"):
            SplitStepEngine(wave, fiber, np.array([-21.8]))
        with pytest.raises(ValueError, match="field units"):
            MultimodeSplitStepEngine([wave], fiber, np.array([-21.8]))

    def test_to_engine_units_makes_peak_power(self):
        target = 1.0
        area = Area(80, "um^2")
        env = Envelope.from_fwhm_peak_power(
            "gaussian", Power(target, "W"), Time(50e-15, "s"), area
        )
        wave = Wave.from_envelope(env, Wavelength(1550, "nm"))
        converted = wave.to_engine_units(area)
        assert converted.amplitude_convention == "power"
        peak = float(np.max(np.abs(converted.envelope_field) ** 2))
        assert peak == pytest.approx(target, rel=1e-12)

    def test_power_envelope_to_engine_units_is_noop(self):
        env = Envelope.from_energy("gaussian", 1e-12, Time(50e-15, "s"))
        area = Area(80, "um^2")
        conv = env.to_engine_units(area)
        assert conv.peak_amplitude == env.peak_amplitude
        assert conv.amplitude_convention == "power"


class TestWidthMetadata:
    """Gap G: Airy metadata vs measurement, plus the parabolic factor."""

    @pytest.mark.parametrize(
        "shape",
        [
            "gaussian",
            "sech",
            "lorentzian",
            "rectangular",
            "triangular",
            "parabolic",
            "cosine",
            "exponential",
            "super-gaussian",
            "airy",
        ],
    )
    def test_metadata_equals_measured_fwhm(self, shape):
        env = Envelope.from_fwhm(shape, 1.0, Time(200e-15, "s"))
        assert env.measured_width().as_fs == pytest.approx(
            env.fwhm.as_fs, rel=1e-12
        )
        grid = TemporalGrid(N=2**12, Tmax=Time(10 * env.pulse_width.as_s, "s"))
        wave = Wave.from_envelope(env, Wavelength(1550, "nm"), grid=grid)
        assert wave.calc_width().as_fs == pytest.approx(env.fwhm.as_fs, rel=1e-12)

    def test_airy_is_documented_main_lobe(self):
        env = Envelope.from_fwhm("airy", 1.0, Time(200e-15, "s"))
        # The oscillating tail re-crosses half-max far out; that outermost
        # crossing is deliberately not the API width.
        from photonics_helper.pulse import _crossing_width

        grid = TemporalGrid(N=2**13, Tmax=Time(2e-12, "s"))
        outer = _crossing_width(grid.t, np.abs(env.field(grid.t)) ** 2)
        assert outer * 1e15 > 2.5 * env.fwhm.as_fs


class TestRamanResponseKwarg:
    def test_scalar_kwarg_matches_fiber_attribute(self):
        from photonics_helper.raman import RamanSpec

        grid = TemporalGrid(N=2**10, Tmax=Time(60e-12, "s"))
        spec = RamanSpec(
            name="Silica", raman_shift_cm=440.0, raman_linewidth_cm=45.0, fR=0.18
        )
        resp = spec.response(grid, fR=0.18, tau1=12.2e-15, tau2=32e-15)
        betas = np.array([-21.8])
        fiber_attr = Fiber(
            n2=2.6e-20,
            alpha=0.0,
            A_eff=Area(80e-12, "m^2"),
            length=Length(5.0, "m"),
            raman_response=resp,
        )
        fiber_none = Fiber(
            n2=2.6e-20,
            alpha=0.0,
            A_eff=Area(80e-12, "m^2"),
            length=Length(5.0, "m"),
        )
        wave = Wave(
            grid=grid,
            envelope=Envelope(
                shape="sech", peak_amplitude=2.0, pulse_width=Time(5e-12, "s")
            ),
            central_wavelength=Wavelength(1550, "nm"),
        )
        a = SplitStepEngine(
            wave, fiber_attr, betas, include_raman=True, step_size=Length(0.1, "m")
        )
        b = SplitStepEngine(
            wave,
            fiber_none,
            betas,
            include_raman=True,
            raman_response=resp,
            step_size=Length(0.1, "m"),
        )
        a.propagate(10)
        b.propagate(10)
        assert np.max(np.abs(a.A - b.A)) < 1e-12
