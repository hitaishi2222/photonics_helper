"""Smooth-UX pass: peak-power factories, auto-grids, Fiber, Raman facade."""

import warnings

import numpy as np

from photonics_helper import (
    AngularFrequency,
    Area,
    Fiber,
    Frequency,
    Length,
    Power,
    Time,
    Wavelength,
)
from photonics_helper.gnlse import _fiber_gamma
from photonics_helper.pulse import ElectricField, Envelope, Wave


def test_envelope_from_peak_power_roundtrip():
    A_eff = Area(80, "um^2")
    env = Envelope.from_peak_power(
        "sech", Power(1, "kW"), Time(50, "fs"), A_eff, refractive_index=1.0
    )
    wave = Wave.from_envelope(env, Wavelength(1550, "nm")).with_effective_area(A_eff)
    assert wave.peak_power() == __import__("pytest").approx(1000.0, rel=0.05)


def test_envelope_from_fwhm_peak_power():
    A_eff = Area(80, "um^2")
    env = Envelope.from_fwhm_peak_power(
        "gaussian", Power(1, "W"), Time(50, "fs"), A_eff
    )
    assert env.pulse_width.as_s > 0
    assert env.peak_amplitude > 0


def test_wave_from_envelope_auto_grid():
    env = Envelope(shape="sech", peak_amplitude=1.0, pulse_width=Time(50, "fs"))
    wave = Wave.from_envelope(env, Wavelength(1550, "nm"))
    assert wave.grid.N == 2**12
    assert wave.repetition_rate is None


def test_wave_from_pulse_train_optional_grid():
    env = Envelope(shape="sech", peak_amplitude=1.0, pulse_width=Time(50, "fs"))
    single = Wave.from_pulse_train(env, Wavelength(1550, "nm"), n_pulses=1)
    assert single.grid.N > 0
    train = Wave.from_pulse_train(
        env,
        Wavelength(1550, "nm"),
        repetition_rate=Frequency(10, "GHz"),
        n_pulses=3,
    )
    assert train.repetition_rate is not None
    # Stored rate is reused when average_power gets no argument.
    assert train.average_power() > 0


def test_electric_field_constructors():
    from photonics_helper.core.grids import TemporalGrid

    env = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=Time(50, "fs"))
    grid = TemporalGrid(N=1024, Tmax=Time(2, "ps"))
    wl = Wavelength(1550, "nm")
    ef = ElectricField.from_envelope(env, grid, wl)
    assert ef.field.shape == (1024,)
    ef2 = ElectricField.from_array(np.asarray(env.field(grid.t)), grid, wl)
    assert np.allclose(ef.field, ef2.field)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        ef3 = ElectricField.from_envelope(np.asarray(env.field(grid.t)), grid, wl)
    assert np.allclose(ef3.field, ef2.field)


def test_fiber_alias_and_gamma_first():
    assert Fiber is Fiber
    f = Fiber.from_gamma(0.045, A_eff=Area(80, "um^2"))
    assert f.gamma == 0.045
    assert f.effective_gamma(AngularFrequency(2e15, "rad/s")) == 0.045
    assert _fiber_gamma(f, 2e15) == 0.045
    # Legacy path still back-computes A_eff.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy = Fiber.from_gamma(
            0.045, 2.6e-20, Wavelength(1550, "nm").to_omega()
        )
    assert legacy.A_eff is not None
    # n2-derived path still works.
    f2 = Fiber.from_n2(2.6e-20, Area(80, "um^2"), Length(0.1, "m"))
    assert f2.effective_gamma(Wavelength(1550, "nm").to_omega()) > 0


def test_raman_facade():
    from photonics_helper.raman.spec import RamanSpec

    spec = RamanSpec.from_database("Silica")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        resp = spec.response()
        freq = resp.frequency
        freq2 = spec.frequency_response()
    assert freq is not None and freq2 is not None
