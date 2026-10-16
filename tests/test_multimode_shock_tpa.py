"""Tests for step9 multimode shock + TPA/carriers (tasks 1.2/1.3/2.2)."""

import numpy as np
import pytest

from photonics_helper.base import Area, Length, Time, Wavelength
from photonics_helper.gnlse import Fiber, SplitStepEngine
from photonics_helper.multimode_gnlse import MultimodeSplitStepEngine
from photonics_helper.pulse import Envelope, TemporalGrid, Wave

WL = Wavelength(1550.0, "nm")


def _grid(t_ps: float = 8.0, n: int = 2**11) -> TemporalGrid:
    return TemporalGrid(N=n, Tmax=Time(t_ps * 1e-12, "s"))


def _sech_wave(grid, peak, width_ps: float = 0.5):
    env = Envelope(
        shape="sech",
        peak_amplitude=np.sqrt(peak),
        pulse_width=Time(width_ps * 1e-12, "s"),
    )
    return Wave(grid=grid, envelope=env, central_wavelength=WL)


def _fiber(length_m: float) -> Fiber:
    return Fiber(
        n2=2.6e-20,
        alpha=0.0,
        A_eff=Area(80e-12, "m^2"),
        length=Length(length_m, "m"),
    )


def test_single_channel_equals_scalar_shock():
    """N=1 shock == scalar shock deck at machine precision (task 1.2)."""
    grid = _grid()
    fiber = _fiber(0.5)
    betas = np.array([-21.8])
    scalar = SplitStepEngine(
        _sech_wave(grid, 4.0),
        fiber,
        betas,
        include_self_steepening=True,
        step_size=Length(0.025, "m"),
    )
    scalar.propagate(20)
    mm = MultimodeSplitStepEngine(
        [_sech_wave(grid, 4.0)],
        fiber,
        betas,
        include_shock=True,
        step_size=Length(0.025, "m"),
    )
    mm.propagate(20)
    ref = scalar.evolution[-1].envelope_field
    got = mm.fields_vs_z()[0][-1, :]
    assert np.max(np.abs(ref - got)) < 1e-9


def test_shock_grid_guard_rejects():
    """Under-resolved shock grid fails loudly (task 1.3)."""
    grid = TemporalGrid(N=2**14, Tmax=Time(1e-12, "s"))
    with pytest.raises(ValueError, match="Ω_max"):
        MultimodeSplitStepEngine(
            [_sech_wave(grid, 4.0)],
            _fiber(0.5),
            np.array([-21.8]),
            include_shock=True,
        )


def test_conserving_shock_rejects_coupled_paths():
    """pcGNLSE variant is only defined on the per-channel default path."""
    grid = _grid()
    fiber = _fiber(0.5)
    with pytest.raises(ValueError, match="conserving_shock"):
        MultimodeSplitStepEngine(
            [_sech_wave(grid, 4.0), _sech_wave(grid, 1.0)],
            fiber,
            [np.array([-21.8]), np.array([-21.8])],
            include_shock=True,
            include_fwm=True,
            conserving_shock=True,
        )
    with pytest.raises(ValueError, match="conserving_shock"):
        MultimodeSplitStepEngine(
            [_sech_wave(grid, 4.0)],
            fiber,
            np.array([-21.8]),
            include_shock=True,
            conserving_shock=True,
            coupling_tensor=np.zeros((1, 1), dtype=complex),
        )


def test_shock_photon_tracking():
    """Shock run records Blow & Wood photon number with small drift."""
    grid = _grid()
    mm = MultimodeSplitStepEngine(
        [_sech_wave(grid, 4.0)],
        _fiber(0.5),
        np.array([-21.8]),
        include_shock=True,
        step_size=Length(0.025, "m"),
    )
    mm.propagate(20)
    assert mm._photon_vs_z is not None and len(mm._photon_vs_z) > 2
    q0 = mm._photon_vs_z[0]
    assert all(abs(q - q0) / q0 < 0.02 for q in mm._photon_vs_z)


def _cw_wave(grid, peak) -> Wave:
    const = float(np.sqrt(peak))

    def func(t, T0, A0):
        return const * np.ones_like(t)

    env = Envelope(
        shape="custom",
        peak_amplitude=np.sqrt(peak),
        func=func,
        pulse_width=Time(1e-12, "s"),
    )
    return Wave(grid=grid, envelope=env, central_wavelength=WL)


def test_tpa_analytic_attenuation():
    """CW TPA follows I(L) = I0/(1 + β·I0·L) per channel (task 2.2)."""
    grid = _grid(40.0, 2**10)
    fiber = _fiber(1.0)
    beta = 1e-11  # m/W
    I0 = 1e12  # W/m²-scale intensity in envelope units
    mm = MultimodeSplitStepEngine(
        [_cw_wave(grid, I0), _cw_wave(grid, I0 / 4)],
        fiber,
        [np.array([0.0]), np.array([0.0])],
        include_free_carriers=True,
        beta_tpa=[beta, beta],
        step_size=Length(0.05, "m"),
    )
    mm.propagate(20)  # L = 1 m
    trajs = mm.fields_vs_z()
    for ch, p0 in ((0, I0), (1, I0 / 4)):
        got = float(np.mean(np.abs(trajs[ch][-1]) ** 2))
        expect = p0 / (1.0 + beta * p0 * 1.0)
        assert abs(got - expect) / expect < 1e-6
