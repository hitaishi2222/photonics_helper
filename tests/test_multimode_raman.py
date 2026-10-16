"""Tests for step8 multimode Raman (task 1.2 / 2.3)."""

import numpy as np
import pytest

from photonics_helper.base import Area, Length, Time, Wavelength
from photonics_helper.gnlse import Fiber, SplitStepEngine
from photonics_helper.multimode_gnlse import MultimodeSplitStepEngine
from photonics_helper.pulse import Envelope, TemporalGrid, Wave
from photonics_helper.raman import RamanSpec

WL = Wavelength(1550.0, "nm")


def _grid(t_ps: float = 60.0, n: int = 2**10) -> TemporalGrid:
    return TemporalGrid(N=n, Tmax=Time(t_ps * 1e-12, "s"))


def _resp(grid):
    spec = RamanSpec(
        name="Silica", raman_shift_cm=440.0, raman_linewidth_cm=45.0, fR=0.18
    )
    return spec.response(grid, fR=0.18, tau1=12.2e-15, tau2=32e-15)


def _sech_wave(grid, peak, width_ps: float = 5.0):
    env = Envelope(
        shape="sech",
        peak_amplitude=np.sqrt(peak),
        pulse_width=Time(width_ps * 1e-12, "s"),
    )
    return Wave(grid=grid, envelope=env, central_wavelength=WL)


def test_single_channel_equals_scalar_with_raman():
    """N=1 with Raman == scalar engine at machine precision (task 1.2)."""
    grid = _grid()
    resp = _resp(grid)
    fiber = Fiber(
        n2=2.6e-20,
        alpha=0.0,
        A_eff=Area(80e-12, "m^2"),
        length=Length(10.0, "m"),
        raman_response=resp,
    )
    betas = np.array([-21.8])
    scalar = SplitStepEngine(
        _sech_wave(grid, 4.0, 5.0),
        fiber,
        betas,
        include_raman=True,
        step_size=Length(0.1, "m"),
    )
    scalar.propagate(20)
    mm = MultimodeSplitStepEngine(
        [_sech_wave(grid, 4.0, 5.0)],
        fiber,
        betas,
        include_raman=True,
        raman_response=resp,
        step_size=Length(0.1, "m"),
    )
    mm.propagate(20)
    ref = scalar.evolution[-1].envelope_field
    got = mm.fields_vs_z()[0][-1, :]
    assert np.max(np.abs(ref - got)) < 1e-9


def test_raman_tensors_require_include_raman():
    """Intermodal tensors ride on include_raman's h_R/f_R — fail at init,
    not with a mid-propagation assert (layer-1/layer-2 contract)."""
    grid = _grid()
    with pytest.raises(ValueError, match="include_raman"):
        MultimodeSplitStepEngine(
            [_sech_wave(grid, 4.0), _sech_wave(grid, 1.0)],
            Fiber(
                n2=2.6e-20,
                alpha=0.0,
                A_eff=Area(80e-12, "m^2"),
                length=Length(1.0, "m"),
            ),
            [np.array([-21.8]), np.array([-21.8])],
            raman_tensors=np.zeros((2, 2, 2, 2)),
        )


def test_raman_conserves_energy():
    """Raman run drifts within integrator tolerance (task 2.3)."""
    grid = _grid()
    resp = _resp(grid)
    fiber = Fiber(
        n2=2.6e-20,
        alpha=0.0,
        A_eff=Area(80e-12, "m^2"),
        length=Length(5.0, "m"),
        raman_response=resp,
    )
    mm = MultimodeSplitStepEngine(
        [_sech_wave(grid, 4.0, 5.0), _sech_wave(grid, 1.0, 5.0)],
        fiber,
        [np.array([-21.8]), np.array([-21.8])],
        include_raman=True,
        raman_response=resp,
        step_size=Length(0.1, "m"),
    )
    mm.propagate(10)
    e0 = sum(float(np.sum(np.abs(a) ** 2)) for a in mm.fields_vs_z())
    # fields_vs_z returns per-channel trajectories; compare first vs last energy
    trajs = mm.fields_vs_z()
    e_start = sum(float(np.sum(np.abs(t[0]) ** 2)) for t in trajs)
    e_end = sum(float(np.sum(np.abs(t[-1]) ** 2)) for t in trajs)
    assert abs(e_end - e_start) / e_start < 0.05
    assert e0 > 0


def test_intermodal_transfer_and_forbidden_control():
    """GV-matched pair with T_R transfers; zero-entry control does not."""
    grid = _grid(40.0, 2**10)
    resp = _resp(grid)
    fiber = Fiber(
        n2=2.6e-20,
        alpha=0.0,
        A_eff=Area(80e-12, "m^2"),
        length=Length(2.0, "m"),
        raman_response=resp,
    )
    betas = [np.array([-21.8]), np.array([-21.8])]

    def cw(peak):
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

    # Allowed: T_R[1,0,0,0] = 1 (pump 0 drives mode 1 through |A0|² memory).
    T = np.zeros((2, 2, 2, 2), dtype=complex)
    T[1, 0, 0, 0] = 1.0
    mm = MultimodeSplitStepEngine(
        [cw(1000.0), cw(1.0)],
        fiber,
        betas,
        include_raman=True,
        raman_response=resp,
        raman_tensors=T,
        step_size=Length(0.02, "m"),
    )
    mm.propagate(40)
    trajs = mm.fields_vs_z()
    p1_start = float(np.mean(np.abs(trajs[1][0]) ** 2))
    p1_end = float(np.mean(np.abs(trajs[1][-1]) ** 2))
    assert p1_end > p1_start * 1.01

    # Forbidden control: all-zero tensor (Antikainen selection-rule analogue).
    T0 = np.zeros((2, 2, 2, 2), dtype=complex)
    mm0 = MultimodeSplitStepEngine(
        [cw(100.0), cw(1.0)],
        fiber,
        betas,
        include_raman=True,
        raman_response=resp,
        raman_tensors=T0,
        step_size=Length(0.02, "m"),
    )
    mm0.propagate(20)
    trajs0 = mm0.fields_vs_z()
    q1_start = float(np.mean(np.abs(trajs0[1][0]) ** 2))
    q1_end = float(np.mean(np.abs(trajs0[1][-1]) ** 2))
    assert abs(q1_end - q1_start) / q1_start < 1e-6
