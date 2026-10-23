"""Per-OAM-class power accounting in the multimode engine (GAP A4)."""

from __future__ import annotations

import numpy as np
import pytest

from photonics_helper.base import Area, Length, Time, Wavelength
from photonics_helper.gnlse import Fiber
from photonics_helper.multimode_gnlse import MultimodeSplitStepEngine
from photonics_helper.pulse import Envelope, TemporalGrid, Wave

N = 1024
TMAX_PS = 40.0
LAM_NM = 532.0
P0_W = 1500.0
SEED = 1e-2


def _waves():
    grid = TemporalGrid(N=N, Tmax=Time(TMAX_PS * 1e-12, "s"))
    a = float(np.sqrt(P0_W))
    waves = []
    for ch in range(3):
        wave = Wave(
            grid=grid,
            envelope=Envelope(
                shape="gaussian",
                peak_amplitude=a,
                pulse_width=Time(10.0e-12, "s"),
            ),
            central_wavelength=Wavelength(LAM_NM, "nm"),
        )
        if ch == 0:
            field = np.full(N, a, dtype=complex)
        elif ch == 1:
            field = SEED * a * np.exp(1j * 2.0 * np.pi * 0.15e12 * grid.t)
        else:
            field = SEED * a * np.conj(np.exp(1j * 2.0 * np.pi * 0.15e12 * grid.t))
        waves.append(wave.with_field(field))
    return waves


def _engine(oam_l, *, include_fwm=False):
    fiber = Fiber(
        n2=2.6e-20, alpha=0.0, A_eff=Area(50e-12, "m^2"), length=Length(1.0, "m")
    )
    return MultimodeSplitStepEngine(
        _waves(),
        fiber,
        betas=[[0.0, 0.0, 0.0, 0.0]] * 3,
        oam_l=oam_l,
        include_fwm=include_fwm,
        fwm_pump_depletion=include_fwm,
        step_size=Length(5e-3, "m"),
    )


def test_power_by_l_sums_channels_in_one_class():
    eng = _engine([0, 1, 1])
    p = eng.power_by_l()
    per_channel = [float(np.sum(np.abs(A) ** 2)) * eng.grid.dt for A in eng.A]
    assert set(p) == {0, 1}
    assert p[0] == pytest.approx(per_channel[0], rel=1e-12)
    assert p[1] == pytest.approx(per_channel[1] + per_channel[2], rel=1e-12)


def test_missing_labels_raise():
    eng = _engine(None)
    with pytest.raises(ValueError, match="oam_l"):
        eng.power_by_l()


def test_class_power_is_flat_without_fwm():
    eng = _engine([0, 1, 2])
    eng.propagate(num_steps=20, nsaves=5)
    hist = eng.power_by_l_vs_z
    assert hist is not None
    z, labels, powers = hist
    assert labels == [0, 1, 2]
    assert powers.shape == (5, 3)
    assert z.shape == (5,)
    # no exchange: each channel's power is conserved, so each class is too
    assert np.allclose(powers, powers[0], rtol=1e-9, atol=1e-18)


def test_allowed_fwm_exchange_changes_the_class_budget():
    # FWM gate: l_m = l_n + l_p - l_q.  With l = [0, 1, 2] the triple
    # (m, n, p, q) = (2, 1, 1, 0) is allowed (2 = 1 + 1 - 0), so classes 1
    # and 2 exchange power.  Class 0 is a bystander.
    eng = _engine([0, 1, 2], include_fwm=True)
    eng.propagate(num_steps=200, nsaves=5)
    hist = eng.power_by_l_vs_z
    assert hist is not None
    _z, _labels, powers = hist
    assert np.ptp(powers[:, 2]) > 1e-3 * powers[0, 2], (
        "class-changing FWM must show in the budget"
    )
    # the exchange is between classes 1 and 2, not a total-energy change
    assert np.allclose(powers.sum(axis=1), powers.sum(axis=1)[0], rtol=1e-9)
