"""Regression tests for the multimode engine's group-delays arm.

Locks the walk-off audit finding (2026-09-30, ISSUES.md #8 addendum):
after the post-#0 convention swap the multimode engine's ``group_delays``
arm delays a channel whose group velocity is lower by exactly
``Delta beta1 * L`` in the channel-0 frame (envelope centroid in the
time domain, read via analysis-kernel-safe bookkeeping).

Methodology caveats baked in (do not regress these either):
- the probe envelope MUST be centered at t = 0 (the window center);
  seam-centered (t = T/2) envelopes bias every time-shift readout by
  wrap spill across the circular boundary,
- the group delay acts on the envelope centroid; with a seam-centered
  envelope the measured "shift" is window-dependent and can even invert
  sign, which is what v1's audit measured before its own probes were
  corrected.
"""
import numpy as np
import pytest

from photonics_helper.base import Area, Length, Time, Wavelength
from photonics_helper.gnlse import FiberProfile
from photonics_helper.multimode_gnlse import MultimodeSplitStepEngine
from photonics_helper.pulse import Envelope, TemporalGrid, Wave

BETA2_PS2_PER_M = 21.7e-3   # ps^2/m
BETA3_PS3_PER_M = -89.5e-6  # ps^3/m (mode-0 values, ps^k/m units)


def _centroid(f, t):
    return float(np.sum(t * np.abs(f)) / np.sum(np.abs(f)))


def test_multimode_group_delays_centroid_walkoff():
    """A slower channel (positive GVM) drifts LATER by GVM * L — exact."""
    n, T_ps = 8192, 400.0
    grid = TemporalGrid(N=n, Tmax=Time(T_ps * 1e-12, "s"))
    tt = np.asarray(grid.t.as_s if hasattr(grid.t, "as_s") else grid.t)
    envelope = np.exp(-(tt / (20e-12)) ** 2)  # centered at t=0, NOT the seam

    waves = []
    for m in range(4):
        wv = Wave(
            grid=grid,
            envelope=Envelope(shape="gaussian", peak_amplitude=1.0,
                              pulse_width=Time(1.0, "s")),
            central_wavelength=Wavelength(1064.0, "nm"),
        )
        waves.append(wv.with_field(np.asarray(1e-3 * envelope, complex)))

    fiber = FiberProfile(n2=1e-9, alpha=0.0, A_eff=Area(1.0, "m^2"),
                         length=Length(0.5, "m"))
    eng = MultimodeSplitStepEngine(
        waves, fiber,
        betas=[[BETA2_PS2_PER_M, BETA3_PS3_PER_M]] * 4,
        betas_unit="ps^k/m",
        coef_model="isotropic", include_fwm=False,
        step_size=Length(1e-3, "m"),
        group_delays=[0.0, 10.8e-12, 0.0, 0.0],
    )

    L = 0.5
    ref = eng._linear_step(np.asarray(1e-3 * envelope, complex), L, 0)
    delayed = eng._linear_step(np.asarray(1e-3 * envelope, complex), L, 1)

    delta = (_centroid(delayed, tt) - _centroid(ref, tt)) * 1e12
    assert delta == pytest.approx(10.8e-12 * L * 1e12, abs=1e-6), delta


def test_multimode_group_delays_sign_flips_with_gvm():
    n, T_ps = 8192, 400.0
    grid = TemporalGrid(N=n, Tmax=Time(T_ps * 1e-12, "s"))
    tt = np.asarray(grid.t.as_s if hasattr(grid.t, "as_s") else grid.t)
    envelope = np.exp(-(tt / (20e-12)) ** 2)
    waves = []
    for m in range(4):
        wv = Wave(
            grid=grid,
            envelope=Envelope(shape="gaussian", peak_amplitude=1.0,
                              pulse_width=Time(1.0, "s")),
            central_wavelength=Wavelength(1064.0, "nm"),
        )
        waves.append(wv.with_field(np.asarray(1e-3 * envelope, complex)))
    fiber = FiberProfile(n2=1e-9, alpha=0.0, A_eff=Area(1.0, "m^2"),
                         length=Length(0.5, "m"))
    eng_neg = MultimodeSplitStepEngine(
        waves, fiber,
        betas=[[BETA2_PS2_PER_M, BETA3_PS3_PER_M]] * 4,
        betas_unit="ps^k/m", coef_model="isotropic", include_fwm=False,
        step_size=Length(1e-3, "m"),
        group_delays=[0.0, -10.8e-12, 0.0, 0.0])

    L = 0.5
    ref = eng_neg._linear_step(np.asarray(1e-3 * envelope, complex), L, 0)
    d = eng_neg._linear_step(
        np.asarray(1e-3 * envelope, complex), L, 1)
    delta = (_centroid(d, tt) - _centroid(ref, tt)) * 1e12
    assert delta == pytest.approx(-10.8e-12 * L * 1e12, abs=1e-6), delta
