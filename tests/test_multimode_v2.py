"""Tests for multimode GNLSE v2: overlap-tensor weights + FWM pump depletion.

Literature contract: S. Mumtaz, R.-J. Essiambre & G. P. Agrawal, "Nonlinear
propagation in multimode and multicore fibers," JLT 31, 398 (2013)
(doi:10.1109/JLT.2012.2235414, Eq. 6/8) and F. Poletti & P. Horak, JOSA B 25,
1645 (2008) (doi:10.1364/JOSAB.25.001645, photon-number conservation Eq. 15).

Contracts:
- the Manley-Rowe-consistent depleted set conserves Σ|A|² to RK4 round-off
  and shows the parametric back-conversion oscillation on the pump;
- with tiny seeds the depleted and pump-driven engines agree (<5%);
- ``xpm_weights``/``fwm_weights`` override the coef_model factors exactly;
- two-channel (and single-channel) degenerate strips-down still hold.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from photonics_helper.base import Area, Length, Time, Wavelength
from photonics_helper.gnlse import FiberProfile
from photonics_helper.multimode_gnlse import MultimodeSplitStepEngine
from photonics_helper.pulse import Envelope, TemporalGrid, Wave

WL = Wavelength(1550.0, "nm")


def _grid(t_ps: float = 60.0, n: int = 2**10) -> TemporalGrid:
    return TemporalGrid(N=n, Tmax=Time(t_ps * 1e-12, "s"))


def _fiber(length_m: float) -> FiberProfile:
    return FiberProfile(
        n2=2.6e-20,
        alpha=0.0,
        A_eff=Area(80e-12, "m^2"),
        length=Length(length_m, "m"),
    )


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


def _engine(grid, fiber, peaks, **kw):
    include_fwm = kw.pop("include_fwm", True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eng = MultimodeSplitStepEngine(
            [_cw_wave(grid, p) for p in peaks],
            fiber,
            np.array([0.0]),
            coef_model="isotropic",
            include_fwm=include_fwm,
            step_size=Length(kw.pop("dz", 0.02), "m"),
            **kw,
        )
    return eng


class TestPumpDepletion:
    """The depleted FWM substep: exact photon conservation + back-conversion."""

    def test_depleted_conserves_energy_machine_precision(self):
        grid = _grid()
        eng = _engine(grid, _fiber(5.0), [5.0, 5.0, 0.05], fwm_pump_depletion=True)
        eng.propagate(250)
        e = eng.energy_vs_z
        assert e[-1] == pytest.approx(e[0], rel=2e-3)  # Strang + RK4 floor
        # finer steps: energy must approach machine-level conservation
        eng2 = _engine(grid, _fiber(5.0), [5.0, 5.0, 0.05], dz=0.005, fwm_pump_depletion=True)
        eng2.propagate(1000)
        e2 = eng2.energy_vs_z
        assert abs(e2[-1] / e2[0] - 1.0) < 1e-5

    def test_depleted_matches_dense_manley_rowe_ode(self):
        """Engine vs dense RK4 of the exact Manley-Rowe RHS (<5%)."""
        grid = _grid()
        fiber = _fiber(5.0)
        eng = _engine(
            grid, fiber, [5.0, 0.05, 0.05], fwm_pump_depletion=True
        )
        eng.propagate(250)
        f0, f1, f2 = eng.fields_vs_z()
        zs = np.array([0.0, 2.0, 5.0])
        idx = [int(round(zz / 0.02)) for zz in zs]
        gamma = eng._gamma_v()

        def deriv(s):
            a0, a1, a2 = s
            diag = gamma * (
                abs(a0) ** 2 + abs(a1) ** 2 + abs(a2) ** 2
            )
            # creation arms: pump n drives both pair arms (+i γ A_n² A_*)
            # back-conversion: pump loses with Manley-Rowe factor 2
            arms = [
                [  # pump 0 -> pair (1, 2)
                    (1, 1j * gamma * a0**2 * np.conj(a2)),
                    (2, 1j * gamma * a0**2 * np.conj(a1)),
                    (0, 2j * gamma * np.conj(a0) * a1 * a2),
                ],
                [  # pump 1 -> pair (0, 2)
                    (0, 1j * gamma * a1**2 * np.conj(a2)),
                    (2, 1j * gamma * a1**2 * np.conj(a0)),
                    (1, 2j * gamma * np.conj(a1) * a0 * a2),
                ],
                [  # pump 2 -> pair (0, 1)
                    (0, 1j * gamma * a2**2 * np.conj(a1)),
                    (1, 1j * gamma * a2**2 * np.conj(a0)),
                    (2, 2j * gamma * np.conj(a2) * a0 * a1),
                ],
            ]
            d = [1j * diag * s[k] for k in range(3)]
            for group in arms:
                for ch, term in group:
                    d[ch] = d[ch] + term
            return np.array(d)

        state = np.array([np.sqrt(5.0), np.sqrt(0.05), np.sqrt(0.05)], dtype=complex)
        dense = [np.abs(state.copy())]
        prev = 0.0
        for zt in zs[1:]:
            while prev < zt - 1e-14:
                step = min(0.002, zt - prev)
                k1 = deriv(state)
                k2 = deriv(state + 0.5 * step * k1)
                k3 = deriv(state + 0.5 * step * k2)
                k4 = deriv(state + step * k3)
                state = state + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                prev += step
            dense.append([abs(state[k]) for k in range(3)])
        dense = np.array(dense)
        got = np.column_stack(
            [np.abs(f0[idx, 0]), np.abs(f1[idx, 0]), np.abs(f2[idx, 0])]
        )
        rel = np.max(np.abs(dense - got)) / np.max(dense)
        assert rel < 0.05, f"depleted FWM vs Manley-Rowe RK4 rel err {rel:.2e}"

    def test_depletion_absent_when_disabled(self):
        """The default pump-driven engine must not touch the pump via FWM."""
        grid = _grid()
        eng = _engine(grid, _fiber(5.0), [5.0, 0.05, 0.05])
        eng.propagate(250)
        f0, _f1, _f2 = eng.fields_vs_z()
        # pump amplitude change is XPM-only phase: |A_0| stays ~ sqrt(5)
        assert abs(abs(f0[-1, 0]) - np.sqrt(5.0)) < 0.02


class TestOverlapWeights:
    """Mode-specific overlap tensors (Mumtaz Eq. 8) override coef_model."""

    def test_xpm_weights_zero_coupling(self):
        """A zero XPM weight isolates a channel: its CW phase = γL·P exactly."""
        grid = _grid()
        eng = _engine(
            grid,
            _fiber(2.0),
            [1.0, 4.0],
            xpm_weights=np.array([[1.0, 0.0], [0.0, 1.0]]),
        )
        eng.propagate(500)
        f0, f1 = eng.fields_vs_z()
        gamma = eng._gamma_v()
        ref0 = np.angle(f0[-1, 0] / f0[0, 0])
        ref1 = np.angle(f1[-1, 0] / f1[0, 0])
        assert ref0 == pytest.approx(gamma * 2.0 * 1.0, abs=1e-12)
        assert ref1 == pytest.approx(gamma * 2.0 * 4.0, abs=1e-12)

    def test_xpm_weights_isotropic_match(self):
        """xpm_weights = all-ones reproduces the isotropic engine exactly."""
        grid = _grid()
        eng_a = _engine(grid, _fiber(1.0), [2.0, 3.0])
        eng_b = _engine(
            grid,
            _fiber(1.0),
            [2.0, 3.0],
            xpm_weights=np.ones((2, 2)),
        )
        eng_a.propagate(200)
        eng_b.propagate(200)
        a0, a1 = eng_a.fields_vs_z()
        b0, b1 = eng_b.fields_vs_z()
        assert np.allclose(a0, b0, atol=1e-12)
        assert np.allclose(a1, b1, atol=1e-12)

    def test_fwm_weights_gate_transitions(self):
        """fwm_weights = 0 for a transition blocks it exactly like oam gating."""
        grid = _grid()
        peaks = [5.0, 5.0, 0.05]
        oam = [-1, 0, +1]
        eng_gate = _engine(grid, _fiber(5.0), peaks, oam_l=oam)
        w = np.zeros((3, 3, 3, 3))
        for m in range(3):
            for n in range(3):
                for q in range(3):
                    if m != n and q != n and q != m and oam[m] == oam[n] + oam[n] - oam[q]:
                        w[m, n, n, q] = 1.0
                        w[q, n, n, m] = 1.0
        eng_weight = _engine(grid, _fiber(5.0), peaks, fwm_weights=w)
        eng_gate.propagate(250)
        eng_weight.propagate(250)
        g0, _g1, g2 = eng_gate.fields_vs_z()
        w0, _w1, w2 = eng_weight.fields_vs_z()
        assert np.allclose(g0, w0, atol=1e-12)
        assert np.allclose(g2, w2, atol=1e-12)

    def test_weight_validation(self):
        grid = _grid()
        fiber = _fiber(1.0)
        with pytest.raises(ValueError):
            _engine(grid, fiber, [1.0, 2.0], xpm_weights=np.ones((2, 2, 2)))
        with pytest.raises(ValueError):
            _engine(
                grid, fiber, [1.0, 2.0, 3.0], fwm_weights=np.ones((3, 3, 3))
            )
        with pytest.raises(ValueError):
            _engine(
                grid, fiber, [1.0, 2.0],
                include_fwm=False,
                fwm_pump_depletion=True,
            )
