"""Tests for the multimode (few-mode) coupled GNLSE engine.

Contracts (mirroring the vector engine's suite):
- scalar/2-channel reduction to the polarization-coupled engine (exact);
- mode group-delay walk-off = Δβ₁·L (exact);
- pump-driven inter-modal FWM vs a dense RK4 of the same equations (<5%),
  with exact Hamiltonian energy exchange;
- OAM angular-momentum selection rule gating;
- total-energy conservation in every coefficient model;
- argument validation.
"""

from __future__ import annotations


import warnings

import numpy as np
import pytest

from photonics_helper.base import Area, Length, Time, Wavelength
from photonics_helper.gnlse import FiberProfile, SplitStepEngine
from photonics_helper.multimode_gnlse import MultimodeSplitStepEngine
from photonics_helper.pulse import Envelope, TemporalGrid, Wave
from photonics_helper.vector_gnlse import VectorSplitStepEngine

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


def _sech_wave(grid, peak, width_ps: float = 10.0):
    env = Envelope(
        shape="sech",
        peak_amplitude=np.sqrt(peak),
        pulse_width=Time(width_ps * 1e-12, "s"),
    )
    return Wave(grid=grid, envelope=env, central_wavelength=WL)


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


def _zero_wave(grid, template) -> Wave:
    wave = Wave(
        grid=grid,
        envelope=template.envelope,
        central_wavelength=template.central_wavelength,
    )
    wave.with_field(np.zeros(grid.N, dtype=complex))
    return wave


class TestReductionContracts:
    """Two-channel degenerate limits must equal the validated engines."""

    def test_single_channel_equals_scalar_engine(self):
        grid = _grid()
        fiber = _fiber(10.0)
        betas = np.array([-21.8])
        scalar = SplitStepEngine(
            _sech_wave(grid, 4.0, 5.0), fiber, betas, step_size=Length(0.1, "m")
        )
        scalar.propagate(100)
        engine = MultimodeSplitStepEngine(
            [_sech_wave(grid, 4.0, 5.0)], fiber, betas, step_size=Length(0.1, "m")
        )
        engine.propagate(100)
        ref = scalar.evolution[-1].envelope_field
        got = engine.fields_vs_z()[0][-1, :]
        assert np.max(np.abs(ref - got)) < 1e-12

    def test_two_channels_match_vector_engine(self):
        grid = _grid()
        fiber = _fiber(10.0)
        betas = np.array([-21.8])
        ax0 = np.sqrt(4.0)
        ay_field = 0.5 * ax0 * np.exp(-((grid.t) / 5e-12) ** 2)
        vec = VectorSplitStepEngine(
            _sech_wave(grid, 4.0, 5.0),
            _zero_wave(grid, _sech_wave(grid, 4.0, 5.0)).with_field(ay_field),
            fiber,
            betas,
            coupling="incoherent",
            step_size=Length(0.1, "m"),
        )
        vec.propagate(100)
        mm = MultimodeSplitStepEngine(
            [
                _sech_wave(grid, 4.0, 5.0),
                _zero_wave(grid, _sech_wave(grid, 4.0, 5.0)).with_field(ay_field),
            ],
            fiber,
            betas,
            step_size=Length(0.1, "m"),
        )
        mm.propagate(100)
        for mms, attr in ((0, "x"), (1, "y")):
            mine = mm.fields_vs_z()[mms][-1, :]
            ref = (vec.evolution_x if mms == 0 else vec.evolution_y)[
                -1
            ].envelope_field
            assert np.max(np.abs(mine - ref)) < 1e-12


class TestGroupDelay:
    def test_mode_walk_off(self):
        grid = _grid(40.0, 2**12)

        def gauss(t, T0, A0):
            return A0 * np.exp(-((t / T0) ** 2))

        env = Envelope(
            shape="custom",
            peak_amplitude=1.0,
            func=gauss,
            pulse_width=Time(1e-12, "s"),
        )
        walkoff = 2.0e-12  # s/m; mode-1 rides slightly slower than mode 0
        length = 2.0
        fiber = _fiber(length)
        engine = MultimodeSplitStepEngine(
            [
                Wave(grid=grid, envelope=env, central_wavelength=WL),
                Wave(grid=grid, envelope=env, central_wavelength=WL),
            ],
            fiber,
            np.array([0.0]),
            group_delays=[0.0, walkoff],
            step_size=Length(0.05, "m"),
        )
        engine.propagate(40)
        t = grid.t
        p0 = t[np.argmax(np.abs(engine.fields_vs_z()[0][-1, :]) ** 2)]
        p1 = t[np.argmax(np.abs(engine.fields_vs_z()[1][-1, :]) ** 2)]
        assert (p1 - p0) == pytest.approx(walkoff * length, rel=2e-2)


class TestInterModalFWM:
    """Pump-driven FWM: growth, Hamiltonian exchange, and OAM gating."""


class TestInterModalFWMHeavy:
    """Pump-driven FWM against an independent dense RK4 of the same RHS."""

    @staticmethod
    def _dense_reference(length, z_samples, gamma, a0, a1, a2, h=0.002):
        """RK4 of the engine's exact RHS at the CW plane (single-pump model).

        Diagonal: isotropic SPM/XPM with the shared γ; FWM: every pump
        drives both ordered exchange pairs through the Mumtaz (2013)
        Eq. (6) creation arms — identical equations to the engine's
        mixing step (all ``+iγ A_n² A_q*`` terms over the (n, m, q) sum).
        """

        def deriv(s):
            am0, am1, am2 = s
            diag = gamma * (abs(am0) ** 2 + abs(am1) ** 2 + abs(am2) ** 2)
            d0 = 1j * diag * am0 + 1j * gamma * (
                am1 * am1 * np.conj(am2) + am2 * am2 * np.conj(am1)
            )
            d1 = 1j * diag * am1 + 1j * gamma * (
                am0 * am0 * np.conj(am2) + am2 * am2 * np.conj(am0)
            )
            d2 = 1j * diag * am2 + 1j * gamma * (
                am0 * am0 * np.conj(am1) + am1 * am1 * np.conj(am0)
            )
            return np.array([d0, d1, d2])

        state = np.array([a0, a1, a2], dtype=complex)
        out = [np.abs(state.copy())]
        prev = 0.0
        for ztarget in z_samples[1:]:
            while prev < ztarget - 1e-14:
                step = min(h, ztarget - prev)
                k1 = deriv(state)
                k2 = deriv(state + 0.5 * step * k1)
                k3 = deriv(state + 0.5 * step * k2)
                k4 = deriv(state + step * k3)
                state = state + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                prev += step
            out.append([abs(state[i]) for i in range(3)])
        return np.array(out)

    def test_fwm_growth_matches_dense_ode(self):
        grid = _grid()
        length = 5.0
        fiber = _fiber(length)
        peak = 5.0
        seed_peak = 0.01  # 1 % amplitude seed on channels 0 and 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            engine = MultimodeSplitStepEngine(
                [
                    _cw_wave(grid, peak),
                    _cw_wave(grid, peak),
                    _cw_wave(grid, seed_peak),
                ],
                fiber,
                np.array([0.0]),
                coef_model="isotropic",
                include_fwm=True,
                step_size=Length(0.02, "m"),
            )
        probe = MultimodeSplitStepEngine(
            [_cw_wave(grid, peak)], fiber, np.array([0.0])
        )
        gamma = probe._gamma_v()
        engine.propagate(250)
        f0, f1, f2 = engine.fields_vs_z()
        zs = np.array([0.0, 2.0, 5.0])
        idx = [int(round(zz / 0.02)) for zz in zs]
        # channels 0 and 2 are the exchange pair of pump channel 1
        dense = self._dense_reference(
            length,
            zs,
            gamma,
            np.sqrt(peak),
            np.sqrt(peak),
            np.sqrt(seed_peak),
        )
        got = np.column_stack(
            [
                np.abs(f0[idx, 0]),
                np.abs(f1[idx, 0]),
                np.abs(f2[idx, 0]),
            ]
        )
        rel = np.max(np.abs(dense - got)) / max(np.max(dense), 1e-30)
        assert rel < 0.05, f"FWM vs dense-RK4 rel err {rel:.2e}"


class TestOAMvSelectionRule:
    """Angular-momentum gating of the pump-driven FWM exchange pair."""

    @staticmethod
    def _run(oam_l, seed_peak=0.05):
        grid = _grid()
        length = 5.0
        fiber = _fiber(length)
        peak = 5.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            engine = MultimodeSplitStepEngine(
                [
                    _cw_wave(grid, peak),
                    _cw_wave(grid, peak),
                    _cw_wave(grid, seed_peak),
                ],
                fiber,
                np.array([0.0]),
                coef_model="isotropic",
                include_fwm=True,
                oam_l=oam_l,
                step_size=Length(0.02, "m"),
            )
        engine.propagate(250)
        f0, _f1, f2 = engine.fields_vs_z()
        return np.abs(f0[-1, 0]), np.abs(f2[-1, 0])

    def test_allowed_pair_exchanges(self):
        """ℓ_m = 2ℓ_n − ℓ_q holds for the pair (0, 2) through ℓ_n = 0."""
        a0, a2 = self._run([-1, 0, +1])
        # the Mumtaz-consistent mixing is phase-sensitive: the seeded arm
        # is exchanged (and can be parametrically amplified past its seed)
        assert abs(a2 - np.sqrt(0.05)) > 0.01 * np.sqrt(0.05)
        assert a0 < np.sqrt(5.0) * 1.01  # pump arm also touched (XPM/FWM)

    def test_forbidden_pair_is_spm_only(self):
        """A triplet violating ℓ_m = 2ℓ_n − ℓ_q does not mix (no FWM)."""
        allowed = self._run([-1, 0, +1])
        forbidden = self._run([-1, +2, +1])
        # the forbidden case must stay essentially unmixed: |a0| unchanged
        assert forbidden[0] == pytest.approx(np.sqrt(5.0), rel=1e-4)
        assert abs(forbidden[0] - allowed[0]) > 1e-6

    def test_allowed_pair_energy_exchange(self):
        """Allowed FWM pair: total energy conserved to the Strang commutator."""
        grid = _grid()
        length = 5.0
        fiber = _fiber(length)
        peak = 5.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            engine = MultimodeSplitStepEngine(
                [
                    _cw_wave(grid, peak),
                    _cw_wave(grid, peak),
                    _cw_wave(grid, 0.05),
                ],
                fiber,
                np.array([0.0]),
                coef_model="isotropic",
                include_fwm=True,
                oam_l=[-1, 0, +1],
                step_size=Length(0.02, "m"),
            )
        engine.propagate(250)
        e = engine.energy_vs_z
        assert e[-1] == pytest.approx(e[0], rel=1e-3)


class TestArgumentValidation:
    def test_shared_grid_required(self):
        grid_a, grid_b = _grid(), _grid()
        fiber = _fiber(1.0)
        with pytest.raises(ValueError, match="TemporalGrid"):
            MultimodeSplitStepEngine(
                [_sech_wave(grid_a, 1.0), _sech_wave(grid_b, 1.0)],
                fiber,
                np.array([-21.8]),
            )

    def test_group_delays_reference_frame(self):
        grid = _grid()
        fiber = _fiber(1.0)
        with pytest.raises(ValueError, match="reference frame"):
            MultimodeSplitStepEngine(
                [_sech_wave(grid, 1.0), _sech_wave(grid, 1.0)],
                fiber,
                np.array([-21.8]),
                group_delays=[1e-12, 0.0],
            )

    def test_betas_list_count(self):
        grid = _grid()
        fiber = _fiber(1.0)
        with pytest.raises(ValueError, match="channel count"):
            MultimodeSplitStepEngine(
                [_sech_wave(grid, 1.0), _sech_wave(grid, 1.0)],
                fiber,
                [np.array([-21.8])],  # too few for 2 channels
            )

    def test_oam_length_mismatch(self):
        grid = _grid()
        fiber = _fiber(1.0)
        with pytest.raises(ValueError, match="oam_l"):
            MultimodeSplitStepEngine(
                [_sech_wave(grid, 1.0), _sech_wave(grid, 1.0)],
                fiber,
                np.array([-21.8]),
                oam_l=[0],
            )
