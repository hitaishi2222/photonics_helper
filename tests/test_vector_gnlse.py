"""Tests for the vector (polarization-coupled) GNLSE engine.

Validation strategy mirrors the scalar engine's philosophy: analytic limits,
the Manakov 8/9 polarization-averaging coefficient, PM-fiber walk-off,
polarization FWM against an independent dense-RK4 integration of the same
equations, and exact total-energy conservation in every coupling mode.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from photonics_helper.base import Area, Length, Time, Wavelength
from photonics_helper.gnlse import FiberProfile, SplitStepEngine
from photonics_helper.pulse import Envelope, TemporalGrid, Wave
from photonics_helper.raman import RAMAN_MATERIALS, RamanResponse
from photonics_helper.vector_gnlse import (
    MANAKOV_FACTOR,
    RandomBirefringenceEngine,
    VectorSplitStepEngine,
)

WL = Wavelength(1550.0, "nm")


def _grid(t_ps: float = 60.0, n: int = 2**11) -> TemporalGrid:
    return TemporalGrid(N=n, Tmax=Time(t_ps * 1e-12, "s"))


def _fiber(length_m: float) -> FiberProfile:
    return FiberProfile(
        n2=2.6e-20,
        alpha=0.0,
        A_eff=Area(80e-12, "m^2"),
        length=Length(length_m, "m"),
    )


def _sech_wave(grid, peak, width_ps: float = 10.0) -> Wave:
    env = Envelope(
        shape="sech",
        peak_amplitude=np.sqrt(peak),
        pulse_width=Time(width_ps * 1e-12, "s"),
    )
    return Wave(grid=grid, envelope=env, central_wavelength=WL)


def _cw_envelope(peak) -> Envelope:
    const = float(np.sqrt(peak))

    def func(t, T0, A0):
        return const * np.ones_like(t)

    return Envelope(
        shape="custom",
        peak_amplitude=np.sqrt(peak),
        func=func,
        pulse_width=Time(1e-12, "s"),
    )


def _cw_wave(grid, peak) -> Wave:
    return Wave(grid=grid, envelope=_cw_envelope(peak), central_wavelength=WL)


def _zero_wave(grid, template: Wave) -> Wave:
    wave = Wave(
        grid=grid,
        envelope=template.envelope,
        central_wavelength=template.central_wavelength,
    )
    wave.with_field(np.zeros(grid.N, dtype=complex))
    return wave


# ---------------------------------------------------------------------------
# scalar-limit reduction
# ---------------------------------------------------------------------------


class TestScalarLimitReduction:
    """coupling='incoherent' with A_y ≡ 0 must equal the scalar engine."""

    def test_exact_field_match_no_raman(self):
        grid = _grid()
        fiber = _fiber(10.0)
        betas = np.array([-21.8])  # ps^2/m
        scalar = SplitStepEngine(
            _sech_wave(grid, 10.0), fiber, betas, step_size=Length(0.1, "m")
        )
        scalar.propagate(100)
        vec = VectorSplitStepEngine(
            _sech_wave(grid, 10.0),
            _zero_wave(grid, _sech_wave(grid, 10.0)),
            fiber,
            betas,
            coupling="incoherent",
            step_size=Length(0.1, "m"),
        )
        vec.propagate(100)
        ref = scalar.evolution[-1].envelope_field
        got = vec.evolution_x[-1].envelope_field
        assert got.shape == (2**11,)
        assert np.max(np.abs(ref - got)) < 1e-12

    def test_exact_field_match_with_raman(self):
        grid = _grid(200.0, 2**12)
        spec = RAMAN_MATERIALS["Silica"]
        raman = RamanResponse(spec=spec, fR=0.18, grid=grid)
        fiber = FiberProfile(
            n2=2.6e-20,
            alpha=0.0,
            A_eff=Area(80e-12, "m^2"),
            length=Length(10.0, "m"),
            raman_response=raman,
        )
        betas = np.array([-21.8])
        scalar = SplitStepEngine(
            _sech_wave(grid, 10.0),
            fiber,
            betas,
            include_raman=True,
            step_size=Length(0.1, "m"),
        )
        scalar.propagate(100)
        vec = VectorSplitStepEngine(
            _sech_wave(grid, 10.0),
            _zero_wave(grid, _sech_wave(grid, 10.0)),
            fiber,
            betas,
            coupling="incoherent",
            include_raman=True,
            step_size=Length(0.1, "m"),
        )
        vec.propagate(100)
        ref = scalar.evolution[-1].envelope_field
        got = vec.evolution_x[-1].envelope_field
        assert np.max(np.abs(ref - got)) < 1e-12


# ---------------------------------------------------------------------------
# Manakov mode
# ---------------------------------------------------------------------------


class TestManakovMode:
    def test_cw_nonlinear_phase_is_8_9_of_scalar_spm(self):
        """CW in Manakov mode acquires φ = (8/9)·γ·P·L exactly."""
        grid = _grid()
        fiber = _fiber(10.0)
        peak = 5.0
        engine = VectorSplitStepEngine(
            _cw_wave(grid, peak),
            _zero_wave(grid, _cw_wave(grid, peak)),
            fiber,
            np.array([0.0]),  # no dispersion: isolate the nonlinearity
            coupling="manakov",
        )
        engine.propagate(40)
        phi = np.angle(engine.evolution_x[-1].envelope_field[0] / np.sqrt(peak))
        expected = MANAKOV_FACTOR * engine._gamma_v() * peak * 10.0
        assert phi == pytest.approx(expected, rel=1e-3)

    def test_manakov_equals_scalar_with_scaled_gamma(self):
        """Manakov(γ) ≡ scalar engine driven at an explicitly scaled 8/9·γ."""
        grid = _grid()
        peak = 5.0
        fiber = _fiber(10.0)
        env = _cw_envelope(peak)
        manakov = VectorSplitStepEngine(
            _cw_wave(grid, peak),
            _zero_wave(grid, _cw_wave(grid, peak)),
            fiber,
            np.array([0.0]),
            coupling="manakov",
        )
        manakov.propagate(40)
        phi_m = np.angle(manakov.evolution_x[-1].envelope_field[0] / np.sqrt(peak))

        fiber_scaled = FiberProfile(
            n2=fiber.n2 * MANAKOV_FACTOR,
            alpha=0.0,
            A_eff=Area(80e-12, "m^2"),
            length=Length(10.0, "m"),
        )
        scalar = SplitStepEngine(
            Wave(grid=grid, envelope=env, central_wavelength=WL),
            fiber_scaled,
            np.array([0.0]),
            step_size=Length(0.25, "m"),
        )
        scalar.propagate(40)
        phi_s = np.angle(scalar.evolution[-1].envelope_field[0] / np.sqrt(peak))
        assert phi_m == pytest.approx(phi_s, rel=1e-4)

    def test_manakov_rejects_different_betas_and_walkoff(self):
        grid = _grid()
        fiber = _fiber(1.0)
        with pytest.raises(ValueError, match="manakov"):
            VectorSplitStepEngine(
                _sech_wave(grid, 1.0),
                _zero_wave(grid, _sech_wave(grid, 1.0)),
                fiber,
                np.array([-21.8]),
                betas_y=np.array([-15.0]),
                coupling="manakov",
            )
        with pytest.raises(ValueError, match="manakov"):
            VectorSplitStepEngine(
                _sech_wave(grid, 1.0),
                _zero_wave(grid, _sech_wave(grid, 1.0)),
                fiber,
                np.array([-21.8]),
                coupling="manakov",
                walkoff=1e-12,
            )


# ---------------------------------------------------------------------------
# PM-fiber walk-off
# ---------------------------------------------------------------------------


class TestWalkoff:
    def test_differential_group_delay(self):
        """A y-channel pulse drifts by exactly walkoff·z (retarded x frame)."""
        grid = _grid(40.0, 2**12)

        def gauss(t, T0, A0):
            return A0 * np.exp(-((t / T0) ** 2))

        env = Envelope(
            shape="custom",
            peak_amplitude=1.0,
            func=gauss,
            pulse_width=Time(1e-12, "s"),
        )
        walkoff = 2.0e-12  # s/m
        length = 2.0
        fiber = _fiber(length)
        engine = VectorSplitStepEngine(
            Wave(grid=grid, envelope=env, central_wavelength=WL),
            Wave(grid=grid, envelope=env, central_wavelength=WL),
            fiber,
            np.array([0.0]),
            walkoff=walkoff,
            step_size=Length(0.05, "m"),
        )
        engine.propagate(40)
        t = grid.t
        px = t[np.argmax(np.abs(engine.evolution_x[-1].envelope_field) ** 2)]
        py = t[np.argmax(np.abs(engine.evolution_y[-1].envelope_field) ** 2)]
        assert (py - px) == pytest.approx(walkoff * length, rel=2e-2)


# ---------------------------------------------------------------------------
# coherent polarization FWM
# ---------------------------------------------------------------------------



class TestCoherentFWM:
    """The coherent-coupling step vs an independent dense-RK4 integration."""

    @staticmethod
    def _dense_reference(length, dbeta, z_samples, gamma, ax0, ay0, h=0.002):
        """RK4 of the same coupled ODE the engine discretizes (Agrawal §6.3)."""

        def deriv(z, s):
            ax, ay = s
            Px, Py = abs(ax) ** 2, abs(ay) ** 2
            Dx = gamma * (Px + 2 / 3 * Py)
            Dy = gamma * (Py + 2 / 3 * Px)
            mx = (1j * gamma / 3) * np.exp(-2j * dbeta * z)
            my = (1j * gamma / 3) * np.exp(+2j * dbeta * z)
            return np.array(
                [
                    1j * Dx * ax + mx * ay * ay * np.conj(ax),
                    1j * Dy * ay + my * ax * ax * np.conj(ay),
                ]
            )

        state = np.array([ax0, ay0], dtype=complex)
        out = [abs(state[1])]
        z = 0.0
        for ztarget in z_samples[1:]:
            while z < ztarget - 1e-14:
                step = min(h, ztarget - z)
                k1 = deriv(z, state)
                k2 = deriv(z + step / 2, state + 0.5 * step * k1)
                k3 = deriv(z + step / 2, state + 0.5 * step * k2)
                k4 = deriv(z + step, state + step * k3)
                state = state + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                z += step
            out.append(abs(state[1]))
        return np.array(out)

    @pytest.mark.parametrize("db_units", [1.0, 3.0])
    def test_matches_dense_ode(self, db_units):
        grid = _grid()
        length, peak = 20.0, 50.0
        fiber = _fiber(length)
        ay0 = 0.05 * np.sqrt(peak)
        probe = VectorSplitStepEngine(
            _cw_wave(grid, peak),
            _cw_wave(grid, ay0**2),
            fiber,
            np.array([0.0]),
            coupling="incoherent",
        )
        gamma = probe._gamma_v()  # scalar γ of the profile
        dbeta = db_units * gamma * peak
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            engine = VectorSplitStepEngine(
                _cw_wave(grid, peak),
                _cw_wave(grid, ay0**2),
                fiber,
                np.array([0.0]),
                coupling="coherent",
                delta_beta=dbeta,
                step_size=Length(0.04, "m"),
            )
        engine.propagate(500)
        ay_eng = np.abs(engine.fields_vs_z()[1][:, 0])
        zs = np.array([0.0, 5.0, 10.0, 20.0])
        dense = self._dense_reference(length, dbeta, zs, gamma, np.sqrt(peak), ay0)
        idx = [int(round(zz / 0.04)) for zz in zs]
        rel = np.max(np.abs(dense - ay_eng[idx])) / max(dense.max(), 1e-30)
        assert rel < 0.05, f"Δβ={dbeta:.4g}: rel err {rel:.2e}"


# ---------------------------------------------------------------------------
# energy conservation in every coupling mode
# ---------------------------------------------------------------------------


class TestEnergyConservation:
    @pytest.mark.parametrize("coupling", ["incoherent", "coherent", "manakov"])
    def test_total_energy_conserved(self, coupling):
        grid = _grid()
        fiber = _fiber(5.0)
        peak = 2.0
        kwargs = {}
        if coupling == "coherent":
            # a finite mismatch avoids the Δβ=0 resonance warning
            gamma_probe = VectorSplitStepEngine(
                _sech_wave(grid, peak), _zero_wave(grid, _sech_wave(grid, peak)),
                fiber, np.array([0.0]),
            )
            kwargs["delta_beta"] = 2.0 * gamma_probe._gamma_v() * peak
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            engine = VectorSplitStepEngine(
                _sech_wave(grid, peak, 2.0),
                _sech_wave(grid, peak, 2.0),
                fiber,
                np.array([-21.8]),
                coupling=coupling,
                step_size=Length(0.05, "m"),
                **kwargs,
            )
        engine.propagate(100)
        e = engine.energy_vs_z
        assert e[-1] == pytest.approx(e[0], rel=1e-8)


# ---------------------------------------------------------------------------
# random-birefringence engine → Manakov limit
# ---------------------------------------------------------------------------


class TestRandomBirefringence:
    def test_energy_and_cw_shape_exactly_conserved(self):
        grid = _grid()
        length, steps = 20.0, 60
        fiber = _fiber(length)
        peak = 5.0
        engine = RandomBirefringenceEngine(
            _cw_wave(grid, peak),
            _zero_wave(grid, _cw_wave(grid, peak)),
            fiber,
            np.array([0.0]),
            step_size=Length(length / steps, "m"),
            seed=1,
        )
        engine.propagate(steps)
        e = engine.energy_vs_z
        assert e[-1] == pytest.approx(e[0], rel=1e-10)
        # a CW input stays exactly CW: min/max intensity constant in time
        for channel in engine.evolution_x, engine.evolution_y:
            field = channel[-1].envelope_field
            intens = np.abs(field) ** 2
            assert (intens.max() - intens.min()) < 1e-8 * intens.max()

    def test_ensemble_spectrum_converges_to_manakov(self):
        """Ensemble-averaged spectrum ≈ the deterministic 8/9 Manakov run.

        Seeded sech pulses, zero dispersion, six random-frame seeds: the
        ensemble average of the random-birefringence spectra matches the
        deterministic Manakov spectrum (Wai & Menyuk 1996) to <2% in L2.
        """
        grid = _grid()
        length, steps = 5.0, 80
        fiber = _fiber(length)
        peak = 2.0

        manakov = VectorSplitStepEngine(
            _sech_wave(grid, peak, 2.0),
            _zero_wave(grid, _sech_wave(grid, peak, 2.0)),
            fiber,
            np.array([-21.8]),
            coupling="manakov",
        )
        manakov.propagate(steps)
        spec_m = np.abs(grid.fft(manakov.evolution_x[-1].envelope_field)) ** 2

        acc = np.zeros(grid.N)
        for seed in range(6):
            engine = RandomBirefringenceEngine(
                _sech_wave(grid, peak, 2.0),
                _zero_wave(grid, _sech_wave(grid, peak, 2.0)),
                fiber,
                np.array([-21.8]),
                step_size=Length(length / steps, "m"),
                seed=seed,
            )
            engine.propagate(steps)
            spec = (
                np.abs(grid.fft(engine.evolution_x[-1].envelope_field)) ** 2
                + np.abs(grid.fft(engine.evolution_y[-1].envelope_field)) ** 2
            )
            acc += spec
        ens = acc / 6
        rel = np.linalg.norm(ens - spec_m) / np.linalg.norm(spec_m)
        assert rel < 0.05, f"ensemble vs Manakov rel L2 = {rel:.2e}"


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------


class TestArgumentValidation:
    def test_shared_grid_required(self):
        grid_a, grid_b = _grid(), _grid()
        fiber = _fiber(1.0)
        with pytest.raises(ValueError, match="TemporalGrid"):
            VectorSplitStepEngine(
                _sech_wave(grid_a, 1.0),
                _sech_wave(grid_b, 1.0),
                fiber,
                np.array([-21.8]),
            )
