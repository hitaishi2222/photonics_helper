"""Literature-conformance regressions from the 2026-10 multimode audit.

Findings covered
----------------
1. ``mode_analysis.overlap_tensors_from_fields`` routed per-mode γ through
   ``coupling_tensor`` as ``½(γ_i+γ_j)·xpm``.  That destroys the per-mode SPM
   scale (``C_ii`` must be ``γ_i``) and averages a quantity that the overlap
   tensor already symmetrises.  The absolute coefficient is
   ``C_ij = γ₀·xpm_ij = (n₂ω₀/c)·Q⁽¹⁾_iijj``.
2. The engine's ``lp_degenerate`` model is the *two-polarization* vector
   strip-down (XPM 2/3).  Two **co-polarized** spatial modes have the
   Mumtaz Eq. (6) scalar limit XPM = 2, SPM = 1, FWM = 1 — added as
   ``coef_model="co_polarized"``.
3. ``build_multimode_deck`` now exposes ``gamma_reference = γ₀`` so the
   S₀₀₀₀-normalised FWM/Raman tensors keep their absolute scale even when
   the caller's ``Fiber.A_eff`` is not ``A_eff,0``.
4. ``_fwm_rate_max`` used ``γ·f`` and silently ignored an absolute
   ``fwm_coupling_tensor``; the RK4 substep count could therefore be
   under-estimated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from photonics_helper.base import Area, Length, Time, Wavelength
from photonics_helper.gnlse import Fiber, SplitStepEngine, _fiber_gamma
from photonics_helper.mode_analysis import (
    build_multimode_deck,
    load_modes,
    overlap_tensors_from_fields,
)
from photonics_helper.multimode_gnlse import MultimodeSplitStepEngine
from photonics_helper.pulse import Envelope, TemporalGrid, Wave

WL = Wavelength(1550.0, "nm")
EXPORT = Path("examples/data/multimode_rib")


def _grid(t_ps: float = 40.0, n: int = 2**10) -> TemporalGrid:
    return TemporalGrid(N=n, Tmax=Time(t_ps * 1e-12, "s"))


def _fiber(length_m: float, n2: float = 2.6e-20) -> Fiber:
    return Fiber(
        n2=n2,
        alpha=0.0,
        A_eff=Area(80e-12, "m^2"),
        length=Length(length_m, "m"),
    )


def _cw(grid, peak) -> Wave:
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


class TestModeAnalysisAbsoluteScale:
    """Finding 1: the coupling tensor must carry the per-mode γ."""

    def test_coupling_diagonal_is_per_mode_gamma(self):
        modes = load_modes(EXPORT)
        p = overlap_tensors_from_fields(modes, n2=1e-18, lambda0=2.0)
        c_diag = np.real(np.diag(p.coupling_tensor))
        assert np.allclose(c_diag, p.gamma, rtol=1e-12, atol=0.0)
        # The old ½(γ_i+γ_j) formula gave γ_0·(A_eff,0/A_eff,i)² — a 0.75×
        # error on the committed rib deck.  The correct diagonal is exactly
        # γ_0·A_eff,0/A_eff,i = γ_i (asserted above); check the old shape
        # is *not* what we now produce.
        assert not np.allclose(
            c_diag[1:],
            p.gamma[0] * (p.A_eff[0] / p.A_eff[1:]) ** 2,
        )

    def test_co_polarized_xpm_is_two_times_the_bare_overlap(self):
        modes = load_modes(EXPORT)
        co = overlap_tensors_from_fields(
            modes, n2=1e-18, lambda0=2.0, xpm_model="co_polarized"
        )
        vec = overlap_tensors_from_fields(
            modes, n2=1e-18, lambda0=2.0, xpm_model="vector"
        )
        n = len(modes)
        off = ~np.eye(n, dtype=bool)
        # SPM slot equal in both models, XPM doubled for co-polarized modes.
        assert np.allclose(co.xpm_weights[~off], vec.xpm_weights[~off])
        assert np.allclose(
            co.xpm_weights[off], 2.0 * vec.xpm_weights[off], rtol=1e-12
        )
        assert np.allclose(
            co.coupling_tensor[off].real,
            2.0 * vec.coupling_tensor[off].real,
            rtol=1e-12,
        )

    def test_invalid_xpm_model_rejected(self):
        modes = load_modes(EXPORT)
        with pytest.raises(ValueError, match="xpm_model"):
            overlap_tensors_from_fields(
                modes, n2=1e-18, lambda0=2.0, xpm_model="circular"
            )

    def test_deck_exposes_gamma_reference(self):
        deck = build_multimode_deck(EXPORT, 2.0, 4, 1e-18)
        p = overlap_tensors_from_fields(
            load_modes(EXPORT), n2=1e-18, lambda0=2.0
        )
        assert deck["gamma_reference"] == pytest.approx(float(p.gamma[0]))
        assert np.allclose(deck["gamma"], p.gamma)
        assert np.allclose(deck["A_eff"], p.A_eff)


class TestCoPolarizedModel:
    """Finding 2: SPM 1, XPM 2 for co-polarized scalar spatial modes."""

    @pytest.mark.parametrize(
        ("model", "xpm"),
        [("co_polarized", 2.0), ("isotropic", 1.0), ("lp_degenerate", 2.0 / 3.0)],
    )
    def test_cw_phase_rate(self, model, xpm):
        grid = _grid()
        p0, p1 = 1.0, 4.0
        eng = MultimodeSplitStepEngine(
            [_cw(grid, p0), _cw(grid, p1)],
            _fiber(2.0),
            [np.array([0.0]), np.array([0.0])],
            coef_model=model,
            step_size=Length(0.1, "m"),
        )
        eng.propagate(20)
        gamma = eng._gamma_v()
        f0, f1 = eng.fields_vs_z()
        phase0 = np.angle(f0[-1, 0] / f0[0, 0])
        phase1 = np.angle(f1[-1, 0] / f1[0, 0])
        assert phase0 == pytest.approx(gamma * (p0 + xpm * p1) * 2.0, abs=1e-12)
        assert phase1 == pytest.approx(gamma * (p1 + xpm * p0) * 2.0, abs=1e-12)

    def test_co_polarized_fwm_factor_is_one(self):
        grid = _grid()
        eng = MultimodeSplitStepEngine(
            [_cw(grid, 1.0), _cw(grid, 1.0)],
            _fiber(1.0),
            np.array([0.0]),
            coef_model="co_polarized",
        )
        assert eng._fwm_factor(0, 0, 0, 0) == 1.0

    def test_two_channel_vector_reduction_still_default(self):
        """The default must remain the vector-GNLSE strip-down (XPM 2/3)."""
        grid = _grid()
        eng = MultimodeSplitStepEngine(
            [_cw(grid, 1.0), _cw(grid, 1.0)],
            _fiber(1.0),
            np.array([0.0]),
        )
        assert eng.coef_model == "lp_degenerate"
        assert eng._xpm_factor(0, 1) == pytest.approx(2.0 / 3.0)


class TestTensorScaleAndStability:
    """Findings 3/4: gamma_reference and the absolute-tensor rate estimate."""

    def test_gamma_reference_matches_absolute_fwm_tensor(self):
        grid = _grid(20.0)
        length = 1.0
        gamma0 = 2.0  # W^-1 m^-1, deliberately not the fiber's γ
        rng = np.random.default_rng(7)
        w = rng.normal(size=(3, 3, 3, 3))
        w = 0.1 * (w + w.transpose(0, 2, 1, 3))  # symmetric pump pair
        waves = [_cw(grid, 4.0), _cw(grid, 0.25), _cw(grid, 0.05)]
        betas = [np.array([0.0])] * 3
        common = dict(
            include_fwm=True,
            fwm_full_tensor=True,
            oam_l=[0, 0, 0],
            step_size=Length(0.05, "m"),
        )
        a = MultimodeSplitStepEngine(
            list(waves),
            _fiber(length),
            betas,
            fwm_weights=w,
            gamma_reference=gamma0,
            **common,
        )
        b = MultimodeSplitStepEngine(
            list(waves),
            _fiber(length),
            betas,
            fwm_coupling_tensor=gamma0 * w,
            # The diagonal SPM/XPM arm is a separate scale from the absolute
            # FWM tensor; pin it to the same γ so the comparison isolates
            # the tensor-scaling contract.
            gamma_reference=gamma0,
            **common,
        )
        a.propagate(20)
        b.propagate(20)
        for m in range(3):
            np.testing.assert_allclose(
                a.fields_vs_z()[m][-1], b.fields_vs_z()[m][-1], atol=1e-12
            )

    def test_gamma_reference_off_keeps_fiber_gamma(self):
        eng = MultimodeSplitStepEngine(
            [_cw(_grid(), 1.0)],
            _fiber(1.0),
            np.array([0.0]),
        )
        assert eng.gamma_reference is None
        assert eng._gamma_v() == pytest.approx(_fiber_gamma(eng.fiber, eng.omega0))

    def test_fwm_rate_sees_absolute_coupling_tensor(self):
        grid = _grid(20.0, 2**8)
        c = np.zeros((3, 3, 3, 3), dtype=complex)
        c[1, 0, 0, 2] = c[2, 0, 0, 1] = 1.0e6
        eng = MultimodeSplitStepEngine(
            [_cw(grid, 4.0), _cw(grid, 1e-3), _cw(grid, 1e-3)],
            _fiber(1.0),
            [np.array([0.0])] * 3,
            include_fwm=True,
            fwm_coupling_tensor=c,
            oam_l=[0, 0, 0],
            step_size=Length(0.05, "m"),
        )
        rate, _ch = eng._fwm_rate_max(eng.A, dz=0.05)
        assert rate == pytest.approx(1.0e6 * 4.0, rel=1e-12)

    def test_photon_number_matches_scalar_engine(self):
        """Finding 5: the multimode (A5) sum must carry the dω measure."""
        grid = _grid(8.0, 2**10)
        fiber = _fiber(0.5)
        betas = np.array([-21.8])
        wave = Wave(
            grid=grid,
            envelope=Envelope(
                shape="sech",
                peak_amplitude=2.0,
                pulse_width=Time(0.5e-12, "s"),
            ),
            central_wavelength=WL,
        )
        scalar = SplitStepEngine(
            wave,
            fiber,
            betas,
            include_self_steepening=True,
            step_size=Length(0.025, "m"),
        )
        mm = MultimodeSplitStepEngine(
            [wave],
            fiber,
            betas,
            include_shock=True,
            step_size=Length(0.025, "m"),
        )
        assert mm._photon_number(mm.A) == pytest.approx(scalar._photon_number(mm.A))

    def test_fwm_substep_count_uses_absolute_tensor(self):
        grid = _grid(20.0, 2**8)
        c = np.zeros((3, 3, 3, 3), dtype=complex)
        c[1, 0, 0, 2] = c[2, 0, 0, 1] = 5.0e4
        eng = MultimodeSplitStepEngine(
            [_cw(grid, 4.0), _cw(grid, 1e-3), _cw(grid, 1e-3)],
            _fiber(1.0),
            [np.array([0.0])] * 3,
            include_fwm=True,
            fwm_coupling_tensor=c,
            oam_l=[0, 0, 0],
            step_size=Length(0.05, "m"),
        )
        # rate·dz = 5e4·4·0.05 = 1e4, so n_stab ≈ 4000 > 1.
        assert eng._fwm_substep_count(eng.A, dz=0.05) > 100
