"""Gap M / Gap N: TPA flags and coefficient units, wrapper reach, invariants.

GAP.md §17 (`beta_tpa` was inert under `include_tpa`, and nothing converted a
measured coefficient to the engine's power convention) and §18.1/§18.3/§18.4
(the Gap J/K keywords were unreachable from the solver wrappers, the photon
number was private, ETDRK4 silently froze z-dependent hooks).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from photonics_helper.base import Area, Length, Time, Wavelength
from photonics_helper.gnlse import Fiber, GNLSESolver, SplitStepEngine, tpa_step
from photonics_helper.pulse import Envelope, TemporalGrid, Wave

E_PJ = 100.0
L_M = 0.01
N_STEPS = 500
BETA_PHYS = 1.0e-13  # m/W
A_EFF_M2 = 1.0e-12  # m^2 — a rib mode


def _deck(energy_pj: float = E_PJ, n: int = 2048) -> tuple[Wave, TemporalGrid]:
    grid = TemporalGrid(N=n, Tmax=Time(8e-12, "s"))
    env = Envelope.from_energy("gaussian", energy_pj * 1e-12, Time(200e-15, "s"))
    wave = Wave(
        grid=grid, envelope=env, central_wavelength=Wavelength(1550.0, "nm")
    ).with_field(np.asarray(env.field(grid.t), complex))
    return wave, grid


def _energy(wave_field: np.ndarray, grid: TemporalGrid) -> float:
    return float(np.sum(np.abs(wave_field) ** 2) * grid.dt)


# ── M3.1: TPA closed form through the a_eff route ─────────────────────────


def test_closed_form_tpa_with_area_hook() -> None:
    """β_TPA/A_eff reproduces the per-sample closed form I₀/(1+βI₀z/A_eff).

    Dispersionless and γ=0, so every step's TPA attenuation is exact and the
    only thing under test is the unit conversion plus the flag wiring.
    """
    wave, grid = _deck()
    intensity = np.abs(np.asarray(wave.envelope_field)) ** 2
    fiber = Fiber(
        gamma=0.0,
        alpha=0.0,
        length=Length(L_M, "m"),
        beta_tpa=BETA_PHYS,
        a_eff=Area(A_EFF_M2, "m^2"),
    )
    engine = SplitStepEngine(
        wave,
        fiber,
        betas=np.array([0.0, 0.0]),
        include_raman=False,
        include_free_carriers=True,
        step_size=Length(L_M / N_STEPS, "m"),
    )
    engine.propagate(N_STEPS)

    e_in = _energy(np.asarray(wave.envelope_field), grid)
    e_out = _energy(np.asarray(engine.A), grid)
    beta_norm = BETA_PHYS / A_EFF_M2
    closed = float(np.sum(intensity / (1.0 + beta_norm * intensity * L_M)) * grid.dt)

    assert abs(e_out / e_in - closed / e_in) <= 1e-3
    # the absorption is macroscopic on this deck, not a rounding effect
    assert e_out / e_in < 0.9


# ── M3.2: the flag that does nothing must say so ───────────────────────────


def test_include_tpa_with_beta_tpa_only_warns() -> None:
    """`include_tpa` carries σ_TPA only: beta-only used to be a silent no-op."""
    wave, _ = _deck()
    fiber = Fiber(gamma=0.0, length=Length(L_M, "m"), beta_tpa=1e-3)
    with pytest.warns(UserWarning, match="include_free_carriers"):
        SplitStepEngine(
            wave,
            fiber,
            betas=np.array([0.0, 0.0]),
            include_raman=False,
            include_tpa=True,
            step_size=Length(L_M / 10, "m"),
        )


def test_include_free_carriers_does_not_warn_and_absorbs() -> None:
    """The route the warning names actually applies β_TPA."""
    wave, grid = _deck()
    fiber = Fiber(gamma=0.0, length=Length(L_M, "m"), beta_tpa=1e-3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine = SplitStepEngine(
            wave,
            fiber,
            betas=np.array([0.0, 0.0]),
            include_raman=False,
            include_free_carriers=True,
            step_size=Length(L_M / 400, "m"),
        )
        engine.propagate(400)
    messages = [str(w.message) for w in caught]
    assert not any("include_free_carriers=True (the" in m for m in messages)
    assert _energy(np.asarray(engine.A), grid) < 0.999 * _energy(
        np.asarray(wave.envelope_field), grid
    )


def test_inactive_legacy_tpa_step_needs_no_carrier_frequency() -> None:
    """σ_TPA == 0 returns before the ω₀ demand (the check order was inverted)."""
    wave, grid = _deck(n=256)
    fiber = Fiber(gamma=0.0, length=Length(L_M, "m"))
    A = np.asarray(wave.envelope_field, dtype=complex)
    A_new, U_new = tpa_step(A.copy(), fiber, grid, dz=1e-4, include_tpa=True)
    assert np.array_equal(A_new, A) and U_new == 0.0
    # with sigma_tpa > 0 the demand is still made
    fiber.sigma_tpa = 1e-11
    with pytest.raises(ValueError, match="omega0.*required when include_tpa=True"):
        tpa_step(A.copy(), fiber, grid, dz=1e-4, include_tpa=True)


# ── M3.3: the conversion is exactly A_eff (TPA) and A_eff² (3PA) ───────────


def test_area_hook_converts_by_area_and_area_squared() -> None:
    area = Area(A_EFF_M2, "m^2")
    physical = Fiber(beta_tpa=BETA_PHYS, sigma_tpa=1e-11, a_eff=area)
    engine_units = Fiber(beta_tpa=BETA_PHYS, sigma_tpa=1e-11)

    assert physical.beta_tpa_engine / engine_units.beta_tpa_engine == pytest.approx(
        1.0 / A_EFF_M2, rel=1e-15
    )
    assert physical.sigma_tpa_engine == pytest.approx(1e-11 / A_EFF_M2, rel=1e-15)
    # a_eff=None keeps the stored number bit-identically (backward compatibility)
    assert engine_units.beta_tpa_engine == BETA_PHYS
    assert engine_units.sigma_tpa_engine == 1e-11

    wave, _ = _deck(n=256)
    alpha3 = 1e-26  # m^3/W^2
    scaled = SplitStepEngine(
        wave, physical, betas=np.array([0.0]), include_raman=False,
        three_photon_alpha_3=alpha3,
    )
    raw = SplitStepEngine(
        wave, engine_units, betas=np.array([0.0]), include_raman=False,
    )
    assert scaled._alpha3_scalar is not None
    assert scaled._alpha3_scalar / (alpha3 / A_EFF_M2**2) == pytest.approx(
        1.0, rel=1e-15
    )
    assert raw._alpha3_scalar is None


# ── M3.4: α₃ == 0 is a no-op even with the area hook ───────────────────────


def test_alpha3_zero_with_area_hook_is_bit_identical() -> None:
    wave, grid = _deck()
    betas = np.array([-20.0, 0.0, 0.0])
    fields = []
    for area in (None, Area(A_EFF_M2, "m^2")):
        fiber = Fiber(gamma=0.05, length=Length(L_M, "m"), a_eff=area)
        engine = SplitStepEngine(
            wave,
            fiber,
            betas=betas,
            include_raman=False,
            step_size=Length(L_M / 20, "m"),
            three_photon_alpha_3=0.0,
        )
        engine.propagate(20)
        fields.append(np.asarray(engine.A))
        assert engine.w3pa == 0.0 and engine.w3pa_energy == 0.0
    assert np.array_equal(fields[0], fields[1])


# ── N1: the Gap J/K keywords are reachable from the wrappers ───────────────


def test_gnlse_solver_passes_through_gamma_omega_and_alpha3() -> None:
    wave, grid = _deck(energy_pj=20.0, n=512)
    fiber = Fiber(gamma=0.05, length=Length(1e-3, "m"))
    betas = np.array([-20.0, 0.0, 0.0])
    dz = Length(1e-5, "m")
    gamma_omega = lambda w: 0.05 * (1.0 + 1e-15 * np.asarray(w))  # noqa: E731

    engine = SplitStepEngine(
        wave, fiber, betas=betas, include_raman=False, step_size=dz,
        gamma_omega=gamma_omega, three_photon_alpha_3=1e-3,
    )
    engine.propagate(100)
    solver = GNLSESolver(
        wave, fiber, betas=betas, include_raman=False, step_size=dz,
        gamma_omega=gamma_omega, three_photon_alpha_3=1e-3,
    )
    solver.propagate(100)

    assert np.array_equal(np.asarray(solver.evolution[-1].envelope_field), engine.A)
    assert _energy(np.asarray(engine.A), grid) != _energy(
        np.asarray(wave.envelope_field), grid
    )


# ── N3: public photon_number(A=None) ──────────────────────────────────────


def test_photon_number_is_public() -> None:
    wave, _ = _deck(energy_pj=50.0, n=1024)
    fiber = Fiber(gamma=0.05, length=Length(1e-3, "m"))
    engine = SplitStepEngine(
        wave,
        fiber,
        betas=np.array([-20.0, 0.0, 0.0]),
        include_raman=False,
        step_size=Length(1e-5, "m"),
    )
    q_in = engine.photon_number(np.asarray(wave.envelope_field))
    assert q_in == engine._photon_number(np.asarray(wave.envelope_field))
    engine.propagate(50)
    # default argument is the engine's current field, i.e. Q_out
    assert engine.photon_number() == engine._photon_number(engine.A)
    assert engine.photon_number() != q_in


# ── N4: ETDRK4 and z-dependent hooks ──────────────────────────────────────


def test_etdrk4_static_hook_keeps_fourth_order() -> None:
    """A static gamma_fn must not cost the scheme its order (≥ 3.5 measured)."""
    wave, _ = _deck(n=1024)
    fiber = Fiber(gamma=2.0, length=Length(2e-3, "m"))
    static_gamma = lambda z: 2.0  # noqa: E731

    def run(n_steps: int) -> np.ndarray:
        engine = SplitStepEngine(
            wave,
            fiber,
            betas=np.array([0.0, 0.0]),
            include_raman=False,
            step_size=Length(2e-3 / n_steps, "m"),
            scheme="etdrk4",
            gamma_fn=static_gamma,
        )
        engine.propagate(n_steps)
        return np.asarray(engine.A)

    reference = run(4096)
    scale = float(np.max(np.abs(reference)))
    coarse = float(np.max(np.abs(run(64) - reference))) / scale
    fine = float(np.max(np.abs(run(128) - reference))) / scale
    assert np.log2(coarse / fine) >= 3.5


def test_gamma_omega_warns_that_hooks_are_ignored() -> None:
    """`gamma_omega` overrides gamma_fn/a_eff_fn; that must not be silent."""
    wave, _ = _deck(n=256)
    fiber = Fiber(gamma=0.05, length=Length(L_M, "m"))
    betas = np.array([-20.0, 0.0, 0.0])
    with pytest.warns(UserWarning, match="gamma_fn.*ignored"):
        SplitStepEngine(
            wave, fiber, betas=betas, include_raman=False,
            gamma_omega=lambda w: 0.05 * np.ones_like(np.asarray(w)),
            gamma_fn=lambda z: 0.05,
        )
    with pytest.warns(UserWarning, match="a_eff_fn.*ignored"):
        SplitStepEngine(
            wave, fiber, betas=betas, include_raman=False,
            gamma_omega=lambda w: 0.05 * np.ones_like(np.asarray(w)),
            a_eff_fn=lambda z: A_EFF_M2,
        )
    # gamma_omega alone stays silent
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SplitStepEngine(
            wave, fiber, betas=betas, include_raman=False,
            gamma_omega=lambda w: 0.05 * np.ones_like(np.asarray(w)),
        )
    assert not any("ignored" in str(w.message) for w in caught)


def test_etdrk4_warns_on_z_dependent_hook() -> None:
    wave, _ = _deck(n=256)
    fiber = Fiber(gamma=0.05, length=Length(1e-3, "m"))
    with pytest.warns(UserWarning, match="freezes gamma_fn"):
        SplitStepEngine(
            wave,
            fiber,
            betas=np.array([-20.0]),
            include_raman=False,
            scheme="etdrk4",
            gamma_fn=lambda z: 0.05 * (1.0 + 100.0 * z),
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SplitStepEngine(
            wave,
            fiber,
            betas=np.array([-20.0]),
            include_raman=False,
            scheme="etdrk4",
            gamma_fn=lambda z: 0.05,
        )
    assert not any("freezes" in str(w.message) for w in caught)
