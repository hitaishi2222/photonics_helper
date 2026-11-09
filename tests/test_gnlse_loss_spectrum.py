"""Gap L: per-frequency linear field-amplitude loss on the analytic-signal axis."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from photonics_helper.base import Area, Length, Time, Wavelength
from photonics_helper.gnlse import Fiber, GNLSESolver, SplitStepEngine
from photonics_helper.pulse import Envelope, TemporalGrid, Wave
from photonics_helper.raman import RamanSpec

WL0 = 1550e-9
LENGTH = 0.02


def _deck(n: int = 256, length: float = LENGTH):
    grid = TemporalGrid(N=n, Tmax=Time(8e-12, "s"))
    env = Envelope.from_energy(
        shape="sech", energy=1e-12, fwhm=Time(200e-15, "s")
    )
    wave = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(WL0, "m"))
    fiber = Fiber(gamma=0.0, alpha=0.0, length=Length(length, "m"))
    return wave, grid, fiber


def _engine(steps: int = 20, *, scheme: str = "strang", **kwargs):
    wave, grid, fiber = _deck(length=kwargs.pop("length", LENGTH))
    return SplitStepEngine(
        wave,
        fiber,
        betas=np.array([0.0]),
        include_raman=False,
        step_size=Length(fiber.length.as_m / steps, "m"),
        scheme=scheme,
        **kwargs,
    ), grid


@pytest.mark.parametrize("scheme", ["strang", "etdrk4"])
def test_alpha_w_ramp_matches_each_spectral_bin(scheme: str):
    engine, grid = _engine(
        40,
        scheme=scheme,
        alpha_w=lambda w: 2.0 + 0.5 * np.tanh(np.asarray(w) / 2e13),
    )
    # A nonvanishing flat spectrum makes every per-bin ratio well conditioned.
    engine.A = np.asarray(grid.ifft(np.ones(grid.N, dtype=complex)))
    a_in = np.asarray(engine.A)
    engine.propagate(40, nsaves=2)
    ratio = np.abs(grid.fft(engine.A) / grid.fft(a_in)) ** 2
    expected = np.exp(-engine._alpha_w * LENGTH)
    assert np.max(np.abs(ratio / expected - 1.0)) <= 1e-9
    assert engine.alpha_w is not None  # the run record names the accounted loss


@pytest.mark.parametrize("scheme", ["strang", "etdrk4"])
@pytest.mark.parametrize("alpha_w", [0.3, lambda w: np.full_like(w, 0.3)])
def test_flat_alpha_w_is_bit_identical_to_fiber_alpha(scheme: str, alpha_w):
    spectral, _ = _engine(50, scheme=scheme, alpha_w=alpha_w)
    legacy, _ = _engine(50, scheme=scheme)
    legacy.fiber.alpha = 0.3
    spectral.propagate(50, nsaves=2)
    legacy.propagate(50, nsaves=2)
    assert np.array_equal(np.asarray(spectral.A), np.asarray(legacy.A))


def test_gnlse_solver_passes_alpha_w_through():
    wave, _, fiber = _deck()
    alpha_w = np.full(wave.grid.N, 0.2)
    solver = GNLSESolver(
        wave,
        fiber,
        betas=np.array([0.0]),
        include_raman=False,
        step_size=Length(LENGTH / 10, "m"),
        alpha_w=alpha_w,
    )
    solver.propagate(10, nsaves=2)
    engine, _ = _engine(10, alpha_w=alpha_w)
    engine.propagate(10, nsaves=2)
    assert np.array_equal(
        np.asarray(solver.evolution[-1].envelope_field), np.asarray(engine.A)
    )


def test_flat_alpha_w_preserves_etdrk4_fourth_order_on_dispersive_deck():
    steps = (1000, 2000, 4000)
    drifts = []
    for n_steps in steps:
        wave, grid, _ = _deck(n=512, length=LENGTH)
        wave.envelope = Envelope.from_energy(
            shape="sech", energy=10e-12, fwhm=Time(200e-15, "s")
        )
        spec = RamanSpec(
            name="Silica", raman_shift_cm=440, raman_linewidth_cm=45, fR=0.18
        )
        fiber = Fiber(
            gamma=10.0,
            alpha=0.0,
            A_eff=Area(0.5e-12, "m^2"),
            length=Length(LENGTH, "m"),
            raman_response=spec.response(grid),
        )
        w = np.asarray(grid.w)
        full_beta = -2e-26 * w**2 / 2 + 1e-40 * w**3 / 6 - 1e-55 * w**4 / 24
        engine = SplitStepEngine(
            wave,
            fiber,
            betas=None,
            full_beta=full_beta,
            include_raman=True,
            gamma_omega=10.0 * (1.0 + 0.3 * np.tanh(w / 2e13)),
            alpha_w=0.3,
            step_size=Length(LENGTH / n_steps, "m"),
            scheme="etdrk4",
        )
        engine.propagate(n_steps, nsaves=2)
        energy = np.asarray(engine.energy_vs_z)
        invariant = energy * np.exp(0.3 * engine.z_array)
        drifts.append(float(abs(invariant[-1] / invariant[0] - 1.0)))
    order = float(-np.polyfit(np.log(steps), np.log(drifts), 1)[0])
    assert order >= 3.9, f"order {order:.3f}; normalized drifts {drifts}"


def test_alpha_w_refuses_double_counting_and_bad_bins():
    wave, _, fiber = _deck()
    fiber.alpha = 0.2
    with pytest.raises(ValueError, match="double count"):
        SplitStepEngine(wave, fiber, betas=np.array([0.0]), alpha_w=0.1)
    with pytest.raises(TypeError, match="channel_alpha"):
        SplitStepEngine(
            wave, Fiber(gamma=0.0), betas=np.array([0.0]), alpha_w=0.1,
            channel_alpha=np.array([0.1]),
        )
    for bad in (np.full(wave.grid.N, -0.1), np.full(wave.grid.N, np.nan)):
        with pytest.raises(ValueError, match="finite and non-negative"):
            SplitStepEngine(wave, fiber=Fiber(gamma=0.0), betas=np.array([0.0]), alpha_w=bad)
    with pytest.raises(ValueError, match="must match the grid"):
        SplitStepEngine(
            wave, Fiber(gamma=0.0), betas=np.array([0.0]), alpha_w=np.zeros(3)
        )


# ─── Conservation-guard predicate (multimode_scg GAPS item 16) ────────────────


def _guard_messages(**kwargs) -> list[str]:
    """Run a deck that drifts >1 % with no loss, plus the given switches.

    The self-steepening RK4 substep is not unitary, so two steps on a strong
    nonlinearity drift by tens of percent while nothing absorbs. Each guard test
    below adds exactly one switch to that deck and asks what the guard says.
    """
    fiber_kwargs = {
        key: kwargs.pop(key) for key in ("alpha", "beta_tpa", "sigma_tpa") if key in kwargs
    }
    wave, _, _ = _deck(n=256, length=LENGTH)
    fiber_kwargs.setdefault("alpha", 0.0)
    fiber = Fiber(gamma=500.0, length=Length(LENGTH, "m"), **fiber_kwargs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine = SplitStepEngine(
            wave,
            fiber,
            betas=np.array([-2e-26]),
            include_raman=False,
            include_self_steepening=True,
            step_size=Length(LENGTH / 2, "m"),
            **kwargs,
        )
        engine.propagate(2, nsaves=2)
    # or the test proves nothing:
    energy = np.asarray(engine.energy_vs_z)
    assert abs(energy[-1] / energy[0] - 1.0) > 0.01
    return [str(w.message) for w in caught]


@pytest.mark.parametrize(
    "switch",
    [
        {"alpha": 0.5},
        {"alpha_w": 5.0},
        {"include_tpa": True, "sigma_tpa": 1e-4},
        {"include_free_carriers": True, "beta_tpa": 1.0},
        {"three_photon_alpha_3": 1e-3},
        {"beta_tpa_pointwise": True, "beta_tpa": 1.0},
    ],
    ids=[
        "alpha",
        "alpha_w",
        "sigma_tpa",
        "free_carriers",
        "three_photon",
        "pointwise_2pa",
    ],
)
def test_guard_treats_every_active_loss_as_lossy(switch):
    """A run that absorbs must never be called "with loss/TPA disabled"."""
    messages = _guard_messages(**switch)
    assert not any("loss/TPA disabled" in m for m in messages), messages


def test_guard_still_warns_on_a_genuinely_lossless_run():
    assert any("loss/TPA disabled" in m for m in _guard_messages())


@pytest.mark.parametrize(
    "switch",
    [
        {"beta_tpa": 1.0},  # bare beta_TPA: no TPA path armed
        {"include_tpa": True, "beta_tpa": 1.0},  # include_tpa without sigma_tpa
    ],
    ids=["inert_beta_tpa", "inert_include_tpa"],
)
def test_guard_does_not_excuse_drift_with_an_inert_switch(switch):
    """A switch that applies nothing must not silence a real drift alarm."""
    messages = _guard_messages(**switch)
    assert any("loss/TPA disabled" in m for m in messages), messages


def test_verify_convergence_reports_a_lossy_run_as_lossy():
    """The convergence verdict and the drift warning must agree (task 6.3)."""
    wave, _, fiber = _deck()
    solver = GNLSESolver(
        wave,
        fiber,
        betas=np.array([0.0]),
        include_raman=False,
        alpha_w=0.3,
        step_size=Length(LENGTH / 200, "m"),
    )
    solver.propagate(200, nsaves=2)
    report = solver.verify_convergence(num_steps=200, warn=False)
    assert "loss/gain" in report.invariant
    assert report.drift_pct == 0.0
