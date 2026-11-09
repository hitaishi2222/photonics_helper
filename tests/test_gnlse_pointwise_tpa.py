"""Carrier-free 2PA attenuation: ``beta_tpa_pointwise``.

The point of the term (change ``add-pointwise-tpa-term``) is that pure 2PA loss
must be expressible *without* the history-dependent carrier state, so the
exponential integrators can carry it. The exactness test below is the one that
pins the physics: with no dispersion and no Kerr, the split-step schemes reduce
to the closed-form solution ``I(z) = I₀/(1 + β_TPA I₀ z)`` and must reproduce it
to round-off. A sign error, a factor 2 or a missing ``dz/2`` fails it.
"""

from __future__ import annotations

import numpy as np
import pytest

from photonics_helper.base import Area, Length, Time, Wavelength
from photonics_helper.gnlse import Fiber, GNLSESolver, SplitStepEngine
from photonics_helper.pulse import Envelope, TemporalGrid, Wave

WL0 = 1550e-9
LENGTH = 0.02
#: Engine-convention β_TPA. Peak |A|² ≈ 8.8 on this deck, so β = 1 absorbs
#: ~15 % over 2 cm — small enough to be physical, large enough to be visible.
BETA = 1.0
STEPS = 100


def _deck(*, energy: float = 2e-12, beta_tpa: float = BETA, gamma: float = 0.0):
    grid = TemporalGrid(N=256, Tmax=Time(8e-12, "s"))
    env = Envelope.from_energy(
        shape="sech", energy=energy, fwhm=Time(200e-15, "s")
    )
    wave = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(WL0, "m"))
    fiber = Fiber(
        gamma=gamma, alpha=0.0, length=Length(LENGTH, "m"), beta_tpa=beta_tpa
    )
    return wave, fiber


def _engine(steps: int = STEPS, scheme: str = "strang", **kwargs):
    beta_tpa = kwargs.pop("beta_tpa", BETA)
    gamma = kwargs.pop("gamma", 0.0)
    wave, fiber = _deck(beta_tpa=beta_tpa, gamma=gamma)
    return SplitStepEngine(
        wave,
        fiber,
        betas=np.array([0.0]),
        include_raman=False,
        step_size=Length(LENGTH / steps, "m"),
        scheme=scheme,
        **kwargs,
    )


def _rel_err_vs_closed_form(
    intensity: np.ndarray, intensity0: np.ndarray, beta: float
) -> float:
    exact = intensity0 / (1.0 + beta * intensity0 * LENGTH)
    mask = intensity0 > 1e-6 * intensity0.max()  # sech tails are 0/0, not physics
    return float(
        np.max(np.abs(intensity[mask] - exact[mask]) / exact[mask])
    )


@pytest.mark.parametrize("scheme", ["strang", "yoshida4"])
def test_split_step_schemes_are_exact_on_a_pure_2pa_deck(scheme: str):
    engine = _engine(scheme=scheme, beta_tpa_pointwise=True)
    intensity_before = np.abs(engine.A) ** 2
    engine.propagate(STEPS, nsaves=2)
    intensity = np.abs(engine.A) ** 2

    assert _rel_err_vs_closed_form(intensity, intensity_before, BETA) < 1e-10
    # ...and the term actually acted, so "exact" cannot be "did nothing".
    loss = 1.0 - float(intensity.max() / intensity_before.max())
    assert loss == pytest.approx(1.0 - 1.0 / (1.0 + BETA * 8.81 * LENGTH), rel=0.05)


@pytest.mark.parametrize("scheme", ["etdrk4", "rk4ip"])
def test_exponential_integrators_carry_2pa(scheme: str):
    """The capability itself: these schemes refused every 2PA path before."""
    engine = _engine(scheme=scheme, beta_tpa_pointwise=True)
    intensity_before = np.abs(engine.A) ** 2
    engine.propagate(STEPS, nsaves=2)
    intensity = np.abs(engine.A) ** 2

    assert _rel_err_vs_closed_form(intensity, intensity_before, BETA) < 1e-8


@pytest.mark.parametrize("steps", [STEPS])
def test_strang_and_etdrk4_agree(steps: int):
    """Same physics on both integrators, not just "both run"."""
    a = _engine(steps=steps, scheme="strang", beta_tpa_pointwise=True)
    b = _engine(steps=steps, scheme="etdrk4", beta_tpa_pointwise=True)
    a.propagate(steps, nsaves=2)
    b.propagate(steps, nsaves=2)
    rel = np.linalg.norm(np.asarray(a.A) - np.asarray(b.A)) / np.linalg.norm(
        np.asarray(b.A)
    )
    assert float(rel) < 1e-6


def test_etdrk4_keeps_fourth_order_with_2pa_in_the_remainder():
    """If 2PA were folded into the diagonal L, the order (and the answer) break.

    The loss must converge to the algebraic ``1/(1+βI₀z)``, not to an
    exponential: a diagonal-operator treatment gives ``exp(−βI₀z)``, which no
    amount of step refinement fixes. Strong β is used so the per-step change is
    measurable at coarse step counts.
    """
    beta = 200.0
    steps = (10, 20, 40)
    errs = []
    for n_steps in steps:
        engine = _engine(
            steps=n_steps, scheme="etdrk4", beta_tpa=beta, beta_tpa_pointwise=True
        )
        intensity_before = np.abs(engine.A) ** 2
        engine.propagate(n_steps, nsaves=2)
        intensity = np.abs(engine.A) ** 2
        exact = intensity_before / (1.0 + beta * intensity_before * LENGTH)
        errs.append(
            float(
                np.linalg.norm(intensity - exact) / np.linalg.norm(exact)
            )
        )
    order = float(-np.polyfit(np.log(steps), np.log(errs), 1)[0])
    assert order >= 3.5, f"order {order:.3f}; errors {errs}"


def test_refuses_double_counting_with_free_carriers():
    wave, fiber = _deck()
    with pytest.raises(ValueError, match="applied twice"):
        SplitStepEngine(
            wave,
            fiber,
            betas=np.array([0.0]),
            beta_tpa_pointwise=True,
            include_free_carriers=True,
        )
    with pytest.raises(ValueError, match="applied twice"):
        GNLSESolver(
            wave,
            fiber,
            betas=np.array([0.0]),
            beta_tpa_pointwise=True,
            include_free_carriers=True,
        )


def test_inert_flag_warns_and_applies_nothing():
    with pytest.warns(UserWarning, match="no-op"):
        engine = _engine(beta_tpa=0.0, beta_tpa_pointwise=True)
    reference = _engine(beta_tpa=0.0)
    engine.propagate(10, nsaves=2)
    reference.propagate(10, nsaves=2)
    assert np.array_equal(np.asarray(engine.A), np.asarray(reference.A))


def test_physical_beta_tpa_is_normalised_by_the_area_hook():
    """``Fiber(beta_tpa=…, a_eff=…)`` must reach the term as β/A_eff."""
    area = 0.5e-12
    beta_physical = 6.2e-15
    wave_phys, _ = _deck(beta_tpa=0.0)
    fiber_phys = Fiber(
        gamma=0.0,
        alpha=0.0,
        length=Length(LENGTH, "m"),
        beta_tpa=beta_physical,
        a_eff=Area(area, "m^2"),
    )
    physical = SplitStepEngine(
        wave_phys,
        fiber_phys,
        betas=np.array([0.0]),
        include_raman=False,
        step_size=Length(LENGTH / 10, "m"),
        beta_tpa_pointwise=True,
    )
    engine_convention = _engine(
        steps=10, beta_tpa=beta_physical / area, beta_tpa_pointwise=True
    )
    physical.propagate(10, nsaves=2)
    engine_convention.propagate(10, nsaves=2)
    rel = np.linalg.norm(np.asarray(physical.A) - np.asarray(engine_convention.A)) / (
        np.linalg.norm(np.asarray(engine_convention.A))
    )
    assert float(rel) < 1e-12


def test_gnlse_solver_passes_the_flag_through():
    wave, fiber = _deck()
    solver = GNLSESolver(
        wave,
        fiber,
        betas=np.array([0.0]),
        include_raman=False,
        step_size=Length(LENGTH / STEPS, "m"),
        beta_tpa_pointwise=True,
    )
    assert solver._make_engine(solver.step_size).beta_tpa_pointwise is True
    solver.propagate(STEPS, nsaves=2)
    energy = np.asarray(solver.energy_vs_z)
    assert energy[-1] < 0.95 * energy[0]  # the 2PA loss really happened

    # ...and it is the same physics the engine direct path gives.
    reference = _engine(steps=STEPS, beta_tpa_pointwise=True)
    reference.propagate(STEPS, nsaves=2)
    solver_spectrum = np.asarray(solver.spectra_vs_z[1][-1])
    reference_spectrum = np.asarray(reference.spectra_vs_z[1][-1])
    assert np.allclose(solver_spectrum, reference_spectrum, rtol=1e-10, atol=0.0)


def test_guard_does_not_call_a_2pa_run_lossless():
    """"drift … with loss/TPA disabled" must not print on a run that absorbs.

    The lossless deck is the self-steepening one: its RK4 shock substep is not
    unitary, so with two steps it drifts by tens of percent and the guard has
    something to say. Arming the carrier-free 2PA term must silence the
    "loss/TPA disabled" wording (the drift is then expected, not a symptom) —
    while the run still drifts, so the suppression is not "no warning at all
    because nothing happened".
    """

    def _warnings(*, armed: bool) -> list[str]:
        import warnings

        wave, fiber = _deck(gamma=500.0, beta_tpa=1.0)
        engine = SplitStepEngine(
            wave,
            fiber,
            betas=np.array([-2e-26]),
            include_raman=False,
            include_self_steepening=True,
            step_size=Length(LENGTH / 2, "m"),
            beta_tpa_pointwise=armed,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            engine.propagate(2, nsaves=2)
        energy = np.asarray(engine.energy_vs_z)
        assert abs(energy[-1] / energy[0] - 1.0) > 0.01  # the guard has a reason
        return [str(w.message) for w in caught]

    assert any("loss/TPA disabled" in m for m in _warnings(armed=False))
    assert not any("loss/TPA disabled" in m for m in _warnings(armed=True))
