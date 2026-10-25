"""Three-photon absorption (``three_photon_alpha_3``) and its ``w3pa`` ledger.

Gap K of ``multimode_scg/GAP.md`` §14. The field form is

    ∂_z A = −α₃ I² A / 2,      I = |A|²   (``|A|²`` in watts, engine units)

which is the standard three-photon absorption equation *per sample*,

    dI/dz = −α₃ I³      ⇒      I(z) = I₀ / sqrt(1 + 2 α₃ I₀² z),

and with a Kerr term ``iγI·A`` the per-sample solution is closed-form too:

    I(z) = I₀/√(1+2α₃I₀²z),   φ(z) = (γ/(α₃I₀))·(√(1+2α₃I₀²z) − 1).

(The acceptance text writes ``I₀/(1 + 2α₃I₀²L)``; for the field form it
specifies the denominator is a **square root** — the linear form is what
``dI/dz = −2α₃I²`` would give. Both numbers are reported by the tests.)

Ledgers exposed after :meth:`SplitStepEngine.propagate`:

* ``w3pa = ∫∫|A|⁴ dt dz`` (the brief's ``∫∫I²dt dz``) — for the flat-field
  deck ``I_out/I_in = exp(−α₃·w3pa/T)`` exactly;
* ``w3pa_energy = ∫∫|A|⁶ dt dz`` — the integral that closes the energy budget:
  ``dE/dz = −α₃∫|A|⁶dt``, so ``E_out = E_in − α₃·w3pa_energy``. This is the
  one to use for a pulse deck; ``w3pa`` is the intensity-squared action.
* ``w3pa_vs_z`` — the cumulative ``w3pa`` per saved step.

No free carriers: 3PA-generated carriers (and their FCA / recombination) are
not modelled — that is ``include_free_carriers``' separate TPA-based model.
"""

from __future__ import annotations

import numpy as np
import pytest

from photonics_helper.base import Area, Length, Time, Wavelength
from photonics_helper.gnlse import Fiber, SplitStepEngine
from photonics_helper.pulse import Envelope, TemporalGrid, Wave

WL0 = 1550e-9
LEN = 0.1  # m
A3 = 5.0  # 1/(W²·m) in engine units (|A|² in W)
GAMMA = 2.0  # 1/(W·m)
I0 = 1.0  # W
TMAX = 1e-11


def _cw_engine(steps: int, scheme: str = "strang", a3=A3, gamma: float = GAMMA, **kw):
    """Flat (CW) 1 W field, no dispersion, 0.1 m: the analytic 3PA deck."""
    grid = TemporalGrid(N=64, Tmax=Time(TMAX, "s"))
    env = Envelope(
        shape="sech", peak_amplitude=np.sqrt(I0), pulse_width=Time(1e-12, "s")
    )
    pulse = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(WL0, "m"))
    pulse.with_field(np.full(grid.N, np.sqrt(I0), dtype=complex))
    fiber = Fiber(
        gamma=gamma, alpha=0.0, A_eff=Area(1e-12, "m^2"), length=Length(LEN, "m")
    )
    engine = SplitStepEngine(
        pulse=pulse,
        fiber=fiber,
        betas=np.array([0.0]),
        step_size=Length(LEN / steps, "m"),
        scheme=scheme,
        three_photon_alpha_3=a3,
        **kw,
    )
    engine.propagate(steps)
    return engine, grid


def _exact(a3=A3, gamma: float = GAMMA, length: float = LEN, i0: float = I0):
    """Closed-form per-sample solution and its two integrals over ``[0, L]``."""
    if a3 == 0.0:
        return (
            np.sqrt(i0) * np.exp(1j * gamma * i0 * length),
            i0**2 * length,
            i0**3 * length,
        )
    q = 1.0 + 2.0 * a3 * i0**2 * length
    i_l = i0 / np.sqrt(q)
    phase = (gamma / (a3 * i0)) * (np.sqrt(q) - 1.0)  # γ∫I dz
    w2 = np.log(q) / (2.0 * a3)  # ∫I² dz per sample
    w3 = (i0 - i_l) / a3  # ∫I³ dz per sample
    return np.sqrt(i_l) * np.exp(1j * phase), w2, w3


def _rel(got, want) -> float:
    return float(np.max(np.abs(got - want)) / np.max(np.abs(want)))


def test_pure_three_photon_matches_the_closed_form():
    """α₃ only, no dispersion: both schemes on the analytic solution at ≤1e-6.

    Strang solves the per-sample ODE in closed form inside the split step, so
    its error here is round-off at every step count (≤5e-14) and no order is
    measurable — the order gate is measured on the coupled deck below and on
    ETDRK4 (which carries the term as a pointwise RHS). ETDRK4's residual
    (≤7e-13 at 100 steps) is its own O(dz⁴).

    The literal ``I₀/(1+2α₃I₀²L)`` from the acceptance text would be 18.4 %
    low on this deck (that is the solution of ``dI/dz = −2α₃I²``, not of the
    field form's ``dI/dz = −α₃I³``); the square-root form matches to round-off.
    """
    exact, _, _ = _exact(gamma=0.0)
    want = np.full(64, exact)
    linear_form = I0 / (1.0 + 2.0 * A3 * I0**2 * LEN)
    exact_intensity = I0 / np.sqrt(1.0 + 2.0 * A3 * I0**2 * LEN)
    errors = {"strang": [], "etdrk4": []}
    for scheme in ("strang", "etdrk4"):
        for steps in (100, 200, 400, 800):
            engine, _ = _cw_engine(steps, scheme=scheme, gamma=0.0, a3=A3)
            errors[scheme].append(_rel(np.asarray(engine.A), want))
        assert max(errors[scheme]) <= 1e-6, f"{scheme}: {errors[scheme]}"
    # The linear-denominator reading of the acceptance formula is not this flow.
    assert abs(linear_form / exact_intensity - 1.0) > 0.1


def test_etdrk4_three_photon_order_is_fourth():
    """ETDRK4 carries ``−α₃I²A/2`` as a pointwise RHS: O(dz⁴) on the log scale.

    Measured 2.6e-7 … 1.7e-12 over 5 → 80 steps, fitted order 4.29.
    """
    exact, _, _ = _exact(gamma=0.0)
    want = np.full(64, exact)
    steps = (5, 10, 20, 40, 80)
    errs = []
    for num_steps in steps:
        engine, _ = _cw_engine(num_steps, scheme="etdrk4", gamma=0.0)
        errs.append(_rel(np.asarray(engine.A), want))
    order = float(-np.polyfit(np.log(steps), np.log(errs), 1)[0])
    assert order >= 3.5, f"ETDRK4 3PA order {order:.2f} (errors {errs})"


def test_strang_kerr_plus_three_photon_is_second_order():
    """Kerr + 3PA on the same deck against the closed-form per-sample flow.

    The 3PA sub-step is placed symmetrically around the exact Kerr phase
    (half-loss · phase · half-loss), so the composition is second order:
    measured errors 2.7e-7 … 4.2e-9 over 100 → 800 steps, fitted order 2.00.
    A single trailing loss sub-step would compose at first order (order 1.00 on
    the same deck) — the placement is what buys the scheme order.
    """
    exact, _, _ = _exact()
    want = np.full(64, exact)
    steps = (100, 200, 400, 800)
    errs = []
    for num_steps in steps:
        engine, _ = _cw_engine(num_steps, scheme="strang")
        errs.append(_rel(np.asarray(engine.A), want))
    order = float(-np.polyfit(np.log(steps), np.log(errs), 1)[0])
    assert order >= 1.9, f"Strang Kerr+3PA order {order:.2f} (errors {errs})"
    assert errs[-1] <= 1e-6, f"finest-step error {errs[-1]:.2e}"


def test_w3pa_ledger_matches_the_analytic_integrals():
    """``w3pa`` … ``∫∫I²dtdz``, ``w3pa_energy`` … ``∫∫I³dtdz``, to ≤1e-6.

    On the flat deck the two exponentials are exact statements:
    ``I_out/I_in = exp(−α₃·w3pa/T)`` (since ``∫I²dz = ln(1+2α₃I₀²L)/(2α₃)``),
    and ``E_out = E_in − α₃·w3pa_energy``. Both close at ≤1e-7 with 1000 steps.
    """
    _, w2, w3 = _exact()
    window = 64 * (TMAX / 64)
    for scheme in ("strang", "etdrk4"):
        engine, grid = _cw_engine(1000, scheme=scheme)
        assert abs(engine.w3pa / (w2 * grid.time_window) - 1.0) <= 1e-6
        assert abs(engine.w3pa_energy / (w3 * grid.time_window) - 1.0) <= 1e-6
        energy = np.asarray(engine.energy_vs_z, dtype=float)
        # Exact energy budget: E_out = E_in − α₃·∫∫I³dt dz.
        clos = abs((energy[-1] - energy[0]) / (-A3 * engine.w3pa_energy) - 1.0)
        assert clos <= 1e-6, f"{scheme}: energy ledger closes to {clos:.2e}"
        # Flat-field exponential form of the intensity ratio.
        exp_form = float(np.exp(-A3 * engine.w3pa / window))
        assert abs(energy[-1] / energy[0] / exp_form - 1.0) <= 1e-6
        assert engine.w3pa_vs_z.shape == engine.z_array.shape
        assert np.all(np.diff(engine.w3pa_vs_z) >= 0.0)


def test_w3pa_vs_z_is_unavailable_without_the_term():
    engine, _ = _cw_engine(10, a3=0.0)
    with pytest.raises(RuntimeError, match="three_photon_alpha_3"):
        engine.w3pa_vs_z


def test_three_photon_zero_is_bit_identical_to_omitting_it():
    """The no-op gate ``tests/test_gnlse_effects.py`` applies to TPA."""
    reference, _ = _cw_engine(50, a3=None)
    zero, _ = _cw_engine(50, a3=0.0)
    zero_callable, _ = _cw_engine(50, a3=np.zeros(64))
    assert np.array_equal(np.asarray(reference.A), np.asarray(zero.A))
    assert np.array_equal(np.asarray(reference.A), np.asarray(zero_callable.A))
    assert reference.w3pa == 0.0
    assert reference.w3pa_energy == 0.0
    assert reference.w3pa_spectral == 0.0
    with_term, _ = _cw_engine(50, a3=A3)
    assert not np.array_equal(np.asarray(reference.A), np.asarray(with_term.A))


def test_three_photon_rejects_negative_coefficient():
    with pytest.raises(ValueError, match="non-negative"):
        _cw_engine(1, a3=-1.0)


def _mixed_engine(steps: int, scheme: str, a3: float):
    """full-β + Raman + γ(Ω) + α₃ on the 10 pJ / 200 fs / 20 mm deck."""
    from photonics_helper.raman import RamanSpec

    length = 20e-3
    grid = TemporalGrid(N=1024, Tmax=Time(8e-12, "s"))
    env = Envelope.from_energy(shape="sech", energy=10e-12, fwhm=Time(200e-15, "s"))
    pulse = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(WL0, "m"))
    spec = RamanSpec(name="Silica", raman_shift_cm=440, raman_linewidth_cm=45, fR=0.18)
    fiber = Fiber(
        gamma=10.0,
        alpha=0.0,
        A_eff=Area(0.5e-12, "m^2"),
        length=Length(length, "m"),
        raman_response=spec.response(grid),
    )
    w = np.asarray(grid.w, dtype=float)
    curve = -2.0e-26 * w**2 / 2.0 + 1.0e-40 * w**3 / 6.0 - 1.0e-55 * w**4 / 24.0
    engine = SplitStepEngine(
        pulse=pulse,
        fiber=fiber,
        betas=None,
        full_beta=curve,
        include_raman=True,
        gamma_omega=10.0 * (1.0 + 0.3 * np.tanh(w / 2e13)),
        three_photon_alpha_3=a3,
        step_size=Length(length / steps, "m"),
        scheme=scheme,
    )
    engine.propagate(steps)
    return engine


def test_mixed_deck_runs_and_the_ledger_closes():
    """Full β + Raman + γ(Ω) + α₃ under Strang and ETDRK4.

    α₃ = 2e-4 1/(W²·m) accumulates ~1 % energy loss on this deck. The energy
    ledger ``E_out = E_in − α₃·w3pa_energy`` closes to ≤1e-3 (measured
    1.1e-5 / 2.4e-6 at 2000 steps). The brief's ``exp(−α₃·w3pa)`` form does
    *not* close on a pulse deck — measured 1.4e-1 deviation at 14 % loss —
    because it weights ``⟨I²⟩`` over the whole window whereas the energy loss
    weights ``⟨I³⟩/⟨I⟩`` (a sech¹²-weighted peak): it is the flat-field /
    sample-wise statement, and the flat deck above tests it exactly.
    """
    for scheme in ("strang", "etdrk4"):
        engine = _mixed_engine(2000, scheme, 2e-4)
        energy = np.asarray(engine.energy_vs_z, dtype=float)
        loss = float(1.0 - energy[-1] / energy[0])
        assert loss > 1e-3, f"{scheme}: deck's 3PA loss {loss:.2e} too small to test"
        clos = abs((energy[-1] - energy[0]) / (-2e-4 * engine.w3pa_energy) - 1.0)
        assert clos <= 1e-3, f"{scheme}: energy ledger closes to {clos:.2e}"
        assert np.all(np.isfinite(np.asarray(engine.A)))
        assert engine.w3pa > 0.0
        assert engine.w3pa_vs_z[-1] == engine.w3pa


@pytest.mark.parametrize("scheme", ["strang", "etdrk4"])
@pytest.mark.parametrize("spectral_input", [False, True])
def test_w3pa_spectral_matches_constant_alpha3_ledger(scheme, spectral_input):
    a3 = np.full(64, A3) if spectral_input else A3
    engine, _ = _cw_engine(1000, scheme=scheme, a3=a3)
    expected = A3 * engine.w3pa_energy
    assert abs(engine.w3pa_spectral / expected - 1.0) <= 1e-10


def _spectral_alpha3(engine):
    w = np.asarray(engine.grid.w)
    return 2e-4 * (1.0 + 0.8 * np.tanh(w / 2e13))


def test_spectral_alpha3_energy_closure_and_legacy_residual():
    probe = _mixed_engine(10, "etdrk4", None)
    a3_w = _spectral_alpha3(probe)
    a3_center = float(a3_w[np.argmin(np.abs(probe.grid.w))])
    engine = _mixed_engine(4000, "etdrk4", a3_w)
    energy = np.asarray(engine.energy_vs_z)
    delta = float(energy[-1] - energy[0])
    spectral_residual = abs((delta + engine.w3pa_spectral) / energy[0])
    legacy_residual = abs(
        (delta + a3_center * engine.w3pa_energy) / energy[0]
    )
    assert spectral_residual <= 1e-6
    # Demonstration that the old constant-coefficient closure is not spectral.
    assert legacy_residual > 1e-6
    assert legacy_residual > 10.0 * spectral_residual


def test_etdrk4_full_beta_gamma_omega_spectral_alpha3_is_fourth_order():
    probe = _mixed_engine(10, "etdrk4", None)
    a3_w = _spectral_alpha3(probe)
    reference = np.asarray(_mixed_engine(8000, "etdrk4", a3_w).A)
    steps = (250, 500, 1000, 2000)
    errors = []
    for n_steps in steps:
        field = np.asarray(_mixed_engine(n_steps, "etdrk4", a3_w).A)
        errors.append(float(np.linalg.norm(field - reference) / np.linalg.norm(reference)))
    order = float(-np.polyfit(np.log(steps), np.log(errors), 1)[0])
    assert order >= 3.9, f"order {order:.3f}; field errors {errors}"
