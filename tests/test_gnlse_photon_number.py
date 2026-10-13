"""The shock-run invariant: Blow & Wood (1989) photon number, not energy.

Background (``formal/CHANGELOG.md`` 2026-10-27, ``CONFORMANCE.md`` D1–D9/F11):
a fissioning Raman+shock run legitimately changes the unweighted energy
``Σ|A|²dt`` by several percent — up to −14 % on a 12 cm N=3 deck — while the
*conserved* quantity, Blow & Wood's Appendix Eq. (A5) frequency-weighted photon
number

    Q = Σ_ω |Â(ω)|² / (ω₀ + ω) · dω ,

stays constant to ~1e-5 relative. Their abstract: "Optical energy is not
conserved by the equation, as indeed it is not conserved in stimulated Raman
scattering". The engine exposes the invariant as
``SplitStepEngine.photon_number_vs_z`` (recorded when self-steepening is on,
where the grid validation guarantees ``ω₀ + Ω > 0``) and monitors *it* in the
drift warning.

With self-steepening **off** the unweighted energy is the invariant again, and
the shock+Raman cross term is absent — the Kerr+shock symbol ``1 + Ωτ`` is
exactly ``ω/ω₀``, not a truncation.

Tests here pin the corrected claims; the numbers are measured, not assumed,
and the bounds are deliberately loose (order-of-magnitude pinning, backend
independent — scipy/pyfftw both in play on CI).
"""

import warnings

import numpy as np
import pytest

from photonics_helper.base import Length, Time, Wavelength
from photonics_helper.gnlse import Fiber, GNLSESolver, SplitStepEngine
from photonics_helper.pulse import Envelope, TemporalGrid, Wave
from photonics_helper.raman import RamanResponse, RamanSpec

WL0 = 1550e-9
OMEGA0 = Wavelength(WL0, "m").to_omega().as_rad_s
BETA2 = -21e-27  # s^2/m
GAMMA = 2.0
T0 = 50e-15 / 1.763
PEAK = 9.0 * abs(BETA2) / (GAMMA * T0**2)  # N = 3 soliton
LENGTH = 2.0 * (np.pi / 2) * T0**2 / abs(BETA2)  # ~12 cm
BETAS = np.array([BETA2 * 1e24])  # ps^2/m convention
REGIME_WARNING = "self-steepening is being run outside"


def _raman(grid: TemporalGrid) -> RamanResponse:
    spec = RamanSpec(
        name="Silica", raman_shift_cm=440.0, raman_linewidth_cm=45.0, fR=0.18
    )
    return spec.response(grid, fR=0.18, tau1=12.2e-15, tau2=32e-15)


def _engine(N: int, Tmax: float, *, shock: bool, raman: bool) -> SplitStepEngine:
    grid = TemporalGrid(N=N, Tmax=Time(Tmax, "s"))
    env = Envelope(
        shape="sech", peak_amplitude=np.sqrt(PEAK), pulse_width=Time(T0, "s")
    )
    pulse = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(WL0, "m"))
    fiber = Fiber.from_gamma(
        gamma=GAMMA,
        length=Length(LENGTH, "m"),
        raman_response=_raman(grid) if raman else None,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # derivation-regime warning: expected here
        engine = SplitStepEngine(
            pulse=pulse,
            fiber=fiber,
            betas=BETAS,
            include_raman=raman,
            include_self_steepening=shock,
            step_size=Length(LENGTH / 400, "m"),
        )
        engine.propagate(num_steps=400, nsaves=5)
    return engine


def _manual_photon_number(engine: SplitStepEngine, A: np.ndarray) -> float:
    """Independent evaluation of Eq. (A5) from a field, not via the accessor."""
    w_abs = engine.omega0 + engine.grid.w
    return float(np.sum(np.abs(engine.grid.fft(A)) ** 2 / w_abs) * engine.grid.dw)


def test_accessor_matches_manual_formula():
    """photon_number_vs_z(0) equals a hand-built A5 evaluation of A(0)."""
    eng = _engine(2048, 24e-12, shock=True, raman=True)
    A0 = np.asarray(eng.evolution[0].envelope_field, dtype=complex)
    recorded = float(eng.photon_number_vs_z[0])
    manual = _manual_photon_number(eng, A0)
    assert recorded == pytest.approx(manual, rel=1e-12)


def test_raman_shock_conserves_photon_not_energy():
    """The corrected claim: with Raman+shock the A5 invariant is conserved and
    the unweighted energy is not."""
    eng = _engine(4096, 48e-12, shock=True, raman=True)
    e = np.asarray(eng.energy_vs_z, dtype=float)
    q = np.asarray(eng.photon_number_vs_z, dtype=float)
    dE = abs(e[-1] / e[0] - 1.0)
    dQ = abs(q[-1] / q[0] - 1.0)
    # photon number: ~0.001 % measured on this deck
    assert dQ < 5e-4, f"photon drift {dQ:.2e}"
    # energy: a few percent, legitimately — magnitude is chaotic, do not pin it
    assert dE > 5e-3, f"expected a real energy change, got {dE:.2e}"


@pytest.mark.parametrize("shock,raman", [(True, False), (False, True)])
def test_control_cases_conserve_the_right_thing(shock: bool, raman: bool):
    """No cross term: with the shock OFF, energy alone is the invariant; with
    the shock on (even without Raman), the invariant is B&W's (A5) photon
    number and the unweighted energy moves by the spectral-centroid conversion
    ``ΔE/E = Δ⟨ω⟩/ω₀`` — measured +1.6 % (blue shift) on this deck,
    grid-independent, with Q conserved to ~1e-4.
    """
    eng = _engine(2048, 24e-12, shock=shock, raman=raman)
    e = np.asarray(eng.energy_vs_z, dtype=float)
    if shock:
        q = np.asarray(eng.photon_number_vs_z, dtype=float)
        assert abs(q[-1] / q[0] - 1.0) < 5e-3, "photon number is the invariant"
        assert abs(e[-1] / e[0] - 1.0) < 5e-2  # centroid conversion, order(s) %
    else:
        assert abs(e[-1] / e[0] - 1.0) < 5e-3  # energy is the invariant


def test_photon_number_not_recorded_without_shock():
    """The A5 accessor is scoped to the operator it belongs to."""
    eng = _engine(2048, 24e-12, shock=False, raman=True)
    with pytest.raises(RuntimeError, match="include_self_steepening"):
        _ = eng.photon_number_vs_z


def test_solver_passthrough():
    """GNLSESolver exposes photon_number_vs_z when shock is on."""
    grid = TemporalGrid(N=2048, Tmax=Time(24e-12, "s"))
    env = Envelope(
        shape="sech", peak_amplitude=np.sqrt(PEAK), pulse_width=Time(T0, "s")
    )
    pulse = Wave(grid=grid, envelope=env, central_wavelength=Wavelength(WL0, "m"))
    fiber = Fiber.from_gamma(
        gamma=GAMMA,
        length=Length(LENGTH, "m"),
        raman_response=_raman(grid),
    )
    solver = GNLSESolver(
        pulse=pulse,
        fiber=fiber,
        betas=BETAS,
        include_raman=True,
        include_self_steepening=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver.propagate(num_steps=100, nsaves=3)
    q = np.asarray(solver.photon_number_vs_z, dtype=float)
    assert len(q) == len(solver.z_array)
    assert abs(q[-1] / q[0] - 1.0) < 5e-4


def test_no_false_drift_warning_on_shock_run():
    """A legitimate multi-percent energy change must not trip the >5% warning:
    the warning monitors the conserved quantity."""
    with warnings.catch_warnings():
        # the derivation-regime warning is expected on this grid; allow only it
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", message=REGIME_WARNING)
        eng = _engine(4096, 24e-12, shock=True, raman=True)  # ΔE ≈ −12 %

    e = np.asarray(eng.energy_vs_z, dtype=float)
    assert abs(e[-1] / e[0] - 1.0) > 0.05  # the old code DID warn on this run

    # ...and it still warns when the conserved quantity genuinely drifts
    eng._photon_vs_z = list(eng._photon_vs_z[:-1]) + [eng._photon_vs_z[0] * 0.8]
    with pytest.warns(UserWarning, match="photon number drift"):
        eng._emit_energy_drift_warning()
