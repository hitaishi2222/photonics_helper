"""Regression tests for the multimode FWM substep integrator (ISSUES.md #11).

Historical bug: with ``include_fwm=True, fwm_pump_depletion=True``, a CW pump
(γP₀ = 4.7 /m) and a single seeded sideband channel (no idler partner) grew
the idler to |A| ~ 1e240 → NaN because ``_fwm_substep_count`` silently
capped the explicit RK4 inner steps at 200, leaving h·λ outside RK4's
stability region for the amplified exchange pair.
"""

import numpy as np
import pytest

from photonics_helper.base import Area, Length, Time, Wavelength
from photonics_helper.gnlse import FiberProfile
from photonics_helper.multimode_gnlse import MultimodeSplitStepEngine
from photonics_helper.pulse import Envelope, TemporalGrid, Wave

N_GRID = 1024
WINDOW_PS = 40.0
LAMBDA0_NM = 532.0
P0_W = 1500.0
GAMMA = 0.00313  # /W/m → γP₀ = 4.7 /m


def _waves(seed_fraction: float = 1e-4, with_idler: bool = False):
    grid = TemporalGrid(N=N_GRID, Tmax=Time(WINDOW_PS * 1e-12, "s"))
    a_pump = float(np.sqrt(P0_W))
    waves = []
    for ch in range(3):
        wave = Wave(
            grid=grid,
            envelope=Envelope(
                shape="gaussian",
                peak_amplitude=a_pump,
                pulse_width=Time(WINDOW_PS * 1e-12 / 4.0, "s"),
            ),
            central_wavelength=Wavelength(LAMBDA0_NM, "nm"),
        )
        if ch == 0:
            field = np.full(N_GRID, a_pump, dtype=complex)
        elif ch == 1:
            field = seed_fraction * a_pump * np.exp(1j * 2.0 * np.pi * 0.15e12 * grid.t)
        else:
            if with_idler:
                field = (
                    seed_fraction
                    * a_pump
                    * np.conj(np.exp(1j * 2.0 * np.pi * 0.15e12 * grid.t))
                ).astype(complex)
            else:
                field = np.zeros(N_GRID, dtype=complex)
        waves.append(wave.with_field(field))
    return waves


def _engine(waves, length_m=1.0, step_m=2.5e-3):
    fiber = FiberProfile(
        n2=2.6e-20, alpha=0.0, A_eff=Area(50e-12, "m^2"), length=Length(length_m, "m")
    )
    return MultimodeSplitStepEngine(
        waves,
        fiber,
        betas=[[0.0, 0.0, 0.0, 0.0]] * 3,
        include_fwm=True,
        fwm_pump_depletion=True,
        step_size=Length(step_m, "m"),
    )


def test_fwm_stable_without_idler_partner():
    """The ISSUES.md #11 crash config must stay physical (no 1e240 blow-up)."""
    eng = _engine(_waves())
    eng.propagate(int(1.0 / 2.5e-3), nsaves=3)
    a_idler = np.asarray(eng.A[2], dtype=complex)
    a_signal = np.asarray(eng.A[1], dtype=complex)
    seed = 1e-4 * np.sqrt(P0_W)
    assert np.all(np.isfinite(np.asarray(eng.A[0])))
    # Parametric 2x2 growth: |idler| ~ sinh(rate·L)·seed with rate ≲ γP₀.
    # The old bug produced |idler|/seed ~ 1e244; this asserts orders of
    # magnitude below that while tolerating the exact sinh level.
    rate_pct = np.max(np.abs(a_idler)) / seed
    assert rate_pct < 3e2, f"idler/seed = {rate_pct:.1e} exceeds the sinh(γP₀·L) level"
    assert np.max(np.abs(a_signal)) < 3e2 * seed


def test_fwm_substep_count_is_stability_driven():
    """The substep count must grow with rate·dz instead of silently capping.

    With γP₀ = 4.7 /m and dz' = 2.5 mm the stability requirement is trivial,
    so the count must stay ≥ 1 and small; with a 40× overdriven pump the old
    code would return the capped 200 while leaving h·λ > 2.5.
    """
    grid = TemporalGrid(N=256, Tmax=Time(10e-12, "s"))
    pump = np.full(N_GRID, 40.0 * np.sqrt(P0_W), dtype=complex)
    sideband = np.zeros(N_GRID, dtype=complex)
    eng = _engine(
        [
            _mk_wave(grid, pump),
            _mk_wave(grid, np.full(N_GRID, 1.0, complex)),
            _mk_wave(grid, sideband),
        ]
    )
    n = eng._fwm_substep_count(
        [pump, np.full(N_GRID, 1e-3, complex), np.zeros(N_GRID, complex)],
        dz=2.5e-3,
    )
    # Stability: n ≥ rate·dz / 2.5 with rate ≈ γ f |A|² (f = 2/3):
    rate = 0.00313 * (2.0 / 3.0) * (40.0**2 * P0_W)
    n_req = int(np.ceil(rate * 2.5e-3 / 2.5))
    assert n >= max(n_req, 1)
    assert rate * 2.5e-3 / n <= 2.5


def _mk_wave(grid, field):
    if not isinstance(field, np.ndarray):
        field = np.asarray(field, dtype=complex)
    return Wave(
        grid=grid,
        envelope=Envelope(
            shape="gaussian",
            peak_amplitude=1.0,
            pulse_width=Time(WINDOW_PS * 1e-12 / 4.0, "s"),
        ),
        central_wavelength=Wavelength(LAMBDA0_NM, "nm"),
    ).with_field(field)


def test_fwm_huge_rate_raises_loudly():
    """A configuration outside RK4 stability must raise, naming the rate."""
    grid = TemporalGrid(N=256, Tmax=Time(2e-12, "s"))
    tiny = np.zeros(256, dtype=complex)
    eng = _engine(
        [
            _mk_wave(grid, np.full(256, 1e6, dtype=complex)),
            _mk_wave(grid, np.full(256, 1e-6, dtype=complex)),
            _mk_wave(grid, tiny),
        ],
        length_m=100.0,
        step_m=50e-3,
    )
    with pytest.raises(ValueError, match="RK4 stability|ISSUES.md #11|/m"):
        eng._fwm_substep_count(
            [np.full(256, 1e6, dtype=complex), np.full(256, 1e-6, dtype=complex), tiny],
            dz=50e-3,
        )
