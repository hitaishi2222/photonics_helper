"""Tests for step10 full-β + full-tensor FWM (tasks 1.2/2.3)."""

from math import factorial

import numpy as np

from photonics_helper.base import Area, Length, Time, Wavelength
from photonics_helper.multimode_gnlse import MultimodeSplitStepEngine
from photonics_helper.pulse import Envelope, TemporalGrid, Wave

WL = Wavelength(1550.0, "nm")


def _grid(t_ps: float = 20.0, n: int = 2**11) -> TemporalGrid:
    return TemporalGrid(N=n, Tmax=Time(t_ps * 1e-12, "s"))


def _sech_wave(grid, peak, width_ps: float = 2.0):
    env = Envelope(
        shape="sech",
        peak_amplitude=np.sqrt(peak),
        pulse_width=Time(width_ps * 1e-12, "s"),
    )
    return Wave(grid=grid, envelope=env, central_wavelength=WL)


def _fiber(length_m: float):
    from photonics_helper.gnlse import Fiber

    return Fiber(
        n2=2.6e-20,
        alpha=0.0,
        A_eff=Area(80e-12, "m^2"),
        length=Length(length_m, "m"),
    )


def _taylor_beta(betas_ps, w_si: np.ndarray) -> np.ndarray:
    """Taylor β(ω) in rad/m from ps^k/m coefficients (Ω in rad/s)."""
    out = np.zeros_like(w_si, dtype=float)
    for k, b in enumerate(betas_ps, start=2):
        out += b * 1e-12**k * w_si**k / factorial(k)
    return out


def test_full_beta_matches_taylor_narrowband():
    """Full-β vs high-order Taylor agree in the narrowband limit (1.2)."""
    grid = _grid()
    fiber = _fiber(2.0)
    b0 = np.array([-21.8, 0.05])
    b1 = np.array([-20.0, 0.04])
    gd = [0.0, 2e-12]
    w = np.asarray(grid.w, dtype=float)
    full = [
        _taylor_beta(b0, w) + 1.0 * gd[0] * w,
        _taylor_beta(b1, w) + 100.0 + 1.0 * gd[1] * w,
    ]
    ref = MultimodeSplitStepEngine(
        [_sech_wave(grid, 4.0), _sech_wave(grid, 1.0)],
        fiber,
        [b0, b1],
        group_delays=gd,
        phase_offsets=[0.0, 100.0],
        step_size=Length(0.1, "m"),
    )
    ref.propagate(10)
    got_eng = MultimodeSplitStepEngine(
        [_sech_wave(grid, 4.0), _sech_wave(grid, 1.0)],
        fiber,
        [b0, b1],
        full_beta=full,
        step_size=Length(0.1, "m"),
    )
    got_eng.propagate(10)
    for m in range(2):
        a = ref.fields_vs_z()[m][-1, :]
        b = got_eng.fields_vs_z()[m][-1, :]
        assert np.max(np.abs(a - b)) / np.max(np.abs(a)) < 1e-6


def test_full_tensor_selection_rules():
    """Forbidden triplets unmixed to machine precision (2.3)."""
    grid = _grid()
    fiber = _fiber(1.0)
    betas = [np.array([-21.8])] * 3
    fw = np.ones((3, 3, 3, 3))  # nonzero even where forbidden
    mm = MultimodeSplitStepEngine(
        [
            _sech_wave(grid, 4.0),
            _sech_wave(grid, 0.0 + 1e-30),
            _sech_wave(grid, 0.0 + 1e-30),
        ],
        fiber,
        betas,
        include_fwm=True,
        oam_l=[0, 1, 3],
        fwm_weights=fw,
        fwm_full_tensor=True,
        step_size=Length(0.05, "m"),
    )
    mm.propagate(10)
    trajs = mm.fields_vs_z()
    # Channel pair (1,2): ℓ mismatch forbids every triplet feeding them.
    assert float(np.max(np.abs(trajs[1][-1]))) < 1e-12
    assert float(np.max(np.abs(trajs[2][-1]))) < 1e-12


def test_degenerate_vs_full_verification():
    """Phase 3/4 check (2.2): degenerate-only weights → identical arms."""
    grid = _grid()
    fiber = _fiber(1.0)
    betas = [np.array([-21.8])] * 3
    # Exactly the (m,n,n,q)/(q,n,n,m) entries the degenerate slice touches.
    fw = np.zeros((3, 3, 3, 3))
    for m, n, q in [
        (1, 0, 2),
        (2, 0, 1),
        (0, 1, 2),
        (2, 1, 0),
        (0, 2, 1),
        (1, 2, 0),
    ]:
        fw[m, n, n, q] = 1.0
        fw[q, n, n, m] = 1.0
    kw = dict(
        include_fwm=True,
        oam_l=[0, 0, 0],
        fwm_weights=fw,
        step_size=Length(0.05, "m"),
    )
    waves = [_sech_wave(grid, 4.0), _sech_wave(grid, 0.5), _sech_wave(grid, 0.25)]
    a = MultimodeSplitStepEngine(list(waves), fiber, betas, **kw)
    a.propagate(10)
    b = MultimodeSplitStepEngine(
        [_sech_wave(grid, 4.0), _sech_wave(grid, 0.5), _sech_wave(grid, 0.25)],
        fiber,
        betas,
        fwm_full_tensor=True,
        **kw,
    )
    b.propagate(10)
    for m in range(3):
        assert (
            np.max(np.abs(a.fields_vs_z()[m][-1, :] - b.fields_vs_z()[m][-1, :])) < 1e-9
        )
    # Off-slice entry: full contraction sees arms the slice misses.
    fw2 = fw.copy()
    fw2[0, 1, 2, 0] = 5.0
    c = MultimodeSplitStepEngine(
        [_sech_wave(grid, 4.0), _sech_wave(grid, 0.5), _sech_wave(grid, 0.25)],
        fiber,
        betas,
        include_fwm=True,
        oam_l=[0, 0, 0],
        fwm_weights=fw2,
        fwm_full_tensor=True,
        step_size=Length(0.05, "m"),
    )
    c.propagate(10)
    diff = np.max(np.abs(a.fields_vs_z()[0][-1, :] - c.fields_vs_z()[0][-1, :]))
    assert diff > 1e-9


def test_shock_applied_once_with_full_tensor_fwm():
    """Shock+FWM: the shock operator acts once on the assembled sum.

    Regression guard: _fwm_rhs must NOT pre-apply (1 + iτΩ) to the FWM arm
    — _nonlinear_rhs_tensor applies it once to the whole nonlinear RHS, so
    a double application would give (1 + iτΩ)² on the FWM polarization.
    """
    grid = _grid()
    fiber = _fiber(1.0)
    with __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        mm = MultimodeSplitStepEngine(
            [
                _sech_wave(grid, 4.0),
                _sech_wave(grid, 1.0),
                _sech_wave(grid, 0.25),
            ],
            fiber,
            [np.array([-21.8])] * 3,
            include_fwm=True,
            fwm_full_tensor=True,
            include_shock=True,
            oam_l=[0, 0, 0],
            step_size=Length(0.05, "m"),
        )
    A = list(mm.A)
    tau = float(mm.tau_shock)

    def shock(arms):
        return [
            arm + tau * mm.grid.ifft(1j * mm.grid.w * mm.grid.fft(arm))
            for arm in arms
        ]

    diag = mm._diagonal_phase(A)
    base = [1j * d * Ai for d, Ai in zip(diag, A)]
    # Raw FWM arm reassembled from the same engine contract (no shock).
    fwm_raw = [np.zeros_like(A[0], dtype=complex) for _ in range(3)]
    for mm_ in range(3):
        for n in range(3):
            for p in range(3):
                for q in range(3):
                    if (n == mm_ and q == p) or (p == mm_ and q == n):
                        continue
                    if not mm._fwm_allowed(mm_, n, p, q):
                        continue
                    c = mm._fwm_coupling(mm_, n, p, q)
                    if c == 0:
                        continue
                    fwm_raw[mm_] = fwm_raw[mm_] + 1j * c * A[n] * A[p] * np.conj(A[q])
    tot = [b + f for b, f in zip(base, fwm_raw)]
    rhs = mm._nonlinear_rhs_tensor(A)
    once = shock(tot)
    twice = shock(shock(tot))
    e_once = max(
        float(np.max(np.abs(o - r))) for o, r in zip(once, rhs)
    )
    e_twice = max(
        float(np.max(np.abs(t - r))) for t, r in zip(twice, rhs)
    )
    assert e_once == 0.0
    assert e_twice > 100.0 * e_once


def test_octave_deck_stable_fine_dt():
    """Octave deck stable at dt ≤ 1 fs (2.3)."""
    grid = TemporalGrid(N=2**12, Tmax=Time(2.048e-12, "s"))
    assert float(grid.dt) <= 1e-15
    fiber = _fiber(0.2)
    w = np.asarray(grid.w, dtype=float)
    full = [_taylor_beta(np.array([-21.8]), w)] * 2
    mm = MultimodeSplitStepEngine(
        [_sech_wave(grid, 4.0, 0.2), _sech_wave(grid, 1.0, 0.2)],
        fiber,
        [np.array([-21.8]), np.array([-21.8])],
        full_beta=full,
        include_fwm=True,
        oam_l=[0, 0],
        fwm_full_tensor=True,
        step_size=Length(0.01, "m"),
    )
    mm.propagate(10)
    for m in range(2):
        fld = mm.fields_vs_z()[m][-1, :]
        assert np.all(np.isfinite(fld))
    e0 = sum(float(np.sum(np.abs(t[0]) ** 2)) for t in mm.fields_vs_z())
    e1 = sum(float(np.sum(np.abs(t[-1]) ** 2)) for t in mm.fields_vs_z())
    assert abs(e1 - e0) / e0 < 0.1
