"""Static full-β(ω) operator for the scalar engines (Gap E).

The multimode engine has carried this since step 10; the scalar
``SplitStepEngine``/``GNLSESolver`` only had Taylor β and the z-dependent
``dispersion_profile``.  This file pins the ported scalar path:

* narrowband agreement with Taylor β at machine precision;
* N=1 multimode ≡ scalar on the same full-β curve;
* the ``full_beta``/``dispersion_profile`` mutual exclusion;
* ``GNLSESolver`` and ``scheme="etdrk4"`` accept the same curve.
"""

from __future__ import annotations

from math import factorial

import numpy as np
import pytest

from photonics_helper.base import Area, Length, Time, Wavelength
from photonics_helper.gnlse import Fiber, GNLSESolver, SplitStepEngine
from photonics_helper.multimode_gnlse import MultimodeSplitStepEngine
from photonics_helper.pulse import Envelope, TemporalGrid, Wave

WL = Wavelength(1550.0, "nm")


def _grid(t_ps: float = 20.0, n: int = 2**10) -> TemporalGrid:
    return TemporalGrid(N=n, Tmax=Time(t_ps * 1e-12, "s"))


def _wave(grid) -> Wave:
    return Wave(
        grid=grid,
        envelope=Envelope(
            shape="sech",
            peak_amplitude=2.0,
            pulse_width=Time(2e-12, "s"),
        ),
        central_wavelength=WL,
    )


def _fiber(length_m: float = 1.0) -> Fiber:
    return Fiber(
        n2=2.6e-20,
        alpha=0.0,
        A_eff=Area(80e-12, "m^2"),
        length=Length(length_m, "m"),
    )


def _curve(grid, betas, beta0=5.0, beta1=2.0e-12):
    """β(ω) from Taylor coefficients plus arbitrary β₀, β₁."""
    w = np.asarray(grid.w, dtype=float)
    out = beta0 + beta1 * w
    for k, b in enumerate(np.asarray(betas), start=2):
        out = out + b * 1e-12**k * w**k / factorial(k)
    return out


class TestScalarFullBeta:
    def test_matches_taylor_narrowband(self):
        grid = _grid()
        fiber = _fiber()
        betas = np.array([-21.8, 0.05])
        curve = _curve(grid, betas)
        ref = SplitStepEngine(
            _wave(grid), fiber, betas, step_size=Length(0.05, "m")
        )
        got = SplitStepEngine(
            _wave(grid),
            fiber,
            None,
            full_beta=curve,
            step_size=Length(0.05, "m"),
        )
        ref.propagate(20)
        got.propagate(20)
        rel = np.max(np.abs(ref.A - got.A)) / np.max(np.abs(ref.A))
        assert rel < 1e-10, f"full-beta vs Taylor rel err {rel:.3e}"

    def test_callable_curve_matches_array(self):
        grid = _grid()
        fiber = _fiber()
        w0 = WL.to_omega().as_rad_s

        def beta_fn(w_abs):
            w = np.asarray(w_abs) - w0
            return 3.0 + 1e-12 * w - 21.8 * 1e-24 * w**2 / 2.0

        a = SplitStepEngine(
            _wave(grid), fiber, None, full_beta=beta_fn, step_size=Length(0.05, "m")
        )
        b = SplitStepEngine(
            _wave(grid),
            fiber,
            None,
            full_beta=beta_fn(grid.w + w0),
            step_size=Length(0.05, "m"),
        )
        a.propagate(10)
        b.propagate(10)
        assert np.allclose(a.A, b.A, atol=1e-14)

    def test_multimode_n1_equivalence(self):
        grid = _grid()
        fiber = _fiber()
        curve = _curve(grid, np.array([-21.8, 0.05]))
        scalar = SplitStepEngine(
            _wave(grid), fiber, None, full_beta=curve, step_size=Length(0.05, "m")
        )
        mm = MultimodeSplitStepEngine(
            [_wave(grid)],
            fiber,
            np.array([0.0]),  # ignored when full_beta is supplied
            full_beta=[curve],
            step_size=Length(0.05, "m"),
        )
        scalar.propagate(20)
        mm.propagate(20)
        rel = np.max(np.abs(scalar.A - mm.fields_vs_z()[0][-1])) / np.max(
            np.abs(scalar.A)
        )
        assert rel < 1e-12, f"scalar vs N=1 multimode rel err {rel:.3e}"

    def test_validation_errors(self):
        grid = _grid()
        fiber = _fiber()
        curve = _curve(grid, np.array([-21.8]))
        with pytest.raises(ValueError, match="mutually exclusive"):
            SplitStepEngine(
                _wave(grid),
                fiber,
                None,
                full_beta=curve,
                dispersion_profile=lambda w, z: w * 0 + 1.0,
            )
        with pytest.raises(ValueError, match="betas or full_beta"):
            SplitStepEngine(_wave(grid), fiber, None)
        with pytest.raises(ValueError, match="match the grid"):
            SplitStepEngine(
                _wave(grid), fiber, None, full_beta=curve[:-1]
            )

    def test_etdrk4_supports_full_beta(self):
        grid = _grid()
        fiber = _fiber()
        curve = _curve(grid, np.array([-21.8]))
        eng = SplitStepEngine(
            _wave(grid),
            fiber,
            None,
            full_beta=curve,
            scheme="etdrk4",
            step_size=Length(0.01, "m"),
        )
        eng.propagate(20)
        assert np.all(np.isfinite(eng.A))
        # ETDRK4 and Strang must agree in the small-step limit.
        strang = SplitStepEngine(
            _wave(grid),
            fiber,
            None,
            full_beta=curve,
            scheme="strang",
            step_size=Length(0.01, "m"),
        )
        strang.propagate(20)
        rel = np.max(np.abs(eng.A - strang.A)) / np.max(np.abs(eng.A))
        assert rel < 0.1

    def test_solver_accepts_full_beta(self):
        grid = _grid()
        curve = _curve(grid, np.array([-21.8]))
        solver = GNLSESolver(
            _wave(grid),
            _fiber(),
            None,
            full_beta=curve,
            include_raman=False,
            step_size=Length(0.05, "m"),
        )
        solver.propagate(10)
        assert solver.evolution is not None
        assert np.all(np.isfinite(solver.evolution[-1].envelope_field))
