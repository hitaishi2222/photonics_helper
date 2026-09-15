"""Regression tests for the GNLSE beta-coefficient unit contract.

See openspec change `add-gnlse-beta-units-validation`:

* solvers interpret `betas` as ``ps^k/m`` by default (Ω in ``rad/ps``),
* ``betas_unit="s^k/m"``/``"SI"`` converts per-order at construction,
* invalid units / non-finite arrays raise clear errors.
"""

import numpy as np
import pytest

from photonics_helper.base import Area, Length, Time, Wavelength
from photonics_helper.fiber import ZDependentDispersion
from photonics_helper.gnlse import (
    FiberProfile,
    GNLSESolver,
    SplitStepEngine,
    TaperedGNLSESolver,
)
from photonics_helper.pulse import Envelope, TemporalGrid, Wave


def _setup():
    grid = TemporalGrid(N=128, Tmax=Time(20e-12, "s"))
    env = Envelope(shape="gaussian", peak_amplitude=1.0, pulse_width=Time(1, "ps"))
    pulse = Wave(
        grid=grid,
        envelope=env,
        central_wavelength=Wavelength(1550, "nm"),
    )
    fiber = FiberProfile(
        n2=1e-19, alpha=0.0, A_eff=Area(5e-11, "m^2"), length=Length(1e-3, "m")
    )
    return pulse, fiber


# ---------------------------------------------------------------------------
# 3.1 / 3.3 — equivalence and default behaviour
# ---------------------------------------------------------------------------

def test_equivalent_units_give_identical_propagation():
    """ps^k/m and equivalent s^k/m inputs propagate identically."""
    pulse, fiber = _setup()
    betas_ps = np.array([0.02, 1e-4])  # ps²/m, ps³/m
    betas_si = betas_ps * np.array([10.0**-24, 10.0**-36])  # s²/m, s³/m

    solver_ps = GNLSESolver(
        pulse=pulse, fiber=fiber, betas=betas_ps, include_raman=False
    )
    solver_si = GNLSESolver(
        pulse=pulse,
        fiber=fiber,
        betas=betas_si,
        include_raman=False,
        betas_unit="s^k/m",
    )

    assert np.allclose(solver_ps.betas, solver_si.betas, rtol=1e-12, atol=1e-15)

    solver_ps.propagate(num_steps=5)
    solver_si.propagate(num_steps=5)
    _, spectra_ps = solver_ps.spectra_vs_z
    _, spectra_si = solver_si.spectra_vs_z
    assert np.allclose(spectra_ps, spectra_si, rtol=1e-10, atol=1e-12)


def test_default_unit_is_backwards_compatible():
    """Omitting betas_unit treats coefficients as ps^k/m (unchanged)."""
    pulse, fiber = _setup()
    betas = np.array([0.02, 1e-4])
    solver = GNLSESolver(pulse=pulse, fiber=fiber, betas=betas, include_raman=False)
    assert np.array_equal(solver.betas, betas)
    assert solver.betas_unit == "ps^k/m"


# ---------------------------------------------------------------------------
# 3.2 — per-order conversion factor
# ---------------------------------------------------------------------------

def test_si_conversion_is_order_correct():
    """Order-2 SI coefficient 5e-28 s²/m becomes 5e-4 ps²/m."""
    pulse, fiber = _setup()
    solver = GNLSESolver(
        pulse=pulse,
        fiber=fiber,
        betas=[5e-28],
        include_raman=False,
        betas_unit="SI",
    )
    assert solver.betas[0] == pytest.approx(5e-4, rel=1e-12)


def test_si_conversion_third_order():
    """Order-3 conversion uses 10**(12*3)."""
    pulse, fiber = _setup()
    solver = GNLSESolver(
        pulse=pulse,
        fiber=fiber,
        betas=[2e-26, 3e-36],
        include_raman=False,
        betas_unit="s^k/m",
    )
    assert solver.betas[0] == pytest.approx(0.02, rel=1e-12)
    assert solver.betas[1] == pytest.approx(3.0, rel=1e-12)


# ---------------------------------------------------------------------------
# 3.4 / 3.5 — validation
# ---------------------------------------------------------------------------

def test_invalid_unit_string_rejected():
    pulse, fiber = _setup()
    with pytest.raises(ValueError) as excinfo:
        GNLSESolver(pulse=pulse, fiber=fiber, betas=[0.02], betas_unit="nm")
    message = str(excinfo.value)
    assert "ps^k/m" in message
    assert "s^k/m" in message
    assert "SI" in message


def test_non_finite_betas_rejected():
    pulse, fiber = _setup()
    with pytest.raises(ValueError) as excinfo:
        GNLSESolver(
            pulse=pulse, fiber=fiber, betas=[0.02, np.nan, np.inf], include_raman=False
        )
    message = str(excinfo.value)
    assert "1" in message and "2" in message


def test_non_numeric_betas_rejected():
    pulse, fiber = _setup()
    with pytest.raises(TypeError):
        GNLSESolver(pulse=pulse, fiber=fiber, betas=["a", "b"], include_raman=False)


def test_non_1d_betas_rejected():
    pulse, fiber = _setup()
    with pytest.raises(ValueError):
        GNLSESolver(pulse=pulse, fiber=fiber, betas=[[0.02, 0.1]], include_raman=False)


# ---------------------------------------------------------------------------
# 3.6 — all three constructors accept the flag
# ---------------------------------------------------------------------------

def test_splitstep_engine_accepts_betas_unit():
    pulse, fiber = _setup()
    engine = SplitStepEngine(
        pulse=pulse,
        fiber=fiber,
        betas=[5e-28],
        betas_unit="s^k/m",
        include_raman=False,
    )
    assert engine.betas[0] == pytest.approx(5e-4, rel=1e-12)


def test_gnlsesolver_accepts_betas_unit():
    pulse, fiber = _setup()
    solver = GNLSESolver(
        pulse=pulse, fiber=fiber, betas=[5e-28], betas_unit="SI", include_raman=False
    )
    assert solver.betas[0] == pytest.approx(5e-4, rel=1e-12)


def test_tapered_solver_accepts_betas_unit():
    pulse, fiber = _setup()
    omega0 = pulse.central_frequency
    omegas = np.linspace(omega0 - 6e13, omega0 + 6e13, 50)
    z_positions = np.linspace(0, 1e-3, 5)
    beta = np.zeros((len(omegas), len(z_positions)))
    for j in range(len(z_positions)):
        beta[:, j] = 1e8 + 0.5 * 0.02e-24 * (omegas - omega0) ** 2
    profile = ZDependentDispersion.from_arrays(
        omegas=omegas,
        z_positions=z_positions,
        beta=beta,
        central_wavelength=pulse.central_wavelength.as_m,
    )
    solver = TaperedGNLSESolver(
        pulse=pulse,
        fiber=fiber,
        dispersion_profile=profile,
        include_raman=False,
        betas_unit="s^k/m",
    )
    assert solver.betas_unit == "s^k/m"

    with pytest.raises(ValueError):
        TaperedGNLSESolver(
            pulse=pulse,
            fiber=fiber,
            dispersion_profile=profile,
            include_raman=False,
            betas_unit="bogus",
        )
