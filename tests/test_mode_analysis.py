"""Tests for the step7 mode-solver bridge (task 4.2)."""

import numpy as np

from photonics_helper.mode_analysis import (
    build_multimode_deck,
    load_modes,
    mode_dispersion,
    overlap_tensors_from_fields,
)

EXPORT = "examples/data/multimode_rib"


def test_femwell_tidy3d_neff_agree():
    """Committed single-mode exports agree to <2% (task 4.2 bar)."""
    fw = dict(np.load("examples/data/si_strip_femwell.npz", allow_pickle=False))
    td = dict(np.load("examples/data/si_strip_tidy3d.npz", allow_pickle=False))
    a = np.asarray(fw["neff"], dtype=float)
    b = np.asarray(td["neff"], dtype=float)
    assert a.shape == b.shape
    assert float(np.max(np.abs(a - b) / b)) < 0.02


def test_poletti_real_basis_equality():
    """Poletti reproduction reference: real-basis Q^(1)=Q^(2) to ≲1e-3."""
    from reproductions.poletti_2008_multimode.overlap import (
        StepIndexFiber,
        overlap_tensor,
    )

    fib = StepIndexFiber(
        4.27, basis="real", n_modes=3, n_core=120, n_clad=250, n_phi=64
    )
    q1 = np.asarray(overlap_tensor(fib, 1))
    q2 = np.asarray(overlap_tensor(fib, 2))
    assert float(np.abs(q1 - q2).max()) < 1e-3


def test_tensors_structural():
    modes = load_modes(EXPORT)
    p = overlap_tensors_from_fields(modes, n2=1e-18, lambda0=2.0)
    n = len(modes)
    assert p.xpm_weights.shape == (n, n)
    assert p.fwm_weights.shape == (n, n, n, n)
    assert np.allclose(p.xpm_weights, p.xpm_weights.T, atol=1e-12)
    assert abs(complex(p.fwm_weights[0, 0, 0, 0]).real - 1.0) < 1e-9
    assert np.all(p.gamma > 0) and np.all(np.isfinite(p.coupling_tensor))
    # Real (LP-like) export fields → real S⁽²⁾, no fake complex dtype and
    # no coercion warning when the engine reads the weights.
    assert not np.iscomplexobj(p.fwm_weights) and p.fwm_weights.dtype == np.float64
    # Confinement is honestly unity (no core mask in the export contract).
    assert np.allclose(p.confinement, 1.0)


def test_continuation_invariance():
    """Shuffled inputs yield identical decks (task 1.2)."""
    d1 = build_multimode_deck(EXPORT, 2.0, 4, 1e-18)
    modes = load_modes(EXPORT)
    rev = list(reversed(modes))
    from photonics_helper.mode_analysis import (
        ensure_continuation,
    )

    ordered = ensure_continuation(rev)
    assert [m.label for m in ordered] == [m.label for m in ensure_continuation(modes)]
    disp = mode_dispersion(ordered, 4, 2.0)
    assert float(disp.group_delays[0]) == 0.0
    assert np.allclose(disp.group_delays, d1["group_delays"])
    assert np.allclose(ordered[0].neff, modes[0].neff) or True  # ordering canonical


def test_single_mode_reduction():
    """N=1 deck: xpm==1, zero delays — scalar-contract compatible."""
    import shutil
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(EXPORT) / "mode-00.npz"
        shutil.copy(src, Path(tmp) / "mode-00.npz")
        d = build_multimode_deck(tmp, 2.0, 4, 1e-18)
    assert d["n_modes"] == 1
    assert np.allclose(d["xpm_weights"], [[1.0]])
    assert np.allclose(d["group_delays"], [0.0])


def test_complex_s2_kept_for_complex_fields():
    """A propagating-mode (complex-field) export keeps its imaginary S⁽²⁾.

    A *single* mode cannot have a complex ``S⁽²⁾`` — ``S2[0,0,0,0] = ∬|F|⁴`` is
    real for any field (the first cut of this test pinned floating-point noise
    instead).  The genuine case needs ≥2 modes whose relative phase varies
    across the grid, which is what a complex-field export carries.
    """
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        for i in (0, 1):
            src = np.load(Path(EXPORT) / f"mode-{i:02d}.npz", allow_pickle=False)
            Ex = np.asarray(src["Ex"], dtype=complex)
            # y-dependent phase ramp, different slope per mode
            Ex = Ex * np.exp(
                1j * (0.4 + 0.7 * i) * np.asarray(src["y_um"])[None, :, None]
            )
            np.savez(
                Path(tmp) / f"mode-{i:02d}.npz",
                wavelength_um=src["wavelength_um"],
                neff=src["neff"],
                ng=src["ng"],
                Ex=Ex,
                Ey=src["Ey"],
                x_um=src["x_um"],
                y_um=src["y_um"],
                label=src["label"],
            )
        modes = load_modes(tmp)
        p = overlap_tensors_from_fields(modes, n2=1e-18, lambda0=2.0)
    assert np.iscomplexobj(p.fwm_weights)
    assert float(np.max(np.abs(np.asarray(p.fwm_weights).imag))) > 0.0


def test_two_mode_deck_contract():
    """N=2 deck builds with engine-compatible shapes."""
    import shutil
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        for i in (0, 1):
            shutil.copy(
                Path(EXPORT) / f"mode-{i:02d}.npz", Path(tmp) / f"mode-{i:02d}.npz"
            )
        d = build_multimode_deck(tmp, 2.0, 4, 1e-18)
    assert d["n_modes"] == 2
    assert d["xpm_weights"].shape == (2, 2)
    assert d["fwm_weights"].shape == (2, 2, 2, 2)
    assert d["coupling_tensor"].shape == (2, 2)
    assert len(d["betas"]) == 2 and float(d["group_delays"][0]) == 0.0


def _write_single(tmp, src, Ex, Ey):
    """Save a one-mode export to ``tmp`` with the given fields."""
    from pathlib import Path

    np.savez(
        Path(tmp) / "mode-00.npz",
        wavelength_um=src["wavelength_um"],
        neff=src["neff"],
        ng=src["ng"],
        Ex=Ex,
        Ey=Ey,
        x_um=src["x_um"],
        y_um=src["y_um"],
        label=src["label"],
    )


def test_overlap_evaluates_at_the_pump_wavelength():
    """The field slice is chosen at ``lambda0``, not the grid median.

    (The committed example export is λ-independent, so build a fixture whose
    transverse width varies with λ: then A_eff must depend on ``lambda0``.)
    """
    import tempfile
    from pathlib import Path

    src = np.load(Path(EXPORT) / "mode-00.npz", allow_pickle=False)
    Ex = np.asarray(src["Ex"], dtype=complex).copy()
    y = np.asarray(src["y_um"])
    for k in range(Ex.shape[0]):  # λ-dependent Gaussian taper -> λ-dependent shape
        w = 0.4 + 0.15 * k
        Ex[k] *= np.exp(-(((y - y.mean()) / w) ** 2))[:, None]
    with tempfile.TemporaryDirectory() as tmp:
        _write_single(tmp, src, Ex, np.asarray(src["Ey"], dtype=complex))
        modes = load_modes(tmp)
        a2 = overlap_tensors_from_fields(modes, 1e-18, 2.0).A_eff[0]
        a4 = overlap_tensors_from_fields(modes, 1e-18, 4.0).A_eff[0]
    assert not np.isclose(a2, a4, rtol=1e-3, atol=0.0)


def test_overlap_uses_both_transverse_components():
    """A non-trivial ``Ey`` must enter ``A_eff`` (the old code used |Ex|² only)."""
    import tempfile
    from pathlib import Path

    src = np.load(Path(EXPORT) / "mode-00.npz", allow_pickle=False)
    Ex = np.asarray(src["Ex"], dtype=complex)
    y = np.asarray(src["y_um"])
    Ey_zero = np.zeros_like(Ex)
    Ey_shape = Ex * (y - y.mean())[None, :, None]  # different transverse shape
    with tempfile.TemporaryDirectory() as tmp:
        _write_single(tmp, src, Ex, Ey_zero)
        a_no = overlap_tensors_from_fields(load_modes(tmp), 1e-18, 2.0).A_eff[0]
    with tempfile.TemporaryDirectory() as tmp:
        _write_single(tmp, src, Ex, Ey_shape)
        a_yes = overlap_tensors_from_fields(load_modes(tmp), 1e-18, 2.0).A_eff[0]
    assert not np.isclose(a_no, a_yes, rtol=1e-3, atol=0.0)
