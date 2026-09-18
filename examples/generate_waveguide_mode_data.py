#!/usr/bin/env python3
"""Generate real waveguide ``n_eff(λ)`` export data with femwell and Tidy3D.

This is the *producer* half of the FEM-import workflow. It solves the same
canonical silicon strip waveguide (500 nm × 220 nm Si on SiO₂) with two
independent eigenmode solvers and writes their ``n_eff(λ)`` tables in the
``WaveguideMode`` file conventions:

    examples/data/si_strip_femwell.csv / .npz
    examples/data/si_strip_tidy3d.csv / .npz
    examples/data/si_strip_meta.json

The committed outputs let ``examples/31_waveguide_mode_import_femwell_tidy3d.py``
run without either solver installed. Re-run this script to regenerate them.

Requirements (optional, heavy)
------------------------------
    pip install femwell tidy3d

femwell needs ``gmsh`` (bundled with the wheel); Tidy3D's mode solver runs
**locally** and needs no API key for this script.

Run with: python examples/generate_waveguide_mode_data.py
"""

# The sys.path bootstrap below intentionally precedes the library imports.
# ruff: noqa: E402

import json
import sys
import tempfile
import time
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# ── Canonical geometry / materials ──────────────────────────────────────
WAVELENGTHS_UM = np.linspace(1.50, 1.60, 21)
N_SI = 3.4777
N_SIO2 = 1.444
W_CORE_UM = 0.5
H_CORE_UM = 0.22


def _group_index(wl_um: np.ndarray, neff: np.ndarray) -> np.ndarray:
    """ng = n_eff − λ·dn_eff/dλ from the tabulated curve (same units cancel)."""
    return neff - wl_um * np.gradient(neff, wl_um)


def _write_export(stem: str, wl_um: np.ndarray, neff: np.ndarray, ng: np.ndarray) -> None:
    """Write one solver's table as CSV (with header) and NPZ."""
    DATA.mkdir(parents=True, exist_ok=True)
    csv_path = DATA / f"{stem}.csv"
    with csv_path.open("w") as fh:
        fh.write("wavelength_um, neff, ng\n")
        for x, y, g in zip(wl_um, neff, ng):
            fh.write(f"{x:.6f}, {y:.10f}, {g:.10f}\n")
    np.savez(
        DATA / f"{stem}.npz",
        wavelength_um=wl_um,
        neff=neff,
        ng=ng,
        central_wavelength_nm=1550.0,
    )
    print(f"  wrote {csv_path.relative_to(_ROOT)} and {stem}.npz")


def run_femwell() -> tuple[np.ndarray, np.ndarray]:
    """Solve the strip with femwell (scikit-fem + Gmsh), mode 0 per λ."""
    from skfem import Basis, ElementTriP0, Mesh

    from femwell.maxwell.waveguide import compute_modes
    from femwell.waveguide import mesh_waveguide

    warnings.filterwarnings("ignore", category=np.exceptions.ComplexWarning)

    with tempfile.TemporaryDirectory() as tmp:
        mesh_file = f"{tmp}/si_strip.msh"
        mesh_waveguide(
            wsim=3.0,
            hclad=1.0,
            hbox=1.0,
            wcore=W_CORE_UM,
            hcore=H_CORE_UM,
            filename=mesh_file,
        )
        mesh = Mesh.load(mesh_file)

    basis0 = Basis(mesh, ElementTriP0(), intorder=4)
    epsilon = basis0.zeros()
    epsilon[basis0.get_dofs(elements="core")] = N_SI**2
    epsilon[basis0.get_dofs(elements="clad")] = N_SIO2**2
    epsilon[basis0.get_dofs(elements="box")] = N_SIO2**2

    neff = np.empty_like(WAVELENGTHS_UM)
    for i, wl in enumerate(WAVELENGTHS_UM):
        modes = compute_modes(basis0, epsilon, wavelength=float(wl), num_modes=1)
        neff[i] = float(modes[0].n_eff)
    return WAVELENGTHS_UM, neff


def run_tidy3d() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the strip with Tidy3D's local mode solver; returns (wl, neff, ng)."""
    import tidy3d as td
    from tidy3d.plugins.mode import ModeSolver

    # Tidy3D 2.12 moved the logging knob; keep both paths quiet.
    try:
        td.config.logging.level = "ERROR"
    except AttributeError:  # pragma: no cover - older Tidy3D
        td.config.logging_level = "ERROR"

    si = td.Medium(permittivity=N_SI**2)
    sio2 = td.Medium(permittivity=N_SIO2**2)
    # Propagation along x → infinite in x, finite width (y) and height (z).
    waveguide = td.Structure(
        geometry=td.Box(center=(0, 0, 0), size=(td.inf, W_CORE_UM, H_CORE_UM)),
        medium=si,
    )
    sim = td.Simulation(
        size=(2.0, 3.0, 2.5),
        structures=[waveguide],
        medium=sio2,
        run_time=1e-12,
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=20, wavelength=1.55),
    )
    plane = td.Box(center=(0, 0, 0), size=(0, 3.0, 2.5))
    solver = ModeSolver(
        simulation=sim,
        plane=plane,
        mode_spec=td.ModeSpec(num_modes=1, group_index_step=True),
        freqs=td.C_0 / WAVELENGTHS_UM,
    )
    data = solver.solve()
    neff = np.asarray(data.n_eff.values, dtype=float).ravel()
    ng = np.asarray(data.n_group.values, dtype=float).ravel()
    return WAVELENGTHS_UM, neff, ng


def main() -> None:
    print("Generating waveguide n_eff(λ) export data")
    print(f"  geometry: Si {W_CORE_UM}×{H_CORE_UM} µm strip, n_Si={N_SI}, n_SiO2={N_SIO2}")
    print(f"  grid: {len(WAVELENGTHS_UM)} wavelengths, {WAVELENGTHS_UM[0]}–{WAVELENGTHS_UM[-1]} µm\n")

    print("femwell:")
    t0 = time.time()
    wl, neff_fw = run_femwell()
    ng_fw = _group_index(wl, neff_fw)
    _write_export("si_strip_femwell", wl, neff_fw, ng_fw)
    print(f"  n_eff(1550 nm) = {np.interp(1.55, wl, neff_fw):.6f}  ({time.time() - t0:.1f}s)\n")

    print("tidy3d:")
    t0 = time.time()
    _, neff_td, ng_td = run_tidy3d()
    _write_export("si_strip_tidy3d", wl, neff_td, ng_td)
    print(f"  n_eff(1550 nm) = {np.interp(1.55, wl, neff_td):.6f}  ({time.time() - t0:.1f}s)\n")

    max_dev = float(np.max(np.abs(neff_fw - neff_td) / neff_td))
    meta = {
        "geometry": {
            "type": "silicon strip on oxide",
            "core_width_um": W_CORE_UM,
            "core_height_um": H_CORE_UM,
            "n_Si": N_SI,
            "n_SiO2": N_SIO2,
        },
        "wavelengths_um": WAVELENGTHS_UM.tolist(),
        "solvers": {
            "femwell": {"mode": 0, "ng_source": "numerical d n_eff/dλ"},
            "tidy3d": {"mode": 0, "ng_source": "group_index_step"},
        },
        "n_eff_1550nm": {
            "femwell": float(np.interp(1.55, wl, neff_fw)),
            "tidy3d": float(np.interp(1.55, wl, neff_td)),
        },
        "max_relative_deviation": max_dev,
    }
    (DATA / "si_strip_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  max |n_eff_femwell − n_eff_tidy3d| / n_eff = {max_dev:.2%}")
    print(f"  wrote {(DATA / 'si_strip_meta.json').relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
