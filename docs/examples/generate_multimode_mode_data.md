---
quadrant: reference
---

# Generate a synthetic N-mode export for the step7 bridge

`examples/generate_multimode_mode_data.py` (89 lines)

This script produces no figure. It prints its result, launches a server, or writes data for another example.

## Source

```python
#!/usr/bin/env python3
"""Generate a synthetic N-mode export for the step7 bridge.

Sibling to ``generate_waveguide_mode_data.py`` (which is hard-wired to
``num_modes=1``): writes ``mode-*.npz`` + ``meta.json`` into
``examples/data/multimode_rib/`` for one rib geometry over 1.2–6 µm with
transverse vector fields per mode:

    examples/data/multimode_rib/mode-00.npz, mode-01.npz, mode-02.npz
    examples/data/multimode_rib/meta.json

Fields are analytic Hermite–Gaussian LP01 / LP11a / LP11b shapes on a
Cartesian grid (no circular-symmetry assumption in the consumer), and
``n_eff(λ)`` tables carry a realistic normal-dispersion slope plus a small
per-mode offset. Swap these writers for real femwell/tidy3d field dumps to
drive physical decks — the file format is the contract
(``wavelength_um``, ``neff``, ``ng``, ``Ex``/``Ey`` ``(n_wl, Ny, Nx)``,
``x_um``/``y_um``, ``label``).

Run with: python examples/generate_multimode_mode_data.py
"""

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "multimode_rib"

WAVELENGTHS_UM = np.linspace(1.2, 6.0, 25)
X_UM = np.linspace(-2.0, 2.0, 48)
Y_UM = np.linspace(-2.0, 2.0, 48)
W0_UM = 0.9  # mode-field radius


def _hg_fields() -> list[np.ndarray]:
    xx, yy = np.meshgrid(X_UM, Y_UM)
    g = np.exp(-(xx**2 + yy**2) / W0_UM**2)
    lp01 = g
    lp11a = (xx / W0_UM) * g
    lp11b = (yy / W0_UM) * g
    return [lp01, lp11a, lp11b]


def _neff_curves() -> list[np.ndarray]:
    base = 2.45 - 0.045 * (WAVELENGTHS_UM - 1.55)
    offsets = [0.0, -0.012, -0.013]
    return [base + o for o in offsets]


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    shapes = _hg_fields()
    curves = _neff_curves()
    labels = [0, 1, -1]
    for i, (shape, neff, lab) in enumerate(zip(shapes, curves, labels)):
        ng = neff - WAVELENGTHS_UM * np.gradient(neff, WAVELENGTHS_UM)
        Ex = np.asarray([shape for _ in WAVELENGTHS_UM], dtype=complex)
        Ey = np.zeros_like(Ex)
        np.savez(
            DATA / f"mode-{i:02d}.npz",
            wavelength_um=WAVELENGTHS_UM,
            neff=neff,
            ng=ng,
            Ex=Ex,
            Ey=Ey,
            x_um=X_UM,
            y_um=Y_UM,
            label=np.asarray(lab),
        )
        print(f"  wrote mode-{i:02d}.npz (label {lab:+d})")
    meta = {
        "geometry": "synthetic 1.5x0.8 um high-contrast rib stand-in",
        "wavelengths_um": WAVELENGTHS_UM.tolist(),
        "n_modes": 3,
        "note": "analytic HG fields; replace with femwell/tidy3d dumps",
    }
    (DATA / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  wrote {DATA.relative_to(_ROOT)}/meta.json")


if __name__ == "__main__":
    main()
```
