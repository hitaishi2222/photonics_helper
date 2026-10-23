---
quadrant: how-to
---

# Install

```bash
pip install photonics-helper
```

Optional extras:

| Extra | What it adds |
|---|---|
| `plotting` | Plotly interactive figures |
| `webapp` | Dash dashboards |
| `fftw` | FFTW3 FFT backend (`pyfftw`) |
| `extras` | `imageio` (GIF/video export) |
| `mode-export` | `femwell` + `tidy3d` for generating waveguide mode exports |
| `docs` | docs toolchain (zensical + mkdocstrings) |
| `all` | everything above |

For the **FFTW3-accelerated GNLSE / Raman solver** (recommended for large
grids and long propagation runs):

```bash
pip install "photonics-helper[fftw]"
```

When `pyfftw` is present, every FFT in `gnlse.py` and `raman.py` executes on
the system FFTW3 library with cached plans; otherwise the solver transparently
falls back to `numpy.fft`. See [`fft backends`](api/fftw.md) and the env vars
`PHOTONICS_FFTW_PLANNER`, `PHOTONICS_FFTW_THREADS`, `PHOTONICS_FFT_BACKEND`.

## Lazy imports (Python 3.15+)

The package is lazy by construction (PEP 562): `import photonics_helper` binds
almost nothing, and each public name is imported from its defining submodule the
first time it is accessed, so the plotting/web stack is never pulled in by a
bare import.

That laziness covers attribute access (`photonics_helper.Wavelength`). A
`from photonics_helper import …` statement with a name you may never use still
imports that name's submodule eagerly. On **Python 3.15+** you can defer it too
with [PEP 810](https://docs.python.org/3.15/reference/simple_stmts.html#lazy-imports):

```text
lazy from photonics_helper import Wavelength  # numpy/scipy only on first use

w = Wavelength(1550.0, "nm")  # reifies here
```

This is Python 3.15 syntax, so it is shown as text rather than run.

or enable it for the whole program without changing source:

```bash
python -X lazy_imports=all your_script.py
```

```text
import sys

sys.set_lazy_imports("all")  # equivalent, at runtime
```

Also 3.15 and later. The environment variable `PYTHON_LAZY_IMPORTS=all` does the
same thing on 3.15 without editing source.

Before 3.15 the `lazy` keyword is a `SyntaxError`. Use the attribute form
(`import photonics_helper as ph` then `ph.Wavelength`), or a `from … import …`
under `if TYPE_CHECKING:` for annotation-only imports.

The guard `tests/test_lazy_import.py::test_pep810_lazy_from_import_defers_submodule`
is skipped below 3.15 and activates automatically on a 3.15 interpreter.

## Development install

```bash
git clone https://github.com/hitaishi2222/photonics_helper
cd photonics_helper
pip install -e ".[dev,docs]"
```
