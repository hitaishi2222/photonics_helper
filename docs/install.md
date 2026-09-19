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

## Development install

```bash
git clone https://github.com/hitaishi2222/photonics_helper
cd photonics_helper
pip install -e ".[dev,docs]"
```
