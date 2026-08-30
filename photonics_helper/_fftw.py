"""FFT backend for photonics_helper with automatic fallback chain.

Every FFT hotspot in :mod:`photonics_helper.gnlse` and :mod:`photonics_helper.raman`
routes through this module. The best available backend is picked automatically
so the package works everywhere:

1. **FFTW3** (``pyfftw``) — fastest; cached plans (one per grid size, reused
   across every split-step iteration) and optional multi-threading. Requires
   the optional ``pyfftw`` package (installable via the ``fftw`` extra).
2. **scipy.fft** — pocketfft with multi-threading; ships with the package's
   core ``scipy`` dependency, so this is the default for installs without
   ``pyfftw``.
3. **numpy.fft** — always available; last-resort fallback.

All backends expose identical array conventions, so solver results are
numerically equivalent (differences are ~1e-14 floating-point rounding).

Conventions
-----------
The functions mirror the *shifted* conventions used by
:class:`~photonics_helper.pulse.TemporalGrid` (input centered at ``t=0``,
output centered at ``ω=0``):

* ``fft(A)``  == ``np.fft.fftshift(np.fft.fft(np.fft.ifftshift(A)))``
* ``ifft(A)`` == ``np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(A)))``

The ``dt`` / ``1/dt`` scaling that appears at call sites (``TemporalGrid.fft``)
is intentionally **not** applied here, keeping this a pure FFT layer.

Configuration
-------------
``PHOTONICS_FFT_BACKEND`` — force a backend: ``fftw``, ``scipy`` or ``numpy``
(default: auto). If the forced backend is unavailable the next one in the
chain is used and a warning is emitted.

``PHOTONICS_FFTW_PLANNER`` — FFTW planner effort, default ``FFTW_ESTIMATE``.
Use ``FFTW_MEASURE`` (or ``FFTW_PATIENT``) for long production runs where the
one-time planning cost is amortized over thousands of transforms.

``PHOTONICS_FFT_THREADS`` — threads per transform for FFTW3 and scipy,
default ``1``. Multi-threading rarely pays off below ~2¹⁶ points; raise it for
very large grids.
"""

from __future__ import annotations

import os
import threading
import warnings

import numpy as np

try:
    import pyfftw

    _HAS_PYFFTW = True
except ImportError:  # pragma: no cover - exercised only on FFTW-less installs
    pyfftw = None
    _HAS_PYFFTW = False

try:
    import scipy.fft as _sp_fft
    import scipy.signal as _sp_signal

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - scipy is a core dependency
    _sp_fft = None
    _sp_signal = None
    _HAS_SCIPY = False

_PLANNER = os.environ.get("PHOTONICS_FFTW_PLANNER", "FFTW_ESTIMATE").upper()
_THREADS = int(os.environ.get("PHOTONICS_FFT_THREADS", "1"))

_warned: set[str] = set()
_user_forced: bool = False


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        warnings.warn(message, UserWarning, stacklevel=3)


def _next_fast_len(n: int) -> int:
    """Smallest fast FFT length ≥ n (scipy when available, else n itself)."""
    if _HAS_SCIPY:
        return int(_sp_fft.next_fast_len(n))
    return n


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class _NumpyBackend:
    """numpy.fft fallback — always available."""

    name = "numpy"

    @staticmethod
    def fft(A) -> np.ndarray:
        return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(A)))

    @staticmethod
    def ifft(A_w) -> np.ndarray:
        return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(A_w)))

    @staticmethod
    def convolve_full(a, b) -> np.ndarray:
        a = np.asarray(a)
        b = np.asarray(b)
        n = a.shape[-1] + b.shape[-1] - 1
        nfft = _next_fast_len(n)
        a_pad = np.zeros(nfft, dtype=np.complex128)
        b_pad = np.zeros(nfft, dtype=np.complex128)
        a_pad[: a.shape[-1]] = a
        b_pad[: b.shape[-1]] = b
        prod = np.fft.fft(a_pad) * np.fft.fft(b_pad)
        out = np.fft.ifft(prod)[:n]
        if np.iscomplexobj(a) or np.iscomplexobj(b):
            return out
        return out.real


class _ScipyBackend:
    """scipy.fft backend (pocketfft, optional threading) — core dependency."""

    name = "scipy"

    @staticmethod
    def fft(A) -> np.ndarray:
        return np.fft.fftshift(_sp_fft.fft(np.fft.ifftshift(A), workers=_THREADS))

    @staticmethod
    def ifft(A_w) -> np.ndarray:
        return np.fft.fftshift(_sp_fft.ifft(np.fft.ifftshift(A_w), workers=_THREADS))

    @staticmethod
    def convolve_full(a, b) -> np.ndarray:
        return _sp_signal.fftconvolve(a, b, mode="full")


class _FftwBackend:
    """FFTW3 backend via pyfftw with cached plans and aligned buffers."""

    name = "fftw"

    def __init__(self) -> None:
        self._plans: dict[tuple, object] = {}
        self._plans_lock = threading.Lock()
        self._exec_lock = threading.Lock()  # shared buffers: serialize execute()

    def _get_plan(self, shape: tuple[int, ...], dtype: np.dtype, direction: str) -> object:
        key = (shape, np.dtype(dtype).str, direction)
        plan = self._plans.get(key)
        if plan is None:
            with self._plans_lock:
                plan = self._plans.get(key)
                if plan is None:
                    inp = pyfftw.empty_aligned(shape, dtype=dtype)
                    out = pyfftw.empty_aligned(shape, dtype=dtype)
                    plan = pyfftw.FFTW(
                        inp,
                        out,
                        axes=(-1,),
                        direction=direction,
                        flags=(_PLANNER,),
                        threads=_THREADS,
                    )
                    self._plans[key] = plan
        return plan

    @staticmethod
    def _roll_into(dst: np.ndarray, src: np.ndarray, *, left: bool) -> None:
        """``dst = roll(src, ±N//2)`` via slice assignment (no temporaries).

        ``left=True``  reproduces ``np.fft.ifftshift``: dst[k] = src[(k + N//2) % N].
        ``left=False`` reproduces ``np.fft.fftshift``:   dst[k] = src[(k - N//2) % N].

        Both are correct for odd lengths too; they are exact inverses of each other.
        """
        n = src.shape[-1]
        half = n // 2
        if left:
            dst[..., : n - half] = src[..., half:]
            dst[..., n - half :] = src[..., :half]
        else:
            dst[..., :half] = src[..., n - half :]
            dst[..., half:] = src[..., : n - half]

    def fft(self, A) -> np.ndarray:
        A = np.asarray(A)
        dtype = np.result_type(A.dtype, np.complex128)
        shape = A.shape
        plan = self._get_plan(shape, dtype, "FFTW_FORWARD")
        with self._exec_lock:
            inp = plan.input_array
            out = plan.output_array
            self._roll_into(inp, A, left=True)  # centered → FFT order
            plan.execute()
            res = np.empty(shape, dtype=dtype)
            self._roll_into(res, out, left=False)  # FFT order → centered
        return res

    def ifft(self, A_w) -> np.ndarray:
        A_w = np.asarray(A_w)
        dtype = np.result_type(A_w.dtype, np.complex128)
        shape = A_w.shape
        n = shape[-1]
        plan = self._get_plan(shape, dtype, "FFTW_BACKWARD")
        with self._exec_lock:
            inp = plan.input_array
            out = plan.output_array
            self._roll_into(inp, A_w, left=True)
            plan.execute()
            res = np.empty(shape, dtype=dtype)
            self._roll_into(res, out, left=False)
        # FFTW's backward transform is unnormalized — apply the 1/N that
        # np.fft.ifft / scipy.fft.ifft include by default.
        return res / n

    def convolve_full(self, a, b) -> np.ndarray:
        a = np.asarray(a)
        b = np.asarray(b)
        n = a.shape[-1] + b.shape[-1] - 1
        nfft = _next_fast_len(n)
        a_pad = np.zeros(nfft, dtype=np.complex128)
        b_pad = np.zeros(nfft, dtype=np.complex128)
        a_pad[: a.shape[-1]] = a
        b_pad[: b.shape[-1]] = b
        # Standard-order (unshifted) transforms: the convolution theorem holds
        # in the raw FFT basis, so the result is the plain linear convolution.
        prod = self._fft_raw(a_pad) * self._fft_raw(b_pad)
        out = self._ifft_raw(prod)
        result = out[:n]
        if np.iscomplexobj(a) or np.iscomplexobj(b):
            return result
        return result.real

    def _fft_raw(self, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a)
        dtype = np.result_type(a.dtype, np.complex128)
        shape = a.shape
        plan = self._get_plan(shape, dtype, "FFTW_FORWARD")
        with self._exec_lock:
            inp = plan.input_array
            out = plan.output_array
            inp[...] = a
            plan.execute()
            return np.array(out, copy=True)

    def _ifft_raw(self, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a)
        dtype = np.result_type(a.dtype, np.complex128)
        shape = a.shape
        n = shape[-1]
        plan = self._get_plan(shape, dtype, "FFTW_BACKWARD")
        with self._exec_lock:
            inp = plan.input_array
            out = plan.output_array
            inp[...] = a
            plan.execute()
            return np.array(out, copy=True) / n


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

_BACKEND_CLASSES = {
    "fftw": _FftwBackend,
    "scipy": _ScipyBackend,
    "numpy": _NumpyBackend,
}

_active_backend = None
_active_name: str | None = None


def _available_backends() -> list[str]:
    """Backends that can be instantiated, best first."""
    order = ["fftw", "scipy", "numpy"]
    if not _HAS_PYFFTW:
        order.remove("fftw")
    if not _HAS_SCIPY:
        order.remove("scipy")
    return order


def set_backend(name: str | None = None) -> str:
    """Select the active FFT backend and return its name.

    Parameters
    ----------
    name : str, optional
        One of ``"fftw"``, ``"scipy"``, ``"numpy"``, or ``None`` for automatic
        selection (best available). An unavailable forced backend falls back
        to the next best with a warning.

    Returns
    -------
    str — the name of the backend now in use.
    """
    global _active_backend, _active_name, _user_forced
    _user_forced = name is not None
    available = _available_backends()
    if name is None:
        candidates = available
    else:
        key = name.strip().lower()
        if key not in _BACKEND_CLASSES:
            raise ValueError(
                f"Unknown FFT backend {name!r}. Choose from {list(_BACKEND_CLASSES)}."
            )
        candidates = [key] + [b for b in available if b != key]

    for cand in candidates:
        try:
            backend = _BACKEND_CLASSES[cand]()
        except Exception as exc:  # pragma: no cover - import-time guard
            _warn_once(
                f"backend:{cand}",
                f"FFT backend {cand!r} failed to initialise ({exc}); skipping.",
            )
            continue
        _active_backend = backend
        _active_name = backend.name
        break
    return _active_name


def _maybe_warn_on_use() -> None:
    """One informational warning per process when FFTW3 is not in use."""
    if _active_name == "fftw":
        return
    if _user_forced or os.environ.get("PHOTONICS_FFT_BACKEND"):  # explicit choice — no nag
        return
    _warn_once(
        "fallback-use",
        f"FFT backend is {_active_name} — install pyfftw for FFTW3 acceleration: "
        "pip install pyfftw (or add the fftw extra)",
    )


def available() -> bool:
    """True when an accelerated backend (FFTW3 or scipy) is active."""
    return _active_name in ("fftw", "scipy")


def backend_name() -> str:
    """Human-readable name of the active backend."""
    if _active_name == "fftw":
        return f"FFTW3 via pyfftw {pyfftw.__version__}"
    if _active_name == "scipy":
        return f"scipy.fft (pocketfft{', workers=' + str(_THREADS) if _THREADS > 1 else ''})"
    return "numpy.fft (fallback)"


def fft(A) -> np.ndarray:
    """Shifted forward FFT — identical to ``np.fft.fftshift(np.fft.fft(np.fft.ifftshift(A)))``.

    Parameters
    ----------
    A : array_like — time-domain signal(s), index 0 at ``t=0`` (centered order).

    Returns
    -------
    Frequency-domain array(s), index 0 at ``ω=0`` (centered order), same
    convention and scaling as ``np.fft.fft`` (unnormalized forward transform).
    """
    _maybe_warn_on_use()
    return _active_backend.fft(A)


def ifft(A_w) -> np.ndarray:
    """Shifted inverse FFT — identical to ``np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(A_w)))``.

    Includes the ``1/N`` normalization, matching ``np.fft.ifft``.

    Parameters
    ----------
    A_w : array_like — frequency-domain signal(s), index 0 at ``ω=0``.

    Returns
    -------
    Time-domain array(s), index 0 at ``t=0``.
    """
    _maybe_warn_on_use()
    return _active_backend.ifft(A_w)


def convolve_full(a, b) -> np.ndarray:
    """Full linear convolution of two 1-D arrays along the last axis.

    Drop-in replacement for ``scipy.signal.fftconvolve(a, b, mode='full')``
    (same output ordering and length ``len(a) + len(b) - 1``) implemented on
    the active backend.

    Parameters
    ----------
    a, b : array_like — 1-D signals to convolve.

    Returns
    -------
    NDArray — full convolution, real dtype when both inputs are real.
    """
    _maybe_warn_on_use()
    return _active_backend.convolve_full(a, b)


# Select the backend once at import time (respects PHOTONICS_FFT_BACKEND).
_set_backend_env = os.environ.get("PHOTONICS_FFT_BACKEND")
if _set_backend_env:
    set_backend(_set_backend_env)
else:
    set_backend(None)
