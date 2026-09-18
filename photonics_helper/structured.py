"""Structured light: Laguerre-Gaussian / OAM transverse modes.

Analytic Laguerre-Gaussian (LG) modes carrying orbital angular momentum (OAM),
the propagation helpers that describe how such a beam evolves (beam waist,
radius of curvature, Gouy phase), discrete overlap integrals on a transverse
``(x, y)`` grid, and transverse-profile visualization.

Conventions
-----------
The normalized LG mode used here is

.. math::

    u_p^l(r, \\varphi) = \\frac{1}{w}
        \\sqrt{\\frac{2\\,p!}{\\pi\\,(p+|l|)!}}
        \\left(\\frac{\\sqrt{2}\\,r}{w}\\right)^{|l|}
        L_p^{|l|}\\!\\left(\\frac{2 r^2}{w^2}\\right)
        e^{-r^2/w^2} e^{i l \\varphi},

so that :math:`\\int |u|^2 \\, r\\,dr\\,d\\varphi = 1` and the OAM phase winds as
``exp(i l phi)`` (a closed loop enclosing the axis accumulates ``2 pi l``).
Propagation to ``z`` substitutes ``w -> w(z)``, adds the wavefront curvature
``exp(-i k r^2 / 2R(z))`` and the Gouy phase ``exp(i (2p+|l|+1) arctan(z/zR))``.

References
----------
- L. Allen, M. W. Beijersbergen, R. J. C. Spreeuw & J. P. Woerdman,
  *Phys. Rev. A* **45**, 8185 (1992) (LG modes carrying OAM).
- A. E. Siegman, *Lasers*, University Science Books (1986), Ch. 16-17
  (Gaussian beam propagation, Gouy phase).
- M. J. Padgett & L. Allen, *Contemp. Phys.* **41**, 275 (2000).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial, pi
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import eval_genlaguerre

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

__all__ = [
    "LaguerreGaussianMode",
    "StructuredField",
    "beam_waist",
    "gouy_phase",
    "overlap",
    "plot_transverse_profile",
    "radius_of_curvature",
    "rayleigh_range",
]


# ---------------------------------------------------------------------------
# Gaussian-beam propagation helpers
# ---------------------------------------------------------------------------
def rayleigh_range(w0: float, wavelength: float) -> float:
    """Return the Rayleigh range ``z_R = pi w0^2 / lambda`` (metres).

    Parameters
    ----------
    w0 : beam waist (metres), must be positive.
    wavelength : vacuum wavelength (metres), must be positive.
    """
    if w0 <= 0.0:
        raise ValueError("w0 must be positive")
    if wavelength <= 0.0:
        raise ValueError("wavelength must be positive")
    return float(pi * w0**2 / wavelength)


def beam_waist(w0: float, z: float, wavelength: float) -> float:
    """Return the 1/e^2 beam radius ``w(z) = w0 sqrt(1 + (z/z_R)^2)`` (metres)."""
    z_r = rayleigh_range(w0, wavelength)
    return float(w0 * np.sqrt(1.0 + (z / z_r) ** 2))


def radius_of_curvature(z: float, z_r: float) -> float:
    """Return the wavefront radius of curvature ``R(z) = z (1 + (z_R/z)^2)``.

    ``R(0)`` is infinite (a flat wavefront at the waist).
    """
    if z == 0.0:
        return np.inf
    return float(z * (1.0 + (z_r / z) ** 2))


def gouy_phase(order: int, z: float, z_r: float) -> float:
    """Return the Gouy phase ``(order) * arctan(z / z_R)`` (radians).

    For a Laguerre-Gaussian mode ``order = 2p + |l| + 1``.
    """
    return float(order * np.arctan(z / z_r))


# ---------------------------------------------------------------------------
# Transverse field container
# ---------------------------------------------------------------------------
@dataclass
class StructuredField:
    """A complex scalar transverse field sampled on a uniform ``(x, y)`` grid.

    Parameters
    ----------
    field : complex 2-D array with shape ``(ny, nx)``.
    dx : grid spacing along ``x`` (metres).
    dy : grid spacing along ``y`` (metres); defaults to ``dx`` when ``None``.
    x0, y0 : optional offsets of the grid centre (metres, default 0).
    """

    field: ComplexArray
    dx: float
    dy: float | None = None
    x0: float = 0.0
    y0: float = 0.0

    def __post_init__(self) -> None:
        data = np.asarray(self.field, dtype=np.complex128)
        if data.ndim != 2:
            raise ValueError("StructuredField.field must be a 2-D array")
        if self.dy is None:
            self.dy = float(self.dx)
        if not (self.dx > 0.0 and self.dy > 0.0):
            raise ValueError("dx and dy must be positive")
        self.field = data

    # -- geometry ----------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int]:
        """Grid shape ``(ny, nx)``."""
        return (int(self.field.shape[0]), int(self.field.shape[1]))

    @property
    def ny(self) -> int:
        return int(self.field.shape[0])

    @property
    def nx(self) -> int:
        return int(self.field.shape[1])

    @property
    def spacing(self) -> tuple[float, float]:
        """``(dx, dy)`` in metres."""
        return float(self.dx), float(self.dx if self.dy is None else self.dy)

    @staticmethod
    def _axis(n: int, d: float, offset: float) -> FloatArray:
        return (np.arange(n, dtype=float) - (n - 1) / 2.0) * d + offset

    @property
    def x(self) -> FloatArray:
        """1-D ``x`` coordinates of the grid columns (metres)."""
        dx, _ = self.spacing
        return self._axis(self.nx, dx, self.x0)

    @property
    def y(self) -> FloatArray:
        """1-D ``y`` coordinates of the grid rows (metres)."""
        _, dy = self.spacing
        return self._axis(self.ny, dy, self.y0)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """``(xmin, xmax, ymin, ymax)`` pixel-edge extent in metres."""
        dx, dy = self.spacing
        x, y = self.x, self.y
        return (
            float(x[0] - dx / 2.0),
            float(x[-1] + dx / 2.0),
            float(y[0] - dy / 2.0),
            float(y[-1] + dy / 2.0),
        )

    # -- physical quantities ----------------------------------------------
    @property
    def intensity(self) -> FloatArray:
        """``|E(x, y)|^2``."""
        return np.abs(self.field) ** 2

    @property
    def power(self) -> float:
        """Discrete power ``sum |E|^2 dx dy`` (arbitrary units)."""
        dx, dy = self.spacing
        return float(self.intensity.sum() * dx * dy)

    def normalize(self) -> StructuredField:
        """Return a copy scaled to unit discrete power."""
        p = self.power
        if p <= 0.0:
            raise ValueError("cannot normalize a zero-power field")
        dx, dy = self.spacing
        return StructuredField(self.field / np.sqrt(p), dx, dy, self.x0, self.y0)

    def centroid(self) -> tuple[float, float]:
        """Intensity-weighted centroid ``(xc, yc)`` in metres."""
        w = self.intensity
        total = w.sum()
        if total <= 0.0:
            raise ValueError("cannot locate the centroid of a zero-power field")
        x_grid, y_grid = np.meshgrid(self.x, self.y)
        return (
            float((w * x_grid).sum() / total),
            float((w * y_grid).sum() / total),
        )

    def second_moment_radius(self) -> float:
        """Intensity-weighted RMS radius ``sqrt(<r^2>)`` in metres.

        For a Laguerre-Gaussian mode this equals
        ``w(z)/sqrt(2) * sqrt(2p + |l| + 1)``.
        """
        w = self.intensity
        total = w.sum()
        if total <= 0.0:
            raise ValueError("cannot size a zero-power field")
        x_grid, y_grid = np.meshgrid(self.x, self.y)
        xc, yc = self.centroid()
        r2 = (x_grid - xc) ** 2 + (y_grid - yc) ** 2
        return float(np.sqrt((w * r2).sum() / total))

    # -- inner product -----------------------------------------------------
    def overlap(self, other: StructuredField, *, normalize: bool = True) -> complex:
        """Return the inner product ``integral conj(E1) E2 dx dy``.

        With ``normalize=True`` (default) both fields are first scaled to unit
        discrete power, so the auto-overlap of a mode is exactly ``1`` and the
        result is the modal overlap coefficient.
        """
        if self.shape != other.shape:
            raise ValueError("fields must share the same grid shape")
        dx, dy = self.spacing
        dx2, dy2 = other.spacing
        if not np.isclose(dx, dx2) or not np.isclose(dy, dy2):
            raise ValueError("fields must share the same grid spacing")
        a = self.normalize() if normalize else self
        b = other.normalize() if normalize else other
        return complex(np.sum(np.conj(a.field) * b.field) * dx * dy)

    def plot(self, **kwargs: Any) -> Any:
        """Plot intensity and phase; see :func:`plot_transverse_profile`."""
        return plot_transverse_profile(self, **kwargs)


# ---------------------------------------------------------------------------
# Laguerre-Gaussian mode
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LaguerreGaussianMode:
    """An analytic Laguerre-Gaussian mode ``LG(p, l)``.

    Parameters
    ----------
    p : radial index (integer >= 0).
    l : azimuthal index (integer); the OAM carried by the mode is ``l * hbar``.
    w0 : beam waist at ``z = 0`` (metres).
    wavelength : vacuum wavelength (metres). Required for any ``z != 0``
        evaluation; ``None`` restricts the mode to its waist.
    """

    p: int
    l: int
    w0: float
    wavelength: float | None = None

    def __post_init__(self) -> None:
        if int(self.p) != self.p or self.p < 0:
            raise ValueError("p must be a non-negative integer")
        if int(self.l) != self.l:
            raise ValueError("l must be an integer")
        if self.w0 <= 0.0:
            raise ValueError("w0 must be positive")
        if self.wavelength is not None and self.wavelength <= 0.0:
            raise ValueError("wavelength must be positive")

    @property
    def order(self) -> int:
        """Gouy order ``2p + |l| + 1``."""
        return 2 * int(self.p) + abs(int(self.l)) + 1

    @property
    def z_r(self) -> float:
        """Rayleigh range (requires ``wavelength``)."""
        if self.wavelength is None:
            raise ValueError("wavelength is required to compute the Rayleigh range")
        return rayleigh_range(self.w0, self.wavelength)

    def waist(self, z: float = 0.0) -> float:
        """Beam radius ``w(z)`` (metres)."""
        if self.wavelength is None:
            if z != 0.0:
                raise ValueError("wavelength is required for z != 0")
            return float(self.w0)
        return beam_waist(self.w0, z, self.wavelength)

    def radius_of_curvature(self, z: float) -> float:
        """Wavefront radius of curvature ``R(z)`` (metres)."""
        return radius_of_curvature(z, self.z_r)

    def gouy(self, z: float) -> float:
        """Gouy phase at ``z`` (radians)."""
        return gouy_phase(self.order, z, self.z_r)

    def field(self, x: FloatArray, y: FloatArray, z: float = 0.0) -> ComplexArray:
        """Evaluate the mode on the Cartesian product of ``x`` and ``y``.

        Returns an array of shape ``(len(y), len(x))``.
        """
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if x_arr.ndim != 1 or y_arr.ndim != 1:
            raise ValueError("x and y must be 1-D coordinate arrays")
        x_grid, y_grid = np.meshgrid(x_arr, y_arr)
        return self._evaluate(x_grid, y_grid, z)

    def evaluate(self, x: FloatArray, y: FloatArray, z: float = 0.0) -> ComplexArray:
        """Evaluate the mode elementwise at coordinates ``x``, ``y``.

        Unlike :meth:`field`, this does not form a grid: ``x`` and ``y`` are
        broadcast together. Use it to sample arbitrary points, e.g. a closed
        loop around the optical axis for the OAM winding number.
        """
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        return self._evaluate(x_arr, y_arr, z)

    def _evaluate(
        self, x_grid: FloatArray, y_grid: FloatArray, z: float
    ) -> ComplexArray:
        p, l = int(self.p), int(self.l)
        w = self.waist(z)
        r2 = x_grid * x_grid + y_grid * y_grid
        rho2 = 2.0 * r2 / (w * w)
        radial = (
            np.sqrt(rho2) ** abs(l)
            * eval_genlaguerre(p, abs(l), rho2)
            * np.exp(-r2 / (w * w))
        )
        norm = np.sqrt(2.0 * factorial(p) / (pi * factorial(p + abs(l)))) / w
        out = norm * radial * np.exp(1j * l * np.arctan2(y_grid, x_grid))

        if self.wavelength is not None:
            k = 2.0 * pi / self.wavelength
            curv = radius_of_curvature(z, self.z_r)
            out = out * np.exp(-1j * k * r2 / (2.0 * curv))
            out = out * np.exp(1j * self.gouy(z))

        return np.asarray(out, dtype=np.complex128)

    def structured(
        self, *, half_width: float | None = None, n: int = 257, z: float = 0.0
    ) -> StructuredField:
        """Sample the mode on a square grid.

        Parameters
        ----------
        half_width : half the physical grid size (metres). Defaults to
            ``6 * w(z)`` so the tails are contained at any plane.
        n : number of samples per axis (default 257, odd to include the axis).
        z : propagation distance (metres, default 0).
        """
        if n < 2:
            raise ValueError("n must be at least 2")
        if half_width is None:
            half_width = 6.0 * self.waist(z)
        if half_width <= 0.0:
            raise ValueError("half_width must be positive")
        axis = np.linspace(-half_width, half_width, n)
        dx = float(axis[1] - axis[0])
        return StructuredField(self.field(axis, axis, z), dx, dx)

    def plot(self, **kwargs: Any) -> Any:
        """Plot the mode; see :func:`plot_transverse_profile`."""
        return plot_transverse_profile(self, **kwargs)


def overlap(
    a: LaguerreGaussianMode | StructuredField,
    b: LaguerreGaussianMode | StructuredField,
    *,
    half_width: float | None = None,
    n: int = 257,
    z: float = 0.0,
    normalize: bool = True,
) -> complex:
    """Return the overlap integral ``integral conj(E_a) E_b dx dy``.

    Accepts two :class:`LaguerreGaussianMode` instances (sampled on a common
    grid) or two :class:`StructuredField` instances (integrated directly).
    Modal auto-overlaps are exactly 1 when ``normalize`` is True.
    """
    mode_pair = isinstance(a, LaguerreGaussianMode) and isinstance(
        b, LaguerreGaussianMode
    )
    field_pair = isinstance(a, StructuredField) and isinstance(b, StructuredField)

    if mode_pair:
        assert isinstance(a, LaguerreGaussianMode)
        assert isinstance(b, LaguerreGaussianMode)
        if half_width is None:
            half_width = 6.0 * max(a.waist(z), b.waist(z))
        field_a = a.structured(half_width=half_width, n=n, z=z)
        field_b = b.structured(half_width=half_width, n=n, z=z)
        return field_a.overlap(field_b, normalize=normalize)

    if field_pair:
        assert isinstance(a, StructuredField)
        assert isinstance(b, StructuredField)
        if half_width is not None or z != 0.0:
            raise ValueError("half_width and z apply only to mode inputs")
        return a.overlap(b, normalize=normalize)

    raise TypeError(
        "overlap expects two LaguerreGaussianMode or two StructuredField instances"
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_transverse_profile(
    source: LaguerreGaussianMode | StructuredField,
    *,
    backend: Literal["matplotlib", "plotly"] = "matplotlib",
    half_width: float | None = None,
    n: int = 257,
    z: float = 0.0,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    theme: Literal["light", "dark"] = "light",
) -> Any:
    """Plot the intensity and phase of a transverse field.

    Parameters
    ----------
    source : a :class:`LaguerreGaussianMode` (sampled on the fly) or an
        already-sampled :class:`StructuredField`.
    backend : ``"matplotlib"`` (default) or ``"plotly"`` (optional dependency).
    half_width, n, z : sampling controls used when ``source`` is a mode.
    figsize : figure size for the matplotlib backend.
    title : optional title override.
    theme : ``"light"`` or ``"dark"`` (plotly template / matplotlib facecolor).

    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
    """
    if backend not in ("matplotlib", "plotly"):
        raise ValueError("backend must be 'matplotlib' or 'plotly'")
    if theme not in ("light", "dark"):
        raise ValueError("theme must be 'light' or 'dark'")

    if isinstance(source, LaguerreGaussianMode):
        field = source.structured(half_width=half_width, n=n, z=z)
        if title is None:
            title = f"LG(p={source.p}, l={source.l}) at z = {z * 1e6:.2f} µm"
    else:
        field = source
        if title is None:
            title = "Transverse profile"

    if backend == "plotly":
        return _plot_transverse_plotly(field, title, theme)
    return _plot_transverse_matplotlib(field, title, figsize, theme)


def _plot_transverse_matplotlib(
    field: StructuredField,
    title: str,
    figsize: tuple[float, float] | None,
    theme: str,
) -> Any:
    import matplotlib.pyplot as plt

    intensity = field.intensity
    phase = np.angle(field.field)
    threshold = 1e-6 * float(intensity.max()) if intensity.size else 0.0
    masked_phase = np.ma.masked_where(intensity < threshold, phase)
    extent_um = tuple(v * 1e6 for v in field.extent)

    fig, axes = plt.subplots(1, 2, figsize=figsize or (9.5, 4.2), layout="constrained")
    if theme == "dark":
        fig.patch.set_facecolor("#1e1e1e")
        for ax in axes:
            ax.set_facecolor("#1e1e1e")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")

    im0 = axes[0].imshow(
        intensity,
        origin="lower",
        extent=extent_um,
        cmap="inferno",
        aspect="equal",
    )
    axes[0].set_title("Intensity |E|²")
    axes[0].set_xlabel("x (µm)")
    axes[0].set_ylabel("y (µm)")
    fig.colorbar(im0, ax=axes[0], shrink=0.9)

    im1 = axes[1].imshow(
        masked_phase,
        origin="lower",
        extent=extent_um,
        cmap="twilight",
        vmin=-pi,
        vmax=pi,
        aspect="equal",
    )
    axes[1].set_title("Phase (rad)")
    axes[1].set_xlabel("x (µm)")
    axes[1].set_ylabel("y (µm)")
    fig.colorbar(im1, ax=axes[1], shrink=0.9)

    fig.suptitle(title)
    return fig


def _plot_transverse_plotly(field: StructuredField, title: str, theme: str) -> Any:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "plotly is required for the interactive backend; install with "
            "`pip install plotly` or `pip install photonics-helper[plotting]`"
        ) from exc

    intensity = field.intensity
    phase = np.angle(field.field)
    threshold = 1e-6 * float(intensity.max()) if intensity.size else 0.0
    masked_phase = np.where(intensity < threshold, np.nan, phase)
    x_um = field.x * 1e6
    y_um = field.y * 1e6

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Intensity |E|²", "Phase (rad)"),
        horizontal_spacing=0.14,
    )
    fig.add_trace(
        go.Heatmap(z=intensity, x=x_um, y=y_um, colorscale="Inferno"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=masked_phase,
            x=x_um,
            y=y_um,
            colorscale="Twilight",
            zmin=-pi,
            zmax=pi,
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="x (µm)")
    fig.update_yaxes(title_text="y (µm)", scaleanchor="x", scaleratio=1.0)
    fig.update_layout(
        title=title,
        template="plotly_white" if theme == "light" else "plotly_dark",
    )
    return fig
