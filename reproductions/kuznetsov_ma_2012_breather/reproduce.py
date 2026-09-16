"""Reproduction: Kuznetsov-Ma soliton dynamics in optical fibre (Kibler et al., 2012).

Reference
---------
B. Kibler, J. Fatome, C. Finot, G. Millot, G. Genty, B. Wetzel, N. Akhmediev,
F. Dias, J. M. Dudley, "Observation of Kuznetsov-Ma soliton dynamics in optical
fibre", Scientific Reports 2, 463 (2012), doi:10.1038/srep00463.

What is reproduced
------------------
The paper reviews the general soliton-on-finite-background (SFB) solution of the
dimensionless self-focusing NLSE

    i psi_xi + (1/2) psi_tautau + |psi|^2 psi = 0,                  (Eq. 1)

    psi(xi, tau) = e^{i xi} [ 1 + ( 2(1-2a) cosh(b xi) + i b sinh(b xi) )
                                 / ( sqrt(2a) cos(nu tau) - cosh(b xi) ) ],  (Eq. 2)

with b = sqrt(8a(1-2a)) and nu = 2 sqrt(1-2a).  For a > 1/2 (the Kuznetsov-Ma
regime) b -> iB and nu -> i/Delta, so Eq. (2) becomes the periodic-on-background
KM soliton (paper Eq. 3), with

    B = sqrt(8a(2a-1)),   Delta = 1 / (2 sqrt(2a-1)),   period Dxi = 2 pi / B.

The minimum- and maximum-compression profiles are (paper Eqs. 4-5)

    psi_min(tau) = 1 + 2(2a-1) / ( sqrt(2a) cosh(tau/Delta) + 1 ),
    psi_max(tau) = 1 - 2(2a-1) / ( sqrt(2a) cosh(tau/Delta) - 1 ).

Dimensional mapping (paper Methods, SMF-28 at 1554.9 nm):

    A(z, T) = sqrt(P0) psi,   T = tau T0,   z = xi L_NL + z_p / 2,
    L_NL = 1/(gamma P0),      T0 = sqrt(|beta2| L_NL),   z_p = L_NL * Dxi.

We construct the exact KM field at its minimum-intensity point
(``A(0, T) = sqrt(P0) psi_min``) and propagate it for one full KM period with
the library's split-step GNLSE solver in the pure-NLSE limit (beta2 only, no
Raman / self-steepening / TPA).  The numerical centre power is compared with the
analytic ``|psi(xi, 0)|^2``, and the full temporal and spectral profiles at
maximum compression are compared with the analytic solution.

A second, lossy run with the paper's SMF-28 parameters (alpha = 0.2 dB/km) shows
the loss-induced deviation discussed in the paper: the ideal KM curve assumes
no loss, and the experiment/simulation follows below it at maximum compression.

Usage
-----
    python reproductions/kuznetsov_ma_2012_breather/reproduce.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from photonics_helper.base import C_MS, Length, Time, Wavelength
from photonics_helper.gnlse import FiberProfile, GNLSESolver
from photonics_helper.pulse import Envelope, TemporalGrid, Wave

HERE = Path(__file__).resolve().parent
PARAMETERS = HERE / "parameters.json"


# ---------------------------------------------------------------------------
# Analytical Kuznetsov-Ma solution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KMParameters:
    """Dimensional quantities of the KM solution on the paper's SMF-28 fibre."""

    a: float
    beta2_ps2_per_m: float
    beta3_ps3_per_m: float
    gamma_per_Wm: float
    alpha_per_m: float
    P0: float
    LNL_m: float
    T0_s: float
    B: float
    Delta: float
    period_m: float

    @property
    def T0_ps(self) -> float:
        return self.T0_s * 1e12

    @property
    def period_km(self) -> float:
        return self.period_m * 1e-3

    @property
    def peak_power_W(self) -> float:
        return self.P0 * abs(self.psi_max(0.0)) ** 2

    @property
    def initial_center_power_W(self) -> float:
        return self.P0 * abs(self.psi_min(0.0)) ** 2

    def _km_psi(self, xi: float, tau: NDArray | float) -> NDArray | complex:
        a, B, D = self.a, self.B, self.Delta
        num = 2 * (1 - 2 * a) * np.cos(B * xi) - 1j * B * np.sin(B * xi)
        den = np.sqrt(2 * a) * np.cosh(np.asarray(tau) / D) - np.cos(B * xi)
        return np.exp(1j * xi) * (1 + num / den)

    def psi(self, z: float, t: NDArray | float) -> NDArray | complex:
        """Exact KM field ``psi`` at physical distance ``z`` [m] and time ``t`` [s]."""
        xi = (z - self.period_m / 2) / self.LNL_m
        return self._km_psi(xi, t / self.T0_s)

    def psi_min(self, tau: NDArray | float) -> NDArray | complex:
        """Minimum-intensity profile, paper Eq. (4)."""
        a, D = self.a, self.Delta
        return 1.0 + 2 * (2 * a - 1) / (np.sqrt(2 * a) * np.cosh(np.asarray(tau) / D) + 1)

    def psi_max(self, tau: NDArray | float) -> NDArray | complex:
        """Maximum-compression profile, paper Eq. (5)."""
        a, D = self.a, self.Delta
        return 1.0 - 2 * (2 * a - 1) / (np.sqrt(2 * a) * np.cosh(np.asarray(tau) / D) - 1)

    def center_power_W(self, z: float) -> float:
        """Analytic intensity at T = 0, ``P0 |psi(xi, 0)|^2``."""
        return float(self.P0 * abs(self._km_psi((z - self.period_m / 2) / self.LNL_m, 0.0)) ** 2)


def derive(params: dict) -> KMParameters:
    """Map the paper's fibre parameters onto the dimensionless KM solution."""
    a = float(params["km_parameter_a"])
    if a <= 0.5:
        raise ValueError(f"KM regime requires a > 1/2, got {a}")

    beta2_pm = float(params["beta2_ps2_per_km"]) * 1e-3  # ps^2/m
    gamma = float(params["gamma_per_W_per_km"]) * 1e-3  # 1/(W m)
    alpha = float(params["loss_dB_per_km"]) / (10.0 * np.log10(np.e)) / 1e3  # 1/m
    P0 = float(params["background_power_W"])

    LNL = 1.0 / (gamma * P0)
    T0 = float(np.sqrt(abs(beta2_pm) * 1e-24 * LNL))
    B = float(np.sqrt(8 * a * (2 * a - 1)))
    Delta = 1.0 / (2.0 * np.sqrt(2 * a - 1))
    period = 2 * np.pi / B * LNL
    return KMParameters(
        a=a,
        beta2_ps2_per_m=beta2_pm,
        beta3_ps3_per_m=float(params["beta3_ps3_per_km"]) * 1e-3,
        gamma_per_Wm=gamma,
        alpha_per_m=alpha,
        P0=P0,
        LNL_m=LNL,
        T0_s=T0,
        B=B,
        Delta=Delta,
        period_m=period,
    )


# ---------------------------------------------------------------------------
# Numerical propagation
# ---------------------------------------------------------------------------


def build_grid(params: dict) -> TemporalGrid:
    return TemporalGrid(N=int(params["grid_N"]), Tmax=Time(float(params["grid_Tmax_ps"]) * 1e-12, "s"))


def build_pulse(params: dict, km: KMParameters, grid: TemporalGrid) -> Wave:
    """Exact KM field at the minimum-intensity point of the cycle."""

    def km_min(t: NDArray, _T0: float, A0: float) -> NDArray:
        return A0 * km.psi_min(t / km.T0_s)

    env = Envelope(
        shape="custom",
        peak_amplitude=float(np.sqrt(km.P0)),
        pulse_width=Time(km.T0_s, "s"),
        func=km_min,
    )
    return Wave(
        grid=grid,
        envelope=env,
        central_wavelength=Wavelength(float(params["central_wavelength_nm"]) * 1e-9, "m"),
    )


def propagate(
    params: dict,
    km: KMParameters,
    grid: TemporalGrid,
    *,
    num_steps: int | None = None,
    lossy: bool = False,
    third_order: bool = False,
    nsaves: int = 41,
) -> tuple[NDArray, list[Wave]]:
    """Propagate one KM period and return ``(z_array, evolution)``."""
    steps = int(num_steps if num_steps is not None else params["num_steps"])
    pulse = build_pulse(params, km, grid)
    fiber = FiberProfile.from_gamma(
        gamma=km.gamma_per_Wm,
        n2=float(params["n2_m2_per_W"]),
        omega0=2 * np.pi * C_MS / (float(params["central_wavelength_nm"]) * 1e-9),
        alpha=km.alpha_per_m if lossy else 0.0,
        length=Length(km.period_m, "m"),
    )
    betas = np.array(
        [
            km.beta2_ps2_per_m,
            km.beta3_ps3_per_m if third_order else 0.0,
        ]
    )
    solver = GNLSESolver(
        pulse=pulse,
        fiber=fiber,
        betas=betas,
        include_raman=False,
        include_self_steepening=False,
        include_tpa=False,
    )
    solver.propagate(num_steps=steps, nsaves=nsaves)
    return solver.z_array, solver.evolution


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(
    params: dict | None = None,
    *,
    num_steps: int | None = None,
    make_plot: bool = True,
) -> dict:
    """Run the exact-KM validation and return the metrics."""
    params = params or json.loads(PARAMETERS.read_text())
    km = derive(params)
    grid = build_grid(params)

    z, evolution = propagate(params, km, grid, num_steps=num_steps, lossy=False, nsaves=41)

    mid = grid.N // 2
    center_num = np.array([abs(w.envelope_field[mid]) ** 2 for w in evolution])
    center_an = np.array([km.center_power_W(zi) for zi in z])

    # Full temporal-intensity comparison over the saved snapshots.
    intensity_rel_l2 = []
    for zi, w in zip(z, evolution):
        A = w.envelope_field
        Aa = km.psi(zi, grid.t) * np.sqrt(km.P0)
        inten, inten_an = np.abs(A) ** 2, np.abs(Aa) ** 2
        intensity_rel_l2.append(float(np.linalg.norm(inten - inten_an) / np.linalg.norm(inten_an)))
    intensity_rel_l2 = np.array(intensity_rel_l2)

    # Spectral comparison at maximum compression (DC background removed).
    i_max = int(np.argmin(np.abs(z - km.period_m / 2)))
    A = evolution[i_max].envelope_field
    Aa = km.psi(z[i_max], grid.t) * np.sqrt(km.P0)
    S_num = np.abs(grid.fft(A - A.mean())) ** 2
    S_an = np.abs(grid.fft(Aa - Aa.mean())) ** 2
    spectrum_rel_l2 = float(np.linalg.norm(S_num - S_an) / np.linalg.norm(S_an))

    peak_num = float(center_num[i_max])
    peak_an = float(center_an[i_max])

    # Lossy run with the paper's SMF-28 loss (qualitative experiment comparison).
    z_lossy, ev_lossy = propagate(params, km, grid, num_steps=num_steps, lossy=True, nsaves=41)
    lossy_center = np.array([abs(w.envelope_field[mid]) ** 2 for w in ev_lossy])
    i_lossy_max = int(np.argmin(np.abs(z_lossy - km.period_m / 2)))
    lossy_peak = float(lossy_center[i_lossy_max])

    result = {
        "T0_ps": km.T0_ps,
        "LNL_m": km.LNL_m,
        "period_km": km.period_km,
        "B": km.B,
        "Delta": km.Delta,
        "initial_center_power_W": km.initial_center_power_W,
        "peak_analytic_W": peak_an,
        "peak_numerical_W": peak_num,
        "peak_rel_err": abs(peak_num - peak_an) / peak_an,
        "center_max_abs_err_W": float(np.max(np.abs(center_num - center_an))),
        "intensity_rel_l2_max": float(np.max(intensity_rel_l2)),
        "spectrum_rel_l2": spectrum_rel_l2,
        "lossy_peak_W": lossy_peak,
        "lossy_peak_reduction": (peak_an - lossy_peak) / peak_an,
    }

    if make_plot:
        _plot(km, grid, z, evolution, center_num, center_an, z_lossy, lossy_center, result)

    tol = params["tolerances"]
    assert abs(result["T0_ps"] - params["derived_reference"]["T0_ps"]) < tol["T0_abs_ps"]
    assert abs(result["period_km"] - params["derived_reference"]["period_km"]) / params[
        "derived_reference"
    ]["period_km"] < tol["period_rel"]
    assert result["peak_rel_err"] < tol["peak_rel"], result["peak_rel_err"]
    assert result["center_max_abs_err_W"] < tol["center_abs_W"], result["center_max_abs_err_W"]
    assert result["intensity_rel_l2_max"] < tol["intensity_rel_l2"], result["intensity_rel_l2_max"]
    assert result["spectrum_rel_l2"] < tol["spectrum_rel_l2"], result["spectrum_rel_l2"]
    assert result["lossy_peak_W"] < result["peak_analytic_W"], "loss must reduce the compression peak"

    print("Kuznetsov-Ma validation passed:")
    print(
        f"  T0 = {result['T0_ps']:.3f} ps, L_NL = {result['LNL_m']:.1f} m, "
        f"period = {result['period_km']:.4f} km (a = {km.a})"
    )
    print(
        f"  centre power: init {result['initial_center_power_W']:.4f} W -> "
        f"peak {result['peak_numerical_W']:.4f} W (analytic {result['peak_analytic_W']:.4f} W)"
    )
    print(
        f"  max centre error {result['center_max_abs_err_W']:.2e} W, "
        f"intensity L2 {result['intensity_rel_l2_max']:.2e}, spectrum L2 {result['spectrum_rel_l2']:.2e}"
    )
    print(
        f"  lossy (0.2 dB/km) compression peak {result['lossy_peak_W']:.4f} W "
        f"({100 * result['lossy_peak_reduction']:.1f} % below ideal)"
    )
    return result


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _plot(km, grid, z, evolution, center_num, center_an, z_lossy, lossy_center, result) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    # (a) Evolution of the intensity profile.
    t_ps = grid.t * 1e12
    profiles = np.array([np.abs(w.envelope_field) ** 2 for w in evolution])
    extent = [t_ps[0], t_ps[-1], z[-1] * 1e-3, 0.0]
    im = axes[0].imshow(profiles, aspect="auto", extent=extent, cmap="magma", vmin=0, vmax=profiles.max())
    axes[0].set_xlim(-15, 15)
    axes[0].set_xlabel("time $T$ (ps)")
    axes[0].set_ylabel("distance $z$ (km)")
    axes[0].set_title("(a) KM breather evolution")
    fig.colorbar(im, ax=axes[0], label="$|A|^2$ (W)")

    # (b) Centre power vs distance: numerical vs analytic.
    axes[1].plot(z * 1e-3, center_an, "k-", lw=2, label="analytic KM, Eq. (3)")
    axes[1].plot(z * 1e-3, center_num, "r--", lw=1.5, label="GNLSE (lossless)")
    axes[1].plot(z_lossy * 1e-3, lossy_center, "b:", lw=1.5, label="GNLSE, SMF-28 + 0.2 dB/km loss")
    axes[1].set_xlabel("distance $z$ (km)")
    axes[1].set_ylabel("centre power $|A(T{=}0)|^2$ (W)")
    axes[1].set_title("(b) Breathing of the central lobe")
    axes[1].legend(fontsize=8)

    # (c) Profiles at minimum / maximum compression.
    z_half = km.period_m / 2
    i_half = int(np.argmin(np.abs(z - z_half)))
    axes[2].plot(t_ps, np.abs(evolution[0].envelope_field) ** 2, "b-", label="GNLSE, $z=0$ (min)")
    axes[2].plot(
        t_ps,
        np.abs(km.psi(0.0, grid.t) * np.sqrt(km.P0)) ** 2,
        "k--",
        label="analytic, $z=0$",
    )
    axes[2].plot(
        t_ps,
        np.abs(evolution[i_half].envelope_field) ** 2,
        "r-",
        label="$z=z_p/2$ (max)",
    )
    axes[2].plot(
        t_ps,
        np.abs(km.psi(z_half, grid.t) * np.sqrt(km.P0)) ** 2,
        "k--",
    )
    axes[2].set_xlim(-12, 12)
    axes[2].set_xlabel("time $T$ (ps)")
    axes[2].set_ylabel("$|A|^2$ (W)")
    axes[2].set_title("(c) Profiles at extrema")
    axes[2].legend(fontsize=8)

    fig.suptitle(
        "Kuznetsov-Ma soliton (Kibler et al., Sci. Rep. 2, 463, 2012) — "
        f"$a={km.a}$, $z_p={km.period_km:.3f}$ km, peak {result['peak_numerical_W']:.2f} W",
        fontsize=11,
    )
    fig.tight_layout()
    out = HERE / "kuznetsov_ma_breather.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def make_contact_sheet() -> None:
    """Render the downloaded paper pages into a single contact sheet."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    pages = sorted((HERE / "paper_pages").glob("page*.png"))
    if not pages:
        return
    n = len(pages)
    ncols = min(5, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 4.1 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, p in zip(axes, pages):
        ax.imshow(mpimg.imread(p))
        ax.set_title(p.stem, fontsize=8)
        ax.axis("off")
    for ax in axes[len(pages) :]:
        ax.axis("off")
    fig.suptitle("Kibler et al., Sci. Rep. 2, 463 (2012) — Kuznetsov-Ma soliton", fontsize=11)
    fig.tight_layout()
    out = HERE / "paper_pages" / "contact_sheet.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    validate()
    make_contact_sheet()
