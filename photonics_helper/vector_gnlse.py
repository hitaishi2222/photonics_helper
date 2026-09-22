"""Vector (polarization-coupled) GNLSE solver.

Split-step Fourier solver for the coupled GNLSE on two polarization
components. The scalar :class:`~photonics_helper.gnlse.SplitStepEngine`
models one linearly-polarized channel; the physics that only exists when
the field is genuinely the two-component vector ``A = (A_x, A_y)`` lives
here:

- **birefringent propagation** — per-axis Taylor dispersion and differential
  group delay (polarization walk-off);
- **cross-phase modulation** — the ``2/3`` anisotropy coefficient of the
  degenerate linearly-polarized mode pair;
- **coherent polarization FWM** — the ``(i/3)γ A⊥²A*`` mixing term with the
  birefringence phase mismatch ``Δβ`` (opt-in; it oscillates away in real
  high-birefringence fiber);
- **Manakov averaging** — the ``8/9`` polarization-averaged nonlinearity
  (Wai & Menyuk 1996), with a **random-birefringence** engine that applies
  the local nonlinear step in a randomly rotated polarization frame.

Scalar-limit contract: with ``A_y ≡ 0``, ``coupling="incoherent"`` and
identical split-step structure, this engine reduces to the scalar
:class:`~photonics_helper.gnlse.SplitStepEngine` to machine precision
(enforced by the test suite), so the scalar engine remains the tool of
record for the single-mode/single-polarization regime and all validated
scalar reproductions are unaffected.

Physics scope (v1)
------------------
- Raman uses the *scalar* per-channel response ``P_j = (1−f_R)|A_j|² +
  f_R h_R ⊛ |A_j|²``; the full vector Raman response (Lin & Agrawal,
  *Opt. Lett.* **31**, 3086 (2006)) is future work.
- Self-steepening, TPA and free carriers are scalar-engine features and
  raise a clear error when requested here.
- Fixed-step propagation (``num_steps`` / ``step_size``), like the
  scalar engine's deterministic paths.

References
----------
G. P. Agrawal, *Nonlinear Fiber Optics*, 5th ed., §6.1–6.3 (coupled GNLSE);
P. K. A. Wai & C. R. Menyuk, *J. Lightw. Technol.* **14**, 148 (1996)
(random evolution and the Manakov model); C. Marcos Marcos, de Sterke &
Sipe et al., *Opt. Express* **19**, 553 (2011).
"""

from __future__ import annotations

import warnings
from math import factorial
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from photonics_helper.base import Length
from photonics_helper.gnlse import (
    BetasUnit,
    FiberProfile,
    _delayed_h_R,
    _normalize_betas,
    _raman_polarization,
    _response_fR,
)

if TYPE_CHECKING:  # pragma: no cover
    from photonics_helper.pulse import TemporalGrid, Wave

__all__ = [
    "MANAKOV_FACTOR",
    "Coupling",
    "RandomBirefringenceEngine",
    "VectorSplitStepEngine",
]

#: Polarization-averaged nonlinearity of the Manakov equation
#: (Wai & Menyuk 1996; Agrawal §6.1): a randomly birefringent fiber with
#: beat length ≪ nonlinear length behaves, on average, like a scalar NLSE
#: driven at ``8/9 γ``. Exposed for tests and user arithmetic.
MANAKOV_FACTOR = 8.0 / 9.0

Coupling = Literal["incoherent", "coherent", "manakov"]

_XPM_FACTOR = 2.0 / 3.0  # degenerate linearly-polarized mode pair (Agrawal §6.3)
_FWM_FACTOR = 1.0 / 3.0  # coherent polarization-FWM mixing strength


class VectorSplitStepEngine:
    """Split-step engine for the coupled two-polarization GNLSE.

    Parameters
    ----------
    pulse_x, pulse_y : Wave
        Envelope fields of the two polarization components
        (``envelope_field`` is used, so ``Wave.with_field`` overrides are
        honoured). Both waves must share one :class:`TemporalGrid`
        (``pulse_y.grid is pulse_x.grid``).
    fiber : FiberProfile
        Shared scalar parameters (``n2``, ``alpha``, ``A_eff``,
        ``confinement_factor``, ``raman_response``, ``length``). Every GNLSE
        γ here is the scalar ``γ`` of this profile.
    betas : array_like, optional
        Taylor coefficients ``[β₂, β₃, …]`` used by *both* axes.
    betas_x, betas_y : array_like, optional
        Per-axis dispersion (overrides *betas*). ``betas_y`` defaults to
        ``betas_x`` (degenerate axes) when only ``betas_x`` is given.
    betas_unit : {"ps^k/m", "s^k/m", "SI"}
        Unit of the dispersion coefficients, as in the scalar engine.
    coupling : {"incoherent", "coherent", "manakov"}
        Nonlinear model (see the module docstring):

        * ``"incoherent"`` (default) — coupled GNLSE with the ``2/3`` XPM
          factor and no coherent mixing: the standard model for
          high-birefringence (PM) fiber, where the FWM term averages out.
          Reduces exactly to the scalar engine when ``A_y ≡ 0``.
        * ``"coherent"`` — adds the polarization four-wave-mixing term
          ``(i/3)γ A⊥²A* e^{∓2iΔβz}``; requires ``delta_beta``. Physically
          meaningful only when the beat length is comparable to or longer
          than the nonlinear length.
        * ``"manakov"`` — polarization-averaged model with the ``8/9``
          factor. Requires identical per-axis dispersion and zero walkoff.
    delta_beta : float
        Birefringent phase mismatch ``β_x − β_y`` (rad/m). ``"coherent"``
        only.
    walkoff : float
        Differential group delay ``β₁_y − β₁_x`` in the retarded frame of
        the x axis (SI, s/m). A y-pulse drifts by ``walkoff · z``.
    include_raman : bool
        Per-channel scalar Raman response.
    step_size : Length | None
        Fixed step size (m). ``None`` uses ``length/num_steps``.
    """

    _SUPPORTED_COUPLING = ("incoherent", "coherent", "manakov")

    def __init__(
        self,
        pulse_x: Wave,
        pulse_y: Wave,
        fiber: FiberProfile,
        betas: NDArray | None = None,
        *,
        betas_x: NDArray | None = None,
        betas_y: NDArray | None = None,
        betas_unit: BetasUnit = "ps^k/m",
        coupling: Coupling = "incoherent",
        delta_beta: float = 0.0,
        walkoff: float = 0.0,
        include_raman: bool = False,
        step_size: Length | None = None,
    ):
        if pulse_y.grid is not pulse_x.grid:
            raise ValueError(
                "pulse_x and pulse_y must share the same TemporalGrid object "
                "(pulse_y.grid is pulse_x.grid). Build both waves on one grid."
            )
        if coupling not in self._SUPPORTED_COUPLING:
            raise ValueError(
                f"Unknown coupling mode {coupling!r}; accepted values are "
                f"{', '.join(repr(c) for c in self._SUPPORTED_COUPLING)}."
            )
        if betas is None and betas_x is None:
            raise ValueError("Provide betas, or per-axis betas_x / betas_y.")

        bx = _normalize_betas(betas if betas_x is None else betas_x, betas_unit)
        by = _normalize_betas(betas if betas_y is None else betas_y, betas_unit)

        if coupling == "manakov":
            if not np.array_equal(bx, by):
                raise ValueError(
                    "coupling='manakov' presumes polarization-averaged "
                    "birefringence: betas_x and betas_y must be identical. "
                    "Use two distinct arrays with coupling='incoherent' for "
                    "a deterministic PM-fiber model."
                )
            if walkoff != 0.0:
                raise ValueError(
                    "coupling='manakov' presumes zero walkoff (random axes "
                    f"average it out); got walkoff={walkoff!r} s/m."
                )
        if coupling == "coherent" and delta_beta == 0.0:
            warnings.warn(
                "coupling='coherent' with delta_beta=0: the polarization "
                "FWM term is fully resonant and will exchange power between "
                "the axes without bound. This is physical only for "
                "zero-birefringence fiber.",
                UserWarning,
                stacklevel=2,
            )
        if delta_beta != 0.0 and coupling != "coherent":
            raise ValueError(
                "delta_beta applies only to coupling='coherent'; got "
                f"coupling={coupling!r} with delta_beta={delta_beta!r}."
            )
        if walkoff != 0.0 and coupling == "manakov":  # unreachable; kept explicit
            raise ValueError("coupling='manakov' requires walkoff == 0")
        if include_raman and fiber.raman_response is None:
            raise ValueError(
                "include_raman=True but fiber.raman_response is None. "
                "Provide a RamanResponse or set include_raman=False."
            )
        if fiber.length.as_m <= 0.0:
            raise ValueError(
                f"fiber.length must be positive, got {fiber.length!r}"
            )
        if step_size is not None and step_size.as_m <= 0.0:
            raise ValueError(f"step_size must be positive, got {step_size!r}")

        self.pulse_x = pulse_x
        self.pulse_y = pulse_y
        self.fiber = fiber
        self.betas_x = bx
        self.betas_y = by
        self.coupling: Coupling = coupling
        self.delta_beta = float(delta_beta)
        self.walkoff = float(walkoff)  # SI s/m (β₁_y − β₁_x)
        self.include_raman = include_raman
        self.step_size = step_size

        self.grid: TemporalGrid = pulse_x.grid
        self.omega0 = pulse_x.central_frequency
        self.A_x = np.array(pulse_x.envelope_field, dtype=complex)
        self.A_y = np.array(pulse_y.envelope_field, dtype=complex)
        if self.A_x.shape != self.A_y.shape:
            raise ValueError(
                f"Field shapes differ between axes: {self.A_x.shape} vs "
                f"{self.A_y.shape}."
            )

        self.evolution_x: list[Wave] = []
        self.evolution_y: list[Wave] = []
        self._z_positions: list[float] = [0.0]
        self._current_z: float = 0.0
        self._energy_vs_z: list[float] | None = None
        self._spectra_vs_z: tuple[NDArray, NDArray] | None = None
        self._h_R_fft_cache: NDArray | None = None

    # ------------------------------------------------------------------
    # step helpers
    # ------------------------------------------------------------------

    def _gamma_v(self) -> float:
        """Scalar γ of the shared profile, at the carrier frequency."""
        from photonics_helper.gnlse import _gamma

        return _gamma(
            self.fiber.n2,
            self.omega0,
            self.fiber.A_eff,
            self.fiber.confinement_factor,
        )

    def _get_h_R_fft(self) -> NDArray:
        """Cached FFT of the Raman response ``h_R(t)`` on ``grid.t``."""
        if self._h_R_fft_cache is None:
            response = self.fiber.raman_response
            assert response is not None  # validated in __init__
            h_R = _delayed_h_R(response, self.grid.t)
            self._h_R_fft_cache = self.grid.fft(h_R)
        return self._h_R_fft_cache

    def _channel_intensity(self, intensity: NDArray) -> NDArray:
        """Kerr + (scalar) Raman driving intensity ``P_j(|A_j|²)``."""
        if self.include_raman and self.fiber.raman_response is not None:
            fR = _response_fR(self.fiber.raman_response)
            return _raman_polarization(intensity, fR, self.grid, self._get_h_R_fft())
        return intensity

    def _linear_step_channel(
        self,
        field: NDArray,
        dz: float,
        betas: NDArray,
        *,
        apply_walkoff: bool,
    ) -> NDArray:
        """Linear step (dispersion + loss + optional walkoff) for one channel.

        Same dispersion convention as the scalar engine:
        ``(+i) Σ β_k Ω^k/k! · dz`` with Ω in rad/ps and β_k in ps^k/m.
        """
        f_w = self.grid.fft(field)
        omega_ps = self.grid.w * 1e-12  # rad/s → rad/ps

        phi = np.zeros_like(omega_ps, dtype=float)
        for k, beta_k in enumerate(betas, start=2):
            phi += beta_k * omega_ps**k / factorial(k)
        phi *= dz
        if apply_walkoff and self.walkoff != 0.0:
            # Retarded frame of the x axis: ∂A_y/∂z gains the drift term
            # −Δβ₁ ∂A_y/∂T. Inside the (+iφ) convention this is a
            # first-order Taylor term β₁Ω applied with a minus sign.
            phi -= self.walkoff * self.grid.w * dz
        f_w = f_w * np.exp(1j * phi)

        alpha = self.fiber.alpha
        if alpha > 0:
            f_w = f_w * np.exp(-alpha * dz / 2)
        return np.asarray(self.grid.ifft(f_w), dtype=complex)

    def _coupled_nonlinear_step(
        self, Ax: NDArray, Ay: NDArray, dz: float
    ) -> tuple[NDArray, NDArray]:
        """Coupled nonlinear step for both channels, at ``self._current_z``.

        Diagonal SPM/XPM terms advance exactly (unitary phase rotation). In
        ``coherent`` mode the non-diagonal polarization-FWM mixing term is
        advanced with frequency-domain RK4 substeps, Strang-split around the
        exact diagonal phase (same structure the scalar engine uses for the
        shock term): the mixing term is ``O(γP/3)`` while the diagonal phase
        is ``O(γP)``, so no stage integrates the stiff part.
        """
        gamma = self._gamma_v()
        Px = self._channel_intensity(np.abs(Ax) ** 2)
        Py = self._channel_intensity(np.abs(Ay) ** 2)

        if self.coupling == "manakov":
            D = MANAKOV_FACTOR * gamma * (Px + Py)
            ph = np.exp(1j * D * dz)
            return Ax * ph, Ay * ph

        Dx = gamma * (Px + _XPM_FACTOR * Py)
        Dy = gamma * (Py + _XPM_FACTOR * Px)

        if self.coupling == "incoherent":
            ph_x = np.exp(1j * Dx * dz)
            ph_y = np.exp(1j * Dy * dz)
            return Ax * ph_x, Ay * ph_y

        # ---- coherent mode ------------------------------------------------
        # exact diagonal halves (Strang) + RK4 for the mixing term evaluated
        # at the step midpoint (Δβ dz ≪ 1 per step; documented approximation).
        z_mid = self._current_z + dz / 2.0
        mix_x = (1j * _FWM_FACTOR * gamma) * np.exp(-2j * self.delta_beta * z_mid)
        mix_y = (1j * _FWM_FACTOR * gamma) * np.exp(+2j * self.delta_beta * z_mid)

        def rhs(a: NDArray, b: NDArray):
            """Mixing RHS: A_x ← (i/3)γ A_y²A_x* e^{−2iΔβz}, and symmetric."""
            return mix_x * b * b * np.conj(a), mix_y * a * a * np.conj(b)

        def rk4_pair(ax: NDArray, ay: NDArray, h: float):
            k1x, k1y = rhs(ax, ay)
            k2x, k2y = rhs(ax + 0.5 * h * k1x, ay + 0.5 * h * k1y)
            k3x, k3y = rhs(ax + 0.5 * h * k2x, ay + 0.5 * h * k2y)
            k4x, k4y = rhs(ax + h * k3x, ay + h * k3y)
            return (
                ax + (h / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x),
                ay + (h / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y),
            )

        # substep the mixing contribution: keep Δφ_mixing ≲ 0.05 rad per
        # substep, AND resolve the mismatch oscillation 2Δβ·dz itself (the
        # phase factors are frozen at the step midpoint, so 2Δβ dz must be
        # well below 1 rad inside the substep grid or the mixing term gets a
        # systematic phase error that loses the filament).
        mix_rate = max(
            float(np.max(np.abs(mix_x) * np.abs(Ax * Ay))),
            float(np.max(np.abs(mix_y) * np.abs(Ax * Ay))),
            1e-30,
        )
        dbeta_rate = 2.0 * abs(self.delta_beta)   # mismatch phase rate (rad/m)
        n_sub = int(
            min(
                2000,
                max(
                    1,
                    int(np.ceil(mix_rate * dz / 0.05)),
                    int(np.ceil(dbeta_rate * dz / 0.05)),
                ),
            )
        )
        h_sub = dz / n_sub

        ax = Ax * np.exp(0.5j * Dx * dz)
        ay = Ay * np.exp(0.5j * Dy * dz)
        for _ in range(n_sub):
            ax, ay = rk4_pair(ax, ay, h_sub)
        # second Strang half: the diagonal phase completes around the mixing
        # substep (missed before: coherent runs decayed at half the
        # nonlinearity and lost the soliton entirely).
        ax = ax * np.exp(0.5j * Dx * dz)
        ay = ay * np.exp(0.5j * Dy * dz)
        return ax, ay

    # ------------------------------------------------------------------
    # propagation
    # ------------------------------------------------------------------

    def propagate(
        self,
        num_steps: int,
        *,
        nsaves: int | None = None,
        show_progress: bool = False,
    ) -> None:
        """Run the coupled split-step propagation for ``num_steps`` steps.

        Snapshot semantics match
        :meth:`~photonics_helper.gnlse.SplitStepEngine.propagate`:
        ``nsaves`` evenly spaced snapshots (including ``z=0`` and ``z=L``);
        with ``nsaves=None`` every step is saved. A total-photon-number
        monitor warns on >5% drift in lossless runs.
        """
        length = self.fiber.length.as_m
        if self.step_size is not None:
            dz = self.step_size.as_m
            n_steps = int(round(length / dz))
            dz = length / n_steps  # exact fit to the fiber end
        else:
            if num_steps < 1:
                raise ValueError(f"num_steps must be >= 1, got {num_steps}")
            dz = length / num_steps
            n_steps = num_steps

        save_z = np.linspace(0.0, length, nsaves) if nsaves is not None else None
        next_save_idx = 1 if nsaves is not None else None

        if show_progress:
            try:
                from tqdm import tqdm
            except ImportError as exc:
                raise ImportError(
                    "show_progress=True requires tqdm. Install with: pip install tqdm"
                ) from exc
            step_iter = tqdm(range(n_steps), desc="vector GNLSE (random)", unit="step")
        else:
            step_iter = range(n_steps)

        self.evolution_x = []
        self.evolution_y = []
        self._z_positions = [0.0]
        self._energy_vs_z = []

        def _snapshot() -> None:
            from photonics_helper.pulse import Wave

            wx = Wave(
                grid=self.grid,
                envelope=self.pulse_x.envelope,
                central_wavelength=self.pulse_x.central_wavelength,
            )
            wx._pulse_train_field = self.A_x.copy()
            wy = Wave(
                grid=self.grid,
                envelope=self.pulse_y.envelope,
                central_wavelength=self.pulse_y.central_wavelength,
            )
            wy._pulse_train_field = self.A_y.copy()
            assert self._energy_vs_z is not None
            self.evolution_x.append(wx)
            self.evolution_y.append(wy)
            total = (
                float(
                    np.sum(np.abs(self.A_x) ** 2 + np.abs(self.A_y) ** 2)
                    * self.grid.dt
                )
            )
            self._energy_vs_z.append(total)

        _snapshot()  # z = 0

        for step in step_iter:
            self._current_z = step * dz
            # Strang split: half linear → coupled nonlinear → half linear.
            self.A_x = self._linear_step_channel(
                self.A_x, dz / 2, self.betas_x, apply_walkoff=False
            )
            self.A_y = self._linear_step_channel(
                self.A_y, dz / 2, self.betas_y, apply_walkoff=True
            )
            self.A_x, self.A_y = self._coupled_nonlinear_step(
                self.A_x, self.A_y, dz
            )
            self.A_x = self._linear_step_channel(
                self.A_x, dz / 2, self.betas_x, apply_walkoff=False
            )
            self.A_y = self._linear_step_channel(
                self.A_y, dz / 2, self.betas_y, apply_walkoff=True
            )
            z = step * dz + dz
            self._z_positions.append(z)

            if save_z is None:
                _snapshot()
            else:
                while (
                    next_save_idx is not None
                    and next_save_idx < len(save_z)
                    and z >= save_z[next_save_idx] - 1e-15
                ):
                    _snapshot()
                    next_save_idx += 1

        if save_z is not None:
            assert next_save_idx is not None
            assert next_save_idx is not None and nsaves is not None
            while next_save_idx < nsaves:  # catch the final snapshot at z = L
                _snapshot()
                next_save_idx += 1
            self._z_positions = save_z[: len(self.evolution_x)].tolist()

        e0, e1 = self._energy_vs_z[0], self._energy_vs_z[-1]
        if e0 > 0.0 and self.fiber.alpha == 0.0:
            drift = abs(e1 / e0 - 1.0)
            if drift > 0.05:
                warnings.warn(
                    f"vector GNLSE total-energy drift {drift * 100:.2f}% "
                    "exceeds 5% with loss disabled; increase num_steps or "
                    "reduce step_size.",
                    UserWarning,
                    stacklevel=2,
                )

        spectra = np.zeros((len(self.evolution_x), self.grid.N))
        for i, wx in enumerate(self.evolution_x):
            spectra[i] = (
                np.abs(self.grid.fft(wx.envelope_field)) ** 2
                + np.abs(self.grid.fft(self.evolution_y[i].envelope_field)) ** 2
            )
        self._spectra_vs_z = (self.grid.w, spectra)

    # ------------------------------------------------------------------
    # readouts
    # ------------------------------------------------------------------

    @property
    def z_array(self) -> NDArray:
        """Propagation distances (m) for each saved snapshot."""
        return np.asarray(self._z_positions, dtype=float)

    def fields_vs_z(self) -> tuple[NDArray, NDArray]:
        """Complex field histories: ``(A_x(z, t), A_y(z, t))``.

        Returns
        -------
        (NDArray, NDArray)
            Arrays of shape ``(n_saves, N)``:: both channels' envelopes at
            every saved snapshot.
        """
        ax = np.array([w.envelope_field for w in self.evolution_x], dtype=complex)
        ay = np.array([w.envelope_field for w in self.evolution_y], dtype=complex)
        return ax, ay

    @property
    def energy_vs_z(self) -> NDArray:
        """Total pulse energy ``∫(|A_x|²+|A_y|²)dt`` at each snapshot."""
        if self._energy_vs_z is None:
            raise RuntimeError("Call propagate() first.")
        return np.asarray(self._energy_vs_z, dtype=float)

    @property
    def spectra_vs_z(self) -> tuple[NDArray, NDArray]:
        """``(ω [rad/s], summed spectrum |A_x|²+|A_y|²)`` per snapshot."""
        if self._spectra_vs_z is None:
            raise RuntimeError("Call propagate() first.")
        return self._spectra_vs_z

    def nonlinear_phase_measure(self, index: Literal[0, 1] = 0) -> float:
        """Nonlinear phase of channel *index*, measured against the input CW.

        Meaningful for CW or constant-amplitude tests; returns
        ``arg(A_z / A_0)`` at the last snapshot.
        """
        if not self.evolution_x:
            raise RuntimeError("Call propagate() first.")
        ref = self.A_x if index == 0 else self.A_y
        z0 = self.pulse_x if index == 0 else self.pulse_y
        ratio = ref / np.asarray(z0.envelope_field, dtype=complex)
        return float(np.angle(ratio[0]))


class RandomBirefringenceEngine(VectorSplitStepEngine):
    """Coupled GNLSE with a random polarisation-frame evolution.

    Segments of length ``segment_length`` (default: a fixed step ``dz``) are
    each propagated with :class:`VectorSplitStepEngine`'s ``incoherent``
    nonlinearity in a randomly rotated polarisation frame: the field is
    transformed by a random ``SU(2)`` rotation (uniform axis on the Poincaré
    sphere, uniform rotation angle), the local coupled-nonlinear step is
    applied, and the field is rotated back. Averaged over segments and
    seeds this reproduces the polarization-averaged Manakov model with the
    ``8/9`` effective nonlinearity (Wai & Menyuk 1996).

    This is the engine to use for validating the Manakov limit: identical
    ensemble spectra to the deterministic Manakov run, depolarization of
    the mean field, exact total-energy conservation at every step.
    """

    def __init__(
        self,
        pulse_x: Wave,
        pulse_y: Wave,
        fiber: FiberProfile,
        betas: NDArray,
        *,
        include_raman: bool = False,
        step_size: Length | None = None,
        seed: int | None = None,
    ):
        super().__init__(
            pulse_x,
            pulse_y,
            fiber,
            betas,
            coupling="incoherent",
            include_raman=include_raman,
            step_size=step_size,
        )
        self.seed = seed
        self._rotation_history: list[float] = []

    @staticmethod
    def _random_su2(rng: np.random.Generator) -> NDArray:
        """Random ``SU(2)`` rotation: uniform axis `n̂` on the sphere, angle φ."""
        u = rng.normal(size=3)
        norm = float(np.linalg.norm(u))
        if norm == 0.0:  # pragma: no cover — astronomically unlikely
            u = np.array([1.0, 0.0, 0.0])
            norm = 1.0
        n = u / norm
        theta = rng.uniform(0.0, 2.0 * np.pi)
        cx, cy, cz = n
        # σ · n̂ = [[cz, cx − i cy], [cx + i cy, −cz]]
        s = np.array(
            [
                [cz, cx - 1j * cy],
                [cx + 1j * cy, -cz],
            ]
        )
        rot = np.cos(theta) * np.eye(2) - 1j * np.sin(theta) * s
        return np.asarray(rot, dtype=complex)

    def propagate(
        self,
        num_steps: int,
        *,
        nsaves: int | None = None,
        show_progress: bool = False,
    ) -> None:
        """Randomly rotated propagation: same signature as the base
        :meth:`VectorSplitStepEngine.propagate`, with an ``SU(2)`` frame
        rotation before/after every nonlinear step.
        """
        rng = np.random.default_rng(self.seed)
        length = self.fiber.length.as_m
        if self.step_size is not None:
            dz = self.step_size.as_m
            n_steps = int(round(length / dz))
            dz = length / n_steps
        else:
            if num_steps < 1:
                raise ValueError(f"num_steps must be >= 1, got {num_steps}")
            n_steps = num_steps
            dz = length / num_steps

        save_z = np.linspace(0.0, length, nsaves) if nsaves is not None else None
        next_save_idx = 1 if nsaves is not None else None

        if show_progress:
            try:
                from tqdm import tqdm
            except ImportError as exc:
                raise ImportError(
                    "show_progress=True requires tqdm. Install with: pip install tqdm"
                ) from exc
            step_iter = tqdm(range(n_steps), desc="vector GNLSE", unit="step")
        else:
            step_iter = range(n_steps)

        self.evolution_x = []
        self.evolution_y = []
        self._z_positions = [0.0]
        self._energy_vs_z = []

        def _snapshot() -> None:
            from photonics_helper.pulse import Wave

            wx = Wave(
                grid=self.grid,
                envelope=self.pulse_x.envelope,
                central_wavelength=self.pulse_x.central_wavelength,
            )
            wx._pulse_train_field = self.A_x.copy()
            wy = Wave(
                grid=self.grid,
                envelope=self.pulse_y.envelope,
                central_wavelength=self.pulse_y.central_wavelength,
            )
            wy._pulse_train_field = self.A_y.copy()
            assert self._energy_vs_z is not None
            self.evolution_x.append(wx)
            self.evolution_y.append(wy)
            self._energy_vs_z.append(
                float(
                    np.sum(np.abs(self.A_x) ** 2 + np.abs(self.A_y) ** 2)
                    * self.grid.dt
                )
            )

        _snapshot()

        for step in step_iter:
            self._current_z = step * dz
            # half linear
            self.A_x = self._linear_step_channel(
                self.A_x, dz / 2, self.betas_x, apply_walkoff=False
            )
            self.A_y = self._linear_step_channel(
                self.A_y, dz / 2, self.betas_y, apply_walkoff=False
            )
            # rotate to a random local polarisation frame (per time sample:
            # the SU(2) matrix mixes the two channel arrays pointwise),
            # apply the coupled nonlinearity there, and rotate back.
            rot = self._random_su2(rng)
            r00, r01, r10, r11 = rot[0, 0], rot[0, 1], rot[1, 0], rot[1, 1]
            local_x = r00 * self.A_x + r01 * self.A_y
            local_y = r10 * self.A_x + r11 * self.A_y
            local_x, local_y = self._coupled_nonlinear_step(local_x, local_y, dz)
            # rotate back: the inverse of an SU(2) matrix is its adjoint
            c00, c01, c10, c11 = (r00.conjugate(), r10.conjugate(), r01.conjugate(), r11.conjugate())
            new_x = c00 * local_x + c01 * local_y
            new_y = c10 * local_x + c11 * local_y
            self.A_x, self.A_y = new_x, new_y

            # trailing half linear
            self.A_x = self._linear_step_channel(
                self.A_x, dz / 2, self.betas_x, apply_walkoff=False
            )
            self.A_y = self._linear_step_channel(
                self.A_y, dz / 2, self.betas_y, apply_walkoff=False
            )

            z = step * dz + dz
            self._z_positions.append(z)
            if save_z is None:
                _snapshot()
            else:
                while (
                    next_save_idx is not None
                    and next_save_idx < len(save_z)
                    and z >= save_z[next_save_idx] - 1e-15
                ):
                    _snapshot()
                    next_save_idx += 1
        if save_z is not None:
            assert next_save_idx is not None
            assert next_save_idx is not None and nsaves is not None
            while next_save_idx < nsaves:  # final snapshot at z = L
                _snapshot()
                next_save_idx += 1
            self._z_positions = save_z[: len(self.evolution_x)].tolist()

        e0, e1 = self._energy_vs_z[0], self._energy_vs_z[-1]
        if e0 > 0.0 and self.fiber.alpha == 0.0:
            drift = abs(e1 / e0 - 1.0)
            if drift > 0.05:
                warnings.warn(
                    f"vector GNLSE total-energy drift {drift * 100:.2f}% "
                    "exceeds 5% with loss disabled; increase num_steps.",
                    UserWarning,
                    stacklevel=2,
                )

        spectra = np.zeros((len(self.evolution_x), self.grid.N))
        for i, wx in enumerate(self.evolution_x):
            spectra[i] = (
                np.abs(self.grid.fft(wx.envelope_field)) ** 2
                + np.abs(self.grid.fft(self.evolution_y[i].envelope_field)) ** 2
            )
        self._spectra_vs_z = (self.grid.w, spectra)
