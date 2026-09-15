"""Distributed Bragg Reflector (DBR) design and transfer-matrix simulation."""

from copy import deepcopy
from typing import Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from dataclasses import field
from pydantic.dataclasses import dataclass

from photonics_helper.base import PI, Length, Wavelength, WavelengthArray
from photonics_helper.materials import RefractiveIndex


@dataclass(config={"arbitrary_types_allowed": True})
class Material(RefractiveIndex):
    """A named material with wavelength-dependent complex refractive index."""

    _name: str = ""

    def __post_init__(self):
        if not self._name:
            object.__setattr__(self, "_name", "unnamed")

    @property
    def name(self) -> str:
        return self._name


@dataclass(config={"arbitrary_types_allowed": True})
class Block:
    """A single layer in a DBR stack.

    Attributes
    ----------
    length : Length — thickness of the layer (m).
    material : Material with refractive index data.
    colour : optional colour string for plotting.
    position : (start, end) coordinates along the stack.
    """

    length: Length  # width of single block
    material: Material  # its Material property
    colour: Optional[str] = None  # colour (Optional) for visualisation
    _position: Optional[Tuple[float, float]] = None

    def __post_init__(self) -> None:
        del self.position

    def __repr__(self) -> str:
        return f"Block:\n  length={self.length.as_m}m, material={self.material.name}, colour={self.colour} \n  position={self.position})"

    @property
    def position(self) -> Tuple[float, float] | None:
        return self._position

    @position.setter
    def position(self, value: Tuple[float, float]) -> None:
        if abs((value[1] - value[0]) - self.length.as_m) > 1e-12:
            raise ValueError("Your position is not compatable with the length of Block")
        else:
            self._position = value

    @position.deleter
    def position(self) -> None:
        self._position = (0, self.length.as_m)


@dataclass(config={"arbitrary_types_allowed": True})
class Pattern:
    """A repeating DBR layer pattern.

    Attributes
    ----------
    style : layer sequence string, e.g. "ABAB" or "ABCABC".
    mapping : dict mapping style characters to Block instances.
    central_wavelength : design central wavelength.
    length : total physical length of the pattern.
    out : ordered list of Blocks composing the pattern.
    """

    style: str  # Block style Ex: "ABAB" / "ABCABC" / "AB_AB"
    mapping: Dict[str, Block]  # Ex {"A": SiO2, "B": Si}
    central_wavelength: Wavelength  # central wavelength for DBR
    _length: float = 0  # length of DBR
    _out: List[Block] = field(default_factory=list)  # Pattern out

    def __post_init__(self) -> None:
        self.make_pattern()

    @property
    def out(self) -> List[Block]:
        return self._out

    @out.deleter
    def out(self) -> None:
        del self._out

    @property
    def length(self) -> float:
        return self._length

    def make_pattern(self) -> None:
        """Build the Block list from style and mapping."""
        self._out = []
        start_pos: float = 0
        end_pos: float = 0
        for block in self.style:
            # Clone template blocks so repeated style letters get distinct instances.
            current_block = deepcopy(self.mapping[block])
            current_block._position = None
            end_pos += current_block.length.as_m
            current_block.position = (start_pos, end_pos)
            self._out.append(current_block)
            start_pos = end_pos
        self._length = end_pos

    def add_block(self, block: Block, mapping: str, index: int) -> None:
        """Insert a block at *index* in the pattern style."""
        self.style = self.style[:index] + mapping + self.style[index:]
        if mapping not in self.mapping:
            self.mapping[mapping] = block
        self._out = []
        self.make_pattern()

    def remove_block(self, index: int) -> None:
        """Remove the block at *index* from the pattern style."""
        self.style = self.style[:index] + self.style[index + 1 :]
        self._out = []
        self.make_pattern()

    def get_index(self, wl_micron: float) -> Tuple[List[float], List[float]]:
        """Return [n, k] arrays for all blocks at the given wavelength (μm)."""
        n = []
        k = []
        for block in self.out:
            n.append(block.material.n_func(wl_micron))
            k.append(block.material.k_func(wl_micron))
        return n, k

    def _get_lengths(self) -> List[float]:
        """Return layer thicknesses."""
        return [block.length.as_m for block in self.out]

    def _get_positions(self) -> List[Tuple[float, float] | None]:
        """Return layer start/end positions."""
        return [block.position for block in self.out]

    def _get_colours(self) -> List[str | None]:
        """Return layer colours."""
        return [block.colour for block in self.out]

    def _get_names(self, type=1) -> List[str]:
        """Return layer material names (type=2 deduplicates for legends)."""
        names = [block.material.name for block in self.out]
        if type == 2:
            # for DBR legends in plots
            temp = []
            for name in names:
                if name not in temp:
                    temp.append(name)
                else:
                    temp.append(f"_{name}")
            return temp
        return names

    def plot_index(self, wl: Wavelength | None = None) -> None:
        if wl is None:
            if self.central_wavelength is None:
                raise ValueError(
                    "Wavelenth reguired[in meters]: set central wavelength (or) assign wavelength of your choice in function."
                )
            wl = self.central_wavelength
        plot_index(self, wl)

    def plot_2d(self, height=100e-9, overlay_index: bool = False) -> None:
        plot_2d(self, height=height, overlay_index=overlay_index)


def plot_index(pattren: Pattern, wl: Wavelength) -> None:
    """Plot refractive index n across the DBR pattern at wavelength *wl*."""
    n, _ = pattren.get_index(wl.as_um)
    x_min = [pos[0] for pos in pattren._get_positions() if pos]
    x_max = [pos[1] for pos in pattren._get_positions() if pos]

    fig, ax = plt.subplots()
    ax.hlines(n, x_min, x_max)
    ax.vlines(x_max[:-1], n[:-1], n[1:])

    ax.set_xlabel("DBR length")
    ax.set_ylabel("refractive index (n)")
    ax.set_xlim(0, max(x_max))
    plt.show()


def plot_2d(pattren: Pattern, height=100e-9, overlay_index: bool = False) -> None:
    """Plot a 2-D bar chart of the DBR pattern.

    Parameters
    ----------
    pattren : Pattern — the DBR pattern.
    height : bar height in metres (default 100 nm).
    overlay_index : if True, overlay refractive index on the bar chart.
    """
    lengths = pattren._get_lengths()
    colors = pattren._get_colours()
    start_points = [pos[0] for pos in pattren._get_positions() if pos]
    names = pattren._get_names(type=2)

    _, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.bar(
        start_points,
        height,
        lengths,
        align="edge",
        color=colors,
        alpha=0.8,
        label=names,
    )
    ax.set_xlabel("DBR length [m]")
    ax.tick_params("y", length=0, labelleft=False)
    ax.legend(loc="upper left", ncols=4, bbox_to_anchor=(0.5, 1.15))

    if overlay_index:
        ax1 = ax.twinx()
        ax1.set_ylabel("Refractive index (n)")

        x_min = [pos[0] for pos in pattren._get_positions() if pos]
        x_max = [pos[1] for pos in pattren._get_positions() if pos]

        if pattren.central_wavelength is None:
            raise ValueError("Wavelenth reguired[in meters]: set central wavelength")
        else:
            n, k = pattren.get_index(pattren.central_wavelength.as_um)

        ax1.hlines(n, x_min, x_max, colors="k")
        ax1.vlines(x_max[:-1], n[:-1], n[1:], colors="k")

    plt.show()


@dataclass(config={"arbitrary_types_allowed": True})
class TMM:
    """Transfer-matrix method for a DBR stack.

    Computes reflection, transmission, and field profiles using the
    2×2 transfer-matrix formalism for stratified media.

    Attributes
    ----------
    pattern : Pattern — the layer stack.
    angle_of_incidence : angle of incidence in radians.
    polarisation : "TE" or "TM".
    n_incident : complex — refractive index of the semi-infinite incident
        medium (default air, ``1.0 + 0j``).
    n_substrate : complex — refractive index of the semi-infinite exit
        (substrate) medium (default air, ``1.0 + 0j``).
    """

    pattern: Pattern
    angle_of_incidence: float
    polarisation: Literal["TE", "TM"]
    n_incident: complex = 1.0 + 0j
    n_substrate: complex = 1.0 + 0j

    def _layer_matrix(
        self, letter_asigned: str, wavelength: Wavelength, thickness: float
    ) -> NDArray:
        """2×2 characteristic matrix for a slab of ``thickness`` metres.

        Uses the standard optical admittance formalism which correctly
        handles absorbing (complex n) layers. For lossless media this
        reduces to the usual cos/sin form. Snell's law uses the configured
        incident medium index.

        References
        ----------
        - Macleod, *Thin-Film Optical Filters*, 4th ed., §2.4–2.5 (optical
          admittance and the characteristic matrix).
        - Born & Wolf, *Principles of Optics*, 7th ed., §1.6 (Snell's law and
          Fresnel admittances for absorbing media).
        """
        mat = self.pattern.mapping[letter_asigned].material
        n_real: float = mat.n_func(wavelength.as_um)
        n_imag: float = mat.k_func(wavelength.as_um)
        n: complex = n_real + 1j * n_imag
        d: float = thickness
        # Phase thickness with complex n
        # Snell's law: n_{layer} sin(θ_layer) = n_incident sin(θ_incident)
        sin_theta_layer = self.n_incident * np.sin(self.angle_of_incidence) / n
        if np.isrealobj(sin_theta_layer):
            sin_theta_layer = np.clip(sin_theta_layer, -1.0, 1.0)
        cos_theta_layer = np.sqrt(1.0 - sin_theta_layer**2)
        delta = 2 * PI * n * d * cos_theta_layer / wavelength.as_m

        # Optical admittance η = n * cos(θ_layer) for TE, n / cos(θ_layer) for TM
        if self.polarisation == "TE":
            eta = n * cos_theta_layer
        else:  # TM
            eta = n / cos_theta_layer

        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        return np.array(
            [[cos_d, -1j * sin_d / eta], [-1j * eta * sin_d, cos_d]],
            dtype=complex,
        )

    def _characteristic_matrix(
        self, letter_asigned: str, wavelength: Wavelength
    ) -> NDArray:
        """2×2 characteristic matrix for one full layer of the pattern."""
        return self._layer_matrix(
            letter_asigned,
            wavelength,
            self.pattern.mapping[letter_asigned].length.as_m,
        )

    def transfer_matrix(self, wavelength: Wavelength) -> NDArray:
        """Compute the full 2×2 characteristic matrix for the stack.

        Uses the optical admittance formalism so that absorbing layers
        (complex n) are handled correctly.  The returned matrix M relates
        the tangential E and H fields at the back and front of the stack::

            [E_front]   [M11  M12] [E_back]
            [H_front] = [M21  M22] [H_back]

        Each layer's characteristic matrix maps the fields at its *back*
        interface to its *front* interface, so the stack matrix is the
        ordered product ``M_1 @ M_2 @ ... @ M_n`` with layer 1 at the
        incident side (Macleod, *Thin-Film Optical Filters*, Eq. 2.55).
        """
        M_total = np.identity(2, dtype=complex)
        for letter in self.pattern.style:
            M_layer = self._characteristic_matrix(letter, wavelength)
            M_total = M_total @ M_layer
        return M_total

    def spectrum(self, wavelengths: WavelengthArray) -> tuple[NDArray, NDArray]:
        """Compute reflection R(λ) and transmission T(λ) across a wavelength range."""
        R = np.empty_like(wavelengths.as_m, dtype=float)
        T = np.empty_like(wavelengths.as_m, dtype=float)

        # Optical admittances of the configured incident and exit media
        eta0 = self._admittance(self.n_incident, self.angle_of_incidence)
        eta_exit = self._admittance(self.n_substrate, self.angle_of_incidence)

        for i, wl_m in enumerate(wavelengths.as_m):
            wl = Wavelength(wl_m, "m")
            M = self.transfer_matrix(wl)

            M11, M12, M21, M22 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
            # Characteristic matrix formalism: r and t from admittance boundary
            denom = eta0 * M11 + eta0 * eta_exit * M12 + M21 + eta_exit * M22
            r = (eta0 * M11 + eta0 * eta_exit * M12 - M21 - eta_exit * M22) / denom
            t = 2.0 * eta0 / denom
            R[i] = np.abs(r) ** 2
            T[i] = (eta_exit.real / eta0.real) * np.abs(t) ** 2

        return R, T

    def _admittance(self, n: complex, theta_incident: float) -> complex:
        """Optical admittance η = n·cos(θ) for TE, n/cos(θ) for TM.

        ``theta_incident`` is the angle in the incident medium; Snell's law is
        applied with the configured ``n_incident`` for both polarisations, so
        the result is consistent for any incident/exit medium index
        (Macleod, *Thin-Film Optical Filters*, §2.4; Born & Wolf,
        *Principles of Optics*, §1.6).
        """
        sin_t = self.n_incident * np.sin(theta_incident) / n
        if np.isrealobj(sin_t):
            sin_t = np.clip(sin_t, -1.0, 1.0)
        cos_t = np.sqrt(1.0 - sin_t**2)
        if self.polarisation == "TE":
            return complex(n * cos_t)
        else:
            return complex(n / cos_t)

    def _reflection_coefficient(self, wavelength: Wavelength) -> complex:
        """Overall reflection coefficient via characteristic matrix formalism."""
        M = self.transfer_matrix(wavelength)
        eta0 = self._admittance(self.n_incident, self.angle_of_incidence)
        eta_exit = self._admittance(self.n_substrate, self.angle_of_incidence)
        M11, M12, M21, M22 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
        denom = eta0 * M11 + eta0 * eta_exit * M12 + M21 + eta_exit * M22
        return complex(
            (eta0 * M11 + eta0 * eta_exit * M12 - M21 - eta_exit * M22) / denom
        )

    def field_profile(
        self, wavelength: Wavelength, return_positions: bool = False
    ) -> NDArray | tuple[list[Length], NDArray]:
        """Compute |E(z)| inside the stack at a single wavelength.

        Returns one value per layer: the magnitude of the total electric
        field at the *front* of each layer (just after the interface).

        Because the characteristic matrix maps fields at the back of a layer
        to its front, forward propagation through a layer uses the inverse
        relation ``solve(M_layer, [E, H])`` — multiplying by ``M_layer`` would
        propagate the wrong way.

        Parameters
        ----------
        wavelength : Wavelength
            Vacuum wavelength (m).
        return_positions : bool
            When True, also return the front-interface positions in metres.

        Returns
        -------
        |E| : NDArray
            Field magnitude at the front of each layer.
        positions : list[Length], optional
            Interface positions, only when ``return_positions=True``.
        """
        r_total = self._reflection_coefficient(wavelength)
        eta0 = self._admittance(self.n_incident, self.angle_of_incidence)

        # Normalize so that the incident forward wave has E = 1.
        # H_forward = eta0 * E_forward,  H_backward = -eta0 * E_backward
        # At the front face: E_total = E_f + E_b,  H_total = eta0*(E_f - E_b)
        # With E_f = 1, E_b = r_total:
        E_cur: complex = 1.0 + r_total
        H_cur: complex = eta0 * (1.0 - r_total)

        field_vals: List[float] = []
        positions: List[Length] = []
        layer_positions = self.pattern._get_positions()

        for idx, letter in enumerate(self.pattern.style):
            M_layer = self._characteristic_matrix(letter, wavelength)
            # Record |E| at the front of this layer
            field_vals.append(np.abs(E_cur))
            pos = layer_positions[idx]
            positions.append(Length(float(pos[0]) if pos is not None else 0.0, "m"))
            # Propagate forward through the layer: [E,H]_back = M^{-1}[E,H]_front
            E_cur, H_cur = np.linalg.solve(M_layer, np.array([E_cur, H_cur]))

        if return_positions:
            return positions, np.array(field_vals, dtype=float)
        return np.array(field_vals, dtype=float)

    def field_profile_z(
        self, wavelength: Wavelength, n_points_per_layer: int = 20
    ) -> tuple[NDArray, NDArray]:
        """Continuous ``|E(z)|`` profile sampled across the whole stack.

        Parameters
        ----------
        wavelength : Wavelength
            Vacuum wavelength (m).
        n_points_per_layer : int
            Number of samples per layer (including both interfaces). Must be
            at least 2. Default 20.

        Returns
        -------
        z : NDArray
            Positions in metres, strictly increasing from ``0`` to the total
            stack length.
        E : NDArray
            ``|E(z)|`` at those positions.

        Notes
        -----
        Within layer ``i`` the fields are obtained from the back interface via
        the characteristic matrix of the remaining thickness: at depth ``x``,
        ``[E, H] = M(d_i − x) · [E_back, H_back]``. At ``x = 0`` this
        reproduces the front field, so the sampled interface values match
        :meth:`field_profile`. The same characteristic-matrix/admittance
        formalism is used throughout (Macleod, *Thin-Film Optical Filters*,
        §2.4).
        """
        if n_points_per_layer < 2:
            raise ValueError("n_points_per_layer must be >= 2")

        r_total = self._reflection_coefficient(wavelength)
        eta0 = self._admittance(self.n_incident, self.angle_of_incidence)
        E_cur: complex = 1.0 + r_total
        H_cur: complex = eta0 * (1.0 - r_total)

        layer_positions = self.pattern._get_positions()
        z_vals: list[float] = []
        e_vals: list[float] = []

        for idx, letter in enumerate(self.pattern.style):
            d = float(self.pattern.mapping[letter].length.as_m)
            M_layer = self._layer_matrix(letter, wavelength, d)
            E_back, H_back = np.linalg.solve(M_layer, np.array([E_cur, H_cur]))

            pos = layer_positions[idx]
            z0 = float(pos[0]) if pos is not None else 0.0

            for x in np.linspace(0.0, d, n_points_per_layer):
                M_sub = self._layer_matrix(letter, wavelength, d - x)
                E_x, _ = M_sub @ np.array([E_back, H_back])
                z = z0 + float(x)
                if z_vals and np.isclose(z, z_vals[-1], rtol=0.0, atol=1e-15):
                    continue
                z_vals.append(z)
                e_vals.append(float(np.abs(E_x)))

            E_cur, H_cur = E_back, H_back

        return np.asarray(z_vals, dtype=float), np.asarray(e_vals, dtype=float)

    def plot_spectrum(self, wavelengths: WavelengthArray, ax=None):
        """Plot reflectance R(λ), transmittance T(λ) and absorption A(λ).

        Parameters
        ----------
        wavelengths : WavelengthArray — wavelength grid.
        ax : matplotlib Axes, optional — axis to draw on.

        Returns
        -------
        fig : matplotlib Figure
        """
        import matplotlib.pyplot as plt

        R, T = self.spectrum(wavelengths)
        wl_nm = wavelengths.as_nm
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure
        ax.plot(wl_nm, R, label="R", color="C0")
        ax.plot(wl_nm, T, label="T", color="C1")
        ax.plot(wl_nm, 1.0 - R - T, label="A", color="C2", alpha=0.6)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Fraction")
        ax.set_title(f"DBR spectrum ({self.polarisation}, {np.rad2deg(self.angle_of_incidence):.1f}°)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return fig
