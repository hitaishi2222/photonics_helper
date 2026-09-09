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
class Pattren:
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
    _out: List[Block] = field(default_factory=list)  # Pattren Out

    def __post_init__(self) -> None:
        self.make_pattren()

    @property
    def out(self) -> List[Block]:
        return self._out

    @out.deleter
    def out(self) -> None:
        del self._out

    @property
    def length(self) -> float:
        return self._length

    def make_pattren(self) -> None:
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
        print(self.style)
        if mapping not in self.mapping:
            self.mapping[mapping] = block
        self._out = []
        self.make_pattren()

    def remove_block(self, index: int) -> None:
        """Remove the block at *index* from the pattern style."""
        self.style = self.style[:index] + self.style[index + 1 :]
        self._out = []
        self.make_pattren()

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


def plot_index(pattren: Pattren, wl: Wavelength) -> None:
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


def plot_2d(pattren: Pattren, height=100e-9, overlay_index: bool = False) -> None:
    """Plot a 2-D bar chart of the DBR pattern.

    Parameters
    ----------
    pattren : Pattren — the DBR pattern.
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


# Public alias for the historical typo in class name.
Pattern = Pattren


@dataclass(config={"arbitrary_types_allowed": True})
class TMM:
    """Transfer-matrix method for a DBR stack.

    Computes reflection, transmission, and field profiles using the
    2×2 transfer-matrix formalism for stratified media.

    Attributes
    ----------
    pattern : Pattren — the layer stack.
    angle_of_incidence : angle of incidence in radians.
    polarisation : "TE" or "TM".
    """

    pattern: Pattren
    angle_of_incidence: float
    polarisation: Literal["TE", "TM"]

    def _characteristic_matrix(
        self, letter_asigned: str, wavelength: Wavelength
    ) -> NDArray:
        """2×2 characteristic matrix for a single layer.

        Uses the standard optical admittance formalism which correctly
        handles absorbing (complex n) layers. For lossless media this
        reduces to the usual cos/sin form.
        """
        mat = self.pattern.mapping[letter_asigned].material
        n_real: float = mat.n_func(wavelength.as_um)
        n_imag: float = mat.k_func(wavelength.as_um)
        n: complex = n_real + 1j * n_imag
        d: float = self.pattern.mapping[letter_asigned].length.as_m
        # Phase thickness with complex n
        # Snell's law for complex n: sin_theta_layer = sin_incident / n
        sin_theta_layer = np.sin(self.angle_of_incidence) / n
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

    def transfer_matrix(self, wavelength: Wavelength) -> NDArray:
        """Compute the full 2×2 characteristic matrix for the stack.

        Uses the optical admittance formalism so that absorbing layers
        (complex n) are handled correctly.  The returned matrix M relates
        the tangential E and H fields at the front and back of the stack:
            [E_front]   [M11  M12] [E_back]
            [H_front] = [M21  M22] [H_back]
        """
        M_total = np.identity(2, dtype=complex)
        for letter in self.pattern.style:
            M_layer = self._characteristic_matrix(letter, wavelength)
            M_total = M_layer @ M_total
        return M_total

    def spectrum(self, wavelengths: WavelengthArray) -> tuple[NDArray, NDArray]:
        """Compute reflection R(λ) and transmission T(λ) across a wavelength range."""
        R = np.empty_like(wavelengths.as_m, dtype=float)
        T = np.empty_like(wavelengths.as_m, dtype=float)

        # Optical admittances of incident (air) and exit (air) media
        eta0 = self._admittance(1.0 + 0j, self.angle_of_incidence)
        eta_exit = self._admittance(1.0 + 0j, 0.0)

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

    def _admittance(self, n: complex, angle: float) -> complex:
        """Optical admittance η = n·cos(θ) for TE, n/cos(θ) for TM."""
        if self.polarisation == "TE":
            cos_t = np.cos(angle)
            return n * cos_t
        else:
            sin_t = np.sin(angle) / n
            if np.isrealobj(sin_t):
                sin_t = np.clip(sin_t, -1.0, 1.0)
            cos_t = np.sqrt(1.0 - sin_t**2)
            return n / cos_t

    def _reflection_coefficient(self, wavelength: Wavelength) -> complex:
        """Overall reflection coefficient via characteristic matrix formalism."""
        M = self.transfer_matrix(wavelength)
        eta0 = self._admittance(1.0 + 0j, self.angle_of_incidence)
        eta_exit = self._admittance(1.0 + 0j, 0.0)
        M11, M12, M21, M22 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
        denom = eta0 * M11 + eta0 * eta_exit * M12 + M21 + eta_exit * M22
        return (eta0 * M11 + eta0 * eta_exit * M12 - M21 - eta_exit * M22) / denom

    def field_profile(self, wavelength: Wavelength) -> NDArray:
        """Compute |E(z)| inside the stack at a single wavelength.

        Returns one value per layer: the magnitude of the total electric
        field at the *front* of each layer (just after the interface).
        """
        r_total = self._reflection_coefficient(wavelength)
        eta0 = self._admittance(1.0 + 0j, self.angle_of_incidence)

        # Normalize so that the incident forward wave has E = 1.
        # H_forward = eta0 * E_forward,  H_backward = -eta0 * E_backward
        # At the front face: E_total = E_f + E_b,  H_total = eta0*(E_f - E_b)
        # With E_f = 1, E_b = r_total:
        E_front = 1.0 + r_total
        H_front = eta0 * (1.0 - r_total)

        field_vals: List[float] = []
        E_cur = E_front
        H_cur = H_front

        for letter in self.pattern.style:
            M_layer = self._characteristic_matrix(letter, wavelength)
            # Record |E| at the front of this layer
            field_vals.append(np.abs(E_cur))
            # Propagate through the layer
            M11, M12 = M_layer[0, 0], M_layer[0, 1]
            M21, M22 = M_layer[1, 0], M_layer[1, 1]
            E_new = M11 * E_cur + M12 * H_cur
            H_new = M21 * E_cur + M22 * H_cur
            E_cur = E_new
            H_cur = H_new
        return np.array(field_vals, dtype=float)
