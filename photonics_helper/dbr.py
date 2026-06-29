"""Distributed Bragg Reflector (DBR) design and transfer-matrix simulation."""

from typing import Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field

from photonics_helper.base import PI, Wavelength, WavelengthArray
from photonics_helper.materials import RefractiveIndex


class Material(RefractiveIndex):
    """A named material with wavelength-dependent complex refractive index."""

    def __init__(
        self, name: str, n: np.ndarray, k: np.ndarray, wl: WavelengthArray
    ) -> None:
        super().__init__(n=n, k=k, wl=wl)
        self._name = name

    @property
    def name(self) -> str:
        return self._name


@dataclass
class Block:
    """A single layer in a DBR stack.

    Attributes
    ----------
    length : thickness of the layer (m).
    material : Material with refractive index data.
    colour : optional colour string for plotting.
    position : (start, end) coordinates along the stack.
    """

    length: float  # width of single block
    material: Material  # its Material property
    colour: Optional[str] = None  # colour (Optional) for visualisation
    _position: Optional[Tuple[float, float]] = None

    def __post_init__(self) -> None:
        del self.position

    def __repr__(self) -> str:
        return f"Block:\n  length={self.length}m, material={self.material.name}, colour={self.colour} \n  position={self.position})"

    @property
    def position(self) -> Tuple[float, float] | None:
        return self._position

    @position.setter
    def position(self, value: Tuple[float, float]) -> None:
        if value[1] - value[0] != self.length:
            raise ValueError("Your position is not compatable with the length of Block")
        else:
            self._position = value

    @position.deleter
    def position(self) -> None:
        self._position = (0, self.length)


@dataclass
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
        start_pos: float = 0
        end_pos: float = 0
        for block in self.style:
            current_block = self.mapping[block]
            end_pos += current_block.length
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
        return [block.length for block in self.out]

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

    def plot_index(self, wl: float = 0) -> None:
        if wl == 0 and self.central_wavelength is None:
            raise ValueError(
                "Wavelenth reguired[in meters]: set central wavelength (or) assign wavelength of your choice in function."
            )
        elif wl != 0:
            plot_index(self, wl)
        else:
            plot_index(self, self.central_wavelength)

    def plot_2d(self, height=100e-9, overlay_index: bool = False) -> None:
        plot_2d(self, height=height, overlay_index=overlay_index)


def plot_index(pattren: Pattren, wl) -> None:
    """Plot refractive index n across the DBR pattern at wavelength *wl*."""
    n, _ = pattren.get_index(wl)
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
            n, k = pattren.get_index(pattren.central_wavelength)

        ax1.hlines(n, x_min, x_max, colors="k")
        ax1.vlines(x_max[:-1], n[:-1], n[1:], colors="k")

    plt.show()


@dataclass
class TMM:
    """Transfer-matrix method for a DBR stack.

    Computes reflection, transmission, and field profiles using the
    2×2 transfer-matrix formalism for stratified media.

    Attributes
    ----------
    pattern : Pattren — the layer stack.
    anlge_of_incidence : angle of incidence in radians.
    polarisation : "TE" or "TM".
    """

    pattern: Pattren
    anlge_of_incidence: float
    polarisation: Literal["TE", "TM"]

    def _interface_matrix(self, n1, n2, angle, pol) -> np.ndarray:
        """2×2 Fresnel interface matrix between media n1 and n2."""
        # Compute transmission angle using Snell's law
        sin_theta2 = n1 * np.sin(angle) / n2
        if np.isrealobj(sin_theta2):
            sin_theta2 = np.clip(sin_theta2, -1.0, 1.0)
        theta2 = np.arcsin(sin_theta2)

        if pol == "TE":
            r = (n1 * np.cos(angle) - n2 * np.cos(theta2)) / (
                n1 * np.cos(angle) + n2 * np.cos(theta2)
            )
            t = (2 * n1 * np.cos(angle)) / (n1 * np.cos(angle) + n2 * np.cos(theta2))
        else:  # TM
            r = (n2 * np.cos(angle) - n1 * np.cos(theta2)) / (
                n2 * np.cos(angle) + n1 * np.cos(theta2)
            )
            t = (2 * n1 * np.cos(angle)) / (n2 * np.cos(angle) + n1 * np.cos(theta2))
        return (1 / t) * np.array([[1, r], [r, 1]], dtype=complex)

    def _propagation_matrix(
        self, letter_asigned: str, wavelength: Wavelength, angle
    ) -> np.ndarray:
        """2×2 phase accumulation matrix for a single layer."""
        n: float = self.pattern.mapping[letter_asigned].material.n_func(wavelength.as_m)
        d: float = self.pattern.mapping[letter_asigned].length
        delta = 2 * PI * n * d * np.cos(self.anlge_of_incidence) / wavelength.as_m
        return np.array([[np.exp(1j * delta), 0], [0, np.exp(-1j * delta)]])

    def transfer_matrix(
        self, wavelength: float, angle: float = 0.0, polarization: str = "TE"
    ) -> np.ndarray:
        """Compute the full 2×2 transfer matrix for the stack at a given wavelength."""
        n_list, k_list = self.pattern.get_index(wavelength)
        n_complex = [n + 1j * k for n, k in zip(n_list, k_list)]

        M_total = np.identity(2, dtype=complex)

        n_prev = 1.0 + 0j  # incident medium (air)
        theta_prev = angle
        letters = list(self.pattern.style)

        for letter, n_curr in zip(letters, n_complex):
            # Interface
            M_int = self._interface_matrix(n_prev, n_curr, theta_prev, polarization)
            M_total = M_int @ M_total

            # Propagation
            wl_obj = Wavelength(wavelength, "m")
            M_prop = self._propagation_matrix(letter, wl_obj, theta_prev)
            M_total = M_prop @ M_total

            # Update angle for next layer
            sin_theta_next = n_prev * np.sin(theta_prev) / n_curr
            if np.isrealobj(sin_theta_next):
                sin_theta_next = np.clip(sin_theta_next, -1.0, 1.0)
            theta_next = np.arcsin(sin_theta_next)
            n_prev = n_curr
            theta_prev = theta_next

        # Exit interface back to air
        M_exit = self._interface_matrix(n_prev, 1.0 + 0j, theta_prev, polarization)
        M_total = M_exit @ M_total
        return M_total

    def spectrum(self, wavelengths: WavelengthArray) -> tuple[NDArray, NDArray]:
        """Compute reflection R(λ) and transmission T(λ) across a wavelength range."""
        R = np.empty_like(wavelengths.as_m, dtype=float)
        T = np.empty_like(wavelengths.as_m, dtype=float)

        n0 = 1.0 + 0j  # incident medium

        for i, wl in enumerate(wavelengths.as_um):
            M = self.transfer_matrix(
                wl, angle=self.anlge_of_incidence, polarization=self.polarisation
            )
            n_sub_real, k_sub = self.pattern.get_index(wl)
            n_sub = n_sub_real[-1] + 1j * k_sub[-1]

            M11, M12, M21, M22 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
            r = (n0 * M11 + n0 * n_sub * M12 - M21 - n_sub * M22) / (
                n0 * M11 + n0 * n_sub * M12 + M21 + n_sub * M22
            )
            t = (2 * n0) / (n0 * M11 + n0 * n_sub * M12 + M21 + n_sub * M22)
            R[i] = np.abs(r) ** 2
            T[i] = (n_sub.real / n0.real) * (np.abs(t) ** 2)

        return R, T

    def _reflection_coefficient(self, wavelength: float) -> complex:
        """Helper to compute overall reflection coefficient."""
        n0 = 1.0 + 0j
        M = self.transfer_matrix(
            wavelength, angle=self.anlge_of_incidence, polarization=self.polarisation
        )
        n_sub_real, k_sub = self.pattern.get_index(wavelength)
        n_sub = n_sub_real[-1] + 1j * k_sub[-1]
        M11, M12, M21, M22 = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
        return (n0 * M11 + n0 * n_sub * M12 - M21 - n_sub * M22) / (
            n0 * M11 + n0 * n_sub * M12 + M21 + n_sub * M22
        )

    def field_profile(self, wavelength: float) -> NDArray:
        """Compute |E(z)| inside the stack at a single wavelength - useful for cavity design."""
        n_list, k_list = self.pattern.get_index(wavelength)
        n_complex = [n + 1j * k for n, k in zip(n_list, k_list)]
        letters = list(self.pattern.style)

        # initial field vector (forward = 1, backward = overall reflected)
        r_total = self._reflection_coefficient(wavelength)
        v = np.array([1.0 + 0j, r_total], dtype=complex)

        field_vals: List[float] = []

        n_prev = 1.0 + 0j
        theta_prev = self.anlge_of_incidence

        for letter, n_curr in zip(letters, n_complex):
            # Interface
            M_int = self._interface_matrix(
                n_prev, n_curr, theta_prev, self.polarisation
            )
            v = M_int @ v

            # Record field magnitude at the start of the layer
            field_vals.append(np.abs(v[0] + v[1]))

            # Propagation through the layer
            wl_obj = Wavelength(wavelength, "m")
            M_prop = self._propagation_matrix(letter, wl_obj, theta_prev)
            v = M_prop @ v

            # Update angle for next layer
            sin_theta_next = n_prev * np.sin(theta_prev) / n_curr
            if np.isrealobj(sin_theta_next):
                sin_theta_next = np.clip(sin_theta_next, -1.0, 1.0)
            theta_next = np.arcsin(sin_theta_next)
            n_prev = n_curr
            theta_prev = theta_next

        return np.array(field_vals, dtype=float)
