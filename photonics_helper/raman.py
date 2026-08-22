"""Raman scattering module — interactive explorer for Raman effects in materials.

This module provides:
- RamanSpec: Material Raman properties with automatic derivation
- RamanResponse: Time-domain Raman response function h_R(t)
- RamanFrequencyResponse: Frequency-domain response H(Ω)
- RamanPulseInteraction: Pulse convolution with Raman response
- PumpWavelengthExplorer: Stokes/Anti-Stokes analysis
- MaterialComparison: Multi-material overlay
- RamanDatabase: SQLite CRUD for material data
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Literal, TYPE_CHECKING
from functools import cached_property
import warnings

import numpy as np
import matplotlib.pyplot as plt
from pydantic.dataclasses import dataclass
from pydantic import model_validator, Field
from numpy.typing import NDArray

from .base import (
    PI,
    C_MS,
    Wavelength,
    Energy,
    Time,
)
from .pulse import TemporalGrid, Wave
from .phonon import PhononMode
from ._fftw import convolve_full as _fftw_convolve_full

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

if TYPE_CHECKING:
    import dash


# ─── Reference Materials ─────────────────────────────────────────────────────

# Hardcoded reference materials for v1 (before DB is populated)
# These will be replaced by DB lookup when materials.db is available
RAMAN_MATERIALS = {
    "Silica": {
        "name": "Silica",
        "crystal": "Amorphous SiO2",
        "bandgap_eV": 9.0,
        "n2": 3.2e-20,
        "raman_shift_cm": 440,
        "raman_linewidth_cm": 45,
        "fR": 0.18,
        "gain_coeff": 1.0e-13,
        "lo_phonon_cm": None,
        "to_phonon_cm": None,
        "references": "Agrawal, Nonlinear Fiber Optics, 5th ed.",
    },
    "CdS": {
        "name": "CdS",
        "crystal": "Wurtzite",
        "bandgap_eV": 2.42,
        "n2": 1.5e-18,
        "raman_shift_cm": 305,
        "raman_linewidth_cm": 12,
        "fR": 0.35,
        "gain_coeff": None,
        "lo_phonon_cm": 302,
        "to_phonon_cm": 293,
        "references": "Pankove, Optical Processes in Semiconductors",
    },
    "GaAs": {
        "name": "GaAs",
        "crystal": "Zincblende",
        "bandgap_eV": 1.42,
        "n2": 2.5e-18,
        "raman_shift_cm": 292,
        "raman_linewidth_cm": 5,
        "fR": 0.07,
        "gain_coeff": None,
        "lo_phonon_cm": 292,
        "to_phonon_cm": 268,
        "references": "Adachi, Optical Properties of Crystalline and Amorphous Semiconductors",
    },
    "Diamond": {
        "name": "Diamond",
        "crystal": "Diamond cubic",
        "bandgap_eV": 5.5,
        "n2": 1.1e-19,
        "raman_shift_cm": 1332,
        "raman_linewidth_cm": 4,
        "fR": 0.0,
        "gain_coeff": 0.11,
        "lo_phonon_cm": None,
        "to_phonon_cm": None,
        "references": "Ferrari, Raman spectroscopy of graphene and graphite",
    },
    "LiNbO3": {
        "name": "LiNbO3",
        "crystal": "Rhombohedral",
        "bandgap_eV": 3.7,
        "n2": 3.0e-20,
        "raman_shift_cm": 254,
        "raman_linewidth_cm": 14,
        "fR": 0.0,
        "gain_coeff": None,
        "lo_phonon_cm": None,
        "to_phonon_cm": None,
        "references": "Fejer, Quasi-phase-matched nonlinear optics",
    },
    "As2Se3": {
        "name": "As2Se3",
        "crystal": "Amorphous",
        "bandgap_eV": 1.8,
        "n2": 3.5e-18,
        "raman_shift_cm": 310,
        "raman_linewidth_cm": 60,
        "fR": 0.55,
        "gain_coeff": None,
        "lo_phonon_cm": None,
        "to_phonon_cm": None,
        "references": "Richardson, Chalcogenide glass fibers for nonlinear optics",
    },
    "Si": {
        "name": "Si",
        "crystal": "Diamond cubic",
        "bandgap_eV": 1.12,
        "n2": 6.0e-19,
        "raman_shift_cm": 520,
        "raman_linewidth_cm": 4,
        "fR": 0.0,
        "gain_coeff": None,
        "lo_phonon_cm": None,
        "to_phonon_cm": None,
        "references": "Burstein, Infrared and Raman studies of silicon",
    },
    "Ge": {
        "name": "Ge",
        "crystal": "Diamond cubic",
        "bandgap_eV": 0.67,
        "n2": 1.2e-18,
        "raman_shift_cm": 300,
        "raman_linewidth_cm": 8,
        "fR": 0.0,
        "gain_coeff": None,
        "lo_phonon_cm": None,
        "to_phonon_cm": None,
        "references": "Nakamura, Raman scattering in germanium",
    },
    "As2S3": {
        "name": "As2S3",
        "crystal": "Amorphous",
        "bandgap_eV": 2.0,
        "n2": 4.0e-18,
        "raman_shift_cm": 345,
        "raman_linewidth_cm": 56,
        "fR": 0.15,
        "gain_coeff": 4.3e-12,
        "lo_phonon_cm": None,
        "to_phonon_cm": None,
        "references": "Xiong et al., Appl. Opt. 48, 5467 (2009); Slusher et al., JOSA B 21, 1146 (2004)",
    },
    "Si3N4": {
        "name": "Si3N4",
        "crystal": "Amorphous/hexagonal",
        "bandgap_eV": 4.9,
        "n2": 2.4e-19,
        "raman_shift_cm": 206,
        "raman_linewidth_cm": 50,
        "fR": None,
        "gain_coeff": None,
        "lo_phonon_cm": None,
        "to_phonon_cm": None,
        "references": "Zymla et al., J. Raman Spectrosc. (2025); Lacava et al., Sci. Rep. 7, 41598 (2017)",
    },
    "SiC_4H": {
        "name": "SiC_4H",
        "crystal": "Hexagonal (4H)",
        "bandgap_eV": 3.23,
        "n2": 8.0e-19,
        "raman_shift_cm": 777,
        "raman_linewidth_cm": 5,
        "fR": 0.0,
        "gain_coeff": None,
        "lo_phonon_cm": 983,
        "to_phonon_cm": 777,
        "references": "Feldman et al., Phys. Rev. 173, 787 (1968); Li et al., Phys. Rev. Appl. 19, 034083 (2023)",
    },
    "YAG": {
        "name": "YAG",
        "crystal": "Cubic garnet",
        "bandgap_eV": 6.5,
        "n2": 7.0e-20,
        "raman_shift_cm": 784,
        "raman_linewidth_cm": 8,
        "fR": 0.0,
        "gain_coeff": None,
        "lo_phonon_cm": None,
        "to_phonon_cm": None,
        "references": "Hurrell et al., Phys. Rev. 173, 851 (1968); Lamaignere et al., Opt. Mater. X 8, 100068 (2020)",
    },
    "BaTiO3": {
        "name": "BaTiO3",
        "crystal": "Tetragonal perovskite",
        "bandgap_eV": 3.2,
        "n2": 1.8e-18,
        "raman_shift_cm": 520,
        "raman_linewidth_cm": 45,
        "fR": 0.0,
        "gain_coeff": None,
        "lo_phonon_cm": None,
        "to_phonon_cm": None,
        "references": "Scalabrin et al., Phys. Status Solidi B 79, 731 (1977); Chaves et al., Phys. Rev. B 10, 3522 (1974)",
    },
    "ZBLAN": {
        "name": "ZBLAN",
        "crystal": "Amorphous fluoride",
        "bandgap_eV": 4.5,
        "n2": 1.5e-20,
        "raman_shift_cm": 580,
        "raman_linewidth_cm": 23,
        "fR": 0.06,
        "gain_coeff": 4.0e-14,
        "lo_phonon_cm": None,
        "to_phonon_cm": None,
        "references": "Petersen et al., JOSA B 28, 2310 (2011); Yan et al., JOSA B 29, 238 (2012)",
    },
    "GaN": {
        "name": "GaN",
        "crystal": "Wurtzite",
        "bandgap_eV": 3.4,
        "n2": 9.0e-19,
        "raman_shift_cm": 568,
        "raman_linewidth_cm": 4,
        "fR": 0.0,
        "gain_coeff": None,
        "lo_phonon_cm": 736,
        "to_phonon_cm": 532,
        "references": "Zeng et al., Appl. Sci. 10, 8814 (2020); Almeida et al., Photonics 6, 69 (2019)",
    },
    "AlN": {
        "name": "AlN",
        "crystal": "Wurtzite",
        "bandgap_eV": 6.2,
        "n2": 2.3e-19,
        "raman_shift_cm": 658,
        "raman_linewidth_cm": 1,
        "fR": 0.0,
        "gain_coeff": None,
        "lo_phonon_cm": 895,
        "to_phonon_cm": 615,
        "references": "Pandit et al., J. Appl. Phys. 102, 113508 (2007); Jung, Tang, Nanophotonics 5, 264 (2016)",
    },
    "InP": {
        "name": "InP",
        "crystal": "Zincblende",
        "bandgap_eV": 1.34,
        "n2": 2.0e-17,
        "raman_shift_cm": 345,
        "raman_linewidth_cm": 3,
        "fR": 0.05,
        "gain_coeff": None,
        "lo_phonon_cm": 345,
        "to_phonon_cm": 307,
        "references": "Artus et al., Phys. Rev. B 50, 11552 (1994); Mobini et al., J. Phys. Photonics (2026)",
    },
    "LiTaO3": {
        "name": "LiTaO3",
        "crystal": "Rhombohedral",
        "bandgap_eV": 4.0,
        "n2": 3.0e-20,
        "raman_shift_cm": 255,
        "raman_linewidth_cm": 12,
        "fR": 0.0,
        "gain_coeff": None,
        "lo_phonon_cm": None,
        "to_phonon_cm": None,
        "references": "Raptis, Phys. Rev. B 38, 10007 (1988); Margueron et al., J. Appl. Phys. 111, 104105 (2012)",
    },
    "KTP": {
        "name": "KTP",
        "crystal": "Orthorhombic (Pna2_1)",
        "bandgap_eV": 3.5,
        "n2": 5.0e-20,
        "raman_shift_cm": 270,
        "raman_linewidth_cm": 20,
        "fR": 0.0,
        "gain_coeff": None,
        "lo_phonon_cm": None,
        "to_phonon_cm": None,
        "references": "Neufeld et al., Crystals 13, 1423 (2023)",
    },
    "AlGaAs": {
        "name": "AlGaAs",
        "crystal": "Zincblende",
        "bandgap_eV": 1.65,
        "n2": 2.0e-17,
        "raman_shift_cm": 284,
        "raman_linewidth_cm": 5,
        "fR": None,
        "gain_coeff": None,
        "lo_phonon_cm": 284,
        "to_phonon_cm": 262,
        "references": "Feng et al., Phys. Rev. B 47, 13466 (1993); Villeneuve et al., Appl. Phys. Lett. 62, 2465 (1993)",
    },
    "Al2O3": {
        "name": "Al2O3",
        "crystal": "Corundum",
        "bandgap_eV": 8.8,
        "n2": 3.1e-20,
        "raman_shift_cm": 418,
        "raman_linewidth_cm": 5,
        "fR": 0.0,
        "gain_coeff": None,
        "lo_phonon_cm": None,
        "to_phonon_cm": None,
        "references": "Watson et al., J. Chem. Phys. 74, 2023 (1981); Major et al., Opt. Lett. 29, 602 (2004)",
    },
    "YLF": {
        "name": "YLF",
        "crystal": "Scheelite (tetragonal)",
        "bandgap_eV": 10.0,
        "n2": 1.7e-20,
        "raman_shift_cm": 262,
        "raman_linewidth_cm": 5,
        "fR": 0.0,
        "gain_coeff": None,
        "lo_phonon_cm": None,
        "to_phonon_cm": None,
        "references": "Miller et al., J. Chem. Phys. 52, 4172 (1970); Salaun et al., J. Phys. Condens. Matter 9, 6941 (1997)",
    },
    "ZnO": {
        "name": "ZnO",
        "crystal": "Wurtzite",
        "bandgap_eV": 3.37,
        "n2": -9.0e-19,
        "raman_shift_cm": 439,
        "raman_linewidth_cm": 7,
        "fR": None,
        "gain_coeff": None,
        "lo_phonon_cm": 574,
        "to_phonon_cm": 380,
        "references": "Cusco et al., Phys. Rev. B 75, 165202 (2007); Zhang et al., JOSA B 14, 1951 (1997)",
    },
    "CdTe": {
        "name": "CdTe",
        "crystal": "Zincblende",
        "bandgap_eV": 1.44,
        "n2": -3.0e-17,
        "raman_shift_cm": 166,
        "raman_linewidth_cm": 15,
        "fR": None,
        "gain_coeff": None,
        "lo_phonon_cm": 166,
        "to_phonon_cm": 141,
        "references": "Tivanov et al., J. Mater. Sci. (2025); Sayadov et al., JOSA B 9, 405 (1992)",
    },
    "Ga2O3": {
        "name": "Ga2O3",
        "crystal": "Monoclinic (beta)",
        "bandgap_eV": 4.85,
        "n2": 4.0e-19,
        "raman_shift_cm": 767,
        "raman_linewidth_cm": 5,
        "fR": None,
        "gain_coeff": None,
        "lo_phonon_cm": None,
        "to_phonon_cm": None,
        "references": "Kranert et al., Sci. Rep. 6, 35964 (2016); Sun et al., Appl. Phys. Lett. (2022)",
    },
    "LBO": {
        "name": "LBO",
        "crystal": "Orthorhombic",
        "bandgap_eV": 7.8,
        "n2": 1.0e-20,
        "raman_shift_cm": 938,
        "raman_linewidth_cm": 15,
        "fR": 0.0,
        "gain_coeff": None,
        "lo_phonon_cm": None,
        "to_phonon_cm": None,
        "references": "Chen et al., JOSA B 6, 616 (1989)",
    },
    "AgGaS2": {
        "name": "AgGaS2",
        "crystal": "Chalcopyrite (tetragonal)",
        "bandgap_eV": 2.73,
        "n2": 3.0e-18,
        "raman_shift_cm": 295,
        "raman_linewidth_cm": 5,
        "fR": 0.0,
        "gain_coeff": None,
        "lo_phonon_cm": None,
        "to_phonon_cm": None,
        "references": "Lockwood, Montgomery, J. Phys. C 8 (1975); Qiao et al., Opt. Mater. 119, 111300 (2021)",
    },
    "AgGaSe2": {
        "name": "AgGaSe2",
        "crystal": "Chalcopyrite (tetragonal)",
        "bandgap_eV": 1.82,
        "n2": 5.0e-18,
        "raman_shift_cm": 180,
        "raman_linewidth_cm": 5,
        "fR": 0.0,
        "gain_coeff": None,
        "lo_phonon_cm": None,
        "to_phonon_cm": None,
        "references": "Miller et al., Phys. Status Solidi B 78, 569 (1976)",
    },
    "InGaAs": {
        "name": "InGaAs",
        "crystal": "Zincblende",
        "bandgap_eV": 0.75,
        "n2": 5.0e-17,
        "raman_shift_cm": 269,
        "raman_linewidth_cm": 5,
        "fR": None,
        "gain_coeff": None,
        "lo_phonon_cm": 269,
        "to_phonon_cm": 255,
        "references": "Estrera et al., J. Appl. Phys. 72, 3692 (1992); Zhang et al., Appl. Phys. Lett. 123, 011103 (2023)",
    },
    "GeO2": {
        "name": "GeO2",
        "crystal": "Amorphous",
        "bandgap_eV": 5.5,
        "n2": 6.0e-20,
        "raman_shift_cm": 420,
        "raman_linewidth_cm": 135,
        "fR": 0.20,
        "gain_coeff": None,
        "lo_phonon_cm": None,
        "to_phonon_cm": None,
        "references": "Davey et al., IEE Proc. J 136, 301 (1989); Bromage et al., IEEE Photon. Technol. Lett. 14, 24 (2002)",
    },
    "GeAsSe": {
        "name": "GeAsSe",
        "crystal": "Amorphous",
        "bandgap_eV": 1.6,
        "n2": 6.0e-18,
        "raman_shift_cm": 250,
        "raman_linewidth_cm": 50,
        "fR": 0.50,
        "gain_coeff": None,
        "lo_phonon_cm": None,
        "to_phonon_cm": None,
        "references": "Richardson, Chalcogenide glass fibers for nonlinear optics; ~2× As2Se3 (Slusher et al., JOSA B 21, 1146)",
    },
}

# Thorlabs optical substrates (refractive index / Sellmeier; no Raman response)
# https://www.thorlabs.com/optical-substrates
THORLABS_SUBSTRATE_MATERIALS = {
    "N-BK7": {
        "name": "N-BK7",
        "crystal": "Borosilicate crown glass",
        "bandgap_eV": None,
        "n2": None,
        "raman_shift_cm": None,
        "raman_linewidth_cm": None,
        "fR": 0.0,
        "gain_coeff": None,
        "lo_phonon_cm": None,
        "to_phonon_cm": None,
        "references": "Thorlabs Optical Substrates; SCHOTT N-BK7 (refractiveindex.info)",
    },
    "N-SF11": {
        "name": "N-SF11",
        "crystal": "Dense flint glass",
        "bandgap_eV": None,
        "n2": None,
        "raman_shift_cm": None,
        "raman_linewidth_cm": None,
        "fR": 0.0,
        "gain_coeff": None,
        "lo_phonon_cm": None,
        "to_phonon_cm": None,
        "references": "Thorlabs Optical Substrates; SCHOTT N-SF11 (refractiveindex.info)",
    },
    "N-F2": {
        "name": "N-F2",
        "crystal": "Flint glass",
        "bandgap_eV": None,
        "n2": None,
        "raman_shift_cm": None,
        "raman_linewidth_cm": None,
        "fR": 0.0,
        "gain_coeff": None,
        "lo_phonon_cm": None,
        "to_phonon_cm": None,
        "references": "Thorlabs Optical Substrates; SCHOTT N-F2 (refractiveindex.info)",
    },
    "F2": {
        "name": "F2",
        "crystal": "Flint glass",
        "bandgap_eV": None,
        "n2": None,
        "raman_shift_cm": None,
        "raman_linewidth_cm": None,
        "fR": 0.0,
        "gain_coeff": None,
        "lo_phonon_cm": None,
        "to_phonon_cm": None,
        "references": "Thorlabs Optical Substrates; SCHOTT F2 (refractiveindex.info)",
    },
    "CaF2": {
        "name": "CaF2",
        "crystal": "Cubic (fluorite)",
        "bandgap_eV": 12.1,
        "n2": None,
        "raman_shift_cm": None,
        "raman_linewidth_cm": None,
        "fR": 0.0,
        "gain_coeff": None,
        "lo_phonon_cm": None,
        "to_phonon_cm": None,
        "references": "Thorlabs Optical Substrates; Daimon & Masumura, Appl. Opt. 41, 5275 (2002)",
    },
    "BaF2": {
        "name": "BaF2",
        "crystal": "Cubic",
        "bandgap_eV": 10.2,
        "n2": None,
        "raman_shift_cm": None,
        "raman_linewidth_cm": None,
        "fR": 0.0,
        "gain_coeff": None,
        "lo_phonon_cm": None,
        "to_phonon_cm": None,
        "references": "Thorlabs Optical Substrates; Malitson, JOSA 54, 628 (1964)",
    },
    "MgF2": {
        "name": "MgF2",
        "crystal": "Tetragonal (o-ray, c-axis oriented)",
        "bandgap_eV": 12.0,
        "n2": None,
        "raman_shift_cm": None,
        "raman_linewidth_cm": None,
        "fR": 0.0,
        "gain_coeff": None,
        "lo_phonon_cm": None,
        "to_phonon_cm": None,
        "references": "Thorlabs Optical Substrates; Dodge, Appl. Opt. 23, 1980 (1984)",
    },
    "ZnSe": {
        "name": "ZnSe",
        "crystal": "Zincblende",
        "bandgap_eV": 2.70,
        "n2": None,
        "raman_shift_cm": None,
        "raman_linewidth_cm": None,
        "fR": 0.0,
        "gain_coeff": None,
        "lo_phonon_cm": None,
        "to_phonon_cm": None,
        "references": "Thorlabs Optical Substrates; Marple, J. Appl. Phys. 35, 539 (1964)",
    },
    "YVO4": {
        "name": "YVO4",
        "crystal": "Tetragonal (o-ray)",
        "bandgap_eV": None,
        "n2": None,
        "raman_shift_cm": None,
        "raman_linewidth_cm": None,
        "fR": 0.0,
        "gain_coeff": None,
        "lo_phonon_cm": None,
        "to_phonon_cm": None,
        "references": "Thorlabs Optical Substrates; Birnbaum & DeShazer, NASA CR (1976)",
    },
    "KBr": {
        "name": "KBr",
        "crystal": "Cubic",
        "bandgap_eV": 7.5,
        "n2": None,
        "raman_shift_cm": None,
        "raman_linewidth_cm": None,
        "fR": 0.0,
        "gain_coeff": None,
        "lo_phonon_cm": None,
        "to_phonon_cm": None,
        "references": "Thorlabs Optical Substrates; Li, J. Phys. Chem. Ref. Data 5, 329 (1976)",
    },
    "Zerodur": {
        "name": "Zerodur",
        "crystal": "Glass ceramic",
        "bandgap_eV": None,
        "n2": None,
        "raman_shift_cm": None,
        "raman_linewidth_cm": None,
        "fR": 0.0,
        "gain_coeff": None,
        "lo_phonon_cm": None,
        "to_phonon_cm": None,
        "references": "Thorlabs Optical Substrates; SCHOTT ZERODUR (refractiveindex.info)",
    },
    "PMMA": {
        "name": "PMMA",
        "crystal": "Amorphous (acrylic)",
        "bandgap_eV": None,
        "n2": None,
        "raman_shift_cm": None,
        "raman_linewidth_cm": None,
        "fR": 0.0,
        "gain_coeff": None,
        "lo_phonon_cm": None,
        "to_phonon_cm": None,
        "references": "Thorlabs Optical Substrates; Szczurowski (refractiveindex.info)",
    },
}


# ─── RamanSpec ────────────────────────────────────────────────────────────────


@dataclass(config={"arbitrary_types_allowed": True})
class RamanSpec:
    """Material Raman scattering properties.

    Stores Raman parameters and auto-derives derived quantities.

    Attributes
    ----------
    name : Material name.
    crystal : Crystal structure (e.g. "Wurtzite", "Diamond cubic").
    bandgap_eV : Bandgap energy in eV.
    n2 : Nonlinear refractive index n₂ (m²/W).
    raman_shift_cm : Raman shift in cm⁻¹.
    raman_linewidth_cm : Raman linewidth (FWHM) in cm⁻¹.
    fR : Raman response fraction (0 to 1).
    gain_coeff : Raman gain coefficient (m/GW).
    tau1 : Oscillation period (s). Auto-derived if None.
    tau2 : Damping time (s). Auto-derived if None.
    alpha : Silica model parameter α (default 0.52).
    lo_phonon_cm : LO phonon wavenumber (cm⁻¹).
    to_phonon_cm : TO phonon wavenumber (cm⁻¹).
    references : Source citation.
    phonon_modes : List of PhononMode for multi-mode Raman materials.
    """

    name: str
    raman_shift_cm: float | None = None
    raman_linewidth_cm: float | None = None
    crystal: str | None = None
    bandgap_eV: Energy | None = None
    n2: float | None = None
    fR: float | None = None
    gain_coeff: float | None = None
    tau1: Time | None = None
    tau2: Time | None = None
    alpha: float = 0.52
    lo_phonon_cm: float | None = None
    to_phonon_cm: float | None = None
    references: str | None = None
    phonon_modes: list | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_types(cls, values) -> dict:
        # Handle ArgsKwargs from pydantic.dataclasses
        if hasattr(values, "kwargs"):
            values = values.kwargs
        if isinstance(values, dict):
            for key, target_type, unit in [
                ("bandgap_eV", Energy, "eV"),
                ("tau1", Time, "s"),
                ("tau2", Time, "s"),
            ]:
                v = values.get(key)
                if v is not None and not isinstance(v, target_type):
                    values[key] = target_type(v, unit)
        return values

    @model_validator(mode="after")
    def _validate(self) -> "RamanSpec":
        """Validate that essential Raman parameters are present.

        Note: raman_shift_cm and raman_linewidth_cm are optional in the
        constructor. They should be provided when creating from raw params,
        but can be None when using from_database() with fallback.
        """
        return self

    @cached_property
    def raman_shift_Hz(self) -> float:
        """Raman shift in Hz."""
        if self.raman_shift_cm is None:
            return 0.0
        return C_MS * self.raman_shift_cm * 100.0

    @cached_property
    def raman_shift_THz(self) -> float:
        """Raman shift in THz."""
        return self.raman_shift_Hz / 1e12

    @cached_property
    def raman_shift_omega(self) -> float:
        """Raman shift in rad/s."""
        return 2 * PI * self.raman_shift_Hz

    @cached_property
    def linewidth_Hz(self) -> float:
        """Raman linewidth (FWHM) in Hz."""
        if self.raman_linewidth_cm is None:
            return 0.0
        return C_MS * self.raman_linewidth_cm * 100.0

    @cached_property
    def linewidth_THz(self) -> float:
        """Raman linewidth (FWHM) in THz."""
        return self.linewidth_Hz / 1e12

    @cached_property
    def quality_factor(self) -> float:
        """Quality factor Q = shift / linewidth."""
        if self.linewidth_Hz == 0 or self.raman_shift_Hz == 0:
            return float("inf") if self.raman_shift_Hz != 0 else 0.0
        return self.raman_shift_Hz / self.linewidth_Hz

    def stokes_wavelength(self, pump_wl: Wavelength) -> Wavelength:
        """Stokes wavelength for a given pump wavelength.

        ν_stokes = ν_pump - ν_Raman
        λ_stokes = c / ν_stokes
        """
        if self.raman_shift_cm is None:
            raise ValueError("Raman shift not available")
        nu_pump = C_MS / pump_wl.as_m
        nu_stokes = nu_pump - self.raman_shift_Hz
        if nu_stokes <= 0:
            raise ValueError("Stokes frequency would be negative")
        return Wavelength(C_MS / nu_stokes, "m")

    def anti_stokes_wavelength(self, pump_wl: Wavelength) -> Wavelength:
        """Anti-Stokes wavelength for a given pump wavelength.

        ν_anti_stokes = ν_pump + ν_Raman
        λ_anti_stokes = c / ν_anti_stokes
        """
        if self.raman_shift_cm is None:
            raise ValueError("Raman shift not available")
        nu_pump = C_MS / pump_wl.as_m
        nu_anti_stokes = nu_pump + self.raman_shift_Hz
        return Wavelength(C_MS / nu_anti_stokes, "m")

    @property
    def phonon_response(self):
        """Multi-mode phonon response, or None if no modes configured.

        Returns
        -------
        PhononResponse or None
        """
        from .phonon import PhononResponse

        if self.phonon_modes:
            return PhononResponse(self.phonon_modes, fR=self.fR)
        return None

    def multi_stokes_wavelengths(self, pump_wl) -> list:
        """Stokes wavelength for each phonon mode.

        Parameters
        ----------
        pump_wl : Wavelength
            Pump wavelength.

        Returns
        -------
        list[Wavelength]
            Stokes wavelengths, one per mode.
        """
        if not self.phonon_modes:
            return []

        from .base import C_MS

        wavelengths = []
        for mode in self.phonon_modes:
            nu_pump = C_MS / pump_wl.as_m
            nu_stokes = nu_pump - C_MS * mode.shift_cm * 100.0
            if nu_stokes > 0:
                wavelengths.append(Wavelength(C_MS / nu_stokes, "m"))
        return wavelengths

    def summary(self) -> str:
        """Text summary of material Raman properties."""
        lines = [
            f"Material: {self.name}",
            f"Crystal: {self.crystal or 'N/A'}",
            f"Bandgap: {self.bandgap_eV or 'N/A'} eV",
            f"n₂: {self.n2 or 'N/A'} m²/W",
            f"Raman shift: {self.raman_shift_cm} cm⁻¹ = {self.raman_shift_THz:.2f} THz",
            f"Linewidth: {self.raman_linewidth_cm} cm⁻¹ = {self.linewidth_THz:.2f} THz",
            f"fR: {self.fR or 'N/A'}",
            f"Gain coeff: {self.gain_coeff or 'N/A'} m/GW",
            f"Q factor: {self.quality_factor:.1f}",
        ]
        if self.lo_phonon_cm:
            lines.append(f"LO phonon: {self.lo_phonon_cm} cm⁻¹")
        if self.to_phonon_cm:
            lines.append(f"TO phonon: {self.to_phonon_cm} cm⁻¹")
        if self.phonon_modes:
            lines.append(f"Multi-mode: {len(self.phonon_modes)} phonon modes")
        return "\n".join(lines)

    def nk(self, wavelength_um: float) -> complex:
        """Interpolated complex refractive index at wavelength (μm).

        Returns n + i·k from the nk_data table, or falls back to Sellmeier
        interpolation if tabulated data is absent.

        Parameters
        ----------
        wavelength_um : Wavelength in μm

        Returns
        -------
        complex : n + i·k

        Raises
        ------
        ValueError : If no data available or wavelength outside valid range
        """
        from photonics_helper.raman import RamanDatabase  # type: ignore[import-not-found, self-import]

        # Query tabulated data first
        db = RamanDatabase()
        wl, n_tab, k_tab = db.get_nk_data(self.name)

        if len(wl) > 0:
            # Interpolate from tabulated data
            from scipy.interpolate import interp1d

            n_func = interp1d(wl, n_tab, kind="cubic", fill_value="extrapolate")
            k_func = interp1d(wl, k_tab, kind="cubic", fill_value="extrapolate")

            # Validate range
            if wavelength_um < wl[0] or wavelength_um > wl[-1]:
                raise ValueError(
                    f"Wavelength {wavelength_um} μm outside valid range [{wl[0]:.3f}, {wl[-1]:.3f}] μm"
                )

            n_val = float(n_func(wavelength_um))
            k_val = float(k_func(wavelength_um))
            return complex(n_val, k_val)

        # Fallback to Sellmeier
        sellmeier_data = db.get_sellmeier(self.name)
        if sellmeier_data:
            return self.nk_from_sellmeier(wavelength_um, sellmeier_data)

        # No data available
        raise ValueError(f"No refractive index data available for {self.name}")

    def nk_from_sellmeier(
        self, wavelength_um: float, sellmeier_data: dict | None = None
    ) -> complex:
        """Compute n + ik from stored Sellmeier coefficients.

        Parameters
        ----------
        wavelength_um : Wavelength in μm
        sellmeier_data : Dict with keys: form, a0, coefficients, wavelengths, valid_from_um, valid_to_um
                       If None, fetches from database

        Returns
        -------
        complex : n + i·0 (k=0 for transparent region)

        Raises
        ------
        ValueError : If wavelength outside valid range or no data available
        """
        from photonics_helper.raman import RamanDatabase  # type: ignore[import-not-found, self-import]
        import numpy as np

        if sellmeier_data is None:
            db = RamanDatabase()
            sellmeier_data = db.get_sellmeier(self.name)

        if sellmeier_data is None:
            raise ValueError(f"No Sellmeier data available for {self.name}")

        # Validate wavelength range
        if (
            wavelength_um < sellmeier_data["valid_from_um"]
            or wavelength_um > sellmeier_data["valid_to_um"]
        ):
            raise ValueError(
                f"Wavelength {wavelength_um} μm outside valid range "
                f"[{sellmeier_data['valid_from_um']:.3f}, {sellmeier_data['valid_to_um']:.3f}] μm"
            )

        # Compute n from Sellmeier equation
        form = sellmeier_data["form"]
        a0 = sellmeier_data["a0"]
        coefficients = sellmeier_data["coefficients"]
        wavelengths = sellmeier_data["wavelengths"]

        if form == "standard":
            # n² = A₀ + Σ Aᵢλ²/(λ² - Bᵢ)
            n_squared = a0
            for A, B in zip(coefficients, wavelengths):
                n_squared += A * wavelength_um**2 / (wavelength_um**2 - B)
            n_val = np.sqrt(n_squared)

        elif form == "alt":
            # n² = A₀ + Σ Aᵢ/(λ² - Bᵢ²)
            n_squared = a0
            for A, B in zip(coefficients, wavelengths):
                n_squared += A / (wavelength_um**2 - B**2)
            n_val = np.sqrt(n_squared)
        else:
            raise ValueError(f"Unknown Sellmeier form: {form}")

        # k = 0 for transparent region (no absorption data in Sellmeier)
        return complex(float(n_val), 0.0)

    def plot_spectrum(
        self,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        shift_range_cm: float = 200,
        n_points: int = 1000,
        figsize: tuple[float, float] | None = None,
    ):
        """Plot Raman intensity vs Raman shift.

        Parameters
        ----------
        backend : "matplotlib" or "plotly"
        shift_range_cm : Range around 0 to plot (cm⁻¹)
        n_points : Number of points
        figsize : Figure size for matplotlib
        """
        if self.raman_shift_cm is None or self.raman_linewidth_cm is None:
            raise ValueError("Raman shift and linewidth required")

        shift = np.linspace(-shift_range_cm, shift_range_cm, n_points)
        center = self.raman_shift_cm
        width = self.raman_linewidth_cm / 2

        # Lorentzian lineshape
        intensity = (width / np.pi) / ((shift - center) ** 2 + width**2)
        intensity /= np.max(intensity)  # Normalize

        if backend == "plotly":
            if not HAS_PLOTLY:
                raise ImportError("plotly required for plotly backend")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=shift, y=intensity, mode="lines", name="Raman"))
            fig.update_layout(
                title=f"Raman Spectrum: {self.name}",
                xaxis_title="Raman shift (cm⁻¹)",
                yaxis_title="Intensity (arb.)",
            )
            return fig
        else:
            fig, ax = plt.subplots(figsize=figsize or (8, 5))
            ax.plot(shift, intensity, linewidth=2, label="Raman peak")
            ax.axvline(
                x=center,
                color="r",
                linestyle="--",
                alpha=0.5,
                label=f"Peak: {center} cm⁻¹",
            )
            ax.axvspan(
                center - width,
                center + width,
                alpha=0.2,
                color="orange",
                label=f"FWHM: {2*width:.1f} cm⁻¹",
            )
            ax.set_xlabel("Raman shift (cm⁻¹)", fontsize=12)
            ax.set_ylabel("Intensity (arb.)", fontsize=12)
            ax.set_title(f"Raman Spectrum: {self.name}", fontsize=14, fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            return fig

    def plot_phonons(
        self,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        figsize: tuple[float, float] | None = None,
    ):
        """Plot Raman-active phonon modes.

        Displays LO, TO, and other Raman-active modes if available.
        """
        if backend == "plotly":
            if not HAS_PLOTLY:
                raise ImportError("plotly required for plotly backend")
            fig = go.Figure()
            if self.lo_phonon_cm:
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=self.lo_phonon_cm,
                        title=dict(text="LO Phonon"),
                        gauge=dict(axis=dict(range=[0, self.lo_phonon_cm * 1.2])),
                    )
                )
            if self.to_phonon_cm:
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=self.to_phonon_cm,
                        title=dict(text="TO Phonon"),
                        gauge=dict(axis=dict(range=[0, self.to_phonon_cm * 1.2])),
                    )
                )
            fig.update_layout(title=f"Phonon Modes: {self.name}")
            return fig
        else:
            fig, axes = plt.subplots(1, 2, figsize=figsize or (10, 4))
            if self.lo_phonon_cm:
                axes[0].barh(["LO"], [self.lo_phonon_cm], color="blue", alpha=0.7)
                axes[0].set_xlabel("Wavenumber (cm⁻¹)")
                axes[0].set_title("LO Phonon")
            else:
                axes[0].text(
                    0.5,
                    0.5,
                    "N/A",
                    ha="center",
                    va="center",
                    transform=axes[0].transAxes,
                )
                axes[0].set_title("LO Phonon")

            if self.to_phonon_cm:
                axes[1].barh(["TO"], [self.to_phonon_cm], color="red", alpha=0.7)
                axes[1].set_xlabel("Wavenumber (cm⁻¹)")
                axes[1].set_title("TO Phonon")
            else:
                axes[1].text(
                    0.5,
                    0.5,
                    "N/A",
                    ha="center",
                    va="center",
                    transform=axes[1].transAxes,
                )
                axes[1].set_title("TO Phonon")

            fig.suptitle(
                f"Raman-Active Phonons: {self.name}", fontsize=14, fontweight="bold"
            )
            plt.tight_layout()
            return fig

    @classmethod
    def from_database(
        cls, name: str, fallback: dict | None = None, db_path: Path | None = None
    ) -> "RamanSpec":
        """Create RamanSpec from SQLite database or fallback.

        Queries materials.db first, then falls back to the hardcoded
        RAMAN_MATERIALS dict, and finally to the provided fallback dict.

        Parameters
        ----------
        name : Material name
        fallback : Optional dict of fallback values if not in DB
        db_path : Optional explicit path to SQLite database
        """
        # Try SQLite database first
        try:
            db = RamanDatabase(db_path=db_path)
            row = db.get_material(name)
            if row:
                return cls(**row)  # type: ignore[arg-type]
        except Exception:
            pass

        # Fall back to hardcoded materials
        if name in RAMAN_MATERIALS:
            return cls(**RAMAN_MATERIALS[name])  # type: ignore[arg-type]

        # Use fallback if provided
        if fallback:
            return cls(name=name, **fallback)

        raise ValueError(f"Material '{name}' not found in database or fallback")


# ─── RamanDatabase ────────────────────────────────────────────────────────────


@dataclass(config={"arbitrary_types_allowed": True})
class RamanDatabase:
    """SQLite database for Raman material data.

    Schema:
    CREATE TABLE raman_specs (
        name              TEXT PRIMARY KEY,
        crystal           TEXT,
        bandgap_eV        REAL,
        n2                REAL,
        raman_shift_cm    REAL,
        raman_linewidth_cm REAL,
        fR                REAL,
        gain_coeff        REAL,
        tau1              REAL,
        tau2              REAL,
        lo_phonon_cm      REAL,
        to_phonon_cm      REAL,
        references        TEXT
    );

    CREATE TABLE nk_data (
        material       TEXT REFERENCES raman_specs(name),
        wavelength_um  REAL,
        n              REAL,
        k              REAL
    );
    """

    db_path: Path | None = None

    def __post_init__(self):
        if self.db_path is None:
            # Try bundled DB first, then next to module
            bundled = Path(__file__).parent / "materials.db"
            if bundled.exists():
                object.__setattr__(self, "db_path", bundled)
            else:
                # Create new DB in user's home directory
                home_db = Path.home() / ".photonics_helper" / "materials.db"
                home_db.parent.mkdir(parents=True, exist_ok=True)
                object.__setattr__(self, "db_path", home_db)
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS raman_specs (
                name              TEXT PRIMARY KEY,
                crystal           TEXT,
                bandgap_eV        REAL,
                n2                REAL,
                raman_shift_cm    REAL,
                raman_linewidth_cm REAL,
                fR                REAL,
                gain_coeff        REAL,
                tau1              REAL,
                tau2              REAL,
                lo_phonon_cm      REAL,
                to_phonon_cm      REAL,
                "references"        TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nk_data (
                material       TEXT REFERENCES raman_specs(name),
                wavelength_um  REAL,
                n              REAL,
                k              REAL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sellmeier (
                material       TEXT PRIMARY KEY REFERENCES raman_specs(name),
                form           TEXT CHECK(form IN ('standard', 'alt')),
                a0             REAL,
                coefficients   TEXT,
                wavelengths    TEXT,
                valid_from_um  REAL,
                valid_to_um    REAL,
                source         TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS phonon_modes (
                material           TEXT REFERENCES raman_specs(name),
                shift_cm           REAL,
                linewidth_cm       REAL,
                symmetry           TEXT,
                relative_strength  REAL DEFAULT 1.0,
                lo_phonon_cm       REAL,
                to_phonon_cm       REAL,
                note               TEXT,
                PRIMARY KEY (material, shift_cm, symmetry)
            )
        """)

        conn.commit()
        conn.close()

    def add_material(self, spec: dict) -> None:
        """INSERT or REPLACE a material entry.

        Parameters
        ----------
        spec : dict with material properties
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO raman_specs
            (name, crystal, bandgap_eV, n2, raman_shift_cm, raman_linewidth_cm,
             fR, gain_coeff, tau1, tau2, lo_phonon_cm, to_phonon_cm, "references")
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                spec.get("name"),
                spec.get("crystal"),
                spec.get("bandgap_eV"),
                spec.get("n2"),
                spec.get("raman_shift_cm"),
                spec.get("raman_linewidth_cm"),
                spec.get("fR"),
                spec.get("gain_coeff"),
                spec.get("tau1"),
                spec.get("tau2"),
                spec.get("lo_phonon_cm"),
                spec.get("to_phonon_cm"),
                spec.get("references"),
            ),
        )

        conn.commit()
        conn.close()

    def get_material(self, name: str) -> dict | None:
        """SELECT one material by name.

        Parameters
        ----------
        name : Material name

        Returns
        -------
        dict with material properties, or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM raman_specs WHERE name = ?", (name,))
        row = cursor.fetchone()

        conn.close()

        if row:
            return dict(row)
        return None

    def list_materials(self) -> list[str]:
        """SELECT DISTINCT name FROM raman_specs.

        Returns
        -------
        list of material names
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM raman_specs ORDER BY name")
        rows = cursor.fetchall()

        conn.close()
        return [row[0] for row in rows]

    def add_phonon_mode(self, material: str, mode: "PhononMode") -> None:
        """Insert a phonon mode for a material.

        Parameters
        ----------
        material : str
            Material name.
        mode : PhononMode
            Phonon mode to add.
        """
        from .phonon import PhononMode

        if not isinstance(mode, PhononMode):
            raise TypeError("mode must be a PhononMode instance")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO phonon_modes
            (material, shift_cm, linewidth_cm, symmetry, relative_strength,
             lo_phonon_cm, to_phonon_cm, note)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                material,
                mode.shift_cm,
                mode.linewidth_cm,
                mode.symmetry,
                mode.relative_strength,
                mode.lo_phonon_cm,
                mode.to_phonon_cm,
                mode.note,
            ),
        )

        conn.commit()
        conn.close()

    def get_phonon_modes(self, material: str) -> list["PhononMode"]:
        """Get all phonon modes for a material.

        Parameters
        ----------
        material : str
            Material name.

        Returns
        -------
        list[PhononMode]
            Phonon modes for the material.
        """
        from .phonon import PhononMode

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM phonon_modes WHERE material = ? ORDER BY shift_cm",
            (material,),
        )
        rows = cursor.fetchall()

        conn.close()

        modes = []
        for row in rows:
            modes.append(
                PhononMode(
                    shift_cm=row["shift_cm"],
                    linewidth_cm=row["linewidth_cm"],
                    symmetry=row["symmetry"],
                    relative_strength=row["relative_strength"],
                    lo_phonon_cm=row["lo_phonon_cm"],
                    to_phonon_cm=row["to_phonon_cm"],
                    note=row["note"],
                )
            )
        return modes

    def list_phonon_materials(self) -> list[str]:
        """Get materials with phonon mode data.

        Returns
        -------
        list[str]
            Material names that have phonon_modes entries.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT DISTINCT material FROM phonon_modes ORDER BY material")
        rows = cursor.fetchall()

        conn.close()
        return [row[0] for row in rows]

    def seed_phonon_data(self) -> int:
        """Seed PHONON_MATERIALS data into the phonon_modes table.

        Returns
        -------
        int
            Number of modes seeded.
        """
        from .phonon import PHONON_MATERIALS

        count = 0
        for material, modes in PHONON_MATERIALS.items():
            for mode in modes:
                self.add_phonon_mode(material, mode)
                count += 1
        return count

    def update_material(self, name: str, **kwargs) -> None:
        """UPDATE material fields.

        Parameters
        ----------
        name : Material name
        **kwargs : Fields to update
        """
        if not kwargs:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Build UPDATE query dynamically
        columns = ", ".join(f"{k} = ?" for k in kwargs.keys())
        values = list(kwargs.values()) + [name]

        cursor.execute(f"UPDATE raman_specs SET {columns} WHERE name = ?", values)

        conn.commit()
        conn.close()

    def delete_material(self, name: str) -> None:
        """DELETE material by name.

        Parameters
        ----------
        name : Material name
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM raman_specs WHERE name = ?", (name,))
        cursor.execute("DELETE FROM nk_data WHERE material = ?", (name,))

        conn.commit()
        conn.close()

    def add_nk_data(self, material: str, wl_um: float, n: float, k: float) -> None:
        """INSERT nk data point.

        Parameters
        ----------
        material : Material name
        wl_um : Wavelength in μm
        n : Real refractive index
        k : Extinction coefficient
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO nk_data (material, wavelength_um, n, k)
            VALUES (?, ?, ?, ?)
        """,
            (material, wl_um, n, k),
        )

        conn.commit()
        conn.close()

    def get_nk_data(self, material: str) -> tuple[NDArray, NDArray, NDArray]:
        """SELECT nk data for a material.

        Parameters
        ----------
        material : Material name

        Returns
        -------
        (wavelength_um, n, k) as numpy arrays
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT wavelength_um, n, k FROM nk_data WHERE material = ? ORDER BY wavelength_um",
            (material,),
        )
        rows = cursor.fetchall()

        conn.close()

        if not rows:
            return np.array([]), np.array([]), np.array([])

        wl = np.array([r[0] for r in rows])
        n = np.array([r[1] for r in rows])
        k = np.array([r[2] for r in rows])

        return wl, n, k

    def add_sellmeier(
        self,
        material: str,
        form: str,
        a0: float,
        coefficients: list[float],
        wavelengths: list[float],
        valid_from_um: float,
        valid_to_um: float,
        source: str,
    ) -> None:
        """INSERT Sellmeier coefficients.

        Parameters
        ----------
        material : Material name
        form : 'standard' or 'alt'
        a0 : Constant term A₀
        coefficients : List of Aᵢ coefficients
        wavelengths : List of Bᵢ wavelengths
        valid_from_um : Valid range start (μm)
        valid_to_um : Valid range end (μm)
        source : Citation
        """
        import json

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO sellmeier (material, form, a0, coefficients, wavelengths, valid_from_um, valid_to_um, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                material,
                form,
                a0,
                json.dumps(coefficients),
                json.dumps(wavelengths),
                valid_from_um,
                valid_to_um,
                source,
            ),
        )

        conn.commit()
        conn.close()

    def get_sellmeier(self, material: str) -> dict | None:
        """SELECT Sellmeier coefficients for a material.

        Parameters
        ----------
        material : Material name

        Returns
        -------
        dict with keys: material, form, a0, coefficients, wavelengths, valid_from_um, valid_to_um, source
        Or None if not found
        """
        import json

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM sellmeier WHERE material = ?", (material,))
        row = cursor.fetchone()

        conn.close()

        if not row:
            return None

        result = dict(row)
        result["coefficients"] = json.loads(result["coefficients"])
        result["wavelengths"] = json.loads(result["wavelengths"])

        return result

    def search_materials(self, query: str) -> list[dict[str, float]]:
        """LIKE search across name, crystal, references.

        Parameters
        ----------
        query : Search string

        Returns
        -------
        list of matching materials
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        search_pattern = f"%{query}%"
        cursor.execute(
            """
            SELECT * FROM raman_specs
            WHERE name LIKE ? OR crystal LIKE ? OR "references" LIKE ?
            ORDER BY name
        """,
            (search_pattern, search_pattern, search_pattern),
        )
        rows = cursor.fetchall()

        conn.close()
        return [dict(row) for row in rows]

    def export_to_dict(self) -> dict[str, dict]:
        """Export all materials as a Python dict.

        Returns
        -------
        dict mapping material names to property dicts
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM raman_specs ORDER BY name")
        rows = cursor.fetchall()

        conn.close()
        return {row["name"]: dict(row) for row in rows}

    def import_from_dict(self, data: dict[str, dict]) -> None:
        """Bulk INSERT from a Python dict.

        Parameters
        ----------
        data : dict mapping material names to property dicts
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for name, spec in data.items():
            cursor.execute(
                """
                INSERT OR REPLACE INTO raman_specs
                (name, crystal, bandgap_eV, n2, raman_shift_cm, raman_linewidth_cm,
                 fR, gain_coeff, tau1, tau2, lo_phonon_cm, to_phonon_cm, "references")
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    spec.get("name", name),
                    spec.get("crystal"),
                    spec.get("bandgap_eV"),
                    spec.get("n2"),
                    spec.get("raman_shift_cm"),
                    spec.get("raman_linewidth_cm"),
                    spec.get("fR"),
                    spec.get("gain_coeff"),
                    spec.get("tau1"),
                    spec.get("tau2"),
                    spec.get("lo_phonon_cm"),
                    spec.get("to_phonon_cm"),
                    spec.get("references"),
                ),
            )

        conn.commit()
        conn.close()


# ─── RamanResponse (Layer 2) ─────────────────────────────────────────────────


@dataclass(config={"arbitrary_types_allowed": True})
class RamanResponse:
    """Time-domain Raman response function h_R(t).

    Implements the standard Silica model (Agrawal):
        h_R(t) = (τ1² + τ2²) / (τ1·τ2²) · exp(-t/τ2) · sin(t/τ1)   for t ≥ 0
        h_R(t) = 0                                                    for t < 0

    Combined response: R(t) = (1 - fR)·δ(t) + fR·h_R(t)
    where δ(t) is approximated as a narrow Gaussian.

    Attributes
    ----------
    spec : RamanSpec
        Material Raman properties.
    fR : float
        Raman response fraction. Overrides spec.fR if set.
    tau1 : float
        Oscillation period (s). Auto-derived from Raman shift if None.
    tau2 : float
        Damping time (s). Auto-derived from linewidth if None.
    grid : TemporalGrid
        Time grid for computations.
    """

    spec: RamanSpec
    fR: float | None = None
    tau1: float | None = None
    tau2: float | None = None
    grid: TemporalGrid | None = None

    @model_validator(mode="after")
    def _derive_tau(self) -> "RamanResponse":
        """Auto-derive τ1, τ2 from Raman shift + linewidth if not provided."""
        if self.fR is None:
            self.fR = self.spec.fR or 0.0

        if self.tau1 is None and self.spec.raman_shift_Hz > 0:
            self.tau1 = 1.0 / (2 * np.pi * self.spec.raman_shift_Hz)

        if self.tau2 is None and self.spec.linewidth_Hz > 0:
            self.tau2 = 1.0 / (np.pi * self.spec.linewidth_Hz)
            warnings.warn(
                f"Auto-derived τ2 = {self.tau2*1e15:.1f} fs from linewidth. "
                f"This approximation (τ2 = 1/(π·linewidth)) assumes weak damping "
                f"and may be inaccurate for materials like Silica where τ2/τ1 is small. "
                f"Consider providing τ1 and τ2 explicitly for accurate Raman responses.",
                stacklevel=2,
            )

        if self.grid is None:
            # Default grid: cover ~20 τ2 for damped oscillation to decay
            from .base import Time
            tau2_val = self.tau2 or 1e-12
            tmax = max(10e-12, 20 * tau2_val)
            self.grid = TemporalGrid(
                N=2**14, Tmax=Time(tmax, unit='s')
            )

        return self

    @property
    def _delta_width(self) -> float:
        """Width parameter ε for the narrow Gaussian approximation of δ(t)."""
        # Use grid.dt * 5 as the delta width — narrow enough to look like a spike
        return self.grid.dt * 5  # type: ignore

    def _h_R(self, t: NDArray) -> NDArray:
        """Raw delayed response h_R(t) (without fR scaling).

        Standard Agrawal exponential-damped form:
        h_R(t) = (τ1² + τ2²) / (τ1·τ2²) · exp(-t/τ2) · sin(t/τ1)   for t ≥ 0
        h_R(t) = 0                                                    for t < 0
        """
        result = np.zeros_like(t, dtype=float)
        mask = t >= 0
        t_pos = t[mask]

        prefactor = (self.tau1**2 + self.tau2**2) / (self.tau1 * self.tau2**2)  # type: ignore
        exponential = np.exp(-t_pos / self.tau2)
        oscillation = np.sin(t_pos / self.tau1)

        result[mask] = prefactor * exponential * oscillation

        integral = np.trapezoid(result[mask], t_pos)
        if integral > 0:
            result[mask] /= integral
        return result

    def instantaneous_response(self, t: NDArray | None = None) -> NDArray:
        """Electronic Kerr response: (1 - fR)·δ(t).

        δ(t) approximated as a narrow Gaussian: exp(-t²/(2ε²)) / (ε·√(2π)).

        Parameters
        ----------
        t : array of time values (s). If None, uses self.grid.t.

        Returns
        -------
        NDArray — instantaneous response values.
        """
        if t is None:
            t = self.grid.t  # type: ignore
        t = np.asarray(t, dtype=float)
        eps = self._delta_width
        delta = np.exp(-(t**2) / (2 * eps**2)) / (eps * np.sqrt(2 * np.pi))
        return (1.0 - self.fR) * delta  # type: ignore

    def delayed_response(self, t: NDArray | None = None) -> NDArray:
        """Lattice oscillation: fR·h_R(t).

        Parameters
        ----------
        t : array of time values (s). If None, uses self.grid.t.

        Returns
        -------
        NDArray — delayed (Raman) response values.
        """
        if t is None:
            t = self.grid.t  # type: ignore
        t = np.asarray(t, dtype=float)
        return self.fR * self._h_R(t)  # type: ignore

    def combined_response(self, t: NDArray | None = None) -> NDArray:
        """Total Raman response R(t) = (1-fR)δ(t) + fR·h_R(t).

        Parameters
        ----------
        t : array of time values (s). If None, uses self.grid.t.

        Returns
        -------
        NDArray — combined response values.
        """
        if t is None:
            t = self.grid.t  # type: ignore
        t = np.asarray(t, dtype=float)
        return self.instantaneous_response(t) + self.delayed_response(t)  # type: ignore

    def plot_components(
        self,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        figsize: tuple[float, float] | None = None,
        t_range_ps: tuple[float, float] | None = None,
    ):
        """3-panel plot: instantaneous, delayed, combined response.

        Parameters
        ----------
        backend : "matplotlib" or "plotly"
        figsize : Figure size for matplotlib
        t_range_ps : Time range in picoseconds as (t_min, t_max). Overrides grid.
        """
        if backend == "plotly":
            if not HAS_PLOTLY:
                raise ImportError("plotly required for plotly backend")
            return self._plot_components_plotly(figsize, t_range_ps)
        else:
            return self._plot_components_matplotlib(figsize, t_range_ps)

    def _plot_components_matplotlib(
        self,
        figsize: tuple[float, float] | None,
        t_range_ps: tuple[float, float] | None,
    ):
        """Matplotlib 3-panel plot."""
        import matplotlib.pyplot as plt

        if t_range_ps is not None:
            t_min, t_max = t_range_ps
            t = np.linspace(t_min * 1e-12, t_max * 1e-12, 5000)
        else:
            t = self.grid.t  # type: ignore

        inst = self.instantaneous_response(t)
        delayed = self.delayed_response(t)
        combined = self.combined_response(t)

        fig, axes = plt.subplots(3, 1, figsize=figsize or (10, 9), sharex=True)
        fig.suptitle(
            f"Raman Response: {self.spec.name}  "
            f"(fR={self.fR:.2f}, τ1={self.tau1*1e15:.2f} fs, τ2={self.tau2*1e15:.2f} fs)",  # type: ignore
            fontsize=13,
            fontweight="bold",
        )

        # Panel 1: Instantaneous
        ax1 = axes[0]
        ax1.plot(t * 1e12, inst, color="#00d4ff", linewidth=1.5)
        ax1.set_ylabel("Amplitude (arb.)", fontsize=10)
        ax1.set_title("Instantaneous (Electronic Kerr) Response", fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(0, color="k", linewidth=0.5)

        # Panel 2: Delayed
        ax2 = axes[1]
        ax2.plot(t * 1e12, delayed, color="#a78bfa", linewidth=1.5)
        ax2.set_ylabel("Amplitude (arb.)", fontsize=10)
        ax2.set_title("Delayed (Lattice Oscillation) Response fR·h_R(t)", fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0, color="k", linewidth=0.5)
        # Shade the oscillation envelope
        envelope = np.abs(delayed) if np.any(delayed != 0) else delayed
        ax2.fill_between(t * 1e12, 0, envelope, alpha=0.1, color="#a78bfa")

        # Panel 3: Combined
        ax3 = axes[2]
        ax3.plot(
            t * 1e12, combined, color="#34d399", linewidth=1.5, label="Combined R(t)"
        )
        ax3.set_xlabel("Time (ps)", fontsize=10)
        ax3.set_ylabel("Amplitude (arb.)", fontsize=10)
        ax3.set_title("Combined Response R(t) = (1-fR)δ(t) + fR·h_R(t)", fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(0, color="k", linewidth=0.5)
        ax3.legend(loc="upper right")

        plt.tight_layout()
        return fig

    def _plot_components_plotly(
        self,
        figsize: tuple[float, float] | None,
        t_range_ps: tuple[float, float] | None,
    ):
        """Plotly 3-panel plot."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        if t_range_ps is not None:
            t_min, t_max = t_range_ps
            t = np.linspace(t_min * 1e-12, t_max * 1e-12, 5000)
        else:
            t = self.grid.t  # type: ignore

        inst = self.instantaneous_response(t)
        delayed = self.delayed_response(t)
        combined = self.combined_response(t)

        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=(
                "Instantaneous (Electronic Kerr) Response",
                "Delayed (Lattice Oscillation) Response fR·h_R(t)",
                "Combined Response R(t) = (1-fR)δ(t) + fR·h_R(t)",
            ),
            shared_xaxes=True,
            vertical_spacing=0.08,
        )

        t_ps = t * 1e12

        fig.add_trace(
            go.Scatter(
                x=t_ps,
                y=inst,
                mode="lines",
                name="Instant.",
                line=dict(color="#00d4ff", width=1.5),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t_ps,
                y=delayed,
                mode="lines",
                name="Delayed",
                line=dict(color="#a78bfa", width=1.5),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t_ps,
                y=combined,
                mode="lines",
                name="Combined",
                line=dict(color="#34d399", width=1.5),
            ),
            row=3,
            col=1,
        )

        for i in range(1, 4):
            fig.add_hline(
                y=0, line_dash="dot", line_color="gray", opacity=0.5, row=i, col=1
            )

        fig.update_yaxes(title_text="Amplitude (arb.)", row=1, col=1)
        fig.update_yaxes(title_text="Amplitude (arb.)", row=2, col=1)
        fig.update_yaxes(title_text="Amplitude (arb.)", row=3, col=1)
        fig.update_xaxes(title_text="Time (ps)", row=3, col=1)

        fig.update_layout(
            title_text=f"Raman Response: {self.spec.name}  "
            f"(fR={self.fR:.2f}, τ1={self.tau1*1e15:.2f} fs, τ2={self.tau2*1e15:.2f} fs)",  # type: ignore
            height=750,
            showlegend=False,
        )
        return fig


# ─── RamanFrequencyResponse (Layer 3) ─────────────────────────────────────────


@dataclass(config={"arbitrary_types_allowed": True})
class RamanFrequencyResponse:
    """Frequency-domain Raman response H(Ω) = F{h_R(t)}.

    Connects time-domain physics to measured Raman spectra via FFT.

    Attributes
    ----------
    response : RamanResponse
        Layer 2 time-domain response.
    grid : TemporalGrid
        Frequency grid (from the temporal grid's fft).
    """

    response: RamanResponse
    grid: TemporalGrid | None = None

    @model_validator(mode="after")
    def _ensure_grid(self) -> "RamanFrequencyResponse":
        if self.grid is None:
            self.grid = self.response.grid
        return self

    @property
    def H(self) -> NDArray:
        """Complex frequency response H(Ω)."""
        h_R_t = (
            self.response.delayed_response(self.grid.t) / self.response.fR  # type: ignore
            if self.response.fR > 0  # type: ignore
            else np.zeros_like(self.grid.t)  # type: ignore
        )
        return self.grid.fft(h_R_t)  # type: ignore

    @property
    def H_real(self) -> NDArray:
        """Real part Re(H(Ω))."""
        return np.real(self.H)

    @property
    def H_imag(self) -> NDArray:
        """Imaginary part Im(H(Ω))."""
        return np.imag(self.H)

    @property
    def H_magnitude(self) -> NDArray:
        """Magnitude |H(Ω)|."""
        return np.abs(self.H)

    @property
    def H_phase(self) -> NDArray:
        """Phase ∠H(Ω)."""
        return np.angle(self.H)

    @property
    def resonance_frequency_THz(self) -> float:
        """Resonance frequency in THz (peak of |Im(H)|)."""
        # Find peak of |Im(H)| in positive frequencies
        positive_mask = self.grid.w > 0  # type: ignore
        w_pos = self.grid.w[positive_mask]  # type: ignore
        imag_pos = self.H_imag[positive_mask]
        peak_idx = np.argmax(np.abs(imag_pos))
        return w_pos[peak_idx] / (2 * np.pi * 1e12)  # rad/s → THz

    @property
    def resonance_FWHM_THz(self) -> float:
        """FWHM of the resonance in THz."""
        mag = self.H_magnitude
        w = self.grid.w / (2 * np.pi * 1e12)  # type: ignore  # rad/s → THz

        # Find peak
        peak_idx = np.argmax(mag)
        peak_val = mag[peak_idx]
        half_max = peak_val / 2

        # Find left and right crossings
        left_idx = np.where(mag[:peak_idx] < half_max)[0]
        right_idx = np.where(mag[peak_idx:] < half_max)[0]

        if len(left_idx) > 0 and len(right_idx) > 0:
            w_left = w[left_idx[-1]]
            w_right = w[peak_idx + right_idx[0]]
            return w_right - w_left

        # Fallback: use grid resolution
        return float(self.grid.dw / (2 * np.pi * 1e12))  # type: ignore

    @property
    def quality_factor(self) -> float:
        """Quality factor Q = f_res / FWHM."""
        fwhm = self.resonance_FWHM_THz
        if fwhm == 0:
            return float("inf")
        return self.resonance_frequency_THz / fwhm

    def plot_real(
        self,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        figsize: tuple[float, float] | None = None,
    ):
        """Plot Re(H(Ω))."""
        if backend == "plotly":
            if not HAS_PLOTLY:
                raise ImportError("plotly required for plotly backend")
            return self._plot_dispersion_plotly(
                "Real Part Re(H)", self.H_real, "Re(H(Ω))", figsize
            )
        return self._plot_dispersion_matplotlib(
            "Real Part Re(H)", self.H_real, "Re(H(Ω))", figsize
        )

    def plot_imag(
        self,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        figsize: tuple[float, float] | None = None,
    ):
        """Plot Im(H(Ω)) — the Raman gain spectrum."""
        if backend == "plotly":
            if not HAS_PLOTLY:
                raise ImportError("plotly required for plotly backend")
            return self._plot_dispersion_plotly(
                "Imaginary Part Im(H)", self.H_imag, "Im(H(Ω))", figsize
            )
        return self._plot_dispersion_matplotlib(
            "Imaginary Part Im(H)", self.H_imag, "Im(H(Ω))", figsize
        )

    def plot_magnitude(
        self,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        figsize: tuple[float, float] | None = None,
    ):
        """Plot |H(Ω)|."""
        if backend == "plotly":
            if not HAS_PLOTLY:
                raise ImportError("plotly required for plotly backend")
            return self._plot_dispersion_plotly(
                "Magnitude |H(Ω)|", self.H_magnitude, "|H(Ω)|", figsize
            )
        return self._plot_dispersion_matplotlib(
            "Magnitude |H(Ω)|", self.H_magnitude, "|H(Ω)|", figsize
        )

    def plot_phase(
        self,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        figsize: tuple[float, float] | None = None,
    ):
        """Plot ∠H(Ω)."""
        if backend == "plotly":
            if not HAS_PLOTLY:
                raise ImportError("plotly required for plotly backend")
            return self._plot_dispersion_plotly(
                "Phase ∠H(Ω)", self.H_phase, "∠H(Ω) (rad)", figsize
            )
        return self._plot_dispersion_matplotlib(
            "Phase ∠H(Ω)", self.H_phase, "∠H(Ω) (rad)", figsize
        )

    def plot_all(
        self,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        figsize: tuple[float, float] | None = None,
    ):
        """4-panel plot: Re, Im, |H|, phase."""
        if backend == "plotly":
            if not HAS_PLOTLY:
                raise ImportError("plotly required for plotly backend")
            return self._plot_all_plotly(figsize)
        return self._plot_all_matplotlib(figsize)

    def _plot_dispersion_matplotlib(
        self,
        title: str,
        data: NDArray,
        ylabel: str,
        figsize: tuple[float, float] | None,
    ):
        """Single-panel dispersion plot (matplotlib)."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize or (10, 5))
        w_THz = self.grid.w / (2 * np.pi * 1e12)  # type: ignore[union-attr]
        ax.plot(w_THz, data, linewidth=1.5, color="#00d4ff")
        ax.set_xlabel("Angular frequency (THz)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", linewidth=0.5)

        # Annotate resonance
        if "Im" in title or "Magnitude" in title:
            f_res = self.resonance_frequency_THz
            ax.axvline(
                f_res,
                color="r",
                linestyle="--",
                alpha=0.7,
                label=f"Resonance: {f_res:.2f} THz",
            )
            ax.legend()

        plt.tight_layout()
        return fig

    def _plot_dispersion_plotly(
        self,
        title: str,
        data: NDArray,
        ylabel: str,
        figsize: tuple[float, float] | None,
    ):
        """Single-panel dispersion plot (plotly)."""
        import plotly.graph_objects as go

        fig = go.Figure()
        w_THz = self.grid.w / (2 * np.pi * 1e12)  # type: ignore[union-attr]
        fig.add_trace(
            go.Scatter(
                x=w_THz,
                y=data,
                mode="lines",
                name="H(Ω)",
                line=dict(color="#00d4ff", width=1.5),
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="Angular frequency (THz)",
            yaxis_title=ylabel,
            height=500,
        )
        return fig

    def _plot_all_matplotlib(self, figsize: tuple[float, float] | None) -> "plt.Figure":
        """4-panel matplotlib plot: Re, Im, |H|, phase."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=figsize or (12, 9))
        fig.suptitle(
            f"Raman Frequency Response: {self.response.spec.name}  "
            f"(f_res={self.resonance_frequency_THz:.2f} THz, Q={self.quality_factor:.1f})",
            fontsize=13,
            fontweight="bold",
        )

        w_THz = self.grid.w / (2 * np.pi * 1e12)  # type: ignore[union-attr]
        colors = ["#00d4ff", "#a78bfa", "#34d399", "#fbbf24"]
        titles = ["Re(H(Ω))", "Im(H(Ω))", "|H(Ω)|", "∠H(Ω)"]
        data = [self.H_real, self.H_imag, self.H_magnitude, self.H_phase]
        ylabels = ["Re(H(Ω))", "Im(H(Ω))", "|H(Ω)|", "∠H(Ω) (rad)"]

        for idx, ax in enumerate(axes.flatten()):
            ax.plot(w_THz, data[idx], linewidth=1.5, color=colors[idx])
            ax.set_xlabel("Angular frequency (THz)", fontsize=9)
            ax.set_ylabel(ylabels[idx], fontsize=9)
            ax.set_title(titles[idx], fontsize=10, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color="k", linewidth=0.5)

            # Annotate resonance on Im and |H| panels
            if idx in (1, 2):
                f_res = self.resonance_frequency_THz
                ax.axvline(
                    f_res,
                    color="r",
                    linestyle="--",
                    alpha=0.7,
                    label=f"Resonance: {f_res:.2f} THz",
                )
                ax.legend(fontsize=8)

        plt.tight_layout()
        return fig

    def _plot_all_plotly(self, figsize: tuple[float, float] | None) -> "go.Figure":
        """4-panel plotly plot: Re, Im, |H|, phase."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=["Re(H(Ω))", "Im(H(Ω))", "|H(Ω)|", "∠H(Ω)"],
            shared_xaxes=True,
            shared_yaxes=False,
        )

        w_THz = self.grid.w / (2 * np.pi * 1e12)  # type: ignore[union-attr]
        colors = ["#00d4ff", "#a78bfa", "#34d399", "#fbbf24"]
        data = [self.H_real, self.H_imag, self.H_magnitude, self.H_phase]
        ylabels = ["Re(H(Ω))", "Im(H(Ω))", "|H(Ω)|", "∠H(Ω) (rad)"]

        for idx in range(4):
            row, col = divmod(idx, 2)
            row += 1
            col += 1
            fig.add_trace(
                go.Scatter(
                    x=w_THz,
                    y=data[idx],
                    mode="lines",
                    name=ylabels[idx],
                    line=dict(color=colors[idx], width=1.5),
                ),
                row=row,
                col=col,
            )
            fig.add_hline(
                y=0, line_dash="dot", line_color="gray", opacity=0.5, row=row, col=col
            )

        fig.update_layout(
            title_text=f"Raman Frequency Response: {self.response.spec.name}",
            height=700,
            showlegend=False,
        )
        return fig


# ─── RamanPulseInteraction (Layer 4) ──────────────────────────────────────────


@dataclass(config={"arbitrary_types_allowed": True})
class RamanPulseInteraction:
    """Pulse interaction with Raman-active medium.

    Computes the nonlinear polarization P_NL(t) = n₂ · (R(t) ⊗ I(t))
    where I(t) = |E(t)|² is the pulse intensity and R(t) is the combined
    Raman response from Layer 2.

    Attributes
    ----------
    pulse : Wave
        Input pulse (from pulse.py).
    response : RamanResponse
        Layer 2 Raman response function.
    spec : RamanSpec
        Layer 1 material properties (provides n₂).
    n2 : float
        Nonlinear refractive index n₂ (m²/W). Overrides spec.n2 if set.
    grid : TemporalGrid
        Time grid. Defaults to pulse.grid if not provided.
    """

    pulse: Wave
    response: RamanResponse
    spec: RamanSpec
    n2: float | None = None
    grid: TemporalGrid | None = None

    @model_validator(mode="after")
    def _ensure_defaults(self) -> "RamanPulseInteraction":
        if self.grid is None:
            self.grid = self.pulse.grid
        if self.n2 is None:
            self.n2 = self.spec.n2 or 0.0
        return self

    @property
    def nonlinear_polarization(self) -> NDArray:
        """Nonlinear polarization P_NL(t) = n₂ · (R(t) ⊗ I(t)).

        Computed via FFT-based convolution on the FFTW backend
        (:func:`photonics_helper._fftw.convolve_full`), which reproduces
        ``scipy.signal.fftconvolve(..., mode='full')``.
        The result is cropped to the central N points to match the grid.

        Returns
        -------
        NDArray — nonlinear polarization values.
        """
        I_t = self.pulse.envelope_intensity
        R_t = self.response.combined_response(self.grid.t)  # type: ignore

        # Full convolution, then extract central N points
        P_full = _fftw_convolve_full(I_t, R_t)

        N = self.grid.N  # type: ignore
        if len(P_full) >= N:
            start = (len(P_full) - N) // 2
            P_NL = P_full[start : start + N]
        else:
            P_NL = np.zeros(N)
            P_NL[: len(P_full)] = P_full

        return self.n2 * P_NL

    def plot_interaction(
        self,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        figsize: tuple[float, float] | None = None,
        t_range_ps: tuple[float, float] | None = None,
    ):
        """4-panel plot: input pulse, Raman response, delayed polarization, output.

        Parameters
        ----------
        backend : "matplotlib" or "plotly"
        figsize : Figure size for matplotlib
        t_range_ps : Time range in picoseconds as (t_min, t_max). Overrides grid.
        """
        if backend == "plotly":
            if not HAS_PLOTLY:
                raise ImportError("plotly required for plotly backend")
            return self._plot_interaction_plotly(figsize, t_range_ps)
        else:
            return self._plot_interaction_matplotlib(figsize, t_range_ps)

    def _plot_interaction_matplotlib(
        self,
        figsize: tuple[float, float] | None,
        t_range_ps: tuple[float, float] | None,
    ):
        """Matplotlib 4-panel plot."""
        import matplotlib.pyplot as plt

        if t_range_ps is not None:
            t_min, t_max = t_range_ps
            t = np.linspace(t_min * 1e-12, t_max * 1e-12, 5000)
            I_t = (
                np.abs(
                    self.pulse.envelope_field[:1]
                    if False
                    else self.pulse.envelope_field
                )
                ** 2
            )
            # For custom time range, recompute intensity on the fly
            from photonics_helper.pulse import TemporalGrid

            custom_grid = TemporalGrid(N=len(t), Tmax=t_max * 1e-12)
            custom_grid.t = t  # type: ignore[reportAttributeAccessIssue]  # override cached_property via instance dict
            I_t = np.abs(self.pulse.envelope_field) ** 2
            # Use grid-based computation for consistency
            t = self.grid.t  # type: ignore
        else:
            t = self.grid.t  # type: ignore

        I_t = self.pulse.envelope_intensity
        R_t = self.response.combined_response(t)
        P_NL = self.nonlinear_polarization

        # Simplified output: E_out = E_in + α·P_NL (where α is a coupling constant)
        # For visualization, we show the envelope of the output
        alpha_coupling = 0.1  # arbitrary coupling strength for visualization
        E_in = self.pulse.envelope_field
        E_out = E_in + alpha_coupling * P_NL
        I_out = np.abs(E_out) ** 2

        fig, axes = plt.subplots(4, 1, figsize=figsize or (10, 12), sharex=True)
        fig.suptitle(
            f"Pulse-Raman Interaction: {self.spec.name}  "
            f"(n₂={self.n2:.2e} m²/W, fR={self.response.fR:.2f})",
            fontsize=13,
            fontweight="bold",
        )

        t_ps = t * 1e12

        # Panel 1: Input pulse intensity
        ax1 = axes[0]
        ax1.plot(t_ps, I_t, color="#00d4ff", linewidth=1.5, label="|E(t)|²")
        ax1.set_ylabel("Intensity (arb.)", fontsize=10)
        ax1.set_title("Input Pulse Intensity |E(t)|²", fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right")

        # Panel 2: Raman response
        ax2 = axes[1]
        ax2.plot(t_ps, R_t, color="#a78bfa", linewidth=1.5, label="R(t)")
        ax2.set_ylabel("Amplitude (arb.)", fontsize=10)
        ax2.set_title("Combined Raman Response R(t)", fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0, color="k", linewidth=0.5)
        ax2.legend(loc="upper right")

        # Panel 3: Nonlinear polarization
        ax3 = axes[2]
        ax3.plot(t_ps, P_NL, color="#34d399", linewidth=1.5, label="P_NL(t)")
        ax3.set_ylabel("P_NL (arb.)", fontsize=10)
        ax3.set_title("Nonlinear Polarization P_NL(t) = n₂·(R⊗I)", fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(0, color="k", linewidth=0.5)
        ax3.legend(loc="upper right")

        # Panel 4: Output pulse
        ax4 = axes[3]
        ax4.plot(
            t_ps, I_t, color="#00d4ff", linewidth=1.0, alpha=0.5, label="Input |E|²"
        )
        ax4.plot(t_ps, I_out, color="#fbbf24", linewidth=1.5, label="Output |E_out|²")
        ax4.set_xlabel("Time (ps)", fontsize=10)
        ax4.set_ylabel("Intensity (arb.)", fontsize=10)
        ax4.set_title(f"Output Pulse (α={alpha_coupling})", fontsize=11)
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc="upper right")

        plt.tight_layout()
        return fig

    def _plot_interaction_plotly(
        self,
        figsize: tuple[float, float] | None,
        t_range_ps: tuple[float, float] | None,
    ):
        """Plotly 4-panel plot."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        t = self.grid.t  # type: ignore
        t_ps = t * 1e12
        I_t = self.pulse.envelope_intensity
        R_t = self.response.combined_response(t)
        P_NL = self.nonlinear_polarization

        alpha_coupling = 0.1
        E_in = self.pulse.envelope_field
        E_out = E_in + alpha_coupling * P_NL
        I_out = np.abs(E_out) ** 2

        fig = make_subplots(
            rows=4,
            cols=1,
            subplot_titles=(
                "Input Pulse Intensity |E(t)|²",
                "Combined Raman Response R(t)",
                "Nonlinear Polarization P_NL(t) = n₂·(R⊗I)",
                f"Output Pulse (α={alpha_coupling})",
            ),
            shared_xaxes=True,
            vertical_spacing=0.08,
        )

        fig.add_trace(
            go.Scatter(
                x=t_ps,
                y=I_t,
                mode="lines",
                name="|E|²",
                line=dict(color="#00d4ff", width=1.5),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t_ps,
                y=R_t,
                mode="lines",
                name="R(t)",
                line=dict(color="#a78bfa", width=1.5),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t_ps,
                y=P_NL,
                mode="lines",
                name="P_NL(t)",
                line=dict(color="#34d399", width=1.5),
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t_ps,
                y=I_t,
                mode="lines",
                name="Input",
                opacity=0.5,
                line=dict(color="#00d4ff", width=1.0),
            ),
            row=4,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t_ps,
                y=I_out,
                mode="lines",
                name="Output",
                line=dict(color="#fbbf24", width=1.5),
            ),
            row=4,
            col=1,
        )

        for i in range(1, 5):
            fig.add_hline(
                y=0, line_dash="dot", line_color="gray", opacity=0.3, row=i, col=1
            )

        fig.update_yaxes(title_text="Intensity (arb.)", row=1, col=1)
        fig.update_yaxes(title_text="Amplitude (arb.)", row=2, col=1)
        fig.update_yaxes(title_text="P_NL (arb.)", row=3, col=1)
        fig.update_yaxes(title_text="Intensity (arb.)", row=4, col=1)
        fig.update_xaxes(title_text="Time (ps)", row=4, col=1)

        fig.update_layout(
            title_text=f"Pulse-Raman Interaction: {self.spec.name}  "
            f"(n₂={self.n2:.2e} m²/W, fR={self.response.fR:.2f})",
            height=800,
            showlegend=True,
        )
        return fig

    def animate(
        self,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        frames: int = 20,
        figsize: tuple[float, float] | None = None,
    ):
        """Animation: pulse enters → instantaneous response → lattice oscillation → pulse exits.

        Shows the time evolution of the pulse and the induced nonlinear polarization.

        Parameters
        ----------
        backend : "matplotlib" or "plotly"
        frames : number of animation frames (default 20)
        figsize : figure size for matplotlib
        """
        if backend == "plotly":
            if not HAS_PLOTLY:
                raise ImportError("plotly required for plotly backend")
            return self._animate_plotly(frames, figsize)
        else:
            return self._animate_matplotlib(frames, figsize)

    def _animate_matplotlib(
        self,
        frames: int,
        figsize: tuple[float, float] | None,
    ):
        """Matplotlib animation."""
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        t = self.grid.t  # type: ignore
        t_ps = t * 1e12
        I_t = self.pulse.envelope_intensity
        P_NL = self.nonlinear_polarization

        fig, axes = plt.subplots(2, 1, figsize=figsize or (10, 8), sharex=True)
        fig.suptitle(
            f"Pulse-Raman Interaction Animation: {self.spec.name}",
            fontsize=13,
            fontweight="bold",
        )

        # Line objects for animation
        (line_pulse,) = axes[0].plot(t_ps, I_t, color="#00d4ff", linewidth=1.5)
        (line_PNL,) = axes[1].plot(
            t_ps, np.zeros_like(t_ps), color="#34d399", linewidth=1.5
        )

        axes[0].set_ylabel("Intensity (arb.)", fontsize=10)
        axes[0].set_title("Input Pulse", fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, np.max(I_t) * 1.1)

        axes[1].set_ylabel("P_NL (arb.)", fontsize=10)
        axes[1].set_title("Nonlinear Polarization", fontsize=11)
        axes[1].grid(True, alpha=0.3)
        pNL_max = np.max(np.abs(P_NL)) if np.max(np.abs(P_NL)) > 0 else 1
        axes[1].set_ylim(-pNL_max * 1.1, pNL_max * 1.1)

        axes[1].set_xlabel("Time (ps)", fontsize=10)

        # Animation frame function
        def update(frame):
            # Shift the pulse by a fraction of the grid
            shift = frame / frames
            # Create a shifted version of the intensity
            I_shifted = np.roll(I_t, int(shift * len(I_t)))
            PNL_shifted = np.roll(P_NL, int(shift * len(P_NL)))

            line_pulse.set_ydata(I_shifted)
            line_PNL.set_ydata(PNL_shifted)

            # Update title with frame info
            fig.suptitle(
                f"Pulse-Raman Interaction: {self.spec.name}  "
                f"(frame {frame}/{frames})",
                fontsize=13,
                fontweight="bold",
            )
            return line_pulse, line_PNL

        anim = animation.FuncAnimation(
            fig, update, frames=frames, interval=100, blit=False
        )
        fig._animation = anim  # type: ignore[attr-defined]  # keep reference alive to prevent GC warning

        plt.tight_layout()
        return fig

    def _animate_plotly(
        self,
        frames: int,
        figsize: tuple[float, float] | None,
    ):
        """Plotly animation."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        t = self.grid.t  # type: ignore
        t_ps = t * 1e12
        I_t = self.pulse.envelope_intensity
        P_NL = self.nonlinear_polarization

        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Input Pulse", "Nonlinear Polarization"),
            shared_xaxes=True,
            vertical_spacing=0.1,
        )

        fig.add_trace(
            go.Scatter(
                x=t_ps,
                y=I_t,
                mode="lines",
                name="Intensity",
                line=dict(color="#00d4ff", width=1.5),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t_ps,
                y=P_NL,
                mode="lines",
                name="P_NL",
                line=dict(color="#34d399", width=1.5),
            ),
            row=2,
            col=1,
        )

        fig.update_yaxes(title_text="Intensity (arb.)", row=1, col=1)
        fig.update_yaxes(title_text="P_NL (arb.)", row=2, col=1)
        fig.update_xaxes(title_text="Time (ps)", row=2, col=1)

        fig.update_layout(
            title_text=f"Pulse-Raman Interaction: {self.spec.name}",
            height=600,
            showlegend=False,
        )
        return fig


# ─── PumpWavelengthExplorer (Layer 5) ─────────────────────────────────────────


@dataclass(config={"arbitrary_types_allowed": True})
class PumpWavelengthExplorer:
    """Stokes/Anti-Stokes analysis vs pump wavelength.

    Demonstrates that the Raman frequency shift is constant, but the
    corresponding wavelength shift depends on the pump wavelength.

    This is the key educational insight of Layer 5: equal frequency shifts
    produce unequal wavelength shifts because λ = c/ν (hyperbolic relationship).

    Attributes
    ----------
    spec : RamanSpec
        Layer 1 material properties.
    """

    spec: RamanSpec

    def pump_to_stokes(self, pump_wl: Wavelength) -> Wavelength:
        """Stokes wavelength for a given pump wavelength.

        ν_stokes = ν_pump − ν_Raman
        λ_stokes = c / ν_stokes

        Parameters
        ----------
        pump_wl : Pump wavelength.

        Returns
        -------
        Wavelength — Stokes wavelength.
        """
        return self.spec.stokes_wavelength(pump_wl)

    def pump_to_anti_stokes(self, pump_wl: Wavelength) -> Wavelength:
        """Anti-Stokes wavelength for a given pump wavelength.

        ν_anti_stokes = ν_pump + ν_Raman
        λ_anti_stokes = c / ν_anti_stokes

        Parameters
        ----------
        pump_wl : Pump wavelength.

        Returns
        -------
        Wavelength — Anti-Stokes wavelength.
        """
        return self.spec.anti_stokes_wavelength(pump_wl)

    def plot_frequency_axis(
        self,
        pump_wl: Wavelength,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        freq_range_THz: float = 50,
        figsize: tuple[float, float] | None = None,
    ):
        """Plot frequency axis: Anti-Stokes | Pump | Stokes (equidistant).

        The frequency axis shows equal spacing between Anti-Stokes, Pump,
        and Stokes because the Raman shift is a constant frequency.

        Parameters
        ----------
        pump_wl : Pump wavelength.
        backend : "matplotlib" or "plotly"
        freq_range_THz : Range in THz to display on each side of pump.
        figsize : Figure size for matplotlib.
        """
        if backend == "plotly":
            if not HAS_PLOTLY:
                raise ImportError("plotly required for plotly backend")
            return self._plot_freq_plotly(pump_wl, freq_range_THz, figsize)
        return self._plot_freq_matplotlib(pump_wl, freq_range_THz, figsize)

    def plot_wavelength_axis(
        self,
        pump_wl: Wavelength,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        wl_range_nm: float = 200,
        figsize: tuple[float, float] | None = None,
    ):
        """Plot wavelength axis: Stokes — Pump — Anti-Stokes (unequal spacing).

        Demonstrates that equal frequency shifts produce unequal wavelength
        shifts due to the hyperbolic λ = c/ν relationship.

        Parameters
        ----------
        pump_wl : Pump wavelength.
        backend : "matplotlib" or "plotly"
        wl_range_nm : Range in nm to display around pump.
        figsize : Figure size for matplotlib.
        """
        if backend == "plotly":
            if not HAS_PLOTLY:
                raise ImportError("plotly required for plotly backend")
            return self._plot_wl_plotly(pump_wl, wl_range_nm, figsize)
        return self._plot_wl_matplotlib(pump_wl, wl_range_nm, figsize)

    def plot_both(
        self,
        pump_wl: Wavelength,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        freq_range_THz: float = 50,
        wl_range_nm: float = 200,
        figsize: tuple[float, float] | None = None,
    ):
        """Side-by-side: frequency axis vs wavelength axis.

        Parameters
        ----------
        pump_wl : Pump wavelength.
        backend : "matplotlib" or "plotly"
        freq_range_THz : Frequency range in THz.
        wl_range_nm : Wavelength range in nm.
        figsize : Figure size for matplotlib.
        """
        if backend == "plotly":
            if not HAS_PLOTLY:
                raise ImportError("plotly required for plotly backend")
            return self._plot_both_plotly(pump_wl, freq_range_THz, wl_range_nm, figsize)
        return self._plot_both_matplotlib(pump_wl, freq_range_THz, wl_range_nm, figsize)

    def plot_vs_pump(
        self,
        pump_range_um: tuple[float, float] = (0.4, 2.0),
        n_points: int = 100,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        figsize: tuple[float, float] | None = None,
    ):
        """Sweep pump wavelength: show how Stokes/Anti-Stokes shift changes.

        Parameters
        ----------
        pump_range_um : (min, max) pump wavelength in μm.
        n_points : Number of pump wavelengths to sweep.
        backend : "matplotlib" or "plotly"
        figsize : Figure size for matplotlib.
        """
        if backend == "plotly":
            if not HAS_PLOTLY:
                raise ImportError("plotly required for plotly backend")
            return self._plot_vs_pump_plotly(pump_range_um, n_points, figsize)
        return self._plot_vs_pump_matplotlib(pump_range_um, n_points, figsize)

    # ── Matplotlib implementations ──

    def _plot_freq_matplotlib(
        self,
        pump_wl: Wavelength,
        freq_range_THz: float,
        figsize: tuple[float, float] | None,
    ):
        """Frequency axis plot (equidistant markers)."""
        import matplotlib.pyplot as plt

        nu_pump = C_MS / pump_wl.as_m / 1e12  # THz
        nu_stokes = nu_pump - self.spec.raman_shift_THz
        nu_anti = nu_pump + self.spec.raman_shift_THz

        fig, ax = plt.subplots(figsize=figsize or (10, 5))

        # Draw a frequency axis line
        ax.hlines(
            0,
            nu_pump - freq_range_THz,
            nu_pump + freq_range_THz,
            colors="gray",
            linewidth=2,
        )

        # Mark Anti-Stokes (blue), Pump (red), Stokes (green)
        ax.vlines(
            nu_anti,
            -0.1,
            0.15,
            colors="#3b82f6",
            linewidth=3,
            label=f"Anti-Stokes ({C_MS/nu_anti*1e12:.0f} nm)",
        )
        ax.vlines(
            nu_pump,
            -0.1,
            0.15,
            colors="#ef4444",
            linewidth=3,
            label=f"Pump ({pump_wl.as_nm:.0f} nm)",
        )
        ax.vlines(
            nu_stokes,
            -0.1,
            0.15,
            colors="#22c55e",
            linewidth=3,
            label=f"Stokes ({C_MS/nu_stokes*1e12:.0f} nm)",
        )

        # Annotations
        ax.annotate(
            "Anti-Stokes\n(higher ν)",
            xy=(nu_anti, 0.18),
            ha="center",
            fontsize=9,
            color="#3b82f6",
        )
        ax.annotate(
            "Pump", xy=(nu_pump, -0.18), ha="center", fontsize=9, color="#ef4444"
        )
        ax.annotate(
            "Stokes\n(lower ν)",
            xy=(nu_stokes, 0.18),
            ha="center",
            fontsize=9,
            color="#22c55e",
        )

        # Mark the equal shift
        ax.annotate(
            f"Δν = {self.spec.raman_shift_THz:.2f} THz",
            xy=((nu_pump + nu_stokes) / 2, -0.05),
            ha="center",
            fontsize=8,
            color="gray",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7),
        )

        ax.set_xlim(nu_pump - freq_range_THz, nu_pump + freq_range_THz)
        ax.set_ylim(-0.3, 0.35)
        ax.set_xlabel("Frequency (THz)", fontsize=12)
        ax.set_title(
            f"Frequency Axis: {self.spec.name}  "
            f"(Δν = {self.spec.raman_shift_THz:.2f} THz = {self.spec.raman_shift_cm:.0f} cm⁻¹)",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_yticks([])
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        return fig

    def _plot_wl_matplotlib(
        self,
        pump_wl: Wavelength,
        wl_range_nm: float,
        figsize: tuple[float, float] | None,
    ):
        """Wavelength axis plot (unequal spacing)."""
        import matplotlib.pyplot as plt

        wl_pump = pump_wl.as_nm
        stokes = self.pump_to_stokes(pump_wl).as_nm
        anti = self.pump_to_anti_stokes(pump_wl).as_nm

        fig, ax = plt.subplots(figsize=figsize or (10, 5))

        # Draw a wavelength axis line
        ax.hlines(0, anti - 50, stokes + 50, colors="gray", linewidth=2)

        # Mark Anti-Stokes (blue), Pump (red), Stokes (green)
        ax.vlines(
            anti,
            -0.1,
            0.15,
            colors="#3b82f6",
            linewidth=3,
            label=f"Anti-Stokes ({anti:.0f} nm)",
        )
        ax.vlines(
            wl_pump,
            -0.1,
            0.15,
            colors="#ef4444",
            linewidth=3,
            label=f"Pump ({wl_pump:.0f} nm)",
        )
        ax.vlines(
            stokes,
            -0.1,
            0.15,
            colors="#22c55e",
            linewidth=3,
            label=f"Stokes ({stokes:.0f} nm)",
        )

        # Annotations
        ax.annotate(
            "Anti-Stokes\n(shorter λ)",
            xy=(anti, 0.18),
            ha="center",
            fontsize=9,
            color="#3b82f6",
        )
        ax.annotate(
            "Pump", xy=(wl_pump, -0.18), ha="center", fontsize=9, color="#ef4444"
        )
        ax.annotate(
            "Stokes\n(longer λ)",
            xy=(stokes, 0.18),
            ha="center",
            fontsize=9,
            color="#22c55e",
        )

        # Mark unequal shifts
        delta_stokes = stokes - wl_pump
        delta_anti = wl_pump - anti
        ax.annotate(
            f"Δλ_S = {delta_stokes:.1f} nm",
            xy=((wl_pump + stokes) / 2, -0.05),
            ha="center",
            fontsize=8,
            color="#22c55e",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
        )
        ax.annotate(
            f"Δλ_AS = {delta_anti:.1f} nm",
            xy=((anti + wl_pump) / 2, -0.12),
            ha="center",
            fontsize=8,
            color="#3b82f6",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
        )

        ax.set_xlim(anti - 50, stokes + 50)
        ax.set_ylim(-0.3, 0.35)
        ax.set_xlabel("Wavelength (nm)", fontsize=12)
        ax.set_title(
            f"Wavelength Axis: {self.spec.name}  "
            f"(Pump = {wl_pump:.0f} nm — unequal spacing!)",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_yticks([])
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        return fig

    def _plot_both_matplotlib(
        self,
        pump_wl: Wavelength,
        freq_range_THz: float,
        wl_range_nm: float,
        figsize: tuple[float, float] | None,
    ):
        """Side-by-side frequency and wavelength axis plots."""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=figsize or (16, 5), gridspec_kw={"width_ratios": [1, 1]}
        )
        fig.suptitle(
            f"Pump Wavelength Explorer: {self.spec.name}  "
            f"(Pump = {pump_wl.as_nm:.0f} nm)",
            fontsize=13,
            fontweight="bold",
        )

        # Frequency axis (left)
        nu_pump = C_MS / pump_wl.as_m / 1e12
        nu_stokes = nu_pump - self.spec.raman_shift_THz
        nu_anti = nu_pump + self.spec.raman_shift_THz

        ax1.hlines(
            0,
            nu_pump - freq_range_THz,
            nu_pump + freq_range_THz,
            colors="gray",
            linewidth=2,
        )
        ax1.vlines(
            nu_anti,
            -0.1,
            0.15,
            colors="#3b82f6",
            linewidth=3,
            label=f"AS ({C_MS/nu_anti*1e12:.0f} nm)",
        )
        ax1.vlines(
            nu_pump,
            -0.1,
            0.15,
            colors="#ef4444",
            linewidth=3,
            label=f"Pump ({pump_wl.as_nm:.0f} nm)",
        )
        ax1.vlines(
            nu_stokes,
            -0.1,
            0.15,
            colors="#22c55e",
            linewidth=3,
            label=f"S ({C_MS/nu_stokes*1e12:.0f} nm)",
        )
        ax1.set_xlabel("Frequency (THz)", fontsize=11)
        ax1.set_title("Frequency Axis (equidistant)", fontsize=11)
        ax1.set_yticks([])
        ax1.set_xlim(nu_pump - freq_range_THz, nu_pump + freq_range_THz)
        ax1.set_ylim(-0.3, 0.35)
        ax1.grid(True, alpha=0.3, axis="x")
        ax1.legend(fontsize=8, loc="upper right")

        # Wavelength axis (right)
        wl_pump = pump_wl.as_nm
        stokes = self.pump_to_stokes(pump_wl).as_nm
        anti = self.pump_to_anti_stokes(pump_wl).as_nm

        ax2.hlines(0, anti - 50, stokes + 50, colors="gray", linewidth=2)
        ax2.vlines(
            anti, -0.1, 0.15, colors="#3b82f6", linewidth=3, label=f"AS ({anti:.0f} nm)"
        )
        ax2.vlines(
            wl_pump,
            -0.1,
            0.15,
            colors="#ef4444",
            linewidth=3,
            label=f"Pump ({wl_pump:.0f} nm)",
        )
        ax2.vlines(
            stokes,
            -0.1,
            0.15,
            colors="#22c55e",
            linewidth=3,
            label=f"S ({stokes:.0f} nm)",
        )
        ax2.set_xlabel("Wavelength (nm)", fontsize=11)
        ax2.set_title("Wavelength Axis (unequal spacing!)", fontsize=11)
        ax2.set_yticks([])
        ax2.set_xlim(anti - 50, stokes + 50)
        ax2.set_ylim(-0.3, 0.35)
        ax2.grid(True, alpha=0.3, axis="x")
        ax2.legend(fontsize=8, loc="upper right")

        plt.tight_layout()
        return fig

    def _plot_vs_pump_matplotlib(
        self,
        pump_range_um: tuple[float, float],
        n_points: int,
        figsize: tuple[float, float] | None,
    ):
        """Sweep pump wavelength: Stokes/Anti-Stokes vs pump."""
        import matplotlib.pyplot as plt

        pump_wls_um = np.linspace(pump_range_um[0], pump_range_um[1], n_points)
        stokes_wls_um = []
        anti_wls_um = []

        for wl_um in pump_wls_um:
            pump = Wavelength(wl_um, "um")
            stokes_wls_um.append(self.pump_to_stokes(pump).as_um)
            anti_wls_um.append(self.pump_to_anti_stokes(pump).as_um)

        fig, ax = plt.subplots(figsize=figsize or (10, 6))

        ax.plot(
            pump_wls_um, stokes_wls_um, color="#22c55e", linewidth=2, label="Stokes"
        )
        ax.plot(
            pump_wls_um,
            pump_wls_um,
            color="#ef4444",
            linewidth=1.5,
            linestyle="--",
            label="Pump = Stokes (identity)",
        )
        ax.plot(
            pump_wls_um, anti_wls_um, color="#3b82f6", linewidth=2, label="Anti-Stokes"
        )

        # Shade the Raman shift regions
        ax.fill_between(
            pump_wls_um,
            pump_wls_um,
            stokes_wls_um,
            alpha=0.15,
            color="#22c55e",
            label="Stokes shift",
        )

        ax.set_xlabel("Pump Wavelength (μm)", fontsize=12)
        ax.set_ylabel("Signal Wavelength (μm)", fontsize=12)
        ax.set_title(
            f"Stokes/Anti-Stokes vs Pump: {self.spec.name}  "
            f"(Δν̃ = {self.spec.raman_shift_cm:.0f} cm⁻¹)",
            fontsize=13,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_aspect("auto")

        plt.tight_layout()
        return fig

    # ── Plotly implementations ──

    def _plot_freq_plotly(
        self,
        pump_wl: Wavelength,
        freq_range_THz: float,
        figsize: tuple[float, float] | None,
    ):
        """Frequency axis plot (plotly)."""
        import plotly.graph_objects as go

        nu_pump = C_MS / pump_wl.as_m / 1e12
        nu_stokes = nu_pump - self.spec.raman_shift_THz
        nu_anti = nu_pump + self.spec.raman_shift_THz

        fig = go.Figure()

        # Use add_shape for the axis line (xrange not supported)
        fig.add_shape(
            type="line",
            x0=nu_pump - freq_range_THz,
            x1=nu_pump + freq_range_THz,
            y0=0,
            y1=0,
            line=dict(color="gray", width=3),
        )

        fig.add_vline(
            x=nu_anti,
            y1=0.15,
            line=dict(color="#3b82f6", width=3),
            annotation_text=f"AS ({C_MS/nu_anti*1e12:.0f} nm)",
        )
        fig.add_vline(
            x=nu_pump,
            y1=0.15,
            line=dict(color="#ef4444", width=3),
            annotation_text=f"Pump ({pump_wl.as_nm:.0f} nm)",
        )
        fig.add_vline(
            x=nu_stokes,
            y1=0.15,
            line=dict(color="#22c55e", width=3),
            annotation_text=f"S ({C_MS/nu_stokes*1e12:.0f} nm)",
        )

        fig.update_layout(
            title=f"Frequency Axis: {self.spec.name} (Δν = {self.spec.raman_shift_THz:.2f} THz)",
            xaxis_title="Frequency (THz)",
            yaxis=dict(showticklabels=False, showgrid=False),
            height=300,
            showlegend=False,
        )
        return fig

    def _plot_wl_plotly(
        self,
        pump_wl: Wavelength,
        wl_range_nm: float,
        figsize: tuple[float, float] | None,
    ):
        """Wavelength axis plot (plotly)."""
        import plotly.graph_objects as go

        wl_pump = pump_wl.as_nm
        stokes = self.pump_to_stokes(pump_wl).as_nm
        anti = self.pump_to_anti_stokes(pump_wl).as_nm

        fig = go.Figure()

        fig.add_shape(
            type="line",
            x0=anti - 50,
            x1=stokes + 50,
            y0=0,
            y1=0,
            line=dict(color="gray", width=3),
        )
        fig.add_vline(
            x=anti,
            y1=0.15,
            line=dict(color="#3b82f6", width=3),
            annotation_text=f"AS ({anti:.0f} nm)",
        )
        fig.add_vline(
            x=wl_pump,
            y1=0.15,
            line=dict(color="#ef4444", width=3),
            annotation_text=f"Pump ({wl_pump:.0f} nm)",
        )
        fig.add_vline(
            x=stokes,
            y1=0.15,
            line=dict(color="#22c55e", width=3),
            annotation_text=f"S ({stokes:.0f} nm)",
        )

        fig.update_layout(
            title=f"Wavelength Axis: {self.spec.name} (Pump = {wl_pump:.0f} nm)",
            xaxis_title="Wavelength (nm)",
            yaxis=dict(showticklabels=False, showgrid=False),
            height=300,
            showlegend=False,
        )
        return fig

    def _plot_both_plotly(
        self,
        pump_wl: Wavelength,
        freq_range_THz: float,
        wl_range_nm: float,
        figsize: tuple[float, float] | None,
    ):
        """Side-by-side frequency and wavelength axis (plotly)."""
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                f"Frequency Axis: {self.spec.name}",
                f"Wavelength Axis: {self.spec.name}",
            ),
        )

        # Frequency axis
        nu_pump = C_MS / pump_wl.as_m / 1e12
        nu_stokes = nu_pump - self.spec.raman_shift_THz
        nu_anti = nu_pump + self.spec.raman_shift_THz

        fig.add_shape(
            type="line",
            x0=nu_pump - freq_range_THz,
            x1=nu_pump + freq_range_THz,
            y0=0,
            y1=0,
            line=dict(color="gray", width=2),
            row=1,
            col=1,
        )
        fig.add_vline(
            x=nu_anti, y1=0.15, line=dict(color="#3b82f6", width=3), row=1, col=1
        )
        fig.add_vline(
            x=nu_pump, y1=0.15, line=dict(color="#ef4444", width=3), row=1, col=1
        )
        fig.add_vline(
            x=nu_stokes, y1=0.15, line=dict(color="#22c55e", width=3), row=1, col=1
        )

        # Wavelength axis
        wl_pump = pump_wl.as_nm
        stokes = self.pump_to_stokes(pump_wl).as_nm
        anti = self.pump_to_anti_stokes(pump_wl).as_nm

        fig.add_shape(
            type="line",
            x0=anti - 50,
            x1=stokes + 50,
            y0=0,
            y1=0,
            line=dict(color="gray", width=2),
            row=1,
            col=2,
        )
        fig.add_vline(
            x=anti, y1=0.15, line=dict(color="#3b82f6", width=3), row=1, col=2
        )
        fig.add_vline(
            x=wl_pump, y1=0.15, line=dict(color="#ef4444", width=3), row=1, col=2
        )
        fig.add_vline(
            x=stokes, y1=0.15, line=dict(color="#22c55e", width=3), row=1, col=2
        )

        fig.update_layout(height=350, showlegend=False)
        return fig

    def _plot_vs_pump_plotly(
        self,
        pump_range_um: tuple[float, float],
        n_points: int,
        figsize: tuple[float, float] | None,
    ):
        """Sweep pump wavelength (plotly)."""
        import plotly.graph_objects as go

        pump_wls_um = np.linspace(pump_range_um[0], pump_range_um[1], n_points)
        stokes_wls_um = []
        anti_wls_um = []

        for wl_um in pump_wls_um:
            pump = Wavelength(wl_um, "um")
            stokes_wls_um.append(self.pump_to_stokes(pump).as_um)
            anti_wls_um.append(self.pump_to_anti_stokes(pump).as_um)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=pump_wls_um,
                y=stokes_wls_um,
                mode="lines",
                name="Stokes",
                line=dict(color="#22c55e", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=pump_wls_um,
                y=pump_wls_um,
                mode="lines",
                name="Pump = Stokes (identity)",
                line=dict(color="#ef4444", width=1.5, dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=pump_wls_um,
                y=anti_wls_um,
                mode="lines",
                name="Anti-Stokes",
                line=dict(color="#3b82f6", width=2),
            )
        )

        fig.update_layout(
            title=f"Stokes/Anti-Stokes vs Pump: {self.spec.name}  "
            f"(Δν̃ = {self.spec.raman_shift_cm:.0f} cm⁻¹)",
            xaxis_title="Pump Wavelength (μm)",
            yaxis_title="Signal Wavelength (μm)",
            height=500,
        )
        return fig


# ─── MaterialComparison (Layer 6) ─────────────────────────────────────────────


# Preset comparison groups for quick multi-material visualization
COMMON_COMPARISONS = {
    "glass_vs_chalcogenide": ["Silica", "As2Se3"],
    "semiconductor": ["CdS", "GaAs", "Si", "Ge"],
    "high_n2": ["Silica", "As2Se3", "Diamond"],
    "high_shift": ["Silica", "Diamond", "Si"],
    "nitride_semiconductor": ["GaN", "AlN", "Si3N4"],
    "nonlinear_crystal": ["LiNbO3", "KTP", "LBO", "BaTiO3"],
    "high_gain": ["Diamond", "As2Se3", "YAG"],
    "wide_bandgap": ["Diamond", "GaN", "SiC_4H", "AlN", "Ga2O3"],
    "iii_v": ["GaAs", "InP", "AlGaAs", "InGaAs"],
    "laser_host": ["YAG", "Al2O3", "YLF"],
    "nlo_crystal": ["LBO", "KTP", "AgGaS2", "AgGaSe2", "LiNbO3"],
    "chalcogenide": ["As2S3", "As2Se3"],
    "ferroelectric": ["LiNbO3", "LiTaO3", "BaTiO3", "KTP"],
    "negative_n2": ["ZnO", "CdTe"],
}


# Color palette for overlay plots (8 distinct colors)
_OVERLAY_COLORS = [
    "#00d4ff",  # cyan
    "#a78bfa",  # purple
    "#34d399",  # green
    "#fbbf24",  # amber
    "#f87171",  # red
    "#fb923c",  # orange
    "#38bdf8",  # sky blue
    "#c084fc",  # violet
]


@dataclass(config={"arbitrary_types_allowed": True})
class MaterialComparison:
    """Multi-material Raman comparison overlay.

    Overlays Raman spectra, time-domain responses, and frequency-domain
    responses for multiple materials on shared axes for direct comparison.

    Attributes
    ----------
    materials : list[RamanSpec]
        List of RamanSpec instances to compare.
    """

    materials: list[RamanSpec] = Field(default_factory=list)

    def add(self, spec: RamanSpec) -> None:
        """Add a material, replacing if a material with the same name exists.

        Parameters
        ----------
        spec : RamanSpec
            Material to add.
        """
        # Remove existing material with the same name
        self.materials = [m for m in self.materials if m.name != spec.name]
        self.materials.append(spec)

    def remove(self, name: str) -> None:
        """Remove a material by name.

        Parameters
        ----------
        name : str
            Material name to remove.
        """
        self.materials = [m for m in self.materials if m.name != name]

    def clear(self) -> None:
        """Remove all materials."""
        self.materials.clear()

    def _color_for_index(self, idx: int) -> str:
        """Get a color for a material index from the palette."""
        return _OVERLAY_COLORS[idx % len(_OVERLAY_COLORS)]

    def plot_spectra_overlay(
        self,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        shift_range_cm: float = 200,
        n_points: int = 1000,
        figsize: tuple[float, float] | None = None,
    ):
        """Overlay Raman spectra for all materials.

        Each material is plotted as a Lorentzian lineshape centered at its
        Raman shift, normalized to unit peak height.

        Parameters
        ----------
        backend : "matplotlib" or "plotly"
        shift_range_cm : Range around 0 to plot (cm⁻¹)
        n_points : Number of points
        figsize : Figure size for matplotlib

        Returns
        -------
        fig or None — None if no materials.
        """
        if not self.materials:
            return None

        if backend == "plotly":
            if not HAS_PLOTLY:
                raise ImportError("plotly required for plotly backend")
            return self._plot_spectra_plotly(shift_range_cm, n_points)
        return self._plot_spectra_matplotlib(shift_range_cm, n_points, figsize)

    def _plot_spectra_matplotlib(
        self,
        shift_range_cm: float,
        n_points: int,
        figsize: tuple[float, float] | None,
    ):
        """Matplotlib spectra overlay."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize or (10, 6))
        name_labels = []

        for idx, spec in enumerate(self.materials):
            if spec.raman_shift_cm is None or spec.raman_linewidth_cm is None:
                continue

            color = self._color_for_index(idx)

            # Use PhononResponse for multi-mode materials
            if spec.phonon_modes:
                pr = spec.phonon_response
                if pr is None:
                    continue
                shift = np.linspace(-shift_range_cm, shift_range_cm, n_points)
                intensity = pr.frequency_domain(shift)
                ax.plot(
                    shift,
                    intensity,
                    linewidth=2,
                    color=color,
                    label=f"{spec.name} ({len(spec.phonon_modes)} modes)",
                )
            else:
                center = spec.raman_shift_cm
                width = spec.raman_linewidth_cm / 2

                shift = np.linspace(-shift_range_cm, shift_range_cm, n_points)
                intensity = (width / np.pi) / ((shift - center) ** 2 + width**2)
                intensity /= np.max(intensity)

                ax.plot(shift, intensity, linewidth=2, color=color, label=spec.name)

            name_labels.append(spec.name)

        ax.set_xlabel("Raman shift (cm⁻¹)", fontsize=12)
        ax.set_ylabel("Intensity (arb.)", fontsize=12)
        ax.set_title(
            f"Raman Spectra Comparison: {', '.join(name_labels)}",
            fontsize=13,
            fontweight="bold",
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def _plot_spectra_plotly(self, shift_range_cm: float, n_points: int):
        """Plotly spectra overlay."""
        import plotly.graph_objects as go

        fig = go.Figure()
        name_labels = []

        for idx, spec in enumerate(self.materials):
            if spec.raman_shift_cm is None or spec.raman_linewidth_cm is None:
                continue

            color = self._color_for_index(idx)

            # Use PhononResponse for multi-mode materials
            if spec.phonon_modes:
                pr = spec.phonon_response
                if pr is None:
                    continue
                shift = np.linspace(-shift_range_cm, shift_range_cm, n_points)
                intensity = pr.frequency_domain(shift)
                label = f"{spec.name} ({len(spec.phonon_modes)} modes)"
            else:
                center = spec.raman_shift_cm
                width = spec.raman_linewidth_cm / 2

                shift = np.linspace(-shift_range_cm, shift_range_cm, n_points)
                intensity = (width / np.pi) / ((shift - center) ** 2 + width**2)
                intensity /= np.max(intensity)
                label = spec.name

            fig.add_trace(
                go.Scatter(
                    x=shift,
                    y=intensity,
                    mode="lines",
                    name=label,
                    line=dict(color=color, width=2),
                )
            )
            name_labels.append(spec.name)

        fig.update_layout(
            title=f"Raman Spectra Comparison: {', '.join(name_labels)}",
            xaxis_title="Raman shift (cm⁻¹)",
            yaxis_title="Intensity (arb.)",
            height=500,
        )
        return fig

    def plot_response_overlay(
        self,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        grid: TemporalGrid | None = None,
        figsize: tuple[float, float] | None = None,
        t_range_ps: tuple[float, float] | None = None,
    ):
        """Overlay h_R(t) for all materials.

        Computes the delayed Raman response for each material and overlays
        them on a shared time axis.

        Parameters
        ----------
        backend : "matplotlib" or "plotly"
        grid : TemporalGrid — shared time grid. Auto-derived if None.
        figsize : Figure size for matplotlib
        t_range_ps : Time range in picoseconds as (t_min, t_max). Overrides grid.

        Returns
        -------
        fig or None — None if no materials.
        """
        if not self.materials:
            return None

        if grid is None:
            # Use the longest τ2 to determine grid size
            max_tau2 = max(
                (1.0 / (np.pi * s.linewidth_Hz) if s.linewidth_Hz > 0 else 1e-12)
                for s in self.materials
            )
            grid = TemporalGrid(N=2**14, Tmax=max(10e-12, 20 * max_tau2))

        if backend == "plotly":
            if not HAS_PLOTLY:
                raise ImportError("plotly required for plotly backend")
            return self._plot_response_plotly(grid, t_range_ps)
        return self._plot_response_matplotlib(grid, figsize, t_range_ps)

    @staticmethod
    def _compute_h_R(spec: "RamanSpec", t: NDArray) -> NDArray:
        """Compute delayed Raman response fR·h_R(t) for a single material.

        Standard Agrawal exponential-damped form:
        h_R(t) = (τ₁² + τ₂²)/(τ₁·τ₂²) · exp(-t/τ₂) · sin(t/τ₁)   for t ≥ 0
        h_R(t) = 0                                                  for t < 0

        Parameters
        ----------
        spec : RamanSpec
            Material with raman_shift_Hz, linewidth_Hz, fR.
        t : NDArray
            Time array.

        Returns
        -------
        NDArray — delayed response fR·h_R(t).
        """
        tau1 = 1.0 / (2 * np.pi * spec.raman_shift_Hz)
        tau2 = 1.0 / (np.pi * spec.linewidth_Hz) if spec.linewidth_Hz > 0 else 1e-12
        prefactor = (tau1**2 + tau2**2) / (tau1 * tau2**2)
        exponential = np.exp(-t / tau2)
        oscillation = np.sin(t / tau1)
        mask = t >= 0
        delayed = np.zeros_like(t)
        raw_h = prefactor * exponential[mask] * oscillation[mask]
        integral = np.trapezoid(raw_h, t[mask])
        if integral > 0:
            raw_h /= integral
        delayed[mask] = spec.fR * raw_h
        return delayed

    def _plot_response_matplotlib(
        self,
        grid: TemporalGrid,
        figsize: tuple[float, float] | None,
        t_range_ps: tuple[float, float] | None,
    ):
        """Matplotlib response overlay."""
        import matplotlib.pyplot as plt

        if t_range_ps is not None:
            t_min, t_max = t_range_ps
            t = np.linspace(t_min * 1e-12, t_max * 1e-12, 5000)
        else:
            t = grid.t

        fig, ax = plt.subplots(figsize=figsize or (10, 6))
        name_labels = []

        for idx, spec in enumerate(self.materials):
            if spec.fR is None or spec.fR == 0:
                continue
            if spec.raman_shift_Hz <= 0:
                continue

            color = self._color_for_index(idx)

            delayed = self._compute_h_R(spec, t)

            # Normalize to peak for visibility
            peak = np.max(np.abs(delayed))
            if peak > 0:
                delayed /= peak

            ax.plot(t * 1e12, delayed, linewidth=1.5, color=color, label=spec.name)
            name_labels.append(spec.name)

        ax.set_xlabel("Time (ps)", fontsize=12)
        ax.set_ylabel("Delayed response h_R(t) (normalized)", fontsize=11)
        ax.set_title(
            f"Raman Response Overlay: {', '.join(name_labels)}",
            fontsize=13,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.legend(fontsize=10)
        plt.tight_layout()
        return fig

    def _plot_response_plotly(
        self, grid: TemporalGrid, t_range_ps: tuple[float, float] | None
    ):
        """Plotly response overlay."""
        import plotly.graph_objects as go

        if t_range_ps is not None:
            t_min, t_max = t_range_ps
            t = np.linspace(t_min * 1e-12, t_max * 1e-12, 5000)
        else:
            t = grid.t

        fig = go.Figure()
        name_labels = []

        for idx, spec in enumerate(self.materials):
            if spec.fR is None or spec.fR == 0:
                continue
            if spec.raman_shift_Hz <= 0:
                continue

            color = self._color_for_index(idx)

            delayed = self._compute_h_R(spec, t)

            peak = np.max(np.abs(delayed))
            if peak > 0:
                delayed /= peak

            fig.add_trace(
                go.Scatter(
                    x=t * 1e12,
                    y=delayed,
                    mode="lines",
                    name=spec.name,
                    line=dict(color=color, width=1.5),
                )
            )
            name_labels.append(spec.name)

        fig.update_layout(
            title=f"Raman Response Overlay: {', '.join(name_labels)}",
            xaxis_title="Time (ps)",
            yaxis_title="Delayed response h_R(t) (normalized)",
            height=500,
        )
        return fig

    def plot_frequency_overlay(
        self,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        grid: TemporalGrid | None = None,
        figsize: tuple[float, float] | None = None,
    ):
        """Overlay Im(H(Ω)) for all materials.

        The imaginary part of the Fourier transform gives the Raman gain
        spectrum. Overlapping multiple materials shows how their gain
        profiles compare.

        Parameters
        ----------
        backend : "matplotlib" or "plotly"
        grid : TemporalGrid — shared time grid. Auto-derived if None.
        figsize : Figure size for matplotlib

        Returns
        -------
        fig or None — None if no materials.
        """
        if not self.materials:
            return None

        if grid is None:
            max_tau2 = max(
                (1.0 / (np.pi * s.linewidth_Hz) if s.linewidth_Hz > 0 else 1e-12)
                for s in self.materials
            )
            grid = TemporalGrid(N=2**14, Tmax=max(10e-12, 20 * max_tau2))

        if backend == "plotly":
            if not HAS_PLOTLY:
                raise ImportError("plotly required for plotly backend")
            return self._plot_frequency_plotly(grid)
        return self._plot_frequency_matplotlib(grid, figsize)

    def _plot_frequency_matplotlib(
        self,
        grid: TemporalGrid,
        figsize: tuple[float, float] | None,
    ):
        """Matplotlib frequency overlay."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize or (10, 6))
        name_labels = []

        for idx, spec in enumerate(self.materials):
            if spec.fR is None or spec.fR == 0:
                continue
            if spec.raman_shift_Hz <= 0:
                continue

            color = self._color_for_index(idx)

            # Compute h_R(t) and FFT
            h_R_t = self._compute_h_R(spec, grid.t)
            H = grid.fft(h_R_t)
            H_imag = np.imag(H)

            w_THz = grid.w / (2 * np.pi * 1e12)
            ax.plot(w_THz, H_imag, linewidth=1.5, color=color, label=spec.name)
            name_labels.append(spec.name)

            # Annotate resonance frequency
            if np.max(np.abs(H_imag)) > 0:
                f_res = w_THz[np.argmax(np.abs(H_imag))]
                ax.axvline(f_res, color=color, linestyle=":", alpha=0.5, linewidth=0.8)

        ax.set_xlabel("Angular frequency (THz)", fontsize=12)
        ax.set_ylabel("Im(H(Ω))", fontsize=12)
        ax.set_title(
            f"Raman Gain Spectrum: {', '.join(name_labels)}",
            fontsize=13,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.legend(fontsize=10)
        plt.tight_layout()
        return fig

    def _plot_frequency_plotly(
        self,
        grid: TemporalGrid,
    ):
        """Plotly frequency overlay."""
        import plotly.graph_objects as go

        fig = go.Figure()
        name_labels = []

        for idx, spec in enumerate(self.materials):
            if spec.fR is None or spec.fR == 0:
                continue
            if spec.raman_shift_Hz <= 0:
                continue

            color = self._color_for_index(idx)

            h_R_t = self._compute_h_R(spec, grid.t)
            H = grid.fft(h_R_t)
            H_imag = np.imag(H)

            w_THz = grid.w / (2 * np.pi * 1e12)
            fig.add_trace(
                go.Scatter(
                    x=w_THz,
                    y=H_imag,
                    mode="lines",
                    name=spec.name,
                    line=dict(color=color, width=1.5),
                )
            )
            name_labels.append(spec.name)

        fig.update_layout(
            title=f"Raman Gain Spectrum: {', '.join(name_labels)}",
            xaxis_title="Angular frequency (THz)",
            yaxis_title="Im(H(Ω))",
            height=500,
        )
        return fig

    def comparison_table(self) -> str:
        """Text table comparing Raman properties across all materials.

        Returns
        -------
        str — formatted comparison table.
        """
        if not self.materials:
            return "No materials in comparison."

        # Header
        header = (
            f"{'Material':<12} {'n₂ (m²/W)':>14} {'fR':>6} "
            f"{'Shift (cm⁻¹)':>13} {'FWHM (cm⁻¹)':>13} "
            f"{'τ1 (fs)':>9} {'τ2 (fs)':>9} {'Q':>8}"
        )
        separator = "─" * len(header)

        lines = [header, separator]

        for spec in self.materials:
            n2_str = f"{spec.n2:.2e}" if spec.n2 else "N/A"
            fr_str = f"{spec.fR:.2f}" if spec.fR is not None else "N/A"
            shift_str = f"{spec.raman_shift_cm:.0f}" if spec.raman_shift_cm else "N/A"
            fwhm_str = (
                f"{spec.raman_linewidth_cm:.0f}" if spec.raman_linewidth_cm else "N/A"
            )

            if spec.raman_shift_Hz > 0:
                tau1_fs = (1.0 / spec.raman_shift_Hz) * 1e15
                tau1_str = f"{tau1_fs:.1f}"
            else:
                tau1_str = "N/A"

            if spec.linewidth_Hz > 0:
                tau2_fs = (1.0 / (np.pi * spec.linewidth_Hz)) * 1e15
                tau2_str = f"{tau2_fs:.1f}"
            else:
                tau2_str = "N/A"

            if spec.quality_factor != float("inf") and spec.quality_factor != 0:
                q_str = f"{spec.quality_factor:.1f}"
            else:
                q_str = "∞" if spec.raman_shift_Hz > 0 else "N/A"

            lines.append(
                f"{spec.name:<12} {n2_str:>14} {fr_str:>6} "
                f"{shift_str:>13} {fwhm_str:>13} "
                f"{tau1_str:>9} {tau2_str:>9} {q_str:>8}"
            )

        return "\n".join(lines)

    def plot_all(
        self,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        grid: TemporalGrid | None = None,
        figsize: tuple[float, float] | None = None,
    ):
        """Full comparison: spectra + response + frequency for all materials.

        3-panel layout:
        1. Raman spectra overlay
        2. Time-domain response overlay
        3. Frequency-domain gain spectrum overlay

        Parameters
        ----------
        backend : "matplotlib" or "plotly"
        grid : TemporalGrid — shared time grid. Auto-derived if None.
        figsize : Figure size for matplotlib

        Returns
        -------
        fig or None — None if no materials.
        """
        if not self.materials:
            return None

        if grid is None:
            max_tau2 = max(
                (1.0 / (np.pi * s.linewidth_Hz) if s.linewidth_Hz > 0 else 1e-12)
                for s in self.materials
            )
            grid = TemporalGrid(N=2**14, Tmax=max(10e-12, 20 * max_tau2))

        if backend == "plotly":
            if not HAS_PLOTLY:
                raise ImportError("plotly required for plotly backend")
            return self._plot_all_plotly(grid)
        return self._plot_all_matplotlib(grid, figsize)

    def _plot_all_matplotlib(
        self,
        grid: TemporalGrid,
        figsize: tuple[float, float] | None,
    ):
        """Matplotlib 3-panel comparison."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=figsize or (12, 12), sharex=False)
        fig.suptitle(
            f"Raman Material Comparison: {', '.join(s.name for s in self.materials)}",
            fontsize=14,
            fontweight="bold",
        )

        # Panel 1: Spectra
        axes[0].set_title("Raman Spectra", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Raman shift (cm⁻¹)", fontsize=10)
        axes[0].set_ylabel("Intensity (arb.)", fontsize=10)
        # Reuse spectra logic inline for the subplot
        for idx, spec in enumerate(self.materials):
            if spec.raman_shift_cm is None or spec.raman_linewidth_cm is None:
                continue
            color = self._color_for_index(idx)
            center = spec.raman_shift_cm
            width = spec.raman_linewidth_cm / 2
            shift = np.linspace(-200, 200, 1000)
            intensity = (width / np.pi) / ((shift - center) ** 2 + width**2)
            intensity /= np.max(intensity)
            axes[0].plot(shift, intensity, linewidth=2, color=color, label=spec.name)
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)

        # Panel 2: Response
        axes[1].set_title("Time-Domain Response h_R(t)", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Time (ps)", fontsize=10)
        axes[1].set_ylabel("Amplitude (normalized)", fontsize=10)
        for idx, spec in enumerate(self.materials):
            if spec.fR is None or spec.fR == 0 or spec.raman_shift_Hz <= 0:
                continue
            color = self._color_for_index(idx)
            delayed = self._compute_h_R(spec, grid.t)
            peak = np.max(np.abs(delayed))
            if peak > 0:
                delayed /= peak
            axes[1].plot(
                grid.t * 1e12, delayed, linewidth=1.5, color=color, label=spec.name
            )
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(0, color="k", linewidth=0.5)
        axes[1].legend(fontsize=9)

        # Panel 3: Frequency
        axes[2].set_title(
            "Raman Gain Spectrum Im(H(Ω))", fontsize=12, fontweight="bold"
        )
        axes[2].set_xlabel("Angular frequency (THz)", fontsize=10)
        axes[2].set_ylabel("Im(H(Ω))", fontsize=10)
        for idx, spec in enumerate(self.materials):
            if spec.fR is None or spec.fR == 0 or spec.raman_shift_Hz <= 0:
                continue
            color = self._color_for_index(idx)
            h_R_t = self._compute_h_R(spec, grid.t)
            H = grid.fft(h_R_t)
            H_imag = np.imag(H)
            w_THz = grid.w / (2 * np.pi * 1e12)
            axes[2].plot(w_THz, H_imag, linewidth=1.5, color=color, label=spec.name)
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(0, color="k", linewidth=0.5)
        axes[2].legend(fontsize=9)

        plt.tight_layout()
        return fig

    def _plot_all_plotly(self, grid: TemporalGrid):
        """Plotly 3-panel comparison."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=(
                "Raman Spectra",
                "Time-Domain Response h_R(t)",
                "Raman Gain Spectrum Im(H(Ω))",
            ),
            shared_xaxes=False,
            vertical_spacing=0.1,
        )

        # Panel 1: Spectra
        for idx, spec in enumerate(self.materials):
            if spec.raman_shift_cm is None or spec.raman_linewidth_cm is None:
                continue
            color = self._color_for_index(idx)
            center = spec.raman_shift_cm
            width = spec.raman_linewidth_cm / 2
            shift = np.linspace(-200, 200, 1000)
            intensity = (width / np.pi) / ((shift - center) ** 2 + width**2)
            intensity /= np.max(intensity)
            fig.add_trace(
                go.Scatter(
                    x=shift,
                    y=intensity,
                    mode="lines",
                    name=spec.name,
                    line=dict(color=color, width=2),
                ),
                row=1,
                col=1,
            )

        # Panel 2: Response
        for idx, spec in enumerate(self.materials):
            if spec.fR is None or spec.fR == 0 or spec.raman_shift_Hz <= 0:
                continue
            color = self._color_for_index(idx)
            delayed = self._compute_h_R(spec, grid.t)
            peak = np.max(np.abs(delayed))
            if peak > 0:
                delayed /= peak
            fig.add_trace(
                go.Scatter(
                    x=grid.t * 1e12,
                    y=delayed,
                    mode="lines",
                    name=spec.name,
                    line=dict(color=color, width=1.5),
                ),
                row=2,
                col=1,
            )

        # Panel 3: Frequency
        for idx, spec in enumerate(self.materials):
            if spec.fR is None or spec.fR == 0 or spec.raman_shift_Hz <= 0:
                continue
            color = self._color_for_index(idx)
            h_R_t = self._compute_h_R(spec, grid.t)
            H = grid.fft(h_R_t)
            H_imag = np.imag(H)
            w_THz = grid.w / (2 * np.pi * 1e12)
            fig.add_trace(
                go.Scatter(
                    x=w_THz,
                    y=H_imag,
                    mode="lines",
                    name=spec.name,
                    line=dict(color=color, width=1.5),
                ),
                row=3,
                col=1,
            )

        fig.update_layout(height=900, showlegend=True)
        return fig


# ─── Dash App (Layer 7) ──────────────────────────────────────────────────────


def app() -> "dash.Dash":  # type: ignore[valid-type]
    """Interactive Dash app tying all 6 Raman layers together.

    Layout::

        ┌─────────────────────────────────────────────────┐
        │  Photonics Helper — Raman Explorer              │
        ├──────────┬──────────────────────────────────────┤
        │ Sidebar  │  Main content area                   │
        │          │                                      │
        │ Material │  [Layer selector tabs]               │
        │ selector │  ┌──────────────────────────────┐    │
        │          │  │                              │    │
        │ fR slider│  │   Active visualization       │    │
        │          │  │                              │    │
        │ τ1 slider│  │                              │    │
        │          │  └──────────────────────────────┘    │
        │ τ2 slider│                                      │
        │          │  [Data table / summary]              │
        │ Pump λ   │                                      │
        │ slider   │                                      │
        │          │                                      │
        │ Compare  │                                      │
        │ [+ Add]  │                                      │
        └──────────┴──────────────────────────────────────┘

    Returns
    -------
    dash.Dash — configured Dash application object.

    Notes
    -----
    Requires `dash` to be installed. Run with::

        if __name__ == "__main__":
            app().run(debug=True)
    """
    try:
        import dash
        from dash import (
            html,
            dcc,
            Input,
            Output,
            State,
        )
    except ImportError as exc:
        raise ImportError(
            "Dash is required for the interactive app. "
            "Install it with: pip install dash>=2.18.0"
        ) from exc

    dash_app = dash.Dash(__name__, title="Photonics Helper — Raman Explorer")

    # ── Build layout ──────────────────────────────────────────────────────────

    sidebar = html.Div(
        [
            html.H4("Material", style={"marginBottom": "5px"}),
            dcc.Dropdown(
                id="material-selector",
                options=[
                    {"label": name, "value": name}
                    for name in sorted(RAMAN_MATERIALS.keys())
                ],
                value="Silica",
                clearable=False,
            ),
            html.Hr(),
            html.H4("Response Parameters", style={"marginBottom": "5px"}),
            html.Label("fR:"),
            dcc.Slider(
                id="fr-slider",
                min=0.0,
                max=1.0,
                step=0.01,
                value=0.18,
                marks={0.0: "0", 0.25: "0.25", 0.5: "0.5", 0.75: "0.75", 1.0: "1.0"},
            ),
            html.Label("τ1 (fs):"),
            dcc.Slider(
                id="tau1-slider",
                min=0.1,
                max=50.0,
                step=0.1,
                marks={1: "1", 5: "5", 10: "10", 20: "20", 50: "50"},
            ),
            html.Label("τ2 (fs):"),
            dcc.Slider(
                id="tau2-slider",
                min=0.5,
                max=100.0,
                step=0.5,
                marks={1: "1", 10: "10", 25: "25", 50: "50", 100: "100"},
            ),
            html.Div(id="tau1-display", style={"fontSize": "11px", "color": "#666"}),
            html.Div(id="tau2-display", style={"fontSize": "11px", "color": "#666"}),
            html.Hr(),
            html.H4("Pump Wavelength", style={"marginBottom": "5px"}),
            dcc.Slider(
                id="pump-wl-slider",
                min=400,
                max=2500,
                step=10,
                marks={
                    500: "500nm",
                    800: "800nm",
                    1000: "1μm",
                    1550: "1550nm",
                    2000: "2μm",
                },
                value=800,
            ),
            html.Div(id="pump-wl-display", style={"fontSize": "11px", "color": "#666"}),
            html.Hr(),
            html.H4("Compare", style={"marginBottom": "5px"}),
            html.Button("+ Add to Compare", id="add-to-compare-btn", n_clicks=0),
            html.Div(id="compare-list", style={"marginTop": "8px", "fontSize": "12px"}),
        ],
        style={
            "width": "280px",
            "minWidth": "280px",
            "padding": "15px",
            "backgroundColor": "#f8f9fa",
            "borderRight": "1px solid #dee2e6",
            "overflowY": "auto",
            "height": "100vh",
        },
    )

    main_content = html.Div(
        [
            html.H2(
                "Photonics Helper — Raman Explorer",
                style={"textAlign": "center", "marginBottom": "5px"},
            ),
            html.P(
                "Interactive Raman scattering explorer — connect equations, intuition, and visualization",
                style={"textAlign": "center", "color": "#666", "marginBottom": "15px"},
            ),
            # Layer selector tabs
            dcc.Tabs(
                id="layer-tabs",
                value="layer-2-response",
                children=[
                    dcc.Tab(label="1 — Material", value="layer-1-material"),
                    dcc.Tab(label="2 — Response", value="layer-2-response"),
                    dcc.Tab(label="3 — Frequency", value="layer-3-frequency"),
                    dcc.Tab(label="4 — Pulse", value="layer-4-pulse"),
                    dcc.Tab(label="5 — Pump λ", value="layer-5-pump"),
                    dcc.Tab(label="6 — Compare", value="layer-6-compare"),
                ],
            ),
            html.Br(),
            # Output area
            html.Div(id="output-container", style={"margin": "10px 0"}),
            # Summary / data table
            html.Div(
                id="summary-container",
                style={
                    "padding": "10px",
                    "backgroundColor": "#fff",
                    "border": "1px solid #dee2e6",
                    "borderRadius": "4px",
                    "fontFamily": "monospace",
                    "whiteSpace": "pre-wrap",
                },
            ),
        ],
        style={"flex": "1", "padding": "10px", "overflowY": "auto"},
    )

    dash_app.layout = html.Div(
        [
            html.Div(
                [sidebar, main_content],
                style={
                    "display": "flex",
                    "minHeight": "100vh",
                },
            ),
        ]
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    @dash_app.callback(
        [Output("tau1-display", "children"), Output("tau2-display", "children")],
        Input("material-selector", "value"),
        Input("fr-slider", "value"),
        Input("tau1-slider", "value"),
        Input("tau2-slider", "value"),
    )
    def _update_tau_displays(selected_material, fr_val, tau1_val, tau2_val):
        """Show τ1/τ2 values in sidebar."""
        spec = RamanSpec.from_database(selected_material)
        # Auto-derived tau1/tau2 from the material
        if spec.raman_shift_Hz > 0:
            auto_tau1_fs = (1.0 / spec.raman_shift_Hz) * 1e15
        else:
            auto_tau1_fs = 0.0
        if spec.linewidth_Hz > 0:
            auto_tau2_fs = (1.0 / (np.pi * spec.linewidth_Hz)) * 1e15
        else:
            auto_tau2_fs = 0.0
        return (
            f"Auto τ1 = {auto_tau1_fs:.2f} fs (slider: {tau1_val:.1f} fs)",
            f"Auto τ2 = {auto_tau2_fs:.2f} fs (slider: {tau2_val:.1f} fs)",
        )

    @dash_app.callback(
        Output("compare-list", "children"),
        Input("add-to-compare-btn", "n_clicks"),
        State("material-selector", "value"),
        prevent_initial_call=True,
    )
    def _add_to_compare(n_clicks, material_name):
        """Add current material to the comparison list (display only)."""
        if n_clicks == 0:
            return html.P("No materials added yet.")
        return html.P(f"Added: {material_name}")

    @dash_app.callback(
        Output("output-container", "children"),
        Output("summary-container", "children"),
        Input("layer-tabs", "value"),
        Input("material-selector", "value"),
        Input("fr-slider", "value"),
        Input("tau1-slider", "value"),
        Input("tau2-slider", "value"),
        Input("pump-wl-slider", "value"),
    )
    def _update_output(layer, material_name, fr_val, tau1_val, tau2_val, pump_wl_nm):
        """Main callback: render the active layer visualization."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from io import BytesIO
        import base64

        spec = RamanSpec.from_database(material_name)
        pump_wl = Wavelength(pump_wl_nm, "nm")

        # Build RamanResponse with user overrides
        resp = RamanResponse(
            spec=spec, fR=fr_val, tau1=tau1_val * 1e-15, tau2=tau2_val * 1e-15
        )

        summary_lines = [spec.summary()]
        img_html = ""

        if layer == "layer-1-material":
            # Layer 1: Material spectrum + phonons
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            # Spectrum
            shift = np.linspace(-200, 800, 1000)
            center = spec.raman_shift_cm or 0
            width = (spec.raman_linewidth_cm or 1) / 2
            intensity = (
                (width / np.pi) / ((shift - center) ** 2 + width**2)
                if width > 0
                else np.zeros_like(shift)
            )
            intensity /= np.max(intensity) if np.max(intensity) > 0 else 1
            ax1.plot(shift, intensity, linewidth=2, color="#00d4ff")
            ax1.axvline(x=center, color="r", linestyle="--", alpha=0.5)
            ax1.set_xlabel("Raman shift (cm⁻¹)")
            ax1.set_ylabel("Intensity (arb.)")
            ax1.set_title(f"Raman Spectrum: {spec.name}")
            ax1.grid(True, alpha=0.3)
            # Phonons
            if spec.lo_phonon_cm:
                ax2.barh(["LO"], [spec.lo_phonon_cm], color="blue", alpha=0.7)
            if spec.to_phonon_cm:
                ax2.barh(["TO"], [spec.to_phonon_cm], color="red", alpha=0.7)
            ax2.set_xlabel("Wavenumber (cm⁻¹)")
            ax2.set_title("Phonon Modes")
            plt.tight_layout()
            summary_lines.append(f"Q factor: {spec.quality_factor:.1f}")
            summary_lines.append(
                f"Stokes @ {pump_wl_nm}nm: {spec.stokes_wavelength(pump_wl).as_nm:.1f} nm"
            )
            summary_lines.append(
                f"Anti-Stokes @ {pump_wl_nm}nm: {spec.anti_stokes_wavelength(pump_wl).as_nm:.1f} nm"
            )

        elif layer == "layer-2-response":
            # Layer 2: Time-domain response
            fig = resp.plot_components(backend="matplotlib", figsize=(12, 8))
            summary_lines.append(f"fR = {fr_val:.2f}")
            summary_lines.append(f"τ1 = {tau1_val:.2f} fs, τ2 = {tau2_val:.2f} fs")
            summary_lines.append(
                f"h_R(0) = {resp.delayed_response(np.array([0.0]))[0]:.2e}"
            )

        elif layer == "layer-3-frequency":
            # Layer 3: Frequency response
            freq_resp = RamanFrequencyResponse(response=resp)
            fig = freq_resp.plot_all(backend="matplotlib", figsize=(12, 9))
            summary_lines.append(
                f"Resonance: {freq_resp.resonance_frequency_THz:.2f} THz"
            )
            summary_lines.append(f"FWHM: {freq_resp.resonance_FWHM_THz:.2f} THz")
            summary_lines.append(f"Q = {freq_resp.quality_factor:.1f}")

        elif layer == "layer-4-pulse":
            # Layer 4: Pulse interaction
            from photonics_helper.pulse import Wave, Envelope
            from photonics_helper.base import Time

            grid = TemporalGrid(N=2**14, Tmax=20e-12)
            envelope = Envelope(
                shape="gaussian", peak_amplitude=1.0, pulse_width=Time(100, "fs")
            )
            wave = Wave(grid=grid, envelope=envelope, central_wavelength=pump_wl)
            interaction = RamanPulseInteraction(pulse=wave, response=resp, spec=spec)
            fig = interaction.plot_interaction(backend="matplotlib", figsize=(12, 10))
            summary_lines.append(
                f"P_NL max: {np.max(np.abs(interaction.nonlinear_polarization)):.2e}"
            )
            summary_lines.append(f"n₂ = {spec.n2 or 'N/A'} m²/W")

        elif layer == "layer-5-pump":
            # Layer 5: Pump wavelength
            explorer = PumpWavelengthExplorer(spec=spec)
            fig = explorer.plot_both(pump_wl, backend="matplotlib", figsize=(14, 5))
            stokes = explorer.pump_to_stokes(pump_wl)
            anti = explorer.pump_to_anti_stokes(pump_wl)
            summary_lines.append(f"Pump: {pump_wl_nm} nm")
            summary_lines.append(
                f"Stokes: {stokes.as_nm:.1f} nm (Δλ = {stokes.as_nm - pump_wl_nm:.1f} nm)"
            )
            summary_lines.append(
                f"Anti-Stokes: {anti.as_nm:.1f} nm (Δλ = {pump_wl_nm - anti.as_nm:.1f} nm)"
            )

        elif layer == "layer-6-compare":
            # Layer 6: Material comparison (current + Silica as default comparison)
            comp = MaterialComparison(
                materials=[spec, RamanSpec.from_database("Silica")]
            )
            grid = TemporalGrid(N=2**14, Tmax=10e-12)
            fig = comp.plot_all(backend="matplotlib", grid=grid, figsize=(12, 11))
            summary_lines.append(comp.comparison_table())

        else:
            fig, ax = plt.subplots()
            ax.text(
                0.5,
                0.5,
                "Select a layer",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            summary_lines.append("Select a layer tab to view visualization.")

        # Convert figure to base64 image
        buf = BytesIO()
        if hasattr(fig, "savefig"):
            fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")  # type: ignore[union-attr]
            plt.close(fig)  # type: ignore[arg-type]
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        img_html = html.Img(
            src=f"data:image/png;base64,{img_base64}", style={"width": "100%"}
        )

        return img_html, "\n".join(summary_lines)

    return dash_app


# ─── Public API ───────────────────────────────────────────────────────────────

__all__ = [
    "RamanSpec",
    "RamanDatabase",
    "RamanResponse",
    "RamanFrequencyResponse",
    "RamanPulseInteraction",
    "PumpWavelengthExplorer",
    "MaterialComparison",
    "COMMON_COMPARISONS",
    "RAMAN_MATERIALS",
    "THORLABS_SUBSTRATE_MATERIALS",
    "app",
]
