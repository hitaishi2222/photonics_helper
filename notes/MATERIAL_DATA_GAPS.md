# Material data gaps: what is missing, what the literature says, and what stays unknown

Audit date: 2026-11-02. Scope: the `raman_specs` table of `photonics_helper/materials.db`
(45 rows, mirrored by `RAMAN_MATERIALS` / `THORLABS_SUBSTRATE_MATERIALS` in
`photonics_helper/raman/reference.py`) plus the phonon mode table.

This follows the convention already used for Ge-Sb-Se in `DATABASE.md`: a value is only
listed as data when a primary measurement exists, the source carries a resolvable DOI, and
the measurement conditions (wavelength, pulse duration, method) are stated. Values with no
source stay `NULL` — an absent number is information, a fabricated one is not.

---

## 1. Gap audit

`NULL` = unknown / never measured. `0.0` = a placeholder that asserts "this material has no
Raman response" (the code treats `None` and `0.0` identically: `PhononResponse.fR` falls back
via `self.spec.fR or 0.0`, so the two are physically the same but only `NULL` is honest).

| Field | `NULL` | `0.0` placeholder | Materials |
|---|---|---|---|
| `n2` | 12 | — | BaF2, CaF2, F2, KBr, MgF2, N-BK7, N-F2, N-SF11, PMMA, YVO4, Zerodur, ZnSe |
| `fR` | 7 | 28 | **NULL:** AlGaAs, CdTe, Ga2O3, InGaAs, Si3N4, Si3N4-Ligentec, ZnO — **0.0:** Diamond, Si, Ge, SiC_4H, GaN, AlN, YAG, LiNbO3, LiTaO3, KTP, LBO, Al2O3, YLF, BaTiO3, AgGaS2, AgGaSe2, … |
| `raman_shift_cm` / `raman_linewidth_cm` | 12 | — | the 12 index-only substrates |
| `gain_coeff` | 40 | — | everything except Silica, Diamond, As2S3, ZBLAN, GeSbSe |
| `bandgap_eV` | 7 | — | F2, N-BK7, N-F2, N-SF11, PMMA, YVO4, Zerodur |
| `lo_phonon_cm` / `to_phonon_cm` | 35 | — | `phonon_modes` carries 59 rows for 11 of the 45 materials |

The 28 `fR = 0.0` rows are the largest silent gap: for diamond, Si, Ge, SiC, GaN, AlN, YAG and
the ferroelectrics there *is* a Raman response, it is just not quantified here.

### 1b. A non-`NULL` value is not necessarily a measured value

The NULL/`0.0` split above misses the rows that *do* hold a number, but a number the source
does not support outright. GeSbSe is the case in point — with Silica it is one of only two rows
that carry every scalar field, and the convention in this file is copied from its `DATABASE.md`
entry, yet it is not solid data either:

| Field | Stored | What the source actually supports |
|---|---|---|
| `fR` | 0.4615 | the published Kramers-Kronig value from the measured Raman spectrum — the one fR number for this glass that a source actually states. A bracket of 0.17–0.46 used to be advertised here from inverting `fR = g_R·Γ_R/(Ω_R·γ)`; that 0.170 is **withdrawn**, because the relation does not reproduce silica (0.049 against a known 0.18) and the two conventions at stake move the answer by exactly the factor 2.7 that separates the two ends. 0.4615 stands, with an unquantified uncertainty |
| `n2` | 8.7e-18 | z-scan of *neighbouring* Ge-Sb-Se compositions (Ge15Sb15Se70 / Ge15Sb10Se75), bracket 7.4–10.0e-18; the 2 µm number is an extrapolation |
| `raman_shift_cm` / `raman_linewidth_cm` | 198.1 / 9.33 | only the dominant band of a 7-mode spectrum, so they are not a single-Lorentzian description (the `phonon_modes` list is) |

Sources: Wu, Ding & Peng, *Appl. Sci.* **13**, 12923 (2023),
[10.3390/app132312923](https://doi.org/10.3390/app132312923); Wang et al., *Opt. Mater.
Express* **4**, 1011 (2014), [10.1364/OME.4.001011](https://doi.org/10.1364/OME.4.001011);
Peng et al., *Appl. Phys. B* **123**, 261 (2017),
[10.1007/s00340-017-6832-6](https://doi.org/10.1007/s00340-017-6832-6).

The lesson generalises: a column value is only as strong as its weakest sourced input, and a
bracket recovered by an inversion is evidence only once the inversion has been validated
against a case whose answer is known. That check is what 0.4615 passes and 0.170 failed.
It also applies to any value in §2a below where a range is quoted.

---

## 2. Non-linear refractive index n2 — 7 of 12 rows fillable

All values in `m²/W` at 1.064 µm unless stated. esu values are converted with the relation the
source papers themselves use (Milam, Weber & Glass 1977, footnote to Table I):

> `n2[m²/W] = (40π/c) · n2[esu] / n`  (equivalently `n2[esu] = 2.39×10⁶ · n · n2[m²/W]`)

which reproduces every tabulated n₂ value checked here (CaF2, BaF2, MgF2, KBr, ZnO, Ga2O3) to
2 decimal places.

### 2a. Measured — safe to seed

| Material | n2 (m²/W) | λ (µm) | Method | Source (DOI) |
|---|---|---|---|---|
| CaF2 | **1.26e-20** (range 1.26–1.90e-20) | 1.064 | TWM / TRI | Adair, Chase & Payne, *Phys. Rev. B* **39**, 3337 (1989), [10.1103/PhysRevB.39.3337](https://doi.org/10.1103/PhysRevB.39.3337); Milam, Weber & Glass, *Appl. Phys. Lett.* **31**, 822 (1977), [10.1063/1.89561](https://doi.org/10.1063/1.89561) |
| BaF2 | **1.91e-20** (range 1.91–2.85e-20) | 1.064 | TWM / TRI | same two papers |
| MgF2 (o) | **7.6e-21** (range 7.6–9.2e-21) | 1.064 | TWM / TRI | same two papers |
| KBr | **8.0e-20** | 1.064 | TWM (2.93×10⁻¹³ esu, n=1.5435) | Adair et al. 1989, op. cit. |
| ZnSe | **1.50e-18** @2 µm, **1.20e-18** @3.9 µm | 2 / 3.9 | Z-scan, 100–260 fs | Ensley & Bambha, *Opt. Express* **27**, 37940 (2019), [10.1364/OE.27.037940](https://doi.org/10.1364/OE.27.037940); Werner et al., *Opt. Express* **27**, 2867 (2019), [10.1364/OE.27.002867](https://doi.org/10.1364/OE.27.002867) |
| YVO4 (o) | **1.56e-19** | 1.030 | SPM phase regression, 210 fs | Kabaciński et al., *Opt. Express* **27**, 11018 (2019), [10.1364/OE.27.011018](https://doi.org/10.1364/OE.27.011018) |
| N-BK7 | **3.4e-20** | 1.064 | TWM, 3 ns | Adair, Chase & Payne, *J. Opt. Soc. Am. B* **4**, 875 (1987), [10.1364/JOSAB.4.000875](https://doi.org/10.1364/JOSAB.4.000875) |

N-BK7 cross-checks (the spread is real, so quote the method): 3.62e-20 (TWM, 1.064 µm, Adair
1987); 3.30e-20 @0.80 µm and 3.50e-20 @2.0 µm (Z-scan, 100 fs, Ensley & Bambha 2019); 3.33e-20
@0.804 µm (common-path interferometry, 47 fs, Van der Linden et al., *Opt. Express* **33**,
53268 (2025), [10.1364/OE.579736](https://doi.org/10.1364/OE.579736)); 2.24e-20 @1.55 µm and
1.47e-20 @0.772 µm (Z-scan, Flom et al., *Appl. Opt.* **54**, F123 (2015),
[10.1364/AO.54.00F123](https://doi.org/10.1364/AO.54.00F123)). Flom's visible-wavelength
values sit a factor ~2 below the others; do not average across methods.

### 2b. No measurement exists — do not invent one

The five remaining rows (F2, N-F2, N-SF11, Zerodur, PMMA) have no measured n₂ in any source
consulted. They stay `NULL`.

If a number is ever required for them, the only defensible route is Boling, Glass & Owyoung's
single-oscillator fit (their Eq. 38, K = 68), evaluated from the manufacturer's `n_d` and Abbe
number for that glass:

```
n2(10⁻¹³ esu) = 68 (n_d − 1)(n_d² + 2)² / { ν_d [1.517 + (n_d²+2)(n_d+1)ν_d / 6n_d]^1/2 }
```

*Boling, Glass & Owyoung, IEEE J. Quantum Electron.* **14**, 601 (1978),
[10.1109/JQE.1978.1069847](https://doi.org/10.1109/JQE.1978.1069847)*; also Eq. (38) of
*Weber, Milam & Smith, Opt. Eng.* **17**, 463 (1978),
[10.1117/12.7972266](https://doi.org/10.1117/12.7972266)*.

Accuracy check: feeding the standard Schott `n_d`/`ν_d` of BK-7, SF-6 and fused silica into
the fit gives 3.17e-20, 2.30e-19 and 2.51e-20 m²/W against Weber's *measured* 3.43e-20,
2.12e-19 and 2.73e-20 m²/W — agreement to about ±8–10%, which is the accuracy Boling et al.
claim for oxide glasses. The fit is outside its validity for a polymer (PMMA) or a
glass-ceramic (Zerodur).

No per-glass numbers are given here: the `n_d`/`ν_d` pairs for the five remaining rows were
not taken from a primary manufacturer datasheet during this audit, and guessing them would
make the result look measured when it is not. If these rows are ever filled, record the
computed value as derived, together with the datasheet it came from.

---

## 3. Raman fraction fR — two measured values, the rest unknown

### 3a. Published

| Material | fR | Basis | Source (DOI) |
|---|---|---|---|
| **Si3N4**, **Si3N4-Ligentec** | **0.20** | fR = 20% used in the soliton model that reproduces the measured Raman self-frequency shift; inferred Raman shock time 2πτ_R ≈ 20 fs (τ_R ≈ 3.2 fs) for amorphous SiN | Karpov et al., *Phys. Rev. Lett.* **116**, 103902 (2016), [10.1103/PhysRevLett.116.103902](https://doi.org/10.1103/PhysRevLett.116.103902) |
| **N-BK7** | **0.22** | 1 − f_e with f_e = 0.78 (nuclear contribution from Raman-scattering cross-sections, Hellwarth/Heiman) | Weber, Milam & Smith 1978, Table 3, [10.1117/12.7972266](https://doi.org/10.1117/12.7972266) |

Weber's Table 3 in full (`fR = 1 − f_e`): fused silica 0.26, BK-7 0.22, SF-6 0.19, ED-4 0.19,
FK-5 0.21, FK-51 0.24, LHG-5 phosphate 0.39, B-101 fluoroberyllate 0.28, BeF2 0.21. Separate
Hellwarth-derived nuclear percentages quoted by Boling: LaSF-7 29%, SF-7 21%, ED-4 15%,
LSO 19%, fused silica 21%.

**No measurement exists for F2, N-F2 or N-SF11 themselves.** The flint family spans 0.19–0.22
across its measured members (SF-6 0.19, SF-7 0.21, BK-7 0.22), which is suggestive but is not
a measurement of those three glasses, so they stay `NULL` rather than inheriting a family
value. (N-BK7 is the exception: Weber's "borosilicate BK-7" has n_d = 1.517, the same
composition, so 0.22 does belong to the row.)

**Do not overwrite silica's 0.18 with 0.26.** They are different definitions: 0.18 is the
fraction used in the fibre GNLSE convention (Stolen et al., *J. Opt. Soc. Am. B* **6**, 1159
(1989), [10.1364/JOSAB.6.001159](https://doi.org/10.1364/JOSAB.6.001159); Lin & Agrawal,
*Opt. Lett.* **31**, 3086 (2006), [10.1364/OL.31.003086](https://doi.org/10.1364/OL.31.003086)),
while 0.26 is the *nuclear share of the total n2* including electrostriction. The same caveat
applies to every `1 − f_e` value above: they are an upper bound on the fR that a GNLSE wants.

### 3b. Not published — stays `NULL`

No primary source was found for **AlGaAs, ZnO, CdTe, InGaAs, Ga2O3**. These should become
`NULL` rather than `0.0` so that `PhononResponse` users see "unknown", not "no Raman". GaAs
carries fR = 0.07 in the table already; AlGaAs must not inherit that number without a source
of its own.

For the 28 `fR = 0.0` crystal rows (diamond, Si, Ge, SiC_4H, GaN, AlN, YAG, LiNbO3, LiTaO3,
KTP, LBO, Al2O3, YLF, BaTiO3, AgGaS2, AgGaSe2): no published fR was found either. If a value
is needed before a measurement exists, derive it and record the derivation — the relation
used for the semiconductor waveguides (valid in bulk with `γ = n2·ω0/c`) is

> `fR = g_R(ω0)·Γ_R / (Ω_R·γ)`

with `g_R` the peak Raman gain, `Ω_R`/`Γ_R` the shift and linewidth of the dominant mode, and
`γ = n2·ω0/c` for a bulk medium (used, e.g., in Sci. Rep. **8**, 17177 (2018),
[10.1038/s41598-018-34344-4](https://doi.org/10.1038/s41598-018-34344-4)). It needs a
measured `g_R`, which is exactly the column that is 40/45 empty — so `gain_coeff` is the
blocker, not fR.

---

## 4. Values already in the table that the literature contradicts

Found while checking the gaps; each is a separate defect worth fixing regardless of this report.

| Material | DB value | Literature | Source |
|---|---|---|---|
| **Ga2O3** | n2 = 4.0e-19 | 5.80×10⁻¹³ esu, n = 1.96 → **1.24e-19** (3.2× lower) | Adair et al. 1989, Table IV, [10.1103/PhysRevB.39.3337](https://doi.org/10.1103/PhysRevB.39.3337) |
| **ZnO** | n2 = **−9.0e-19** | measured **positive**: 4.92e-19 (e) / 5.26e-19 (o) | Adair et al. 1989, op. cit. |
| **Si** | n2 = 6.0e-19 | 4.84e-18 @1.55 µm, 2.71e-18 @1.03 µm (8× higher) | Bristow, Rotenberg & van Driel, *Appl. Phys. Lett.* **90**, 191104 (2007), [10.1063/1.2737359](https://doi.org/10.1063/1.2737359) |
| **CdTe** | n2 = −3.0e-17 | −1.34e-18 @1.064 µm (Sheik-Bahae et al., *IEEE J. QE* **27**, 1296 (1991), [10.1109/3.89946](https://doi.org/10.1109/3.89946)); +3.27e-17 @1.55 µm (Tatsuura et al., *Appl. Phys. Lett.* **87**, 251110 (2005), [10.1063/1.2151256](https://doi.org/10.1063/1.2151256)) — the sign flips across the two-photon edge, so the row needs the wavelength it belongs to |
| **Diamond** | gain_coeff = 0.11 | inconsistent with the rest of the column, which holds 1e-13 … 3e-11 m/W (Silica 1.0e-13, As2S3 4.3e-12, GeSbSe 2.9e-11); `RamanSpec` documents the field as m/GW while the stored values are m/W, so both the value and the unit need checking against a primary Raman-gain source | unit/scale defect, not a literature conflict |

---

## 5. What this does not fix

* `gain_coeff` (40 NULL) and the LO/TO phonon columns (35 NULL) are untouched: they are a
  much larger literature job than n2/fR and should not be approximated.
* `bandgap_eV` for the five glasses/PMMA/Zerodur is not a meaningful single number (they are
  amorphous wide-gap dielectrics); leaving it `NULL` is correct.
* No attempt was made to make `fR` wavelength-dependent; the published values above are all
  single-wavelength or spectrally averaged.

---

## 6. Ready-to-apply patch

`photonics_helper/raman/reference.py` (measured values only; nothing from §2b):

```python
# n2 (m^2/W) fills — measured
"CaF2":  {"n2": 1.26e-20},  # Adair 1989 doi:10.1103/PhysRevB.39.3337
"BaF2":  {"n2": 1.91e-20},  # Adair 1989; Milam 1977 doi:10.1063/1.89561 gives 2.85e-20
"MgF2":  {"n2": 7.6e-21},   # Adair 1989, ordinary ray
"KBr":   {"n2": 8.0e-20},   # Adair 1989
"ZnSe":  {"n2": 1.20e-18},  # Werner 2019 doi:10.1364/OE.27.002867, @3.9 um
"YVO4":  {"n2": 1.56e-19},  # Kabacinski 2019 doi:10.1364/OE.27.011018, o-ray @1.03 um
"N-BK7": {"n2": 3.4e-20},   # Adair 1987 doi:10.1364/JOSAB.4.000875
# fR fills
"Si3N4":          {"fR": 0.20},  # Karpov 2016 doi:10.1103/PhysRevLett.116.103902
"Si3N4-Ligentec": {"fR": 0.20},  # same
"N-BK7":          {"fR": 0.22},  # Weber 1978 doi:10.1117/12.7972266, 1 - f_e
```

`raman_specs` row count stays 46 and no golden test pins any of these numbers
(`tests/test_material_data_golden.py` pins only Silica/GeAsSe/GeSbSe/As2S3 `fR`), so
re-seeding is test-safe; the `0.0 → NULL` change for fR is *not* test-covered either but
should be done deliberately, since it changes what `PhononResponse` reports to users.
