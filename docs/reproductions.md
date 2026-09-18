# Reproductions

Self-contained, validated reproductions of results from the literature. Each
folder under `reproductions/` contains a `parameters.json`, a `reproduce.py`
that asserts its result against an analytic/closed-form reference, and a
`README.md` with the ground truth and outcome. Run them all with:

```bash
python -m pytest tests/test_reproductions.py
```

| Reproduction | Reference | Stack |
|---|---|---|
| `stolen_lin_1978_spm` | Stolen & Lin, *Phys. Rev. A* **17**, 1448 (1978) | GNLSE (Kerr) + pulse |
| `macleod_quarter_wave_dbr` | Macleod, *Thin-Film Optical Filters* | `dbr` TMM |
| `gordon_1986_ssfs` | Gordon, *Opt. Lett.* **11**, 662 (1986) | GNLSE + Raman + soliton |
| `dudley_2006_cherenkov_dw` | Akhmediev & Karlsson, *Phys. Rev. A* **51**, 2602 (1995) | phase_matching + GNLSE |
| `dudley_2006_scg` | Dudley, Genty & Coen, *Rev. Mod. Phys.* **78**, 1135 (2006) | GNLSE + Raman + phase_matching + soliton |
| `kuznetsov_ma_2012_breather` | Kibler et al., *Sci. Rep.* **2**, 463 (2012) | GNLSE (NLSE) + phase_matching |
| `narhi_2016_mi_breathers` | Närhi et al., *Nat. Commun.* **7**, 13675 (2016) | GNLSE + phase_matching |
| `tomlinson_1985_wave_breaking` | Tomlinson, Stolen & Johnson, *Opt. Lett.* **10**, 457 (1985) | GNLSE (NLSE) |
| `shg_textbook` | Boyd, *Nonlinear Optics*; Fejer et al., *IEEE JQE* **28**, 2631 (1992) | `chi2` (RK4IP) |

Reproducing the papers also surfaced real bugs (inverted GNLSE dispersion sign,
Raman response direction); see the repository `REPORT.md` for the full list.
