# Multimode (few-mode) coupled GNLSE

`photonics_helper.multimode_gnlse` propagates *N* simultaneously guided
spatial modes as coupled modal envelopes — the nonlinear complement of
[`structured.py`](api/structured.md) (details and references there):

- per-mode Taylor dispersion (`betas[fiber_mode]`), modal group delay
  (`group_delays`, s/m — walk-off in the retarded frame of channel 0), and
  shared scalar loss;
- SPM/XPM coefficient sets: co-polarized scalar spatial modes
  (`coef_model="co_polarized"` — SPM 1, XPM 2, FWM 1; Mumtaz Eq. 6 scalar
  limit), the two-polarization degenerate LP pair (`"lp_degenerate"` — the
  vector-GNLSE strip-down: SPM 1, XPM 2/3, FWM 2/3), or the isotropic
  all-ones model (`"isotropic"`). `lp_degenerate` is **not** the XPM of two
  co-polarized spatial modes; pass `coef_model="co_polarized"` or the
  mode-analysis overlap weights for a few-mode LP deck;
- `gamma_reference` (W⁻¹m⁻¹) overrides the fiber-derived scalar γ for every
  tensor-normalised term. The mode-analysis deck returns
  `gamma_reference = γ₀`; use it whenever `fwm_weights`/`raman_tensors` came
  from `S⁽²⁾/S_0000`, otherwise a `Fiber.A_eff` different from `A_eff,0`
  mis-scales the tensor arms; the absolute `coupling_tensor` diagonal is
  exactly the per-mode `γ_m` (`C_ii = γ_i`), not a `½(γ_i+γ_j)` average;
- opt-in **pump-driven inter-modal FWM** (`include_fwm=True`): each pump
  channel `n` exchanges the pair `(m, q)` through two pump photons —
  conjugate (Hamiltonian) partners, energy conserving per pair — gated by
  the angular-momentum rule `ℓ_m = 2ℓ_n − ℓ_q` when OAM indices are
  supplied (`oam_l`);
- contracts (regression-tested, `tests/test_multimode_gnlse.py`):
  single-channel = scalar engine (machine precision), two-channel
  LP-degenerate = `vector_gnlse.VectorSplitStepEngine` (machine
  precision), mode walk-off = `Δβ₁·L`, FWM vs an independent dense RK4 to
  <5%, forbidden FWM triplets unmixed to machine precision.

Scope (v1): fixed-step propagation; Raman is the scalar per-channel
response (`include_raman=True` + `raman_response`, N=1 == scalar engine)
with an opt-in intermodal delayed-coupling tensor (`raman_tensors`,
Wright JSTQE Eq. 1 kernel, step7 `S⁽²⁾` admissible as `T_R`); pump
depletion is implemented (`fwm_pump_depletion=True`, Manley–Rowe
consistent); self-steepening + time-resolved TPA/FCA are implemented
per-channel (`include_shock`, `include_tpa`/`include_free_carriers`);
full-β(ω) dispersion (`full_beta`) and the full-`S⁽²⁾` FWM contraction
(`fwm_full_tensor=True`) are implemented. All four GAP steps have landed —
no switched-off-physics disclaimer.
