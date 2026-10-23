---
quadrant: reference
---

# Examples

Every runnable script in `examples/`, 48 in total, grouped by what they demonstrate. Each entry shows the figure it produces and its source.

Run one with `python examples/<name>.py`. They write into `examples/images/` and share no state.

## Pulses, units and dispersion

- [Pulse Envelope Visualization](examples/01_pulse_visualization.md). 1 figure(s).
- [Light vs Dark Theme Comparison](examples/02_pulse_theme_comparison.md). 2 figure(s).
- [Pulse Train Visualization](examples/03_pulse_train.md). 3 figure(s).
- [Physics-backed verification examples for the pulse module](examples/16_pulse_verification_examples.md). 1 figure(s).
- [Group Dispersion (n_g, v_g)](examples/17_group_dispersion.md). 1 figure(s).
- [Build on the core: group-velocity dispersion from a material you define](examples/33_build_on_core.md).
- [Interactive Pulse Visualization Dashboard](examples/pulse_visualization.md).

## Raman response and material data

- [Raman Response Function (Layer 2)](examples/05_raman_response.md). 3 figure(s).
- [Raman Frequency Response (Layer 3)](examples/06_raman_frequency_response.md). 3 figure(s).
- [Pulse-Raman Interaction (Layer 4)](examples/07_raman_pulse_interaction.md). 4 figure(s).
- [Pump Wavelength Explorer (Layer 5)](examples/08_raman_pump_wavelength_explorer.md). 6 figure(s).
- [Material Comparison (Layer 6)](examples/09_raman_material_comparison.md). 4 figure(s).
- [Material Catalog Explorer](examples/11_raman_material_catalog.md). 1 figure(s).

## GNLSE propagation

- [GNLSE Basic Propagation](examples/12_gnlse_basic_propagation.md). 2 figure(s).
- [GNLSE Soliton Evolution](examples/13_gnlse_soliton_evolution.md). 2 figure(s).
- [GNLSE Raman Scattering and Self-Steepening](examples/14_gnlse_raman_and_steepening.md). 2 figure(s).
- [GNLSE Two-Photon Absorption and Visualization](examples/15_gnlse_tpa_and_visualization.md). 5 figure(s).
- [NLSE_simple (photonics_helper)](examples/20_gnlse_nlse_simple.md). 1 figure(s).
- [NLSE_dudley (photonics_helper)](examples/21_gnlse_nlse_dudley.md). 1 figure(s).
- [GNLSE Beta-Coefficient Units (`betas_unit`)](examples/23_gnlse_beta_units.md). 1 figure(s).
- [Modulation-Instability Gain (Corrected Convention)](examples/26_mi_gain_convention.md). 1 figure(s).
- [GNLSE spectrogram (cross-correlation FROG)](examples/27_gnlse_spectrogram.md). 1 figure(s).
- [Optical Wave Breaking (normal dispersion)](examples/34_wave_breaking_theory.md). 2 figure(s).
- [the photon-conserving shock, and the self-steepening grid guard](examples/39_conserving_shock_self_steepening.md). 2 figure(s).
- [the MI gain and its dispersion-offset convention](examples/40_mi_gain_offset_convention.md). 2 figure(s).
- [GNLSE solver scheme comparison (strang / yoshida4 / etdrk4 / rk4ip)](examples/45_gnlse_scheme_comparison.md). 1 figure(s).

## Solitons and breathers

- [Soliton analysis example: soliton fission in chalcogenide waveguide](examples/18_soliton_analysis_example.md).
- [The Soliton-on-Finite-Background (Breather) Family](examples/30_breather_families.md). 3 figure(s).

## Few-mode, polarization and structured light

- [Structured light: Laguerre-Gaussian / OAM modes](examples/28_structured_light.md). 2 figure(s).
- [Waveguide mode import from an external FEM solver (``WaveguideMode``)](examples/29_waveguide_mode_import.md). 1 figure(s).
- [Import *real* waveguide mode data from femwell and Tidy3D](examples/31_waveguide_mode_import_femwell_tidy3d.md). 1 figure(s).
- [the vector / polarisation GNLSE: three couplings, one flag](examples/37_vector_polarization_gnlse.md). 2 figure(s).
- [building a few-mode / multimode GNLSE one parameter at a time](examples/38_multimode_fewmode_gnlse.md). 2 figure(s).
- [Generate a synthetic N-mode export for the step7 bridge](examples/generate_multimode_mode_data.md).
- [Generate real waveguide ``n_eff(λ)`` export data with femwell and Tidy3D](examples/generate_waveguide_mode_data.md).

## Devices: FROG, TMM, phase matching, inverse design

- [FROG (Frequency-Resolved Optical Gating)](examples/04_frog.md).
- [Phase-matching diagnostics example](examples/22_phase_matching_diagnostics.md). 4 figure(s).
- [TMM Configurable Media & Continuous Field Profile](examples/24_tmm_media_and_field_profile.md). 1 figure(s).
- [FROG Retrieval of Chirped Pulses (Robust PCGPA)](examples/25_frog_chirped_retrieval.md). 1 figure(s).
- [QPM design and two-wave inverse fit](examples/42_inverse_design_qpm_fwm.md). 2 figure(s).

## Tapers, validation, noise and cascades

- [GNLSE propagation in a tapered waveguide: femwell-dispersion driven](examples/19_gnlse_tapered_waveguide.md). 1 figure(s).
- [verifying the GNLSE solver against closed forms and against itself](examples/35_gnlse_validation_harness.md). 2 figure(s).
- [ASE and Raman noise](examples/36_noise_ase_raman.md). 2 figure(s).
- [multi-phonon Raman: when the modal sum beats the two exponentials](examples/41_multi_phonon_raman.md). 2 figure(s).
- [querying the material catalog](examples/43_material_catalog_query.md). 2 figure(s).
- [RIN transfer in a dual-order CW Raman cascade](examples/44_cw_cascade_rin_transfer.md).

## Interactive dashboards

- [Dash Interactive App (Layer 7)](examples/10_raman_dash_app.md).
- [Unified Dashboard (Raman Explorer + GNLSE Viewer)](examples/32_unified_dashboard.md).
