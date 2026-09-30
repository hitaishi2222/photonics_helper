**dark-ODE overlay (Huang supplement S55/S63/S73/S81/S82/S93-S97) — v1.**

Source of the transcription: pdftotext of `huang_pcgnlse.pdf` p.15-21 +
200-dpi renders of p.17-20 (reads for the fraction blocks the text layer
mangles: S60/S64, S76-S80, S94-S100). Key visual-verified corrections over
the earlier vision pass:

- S60 ``(tA_1)`` = 16 P0^2 rho Bd^4/(5E) + 3 P0 Q^2 rho + 3 P0 Q^4/(2 E Bd^2).
- S62 ``eta_1`` = 4 P0^2 Bd^4 eta/(35 E rho^2)  (the glyph is eta, not Omega).
- S64 ``eta_2`` = Omega Q^2/Bd - rho Omega;
  ``eta_3`` = 4P0 Bd^2/(rho E) + 12P0 Bd^2 Q Omega/E + 3Q^2 Omega^2/(Bd^4 E)
          + 4P0 Bd^2 rho Omega^2/E + pi^2 Bd^2 C^2/(2 E rho) + 3P0 C^2 Q^2 I2/(rho^4 E).
- S76 ``A_2`` = P0 pi^2 Bd Q C/rho + P0 pi^2 Bd^2 Omega C/E
            + 6 P0 C Omega Q^2 I_tau/(rho^3 E).
- S79 ``rho_1`` = 2P0^2 Bd^4 rho^3?NO: 2P0^2 Bd^4 rho/(35 E rho^2)... i.e.
  2P0^2 Bd^4 rho^3 is WRONG; printed numerator is 2P0^2 Bd^4 rho over
  35 E rho^2, plus (P0^2 Bd^2 rho/(rho E)) A_3 + (P0^2/(2 rho E)) A_2.
- S80 ``A_3`` = 7 pi Bd^2/120 - pi^2/2 + 9 pi/20 + (pi - pi^2)/2 Q^2.
- S87 ``I_1`` = (s_D P0/rho) [3 - 2Bd^2 + (4Bd^2-3) R/(Bd Q)]
             + 4 s_D Omega [R - Bd Q].
- S88 ``I_2`` = s_D P0 rho [ (2/3)Bd^2 - 1/2 - (2Bd^2-1) R/(2 Bd (1-Bd^2)^{1/2}) ]
  -- NOTE the leading rho vs 1/rho ambiguity; S89 multiplies (s_D P0/E)(I1+I2)
  so the (de)normalization is testable. Implemented with rho (visual read).
- S89 ``E dCtot/dxi|sD`` = (s_D P0/E)(I1+I2) + s_D Omega^2 + s_D pi^2 C^2/(12 rho^2).
- S90 ``C_2`` = -C^2 A_4/rho^3 + 9 Omega C^2/rho^2 + 36 E Omega^3/pi^2
             + 108 Omega^2 A_5/(pi^2 rho^2 Bd^2) + 12 Omega A_6/(pi^2 rho^2)
             + 12 A_7/(pi^2 rho^3).
- S91 A_4 = 3R - 12R^3/pi^2 - 9 Bd Q;  A_5 = R - Bd Q;
       A_6 = 9/Bd^2 - 6 - 3(3-4Bd^2) R/(Bd^3 Q).
- S92 A_7 = 3(7-7Bd^2-2Bd^4+Bd^6)/(4 Bd Q)
          - (6-13Bd^2+9Bd^4+Bd^8)/(4 Bd^2 Q^2) - 3 Q R^2/Bd^3.
- S93 ``C_3`` = 20 P0 Bd sqrt(1-Bd^2)/(pi^2 rho).
- S94 ``C_4`` = -(12 P0/pi^2) [rho + (2 - rho/3)/(Bd^2 - 1)].
- S96 ``C_5`` = (4/pi^2)[2P0(3-Bd^2) Omega - 3P0 Omega/rho];
  ``A_8`` = -4P0 Bd^2 C/(pi^2 rho) - 4P0 Bd^2/5.
- S97 ``dC/dxi`` = -(C/E) dE/dxi - (M/E) deta/dxi
                + s_D C1 + delta C2 + sigma C3 + |s_gamma| C4 + |sigma| C5 + A_8
  with C1 = (12 P0/(pi^2 E))(A_9 + A_10) + 12 Omega^2/(pi^2 E) + C^2/rho^2,
       A_9  = (1/rho)(3 - 2Bd^2 + (4Bd^2-3) R/(Bd Q)) + 4 Omega (R - Bd Q),
       A_10 = (2/3) Bd^2 - 1/2 - (2Bd^2-1) R/(2 Bd Q).

KNOWN PRINTED-TYPO FAMILY (arbitrated numerically against the direct pc-PDE
decks, see ``overlay_report()`` output):

- S73 box prints ``dM/dxi = sigma M1 + |s_gamma| tauR M2 + |sigma| tauR M3``
  but its own derivation S69-S72 pairs pure-Raman(|s_gamma|tau_R)->M1,
  pure-SS(sigma)->M2, SS-Raman(|sigma|tau_R)->M3. Both pairings are
  implemented (``moment_pairing='printed'|'derived'``).
- S63 collects the GVD contribution as +s_D eta_1 while the printed S57
  derivation gives +s_D Omega for that term. Etas checked per-deck.

Also ill-defined in the printed supplement: A_2's ``I_tau`` (a "cutoff
integral over a finite region") — irrelevant for the sigma=0 decks (the term
enters only via delta A_2 = O(delta) and sigma-needing rho_1); recorded.

## Post-script: Table-II P0 correction (same session)

The printed S63/S73/S81/S97 machinery consumes the background power P0.
The paper's own Table II (p-07) fixes E0 = 1 (I/II: rho0 = 1; III: rho0 = 2)
with Bd = 0.9, so S48 (E = 2 P0 Bd² rho) REQUIRES P0 = 0.6173 (I/II) or
0.3086 (III) — the v1 decks' hardcoded P0 = 1.0 violated the energy
constraint outright. Any future revisitation must run the overlay on the
corrected decks (now in parameters.json / run_case); note the instability
persists at reduced scale, so the paper's ±200-window / adaptive-dz setup
remains the suspected missing ingredient (caveats 2-3 family), not an
equation error.

Fig. 4 (dark Case I) qualitative anchors transcribed from p-08 for the
overlay: E flat at 1.0; η → ~ −100 over ξ = 10; M = M_core constant at
Ω(0) = 0; Ω̃ monotonically redshifts to ~ −1.5; ρ constant; chirp opposite
the delay with a rapid rise near ξ ≈ 1 attributed to GVD (their text).
