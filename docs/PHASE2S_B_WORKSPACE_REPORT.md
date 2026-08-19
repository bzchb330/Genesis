# Phase 2S half-scale B workspace and positive controls

The unchanged Allegro kinematic/self-collision workspace cloud was reused as an algorithm input. Every B-dependent envelope, surface-access test, opposition test, Ferrari–Canny calculation, collision check, placement, and dynamic trajectory was recomputed using the compiled half-scale cylinder (radius 0.0125 m, half-height 0.0200 m).

- Geometry candidates: 8,192
- Initially valid candidates: 6,274
- Candidates with multi-finger surface access: 5,055
- Candidates with opposing-contact geometry: 1,705
- Candidates with positive Ferrari–Canny evidence: 973
- Diverse geometry proposals retained: 50
- Historical Phase 2.6 successes replayed under half-scale geometry: 3
- Historical successes passing half-scale revalidation: 0
- Fresh strict dynamic successes: 24 after 4,392 evaluated candidates (8,192 maximum)
- Strict-success access topologies: index+middle+ring+thumb 9, middle+ring+thumb 4, middle+ring 4, index+thumb 3, index+middle+thumb 2, ring+thumb 2

The first ten strict profiles were checked with 20 deterministic local perturbations each over ±0.001 m per axis and ±0.10 rad yaw. Perturbation successes by profile were 11, 18, 12, 15, 7, 6, 14, 14, 13, and 12; all 200 perturbations had zero initial-collision rate in the B-only scene. The demonstrated final-contact topologies included index+middle, index, middle, index+middle+thumb, index+thumb, middle+thumb, and ring+thumb.

The common-region selector used only these B-only results and matched-state geometry. It did not inspect calibration or formal dynamic outcomes. Region 08 maximized the smaller of the two group access fractions while retaining zero A overlap: FINGERTIP access 1.0000 and PALMAR_SECURED access 0.9461. Its B-only evidence was 15/21 including the strict nominal source. Exact bounds and provenance are frozen in `PHASE2S_B_DISTRIBUTION_FREEZE.md`.
