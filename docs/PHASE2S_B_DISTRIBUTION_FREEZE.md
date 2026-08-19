# Phase 2S common B distribution freeze

The common region was selected before any Phase 2S formal dynamic outcome. Selection used only strict half-scale B-only evidence, matched-state geometry, and initial-overlap checks.

- Selected local region: `half_scale_region_08`
- Source strict B-only profile: `phase2S_b_only_08.yaml`
- xyz bounds [m]: `{"x": [0.1032456153804166, 0.1052456153804166], "y": [0.06388611422793093, 0.06588611422793093], "z": [0.17787366839456856, 0.17987366839456856]}`
- yaw bounds [rad]: `[-0.1, 0.1]`
- B-only evidence: 15/21 (0.7142857142857143)
- Geometry access: `{"FINGERTIP": {"access_fraction": 1.0, "initial_A_overlap_pairs": 0, "initial_hand_overlap_pairs": 0, "reachable_pairs": 20000, "state_placement_pairs": 20000, "states_with_any_access": 100}, "PALMAR_SECURED": {"access_fraction": 0.9461, "initial_A_overlap_pairs": 0, "initial_hand_overlap_pairs": 0, "reachable_pairs": 18922, "state_placement_pairs": 20000, "states_with_any_access": 95}}`
- Fixture: kinematic free-joint pose support until scripted release; final 500-step hold unsupported
- Seed namespaces: geometry 20260902, calibration 20260903, formal 20260904
- Config hash: `e6d88c23bb9e4775011594fef032622d4a7195ffad1f1cc5257c03c863971b6d`
- Git SHA at freeze: `d6c20896cd355fa2430017e7e116e597f4d628bb`

The exact same frozen distribution and seeds apply to both endpoint groups and may not be moved based on formal outcomes.
