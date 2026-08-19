# Phase 2R common B distribution freeze

The region was selected before any Phase 2R dynamic second-grasp outcomes. Selection used only the three Phase 2.6 B-only-positive regions, matched-state geometry, and initial-overlap checks.

- Selected source region: `index_thumb_region`
- xyz bounds [m]: `{"x": [0.04535998713890449, 0.04735998713890449], "y": [0.08409904934597245, 0.08609904934597246], "z": [0.22399001120990503, 0.22599001120990503]}`
- yaw bounds [rad]: `[-0.1, 0.1]`
- B-only robustness evidence: 0.45
- Geometry access: `{"FINGERTIP": {"access_fraction": 0.19, "initial_A_overlap_pairs": 0, "initial_hand_overlap_pairs": 15052, "reachable_pairs": 3800, "state_placement_pairs": 20000, "states_with_any_access": 19}, "PALMAR_SECURED": {"access_fraction": 0.83585, "initial_A_overlap_pairs": 0, "initial_hand_overlap_pairs": 0, "reachable_pairs": 16717, "state_placement_pairs": 20000, "states_with_any_access": 86}}`
- Fixture: kinematic free-joint pose support until scripted release; final hold unsupported
- Seed namespaces: geometry 20260820, calibration 20260821, formal 20260822
- Config hash: `490f901aadf0497ed1661466a4e0c67db045c228050a74808829bcc0e64dc436`
- Git SHA at freeze: `ef66d26f964ad2f4ca5d85bdd1e9923338a5b7bb`

The exact same frozen distribution and seeds apply to both endpoint-state groups. It may not be changed based on dynamic outcomes.
