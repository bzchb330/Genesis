# Updated evidence for a future PI decision about J

State descriptors are measurements of an endpoint. Candidate resource-metric components are descriptors the PI may later choose to include in J; Phase 2T does not make that choice.

## occupied_finger_count

- Physical meaning: load-bearing digits above 0.20 N.
- Units: count.
- Measured Phase 2T range/group effect: `{"FINGERTIP": {"maximum": 2, "mean": 2, "minimum": 2}, "PALMAR_SECURED": {"maximum": 2, "mean": 2, "minimum": 2}, "mean_difference_palmar_minus_fingertip": 0}`.
- Relationship with second-grasp success: not identifiable in Phase 2T because the pair-specific B-only gate failed before formal outcomes.
- Redundancy/correlation: descriptive overlap with the other endpoint components was retained for PI review; no component was collapsed or weighted.
- Sim-to-real measurability: digit contacts and palm contact require tactile/contact sensing; COM pose requires object tracking; workspace/volume requires a calibrated geometric model; Ferrari-Canny requires contact geometry and friction assumptions.

## free_finger_count

- Physical meaning: digits below the occupied threshold.
- Units: count.
- Measured Phase 2T range/group effect: `{"FINGERTIP": {"maximum": 2, "mean": 2, "minimum": 2}, "PALMAR_SECURED": {"maximum": 2, "mean": 2, "minimum": 2}, "mean_difference_palmar_minus_fingertip": 0}`.
- Relationship with second-grasp success: not identifiable in Phase 2T because the pair-specific B-only gate failed before formal outcomes.
- Redundancy/correlation: descriptive overlap with the other endpoint components was retained for PI review; no component was collapsed or weighted.
- Sim-to-real measurability: digit contacts and palm contact require tactile/contact sensing; COM pose requires object tracking; workspace/volume requires a calibrated geometric model; Ferrari-Canny requires contact geometry and friction assumptions.

## free_finger_workspace_vol_m3

- Physical meaning: collision-free sampled fingertip reachability.
- Units: m^3.
- Measured Phase 2T range/group effect: `{"FINGERTIP": {"count": 148, "maximum": 0.00032262500000000005, "mean": 0.000290191722972973, "median": 0.0002893125000000001, "minimum": 0.00027537500000000007, "standard_deviation": 6.94408571612837e-06}, "PALMAR_SECURED": {"count": 151, "maximum": 0.00035037500000000005, "mean": 0.00021496357615894042, "median": 0.00017937500000000004, "minimum": 0.000135875, "standard_deviation": 6.820502423667797e-05}, "mean_difference_palmar_minus_fingertip": -7.52281468140326e-05, "standardized_mean_difference": -1.5518141328870991}`.
- Relationship with second-grasp success: not identifiable in Phase 2T because the pair-specific B-only gate failed before formal outcomes.
- Redundancy/correlation: descriptive overlap with the other endpoint components was retained for PI review; no component was collapsed or weighted.
- Sim-to-real measurability: digit contacts and palm contact require tactile/contact sensing; COM pose requires object tracking; workspace/volume requires a calibrated geometric model; Ferrari-Canny requires contact geometry and friction assumptions.

## free_palm_volume_m3

- Physical meaning: unoccupied configured palm-frame voxel volume.
- Units: m^3.
- Measured Phase 2T range/group effect: `{"FINGERTIP": {"count": 148, "maximum": 0.0033327500000000006, "mean": 0.003331269425675676, "median": 0.0033312500000000004, "minimum": 0.0033295000000000004, "standard_deviation": 6.009333793450886e-07}, "PALMAR_SECURED": {"count": 151, "maximum": 0.0033338750000000005, "mean": 0.003331099337748345, "median": 0.0033311250000000008, "minimum": 0.0033272500000000003, "standard_deviation": 1.1565319819034952e-06}, "mean_difference_palmar_minus_fingertip": -1.7008792733107236e-07, "standardized_mean_difference": -0.1845575370411021}`.
- Relationship with second-grasp success: not identifiable in Phase 2T because the pair-specific B-only gate failed before formal outcomes.
- Redundancy/correlation: descriptive overlap with the other endpoint components was retained for PI review; no component was collapsed or weighted.
- Sim-to-real measurability: digit contacts and palm contact require tactile/contact sensing; COM pose requires object tracking; workspace/volume requires a calibrated geometric model; Ferrari-Canny requires contact geometry and friction assumptions.

## COM_to_palm_origin_distance_m

- Physical meaning: object COM location relative to palm origin.
- Units: m.
- Measured Phase 2T range/group effect: `{"FINGERTIP": {"count": 148, "maximum": 0.07057226225455089, "mean": 0.06833660049052977, "median": 0.0681192363207583, "minimum": 0.06562216085841603, "standard_deviation": 0.0013118197103186785}, "PALMAR_SECURED": {"count": 151, "maximum": 0.03198169296939009, "mean": 0.03039397799066982, "median": 0.03039351091699753, "minimum": 0.029006305775299304, "standard_deviation": 0.00045165435487023233}, "mean_difference_palmar_minus_fingertip": -0.03794262249985995, "standardized_mean_difference": -38.67608220743378}`.
- Relationship with second-grasp success: not identifiable in Phase 2T because the pair-specific B-only gate failed before formal outcomes.
- Redundancy/correlation: descriptive overlap with the other endpoint components was retained for PI review; no component was collapsed or weighted.
- Sim-to-real measurability: digit contacts and palm contact require tactile/contact sensing; COM pose requires object tracking; workspace/volume requires a calibrated geometric model; Ferrari-Canny requires contact geometry and friction assumptions.

## palm_A_contact_fraction

- Physical meaning: fraction of unsupported hold with real palm-A contact.
- Units: unitless.
- Measured Phase 2T range/group effect: `{"FINGERTIP": {"count": 148, "maximum": 0.0, "mean": 0.0, "median": 0.0, "minimum": 0.0, "standard_deviation": 0.0}, "PALMAR_SECURED": {"count": 151, "maximum": 1.0, "mean": 0.9974039735099338, "median": 1.0, "minimum": 0.98, "standard_deviation": 0.003715156000218603}, "mean_difference_palmar_minus_fingertip": 0.9974039735099338, "standardized_mean_difference": 379.6724084855565}`.
- Relationship with second-grasp success: not identifiable in Phase 2T because the pair-specific B-only gate failed before formal outcomes.
- Redundancy/correlation: descriptive overlap with the other endpoint components was retained for PI review; no component was collapsed or weighted.
- Sim-to-real measurability: digit contacts and palm contact require tactile/contact sensing; COM pose requires object tracking; workspace/volume requires a calibrated geometric model; Ferrari-Canny requires contact geometry and friction assumptions.

## ferrari_canny_epsilon

- Physical meaning: force-closure quality descriptor.
- Units: normalized epsilon.
- Measured Phase 2T range/group effect: `{"FINGERTIP": {"count": 148, "maximum": 0.18502567006187026, "mean": 0.14680275258301226, "median": 0.1488315662502332, "minimum": 0.04876162484780513, "standard_deviation": 0.023808746672177437}, "PALMAR_SECURED": {"count": 151, "maximum": 0.2531812596102272, "mean": 0.12415835311709718, "median": 0.11713511725855599, "minimum": 0.0003811358271476051, "standard_deviation": 0.0756372037801955}, "mean_difference_palmar_minus_fingertip": -0.02264439946591508, "standardized_mean_difference": -0.4038545314775473}`.
- Relationship with second-grasp success: not identifiable in Phase 2T because the pair-specific B-only gate failed before formal outcomes.
- Redundancy/correlation: descriptive overlap with the other endpoint components was retained for PI review; no component was collapsed or weighted.
- Sim-to-real measurability: digit contacts and palm contact require tactile/contact sensing; COM pose requires object tracking; workspace/volume requires a calibrated geometric model; Ferrari-Canny requires contact geometry and friction assumptions.

## TODO(PI)

Choose whether J should represent: (a) digit availability, (b) future reachable manipulation volume, (c) palm storage capacity, (d) a combination, and define functional form / normalization only after PI review.
