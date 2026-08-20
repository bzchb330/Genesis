# Phase 2W static wrist results

## Outcome

Phase 2W stopped at `PHASE2W_NO_STATIC_WRIST_B_CONTROL`. Static orientation changed the collision geometry enough to create common index+thumb opposition regions, but none of the ten pre-outcome-selected wrist/B candidates produced the required strict B-only positive control in 8,192 dynamic candidates. No wrist or B region was frozen, and no A+B calibration or formal comparison was run.

Dynamic wrist reorientation was **not** simulated. Candidate endpoint states were initialized directly at each static rigid root orientation while preserving the finger configuration and object-A pose relative to the palm, then revalidated under unchanged world-fixed gravity. This experiment tests only whether a static post-reorientation endpoint removes the Phase 2T-R geometric blockage.

## Wrist search and endpoint stability

- Coarse grid: 125 Euler combinations deduplicated to 93 physical orientations; 18 passed both groups at >=10/20.
- Refined grid around the five pre-outcome-ranked coarse poses: 114 deduplicated orientations; 60 passed.
- Highest-ranked failed candidate (diagnostic, not frozen): relative RPY `[-22.5, 22.5, -22.5]` degrees; quaternion wxyz `[0.9508815466352549, -0.15033622173376132, 0.22499405578410392, -0.15033622173376132]`.
- Palm normal world: `[0.8535533905932741, -0.3535533905932738, -0.38268343236508984]`.
- Gravity in palm frame: `[3.7541244715015316, 3.4683587617200162, -8.373358761720018]`; palm-normal/gravity angle 67.500 degrees.
- At that candidate, FINGERTIP survival was 16/20 and PALMAR survival was 20/20.

## Transformed index/thumb workspace and common B geometry

- FINGERTIP index/thumb volumes: 1.8125e-05 / 1.7875e-05 m^3.
- PALMAR index/thumb volumes: 2.2375e-05 / 2.2e-05 m^3.
- Opposition-region volume at the highest-ranked failed candidate: 1.1e-05 m^3.
- Common access fractions: FINGERTIP 0.560000; PALMAR 0.490000.
- Initial hand/A overlap fractions: FINGERTIP 0.000000/0.000000; PALMAR 0.000000/0.000000.
- Final geometry mapping evaluated 390,000 B poses across 78 endpoint-eligible wrist orientations. All had nonzero collision-free opposition candidates; the two-point Ferrari-Canny approximation remained zero and was retained as a limitation rather than misused as the explicit geometric stop condition.

## B-only dynamic control

- Ten wrist/B candidates were selected without A+B outcomes.
- Total candidates: 8,192; strict successes: 0.
- The two highest-ranked candidates were expanded to 2,048 each; the other eight used 512 each.
- Failure mechanisms: `{"B_ROTATED_OUT": 2463, "B_SLIPPED_TO_TABLE": 2372, "CONTACT_LOST_IMMEDIATELY_AFTER_RELEASE": 384, "NO_B_CONTACT_BEFORE_RELEASE": 2869, "OTHER": 104}`.
- Geometry-centered proposals established both index and thumb before release in 1492/2734 F-centered and 1772/2724 P-centered trials, but none survived the unchanged strict 500-step gate.
- Robustness was not run because no wrist pose reached three strict successes.

## Stopped stages

No wrist/B freeze, full-population replay, additional endpoint sampling, calibration split, A+B controller calibration, controller freeze, matching, formal trials, McNemar test, bootstrap, or representative videos were produced. No Phase 2U experiment, scalar J, transfer, dynamic wrist controller, finger gaiting, object C, or RL training was implemented.

## Exploratory palm-space diagnostics

At the highest-ranked failed candidate only (not a formal endpoint comparison):

- FINGERTIP: `{"A_load_distribution_N_mean": [0.0, 2.7530014954576223, 2.578901581951774, 0.0], "COM_to_palm_surface_distance_m_mean": 0.02529553710609715, "ferrari_canny_epsilon_mean": 0.060339487623988435, "free_palm_volume_m3_mean": 0.0033515234375000006, "largest_connected_free_palm_component_m3_mean": 0.0033515234375000006, "largest_inscribed_free_space_radius_m_mean": 0.03619545665030465, "minimum_object_to_palm_boundary_margin_m_mean": 0.030646489630669776, "occupied_palm_voxel_fraction_mean": 0.0648651123046875, "palm_contact_fraction_mean": 0.0, "state_count": 16}`
- PALMAR: `{"A_load_distribution_N_mean": [0.0, 3.1205635082282073, 2.0069626548798327, 0.0], "COM_to_palm_surface_distance_m_mean": -0.0007243328677710502, "ferrari_canny_epsilon_mean": 0.21641560385729358, "free_palm_volume_m3_mean": 0.0033513187500000007, "largest_connected_free_palm_component_m3_mean": 0.0033513187500000007, "largest_inscribed_free_space_radius_m_mean": 0.03956908882425127, "minimum_object_to_palm_boundary_margin_m_mean": 0.03721221502640892, "occupied_palm_voxel_fraction_mean": 0.06492222377232143, "palm_contact_fraction_mean": 1.0, "state_count": 20}`

The existing scalar free-palm volume remains nearly identical between groups, while signed COM-to-palm distance and occupied-palm spatial structure differ. These descriptors remain exploratory and were not used to tune wrist/B selection or define J.

## Limitations

Static feasibility does not establish dynamic wrist planning or control. Workspace access used collision-filtered Monte Carlo fingertip samples and a two-point force-closure approximation; dynamic B-only trials remained the decisive positive-control gate. The stopped design provides no PALMAR-versus-FINGERTIP sequential outcome estimate.
