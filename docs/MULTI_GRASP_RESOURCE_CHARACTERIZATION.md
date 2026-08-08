# Multi-Grasp Resource Characterization

## Scope and reproducibility

The run used `configs/resource_probing_a.yaml`, unchanged scene physics, 20 seeds, and three source-distinct retained object-A profiles. Every selected profile retained the pre-existing finite engineering window in all 20 validation seeds. That statement is not a scientific success criterion. The selected files are:

- `resource_grasp_A_01`: source 01, local variant 015
- `resource_grasp_A_02`: source 02, local variant 022
- `resource_grasp_A_03`: source 03, local variant 013

Selection was source-stratified rather than simply taking the three highest values of one score. The run produced 3,000 raw resource rows, 384 seeded reachability samples, 240 independent single-finger object-B approach probes, and 12 figures.

## Contact geometry and penetration

Mean active contact locations in world xyz metres were: grasp 01 index (0.0787, 0.0465, 0.1700), middle (0.0484, 0.0007, 0.1607), and thumb (0.0694, 0.0462, 0.1551); grasp 02 index (0.0873, 0.0414, 0.1717), middle (0.0792, 0.0065, 0.1843), ring (0.0766, -0.0145, 0.1645), and thumb (0.0584, 0.0334, 0.1670); grasp 03 index (0.0469, 0.0486, 0.1824), middle (0.0580, -0.0016, 0.1793), and thumb (0.0802, 0.0440, 0.1528). These averages describe intermittent contacts and are not fixed contact targets.

Across original candidates 01/02/03/05, mean maximum penetration ranged from 0.00836 to 0.01888 m and mean steady-hold penetration from 0.00144 to 0.00657 m. Lower-penetration retained local variants were found for sources 02, 03, and 05; source 01 improved only marginally, and no variant eliminated penetration. Exact per-seed results and support-release timing are in `CONTACT_PENETRATION_ANALYSIS.md`.

## Object-A raw resource differences

| Grasp | Minimum joint margin [rad] | Mean actuator utilization [1] | Mean two-sided reserve [N m] | Mean active A fingers |
|---|---:|---:|---:|---:|
| 01 | 0.103454 | 0.171056 | 0.828944 | 2.159 |
| 02 | 0.160714 | 0.056136 | 0.943864 | 3.211 |
| 03 | 0.003547 | 0.123438 | 0.876562 | 2.998 |

Finger contact occupancy and mean normal force also differ materially:

| Grasp | Index | Middle | Ring | Thumb |
|---|---|---|---|---|
| 01 contact fraction / force [N] | 0.475 / 2.737 | 0.996 / 1.377 | 0.000 / 0.000 | 0.688 / 2.178 |
| 02 contact fraction / force [N] | 0.381 / 0.806 | 0.957 / 0.745 | 0.875 / 1.052 | 0.999 / 1.230 |
| 03 contact fraction / force [N] | 0.998 / 1.772 | 1.000 / 6.214 | 0.000 / 0.000 | 1.000 / 3.868 |

These profiles are physically distinct in joint occupancy, contact topology, force distribution, and control demand. No ranking or retained-resource label is assigned.

## Fingertip reachable workspaces

Palm-relative sampled axis extents varied by grasp and finger. Values below are x/y/z metres from 32 deterministic samples per cell:

| Grasp | Index | Middle | Ring | Thumb |
|---|---|---|---|---|
| 01 | 0.0225 / 0.0147 / 0.0305 | 0.0152 / 0.0165 / 0.0344 | 0.0138 / 0.0180 / 0.0349 | 0.0143 / 0.0348 / 0.0136 |
| 02 | 0.0180 / 0.0183 / 0.0372 | 0.0123 / 0.0169 / 0.0405 | 0.0142 / 0.0199 / 0.0392 | 0.0177 / 0.0335 / 0.0142 |
| 03 | 0.0124 / 0.0142 / 0.0301 | 0.0165 / 0.0154 / 0.0305 | 0.0137 / 0.0170 / 0.0328 | 0.0126 / 0.0263 / 0.0140 |

These bounded samples characterize local commanded motion; they are not a collision-free workspace proof or a gating rule.

## Independent finger approaches toward object B

Each grasp/finger pair was probed for all 20 object-B seeds. Only the chosen finger's configured joints moved toward the closest sampled reachability posture; no sequential policy, full B grasp, finger gate, or tactile controller was introduced.

No physical fingertip-B contact occurred in any of the 240 probes. The signed-distance test and a direct contact/separation unit test confirm the detector is operational. The observed minimum distances remained positive because the configured B placements were outside the sampled fingertip workspaces:

| Grasp | Minimum signed distance range [m] | Mean [m] | B contacts |
|---|---:|---:|---:|
| 01 | 0.107–0.166 | 0.136 | 0/80 |
| 02 | 0.120–0.170 | 0.146 | 0/80 |
| 03 | 0.111–0.155 | 0.132 | 0/80 |

The result is a reachability diagnosis, not a failure threshold and not permission to change object placement.

Mean minimum distance by finger further shows geometry-dependent variation: grasp 01 index/middle/ring/thumb = 0.156/0.115/0.136/0.137 m; grasp 02 = 0.151/0.161/0.144/0.128 m; grasp 03 = 0.138/0.120/0.128/0.143 m.

Object-A disturbance was logged during every approach:

| Grasp | Mean max translation [m] | Mean max rotation [rad] | Mean final vertical displacement [m] | Mean max force redistribution [N] | Complete contact loss / table contact |
|---|---:|---:|---:|---:|---:|
| 01 | 0.002832 | 0.08581 | -0.001086 | 2.972 | 0 / 0 |
| 02 | 0.005123 | 0.16615 | -0.001709 | 1.443 | 0 / 0 |
| 03 | 0.004334 | 0.12266 | -0.002906 | 1.788 | 0 / 0 |

## Exploratory associations

Pearson coefficients are marked `exploratory_only`. With only three grasp families and repeated correlated probes, they are descriptive and do not justify a metric, controller, or recommendation. Examples include non-A-contacting fingers versus A translation (-0.454), mean actuator reserve versus A translation (+0.442), and minimum joint margin versus B distance (+0.382). Correlations with B contact are undefined because all contact indicators are zero.

## Artifacts and boundary

The ignored output directory contains raw CSV/JSON, 12 reachability NPZ files, and figures for contact patterns, joint occupancy/margins, actuator demand/reserve, A force distribution, fingertip workspaces, B distances/contact counts, and A disturbance. The committed `manifest.json` is intentionally not included because experiment outputs are reproducible generated data.

No resource metric J, scientific retention/success threshold, reward, tactile control law, finger gating, sequential grasp policy, or PPO experiment was defined. The decisions still required from the PI are listed in `RESOURCE_METRIC_PI_DECISION.md` and `PI_DECISIONS.md`.
