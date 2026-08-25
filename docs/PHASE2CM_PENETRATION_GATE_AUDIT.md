# Phase 2CM-P penetration-gate audit

## Result

The frozen penetration threshold remains `0.003 m`. Of the 200 frozen Phase 2CM states, **189** already exceeded the gate at the exact release boundary before any post-release integration, while **189** exceeded it after the first post-release step. The primary replay classified 190 first failures as penetration failures.

No threshold, physics parameter, contact geometry, controller, or primary result was changed by this audit.

## Exact implementation audit

- **Phase 2W:** `seqgrasp/experiments/phase2_5_trajectory.py` takes the maximum of `-contact.distance` over every extracted contact whose body pair contains `object_b`. This includes intended fingertips, other hand links/palm, table, and object A, and it accumulates across the complete pre- and post-release trajectory.
- **Phase 2H:** `seqgrasp/experiments/phase2h_visuals.py` uses `B_penetration_depths_m`, which comes from the configured fingertip groups only. It includes index, middle, ring, and thumb fingertip–B contacts but excludes palm–B, table–B, B–A, and unconfigured other hand geoms. `np.maximum.accumulate` is applied only to the post-release slice beginning at `fixture_release_timestep`.
- **Phase 2CM primary:** `seqgrasp/experiments/phase2cm.py` starts a new cumulative maximum at zero and includes every contact involving B, but it first evaluates the gate after the first post-release `mj_step`; it does not test the saved boundary state itself.

Therefore intended fingertip–B solver overlap is included by all three paths. Phase 2CM's pair scope matches the broad Phase 2W summary, not Phase 2H's fingertip-only strict-series scope.

## Frozen 200 release-state distribution

| statistic | penetration [m] |
|---|---:|
| median | 0.00707941221266 |
| mean | 0.00785669928922 |
| p90 | 0.0136726071667 |
| p95 | 0.014805704096 |
| p99 | 0.0176175920738 |
| maximum | 0.0193051096284 |

| interval | N |
|---|---:|
| less than or equal 1 mm | 0 |
| greater 1 to 2 mm | 5 |
| greater 2 to 3 mm | 6 |
| greater 3 to 4 mm | 14 |
| greater 4 to 5 mm | 19 |
| greater than 5 mm | 156 |

### Violating responsible pairs

| pair | states where pair is maximum and >3 mm | states with any >3 mm contact of pair |
|---|---:|---:|
| index-B | 131 | 150 |
| thumb-B | 58 | 82 |
| middle-B | 0 | 0 |
| ring-B | 0 | 0 |
| palm-B | 0 | 0 |
| B-table | 0 | 0 |
| B-A | 0 | 0 |
| other | 0 | 0 |

## Original Phase 2H comparison

Reconstructed-versus-frozen release-boundary penetration differences had median `0 m`, mean `0 m`, and maximum absolute difference `0 m`. This tests the same pre-integration boundary in independently reconstructed original Phase 2H trajectories.

Pre-release pinch penetration is not accumulated into Phase 2H strict survival. It can nevertheless cause an immediate strict failure when the overlap persists into the first post-release sample, which is the sample at array index `fixture_release_timestep`.

## All 1,521 eligible states

At the pre-integration release boundary, **137/1521 (9.007%)** satisfy penetration `<= 0.003 m`. There are not at least 200 such states.

| wrist pose | eligible N | penetration-valid boundary N | valid % | first-post valid N |
|---|---:|---:|---:|---:|
| `coarse_r+90_p-45_y+0` | 11 | 0 | 0.000 | 0 |
| `coarse_r+90_p-45_y+90` | 39 | 7 | 17.949 | 7 |
| `coarse_r+90_p-45_y-45` | 18 | 2 | 11.111 | 2 |
| `coarse_r+90_p-45_y-90` | 79 | 0 | 0.000 | 0 |
| `refined_coarse_r+0_p+45_y-45_dr-22.5_dp-22.5_dy+0` | 209 | 1 | 0.478 | 1 |
| `refined_coarse_r+0_p+45_y-45_dr-22.5_dp-22.5_dy+22.5` | 94 | 0 | 0.000 | 0 |
| `refined_coarse_r+0_p+45_y-45_dr-22.5_dp-22.5_dy-22.5` | 759 | 101 | 13.307 | 101 |
| `refined_coarse_r+90_p-45_y+45_dr+0_dp+0_dy+22.5` | 91 | 12 | 13.187 | 12 |
| `refined_coarse_r-45_p+45_y+0_dr+22.5_dp-22.5_dy+0` | 81 | 9 | 11.111 | 9 |
| `refined_coarse_r-45_p+45_y+0_dr+22.5_dp-22.5_dy+22.5` | 140 | 5 | 3.571 | 5 |

## Physical interpretation and next step

MuJoCo's negative contact distance is solver overlap for the compiled rigid collision primitives. At the configured gripping interface, some negative distance is the intended compliant-contact representation rather than automatically proving gross geometric invalidity. The same signal can also identify invalid overlap with the table, object A, palm, or non-tip hand geometry. The pair identity and magnitude must therefore be kept explicit.

This audit does not decide whether intended fingertip solver overlap should be inside the scientific penetration gate. The next experiment should be specified only after the PI confirms the intended pair scope. Without changing the 3 mm threshold, a later paired freeze can require release-boundary validity under that confirmed definition, provided the eligible population remains at least 200; no new states are selected here.
