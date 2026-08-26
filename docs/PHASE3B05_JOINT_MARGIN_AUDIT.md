# Phase 3B-0.5 Joint-Margin Audit

The audit covers all 500 stored Phase 3B-0 release states and the sampled
states of a fresh deterministic Phase 3A successful handoff trajectory.
No model, limit, gain, controller keyframe, or stored result was changed.

Exact definition: `min(qpos - lower_limit, upper_limit - qpos) for each of the first 24 compiled hinge joints`.
Compiled solver tolerance: 1e-08.

| Joint | Type | Limits (rad) | Outside / 500 | Min / median / max margin (rad) | Tendon affected |
|---|---|---:|---:|---:|---|
| rh_WRJ2 | mjJNT_HINGE | [-0.523599, 0.174533] | 0 | 0.2121321 / 0.2134741 / 0.2147391 | False  |
| rh_WRJ1 | mjJNT_HINGE | [-0.698132, 0.488692] | 0 | 0.07383575 / 0.07697179 / 0.08012775 | False  |
| rh_FFJ4 | mjJNT_HINGE | [-0.349066, 0.349066] | 0 | 0.04688925 / 0.04879132 / 0.05123292 | False  |
| rh_FFJ3 | mjJNT_HINGE | [-0.261799, 1.5708] | 0 | 0.7667663 / 0.7938083 / 0.81894 | False  |
| rh_FFJ2 | mjJNT_HINGE | [0.0, 1.5708] | 0 | 0.5192009 / 0.5391023 / 0.5593682 | True rh_FFJ0 |
| rh_FFJ1 | mjJNT_HINGE | [0.0, 1.5708] | 0 | 0.4609331 / 0.4753694 / 0.4908522 | True rh_FFJ0 |
| rh_MFJ4 | mjJNT_HINGE | [-0.349066, 0.349066] | 0 | 0.2956206 / 0.2956247 / 0.295629 | False  |
| rh_MFJ3 | mjJNT_HINGE | [-0.261799, 1.5708] | 0 | 0.1681435 / 0.1681843 / 0.1681933 | False  |
| rh_MFJ2 | mjJNT_HINGE | [0.0, 1.5708] | 0 | 0.4572272 / 0.4572325 / 0.4572374 | True rh_MFJ0 |
| rh_MFJ1 | mjJNT_HINGE | [0.0, 1.5708] | 0 | 0.01174412 / 0.01174713 / 0.01175033 | True rh_MFJ0 |
| rh_RFJ4 | mjJNT_HINGE | [-0.349066, 0.349066] | 0 | 0.2439845 / 0.2439897 / 0.2439923 | False  |
| rh_RFJ3 | mjJNT_HINGE | [-0.261799, 1.5708] | 0 | 0.5139453 / 0.5139949 / 0.5140123 | False  |
| rh_RFJ2 | mjJNT_HINGE | [0.0, 1.5708] | 500 | -0.0006110243 / -0.0006072409 / -0.0006030771 | True rh_RFJ0 |
| rh_RFJ1 | mjJNT_HINGE | [0.0, 1.5708] | 0 | 0.02496148 / 0.02496298 / 0.02496357 | True rh_RFJ0 |
| rh_LFJ5 | mjJNT_HINGE | [0.0, 0.785398] | 0 | 0.1119399 / 0.1119879 / 0.1120296 | False  |
| rh_LFJ4 | mjJNT_HINGE | [-0.349066, 0.349066] | 0 | 0.004181275 / 0.004189428 / 0.004196932 | False  |
| rh_LFJ3 | mjJNT_HINGE | [-0.261799, 1.5708] | 0 | 0.3846984 / 0.3847316 / 0.3847671 | False  |
| rh_LFJ2 | mjJNT_HINGE | [0.0, 1.5708] | 500 | -0.0006723832 / -0.0006715983 / -0.00067089 | True rh_LFJ0 |
| rh_LFJ1 | mjJNT_HINGE | [0.0, 1.5708] | 500 | -0.0002186327 / -0.0002184866 / -0.00021817 | True rh_LFJ0 |
| rh_THJ5 | mjJNT_HINGE | [-1.0472, 1.0472] | 0 | 0.7732048 / 0.7987765 / 0.8245481 | False  |
| rh_THJ4 | mjJNT_HINGE | [0.0, 1.22173] | 0 | 0.2995944 / 0.3127144 / 0.3270894 | False  |
| rh_THJ3 | mjJNT_HINGE | [-0.20944, 0.20944] | 500 | -0.004631104 / -0.003990165 / -0.003281206 | False  |
| rh_THJ2 | mjJNT_HINGE | [-0.698132, 0.698132] | 0 | 0.2156159 / 0.218909 / 0.2226675 | False  |
| rh_THJ1 | mjJNT_HINGE | [-0.261799, 1.5708] | 0 | 0.3820379 / 0.3843803 / 0.3857982 | False  |

## Cause

The negative values are real qpos excursions beyond compiled hinge limits;
the indexing and margin formula are correct. They are not floating-point
noise at the compiled solver-tolerance scale. The official `pre grasp`
keyframe already contains out-of-range components (including negative distal
flexion coordinates and LFJ4 below its lower bound). MuJoCo joint limits are
soft constraints, so the unchanged settling/contact dynamics leave RFJ2,
LFJ2, LFJ1, and THJ3 slightly outside at all 500 releases. Coupled distal
coordinates are additionally affected by fixed J2+J1 tendons. This is a
physical generalized-coordinate soft-constraint excursion initiated by the
source keyframe, not a metric, qpos-indexing, tendon-indexing, or semantic-map bug.

Affected release joints: rh_RFJ2, rh_LFJ2, rh_LFJ1, rh_THJ3.
The same per-joint audit was performed across the Phase 3A handoff samples;
the affected sampled trajectory joints are listed below.

| Phase 3A handoff joint | Outside sampled states | Samples | Minimum margin (rad) |
|---|---:|---:|---:|
| rh_FFJ4 | 2 | 647 | -0.0001121498 |
| rh_FFJ1 | 4 | 647 | -0.004529646 |
| rh_RFJ2 | 266 | 647 | -0.000679945 |
| rh_LFJ4 | 9 | 647 | -0.00420035 |
| rh_LFJ2 | 272 | 647 | -0.003911227 |
| rh_LFJ1 | 258 | 647 | -0.0009024702 |
| rh_THJ4 | 59 | 647 | -0.004546545 |
| rh_THJ3 | 415 | 647 | -0.004710055 |

No fix was made: changing the keyframe, constraints, or model tolerance would
be a controller/physics design change, not an authorized software correction.
