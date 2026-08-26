# Phase 3B-0 Reset-Range Audit

## Decision

Phase 3B-0 candidate generation is authorized, but only inside the conservative
position domain already exercised by Phase 3A. Phase 3A used a nominal object
centre and six independent Cartesian perturbations of plus or minus 4 mm. The
Phase 3B-0 sampler therefore uses deterministic interpolation inside their
convex hull (the L1 ball `|dx| + |dy| + |dz| <= 0.004 m`). This does not include
the unexercised corners of the enclosing Cartesian box.

No repository evidence supports perturbing object orientation, wrist pose,
hand keyframes, controller timing, or contact thresholds. Those quantities
remain fixed. The mandatory `PHASE3B0_MISSING_RESET_RANGE_PI` stop does not
apply because Phase 3A contains seven explicit, physically exercised position
samples rather than a single nominal state.

## Exact exercised object-position domain

Source: `configs/phase3A_shadow_hand.yaml:10-38` and
`seqgrasp/phase3/experiments.py:237-243`.

- nominal world position: `[0.379, -0.040, 0.023]` m;
- exercised offsets: `[0,0,0]` and plus/minus 0.004 m independently on x, y,
  and z;
- exercised x extrema: `[0.375, 0.383]` m;
- exercised y extrema: `[-0.044, -0.036]` m;
- exercised z extrema: `[0.019, 0.027]` m;
- authorized interpolation domain: convex hull of the seven exercised points;
- object quaternion: fixed `[1, 0, 0, 0]` (no exercised orientation range).

The separate handoff demonstration position `[0.375, -0.036, 0.018]` m is not
part of the minimal-acquisition cohort and is not used to broaden this domain.

## Hand and wrist initialization

Sources: `configs/hand_shadow_right.yaml`,
`assets/hands/shadow_right/keyframes.xml`, and
`seqgrasp/phase3/experiments.py:91-103`.

- hand model: official MuJoCo Menagerie right Shadow Hand E3M5;
- vendored source commit: `c1a4eeb85694ae1dffe33ff1797d4e528928a133`;
- forearm mount position: `[0, 0, 0]`;
- forearm mount quaternion: `[0, 1, 0, 1]` as preserved from the model source;
- wrist initial joints (`rh_WRJ2`, `rh_WRJ1`): `[-0.03896, -0.5694]` rad from
  the `pre grasp` keyframe;
- wrist approach/closing target: unchanged because the acquisition controller
  commands only thumb and index actuators;
- no wrist sampling range exists and none is introduced.

The complete 24-joint `pre grasp` and `two finger pinch` keyframes remain the
exact arrays stored in `assets/hands/shadow_right/keyframes.xml`; Phase 3B-0
does not perturb them.

## Thumb and index approach

Source: `seqgrasp/phase3/experiments.py:106-132`.

- active acquisition groups: thumb and index only;
- initial actuator targets: actuator-space projection of `pre grasp`;
- closing targets: actuator-space projection of `two finger pinch`;
- trajectory: linear interpolation over 180 simulation steps;
- independent contact latch threshold: 0.02 N normal force;
- once a finger latches, its target is held at the current actuator command;
- fixture settling after closure: 50 simulation steps;
- middle, ring, and little remain at their `pre grasp` targets;
- no approach-keyframe or controller-parameter sampling range exists.

The configuration also contains `approach_steps: 80`, but the validated
minimal-acquisition code does not execute a separate approach loop. Phase 3B-0
preserves the actual validated code path and records this discrepancy rather
than adding a new motion.

## Fixture and release

Sources: `seqgrasp/phase3/model.py:97-129,172-189` and
`seqgrasp/phase3/experiments.py:91-103,187-202`.

- fixture: equality weld `phase3_object_fixture` between the dynamic object and
  mocap body `phase3_fixture_anchor`;
- fixture pose: exactly the sampled object position and fixed object quaternion;
- object linear and angular velocity are zeroed during initial setup;
- fixture remains active through closing and 50-step settling;
- the release state is sampled immediately before the equality is disabled;
- after release, object qpos is never assigned and the fixture is never
  reactivated.

## Fixed physics and execution values

- timestep: 0.002 s;
- frame skip in the Gymnasium environment: 5 (not used to alter the diagnostic
  simulation step);
- gravity: `[0, 0, -9.81]` m/s^2;
- floor z: -0.10 m;
- object: ellipsoid size `[0.03, 0.04, 0.02]` m;
- object friction: `[0.5, 0.01, 0.003]`;
- object compiled contact dimension: 6;
- fingertip compiled contact dimension: 3;
- fingertip friction: `[1.0, 0.005, 0.0001]`;
- actuator-displacement limit: 0.04 actuator-coordinate units;
- stiffness-scale bounds: `[0.2, 1.0]`;
- historical penetration reference: 0.003 m, retained as a diagnostic only.

## Sampled and fixed dimensions

| Quantity | Phase 3B-0 treatment | Basis |
|---|---|---|
| Object x/y/z | Deterministically sampled inside the seven-point convex hull | Explicit Phase 3A cohort |
| Object roll/pitch/yaw | Fixed at zero | No exercised range |
| Wrist configuration | Fixed at `pre grasp` | No exercised range |
| Thumb approach | Fixed contact-aware `pre grasp` to `two finger pinch` path | Validated Phase 3A controller |
| Index approach | Fixed contact-aware `pre grasp` to `two finger pinch` path | Validated Phase 3A controller |
| Middle/ring/little | Fixed `FREE`, no object support | Required minimal-acquisition protocol |
| Fixture timing | Fixed close plus settle sequence | Validated Phase 3A controller |

This audit authorizes interpolation only. Any future extrapolation, orientation
variation, wrist variation, keyframe perturbation, or controller variation
requires PI input.
