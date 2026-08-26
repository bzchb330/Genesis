# Phase 3C-0.6 results

## Outcome

Primary classification: **SP-C**. The D0 sphere reached configured volumes in 390/2,000 trials, but only four transient storage states were detected and all four lost the sphere during thumb release. Valid thumb recovery and every 10-1,000-step survival checkpoint were 0. W2, W3, and D1-D4 were not run because the stated D0 progression gate was not reached.

## Frozen physical setup

- Branch: `codex/phase3C06-sphere-palmodigital-storage`
- Base commit: `7baac924a14ff863c7d1b0bb9bfc67734390609d`
- Frozen acquisition cohort: 50 IDs (`C06_D0_STATE_00000` through `C06_D0_STATE_00049`) before storage outcomes
- D0: diameter 0.035000 m, radius 0.017500 m, density 1000.0 kg/m3, compiled mass 0.022449297504 kg
- Matrix: 4 pockets x 2 preshape conditions x (W0 + 4 W1 commands) x 50 states = 2,000 trials
- World gravity, friction, compliance, official MJCF, collision geometry, and joint limits were unchanged. Object B, RL, rewards, and scalar J were absent.

## Matched pocket comparison

| target | center in palm (m) | half extents (m) | entry | transient stable capture | thumb recovery |
|---|---|---|---:|---:|---:|
| old_palm_center | (0.02, -0.025, 0.075) | (0.05, 0.05, 0.035) | 268/500 | 4/500 | 0/500 |
| middle_ring | (0.0, -0.0175, 0.092625) | (0.013125, 0.009625, 0.013125) | 122/500 | 0/500 | 0/500 |
| ring_little | (-0.022, -0.0175, 0.086375) | (0.013125, 0.009625, 0.013125) | 0/500 | 0/500 | 0/500 |
| ulnar_palmodigital | (-0.024625, -0.021875, 0.082875) | (0.013125, 0.009625, 0.013125) | 0/500 | 0/500 | 0/500 |

The old palm-center control produced all four transient captures. Middle/ring had 122 entries but no stable capture. Ring/little and adjacent ulnar-palmodigital targets had zero entries. Thus the tested palmodigital hypothesis is not supported by this controller/geometry result.

## Preshape and wrist

NO_PRESHAPE and PRESHAPE each produced 195/1,000 entries, 2/1,000 transient captures, and 0/1,000 thumb recoveries. Preshaping was not beneficial. W0 produced 0/400 transient captures. W1 produced 4/1,600, all under `[+5,+5]`; this is a direction-specific temporary settling effect, not a recovery benefit. Native wrist insufficiency and forearm-rotation necessity are not established because the transfer controller failed to reach the ulnar targets.

## Contacts, penetration, and survival

- First storage-finger contact: N=832, median step 234.0, range 135-298.
- Ring contact: 784/2,000; little contact: 128/2,000; palm/root contact: 406/2,000; alternate support: 832/2,000.
- Thumb release attempts: 4; valid thumb recoveries: 0; index release was not attempted because the primary thumb milestone failed.
- Maximum penetration across the full matrix: 0.007293006 m = 0.416743 R0. Penetration acceptability remains `TODO(PI)`; no new threshold or automatic gross-overlap label is applied.
- MuJoCo contact penetration is solver overlap and is not biological skin deformation. Multi-millimeter overlap is reported as a model warning, not justified as human compliance.
- Survival at 10, 25, 50, 100, 200, 300, 500, 750, and 1,000 steps: all 0/4 release attempts; losses occurred during the thumb-release ramp.

## Failure taxonomy

| label | trial count |
|---|---:|
| ACQUISITION_FAILED | 0 |
| TRANSFER_CORRIDOR_BLOCKED | 311 |
| PRESHAPE_TOO_EARLY | 0 |
| PRESHAPE_TOO_LATE | 0 |
| POCKET_NOT_REACHED | 1610 |
| POCKET_GEOMETRY_MISALIGNED | 0 |
| NO_STORAGE_FINGER_CONTACT | 1192 |
| NO_LOAD_BEARING_SUPPORT | 1168 |
| SPHERE_ROLLED_OUT | 0 |
| SPHERE_SLID_OUT | 0 |
| WRIST_DIRECTION_UNFAVORABLE | 0 |
| EXCESSIVE_PENETRATION | 0 |
| JOINT_BOUNDARY_LIMIT | 2000 |
| WRIST_DOF_LIMIT | 0 |
| LOSS_DURING_THUMB_RELEASE | 4 |
| LOSS_AFTER_THUMB_RELEASE | 0 |
| OTHER | 0 |

The joint-boundary diagnostic fired in all trials because the official open/target keyframes touch compiled joint bounds; the raw minimum margin is retained and the label is not treated as a new scientific exclusion rule.

## Decision

Rigid-contact geometry alone was insufficient for reproducible palmodigital storage and thumb recovery under this bounded protocol. A compliant-skin ablation is not yet justified: the intended ring/little and ulnar geometries were not reached, so transfer geometry/control must be resolved first. Object B must not be introduced next, and RL remains premature. The recommended next step is PI review of the unreachable ulnar transfer geometry and whether an explicit forearm reorientation DOF or a different scripted transfer family should be tested; no physics criterion should be relaxed.
