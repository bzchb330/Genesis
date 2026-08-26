# Phase 3B-1A pre-grasp feasible projection

## Purpose

The official Shadow Hand `pre grasp` keyframe contains five small generalized-coordinate violations of the compiled joint ranges. Phase 3B-1A leaves the vendored MJCF, joint limits, actuator limits, tendons, and physics unchanged. At runtime it solves a minimum-L2 constrained projection over all 24 hand coordinates.

The constraints are the compiled per-joint lower and upper limits plus the four fixed-tendon actuator-coordinate ranges. The fixed-tendon rows couple each J2+J1 pair, so the implementation does not independently clip generalized coordinates.

## Projection result

| Joint | Requested (rad) | Projected (rad) | Change (rad) |
|---|---:|---:|---:|
| `rh_FFJ4` | -0.349700 | -0.349066 | +0.000634 |
| `rh_FFJ1` | -0.008296 | 0.000000 | +0.008296 |
| `rh_LFJ4` | -0.359300 | -0.349066 | +0.010234 |
| `rh_LFJ2` | -0.006910 | 0.000000 | +0.006910 |
| `rh_LFJ1` | -0.001588 | 0.000000 | +0.001588 |

The L2 projection magnitude is 0.0149743171 rad and the largest coordinate change is 0.010234 rad. The projected initial minimum compiled joint margin is 0.0 rad. All four fixed-tendon constraints are feasible.

## Command clipping and dynamic margins

The projected initialization places `rh_A_FFJ4`, `rh_A_LFJ4`, and `rh_A_LFJ0` at compiled command boundaries. Thus LF boundary command behavior remains, but it is a feasible coordinate target rather than an out-of-range initialization. The deterministic projected Phase 3A replay has a final minimum dynamic joint margin of -0.0014596914 rad. The matched 50-state projected cohort has a minimum release margin of -0.0070048273 rad, compared with -0.0079534304 rad before projection. These negative dynamic margins arise under the unchanged soft-limit dynamics; they are logged rather than silently clipped.

## Sanitation revalidation

The deterministic Phase 3A handoff remains resource-recovered after projection, with alternate support, thumb release, 2.1893298 rad available thumb motion, and no floor contact.

For the matched 50-state Conservative cohort:

| Metric | Before | After |
|---|---:|---:|
| valid acquisition | 50/50 | 50/50 |
| thumb-index-only release topology | 50/50 | 50/50 |
| 250-step retention | 25/50 | 27/50 |
| median intended penetration | 0.6974 mm | 0.7174 mm |
| maximum intended penetration | 1.0214 mm | 1.1106 mm |

All frozen engineering revalidation criteria passed. No vendored asset or physics parameter was modified.

Machine-readable evidence: `outputs/phase3B1A/projection/revalidation.json` (generated and ignored).
