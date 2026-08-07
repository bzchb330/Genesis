# Single-Object Grasp Characterization

## Diagnostic setup

- Profile: unchanged engineering reference `grasp_A_01`.
- Seeds: integers 0 through 9; 950 deterministic samples per run at 0.002 s.
- Stages: open/pregrasp 0.20 s, close 0.50 s, establish contact 0.20 s, retention attempt 0.50 s, hold 0.50 s.
- Object A is fixture-held through establish-contact; support release is explicit at simulation time 0.900 s.
- No controller gains, posture fractions, timing, fixture pose, or force scaling were tuned.
- No scientific success/failure labels were assigned.

Raw CSV, NPZ, resource-state NPZ, JSON, per-run figures, and aggregate figures are generated under the gitignored `outputs/scripted_grasp_a/seeded_runs/` directory.

## Descriptive 10-seed statistics

| Quantity after support release | Minimum | Maximum | Mean |
|---|---:|---:|---:|
| Object center height (m), pooled | 0.021854 | 0.164919 | 0.115163 |
| Vertical displacement from release (m), pooled | -0.143146 | -0.000081 | -0.049837 |
| Translational drift from release (m), pooled | 0.000092 | 0.179855 | 0.059059 |
| Orientation change (rad), pooled | 0.002414 | 1.897152 | 0.727887 |
| Simultaneously contacting fingers | 0 | 4 | 2.9094 |
| Longest continuous fingertip contact per run (s) | 0.540 | 0.896 | 0.6988 |
| Duration above table-resting center height per run (s) | 0.754 | 1.000 | 0.8970 |
| Vertical velocity immediately after release (m/s) | -0.042926 | -0.040735 | -0.041729 |
| Time to first complete fingertip-contact loss (s) | 0.540 | 0.896 | 0.6988 |

Immediate downward displacement one physics step after release ranged from -0.00008585 m to -0.00008147 m (mean -0.00008346 m). Final vertical displacement ranged from -0.140765 m to -0.120912 m (mean -0.138319 m); final center height ranged from 0.024235 m to 0.044088 m (mean 0.026681 m).

| Finger | Minimum peak (N) | Maximum peak (N) | Mean peak (N) |
|---|---:|---:|---:|
| index | 1.6499 | 2.5739 | 2.2533 |
| middle | 6.1334 | 6.2591 | 6.1875 |
| ring | 2.8815 | 4.5614 | 3.9864 |
| thumb | 6.2429 | 6.5610 | 6.3898 |

## Before support removal

The fixture held object A at center height 0.165 m. Immediately before release, all ten runs reported index, middle, and thumb contact; the ring flag was zero. Mean force was `[1.6835, 6.1805, 0.0000, 6.4112] N` in configured finger order.

## Immediately after support removal

The same index/middle/thumb configuration remained on the first unsupported sample in every run. Mean force was `[1.6583, 6.1875, 0.0000, 6.3095] N`. Vertical velocity was negative in every run, and the object began moving downward immediately.

## Later unsupported behavior

- Every run exhibited a first sample with zero configured-fingertip contacts between 0.540 s and 0.896 s after release.
- Object center height moved downward substantially relative to release in every run.
- Nine final heights were near the configured table-resting center height; one was 0.044088 m at the end of the fixed observation window.
- No run left the configured workspace or triggered early mechanical termination.
- Lateral drift, orientation change, contact duration, and additional contacts varied by seed.

These are observations, not scientific failure labels. Returning near table height, contact loss, downward motion, or drift only become criteria if the PI selects them.

## Informative signals and remaining decisions

Acquisition evidence can use fixture/contact stage, table clearance, active-finger count, raw force, and persistence before release. Retention evidence can use release time, table clearance, XYZ drift, quaternion change, velocity, uninterrupted contact duration, and time to contact loss.

The PI must define acquisition evidence, unsupported hold duration, clearance requirements, allowable translation/rotation, contact continuity, temporary-loss handling, and loss/drop semantics. `SINGLE_GRASP_PI_DECISION.md` maps each option to measured signals without choosing values.
