# Phase 3B-0.5 Results

This phase is pre-RL engineering calibration only. Physics, collision
geometry, actuator/joint limits, finger-controller keyframes, reward weights,
and the Phase 3B-0 dataset were not changed.

- base commit: `6520796fdc7a2709c53ecb7667361aae2c0135b8`
- feasibility candidates: 800
- valid releases: 705
- active matched trials: 240
- full diagnostic handoff fraction: 0.000000
- palm-contact fraction: 0.166667
- support-shift fraction: 0.954167
- thumb-release fraction: 0.858333
- index-release fraction: 0.891667

## Effective diversity sensitivity

No nonzero threshold is selected. Retained effective N across all valid
explored states:

| Dimensionless RMS threshold | Effective N |
|---:|---:|
| 0.0 | 705 |
| 0.01 | 700 |
| 0.025 | 648 |
| 0.05 | 539 |
| 0.1 | 393 |
| 0.15 | 256 |
| 0.2 | 175 |

## Recovered-finger persistence

| Finger | Horizon (steps) | Contact-free | Object retained | Combined with available motion |
|---|---:|---:|---:|---:|
| thumb | 10 | 0.750 | 0.750 | 0.500 |
| thumb | 25 | 0.750 | 0.625 | 0.500 |
| thumb | 50 | 0.750 | 0.625 | 0.500 |
| thumb | 100 | 0.750 | 0.500 | 0.500 |
| thumb | 150 | 0.750 | 0.500 | 0.500 |
| thumb | 250 | 0.750 | 0.500 | 0.500 |
| thumb | 500 | 0.750 | 0.500 | 0.500 |
| index | 10 | 0.750 | 0.500 | 0.375 |
| index | 25 | 0.750 | 0.500 | 0.375 |
| index | 50 | 0.750 | 0.500 | 0.375 |
| index | 100 | 0.750 | 0.500 | 0.375 |
| index | 150 | 0.750 | 0.500 | 0.375 |
| index | 250 | 0.750 | 0.500 | 0.375 |
| index | 500 | 0.750 | 0.500 | 0.375 |

## Usable-motion probes

The released finger moved toward the unchanged pre-grasp target and returned;
it was never moved toward object B. Each scale reports joint-space availability,
Jacobian-derived fingertip envelope, selected-finger contact clearance, and
retained-object behavior in the machine summary.

| Finger / scale | Trials | Collision-free | Retained | Median joint range (rad) | Median Jacobian envelope (m) |
|---|---:|---:|---:|---:|---:|
| thumb_0.25 | 8 | 0.500 | 0.500 | 2.19474 | 0.0273101 |
| thumb_0.5 | 8 | 0.500 | 0.500 | 2.19514 | 0.027292 |
| thumb_1.0 | 8 | 0.500 | 0.500 | 2.19534 | 0.027234 |
| index_0.25 | 8 | 0.375 | 0.500 | 0.661285 | 0.0314489 |
| index_0.5 | 8 | 0.375 | 0.500 | 0.661397 | 0.0314488 |
| index_1.0 | 8 | 0.375 | 0.500 | 0.661695 | 0.0314501 |

## E2 / E3 / E6 sensitivity

The raw paired tables are stored in `outputs/phase3B05/summary.json`. Candidate
options are recommendations for PI review only:

- E2: [0.5, 1.0]x; 0.5x-1.0x bracket the two highest observed retention fractions without the 1.5x joint-margin deterioration; zero complete handoffs prevents a final bound.
- E3: [0.75, 1.0]x; 0.75x-1.0x retain more palm/support evidence than 0.25x; zero complete handoffs prevents freezing a lower limit.
- E6: [1.0]x; 1.0x is the scripted reference and had the highest observed retention; neither slower nor faster rates established complete handoff.

No condition completed the full palm-contact handoff diagnostic across the
expanded matched cohort, so none of these options is a validated final bound.

## Contact gaps and orientation

Passive: 2877 gaps, median/p95/max duration 0.006/0.06/0.092 s, re-established fraction 0.937435.
Active: 3745 gaps, median/p95/max duration 0.004/0.076/2.696 s, re-established fraction 0.969559.
Recovered gaps remain distinct from permanent loss. Orientation is reported
with total change, D2 symmetry-aware change, angular speed, sustained angular
speed, and later retention; rotation alone is not a failure criterion.

## Readiness

**PPO_NOT_READY**

- no expanded-reset active trial completed the full palm-contact handoff diagnostic, so E2/E3/E6 cannot be frozen from successful matched handoffs
- A3/A4/A5/A6/B1/B2/B5/C1/C2/E2/E3/E6 recommendations remain explicitly unfrozen pending PI decision
- the official pre-grasp keyframe starts several free-joint/tendon coordinates outside compiled limits and was not altered in this audit

Phase 3A's deterministic handoff reproduced successfully. Phase 3B-0 does not
require revalidation because no baseline implementation or physics was changed.
Raw artifacts and MP4s remain under ignored `outputs/phase3B05/`; reports and
the 14 vector figures are under `docs/` for PI review.
