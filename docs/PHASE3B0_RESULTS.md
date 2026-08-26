# Phase 3B-0 Results

## Scope

This phase generated and characterized single-object Shadow-Hand minimal
thumb-index acquisition states. It did not run RL, introduce object B, define
scalar `J`, choose reward weights, or modify physics.

## Dataset

- attempts: 500
- episodes reaching fixture release: 500
- raw valid thumb-index release states: 500
- acquisition success fraction: 1.000000
- exact-state unique count: 500
- exact duplicate fraction: 0.000000
- pose/contact regions: 10
- largest region fraction: 0.100000
- TRAIN/VALIDATION/TEST: 300/100/100
- zero split ID overlap: True
- zero serialized-state overlap: True

The nonzero near-duplicate thresholds are sensitivity analyses only. At
dimensionless RMS thresholds 0.01, 0.025, 0.05, and 0.1, the retained
counts are 468,
243,
70, and
17. No
near-duplicate threshold is frozen.

## Release physics

- intended-grip penetration: median=0.000654409 m, p90=0.000691126 m, p95=0.000696098 m, p99=0.00070273 m, maximum=0.00081362 m
- maximum gross/non-grip release penetration: 0 m for all accepted states
- thumb release force: median=1.21386 N, p90=1.34531 N, p95=1.36403 N, p99=1.38599 N, maximum=1.41393 N
- index release force: median=1.20648 N, p90=1.24497 N, p95=1.25258 N, p99=1.27437 N, maximum=1.29888 N
- total release force: median=2.42479 N, p90=2.55741 N, p95=2.59201 N, p99=2.64555 N, maximum=2.71281 N
- minimum joint margin: median=-0.00399017 rad, p90=-0.00361793 rad, p95=-0.00355033 rad, p99=-0.00345302 rad, maximum=-0.00328121 rad
- saturated actuators at release: median=2, p90=2, p95=2, p99=2, maximum=2

The slightly negative joint margin and persistent saturation are descriptive
properties of the unchanged validated acquisition/controller state. They are not
silently reclassified or tuned away.

## Unsupported retention

| Steps | Survival fraction | Translation median / p95 (m) | Rotation median / p95 (rad) |
|---:|---:|---:|---:|
| 1 | 1.000000 | 7.33972e-06 / 9.36997e-06 | 0.000618121 / 0.000710566 |
| 5 | 1.000000 | 0.000134512 / 0.000165405 | 0.00700778 / 0.00803916 |
| 10 | 1.000000 | 0.000457002 / 0.000565683 | 0.0207508 / 0.0247305 |
| 25 | 1.000000 | 0.00219336 / 0.00322596 | 0.0975812 / 0.145066 |
| 50 | 1.000000 | 0.0119193 / 0.0182614 | 0.521522 / 0.728042 |
| 100 | 1.000000 | 0.0386286 / 0.0517394 | 1.38885 / 1.43242 |
| 200 | 0.706000 | 0.0380201 / 0.0413037 | 1.39751 / 1.44494 |
| 300 | 0.676000 | 0.0378401 / 0.0405814 | 1.39444 / 1.43861 |
| 500 | 0.674000 | 0.037815 / 0.0406055 | 1.39015 / 1.43317 |
| 750 | 0.662000 | 0.0378471 / 0.0406934 | 1.38456 / 1.42715 |
| 1000 | 0.646000 | 0.0378879 / 0.0408235 | 1.37858 / 1.42163 |

All 500 trajectories contained at least one complete
contact gap. Across 2877 gaps, duration median/p95/max was
0.006/0.06/0.092 s;
2697 gaps (0.937435) re-established contact.

## Resource precursor

The free identity was `middle+ring+little` for all 500 release states.
Available motion across those free digits was median=1.5659 rad, p90=1.86558 rad, p95=1.86558 rad, p99=1.86558 rad, maximum=1.86559 rad.
No scalar resource score was calculated.

## Phase 3A reproduction

Exact chain reproduced: True.
Checks: `{"dynamic_palmward_motion": true, "dynamics_only_after_release": true, "object_retained": true, "palm_contact": true, "support_shift": true, "thumb_index_acquisition": true, "thumb_released": true, "thumb_remains_free": true, "thumb_unloaded": true}`.

## Readiness

**Phase 3B-1 PPO is not ready to begin.** The raw target exists and the
Phase 3A handoff is reproducible, but orientation, wrist, and controller
dimensions were not authorized to vary; near-duplicate sensitivity reduces
the effective cohort substantially at nonzero thresholds; joint-margin and
actuator-saturation observations require PI interpretation; and the success,
safety, resource-recovery, action-bound, and reward decisions remain unfrozen.

Items that remain `INSUFFICIENT DATA` are C1 release persistence, E2 learned
actuator-displacement bounds, E3 stiffness lower bounds, and E6 learned action-
rate bounds. C2 has acquisition-state precursor evidence but not an explicit
released-finger motion probe.
