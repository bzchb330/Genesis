# Phase 2 formal experiment freeze

This protocol was frozen after geometry-only preflight and before dynamic pilot/formal outcomes. Dynamic results must not be used to change the distribution or classifier.

## Frozen inputs

- Source code baseline at freeze: `8c8116fac57d7f2982c8c23ca3ec7c5113586883` plus the uncommitted implementation described by this document; the final implementation commit is recorded in the final report.
- Physics/config bundle hash: `ad1add3a3087bb89e1f9316e151fd1514923324e405f1211415377cc197ccbdd`.
- Dataset: 227 accepted A grasps; original 200 preserved plus 27 strict-threshold extension acceptances.
- Occupied counts: 2 -> 30, 3 -> 138, 4 -> 59.
- B centre: independent uniform x `[0.055, 0.065]`, y `[0.115, 0.125]`, z `[0.215, 0.225]` m in the world frame.
- B orientation: vertical cylinder, uniform yaw `[0, 6.283185307179586)` rad, zero roll/pitch.
- B fixture: kinematic free-joint pose support through approach and closure; release timestep 900.
- Timing: 500 approach + 400 close + 500 unsupported hold = 1,400 simulation steps.
- Seeds: dataset `20260808`; B/calibration `20260809`; K=20 deterministic B indices per A grasp.
- Workers: at most 8; persistence is incremental, locked, deduplicated, and resumable.
- Resource definitions: `allegro_palm_axis_transform_v1`, 10,000 free-finger Monte Carlo samples, 0.005 m workspace voxels, strict occupied threshold `>0.20 N`; scalar J is undefined.

## Frozen A retention

A retention is unchanged: no A-table contact or complete finger-contact loss, at least two final supporting fingers, total final A normal force `>0.20 N`, A translation `<=0.005 m`, A orientation change `<=0.20 rad`, A penetration `<=0.003 m`, and finite numerical state. Occupied support fingers retain their accepted hold impedance targets. Free-finger commands may physically disturb A; no A retuning is permitted.

## Frozen functional B acquisition

B is acquired only if all ten PI conditions hold: fixture released and inactive in the full final hold; at least one initially free finger contact; at least one total hand-supporting contact (finger, palm, or mechanically valid hand geometry); total B-hand normal force `>0.20 N`; no table contact; no complete B-hand contact loss during the final retention window; B penetration `<=0.003 m`; maximum translation after release `<=0.005 m`; maximum orientation change after release `<=0.20 rad`; and numerical stability. Two free-finger contacts are not required.

Every trial is exactly one of `BOTH_RETAINED`, `A_DROPPED`, `B_NOT_ACQUIRED`, `BOTH_LOST`, or `INVALID`. Pilot records carry `pilot_only: true` in a separate `correlation/pilot/` store. Formal records carry `pilot_only: false` in `correlation/formal/`; analysis additionally filters the flag.

No scalar J, reward tuning, policy training, reinforcement learning, or PPO is part of this freeze.
