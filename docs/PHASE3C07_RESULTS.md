# Phase 3C-0.7 results

## Outcome

Primary classification: **PC-F**. The 25-mm geometry is statically plausible, but native WRJ1/WRJ2 cannot supply the lateral palm-frame gravity component requested by the pocket transport direction. This is reported as `PHASE3C07_FOREARM_DOF_LIMIT`; W2/W3 were not run.

## Frozen protocol results

- Frozen acquisition states: `50`; thumb/index contact `50/50`, unused-finger contact `0/50`, fixture-off dual-contact/no-floor retention `50/50` through the inherited 50-step hold.
- T0 old-target pocket entry: `0/50`.
- T1 fixed-wrist pocket entry: `0/50`.
- W1 wrist-assisted pocket entry: `0/400` across eight matched directions.
- P0/P1 were run only if transport reached or approached the pocket: N=`0` / `0`.
- Load-bearing cages: `0`; unique states `0`.
- Cage hold survival at 100/500/1000: `{'100': 0, '500': 0, '1000': 0}`.
- Maximum raw penetration by surface: `{'thumb': 0.0016461502691473649, 'index': 0.002846787835429984, 'middle': 0.0, 'ring': 0.0, 'little': 0.0, 'palm': 0.0}` m.
- Penetration acceptability: **TODO(PI)**; no threshold was invented.
- Thumb/index release: never performed. Object B, RL, compliant skin, and physics changes: none.

## Interpretation

The experiment distinguishes static fit from dynamic reachability. A smaller object can fit the geometry-derived pocket, but that does not by itself demonstrate transport or a cage. Thumb release and size progression remain premature unless the measured transport/cage chain supports PC-A or strong PC-B. Compliant skin is not justified by a native-DOF transport failure.

## Truthful videos

- `outputs/phase3C07/videos/25mm_thumb_index_acquisition.mp4`
- `outputs/phase3C07/videos/fixed_wrist_failed_pocket_transport.mp4`
- `outputs/phase3C07/videos/pocket_not_reached_failure.mp4`
