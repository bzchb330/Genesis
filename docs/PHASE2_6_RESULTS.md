# Phase 2.6 B-graspable workspace redesign results

## Status

`PHASE2_6_SEQUENTIAL_INTERSECTION_FAILED`

Phase 2.6 established robust B-only graspability under frozen physics but found zero A+B `BOTH_RETAINED` controls after the required expanded 8,192-candidate sequential search. Per the preregistered stop rule, population calibration, controller freeze, formal-v3, correlation inference, and J evidence generation were not started.

## Dense fixed-hand geometry

The palm origin was `[0,0,0.18] m`. Exactly 200,000 self-collision-valid configurations were retained for each finger. World-coordinate fingertip workspace bounds were:

- index: low `[-0.032464,-0.005265,0.134005]`, high `[0.111296,0.095882,0.305835] m`
- middle: low `[-0.030465,-0.050092,0.137632]`, high `[0.111386,0.050240,0.308496] m`
- ring: low `[-0.031158,-0.095836,0.133702]`, high `[0.111383,0.005663,0.305807] m`
- thumb: low `[-0.007272,-0.052091,0.075928]`, high `[0.125366,0.155587,0.192944] m`

The old Phase 2 box lies near the high-y index/thumb overlap and above the thumb global z maximum at most of its range, explaining why it was principally single-tip/index-thumb accessible and failed to provide demonstrated stable multi-contact acquisition.

## Multi-contact B-only search

Of 10,000 envelope-derived B centers, 6,515 had valid open-hand initial geometry, 7,104 had at least two-finger surface access, 1,644 had at least 120 degrees of contact opposition, and 1,144 had positive Ferrari-Canny evidence. Fifty diverse poses covered seven accessibility topologies.

The joint pose/trajectory search evaluated 8,192 candidates and produced three strict B-only controls at three distinct poses. Geometry topology labels were index+thumb and all-four accessible. Actual successful final contact topologies under 60 local perturbations were index+thumb, middle+thumb, and index+middle+thumb. Perturbation success was 34/60 overall (56.7%): 45%, 80%, and 45% by profile.

## A-held intersection and frozen distribution

After correcting state restoration following free-finger workspace sampling, the index-thumb local box was the only tested B-only region with both nonzero A-held access and zero initial A overlap. It was accessible to 10/60 stratified A grasps (16.7%). The exact frozen bounds are recorded in `configs/phase2_6_frozen_B_distribution.yaml` and were fixed before sequential dynamic outcomes.

## Sequential result

The expanded search evaluated 8,192 deterministic trajectory variants across 10 geometry-accessible A grasps and 20 frozen-box B poses. `BOTH_RETAINED=0`. Interaction diagnoses were:

- `A_DESTABILIZED`: 4,112
- `B_SLIP`: 1,740
- `B_CONTACT_LOST`: 1,406
- `FREE_FINGER_SET_INSUFFICIENT`: 934
- `A_BLOCKS_B_APPROACH`: 0
- `PALM_SUPPORT_OCCUPIED`: 0 observed
- `OTHER`: 0

This is an engineering interaction failure after B-only feasibility was demonstrated. It does not test the resource-correlation hypothesis and does not justify defining J.

## Scientific separation

Phase 2 and Phase 2.5 negative controls remain unchanged. All Phase 2.6 searches are calibration-only and raw outputs remain ignored. Physics, geometry, mass, friction, solver settings, gravity, timestep, controller gains, limits, tactile definitions, fixture logic, and frozen acquisition criteria were unchanged. No RL or physics tuning was performed.
