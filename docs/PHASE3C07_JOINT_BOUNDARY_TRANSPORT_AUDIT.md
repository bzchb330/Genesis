# Phase 3C-0.7 joint-boundary transport audit

Exact compiled-boundary events were localized by joint, timestep, and stage in every trial.

- Event counts by joint: `{'rh_LFJ5': 130753, 'rh_MFJ1': 171418, 'rh_RFJ1': 145486, 'rh_LFJ1': 44912, 'rh_LFJ2': 43527, 'rh_RFJ2': 6681}`.
- Transport-limiting events: `0/500`. All observed exact-boundary joints were passive open middle/ring/little joints; none was a thumb/index transport boundary, so causation was not assigned.
- Native wrist diagnostic: `PHASE3C07_FOREARM_DOF_LIMIT`.
- Desired direction: `[-0.7773763338783912, 0.6284439191228217, -0.027281423044633563]`.
- Best native-wrist gravity direction: `[-1.1102230246251573e-16, 0.9990482316709892, -0.043619156285622795]`.
- Residual orientation angle: `51.021°`.
- Missing component: native WRJ1/WRJ2 cannot generate the required palm-x gravity component. W2/W3 expansion was stopped as required; no MJCF or joint limit was changed.

Boundary association with progress remains descriptive; the experiment does not alter limits or invent a margin threshold.
