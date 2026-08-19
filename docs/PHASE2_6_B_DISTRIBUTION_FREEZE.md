# Phase 2.6 sequential B-distribution freeze

The sequential distribution is frozen before any A+B dynamic outcome is evaluated.

- x: `[0.04535998713890449, 0.04735998713890449] m`
- y: `[0.08409904934597245, 0.08609904934597246] m`
- z: `[0.22399001120990503, 0.22599001120990503] m`
- yaw: `[-0.10, 0.10] rad`
- orientation: vertical cylinder
- fixture: unchanged Phase 2 kinematic presentation followed by complete release
- formal-v3 seed namespace: `20261300`

Selection used B-only evidence and A-held geometry only. The source B-only control survived the frozen 500-step unsupported hold and had 45% success over 20 local perturbations. In the corrected geometry audit, which restores each accepted A posture after free-finger sampling, 10/60 stratified accepted A grasps had access in this local box; every sampled pose was accessible to 16.7%, and no tested pose initially overlapped A. This is slightly below the approximate 20% engineering target but is the only tested region with both nonzero access and zero A overlap.

The other 80%-robust B-only local box overlapped A in all 1,200 geometry trials, while the remaining box had only 6.7% typical access and 314 initial A overlaps. The selected box therefore meets the supplied nondegeneracy target without consulting sequential success or resource-correlation strength.

The authoritative machine-readable distribution is `configs/phase2_6_frozen_B_distribution.yaml`. It must not be changed using A+B calibration or formal-v3 outcomes.
