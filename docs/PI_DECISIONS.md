# PI Decision Register

This register enumerates every active `TODO(PI)`. Diagnostic defaults are engineering probes only and do not answer any item below.

| File:line | Function/class or setting | Current placeholder behavior | Scientific decision required | Downstream dependencies | Repository works without it? |
|---|---|---|---|---|---|
| `configs/task_sequential.yaml:16` | `tactile_normalization` | `null`; forces remain raw newtons | Whether and how to normalize total normal force | Observation scaling, policy inputs, plots intended for learning | Yes |
| `configs/task_sequential.yaml:18` | `extra_tactile_feature_dim` | Zero dimensions | Any additional tactile features and their physics/dimensions | Observation contract, future policies | Yes |
| `configs/task_sequential.yaml:20` | phase threshold placeholders | `null`; automatic transitions disabled | Per-phase transition criteria and thresholds | Sequential task progression, evaluation | Yes, as scaffold/diagnostics |
| `configs/task_sequential.yaml:23` | `drop_height_threshold` | `null`; retained-object drop test disabled | Definition of object loss/drop | Early termination, drop reporting | Yes |
| `configs/task_sequential.yaml:25` | `reward_weights` | Every weight is `0.0` | Reward weights after terms are scientifically defined | RL training | Yes; reward remains zero |
| `seqgrasp/sensing/tactile_features.py:10` | `compute_tactile_features` extra slot | Correctly shaped zeros | Physics and units for any feature beyond flag and total normal force | Future observation variants | Yes |
| `seqgrasp/control/retention.py:12` | `ZeroRetentionController.residual` | Zero residual | Closed-loop tactile retention strategy | Persistent retention control | Yes; open-loop diagnostics remain available |
| `seqgrasp/env/resource.py:30` | `compute_resource_metric` | Returns `None`; expanded raw state remains exposed | Definition, admissible inputs, aggregation, and units of resource metric J | Reward resource term, resource evaluation | Yes |
| `seqgrasp/env/rewards.py:4` | `compute_reward` | All term bodies return zero | Retention, progress, J, regularization, and failure term definitions | Meaningful RL objective | Yes; APIs/logging work |
| `seqgrasp/env/termination.py:11` | `update_phase` | Returns current phase and no reason | Scientific phase transition rules | Sequential phase advancement | Yes |
| `seqgrasp/env/termination.py:18` | `failure_reason` retained-object branch | Workspace exits work; drop branch disabled | Loss/drop criterion for a held object | Early termination, drop metrics | Yes |
| `scripts/evaluate.py:12` | evaluation aggregation | No inferred threshold | Success and drop evaluation criteria | Scientific benchmark reporting | Yes; neutral diagnostics work |
| `seqgrasp/env/grasp_criteria.py:16` | `is_grasp_acquired` | Returns `None` | Acquisition evidence, combination, and persistence | Grasp-state transition and evaluation | Yes |
| `seqgrasp/env/grasp_criteria.py:20` | `is_object_retained` | Returns `None` | Unsupported retention definition | Persistent-retention reporting and sequential task | Yes |
| `seqgrasp/env/grasp_criteria.py:24` | `is_object_lost` | Returns `None` | Loss/drop definition beyond workspace exit | Early termination and evaluation | Yes |

## Phase 2 hard-gate and later-experiment inputs

These entries are consolidated in `PHASE2_PI_INPUTS_REQUIRED.md`. They remain null and prevent Parts B–F execution.

| File:line | Setting | Current placeholder behavior | Scientific decision required | Downstream dependency | Repository works without it? |
|---|---|---|---|---|---|
| `configs/phase2_physics_validation.yaml:10` | `penetration_tolerance_m` | `null`; penetration is reported only | Maximum physically valid penetration | Part A gate, Part D INVALID | Part A reports `PI_INPUT_REQUIRED` |
| `configs/phase2_physics_validation.yaml:11` | `maximum_vertical_drift_m` | `null`; raw drift is reported | Stable-hold vertical drift limit | Part A/B stability | Same |
| `configs/phase2_physics_validation.yaml:12` | `maximum_translational_drift_m` | `null`; raw drift is reported | Stable-hold 3-D drift limit | Part A/B stability | Same |
| `configs/phase2_physics_validation.yaml:13` | `maximum_orientation_drift_rad` | `null`; raw angle is reported | Stable-hold rotation limit | Part A/B stability | Same |
| `configs/phase2_physics_validation.yaml:14` | `minimum_active_object_contacts` | `null`; raw contact range is reported | Required contact count | Part A gate | Same |
| `configs/phase2_physics_validation.yaml:15` | `allow_table_recontact` | `null`; event is reported | Table-contact retention policy | Part A/B/D | Same |
| `configs/phase2_physics_validation.yaml:16` | `allow_complete_contact_loss` | `null`; event is reported | Temporary total-contact-loss policy | Part A/B/D | Same |
| `configs/phase2_physics_validation.yaml:19` | contact sweep ranges/scope | All five fields are `null`; no sweep trials run | Target geoms and friction, solref, solimp, timestep ranges | Part A sweep and physics selection | Sweep reports `PI_INPUT_REQUIRED` |
| `configs/phase2_physics_validation.yaml:32` | `occupied_finger_force_threshold_N` | `null` | Load-bearing finger threshold | Part C1 | Blocked after Part A |
| `configs/phase2_physics_validation.yaml:33` | `tactile_binary_force_threshold_N` | `null` | Binary tactile-contact threshold | Part E1 | Blocked after Part A |
| `configs/phase2_physics_validation.yaml:34` | `short_hold_drift_tolerance_m` | `null` | Dataset stability tolerance | Part B2 | Blocked after Part A |
| `configs/phase2_physics_validation.yaml:35` | `grasp_acquisition_threshold` | `null` | Acquisition signals, thresholds, and persistence | Part B/D | Blocked after Part A |
| `configs/phase2_physics_validation.yaml:36` | `retained_object_threshold` | `null` | End-of-hold retention definition | Part D outcomes | Blocked after Part A |
| `configs/phase2_physics_validation.yaml:37` | `object_loss_drop_threshold` | `null` | Loss/drop definition | Part D outcomes | Blocked after Part A |
| `configs/phase2_physics_validation.yaml:38` | `invalid_penetration_threshold_m` | `null` | Penetration causing INVALID | Part D outcomes | Blocked after Part A |
| `configs/phase2_physics_validation.yaml:39` | `B_placement_low_m` | `null` | Lower bound of reachable Phase 2 B distribution | Part D2 | Blocked after Part A |
| `configs/phase2_physics_validation.yaml:40` | `B_placement_high_m` | `null` | Upper bound of reachable Phase 2 B distribution | Part D2 | Blocked after Part A |
| `configs/phase2_physics_validation.yaml:41` | `workspace_monte_carlo_samples` | `null` | Production sample budget after convergence evidence | Part C2 | Blocked after Part A |
| `configs/phase2_physics_validation.yaml:42` | `workspace_voxel_size_m` | `null` | Reachable-workspace voxel resolution | Part C2 | Blocked after Part A |
| `configs/phase2_physics_validation.yaml:43` | `workspace_collision_tolerance_m` | `null` | Collision-rejection clearance | Part C2 | Blocked after Part A |
| `configs/phase2_physics_validation.yaml:44` | `free_palm_box_low_m` | `null` | Palm-frame voxel-box lower bounds | Part C3 | Blocked after Part A |
| `configs/phase2_physics_validation.yaml:45` | `free_palm_box_high_m` | `null` | Palm-frame voxel-box upper bounds | Part C3 | Blocked after Part A |
| `configs/phase2_physics_validation.yaml:46` | `free_palm_voxel_size_m` | `null` | Free-palm voxel resolution | Part C3 | Blocked after Part A |
| `configs/phase2_physics_validation.yaml:47` | `second_grasp_trials_per_grasp` | `null` | B placements per accepted A grasp | Part D4 | Blocked after Part A |
