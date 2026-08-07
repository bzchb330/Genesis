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
| `seqgrasp/env/resource.py:21` | `compute_resource_metric` | Returns `None`; raw state remains exposed | Definition and units of resource metric J | Reward resource term, resource evaluation | Yes |
| `seqgrasp/env/rewards.py:4` | `compute_reward` | All term bodies return zero | Retention, progress, J, regularization, and failure term definitions | Meaningful RL objective | Yes; APIs/logging work |
| `seqgrasp/env/termination.py:11` | `update_phase` | Returns current phase and no reason | Scientific phase transition rules | Sequential phase advancement | Yes |
| `seqgrasp/env/termination.py:18` | `failure_reason` retained-object branch | Workspace exits work; drop branch disabled | Loss/drop criterion for a held object | Early termination, drop metrics | Yes |
| `scripts/evaluate.py:12` | evaluation aggregation | No inferred threshold | Success and drop evaluation criteria | Scientific benchmark reporting | Yes; neutral diagnostics work |
| `seqgrasp/env/grasp_criteria.py:16` | `is_grasp_acquired` | Returns `None` | Acquisition evidence, combination, and persistence | Grasp-state transition and evaluation | Yes |
| `seqgrasp/env/grasp_criteria.py:20` | `is_object_retained` | Returns `None` | Unsupported retention definition | Persistent-retention reporting and sequential task | Yes |
| `seqgrasp/env/grasp_criteria.py:24` | `is_object_lost` | Returns `None` | Loss/drop definition beyond workspace exit | Early termination and evaluation | Yes |
