# PI Decision Register

This register enumerates every active implementation marker reserved for PI scientific input. Diagnostic defaults are engineering probes only and do not answer any item below. Phase 2 working experiment definitions supplied on 2026-08-08 are recorded separately and do not define scalar J.

| File:line | Function/class or setting | Current placeholder behavior | Scientific decision awaiting PI input | Repository works without it? |
|---|---|---|---|---|
| `configs/task_sequential.yaml:16` | `tactile_normalization` | `null`; raw force remains in N | Whether and how policy observations should normalize total normal force | Yes |
| `configs/task_sequential.yaml:18` | `extra_tactile_feature_dim` | Zero dimensions | Physics, units, and dimensions of any general-policy tactile features beyond the Phase 2 three-feature study | Yes |
| `configs/task_sequential.yaml:20` | phase thresholds | `null`; automatic transitions disabled | General sequential-task phase transition evidence and persistence | Yes |
| `configs/task_sequential.yaml:23` | `drop_height_threshold` | `null`; general drop-height test disabled | General retained-object loss/drop definition | Yes |
| `configs/task_sequential.yaml:25` | `reward_weights` | All weights `0.0` | Scientifically justified reward weights after terms are defined | Yes; reward is neutral |
| `scripts/evaluate.py:12` | evaluation aggregation | Success/drop rates reported unavailable | General benchmark success and drop criteria | Yes |
| `seqgrasp/control/retention.py:12` | `ZeroRetentionController.residual` | Returns a zero residual | Closed-loop tactile retention control law | Yes |
| `seqgrasp/env/grasp_criteria.py:16` | `is_grasp_acquired` | Returns `None` | General acquisition evidence and persistence | Yes |
| `seqgrasp/env/grasp_criteria.py:20` | `is_object_retained` | Returns `None` | General unsupported persistent-retention definition | Yes |
| `seqgrasp/env/grasp_criteria.py:24` | `is_object_lost` | Returns `None` | General loss/drop semantics beyond workspace exit | Yes |
| `seqgrasp/env/resource.py:30` | `compute_resource_metric` | Returns `None`; raw components remain separate | Definition, admissible inputs, aggregation, weights, and units of scalar resource metric J | Yes |
| `seqgrasp/env/rewards.py:4` | reward term bodies | Returns a zero total and zero terms | Retention, progress, J, regularization, and failure term definitions | Yes |
| `seqgrasp/env/termination.py:11` | `update_phase` | Keeps current phase | General phase transition rules | Yes |
| `seqgrasp/env/termination.py:18` | `failure_reason` retained-object branch | Workspace exit active; drop branch disabled | General loss/drop criterion for an already-held object | Yes |
| `seqgrasp/sensing/tactile_features.py:45` | general `extra_pi_features` | Correctly shaped zeros | Physics and units of future general-policy tactile features | Yes |

The Phase 2 study uses PI-supplied, experiment-scoped retention/acquisition definitions and exactly three raw tactile features. Those choices do not silently fill the general environment stubs above.
