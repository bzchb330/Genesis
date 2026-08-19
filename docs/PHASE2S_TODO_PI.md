# Remaining TODO(PI) decisions

Phase 2S does not resolve any baseline scientific placeholder. The line references below are audited against the committed source at validation time.

| File and line | Current placeholder behavior | Scientific decision required from the PI |
|---|---|---|
| `configs/task_sequential.yaml:16` | `tactile_normalization: null`; raw normal force in newtons is preserved. | Whether tactile force should be normalized, and the physical/statistical scale to use. |
| `configs/task_sequential.yaml:18` | `extra_tactile_feature_dim: 0`; no additional features enter the observation. | The dimensions, units, and physics of any additional tactile features. |
| `configs/task_sequential.yaml:20` | Phase-transition thresholds are `null`; automatic transitions are disabled. | Observable transition conditions and persistence thresholds for each task phase. |
| `configs/task_sequential.yaml:23` | `drop_height_threshold: null`; the optional height-drop test is disabled. | The physical height/loss criterion for a retained object to count as dropped. |
| `configs/task_sequential.yaml:25` | All scientific reward weights remain zero. | Reward-term definitions, units/scaling, and weights. |
| `seqgrasp/control/retention.py:12` | The retention controller returns an all-zero residual. | The closed-loop tactile retention law. |
| `seqgrasp/sensing/tactile_features.py:45` | Candidate extra tactile features are correctly shaped zeros. | The physical definitions and units of candidate extra features. |
| `seqgrasp/env/termination.py:11` | `update_phase` leaves the current phase unchanged. | Phase-transition evidence and thresholds. |
| `seqgrasp/env/termination.py:18` | Drop-by-height remains disabled unless the config value is supplied; workspace exit still terminates. | The retained-object drop/loss height and any persistence rule. |
| `scripts/evaluate.py:12` | Seeded raw outcomes and per-phase terminations are reported; success/drop rates are `None`. | Scientific episode-level success and drop classifications. |
| `seqgrasp/env/rewards.py:4` | Retention, progress, resource, regularization, and failure terms are structural zeros. | Each term’s physical/statistical definition and all weights, including whether any resource term should exist. |
| `seqgrasp/env/resource.py:30` | `compute_resource_metric` returns `None`. | Whether to define a scalar resource metric, and if so its components, units, normalization, and aggregation. Phase 2S intentionally defines no scalar J. |
| `seqgrasp/env/grasp_criteria.py:16` | Grasp-acquisition classification returns `None`. | Contact/force/motion evidence and persistence needed to declare acquisition. |
| `seqgrasp/env/grasp_criteria.py:20` | Unsupported-retention classification returns `None`. | The unsupported retention criteria and persistence duration. |
| `seqgrasp/env/grasp_criteria.py:24` | Object-loss classification returns `None`. | Loss/drop criteria beyond mechanical workspace exit. |
