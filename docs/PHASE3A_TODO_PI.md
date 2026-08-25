# Phase 3A TODO(PI) Decisions

No item below is resolved by Phase 3A.

| File and line | Placeholder behavior | Scientific decision awaiting PI input |
|---|---|---|
| `configs/phase3A_shadow_hand.yaml:39` | The historical 0.003 m value is diagnostic only; final event thresholds are absent. | Define final Shadow acquisition, secure-support, support-transfer, slip, loss, meaningful-motion, and penetration thresholds/persistence. |
| `configs/phase3A_shadow_hand.yaml:46` | `object_progress_to_palm` reward weight is 0.0. | Choose weight/scaling, if approved. |
| `configs/phase3A_shadow_hand.yaml:47` | `valid_support` reward weight is 0.0. | Choose weight/scaling and valid-support definition, if approved. |
| `configs/phase3A_shadow_hand.yaml:48` | `palm_contact` reward weight is 0.0. | Choose weight/scaling, if approved. |
| `configs/phase3A_shadow_hand.yaml:49` | `support_transfer` reward weight is 0.0. | Choose weight/scaling and transfer definition, if approved. |
| `configs/phase3A_shadow_hand.yaml:50` | post-support acquisition-finger-release reward weight is 0.0. | Choose weight/scaling and secure-before-release definition, if approved. |
| `configs/phase3A_shadow_hand.yaml:51` | recovered-resource reward weight is 0.0. | Choose weight/scaling and meaningful resource definition, if approved. |
| `configs/phase3A_shadow_hand.yaml:52` | complete-object-loss reward weight is 0.0. | Choose weight/scaling and object-loss definition, if approved. |
| `configs/phase3A_shadow_hand.yaml:53` | unsafe-penetration reward weight is 0.0. | Choose weight/scaling and Shadow-specific unsafe-penetration definition, if approved. |
| `configs/phase3A_shadow_hand.yaml:54` | joint-limit reward weight is 0.0. | Choose weight/scaling, if approved. |
| `configs/phase3A_shadow_hand.yaml:55` | violent-action reward weight is 0.0. | Choose weight/scaling and action-safety normalization, if approved. |
| `seqgrasp/phase3/rewards.py:24` | Raw terms are returned; their weighted sum is zero. | Approve the combined scientific reward formulation, signs, normalization, and phase dependence. |

Historical Allegro/Phase 1-2 TODO(PI) entries remain unchanged in their original
files and reports. In particular, Phase 3A does not define scalar resource metric
`J`, historical grasp/drop semantics, tactile feature physics, or the historical
closed-loop retention law.
