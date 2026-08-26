# Phase 3B-1A.5 evaluation-validity stop

## Stop token

`PHASE3B1A5_EVALUATION_VALIDITY_COMPROMISED`

Phase 3B-1A.5 stopped at the curriculum-indicator audit before generating new datasets or running any DEV evaluation.

## Audit result

| Policy seed | Selected checkpoint | Saved stage | Selected-validation stage | Final reached stage | TEST-supplied stage | Match |
|---:|---|---:|---:|---:|---:|---|
| 33101 | `step_0275000.npz` | 5 | 5 | 5 | 5 | yes |
| 33102 | `step_0175000.npz` | 2 | 2 | 2 | 2 | yes |
| 33103 | `step_0125000.npz` | 2 | 2 | 5 | 5 | **no** |

The checkpoint metadata is correct, and checkpoint selection/report generation preserved the selected-validation stage. The defect is in the TEST call path in `scripts/run_phase3b1a.py`: `load_selected(...)` loads the weights but discards returned checkpoint metadata, and `pilot()` passes `curriculum_stage_reached` rather than the selected checkpoint's saved `curriculum_stage` to `evaluate_policy(...)`.

Classification:

- A, reporting-only mismatch: no;
- B, checkpoint metadata bug: no;
- C, evaluator restoration bug: yes;
- D, actor-input mismatch: yes;
- E, actual policy behavior mismatch: possible and directly evidenced at the action level.

## Action-level evidence

For the seed-33103 selected checkpoint, the same VALIDATION reset (`reset_index=0`, reset seed 123456) was reconstructed twice. All physical state inputs were identical; only the curriculum stage supplied to the actor changed from 2 to 5.

- maximum actor-observation difference: 1.0 (the one-hot stage coordinates);
- deterministic first-action L2 difference: 0.2410271913;
- maximum absolute action-coordinate difference: 0.1159560531;
- actions bitwise equal: false.

Seeds 33101 and 33102 had identical saved and supplied stages and produced identical observations and actions in this audit.

## Consequences

The seed-33103 validation-selected checkpoint and its 12% historical validation recovery remain results under the saved stage-2 actor contract. Its one-time 0% TEST result was generated under a different stage-5 actor input and is not a valid stage-matched estimate for that selected checkpoint.

The seed-33101 and seed-33102 TEST evaluations were stage-matched. Thus 0/200 stage-matched old-TEST episodes remain recorded, while the combined 0/300 result must carry this validity qualification.

The old TEST set may not be rerun. Corrected final-policy evaluation requires a new untouched holdout after the evaluator restores the saved checkpoint stage. Phase 3B-1A requires targeted evaluation revalidation, but it cannot be performed on the burned TEST cohort.

## Work intentionally not started

Because the protocol mandates an immediate stop when the mismatch can alter actions, this run did not:

- audit release-identity pairing beyond the already-known historical issue;
- generate ID-DEV, OOD-DEV, or NEW-FINAL-HOLDOUT;
- evaluate any controller on new DEV states;
- rerun any old TEST state;
- retrain PPO;
- change rewards, curriculum, actions, physics, or success criteria;
- generate Phase 3B-1A.5 figures or videos.
