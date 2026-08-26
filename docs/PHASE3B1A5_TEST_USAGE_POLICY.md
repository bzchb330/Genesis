# Phase 3B-1A.5 TEST usage policy

## Status

`BURNED_FINAL_TEST_PHASE3B1A`

The original Phase 3B-1A TEST cohort was evaluated once after all three training runs and validation-only checkpoint selections completed. It is permanently burned for development use.

## Prohibited uses

The original TEST reset IDs and their outcomes must not be used for:

- hyperparameter or checkpoint selection;
- reward or curriculum modification;
- reset-range, observation, or architecture design;
- debugging controller performance;
- rerunning PPO, scripted, zero-action, or random controllers;
- selecting examples, trajectories, thresholds, or future training changes.

Only already-recorded aggregate and state-level results may be analyzed. No simulator policy replay is permitted on these states.

## Permitted use

The frozen reset metadata may be read without stepping the simulator to characterize the initial-state geometry and observation distribution. Already-recorded TEST results may be joined to those descriptors. Such analysis must retain the label `BURNED_FINAL_TEST_PHASE3B1A` and must not be used as a development objective.

## Evaluation-validity qualification

The Phase 3B-1A.5 stage-restoration audit found that the selected seed-33103 checkpoint was saved and selected at curriculum stage 2 (`SUPPORT`), but the one-time TEST evaluator supplied stage 5 (`RECOVER`) to the actor. Because curriculum stage is part of the 131-dimensional actor observation, the seed-33103 TEST result is not a stage-matched evaluation of the validation-selected policy input contract.

The old TEST set remains burned despite this issue. It must not be rerun to repair the result. Any corrected final evaluation must use a newly frozen holdout after an evaluator-restoration fix and renewed PI authorization.

## Source artifacts

- Split manifest: `outputs/phase3B1A/resets/split.json` (generated and ignored)
- Historical evaluation: `outputs/phase3B1A/pilot_results.json` (generated and ignored)
- Seed-33103 checkpoint: `checkpoints/phase3B1A/seed_33103/step_0125000.npz` (generated and ignored)

This policy is a hard protection boundary for all subsequent Phase 3 development.
