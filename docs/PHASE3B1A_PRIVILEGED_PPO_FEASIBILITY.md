# Phase 3B-1A privileged PPO feasibility pilot

## Status

Phase 3B-1A completed the frozen 3-seed, 900,000-step single-object pilot. By the predeclared first-matching classification rule it is **PPO-A**: every validation-selected policy repeatedly produced `RESOURCE_RECOVERED` on held-out validation states (9%, 10%, and 12%). This is a privileged-actor feasibility baseline, not a deployable policy.

The stronger generalization result is negative: the one-time pose-disjoint TEST pass produced 0/100 resource recoveries for each seed. The scripted validation baseline reached 15%, zero action 10%, and random bounded action 9%; therefore the pilot does not establish that PPO outperforms these controls.

## Fixed protocol

- Branch: `codex/phase3B1A-privileged-ppo-feasibility`
- Base commit: `3dedfcc8ddc82f1b8b7f9458b382616ae35be38a`
- Fresh resets: 300 TRAIN / 100 VALIDATION / 100 TEST, with zero ID and state-hash overlap
- Privileged actor: 131 dimensions
- Privileged critic: 140 dimensions
- Action: 26 dimensions (20 bounded target increments + 6 stiffness scales)
- Target displacement/rate cap: 0.0005 rad per control step
- Stiffness: [0.75, 1.0]
- Horizon: 1,000 control/simulation steps = 2.0 s
- PPO: in-repository NumPy implementation, separate 64-unit actor and critic MLPs
- Seeds: 33101, 33102, 33103
- Budget: 300,000 steps per seed; evaluation every 25,000 steps

## Pre-training gates

The feasible-projection sanitation gate passed. The pre-training reward exploit audit exercised zero action, immediate opening, maximal closure, crushing contact, rapid oscillation, and palmward motion while losing support. The recorded Phase 3A trajectory received 15.5698023 structured return; the maximum pathological return was 6.0, a margin of 9.5698023. The fixed-seed 10,000-step smoke run passed with 10 PPO updates, finite diagnostics, deterministic checkpoint evaluation, safe intended penetration, and no catastrophic joint excursion.

## Curriculum outcome

Seed 33101 reached RECOVER at 125,000 steps. Seed 33102 reached SUPPORT at 75,000 steps but never met its 20% validation transition criterion, so it remained there through 300,000 steps. Seed 33103 reached SUPPORT at 125,000, UNLOAD at 150,000, and RECOVER at 175,000 steps. No stage was skipped.

The frozen validation-only checkpoint rule was lexicographic: resource recovery, RECOVER, RELEASE, UNLOAD, SUPPORT, MIGRATE, retention, then mean return.

| Seed | Selected step | RETAIN | MIGRATE | SUPPORT | UNLOAD | RELEASE | RECOVER | Thumb | Index |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 33101 | 275,000 | 100% | 27% | 24% | 24% | 24% | 9% | 3% | 6% |
| 33102 | 175,000 | 100% | 23% | 16% | 16% | 16% | 10% | 6% | 4% |
| 33103 | 125,000 | 100% | 24% | 20% | 20% | 20% | 12% | 5% | 7% |

A labeled post-selection VALIDATION diagnostic (not used for selection) measured true palm-contact rates of 14%, 12%, and 16%, and any-alternate-support rates of 24%, 14%, and 19% for seeds 33101-33103.

## Safety and control diagnostics at selected validation checkpoints

| Seed | Retained | Table drop | Gross collision | Intended penetration median / p95 / max (mm) | Action-bound hits | Mean stiffness | Median target clips | Minimum joint margin (rad) | Median contact gaps |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 33101 | 10% | 12% | 48% | 0.712 / 0.849 / 1.114 | 0% | 0.8849 | 69.0 | -0.006526 | 5 |
| 33102 | 12% | 11% | 53% | 0.722 / 0.849 / 1.115 | 0% | 0.8660 | 52.5 | -0.006526 | 5 |
| 33103 | 15% | 16% | 44% | 0.725 / 0.853 / 1.234 | 0% | 0.8773 | 123.5 | -0.006526 | 5 |

Intended-grip penetration stayed well below the unchanged 3 mm ceiling. Gross-collision rates are terminal non-grip penetrations above the unchanged 3 mm ceiling, not any nonzero solver overlap. Negative joint margins remained inside the unchanged -0.02 rad catastrophic excursion limit. Actuator target clipping remained frequent even though policy actions did not saturate; this distinction is preserved in the logs.

## Baselines and TEST

On VALIDATION, `RESOURCE_RECOVERED` was 10% for zero action, 15% for the deterministic scripted handoff, and 9% for random bounded action. Selected PPO checkpoints achieved 9%, 10%, and 12%.

The TEST cohort was untouched until all three 300,000-step runs and validation-only selections were complete. Each selected checkpoint was then evaluated exactly once:

| Seed | TEST MIGRATE | TEST SUPPORT | TEST RECOVER | TEST retention |
|---:|---:|---:|---:|---:|
| 33101 | 14% | 1% | 0% | 10% |
| 33102 | 8% | 0% | 0% | 10% |
| 33103 | 6% | 0% | 0% | 11% |

The phase therefore demonstrates in-distribution privileged control feasibility but no pose-disjoint resource-recovery generalization.

## Evaluation validity note

The original evaluation payload used the SUPPORT indicator as a palm-contact proxy. Post-selection VALIDATION diagnostics corrected this by logging palm identity separately. TEST was not replayed, so identity-specific TEST palm-contact rate is unavailable; the stored TEST SUPPORT rates remain exact.

The seed-33103 selected checkpoint was saved while the curriculum observation was SUPPORT, while the one-time TEST runner supplied the final reached RECOVER stage indicator. This stage-indicator mismatch is preserved and reported; TEST was not rerun because the protocol permits exactly one pass. It is an evaluation-validity blocker for using the seed-33103 TEST result as a calibrated stage-matched estimate.

Checkpoint-time validation used every frozen validation state, but its deterministic RNG seed included the checkpoint step. Consequently, the randomly selected release-finger identity was deterministic but not paired across checkpoints. The post-selection fixed-seed diagnostic still found repeated recovery for all three selected policies (8%, 6%, and 11%), so the qualitative PPO-A observation persists, but the original validation curves and selection comparisons are not release-identity-paired. Future work must freeze the per-state release identity before training.

## Decision

Do not start Phase 3B-1B yet. The immediate blocker is 0/300 combined pose-disjoint TEST resource recoveries, together with no PPO advantage over the scripted baseline, the seed-33103 stage-indicator mismatch, and non-paired checkpoint-time release identities. The next authorized experiment should first address reset-distribution generalization and freeze a stage-matched, release-identity-paired evaluation contract. This recommendation does not change any Phase 3B-1A criterion or result.

Machine-readable results are under `outputs/phase3B1A/` and checkpoints under `checkpoints/phase3B1A/`; both are generated and ignored. Videos under `videos/phase3B1A/` are generated and ignored. One requested category, support achieved without release, was not observed in the selected validation diagnostic cohort and no video was fabricated.
