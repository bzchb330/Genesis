# Phase 2.5 second-grasp acquisition calibration result

## Termination status

`PHASE2_5_B_ONLY_CONTROL_FAILED`

The required B-only control searched 2,048 deterministic Latin-hypercube trajectory candidates. None satisfied the frozen functional acquisition conjunction for the complete 500-step unsupported hold. Per the preregistered stop rule, A-held+B search, population calibration, controller freeze, formal-v2 execution, correlation statistics, and J evidence generation were not started.

## Preserved Phase 2 result

The original 4,540-trial result remains separate and unchanged: `BOTH_RETAINED=0`, `A_DROPPED=0`, `B_NOT_ACQUIRED=2700`, `BOTH_LOST=1834`, and `INVALID=6`. Its 0% positive-class rate made logistic inference unidentifiable. This is an engineering negative result, not evidence against the resource hypothesis.

## B-only control

The geometry-only positive-control pose was `[0.060, 0.115, 0.215] m`, inside the unchanged Phase 2 box. A 50,000-sample-per-finger collision-distance audit found near-surface samples only for index and thumb; no dynamic outcome was used to select the pose or contact anchors.

Failure counts after 2,048 candidates were:

- `CONTACT_LOST_IMMEDIATELY_AFTER_RELEASE`: 1,076
- `B_SLIPPED_TO_TABLE`: 567
- `NO_B_CONTACT_BEFORE_RELEASE`: 402
- `CONTACT_FORCE_TOO_LOW`: 2
- `SINGLE_UNOPPOSED_CONTACT`: 1

The lexicographically highest candidate was 254. It maintained some unsupported hand contact for 148/500 steps, first lost all hand contact 142 steps after release, contacted the table, translated 0.344866 m, rotated 3.136238 rad, and reached 0.025307 m maximum penetration. It therefore failed several frozen criteria and is not a positive control. It had no actuator saturation and remained numerically finite.

## Diagnostic artifacts

Candidate 254 was rerun with a complete log and an exact 700-step focus window (200 steps before release and 500 after):

- `outputs/phase2_5/diagnostics/b_only_candidate_0254/fixture_release_window.csv`
- `outputs/phase2_5/diagnostics/b_only_candidate_0254/fixture_release_window.npz`
- `outputs/phase2_5/diagnostics/b_only_candidate_0254/fixture_release_diagnostic.pdf`
- `outputs/phase2_5/diagnostics/b_only_candidate_0254/representative_failure.mp4`

The 140-frame video rendered successfully. These generated artifacts and all raw search records are ignored by Git.

## Freeze and separation audit

The MuJoCo model, geometry, masses, friction, contact solver parameters, timestep, gravity, controller gains, actuator and joint limits, tactile definitions, penetration threshold, and retention/acquisition thresholds were not changed. The formal B distribution remains x=`[0.055,0.065]`, y=`[0.115,0.125]`, z=`[0.215,0.225]` m with a vertical cylinder and yaw in `[0,2*pi)`. Phase 2.5 output uses experiment ID `phase2_5_calibration` and a config-hash-scoped resumable store; it is not mixed with `phase2_original_zero_success`.

No scalar J was defined and no RL, reward tuning, policy training, physics tuning, calibration population, or formal-v2 inference was performed.
