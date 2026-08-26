# Phase 3B-0.5 PI Decision Update

`READY_FOR_PI_DECISION` means the requested empirical evidence now exists; it
does not adopt a threshold or scientific definition. All recommendations remain
nonbinding until the PI freezes them.

| Item | Status | Evidence produced |
|---|---|---|
| A3 | READY_FOR_PI_DECISION | passive and active retention horizons |
| A4 | READY_FOR_PI_DECISION | palm-relative translation traces |
| A5 | READY_FOR_PI_DECISION | total, D2-aware, angular-speed, and stabilization traces |
| A6 | READY_FOR_PI_DECISION | active/passive gap duration, motion, recontact, and retention |
| B1 | READY_FOR_PI_DECISION | pair-aware intended penetration distributions |
| B2 | READY_FOR_PI_DECISION | gross-contact distributions |
| B5 | READY_FOR_PI_DECISION | raw floor, numeric, topology, and retention events |
| C1 | READY_FOR_PI_DECISION | thumb/index release persistence at seven horizons |
| C2 | READY_FOR_PI_DECISION | joint/Jacobian envelope and three post-release motion probes |
| E2 | INSUFFICIENT_DATA | paired 0.25x/0.5x/1x/1.5x target-step trials |
| E3 | INSUFFICIENT_DATA | paired 1x/0.75x/0.5x/0.25x stiffness trials |
| E6 | INSUFFICIENT_DATA | paired 0.5x/1x/2x rate trials |

## Engineering options for PI review

- E2: [0.5, 1.0]x. **RECOMMENDATION ONLY - PI NOT YET FROZEN**. 0.5x-1.0x bracket the two highest observed retention fractions without the 1.5x joint-margin deterioration; zero complete handoffs prevents a final bound.
- E3: [0.75, 1.0]x. **RECOMMENDATION ONLY - PI NOT YET FROZEN**. 0.75x-1.0x retain more palm/support evidence than 0.25x; zero complete handoffs prevents freezing a lower limit.
- E6: [1.0]x. **RECOMMENDATION ONLY - PI NOT YET FROZEN**. 1.0x is the scripted reference and had the highest observed retention; neither slower nor faster rates established complete handoff.

## PPO readiness

**PPO_NOT_READY**

Remaining blockers:

- no expanded-reset active trial completed the full palm-contact handoff diagnostic, so E2/E3/E6 cannot be frozen from successful matched handoffs
- A3/A4/A5/A6/B1/B2/B5/C1/C2/E2/E3/E6 recommendations remain explicitly unfrozen pending PI decision
- the official pre-grasp keyframe starts several free-joint/tendon coordinates outside compiled limits and was not altered in this audit

No PPO code or run was started.
