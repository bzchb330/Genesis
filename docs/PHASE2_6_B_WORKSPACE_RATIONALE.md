# Phase 2.6 B-workspace rationale

The original Phase 2 B region was calibrated using single-fingertip geometric reachability. That preflight was nondegenerate, but it did not establish simultaneous access by multiple supporting contacts.

Phase 2.5 supplied the missing engineering control. At its geometry-selected B-only pose, a deterministic 50,000-sample-per-finger audit found near-surface access only for the index and thumb; middle and ring had none. A structured search of 2,048 trajectory candidates then produced zero acquisitions that survived the complete 500-step unsupported hold. The dominant mechanisms were immediate post-release contact loss and slip to the table.

Phase 2.6 therefore redesigns the presentation region using multi-contact graspability rather than single-fingertip reachability. The redesign is performed with B alone, before any new A+B formal outcomes exist. It uses no resource components, correlation strength, scalar J, or reinforcement learning and is not outcome-dependent resource tuning.

All Phase 2 and Phase 2.5 raw datasets, reports, plots, and videos remain separate historical negative controls. Phase 2.6 preserves the validated MuJoCo model, hand and object geometry, mass, friction, gravity, contact solver settings, timestep, controller gains, actuator and joint limits, table, penetration threshold, and fixture mechanism.
