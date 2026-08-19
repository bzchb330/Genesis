# Phase 2R rationale: palmar-secured versus fingertip endpoint states

The original Phase 2 experiment, the Phase 2.5 B-only failure diagnosis, and
the Phase 2.6 B-graspable-workspace study remain preserved as separate
negative/engineering evidence. Those experiments evaluated acquisition of B
while A remained in its original fingertip acquisition-state grasp.

Phase 2R tests the narrower endpoint-state hypothesis: does a physically
palmar-secured A state preserve more capability for a subsequent acquisition
than a fingertip acquisition state under the same B distribution and the same
generic B controller?

The two endpoint states are sampled or replayed directly. No in-hand transfer,
sliding, finger gaiting, contact-switching policy, wrist policy, or
gravity-assisted transfer controller is simulated. Direct initialization is
used only to isolate the effect of the established endpoint grasp state on
future manipulation capability.

A temporary free-joint pose fixture is permitted only while the retaining
fingers close around a candidate palmar state. It is removed before the stable
measurement window. Accepted states use the unchanged MuJoCo dynamics,
gravity, friction, palm/finger contact, a free object joint, and zero equality
constraints. They may not use table support. The model, geometry, masses,
contact parameters, timestep, gravity, limits, gains, and validity thresholds
remain the validated Phase 2 baseline.

This experiment does not demonstrate the control process required to transfer
an object from fingertip acquisition to palmar secure storage. It evaluates
whether the post-transfer endpoint state provides greater capability for
subsequent acquisition.

No scalar resource objective is defined. `compute_J(...)` remains a TODO(PI)
scientific decision.
