# Phase 2CM contact-model rationale

## Question

Phase 2H showed that some object-B trials established the intended index–thumb contact topology but then lost the frozen strict retention state, frequently through rotation. Phase 2CM tests the narrower engineering hypothesis that the contact dimensionality represented by the existing MuJoCo fingertip collision primitives affects that post-release retention behavior.

This is an isolated contact-model audit and paired replay, not a controller search and not a new task definition. The existing Phase 2W trial IDs, wrist poses, object poses, trajectories, controller, timing, friction, mass, geometry, gravity, solver settings, and strict success gates remain frozen.

## Gate and variants

The actual compiled Phase 2W/2H model and actual runtime fingertip–B contacts are audited first. If those contacts already have dimension 4 or greater, the protocol stops with `PHASE2CM_BASELINE_ALREADY_TORSIONAL`, because a nominal 3-to-4 comparison would be misleading. If the baseline contact dimension is 3, the paired replay uses CM3, CM4, and CM6. Only the configured fingertip collision geoms' compiled `condim` values differ; all other compiled fields are checked programmatically for equality.

## Paired design

Eligible existing trials must have index and thumb contact at release, no middle or ring assistance at release, and a numerically valid release state. Selection is determined only by a SHA-256 ordering of trial IDs, balanced across the ten Phase 2W wrist candidates where the available eligible population permits. IDs are frozen before counterfactual outcomes are run. No survival, failure label, visual appearance, or future CM result enters selection.

Each exact pre-release state is reconstructed by replaying its existing trial. The complete MuJoCo integration state, controller command, desired joint target, object-B pose and velocity, and wrist pose are saved. CM3, CM4, and CM6 start from that same state and use the unchanged post-release controller commands for 500 steps. The existing Phase 2W strict criteria are applied without modification.

## Scope exclusions

Phase 2CM does not start reinforcement learning, tune friction, change object mass, modify fingertip or object geometry, modify the controller, redefine success, define a scalar resource metric `J`, or implement transfer.
