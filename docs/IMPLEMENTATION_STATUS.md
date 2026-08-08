# Implementation Status

## A. Fully functional

- Typed YAML loading, configured Menagerie hand composition, cube/cylinder scene, deterministic placement.
- Gymnasium reset/step API, inspectable observations, official MuJoCo stepping and contact extraction.
- Binary per-finger contact and total normal-force reference tactile features.
- Joint impedance and residual torque controllers, saturation, dimension and finite-value validation.
- Raw physical logging, plotting, scene placement checks, deterministic tests, offscreen renderer integration.

## B. Functional engineering scaffold

- Five-phase state machine with explicit phase/reason instrumentation but no PI transition rules.
- Reward breakdown and YAML weights with zero scientific terms.
- `RetentionController` protocol with a zero-output implementation.
- Expanded raw `ResourceState`, deterministic reachability/contact probes, and undefined `compute_resource_metric` hook.
- Separate nullable acquisition, unsupported-retention, and loss criterion interfaces.
- Optional Stable-Baselines3 PPO entry points; integration is present but untrained.

## C. Diagnostic-only

- `diagnostic_grasp_a.yaml` time schedule, joint-range fractions, kinematic object fixture, and fixture jitter.
- Scripted object-A pregrasp/closing/contact/release/hold probe.
- Multi-seed raw-measurement runner, plots, optional MP4, and placement renderer.
- Explicit support-release samples, descriptive 10-seed statistics, and aggregate figures.
- Reproducible engineering-only grasp-posture search, top-candidate profiles, object-specific contact mechanics, and 20-seed descriptive validation.
- Fixed-physics penetration refinement, three source-distinct resource profiles, 3,000 raw post-release samples, 384 reachability samples, and 240 independent single-finger B probes.
- Phase 2 fixed-base physics replay, 1000-step raw hold report, compiled contact-parameter override interface, and crash-tolerant resumable JSONL infrastructure.
- These values and outputs are not success criteria or scientific thresholds.

## D. Blocked by TODO(PI)

- Automatic sequential phase transitions, scientific drop/success decisions, resource metric J.
- Closed-loop tactile retention law, reward definitions/weights, additional tactile features/normalization.
- Scientific RL evaluation and training protocol.
- Phase 2 physics gate and sweep selection pending the consolidated PI inputs; Parts B-F are intentionally not started before that hard gate passes.

## E. Not yet implemented

- A trained policy, tuned hyperparameters, learned retention, real-hardware transport, vision, taxels, or a third object.
- Scientifically validated grasping performance. The diagnostic establishes engineering signal flow only.
