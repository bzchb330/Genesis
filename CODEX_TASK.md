# Codex Task: Sequential Multi-Object Dexterous Grasping Scaffold

Implement the research-code scaffold specified in `codex_task_brief_seqgrasp.pdf` supplied by the PI. Treat the brief as authoritative. Do not invent scientific design decisions that are explicitly reserved for `TODO(PI)`.

## Scope

Build infrastructure only for sequential multi-object grasping with persistent retention: a dexterous hand grasps object A, retains A, then reaches for and grasps object B without dropping A. Initial development uses the MuJoCo Menagerie Allegro Hand as a swappable placeholder model.

## Non-negotiable constraints

- Python 3.10+.
- MuJoCo 3.x using the official `mujoco` Python bindings only.
- Never use `mujoco_py`, `MjSim`, `sim.step()`, or legacy access patterns.
- Use documented MuJoCo 3.x APIs. If uncertain, consult the official MuJoCo documentation before implementing.
- Gymnasium API only: `reset() -> (obs, info)` and `step(action) -> (obs, reward, terminated, truncated, info)`.
- Headless rendering must support `MUJOCO_GL=egl` and `MUJOCO_GL=osmesa` and be documented.
- All configuration lives in YAML and is loaded into typed dataclasses. Do not scatter magic numbers in Python.
- Git-clean repository: no absolute paths, notebook checkpoints, trained weights, videos, or generated artifacts committed.
- Do not choose reward weights, tactile-feature definitions beyond the two explicitly permitted reference features, or evaluation thresholds.

## Hand asset strategy

Use the official `google-deepmind/mujoco_menagerie` Allegro Hand MJCF as the development placeholder. Prefer vendoring the required Allegro files into `assets/hands/allegro/` for a self-contained install, and preserve/document the upstream BSD-2-Clause license and provenance. If a submodule is used instead, document why.

Hand model path, actuator names, fingertip body names, finger-to-geom mapping, and DoF count must all come from configuration so another hand can replace Allegro by YAML edits only.

## Required repository layout

```text
seqgrasp/
  README.md
  pyproject.toml
  configs/
    hand_allegro.yaml
    scene_two_object.yaml
    task_sequential.yaml
    train_ppo.yaml
  assets/
    hands/allegro/...
    objects/...
    scenes/two_object_table.xml
  seqgrasp/
    __init__.py
    config.py
    scene_builder.py
    env/
      __init__.py
      sequential_grasp_env.py
      observations.py
      rewards.py
      termination.py
    sensing/
      __init__.py
      contact.py
      tactile_features.py
    control/
      __init__.py
      base_controller.py
      residual.py
    viz/
      renderer.py
      viewer.py
    utils/
      logging.py
      seeding.py
  scripts/
    check_install.py
    run_random_policy.py
    render_episode.py
    train.py
    evaluate.py
  notebooks/
    01_inspect_scene.ipynb
  tests/
    test_scene_builds.py
    test_env_api.py
    test_contact_extraction.py
    test_determinism.py
```

## Scene builder

Programmatically compose the final MJCF/MjModel from the configured hand, fixed-base mount pose, table, and N configured rigid objects. The in-scope scene has exactly two objects: one cube and one cylinder. Geometry, size, mass, friction, and initial placement are config-driven. Support deterministic randomized object placement from a supplied seed.

## Environment

Implement a Gymnasium environment with these phases exposed in `info` as an integer:

- `0 APPROACH_A`
- `1 GRASP_A`
- `2 APPROACH_B`
- `3 GRASP_B`
- `4 HOLD`

Phase-transition conditions belong in `env/termination.py`. Early termination occurs on loss/drop of an already-held object or any object leaving the configured workspace. Any threshold requiring a scientific choice must remain `TODO(PI)` and config-backed.

Action space: continuous, one value per actuated joint. Support `direct_torque` and `residual` modes, defaulting to residual.

Observation space: flat `gymnasium.spaces.Box`, assembled from named, documented components that can be toggled in config. Include joint positions/velocities, low-dimensional tactile features, palm/end-effector pose, phase one-hot, and optionally privileged target-object position. Privileged simulation-only observations must be independently disableable.

Every observation component must expose a stable name, dimension, and unit so vector layout is inspectable rather than anonymous.

## Contact sensing

From `mujoco.MjData`, extract every contact each timestep. Resolve geom/body identities, compute the 6D contact wrench using documented `mujoco.mj_contactForce`, rotate force/torque into the world frame from the contact frame, and return structured records containing at least geom IDs/names, body IDs/names, contact position, normal, normal-force magnitude, and tangential-force magnitude.

Provide a helper to group contact records by finger according to a config-supplied mapping from finger name to geom names.

Do not simulate a taxel array.

## Tactile feature interface

Implement:

```python
def compute_tactile_features(contacts_by_finger, cfg) -> dict[str, np.ndarray]:
    ...
```

Requirements: stable ordering, fixed output dimensions, correct zero-contact handling, normalization hook, documented dimensions/units.

Only these two reference features may be fully implemented now:

1. binary contact flag per finger;
2. total normal force per finger.

Leave other candidate features as clearly documented `TODO(PI)` stubs that return correctly shaped zeros. Do not define their physics.

## Controllers

`base_controller.py`: joint-space impedance controller mapping desired joint configuration to torque with configurable stiffness, damping, and torque saturation. It is the safety baseline.

`residual.py`: add a bounded/scaled policy residual to base-controller torque; limits are config-driven.

## Reward scaffold

Implement the machinery and logging, not the scientific reward design. `compute_reward(state, cfg)` must return `(total_reward, per_term_breakdown)`.

Include structural placeholder terms for retention, phase-dependent task progress, resource term `J`, action/energy regularization, and failure penalty. Keep unresolved weights at `0.0` in YAML with comments and mark the bodies `TODO(PI)` where appropriate. Do not infer weights or resource metric definitions.

## Scripts

- `check_install.py`: print Python, MuJoCo, Gymnasium versions; report active headless backend; load configs; build the scene; instantiate/reset/check the env; optionally render one offscreen frame when backend supports it; return a useful diagnostic exit status.
- `run_random_policy.py`: run 1000 environment steps with seeded random actions and print observation metadata and reward-term breakdowns.
- `render_episode.py`: run random or loaded policy and write MP4; output files must be gitignored.
- `train.py`: PPO entry point using one maintained library (Stable-Baselines3 preferred unless compatibility blocks it). Keep integration thin and swappable; support vectorized envs, seeding, checkpointing, TensorBoard. Do not tune or train a policy as part of this task.
- `evaluate.py`: load a checkpoint, run N seeded episodes, report success rate, drop rate, and per-phase failure counts. Any scientific success/drop threshold not specified in the brief must remain config-backed `TODO(PI)` rather than invented.

## Notebook

`01_inspect_scene.ipynb` is inspection only: build the scene, render a few offscreen frames using `mujoco.Renderer`, and display with `mediapy`. Explicitly state that interactive `mujoco.viewer.launch()` belongs in `viz/viewer.py` and should not be called from the notebook.

## Tests

Implement and make pass:

1. every config builds its scene without error;
2. environment passes `gymnasium.utils.env_checker.check_env`;
3. fixed seed + fixed action sequence produces identical trajectories;
4. contact extraction returns nonzero normal force for a scripted closing/contact motion and zero when the hand is open/far away;
5. observation-vector dimension equals the sum of declared component dimensions.

Tests should exercise real MuJoCo objects and the official API, not mocks for the core physics path.

## Required execution loop

Work iteratively. After implementation, execute and debug until the implementable requirements pass:

```bash
pip install -e .
python scripts/check_install.py
python scripts/run_random_policy.py
pytest
```

Do not mask failures simply to satisfy tests. If a requirement cannot be completed because it depends on a `TODO(PI)` scientific decision, leave an explicit documented stub and keep unrelated infrastructure operational.

## Final report

At completion, provide:

- architecture summary;
- exact commands executed and their results;
- any limitations/known issues;
- a table of every remaining `TODO(PI)` with file, line, current placeholder behavior, and the PI decision required;
- confirmation that no reward weights, tactile definitions, or evaluation thresholds were invented.

## Explicit non-goals

Do not tune hyperparameters. Do not train a working policy. Do not define unspecified reward weights. Do not implement tactile features beyond the two reference features. Do not simulate taxels. Do not add vision/object detection. Do not add a third object. Do not implement finger-gating scientific logic. Keep architecture extensible to N objects, but only two objects are in scope now.
