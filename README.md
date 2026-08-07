# SeqGrasp

SeqGrasp is a Python 3.10+/MuJoCo 3/Gymnasium research scaffold for retaining object A while reaching for object B with a fixed-base dexterous hand. It provides engineering infrastructure and diagnostics without defining PI-owned scientific criteria.

The architecture separates typed YAML configuration, MJCF composition, Gymnasium environment logic, contact/tactile sensing, torque controllers, reward/termination placeholders, diagnostics, visualization, and training adapters. The Allegro Hand V3 in `assets/hands/allegro` is vendored from MuJoCo Menagerie with its BSD-2-Clause license and provenance.

## Installation and validation

```bash
pip install -e .
python scripts/check_install.py
python scripts/run_random_policy.py
pytest -v
```

The random-policy smoke test uses seeded actions and completes 1000 environment steps. PPO integration is optional via `pip install -e .[train]`. **No PPO policy has been trained.**

## Scripted object-A diagnostic

```bash
python scripts/scripted_grasp_a.py
python scripts/scripted_grasp_a.py --seed 7 --video
python scripts/run_grasp_diagnostics.py --num-seeds 3
python scripts/check_scene_placements.py --seed 0 --render outputs/placements.png
```

The diagnostic logs CSV, compressed NPZ, JSON metadata, and separate plots under ignored `outputs/`. It records raw joint, command, object, contact, tactile, palm, phase, and termination signals. Video uses official `mujoco.Renderer`; rendering/encoding failures warn but do not invalidate physical logging.

`configs/diagnostic_grasp_a.yaml` is explicitly `diagnostic_only: true`. Its time schedule, fixture pose, joint-range fractions, and jitter are engineering probes—not scientific grasp thresholds, optimized control values, or evaluation criteria. The temporary kinematic fixture adapts the probe to a fixed-base hand; it is released for the retention-attempt trace.

## Rendering platforms

Windows uses the normal MuJoCo OpenGL backend. On Linux/headless systems select a backend before Python starts:

```bash
MUJOCO_GL=egl python scripts/scripted_grasp_a.py --video
MUJOCO_GL=osmesa python scripts/scripted_grasp_a.py --video
```

EGL requires a working GPU/EGL driver; OSMesa requires its system library. Videos and results are gitignored.

## Configuration and observation contract

All model, scene, task, diagnostic, and training settings live in typed YAML. Observation components expose stable name, dimension, unit, source, privileged status, and enabled state; see `docs/OBSERVATION_CONTRACT.md` and `docs/observation_spec.json`. Privileged target position is independently disabled by default.

To replace Allegro, update configured MJCF path, actuator/joint order, palm/fingertip bodies, geom mapping, DoF, mount, and diagnostic-only trajectory mapping. See `docs/HAND_SWAP_CHECKLIST.md`.

## Scientific boundary

Only binary contact and total normal force per finger are implemented tactile features. Raw newtons are preserved unless the PI later chooses a normalization. Reward terms and weights, metric J, retention law, phase/drop/success criteria, additional tactile physics, and RL protocol remain unresolved. See `docs/PI_DECISIONS.md` and `docs/NEXT_PI_ACTIONS.md`.
