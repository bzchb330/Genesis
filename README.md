# SeqGrasp

Research-code infrastructure for a fixed-base Allegro Hand to retain object A while grasping object B. It uses Python 3.10+, MuJoCo 3 official bindings, Gymnasium, typed YAML configuration, structured contact records, inspectable observations, and intentionally unchosen scientific placeholders.

The Allegro Hand V3 files in `assets/hands/allegro` are vendored from `google-deepmind/mujoco_menagerie/wonik_allegro` (retrieved 2026-08-07). Their upstream BSD-2-Clause `LICENSE` and `README.md` are preserved. Model path, actuators, joint/DoF metadata, fingertip bodies, and finger geom mappings live in `configs/hand_allegro.yaml`.

## Install and check

```bash
pip install -e .
python scripts/check_install.py
python scripts/run_random_policy.py
pytest
```

Headless Linux rendering is selected before Python starts:

```bash
MUJOCO_GL=egl python scripts/check_install.py
MUJOCO_GL=osmesa python scripts/check_install.py
```

EGL requires a working GPU/EGL driver; OSMesa requires its system library. `render_episode.py` writes ignored MP4s under `outputs/`. Install `.[train]` for the thin Stable-Baselines3 PPO entry points. All YAML paths are repository-relative; no working-directory assumption is made.

Observation layout is available as `env.observation_metadata`: each component has a stable name, dimension, and unit. Privileged target position is independently toggleable. Phases are integer values 0–4. Null PI thresholds disable the corresponding transition/drop rule while retaining workspace failure handling.

## Scientific boundary

Every unresolved scientific choice is explicitly marked for PI input. Reward weights default to zero. Only per-finger binary contact and summed normal force are implemented tactile features; no taxel model exists. Evaluation does not invent thresholds.
