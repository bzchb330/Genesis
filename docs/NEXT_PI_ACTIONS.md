# Next PI Actions

Only scientific decisions are listed here. The current diagnostics provide raw evidence but assign no success label.

## Priority 1: Scientific definition of a valid single-object grasp / retention event

- **Question:** Which contacts, object motion, duration, and loss conditions define grasp and retention?
- **Waiting code:** `env/termination.py`, `configs/task_sequential.yaml`, `scripts/evaluate.py`.
- **Evidence available:** time histories of object pose/velocity, contacts, per-finger normal force, and release/hold behavior.
- **Blocked:** scientific single-object metrics, drop labels, and sequential progression.

## Priority 2: Definition of resource-awareness metric J

- **Question:** What physical quantity or combination defines J, with what units and aggregation?
- **Waiting code:** `env/resource.py`, `env/rewards.py`.
- **Evidence available:** raw joints, commands/limits, contacts, tactile references, phase, and object poses through `ResourceState`.
- **Blocked:** resource-aware reward/evaluation.

## Priority 3: Final tactile feature set / normalization choices

- **Question:** Keep raw newtons or select a normalization scale; add which scientifically justified features, if any?
- **Waiting code:** `sensing/tactile_features.py`, task YAML, observation contract.
- **Evidence available:** deterministic binary contact and raw total-normal-force traces by finger.
- **Blocked:** final policy observation design and sim-to-real scaling.

## Priority 4: Closed-loop tactile retention strategy

- **Question:** What controller law should turn existing tactile/joint/phase signals into residual torque?
- **Waiting code:** `control/retention.py`.
- **Evidence available:** open-loop scripted release/hold traces and controller saturation logs.
- **Blocked:** active persistent retention.

## Priority 5: Reward term definitions and weights

- **Question:** Define retention, phase progress, J, regularization, and failure terms, then their weights.
- **Waiting code:** `env/rewards.py`, task YAML.
- **Evidence available:** per-timestep raw state and structurally separate zero-valued reward breakdown.
- **Blocked:** scientifically meaningful RL optimization.

## Priority 6: Phase-transition and evaluation criteria

- **Question:** What triggers each of the five transitions, and what defines success/drop for evaluation?
- **Waiting code:** `env/termination.py`, `scripts/evaluate.py`, task YAML.
- **Evidence available:** phase/reason instrumentation and diagnostic stage transition logs.
- **Blocked:** autonomous sequential episodes and benchmark rates.

## Priority 7: RL training protocol and hyperparameters

- **Question:** After objectives are approved, which protocol, budgets, curricula, seeds, and hyperparameters should be used?
- **Waiting code:** `configs/train_ppo.yaml`, `scripts/train.py`.
- **Evidence available:** validated environment/controller/sensing engineering baseline; no trained checkpoint.
- **Blocked:** policy training and comparative claims.
