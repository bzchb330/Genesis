from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import ROOT
from .phase3.control import actuator_target_from_qpos
from .phase3.env import load_keyframe_qpos
from .phase3b1a_env import CURRICULUM_STAGES, Phase3B1APrivilegedEnv


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


class Adam:
    def __init__(self, parameters: list[np.ndarray], learning_rate: float) -> None:
        self.parameters = parameters
        self.learning_rate = float(learning_rate)
        self.m = [np.zeros_like(value) for value in parameters]
        self.v = [np.zeros_like(value) for value in parameters]
        self.steps = 0

    def step(self, gradients: list[np.ndarray], maximum_norm: float) -> float:
        norm = float(np.sqrt(sum(np.sum(value * value) for value in gradients)))
        scale = min(1.0, float(maximum_norm) / max(norm, 1e-12))
        self.steps += 1
        for index, (parameter, gradient) in enumerate(zip(self.parameters, gradients)):
            gradient = gradient * scale
            self.m[index] = 0.9 * self.m[index] + 0.1 * gradient
            self.v[index] = 0.999 * self.v[index] + 0.001 * gradient * gradient
            m_hat = self.m[index] / (1.0 - 0.9**self.steps)
            v_hat = self.v[index] / (1.0 - 0.999**self.steps)
            parameter -= self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
        return norm


class MLP:
    def __init__(self, input_dimension: int, hidden_units: int, output_dimension: int, rng: np.random.Generator) -> None:
        self.w1 = rng.normal(0.0, np.sqrt(2.0 / input_dimension), (input_dimension, hidden_units))
        self.b1 = np.zeros(hidden_units)
        self.w2 = rng.normal(0.0, 0.01, (hidden_units, output_dimension))
        self.b2 = np.zeros(output_dimension)

    @property
    def parameters(self) -> list[np.ndarray]:
        return [self.w1, self.b1, self.w2, self.b2]

    def forward(self, values: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        hidden = np.tanh(values @ self.w1 + self.b1)
        return hidden @ self.w2 + self.b2, (values, hidden)

    def backward(self, gradient: np.ndarray, cache: tuple[np.ndarray, np.ndarray]) -> list[np.ndarray]:
        values, hidden = cache
        gw2 = hidden.T @ gradient
        gb2 = np.sum(gradient, axis=0)
        hidden_gradient = (gradient @ self.w2.T) * (1.0 - hidden * hidden)
        gw1 = values.T @ hidden_gradient
        gb1 = np.sum(hidden_gradient, axis=0)
        return [gw1, gb1, gw2, gb2]


@dataclass
class PPOCheckpoint:
    path: str
    environment_steps: int
    curriculum_stage: int
    validation_resource_recovery_rate: float
    validation_return: float


class NumPyPPO:
    """Small, dependency-free, auditable PPO with separate actor and critic MLPs."""

    def __init__(self, actor_dimension: int, critic_dimension: int, action_dimension: int, config: dict[str, Any], seed: int) -> None:
        self.config = config
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        hidden = int(config["hidden_units"])
        self.actor = MLP(actor_dimension, hidden, action_dimension, self.rng)
        self.critic = MLP(critic_dimension, hidden, 1, self.rng)
        self.log_standard_deviation = np.full(action_dimension, -0.75)
        self.actor_adam = Adam([*self.actor.parameters, self.log_standard_deviation], config["actor_learning_rate"])
        self.critic_adam = Adam(self.critic.parameters, config["critic_learning_rate"])

    def act(self, observation: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, float, np.ndarray]:
        values = np.atleast_2d(observation).astype(np.float64)
        mean, _ = self.actor.forward(values)
        latent = mean if deterministic else mean + np.exp(self.log_standard_deviation) * self.rng.normal(size=mean.shape)
        action = np.tanh(latent)
        log_probability = self._log_probability(latent, mean, action)
        return action[0].astype(np.float32), float(log_probability[0]), latent[0]

    def value(self, critic_observation: np.ndarray) -> float:
        value, _ = self.critic.forward(np.atleast_2d(critic_observation).astype(np.float64))
        return float(value[0, 0])

    def _log_probability(self, latent: np.ndarray, mean: np.ndarray, action: np.ndarray) -> np.ndarray:
        standard_deviation = np.exp(self.log_standard_deviation)
        gaussian = -0.5 * (((latent - mean) / standard_deviation) ** 2 + 2.0 * self.log_standard_deviation + np.log(2.0 * np.pi))
        jacobian = np.log(np.maximum(1e-6, 1.0 - action * action))
        return np.sum(gaussian - jacobian, axis=1)

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        advantages = batch["advantages"].copy()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        size = len(advantages)
        minibatch = int(self.config["minibatch_size"])
        clip_range = float(self.config["clip_range"])
        entropy_coefficient = float(self.config["entropy_coefficient"])
        maximum_gradient_norm = float(self.config["maximum_gradient_norm"])
        measurements: dict[str, list[float]] = {name: [] for name in ("policy_loss", "value_loss", "entropy", "kl_divergence", "clip_fraction", "actor_gradient_norm", "critic_gradient_norm")}
        for _ in range(int(self.config["update_epochs"])):
            for start in range(0, size, minibatch):
                indices = self.rng.permutation(size)[start : start + minibatch]
                observations = batch["actor_observations"][indices]
                critic_observations = batch["critic_observations"][indices]
                latent = batch["latent_actions"][indices]
                action = batch["actions"][indices]
                old_log_probability = batch["log_probabilities"][indices]
                selected_advantages = advantages[indices]
                mean, actor_cache = self.actor.forward(observations)
                log_probability = self._log_probability(latent, mean, action)
                ratio = np.exp(np.clip(log_probability - old_log_probability, -20.0, 20.0))
                clipped_ratio = np.clip(ratio, 1.0 - clip_range, 1.0 + clip_range)
                surrogate = np.minimum(ratio * selected_advantages, clipped_ratio * selected_advantages)
                policy_loss = -float(np.mean(surrogate))
                active = ~(((selected_advantages >= 0.0) & (ratio > 1.0 + clip_range)) | ((selected_advantages < 0.0) & (ratio < 1.0 - clip_range)))
                dloss_dlogp = -(ratio * selected_advantages * active) / len(indices)
                variance = np.exp(2.0 * self.log_standard_deviation)
                mean_gradient = dloss_dlogp[:, None] * (latent - mean) / variance
                actor_gradients = self.actor.backward(mean_gradient, actor_cache)
                logstd_gradient = np.sum(dloss_dlogp[:, None] * (((latent - mean) ** 2) / variance - 1.0), axis=0)
                logstd_gradient -= entropy_coefficient
                actor_norm = self.actor_adam.step([*actor_gradients, logstd_gradient], maximum_gradient_norm)
                self.log_standard_deviation[:] = np.clip(self.log_standard_deviation, -3.0, 0.5)

                predicted, critic_cache = self.critic.forward(critic_observations)
                residual = predicted[:, 0] - batch["returns"][indices]
                value_loss = 0.5 * float(np.mean(residual * residual))
                critic_gradient = residual[:, None] / len(indices)
                critic_norm = self.critic_adam.step(self.critic.backward(critic_gradient, critic_cache), maximum_gradient_norm)
                measurements["policy_loss"].append(policy_loss)
                measurements["value_loss"].append(value_loss)
                measurements["entropy"].append(float(np.sum(self.log_standard_deviation + 0.5 * np.log(2.0 * np.pi * np.e))))
                measurements["kl_divergence"].append(float(np.mean(old_log_probability - log_probability)))
                measurements["clip_fraction"].append(float(np.mean(np.abs(ratio - 1.0) > clip_range)))
                measurements["actor_gradient_norm"].append(actor_norm)
                measurements["critic_gradient_norm"].append(critic_norm)
        variance = float(np.var(batch["returns"]))
        explained_variance = 1.0 - float(np.var(batch["returns"] - batch["values"])) / variance if variance > 1e-12 else 0.0
        result = {name: float(np.mean(values)) for name, values in measurements.items()}
        result["explained_variance"] = explained_variance
        result["standard_deviation_mean"] = float(np.mean(np.exp(self.log_standard_deviation)))
        return result

    def save(self, path: str | Path, *, environment_steps: int, curriculum_stage: int) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            actor_w1=self.actor.w1,
            actor_b1=self.actor.b1,
            actor_w2=self.actor.w2,
            actor_b2=self.actor.b2,
            critic_w1=self.critic.w1,
            critic_b1=self.critic.b1,
            critic_w2=self.critic.w2,
            critic_b2=self.critic.b2,
            log_standard_deviation=self.log_standard_deviation,
            seed=self.seed,
            environment_steps=environment_steps,
            curriculum_stage=curriculum_stage,
        )
        return path

    def load(self, path: str | Path) -> dict[str, int]:
        with np.load(path) as payload:
            for name, value in (
                ("w1", payload["actor_w1"]), ("b1", payload["actor_b1"]),
                ("w2", payload["actor_w2"]), ("b2", payload["actor_b2"]),
            ):
                getattr(self.actor, name)[:] = value
            for name, value in (
                ("w1", payload["critic_w1"]), ("b1", payload["critic_b1"]),
                ("w2", payload["critic_w2"]), ("b2", payload["critic_b2"]),
            ):
                getattr(self.critic, name)[:] = value
            self.log_standard_deviation[:] = payload["log_standard_deviation"]
            return {"environment_steps": int(payload["environment_steps"]), "curriculum_stage": int(payload["curriculum_stage"])}


def collect_rollout(agent: NumPyPPO, env: Phase3B1APrivilegedEnv, steps: int, observation: np.ndarray | None = None, critic_observation: np.ndarray | None = None) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if observation is None or critic_observation is None:
        observation, info = env.reset()
        critic_observation = info["critic_observation"]
    storage: dict[str, list[Any]] = {name: [] for name in ("actor_observations", "critic_observations", "actions", "latent_actions", "log_probabilities", "values", "rewards", "terminals")}
    completed: list[dict[str, Any]] = []
    gamma = float(agent.config["gamma"])
    gae_lambda = float(agent.config["gae_lambda"])
    for _ in range(steps):
        action, log_probability, latent = agent.act(observation)
        value = agent.value(critic_observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        for name, value_to_store in (
            ("actor_observations", observation), ("critic_observations", critic_observation),
            ("actions", action), ("latent_actions", latent), ("log_probabilities", log_probability),
            ("values", value), ("rewards", reward), ("terminals", terminated),
        ):
            storage[name].append(value_to_store)
        if terminated or truncated:
            completed.append(info.get("episode_metrics", {}))
            next_observation, reset_info = env.reset()
            next_critic = reset_info["critic_observation"]
        else:
            next_critic = info["critic_observation"]
        observation, critic_observation = next_observation, next_critic
    batch = {name: np.asarray(values, dtype=np.float64) for name, values in storage.items()}
    next_value = agent.value(critic_observation)
    advantages = np.zeros(steps)
    last = 0.0
    for index in range(steps - 1, -1, -1):
        nonterminal = 1.0 - float(batch["terminals"][index])
        following = next_value if index == steps - 1 else batch["values"][index + 1]
        delta = batch["rewards"][index] + gamma * following * nonterminal - batch["values"][index]
        last = delta + gamma * gae_lambda * nonterminal * last
        advantages[index] = last
    batch["advantages"] = advantages
    batch["returns"] = advantages + batch["values"]
    return batch, observation, critic_observation, completed


def summarize_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        return {"episodes": 0}
    stage = np.asarray([episode.get("stage_ever", [0] * 6) for episode in episodes], dtype=float)
    def rate(name: str) -> float:
        return float(np.mean([bool(episode.get(name, False)) for episode in episodes]))
    numeric_names = (
        "return", "maximum_intended_penetration_m", "maximum_gross_penetration_m",
        "minimum_joint_margin_rad", "action_bound_hit_fraction", "stiffness_mean",
        "target_clip_hits", "contact_gap_count",
    )
    result: dict[str, Any] = {
        "episodes": len(episodes),
        "stage_rates": {name: float(stage[:, index].mean()) for index, name in enumerate(CURRICULUM_STAGES)},
        "palm_contact_rate": rate("palm_contact_achieved"),
        "alternate_support_rate": rate("alternate_support_achieved"),
        "resource_recovery_rate": rate("resource_recovered"),
        "thumb_recovery_rate": rate("thumb_recovered"),
        "index_recovery_rate": rate("index_recovered"),
        "object_retention_rate": rate("object_retained"),
        "table_drop_rate": rate("table_drop"),
        "gross_collision_rate": rate("gross_collision"),
    }
    for name in numeric_names:
        values = np.asarray([episode.get(name, np.nan) for episode in episodes], dtype=float)
        result[name] = {
            "mean": float(np.nanmean(values)),
            "median": float(np.nanmedian(values)),
            "p95": float(np.nanpercentile(values, 95)),
            "minimum": float(np.nanmin(values)),
            "maximum": float(np.nanmax(values)),
        }
    return _jsonable(result)


def evaluate_policy(agent: NumPyPPO | None, split: str, *, seed: int, curriculum_stage: int, policy: Callable[[np.ndarray, Phase3B1APrivilegedEnv, int], np.ndarray] | None = None, maximum_episodes: int | None = None, capture_trajectories: bool = False) -> dict[str, Any]:
    env = Phase3B1APrivilegedEnv(split, curriculum_stage=curriculum_stage)
    count = len(env.state_paths) if maximum_episodes is None else min(maximum_episodes, len(env.state_paths))
    episodes: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    for reset_index in range(count):
        observation, info = env.reset(seed=seed + reset_index, options={"reset_index": reset_index})
        trajectory = {"reset_index": reset_index, "candidate_id": info["candidate_id"], "samples": []}
        total_return = 0.0
        for step in range(int(env.pilot["episode"]["simulation_steps"])):
            action = policy(observation, env, step) if policy is not None else agent.act(observation, deterministic=True)[0]  # type: ignore[union-attr]
            observation, reward, terminated, truncated, info = env.step(action)
            total_return += reward
            if capture_trajectories and step % 5 == 0:
                trajectory["samples"].append({
                    "step": step, "action": np.asarray(action).tolist(),
                    "potentials": info["stage_potentials"].tolist(),
                    "stiffness": info["stiffness_scales"].tolist(),
                    "penetration": info["penetration"],
                    "stage_ever": info["stage_ever"].tolist(),
                    "contact_flags": info["contact_flags"].tolist(),
                    "normal_forces_n": info["normal_forces_n"].tolist(),
                    "support_load_fraction": info["support_load_fraction"].tolist(),
                    "relative_position_m": info["relative_position_m"].tolist(),
                    "linear_velocity_m_s": info["linear_velocity_m_s"].tolist(),
                    "angular_velocity_rad_s": info["angular_velocity_rad_s"].tolist(),
                    "minimum_joint_margin_rad": info["minimum_joint_margin_rad"],
                })
            if terminated or truncated:
                metrics = dict(info.get("episode_metrics", {}))
                metrics["return"] = total_return
                metrics["candidate_id"] = trajectory["candidate_id"]
                metrics["selected_release_finger"] = info["selected_release_finger"]
                episodes.append(metrics)
                trajectory["result"] = metrics
                break
        if capture_trajectories:
            trajectories.append(trajectory)
    env.close()
    return {"split": split, "seed": seed, "curriculum_stage": curriculum_stage, "summary": summarize_episodes(episodes), "episodes": _jsonable(episodes), "trajectories": _jsonable(trajectories)}


def zero_policy(_observation: np.ndarray, _env: Phase3B1APrivilegedEnv, _step: int) -> np.ndarray:
    return np.zeros(26, dtype=np.float32)


def random_policy(seed: int) -> Callable[[np.ndarray, Phase3B1APrivilegedEnv, int], np.ndarray]:
    rng = np.random.default_rng(seed)
    return lambda _observation, _env, _step: rng.uniform(-1.0, 1.0, 26).astype(np.float32)


def _actuator_action_toward(env: Phase3B1APrivilegedEnv, targets: np.ndarray, stiffness: float = 1.0) -> np.ndarray:
    delta = float(env.pilot["action"]["target_delta_limit_rad_per_control_step"])
    action = np.zeros(26, dtype=np.float32)
    action[:20] = np.clip((targets - env.desired) / delta, -1.0, 1.0)
    low = float(env.pilot["action"]["stiffness_scale_minimum"])
    high = float(env.pilot["action"]["stiffness_scale_maximum"])
    action[20:] = 2.0 * (float(stiffness) - low) / (high - low) - 1.0
    return np.clip(action, -1.0, 1.0)


def keyframe_policy(keyframe: str, *, open_selected_after: int | None = None, stiffness: float = 1.0) -> Callable[[np.ndarray, Phase3B1APrivilegedEnv, int], np.ndarray]:
    def policy(_observation: np.ndarray, env: Phase3B1APrivilegedEnv, step: int) -> np.ndarray:
        targets = actuator_target_from_qpos(env.scene, load_keyframe_qpos(keyframe))
        if open_selected_after is not None and step >= open_selected_after:
            open_targets = actuator_target_from_qpos(env.scene, load_keyframe_qpos("open hand"))
            targets[env.scene.actuator_ids[env.selected_release_finger]] = open_targets[env.scene.actuator_ids[env.selected_release_finger]]
        return _actuator_action_toward(env, targets, stiffness)
    return policy


def immediate_open_policy(_observation: np.ndarray, env: Phase3B1APrivilegedEnv, _step: int) -> np.ndarray:
    targets = env.desired.copy()
    open_targets = actuator_target_from_qpos(env.scene, load_keyframe_qpos("open hand"))
    for finger in ("thumb", "index"):
        targets[env.scene.actuator_ids[finger]] = open_targets[env.scene.actuator_ids[finger]]
    return _actuator_action_toward(env, targets)


def oscillating_policy(_observation: np.ndarray, _env: Phase3B1APrivilegedEnv, step: int) -> np.ndarray:
    return np.full(26, 1.0 if (step // 5) % 2 == 0 else -1.0, dtype=np.float32)


def palmward_losing_support_policy(_observation: np.ndarray, env: Phase3B1APrivilegedEnv, _step: int) -> np.ndarray:
    targets = actuator_target_from_qpos(env.scene, load_keyframe_qpos("open hand"))
    targets[env.scene.actuator_ids["wrist"]] = env.model.actuator_ctrlrange[env.scene.actuator_ids["wrist"], 1]
    return _actuator_action_toward(env, targets)


def scripted_handoff_policy(_observation: np.ndarray, env: Phase3B1APrivilegedEnv, step: int) -> np.ndarray:
    if step < 300:
        keyframe = "three finger pinch"
        open_after = None
    else:
        keyframe = "three finger pinch"
        open_after = 600
    return keyframe_policy(keyframe, open_selected_after=open_after)(_observation, env, step)


def phase3a_structured_return(path: str | Path = ROOT / "outputs/phase3A/contact_handoff.json") -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = payload["samples"]
    release_step = int(payload["fixture_release_state"]["step"])
    samples = [sample for sample in samples if int(sample["step"]) >= release_step]
    initial_distance = float(np.linalg.norm(np.asarray(samples[0]["object_position"]) - np.asarray(samples[0]["palm_position"])))
    stage = 0
    previous = 0.0
    total = 0.0
    trace = []
    alternate_seen = False
    released_persistence = 0
    completion = 0.95
    for sample in samples:
        flags = np.asarray(sample["contact_flags"], dtype=float)
        forces = np.asarray(sample["normal_forces_n"], dtype=float)
        fractions = np.asarray(sample["support_load_fraction"], dtype=float)
        retained = bool(np.any(flags) and not sample.get("floor_contact", False))
        valid_support = float(retained and np.sum(forces) >= 0.02)
        distance = float(np.linalg.norm(np.asarray(sample["object_position"]) - np.asarray(sample["palm_position"])))
        migration = valid_support * float(np.clip((initial_distance - distance) / 0.02, 0.0, 1.0))
        alternate = bool(np.any(flags[2:]))
        alternate_seen |= alternate
        support = valid_support * max(float(alternate), float(np.clip(np.sum(fractions[2:]) / 0.5, 0.0, 1.0)))
        selected_load = float(fractions[0])
        unload = support * float(np.clip((0.5 - selected_load) / 0.5, 0.0, 1.0))
        released = bool(alternate_seen and not flags[0] and retained)
        released_persistence = released_persistence + 1 if released else 0
        recovery = float(released) * float(np.clip(released_persistence / 20.0, 0.0, 1.0))
        potentials = np.asarray((valid_support, migration, support, unload, float(released), recovery))
        current = float(potentials[stage])
        total += current - previous
        previous = current
        if current >= completion:
            total += 1.0
            if stage < 5:
                stage += 1
                previous = float(potentials[stage])
            else:
                total += 5.0
                trace.append({"step": sample["step"], "stage": stage, "potentials": potentials.tolist()})
                break
        trace.append({"step": sample["step"], "stage": stage, "potentials": potentials.tolist()})
    return {"return": float(total), "final_stage": CURRICULUM_STAGES[stage], "resource_recovered": bool(stage == 5 and trace[-1]["potentials"][5] >= completion), "trace": trace}


def run_reward_exploit_test(output_path: str | Path = ROOT / "outputs/phase3B1A/reward_exploit_test.json", episodes_per_policy: int = 3) -> dict[str, Any]:
    policies = {
        "zero_action": zero_policy,
        "immediately_open_thumb_index": immediate_open_policy,
        "close_all_fingers_maximally": keyframe_policy("close hand"),
        "crushing_contact": keyframe_policy("two finger pinch", stiffness=1.0),
        "rapidly_oscillate_controls": oscillating_policy,
        "palmward_motion_losing_support": palmward_losing_support_policy,
    }
    results = {}
    for offset, (name, policy) in enumerate(policies.items()):
        evaluation = evaluate_policy(None, "validation", seed=33200 + offset * 10, curriculum_stage=0, policy=policy, maximum_episodes=episodes_per_policy)
        returns = [episode["return"] for episode in evaluation["episodes"]]
        results[name] = {"returns": returns, "mean_return": float(np.mean(returns)), "maximum_return": float(np.max(returns)), "summary": evaluation["summary"]}
    phase3a = phase3a_structured_return()
    maximum_pathological = max(value["maximum_return"] for value in results.values())
    margin = float(phase3a["return"] - maximum_pathological)
    payload = {
        "status": "PASS" if phase3a["resource_recovered"] and margin >= 2.0 else "PHASE3B1A_REWARD_EXPLOIT_DETECTED",
        "criterion": "Phase 3A structured return exceeds every pathological-policy return by at least 2.0",
        "phase3a": phase3a,
        "pathological_policies": results,
        "maximum_pathological_return": maximum_pathological,
        "return_margin": margin,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return payload
