from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import yaml

from .config import ROOT
from .phase3.config import FINGERS, SUPPORT_SURFACES
from .phase3.contacts import extract_shadow_contacts, object_velocity
from .phase3.model import build_shadow_scene, set_fixture
from .phase3.resource import compute_resource_snapshot
from .phase3.roles import RoleState
from .phase3b0 import (
    _joint_margins,
    contact_records,
    pair_aware_penetration,
    palm_relative_pose,
    restore_release_state,
)


CURRICULUM_STAGES = ("RETAIN", "MIGRATE", "SUPPORT", "UNLOAD", "RELEASE", "RECOVER")
PENETRATION_LABELS = (
    "thumb_object",
    "index_object",
    "other_finger_object",
    "palm_object",
    "table_object",
    "other_object",
    "maximum_intended_grip",
    "maximum_gross_non_grip",
)


def load_phase3b1a_config() -> dict[str, Any]:
    with (ROOT / "configs/phase3B1A_ppo.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


@dataclass(frozen=True)
class ObservationContract:
    actor_components: tuple[tuple[str, int], ...] = (
        ("joint_positions", 24),
        ("joint_velocities", 24),
        ("wrist_state", 4),
        ("contact_flags", 6),
        ("normal_forces", 6),
        ("previous_action", 26),
        ("object_pose_relative_to_palm", 7),
        ("object_velocity", 6),
        ("support_load_fraction", 6),
        ("contact_topology_counts", 6),
        ("pair_aware_penetration", 8),
        ("selected_release_finger", 2),
        ("curriculum_stage", 6),
    )
    critic_extra_components: tuple[tuple[str, int], ...] = (
        ("all_stage_potentials", 6),
        ("acquisition_available_motion", 2),
        ("normalized_episode_time", 1),
    )

    @property
    def actor_dimension(self) -> int:
        return sum(dimension for _, dimension in self.actor_components)

    @property
    def critic_dimension(self) -> int:
        return self.actor_dimension + sum(dimension for _, dimension in self.critic_extra_components)


OBSERVATION_CONTRACT = ObservationContract()


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


class Phase3B1APrivilegedEnv(gym.Env):
    """Single-object privileged-actor feasibility environment."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 500}

    def __init__(
        self,
        split: str = "train",
        *,
        split_path: str | Path = ROOT / "outputs/phase3B1A/resets/split.json",
        curriculum_stage: int = 0,
        render_mode: str | None = None,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        self.pilot = load_phase3b1a_config()
        self.scene = build_shadow_scene()
        self.model, self.data = self.scene.model, self.scene.data
        self.split_name = split
        self.split_path = _resolve(split_path)
        split_payload = json.loads(self.split_path.read_text(encoding="utf-8"))
        self.state_paths = tuple(_resolve(path) for path in split_payload["state_paths"][split])
        self.candidate_ids = tuple(int(value) for value in split_payload[split])
        if not self.state_paths:
            raise ValueError(f"empty {split} reset split")
        self.curriculum_stage = int(curriculum_stage)
        if not 0 <= self.curriculum_stage < len(CURRICULUM_STAGES):
            raise ValueError("invalid curriculum stage")
        self.render_mode = render_mode
        self._renderer: mujoco.Renderer | None = None
        self._camera: mujoco.MjvCamera | None = None
        self._base_gain = self.model.actuator_gainprm[:, 0].copy()
        self._base_bias = self.model.actuator_biasprm[:, 1].copy()
        self.action_space = spaces.Box(-1.0, 1.0, (26,), np.float32)
        self.observation_space = spaces.Box(-10.0, 10.0, (OBSERVATION_CONTRACT.actor_dimension,), np.float32)
        self.critic_observation_space = spaces.Box(-10.0, 10.0, (OBSERVATION_CONTRACT.critic_dimension,), np.float32)
        self.previous_action = np.zeros(26, dtype=np.float64)
        self.desired = np.zeros(self.model.nu, dtype=np.float64)
        self.steps = 0
        self.selected_release_finger = "thumb"
        self.initial_palm_distance = 0.0
        self.previous_stage_potential = 0.0
        self.stage_bonus_given = False
        self.alternate_support_seen = False
        self.release_persistence_steps = 0
        self.stage_ever = np.zeros(6, dtype=np.int8)
        self.event_steps: dict[str, int] = {}
        self.gap_start: int | None = None
        self.contact_gaps: list[dict[str, Any]] = []
        self._last_relative_position = np.zeros(3)
        self._last_linear_velocity = np.zeros(3)
        self._episode_metrics: dict[str, Any] = {}

    def set_curriculum_stage(self, stage: int) -> None:
        if not 0 <= stage < len(CURRICULUM_STAGES):
            raise ValueError("invalid curriculum stage")
        self.curriculum_stage = int(stage)

    @property
    def physical_episode_duration_s(self) -> float:
        episode = self.pilot["episode"]
        return float(episode["simulation_steps"] * episode["control_decimation"] * episode["timestep_s"])

    def _apply_stiffness(self, scales: np.ndarray) -> None:
        for scale, group in zip(scales, ("wrist", *FINGERS)):
            ids = self.scene.actuator_ids[group]
            self.model.actuator_gainprm[ids, 0] = self._base_gain[ids] * scale
            self.model.actuator_biasprm[ids, 1] = self._base_bias[ids] * scale

    def _available_motion(self, finger: str) -> float:
        ids = self.scene.joint_ids[finger]
        qpos = self.data.qpos[self.model.jnt_qposadr[ids]]
        limits = self.model.jnt_range[ids]
        return float(np.sum(np.maximum(0.0, np.minimum(qpos - limits[:, 0], limits[:, 1] - qpos))))

    def _state_features(self) -> dict[str, Any]:
        contacts = extract_shadow_contacts(self.scene)
        records = contact_records(self.scene, contacts)
        penetration = pair_aware_penetration(records)
        relative_position, relative_quaternion = palm_relative_pose(self.scene)
        linear, angular = object_velocity(self.scene)
        topology = np.asarray(
            [len(contacts.records_by_surface[surface]) for surface in SUPPORT_SURFACES], dtype=np.float64
        )
        table_contact = any(record["surface"] == "table" for record in records)
        selected_index = FINGERS.index(self.selected_release_finger)
        alternative_indices = [index for index in range(6) if index != selected_index]
        alternate_support = bool(np.any(contacts.contact_flags[2:]))
        selected_free = not bool(contacts.contact_flags[selected_index])
        available = np.asarray([self._available_motion("thumb"), self._available_motion("index")])
        finite = bool(
            np.all(np.isfinite(self.data.qpos))
            and np.all(np.isfinite(self.data.qvel))
            and np.all(np.isfinite(self.data.ctrl))
        )
        margins, _ = _joint_margins(self.scene)
        safety = self.pilot["safety"]
        workspace_exit = float(np.linalg.norm(relative_position)) > float(safety["palm_relative_workspace_radius_m"])
        catastrophic_joint = float(margins.min()) < -float(safety["catastrophic_joint_excursion_rad"])
        intended_unsafe = penetration["maximum_intended_grip"] > float(safety["intended_grip_penetration_ceiling_m"])
        gross_unsafe = penetration["maximum_gross_non_grip"] > float(safety["gross_non_grip_penetration_ceiling_m"])
        hard_failure = bool(not finite or table_contact or workspace_exit or catastrophic_joint or intended_unsafe or gross_unsafe)
        hand_contact = bool(np.any(contacts.contact_flags))
        # A complete hand-object gap is tracked and may recover, but it is not
        # itself evidence of retention and cannot satisfy the recovery reward.
        retained = bool(not hard_failure and hand_contact)
        total_force = float(np.sum(contacts.normal_forces))
        valid_support = float(retained and total_force >= 0.02)
        palm_distance = float(np.linalg.norm(relative_position))
        migration = valid_support * float(np.clip((self.initial_palm_distance - palm_distance) / 0.02, 0.0, 1.0))
        alternate_fraction = float(np.sum(contacts.support_load_fraction[2:]))
        support = valid_support * max(float(alternate_support), float(np.clip(alternate_fraction / 0.5, 0.0, 1.0)))
        selected_load = float(contacts.support_load_fraction[selected_index])
        unload = support * float(np.clip((0.5 - selected_load) / 0.5, 0.0, 1.0))
        release = float(self.alternate_support_seen and selected_free and retained)
        recovery = release * float(
            np.clip(
                self.release_persistence_steps / float(safety["recovered_persistence_steps"]),
                0.0,
                1.0,
            )
        ) * float(available[selected_index] > 0.0)
        potentials = np.asarray((valid_support, migration, support, unload, release, recovery), dtype=np.float64)
        penetration_vector = np.asarray([penetration[name] for name in PENETRATION_LABELS], dtype=np.float64)
        return {
            "contacts": contacts,
            "records": records,
            "penetration": penetration,
            "penetration_vector": penetration_vector,
            "relative_position": relative_position,
            "relative_quaternion": relative_quaternion,
            "linear_velocity": linear,
            "angular_velocity": angular,
            "topology": topology,
            "table_contact": table_contact,
            "alternate_support": alternate_support,
            "alternative_indices": alternative_indices,
            "selected_index": selected_index,
            "selected_free": selected_free,
            "available_motion": available,
            "finite": finite,
            "joint_margins": margins,
            "workspace_exit": workspace_exit,
            "catastrophic_joint": catastrophic_joint,
            "intended_unsafe": intended_unsafe,
            "gross_unsafe": gross_unsafe,
            "hard_failure": hard_failure,
            "retained": retained,
            "potentials": potentials,
            "palm_distance": palm_distance,
        }

    def _observations(self, state: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        qpos = self.data.qpos[:24]
        qvel = self.data.qvel[:24]
        wrist_ids = np.asarray(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.scene.config.hand.wrist_joints
            ]
        )
        wrist = np.r_[
            self.data.qpos[self.model.jnt_qposadr[wrist_ids]],
            self.data.qvel[self.model.jnt_dofadr[wrist_ids]],
        ]
        selected = np.zeros(2); selected[0 if self.selected_release_finger == "thumb" else 1] = 1.0
        stage = np.zeros(6); stage[self.curriculum_stage] = 1.0
        actor = np.concatenate(
            (
                np.clip(qpos / np.pi, -2.0, 2.0),
                np.clip(qvel / 20.0, -2.0, 2.0),
                np.clip(wrist / np.asarray([np.pi, np.pi, 20.0, 20.0]), -2.0, 2.0),
                state["contacts"].contact_flags,
                np.clip(state["contacts"].normal_forces / 5.0, 0.0, 2.0),
                self.previous_action,
                np.r_[np.clip(state["relative_position"] / 0.1, -2.0, 2.0), state["relative_quaternion"]],
                np.clip(np.r_[state["linear_velocity"], state["angular_velocity"]] / 10.0, -2.0, 2.0),
                state["contacts"].support_load_fraction,
                np.clip(state["topology"] / 4.0, 0.0, 2.0),
                np.clip(state["penetration_vector"] / 0.003, 0.0, 2.0),
                selected,
                stage,
            )
        ).astype(np.float32)
        critic = np.concatenate(
            (
                actor,
                state["potentials"],
                np.clip(state["available_motion"] / np.pi, 0.0, 2.0),
                np.asarray([self.steps / float(self.pilot["episode"]["simulation_steps"])]),
            )
        ).astype(np.float32)
        if actor.shape != (OBSERVATION_CONTRACT.actor_dimension,) or critic.shape != (OBSERVATION_CONTRACT.critic_dimension,):
            raise AssertionError("Phase 3B-1A observation contract mismatch")
        if not np.all(np.isfinite(actor)) or not np.all(np.isfinite(critic)):
            raise FloatingPointError("nonfinite privileged observation")
        return actor, critic

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if options and "reset_index" in options:
            reset_index = int(options["reset_index"])
        else:
            reset_index = int(self.np_random.integers(len(self.state_paths)))
        reset_index %= len(self.state_paths)
        restore_release_state(self.scene, self.state_paths[reset_index])
        set_fixture(self.scene, False)
        self.model.actuator_gainprm[:, 0] = self._base_gain
        self.model.actuator_biasprm[:, 1] = self._base_bias
        self.desired = self.data.ctrl.copy()
        self.previous_action[:] = 0.0
        self.steps = 0
        self.selected_release_finger = "thumb" if int(self.np_random.integers(2)) == 0 else "index"
        self.stage_bonus_given = False
        self.alternate_support_seen = False
        self.release_persistence_steps = 0
        self.stage_ever[:] = 0
        self.event_steps = {}
        self.gap_start = None
        self.contact_gaps = []
        self._episode_metrics = {
            "action_bound_hits": 0,
            "action_coordinates": 0,
            "target_clip_hits": 0,
            "stiffness_values": [],
            "minimum_joint_margin_rad": float("inf"),
            "maximum_intended_penetration_m": 0.0,
            "maximum_gross_penetration_m": 0.0,
            "table_drop": False,
            "gross_collision": False,
            "palm_contact_achieved": False,
            "alternate_support_achieved": False,
            "return": 0.0,
        }
        mujoco.mj_forward(self.model, self.data)
        state = self._state_features()
        self.initial_palm_distance = state["palm_distance"]
        self.previous_stage_potential = float(state["potentials"][self.curriculum_stage])
        self._last_relative_position = state["relative_position"].copy()
        self._last_linear_velocity = state["linear_velocity"].copy()
        actor, critic = self._observations(state)
        return actor, {
            "critic_observation": critic,
            "candidate_id": self.candidate_ids[reset_index],
            "split": self.split_name,
            "selected_release_finger": self.selected_release_finger,
            "curriculum_stage": CURRICULUM_STAGES[self.curriculum_stage],
        }

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        action_cfg = self.pilot["action"]
        delta_limit = float(action_cfg["target_delta_limit_rad_per_control_step"])
        rate_cap = float(action_cfg["target_rate_cap_rad_per_control_step"])
        delta = np.clip(action[:20] * delta_limit, -rate_cap, rate_cap)
        proposed = self.desired + delta
        clipped = np.clip(proposed, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
        self._episode_metrics["target_clip_hits"] += int(np.sum(clipped != proposed))
        self.desired = clipped
        stiffness_min = float(action_cfg["stiffness_scale_minimum"])
        stiffness_max = float(action_cfg["stiffness_scale_maximum"])
        stiffness = stiffness_min + 0.5 * (action[20:] + 1.0) * (stiffness_max - stiffness_min)
        self._apply_stiffness(stiffness)
        self.data.ctrl[:] = self.desired
        self._episode_metrics["action_bound_hits"] += int(np.sum(np.abs(action) >= 0.999))
        self._episode_metrics["action_coordinates"] += len(action)
        self._episode_metrics["stiffness_values"].extend(float(value) for value in stiffness)
        for _ in range(int(self.pilot["episode"]["control_decimation"])):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        state = self._state_features()
        if state["alternate_support"]:
            self.alternate_support_seen = True
        if (
            self.alternate_support_seen
            and state["selected_free"]
            and state["retained"]
            and state["available_motion"][state["selected_index"]] > 0.0
        ):
            self.release_persistence_steps += 1
        else:
            self.release_persistence_steps = 0
        state = self._state_features()
        potentials = state["potentials"]
        completion = float(self.pilot["curriculum"]["completion_potential"])
        for index, potential in enumerate(potentials):
            if potential >= completion and not self.stage_ever[index]:
                self.stage_ever[index] = 1
                self.event_steps[CURRICULUM_STAGES[index]] = self.steps

        current = float(potentials[self.curriculum_stage])
        delta_potential = current - self.previous_stage_potential
        completion_bonus = 0.0
        if current >= completion and not self.stage_bonus_given:
            completion_bonus = float(self.pilot["reward"]["stage_completion_bonus"])
            self.stage_bonus_given = True
        recovered = bool(potentials[5] >= completion)
        terminal_outcome = 0.0
        if recovered:
            terminal_outcome = float(self.pilot["reward"]["resource_recovered_bonus"])
        elif state["hard_failure"]:
            terminal_outcome = float(self.pilot["reward"]["hard_failure_penalty"])
        reward = float(delta_potential + completion_bonus + terminal_outcome)
        self.previous_stage_potential = current
        self.previous_action = action.copy()
        self._episode_metrics["return"] += reward
        self._episode_metrics["minimum_joint_margin_rad"] = min(
            self._episode_metrics["minimum_joint_margin_rad"], float(state["joint_margins"].min())
        )
        self._episode_metrics["maximum_intended_penetration_m"] = max(
            self._episode_metrics["maximum_intended_penetration_m"], float(state["penetration"]["maximum_intended_grip"])
        )
        self._episode_metrics["maximum_gross_penetration_m"] = max(
            self._episode_metrics["maximum_gross_penetration_m"], float(state["penetration"]["maximum_gross_non_grip"])
        )
        self._episode_metrics["table_drop"] |= bool(state["table_contact"])
        self._episode_metrics["gross_collision"] |= bool(state["gross_unsafe"])
        self._episode_metrics["palm_contact_achieved"] |= bool(state["contacts"].contact_flags[5])
        self._episode_metrics["alternate_support_achieved"] |= bool(state["alternate_support"])

        any_contact = bool(np.any(state["contacts"].contact_flags))
        if not any_contact and self.gap_start is None:
            self.gap_start = self.steps
            self._gap_origin = state["relative_position"].copy()
        elif any_contact and self.gap_start is not None:
            self.contact_gaps.append(
                {
                    "start_step": self.gap_start,
                    "duration_steps": self.steps - self.gap_start,
                    "displacement_m": float(np.linalg.norm(state["relative_position"] - self._gap_origin)),
                    "reestablished": True,
                    "recontact_identity": [
                        surface for surface, flag in zip(SUPPORT_SURFACES, state["contacts"].contact_flags) if flag
                    ],
                    "object_speed_m_s": float(np.linalg.norm(state["linear_velocity"])),
                }
            )
            self.gap_start = None

        terminated = bool(state["hard_failure"] or recovered)
        truncated = bool(self.steps >= int(self.pilot["episode"]["simulation_steps"]))
        actor, critic = self._observations(state)
        info = {
            "critic_observation": critic,
            "curriculum_stage": CURRICULUM_STAGES[self.curriculum_stage],
            "stage_potentials": potentials.copy(),
            "stage_ever": self.stage_ever.copy(),
            "reward_components": {
                "potential_delta": delta_potential,
                "stage_completion_bonus": completion_bonus,
                "terminal_outcome": terminal_outcome,
            },
            "resource_recovered": recovered,
            "thumb_recovered": bool(recovered and self.selected_release_finger == "thumb"),
            "index_recovered": bool(recovered and self.selected_release_finger == "index"),
            "object_retained": state["retained"],
            "table_drop": state["table_contact"],
            "gross_collision": state["gross_unsafe"],
            "penetration": state["penetration"],
            "minimum_joint_margin_rad": float(state["joint_margins"].min()),
            "action_bound_hits": int(np.sum(np.abs(action) >= 0.999)),
            "target_clip_hits": int(np.sum(clipped != proposed)),
            "stiffness_scales": stiffness.copy(),
            "contact_gaps": list(self.contact_gaps),
            "selected_release_finger": self.selected_release_finger,
            "contact_flags": state["contacts"].contact_flags.copy(),
            "normal_forces_n": state["contacts"].normal_forces.copy(),
            "support_load_fraction": state["contacts"].support_load_fraction.copy(),
            "relative_position_m": state["relative_position"].copy(),
            "linear_velocity_m_s": state["linear_velocity"].copy(),
            "angular_velocity_rad_s": state["angular_velocity"].copy(),
            "palm_contact": bool(state["contacts"].contact_flags[5]),
            "alternate_support": bool(state["alternate_support"]),
        }
        if terminated or truncated:
            stiffness_values = np.asarray(self._episode_metrics["stiffness_values"])
            info["episode_metrics"] = {
                **self._episode_metrics,
                "stage_ever": self.stage_ever.tolist(),
                "event_steps": dict(self.event_steps),
                "resource_recovered": recovered,
                "thumb_recovered": bool(recovered and self.selected_release_finger == "thumb"),
                "index_recovered": bool(recovered and self.selected_release_finger == "index"),
                "object_retained": bool(state["retained"]),
                "contact_gap_count": len(self.contact_gaps) + int(self.gap_start is not None),
                "contact_gaps": list(self.contact_gaps),
                "action_bound_hit_fraction": self._episode_metrics["action_bound_hits"] / max(1, self._episode_metrics["action_coordinates"]),
                "stiffness_mean": float(np.mean(stiffness_values)) if len(stiffness_values) else float("nan"),
                "stiffness_min": float(np.min(stiffness_values)) if len(stiffness_values) else float("nan"),
                "stiffness_max": float(np.max(stiffness_values)) if len(stiffness_values) else float("nan"),
                "simulated_steps": self.steps,
            }
        return actor, reward, terminated, truncated, info

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._camera = mujoco.MjvCamera()
            self._camera.lookat[:] = (0.34, -0.02, 0.01)
            self._camera.distance = 0.36
            self._camera.azimuth = 145
            self._camera.elevation = -18
        self._renderer.update_scene(self.data, camera=self._camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def rollout_policy(
    env: Phase3B1APrivilegedEnv,
    policy: Callable[[np.ndarray, Phase3B1APrivilegedEnv, int], np.ndarray],
    *,
    seed: int,
    reset_index: int = 0,
) -> dict[str, Any]:
    observation, _ = env.reset(seed=seed, options={"reset_index": reset_index})
    total = 0.0
    final_info: dict[str, Any] = {}
    for step in range(int(env.pilot["episode"]["simulation_steps"])):
        action = policy(observation, env, step)
        observation, reward, terminated, truncated, final_info = env.step(action)
        total += reward
        if terminated or truncated:
            break
    return {"return": total, "steps": step + 1, "info": final_info, "episode_metrics": final_info.get("episode_metrics", {})}
