from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from ..config import ROOT
from .config import FINGERS, SUPPORT_SURFACES, Phase3Config, load_phase3_config
from .contacts import extract_shadow_contacts, object_velocity
from .control import VariableImpedanceController
from .model import ShadowScene, build_shadow_scene, set_fixture, set_object_pose
from .resource import compute_resource_snapshot
from .rewards import Phase3RewardTerms, weighted_reward
from .roles import FingerRole, ManipulationPhase, RoleState


@dataclass(frozen=True)
class ObservationComponent:
    name: str
    dimension: int
    actor_available: bool


def load_keyframe_qpos(name: str) -> np.ndarray:
    root = ET.parse(ROOT / "assets/hands/shadow_right/keyframes.xml").getroot()
    key = root.find(f".//key[@name='{name}']")
    if key is None:
        raise ValueError(f"missing Shadow keyframe {name}")
    return np.fromstring(key.get("qpos", ""), sep=" ")


class Phase3ShadowHandEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        config: Phase3Config | None = None,
        render_mode: str | None = None,
        fixed_impedance: bool | None = None,
    ) -> None:
        self.config = config or load_phase3_config()
        self.scene: ShadowScene = build_shadow_scene(self.config)
        self.model, self.data = self.scene.model, self.scene.data
        self.render_mode = render_mode
        self.fixed_impedance = (
            self.config.control["fixed_impedance_compatibility"]
            if fixed_impedance is None
            else bool(fixed_impedance)
        )
        self.controller = VariableImpedanceController(self.scene)
        self.roles = RoleState()
        self.steps = 0
        self.previous_action = np.zeros(self.controller.action_dimension, dtype=np.float64)
        self.previous_palm_distance = 0.0
        self._renderer: mujoco.Renderer | None = None
        self._camera: mujoco.MjvCamera | None = None
        self.action_space = spaces.Box(-1.0, 1.0, (self.controller.action_dimension,), np.float32)
        actor_dimension = sum(component.dimension for component in self.observation_metadata if component.actor_available)
        self.observation_space = spaces.Box(-np.inf, np.inf, (actor_dimension,), np.float32)
        self.metadata = {
            "render_modes": ["rgb_array"],
            "render_fps": round(1.0 / (self.config.raw["timestep"] * self.config.raw["frame_skip"])),
        }

    @property
    def observation_metadata(self) -> tuple[ObservationComponent, ...]:
        action_dim = self.controller.action_dimension
        return (
            ObservationComponent("hand_joint_positions", 24, True),
            ObservationComponent("hand_joint_velocities", 24, True),
            ObservationComponent("wrist_state", 4, True),
            ObservationComponent("finger_contact_flags", 5, True),
            ObservationComponent("finger_normal_forces", 5, True),
            ObservationComponent("palm_contact", 1, True),
            ObservationComponent("palm_normal_force", 1, True),
            ObservationComponent("finger_roles_one_hot", len(FINGERS) * len(FingerRole), True),
            ObservationComponent("manipulation_phase_one_hot", len(ManipulationPhase), True),
            ObservationComponent("previous_action", action_dim, True),
            ObservationComponent("object_pose_relative_to_palm", 7, False),
            ObservationComponent("object_velocity", 6, False),
            ObservationComponent("full_contact_counts", len(SUPPORT_SURFACES), False),
            ObservationComponent("support_vector", len(SUPPORT_SURFACES), False),
            ObservationComponent("support_load_fraction", len(SUPPORT_SURFACES), False),
            ObservationComponent("penetration", 1, False),
            ObservationComponent("tip_and_palm_object_relations", len(SUPPORT_SURFACES) * 3, False),
        )

    def _role_one_hot(self) -> np.ndarray:
        output = np.zeros((len(FINGERS), len(FingerRole)), dtype=np.float64)
        for index, finger in enumerate(FINGERS):
            output[index, int(self.roles.fingers[finger])] = 1.0
        return output.ravel()

    def _observations(self) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
        contacts = extract_shadow_contacts(self.scene)
        qpos = self.data.qpos[:24].copy()
        qvel = self.data.qvel[:24].copy()
        wrist_ids = np.asarray(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.config.hand.wrist_joints]
        )
        wrist_addresses = self.model.jnt_qposadr[wrist_ids]
        wrist_dof_addresses = self.model.jnt_dofadr[wrist_ids]
        wrist = np.r_[self.data.qpos[wrist_addresses], self.data.qvel[wrist_dof_addresses]]
        phase_one_hot = np.zeros(len(ManipulationPhase), dtype=np.float64)
        phase_one_hot[int(self.roles.phase)] = 1.0
        actor = np.concatenate(
            (
                qpos,
                qvel,
                wrist,
                contacts.contact_flags[:5],
                contacts.normal_forces[:5],
                contacts.contact_flags[5:6],
                contacts.normal_forces[5:6],
                self._role_one_hot(),
                phase_one_hot,
                self.previous_action,
            )
        ).astype(np.float32)
        palm_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.config.hand.palm_body)
        relative_position = self.data.xpos[self.scene.object_body_id] - self.data.xpos[palm_id]
        linear_velocity, angular_velocity = object_velocity(self.scene)
        relations = []
        for surface in SUPPORT_SURFACES:
            body_name = (
                self.config.hand.palm_body
                if surface == "palm"
                else self.config.hand.fingertip_bodies[surface]
            )
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            relations.extend(self.data.xpos[self.scene.object_body_id] - self.data.xpos[body_id])
        privileged = {
            "object_pose_relative_to_palm": np.r_[relative_position, self.data.xquat[self.scene.object_body_id]].copy(),
            "object_velocity": np.r_[linear_velocity, angular_velocity],
            "full_contact_counts": np.asarray(
                [len(contacts.records_by_surface[surface]) for surface in SUPPORT_SURFACES], dtype=np.float64
            ),
            "support_vector": contacts.support_vector.copy(),
            "support_load_fraction": contacts.support_load_fraction.copy(),
            "penetration": contacts.maximum_penetration,
            "tip_and_palm_object_relations": np.asarray(relations, dtype=np.float64),
        }
        return actor, privileged

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:24] = load_keyframe_qpos("pre grasp")
        offsets = self.config.diagnostic["cohort_offsets_m"]
        offset = np.asarray(offsets[int(self.np_random.integers(len(offsets)))], dtype=np.float64)
        set_object_pose(self.scene, np.asarray(self.config.object["initial_pos"], dtype=np.float64) + offset)
        set_fixture(self.scene, True)
        self.roles = RoleState()
        self.roles.begin_probe()
        self.steps = 0
        self.previous_action[:] = 0.0
        self.controller.reset()
        mujoco.mj_forward(self.model, self.data)
        palm_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.config.hand.palm_body)
        self.previous_palm_distance = float(
            np.linalg.norm(self.data.xpos[self.scene.object_body_id] - self.data.xpos[palm_id])
        )
        observation, privileged = self._observations()
        return observation, {
            "phase": self.roles.phase.name,
            "privileged_observation": privileged,
            "fixture_active": True,
        }

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        self.controller.step(action, fixed_impedance=self.fixed_impedance)
        for _ in range(int(self.config.raw["frame_skip"])):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        contacts = extract_shadow_contacts(self.scene)
        resources = compute_resource_snapshot(self.scene, contacts, self.roles)
        palm_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.config.hand.palm_body)
        palm_distance = float(np.linalg.norm(self.data.xpos[self.scene.object_body_id] - self.data.xpos[palm_id]))
        joint_limits = self.model.jnt_range[:24]
        joint_margin_violation = np.maximum(joint_limits[:, 0] - self.data.qpos[:24], 0.0).sum()
        joint_margin_violation += np.maximum(self.data.qpos[:24] - joint_limits[:, 1], 0.0).sum()
        reference = float(self.config.diagnostic["reference_penetration_m"])
        reward_terms = Phase3RewardTerms(
            object_progress_to_palm=self.previous_palm_distance - palm_distance,
            valid_support=float(contacts.normal_forces.sum()),
            palm_contact=float(contacts.normal_forces[5]),
            support_transfer=float(contacts.support_load_fraction[2:].sum()),
            acquisition_finger_release_after_support=0.0,
            recovered_resource=float(resources.free_finger_mask[:2].sum()),
            complete_object_loss=0.0,
            unsafe_penetration=max(0.0, contacts.maximum_penetration - reference),
            joint_limit=float(joint_margin_violation),
            violent_action=float(np.linalg.norm(action - self.previous_action)),
        )
        reward = weighted_reward(reward_terms, self.config.raw["reward_weights"])
        self.previous_palm_distance = palm_distance
        self.previous_action = action.copy()
        observation, privileged = self._observations()
        truncated = self.steps >= int(self.config.raw["episode_steps"])
        return observation, reward, False, truncated, {
            "phase": self.roles.phase.name,
            "reward_terms": reward_terms.as_dict(),
            "privileged_observation": privileged,
            "free_finger_mask": resources.free_finger_mask.copy(),
            "n_free": resources.n_free,
        }

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model,
                height=int(self.config.raw["render_height"]),
                width=int(self.config.raw["render_width"]),
            )
            self._camera = mujoco.MjvCamera()
            self._camera.lookat[:] = (0.34, -0.02, 0.01)
            self._camera.distance = 0.38
            self._camera.azimuth = 145
            self._camera.elevation = -18
        self._renderer.update_scene(self.data, camera=self._camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
            self._camera = None
