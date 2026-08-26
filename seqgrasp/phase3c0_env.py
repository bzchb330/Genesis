"""Separate, reward-free Phase 3C interface for future learning work."""
from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from .phase3.config import FINGERS
from .phase3.control import VariableImpedanceController, actuator_target_from_qpos
from .phase3c0 import (
    Phase3CFingerRole,
    Phase3CMultiScene,
    Phase3CRoles,
    build_phase3c_multiscene,
    gravity_in_palm_frame,
    multi_object_support_graph,
    object_pose_in_palm,
    open_hand_configuration,
    phase3c_action_contract,
    phase3c_observation_contract,
    set_phase3c_object_pose,
    storage_aperture,
)


class Phase3COpenCorridorEnv(gym.Env):
    """Mechanics-only environment scaffold; Phase 3C-0 defines no reward."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        self.scene: Phase3CMultiScene = build_phase3c_multiscene()
        self.model, self.data = self.scene.model, self.scene.data
        self.controller = VariableImpedanceController(self.scene)  # structural protocol compatibility
        self.roles = Phase3CRoles()
        self.steps = 0
        self.previous_action = np.zeros(self.controller.action_dimension, dtype=np.float64)
        self.observation_metadata = phase3c_observation_contract(self.controller.action_dimension)
        dimension = sum(item.dimension for item in self.observation_metadata if item.actor_available)
        self.observation_space = spaces.Box(-np.inf, np.inf, (dimension,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (self.controller.action_dimension,), np.float32)
        self.action_contract = phase3c_action_contract(self.scene)

    def _object_contact_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        flags = np.zeros((2, 6), dtype=float)
        forces = np.zeros((2, 6), dtype=float)
        graph = multi_object_support_graph(self.scene)
        surface_index = {name: i for i, name in enumerate((*FINGERS, "palm"))}
        object_index = {"A": 0, "B": 1}
        for edge in graph["edges"]:
            row, col = object_index[edge["object"]], surface_index[edge["support"]]
            flags[row, col] = 1.0
            forces[row, col] += edge["normal_force_n"]
        return flags, forces, np.concatenate((flags.ravel(), forces.ravel()))

    def _observation(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        flags, forces, graph_vector = self._object_contact_arrays()
        wrist_ids = np.asarray([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                                for name in self.scene.config.hand.wrist_joints])
        role_hot = np.zeros((len(FINGERS), len(Phase3CFingerRole)), dtype=float)
        role_index = {role: index for index, role in enumerate(Phase3CFingerRole)}
        for row, finger in enumerate(FINGERS):
            role_hot[row, role_index[self.roles.fingers[finger]]] = 1.0
        actor = np.concatenate((
            self.data.qpos[:24], self.data.qvel[:24],
            self.data.qpos[self.model.jnt_qposadr[wrist_ids]],
            self.data.qvel[self.model.jnt_dofadr[wrist_ids]],
            flags[:, :5].ravel(), flags[:, 5], forces.ravel(),
            graph_vector[:12], flags.ravel()[:6], role_hot.ravel(), self.previous_action,
        )).astype(np.float32)
        poses = []
        velocities = []
        for label in ("A", "B"):
            position, rotation = object_pose_in_palm(self.scene, self.scene.object_body_ids[label])
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, rotation.ravel())
            poses.extend((*position, *quat))
            dof = self.model.jnt_dofadr[self.scene.object_joint_ids[label]]
            velocities.extend(self.data.qvel[dof:dof + 6])
        aperture = storage_aperture(self.scene)
        contact_clearance = min(
            (float(self.data.contact[index].dist) for index in range(self.data.ncon)),
            default=float("inf"),
        )
        privileged = {
            "objects_pose_in_palm": np.asarray(poses),
            "objects_velocity": np.asarray(velocities),
            "storage_aperture_geometry": np.asarray([
                *aperture["centroid_palm_m"], *aperture["normal_palm"],
                *aperture["basis_u_palm"], *aperture["basis_v_palm"],
                aperture["effective_width_m"], aperture["effective_height_m"],
                aperture["minimum_node_clearance_m"],
            ]),
            "insertion_corridor_geometry": np.zeros(8, dtype=np.float64),
            "gravity_in_palm_frame": gravity_in_palm_frame(self.scene),
            "contact_graph": graph_vector,
            "exact_collision_clearance": np.asarray([contact_clearance]),
        }
        return actor, privileged

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        open_qpos, _ = open_hand_configuration()
        self.data.qpos[:24] = open_qpos
        self.data.qvel[:] = 0.0
        base = np.asarray(self.scene.config.object["initial_pos"], dtype=float)
        offsets = np.asarray(self.scene.config.diagnostic["cohort_offsets_m"], dtype=float)
        offset = offsets[int(self.np_random.integers(len(offsets)))]
        set_phase3c_object_pose(self.scene, "A", base + offset)
        set_phase3c_object_pose(self.scene, "B", base + np.asarray([0.0, 0.12, 0.0]))
        self.data.ctrl[:] = actuator_target_from_qpos(self.scene, open_qpos)
        mujoco.mj_forward(self.model, self.data)
        self.controller.reset()
        self.roles = Phase3CRoles()
        self.steps = 0
        self.previous_action[:] = 0.0
        observation, privileged = self._observation()
        return observation, {"privileged_observation": privileged, "state": self.roles.state.value}

    def step(self, action):
        action = np.asarray(action, dtype=float)
        self.controller.step(action)
        for _ in range(int(self.scene.config.raw["frame_skip"])):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        self.previous_action = action.copy()
        observation, privileged = self._observation()
        truncated = self.steps >= int(self.scene.config.raw["episode_steps"])
        return observation, 0.0, False, truncated, {
            "privileged_observation": privileged,
            "state": self.roles.state.value,
            "reward_defined": False,
        }
