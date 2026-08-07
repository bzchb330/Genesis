from __future__ import annotations
from dataclasses import dataclass
import mujoco
import numpy as np

@dataclass(frozen=True)
class HandIndices:
    actuator_ids: np.ndarray
    joint_ids: np.ndarray
    qpos_addresses: np.ndarray
    qvel_addresses: np.ndarray

def resolve_hand_indices(model: mujoco.MjModel, hand_cfg) -> HandIndices:
    if len(hand_cfg.actuator_names) != hand_cfg.dof_count or len(hand_cfg.joint_names) != hand_cfg.dof_count:
        raise ValueError("configured actuator/joint names must match dof_count")
    actuator_ids = np.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in hand_cfg.actuator_names], dtype=int)
    joint_ids = np.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in hand_cfg.joint_names], dtype=int)
    if np.any(actuator_ids < 0) or np.any(joint_ids < 0): raise ValueError("configured hand actuator or joint is missing")
    if np.any(model.jnt_type[joint_ids] != mujoco.mjtJoint.mjJNT_HINGE): raise ValueError("hand joints must be one-DoF hinges")
    return HandIndices(actuator_ids, joint_ids, model.jnt_qposadr[joint_ids].copy(), model.jnt_dofadr[joint_ids].copy())

def hand_state(data: mujoco.MjData, indices: HandIndices) -> tuple[np.ndarray, np.ndarray]:
    return data.qpos[indices.qpos_addresses].copy(), data.qvel[indices.qvel_addresses].copy()
