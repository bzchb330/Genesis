from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import mujoco

@dataclass(frozen=True)
class ResourceState:
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    actuator_commands: np.ndarray
    actuator_limits: np.ndarray
    joint_limits: np.ndarray
    contacts: list[Any]
    tactile_features: dict[str, np.ndarray]
    phase: int
    object_poses: dict[str, np.ndarray]

def compute_resource_metric(state: ResourceState, cfg) -> None:
    """Stable hook for resource metric J without defining a scientific score."""
    # TODO(PI): define resource-awareness metric J and its units.
    return None

def build_resource_state(model, data, cfg, phase, indices=None) -> ResourceState:
    from ..control import hand_state, resolve_hand_indices
    from ..sensing import compute_tactile_features, extract_contacts, group_contacts_by_finger
    indices=indices or resolve_hand_indices(model,cfg.hand); q,qvel=hand_state(data,indices)
    contacts=extract_contacts(model,data); tactile=compute_tactile_features(group_contacts_by_finger(contacts,cfg.hand.finger_geom_mapping),cfg)
    poses={}
    for obj in cfg.scene.objects:
        bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,obj.name); poses[obj.name]=np.r_[data.xpos[bid],data.xquat[bid]].copy()
    limits=np.tile(np.asarray([-cfg.task.torque_limit,cfg.task.torque_limit]),(cfg.hand.dof_count,1))
    return ResourceState(q,qvel,data.ctrl[indices.actuator_ids].copy(),limits,model.jnt_range[indices.joint_ids].copy(),contacts,tactile,int(phase),poses)
