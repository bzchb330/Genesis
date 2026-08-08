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
    distance_to_joint_limits: np.ndarray
    normalized_joint_range_position: np.ndarray
    absolute_control_utilization: np.ndarray
    remaining_positive_control_range: np.ndarray
    remaining_negative_control_range: np.ndarray
    active_object_fingers: np.ndarray
    inactive_object_fingers: np.ndarray
    object_contact_count_per_finger: np.ndarray
    object_normal_force_per_finger: np.ndarray

def compute_resource_metric(state: ResourceState, cfg) -> None:
    """Stable hook for resource metric J without defining a scientific score."""
    # TODO(PI): define resource-awareness metric J and its units.
    return None

def build_resource_state(model, data, cfg, phase, indices=None) -> ResourceState:
    from ..control import hand_state, resolve_hand_indices
    from ..sensing import compute_tactile_features, extract_contacts, group_contacts_by_finger
    indices=indices or resolve_hand_indices(model,cfg.hand); q,qvel=hand_state(data,indices)
    contacts=extract_contacts(model,data); grouped=group_contacts_by_finger(contacts,cfg.hand.finger_geom_mapping); tactile=compute_tactile_features(grouped,cfg)
    poses={}
    for obj in cfg.scene.objects:
        bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,obj.name); poses[obj.name]=np.r_[data.xpos[bid],data.xquat[bid]].copy()
    limits=np.tile(np.asarray([-cfg.task.torque_limit,cfg.task.torque_limit]),(cfg.hand.dof_count,1)); joint_limits=model.jnt_range[indices.joint_ids].copy(); distance=np.stack([q-joint_limits[:,0],joint_limits[:,1]-q],axis=1); controls=data.ctrl[indices.actuator_ids].copy(); object_name=cfg.diagnostic.object_name; counts=np.asarray([sum(object_name in {record.body1_name,record.body2_name} for record in grouped[finger]) for finger in cfg.hand.finger_geom_mapping]); forces=np.asarray([sum(record.normal_force for record in grouped[finger] if object_name in {record.body1_name,record.body2_name}) for finger in cfg.hand.finger_geom_mapping]); active=(counts>0).astype(np.int8)
    return ResourceState(q,qvel,controls,limits,joint_limits,contacts,tactile,int(phase),poses,distance,(q-joint_limits[:,0])/(joint_limits[:,1]-joint_limits[:,0]),np.abs(controls)/cfg.task.torque_limit,limits[:,1]-controls,controls-limits[:,0],active,1-active,counts,forces)
