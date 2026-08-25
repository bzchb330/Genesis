from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .config import FINGERS
from .contacts import ShadowContactState
from .model import ShadowScene
from .roles import FingerRole, RoleState


@dataclass(frozen=True)
class FingerResource:
    finger: str
    role: FingerRole
    contact: bool
    support_force: float
    joint_margin: float
    local_object_displacement: np.ndarray
    local_reachable_workspace: np.ndarray
    available_motion_range: float
    free_for_another_action: bool


@dataclass(frozen=True)
class ResourceSnapshot:
    fingers: dict[str, FingerResource]
    free_finger_mask: np.ndarray
    n_free: int


def _finger_joint_margin(scene: ShadowScene, finger: str) -> tuple[float, float]:
    ids = scene.joint_ids[finger]
    addresses = scene.model.jnt_qposadr[ids]
    qpos = scene.data.qpos[addresses]
    limits = scene.model.jnt_range[ids]
    widths = limits[:, 1] - limits[:, 0]
    normalized = np.minimum(qpos - limits[:, 0], limits[:, 1] - qpos) / widths
    return float(np.min(normalized)), float(np.sum(2.0 * np.maximum(normalized, 0.0) * widths))


def _linearized_local_workspace(scene: ShadowScene, finger: str, body_id: int) -> np.ndarray:
    """Raw first-order fingertip displacement envelope at the current state.

    Rows are world x/y/z and columns are minimum/maximum displacement. This is
    a local kinematic diagnostic, not a binary reachability claim.
    """
    jacobian = np.zeros((3, scene.model.nv), dtype=np.float64)
    rotation_jacobian = np.zeros((3, scene.model.nv), dtype=np.float64)
    mujoco.mj_jacBody(scene.model, scene.data, jacobian, rotation_jacobian, body_id)
    ids = scene.joint_ids[finger]
    qpos = scene.data.qpos[scene.model.jnt_qposadr[ids]]
    limits = scene.model.jnt_range[ids]
    lower_delta = limits[:, 0] - qpos
    upper_delta = limits[:, 1] - qpos
    dofs = scene.model.jnt_dofadr[ids]
    envelope = np.zeros((3, 2), dtype=np.float64)
    for axis in range(3):
        coefficients = jacobian[axis, dofs]
        first = coefficients * lower_delta
        second = coefficients * upper_delta
        envelope[axis, 0] = np.minimum(first, second).sum()
        envelope[axis, 1] = np.maximum(first, second).sum()
    return envelope


def compute_resource_snapshot(
    scene: ShadowScene, contacts: ShadowContactState, roles: RoleState
) -> ResourceSnapshot:
    object_position = scene.data.xpos[scene.object_body_id]
    resources: dict[str, FingerResource] = {}
    mask = np.zeros(len(FINGERS), dtype=np.int8)
    for index, finger in enumerate(FINGERS):
        body_id = mujoco.mj_name2id(
            scene.model, mujoco.mjtObj.mjOBJ_BODY, scene.config.hand.fingertip_bodies[finger]
        )
        margin, motion_range = _finger_joint_margin(scene, finger)
        contact = bool(contacts.contact_flags[index])
        free = roles.fingers[finger] == FingerRole.FREE and not contact and motion_range > 0.0
        mask[index] = int(free)
        resources[finger] = FingerResource(
            finger=finger,
            role=roles.fingers[finger],
            contact=contact,
            support_force=float(contacts.normal_forces[index]),
            joint_margin=margin,
            local_object_displacement=(object_position - scene.data.xpos[body_id]).copy(),
            local_reachable_workspace=_linearized_local_workspace(scene, finger, body_id),
            available_motion_range=motion_range,
            free_for_another_action=free,
        )
    return ResourceSnapshot(resources, mask, int(mask.sum()))
