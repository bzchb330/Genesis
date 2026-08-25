from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .config import FINGERS
from .contacts import extract_shadow_contacts
from .model import ShadowScene


def actuator_target_from_qpos(scene: ShadowScene, qpos: np.ndarray) -> np.ndarray:
    """Map a 24-joint Shadow pose to its 20 joint/tendon position targets."""
    model = scene.model
    target = np.zeros(model.nu, dtype=np.float64)
    for actuator_id in range(model.nu):
        transmission = int(model.actuator_trntype[actuator_id])
        transmission_id = int(model.actuator_trnid[actuator_id, 0])
        if transmission == int(mujoco.mjtTrn.mjTRN_JOINT):
            target[actuator_id] = qpos[model.jnt_qposadr[transmission_id]]
        elif transmission == int(mujoco.mjtTrn.mjTRN_TENDON):
            # Fixed tendons in E3M5 couple J2 + J1 with unit coefficients.
            tendon_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, transmission_id)
            prefix = tendon_name[:-1]
            joint_ids = [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{suffix}")
                for suffix in ("2", "1")
            ]
            target[actuator_id] = sum(qpos[model.jnt_qposadr[joint_id]] for joint_id in joint_ids)
        else:
            raise ValueError(f"unsupported Shadow actuator transmission {transmission}")
    return np.clip(target, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])


@dataclass
class VariableImpedanceController:
    scene: ShadowScene

    def __post_init__(self) -> None:
        self.base_gain = self.scene.model.actuator_gainprm[:, 0].copy()
        self.base_bias = self.scene.model.actuator_biasprm[:, 1].copy()
        self.desired = self.scene.data.ctrl.copy()

    @property
    def group_order(self) -> tuple[str, ...]:
        return ("wrist", *FINGERS)

    @property
    def action_dimension(self) -> int:
        return self.scene.model.nu + len(self.group_order)

    def reset(self) -> None:
        self.desired = actuator_target_from_qpos(self.scene, self.scene.data.qpos)
        self.apply_stiffness(np.ones(len(self.group_order)))
        self.scene.data.ctrl[:] = self.desired

    def apply_stiffness(self, scales: np.ndarray) -> None:
        scales = np.asarray(scales, dtype=np.float64)
        if scales.shape != (len(self.group_order),):
            raise ValueError("stiffness scale must have one value per wrist/finger group")
        lower, upper = self.scene.config.control["stiffness_scale_bounds"]
        scales = np.clip(scales, lower, upper)
        for scale, group in zip(scales, self.group_order):
            ids = self.scene.actuator_ids[group]
            self.scene.model.actuator_gainprm[ids, 0] = self.base_gain[ids] * scale
            self.scene.model.actuator_biasprm[ids, 1] = self.base_bias[ids] * scale

    def step(self, action: np.ndarray, *, fixed_impedance: bool = False) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.action_dimension,):
            raise ValueError(f"Phase 3 action must have shape {(self.action_dimension,)}")
        nu = self.scene.model.nu
        delta_limit = float(self.scene.config.control["delta_q_limit"])
        self.desired += np.clip(action[:nu], -1.0, 1.0) * delta_limit
        self.desired = np.clip(
            self.desired,
            self.scene.model.actuator_ctrlrange[:, 0],
            self.scene.model.actuator_ctrlrange[:, 1],
        )
        scales = np.ones(len(self.group_order)) if fixed_impedance else action[nu:]
        if not fixed_impedance:
            lower, upper = self.scene.config.control["stiffness_scale_bounds"]
            scales = lower + (np.clip(scales, -1.0, 1.0) + 1.0) * 0.5 * (upper - lower)
        self.apply_stiffness(scales)
        self.scene.data.ctrl[:] = self.desired
        return self.desired.copy()


@dataclass
class ContactAwareCloser:
    """Independent contact latch for diagnostic thumb/index closure."""

    scene: ShadowScene
    force_threshold_n: float

    def __post_init__(self) -> None:
        self.latched = {"thumb": False, "index": False}
        self.latched_targets: dict[str, np.ndarray] = {}

    def reset(self) -> None:
        self.latched = {"thumb": False, "index": False}
        self.latched_targets.clear()

    def limit_target(self, proposed_target: np.ndarray) -> np.ndarray:
        proposed_target = np.asarray(proposed_target, dtype=np.float64).copy()
        contacts = extract_shadow_contacts(self.scene)
        for finger in ("thumb", "index"):
            surface_index = (*FINGERS, "palm").index(finger)
            if not self.latched[finger] and contacts.normal_forces[surface_index] >= self.force_threshold_n:
                self.latched[finger] = True
                ids = self.scene.actuator_ids[finger]
                self.latched_targets[finger] = self.scene.data.ctrl[ids].copy()
            if self.latched[finger]:
                ids = self.scene.actuator_ids[finger]
                proposed_target[ids] = self.latched_targets[finger]
        return proposed_target
