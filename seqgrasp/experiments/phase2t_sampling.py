from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from ..config import ConfigBundle, ROOT, load_configs
from ..control import JointImpedanceController, hand_state, resolve_hand_indices
from ..diagnostics.multi_grasp import load_grasp_profile
from ..diagnostics.scripted_grasp import _joint_target
from ..phase2_config import Phase2Config
from ..phase2s_config import Phase2SConfig
from ..scene_builder import build_scene
from .phase2r import GraspStateType, classify_grasp_state, measure_stable_hold
from .resource_components import FINGER_ORDER


def _object_addresses(model: mujoco.MjModel) -> tuple[int, int]:
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_a_free")
    return int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])


def _open_joint_configuration(source: dict, model, cfg, indices) -> np.ndarray:
    profile_path = ROOT / source["proposal_profile_path"]
    _, profile = load_grasp_profile(profile_path)
    return _joint_target(model, cfg, indices, profile.open_joint_fractions)


def evaluate_two_finger_fingertip_candidate(
    phase2s: Phase2SConfig,
    phase2: Phase2Config,
    source: dict,
    support_pair: tuple[str, str],
    attempt_index: int,
    seed: int,
) -> dict:
    """Target one exact two-finger support pair and validate after fixture removal."""

    base_cfg = load_configs(scene_filename=phase2s.scene_filename)
    cfg = replace(base_cfg, hand=replace(
        base_cfg.hand,
        mount_pos=list(source["initial_palm_position_m"]),
        mount_quat=list(source["initial_palm_quaternion"]),
    ))
    model, data = build_scene(cfg)
    indices = resolve_hand_indices(model, cfg.hand)
    open_joint = _open_joint_configuration(source, model, cfg, indices)
    source_target = np.asarray(source["retaining_joint_target_rad"], dtype=float)
    ranges = model.jnt_range[indices.joint_ids]
    groups = {
        finger: np.arange(index * 4, index * 4 + 4, dtype=int)
        for index, finger in enumerate(FINGER_ORDER)
    }
    active = np.concatenate([groups[finger] for finger in support_pair])
    inactive = np.asarray(sorted(set(range(16)) - set(active.tolist())), dtype=int)
    pair_index = [
        ("index", "middle"), ("index", "ring"), ("index", "thumb"),
        ("middle", "ring"), ("middle", "thumb"), ("ring", "thumb"),
    ].index(tuple(support_pair))
    local_index = attempt_index // 6
    rng = np.random.default_rng(np.random.SeedSequence([seed, pair_index, local_index]))
    focused = local_index % 5 != 0
    scale = float(rng.uniform(0.94, 1.08) if focused else rng.uniform(0.78, 1.22))
    target_joint = open_joint.copy()
    target_joint[active] = open_joint[active] + scale * (source_target[active] - open_joint[active])
    source_occupied = {
        finger for finger, flag in zip(FINGER_ORDER, source["occupied_finger_mask"]) if flag
    }
    mapped_donor = None
    for finger in support_pair:
        if finger == "thumb" or finger in source_occupied:
            continue
        mapped_donor = "ring"
        donor = groups[mapped_donor]
        receiver = groups[finger]
        target_joint[receiver] = open_joint[receiver] + scale * (source_target[donor] - open_joint[donor])
    target_joint[active] += rng.uniform(-0.018 if focused else -0.06, 0.018 if focused else 0.06, len(active))
    target_joint[inactive] = open_joint[inactive]
    target_joint = np.clip(target_joint, ranges[:, 0], ranges[:, 1])

    base_position = np.asarray(source["final_object_position_m"], dtype=float)
    if mapped_donor is not None:
        data.qpos[indices.qpos_addresses] = np.asarray(source["final_joint_configuration_rad"], dtype=float)
        mujoco.mj_forward(model, data)
        donor_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY,
            cfg.hand.fingertip_bodies[FINGER_ORDER.index(mapped_donor)],
        )
        donor_tip = data.xpos[donor_body].copy()
        receiver_name = next(finger for finger in support_pair if finger not in source_occupied and finger != "thumb")
        data.qpos[indices.qpos_addresses] = target_joint
        mujoco.mj_forward(model, data)
        receiver_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY,
            cfg.hand.fingertip_bodies[FINGER_ORDER.index(receiver_name)],
        )
        base_position = base_position + (data.xpos[receiver_body] - donor_tip)
    position_span = 0.0007 if focused else 0.0025
    fixture_position = base_position + rng.uniform(-position_span, position_span, 3)
    base_rotation = Rotation.from_quat(np.asarray(source["final_object_quaternion"], dtype=float), scalar_first=True)
    angle_span = 0.025 if focused else 0.10
    perturbation = Rotation.from_rotvec(rng.uniform(-angle_span, angle_span, 3))
    fixture_quaternion = (perturbation * base_rotation).as_quat(scalar_first=True)

    qadr, vadr = _object_addresses(model)
    acquisition_start = np.asarray(source["final_joint_configuration_rad"], dtype=float)
    data.qpos[indices.qpos_addresses] = acquisition_start
    data.qvel[indices.qvel_addresses] = 0.0
    data.qpos[qadr:qadr + 3] = fixture_position
    data.qpos[qadr + 3:qadr + 7] = fixture_quaternion
    data.qvel[vadr:vadr + 6] = 0.0
    mujoco.mj_forward(model, data)
    controller = JointImpedanceController(
        cfg.task.impedance_stiffness, cfg.task.impedance_damping, cfg.task.torque_limit,
    )
    close_steps = phase2s.state.fixture_close_steps
    contact_steps = phase2s.state.fixture_contact_steps
    for step in range(close_steps + contact_steps):
        alpha = min(1.0, (step + 1) / close_steps)
        desired = (1.0 - alpha) * acquisition_start + alpha * target_joint
        q, qvel = hand_state(data, indices)
        data.ctrl[indices.actuator_ids] = controller.torque(desired, q, qvel)
        data.qpos[qadr:qadr + 3] = fixture_position
        data.qpos[qadr + 3:qadr + 7] = fixture_quaternion
        data.qvel[vadr:vadr + 6] = 0.0
        mujoco.mj_step(model, data)
    data.qpos[qadr:qadr + 3] = fixture_position
    data.qpos[qadr + 3:qadr + 7] = fixture_quaternion
    data.qvel[vadr:vadr + 6] = 0.0
    mujoco.mj_forward(model, data)
    measured = measure_stable_hold(
        cfg, model, data, indices, target_joint, phase2s.state.stable_hold_steps,
        phase2.resources, phase2.dataset.friction_cone_edges,
        phase2.dataset.convex_hull_tolerance,
    )
    free_pair = tuple(finger for finger in FINGER_ORDER if finger not in support_pair)
    record = {
        **source,
        **measured,
        "grasp_state_id": f"phase2T_fingertip_2free_{attempt_index:05d}",
        "grasp_state_subtype": "FINGERTIP_ELIGIBLE_2FREE",
        "source_phase2s_grasp_state_id": source["grasp_state_id"],
        "support_pair": list(support_pair),
        "free_finger_set": list(free_pair),
        "actual_A_contact_topology": [
            finger for finger, flag in zip(FINGER_ORDER, measured["per_finger_A_contact_flags"]) if flag
        ],
        "targeted_attempt_index": int(attempt_index),
        "sampling_mode": "focused_two_finger_basin" if focused else "broad_two_finger_basin",
        "initial_object_position_m": fixture_position.tolist(),
        "initial_object_quaternion": fixture_quaternion.tolist(),
        "open_joint_configuration_rad": open_joint.tolist(),
        "retaining_joint_target_rad": target_joint.tolist(),
        "target_closure_scale": scale,
        "mapped_source_finger": mapped_donor,
        "fixture_method": "temporary_free_joint_pose_reset_during_initialization_only",
        "fixture_release_timestep": close_steps + contact_steps,
        "revalidated_with_half_scale_geometry": True,
    }
    classified = classify_grasp_state(
        record, GraspStateType.FINGERTIP, phase2s.state,
        phase2.dataset.convex_hull_tolerance,
    )
    expected_mask = [finger in support_pair for finger in FINGER_ORDER]
    extra_checks = {
        "exactly_two_load_bearing_fingers": measured["occupied_finger_count"] == 2,
        "occupied_pair_matches_target": measured["occupied_finger_mask"] == expected_mask,
        "exactly_two_free_fingers": measured["free_finger_count"] == 2,
        "minimum_two_physical_finger_contacts": sum(measured["per_finger_A_contact_flags"]) >= 2,
    }
    checks = {**classified["checks"], **extra_checks}
    rejection = next((name for name, passed in checks.items() if not passed), None)
    return {
        **classified,
        "checks": checks,
        "accepted": rejection is None,
        "rejection_reason": rejection,
        "second_grasp_digit_eligible": rejection is None,
    }


def evaluate_two_free_palmar_candidate(
    phase2s: Phase2SConfig,
    phase2: Phase2Config,
    source: dict,
    attempt_index: int,
    seed: int,
) -> dict:
    """Perturb a validated two-finger palmar basin and require the exact mask."""

    cfg = load_configs(scene_filename=phase2s.scene_filename)
    model, data = build_scene(cfg)
    indices = resolve_hand_indices(model, cfg.hand)
    qadr, vadr = _object_addresses(model)
    open_joint = np.asarray(source["open_joint_configuration_rad"], dtype=float)
    acquisition_start = np.asarray(source["final_joint_configuration_rad"], dtype=float)
    source_target = np.asarray(source["retaining_joint_target_rad"], dtype=float)
    expected_mask = np.asarray(source["occupied_finger_mask"], dtype=bool)
    active = np.flatnonzero(np.repeat(expected_mask, 4))
    inactive = np.flatnonzero(~np.repeat(expected_mask, 4))
    rng = np.random.default_rng(np.random.SeedSequence([seed, attempt_index]))
    focused = attempt_index % 5 != 0
    scale = float(rng.uniform(0.96, 1.05) if focused else rng.uniform(0.85, 1.15))
    target_joint = open_joint.copy()
    target_joint[active] = open_joint[active] + scale * (source_target[active] - open_joint[active])
    target_joint[active] += rng.uniform(-0.012 if focused else -0.04, 0.012 if focused else 0.04, len(active))
    target_joint[inactive] = open_joint[inactive]
    target_joint = np.clip(target_joint, model.jnt_range[indices.joint_ids, 0], model.jnt_range[indices.joint_ids, 1])
    span = 0.0006 if focused else 0.002
    fixture_position = np.asarray(source["final_object_position_m"], dtype=float) + rng.uniform(-span, span, 3)
    angle_span = 0.02 if focused else 0.07
    base_rotation = Rotation.from_quat(source["final_object_quaternion"], scalar_first=True)
    fixture_quaternion = (Rotation.from_rotvec(rng.uniform(-angle_span, angle_span, 3)) * base_rotation).as_quat(scalar_first=True)
    data.qpos[indices.qpos_addresses] = acquisition_start
    data.qvel[indices.qvel_addresses] = 0.0
    data.qpos[qadr:qadr + 3] = fixture_position
    data.qpos[qadr + 3:qadr + 7] = fixture_quaternion
    data.qvel[vadr:vadr + 6] = 0.0
    mujoco.mj_forward(model, data)
    controller = JointImpedanceController(
        cfg.task.impedance_stiffness, cfg.task.impedance_damping, cfg.task.torque_limit,
    )
    close_steps = phase2s.state.fixture_close_steps
    contact_steps = phase2s.state.fixture_contact_steps
    for step in range(close_steps + contact_steps):
        alpha = min(1.0, (step + 1) / close_steps)
        desired = (1.0 - alpha) * acquisition_start + alpha * target_joint
        q, qvel = hand_state(data, indices)
        data.ctrl[indices.actuator_ids] = controller.torque(desired, q, qvel)
        data.qpos[qadr:qadr + 3] = fixture_position
        data.qpos[qadr + 3:qadr + 7] = fixture_quaternion
        data.qvel[vadr:vadr + 6] = 0.0
        mujoco.mj_step(model, data)
    data.qpos[qadr:qadr + 3] = fixture_position
    data.qpos[qadr + 3:qadr + 7] = fixture_quaternion
    data.qvel[vadr:vadr + 6] = 0.0
    mujoco.mj_forward(model, data)
    measured = measure_stable_hold(
        cfg, model, data, indices, target_joint, phase2s.state.stable_hold_steps,
        phase2.resources, phase2.dataset.friction_cone_edges,
        phase2.dataset.convex_hull_tolerance,
    )
    occupied_pair = tuple(finger for finger, flag in zip(FINGER_ORDER, expected_mask) if flag)
    free_pair = tuple(finger for finger, flag in zip(FINGER_ORDER, expected_mask) if not flag)
    record = {
        **source,
        **measured,
        "grasp_state_id": f"phase2T_palmar_2free_{attempt_index:05d}",
        "grasp_state_subtype": "PALMAR_SECURED_2FREE",
        "source_phase2s_grasp_state_id": source["grasp_state_id"],
        "support_pair": list(occupied_pair),
        "free_finger_set": list(free_pair),
        "actual_A_contact_topology": [
            finger for finger, flag in zip(FINGER_ORDER, measured["per_finger_A_contact_flags"]) if flag
        ],
        "targeted_attempt_index": int(attempt_index),
        "sampling_mode": "focused_two_free_palmar_basin" if focused else "broad_two_free_palmar_basin",
        "initial_object_position_m": fixture_position.tolist(),
        "initial_object_quaternion": fixture_quaternion.tolist(),
        "open_joint_configuration_rad": open_joint.tolist(),
        "retaining_joint_target_rad": target_joint.tolist(),
        "target_closure_scale": scale,
        "fixture_method": "temporary_free_joint_pose_reset_during_initialization_only",
        "fixture_release_timestep": close_steps + contact_steps,
        "revalidated_with_half_scale_geometry": True,
    }
    classified = classify_grasp_state(
        record, GraspStateType.PALMAR_SECURED, phase2s.state,
        phase2.dataset.convex_hull_tolerance,
    )
    exact_mask = measured["occupied_finger_mask"] == expected_mask.tolist()
    extra_checks = {
        "exactly_two_load_bearing_fingers": measured["occupied_finger_count"] == 2,
        "occupied_pair_matches_target": exact_mask,
        "exactly_two_free_fingers": measured["free_finger_count"] == 2,
    }
    checks = {**classified["checks"], **extra_checks}
    rejection = next((name for name, passed in checks.items() if not passed), None)
    return {
        **classified,
        "checks": checks,
        "accepted": rejection is None,
        "rejection_reason": rejection,
        "second_grasp_digit_eligible": rejection is None,
    }
