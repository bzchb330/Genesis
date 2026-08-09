from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import mujoco
import numpy as np

from ..control import JointImpedanceController, hand_state
from ..config import ROOT
from ..diagnostics.grasp_search import load_search_config
from ..diagnostics.scripted_grasp import _joint_target
from ..sensing import compute_phase2_tactile_features, extract_contacts, group_contacts_by_finger
from .resource_components import FINGER_ORDER, reconstruct_grasp
from .resumable import stable_trial_id


OUTCOMES = ("BOTH_RETAINED", "A_DROPPED", "B_NOT_ACQUIRED", "BOTH_LOST", "INVALID")


def fixture_is_active(step: int, release_timestep: int) -> bool:
    return step < release_timestep


def correlation_trial_id(record: dict, placement_index: int, pilot_only: bool) -> str:
    return stable_trial_id("phase2-second-grasp", {
        "mode": "pilot" if pilot_only else "formal",
        "grasp_id": record["grasp_id"],
        "placement_index": placement_index,
        "config_hash": record["config_hash"],
    })


def formal_nonpilot_records(rows: list[dict]) -> list[dict]:
    """Defense-in-depth filter used by every inferential analysis."""

    return [row for row in rows if row.get("pilot_only") is False]


@dataclass(frozen=True)
class BPlacement:
    index: int
    position_m: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]
    yaw_rad: float


def cylinder_lowest_point_z(center, quaternion, radius: float, half_length: float) -> float:
    """Exact world-z support point for a finite cylinder with local z axis."""

    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=float))
    axis_z = abs(matrix.reshape(3, 3)[2, 2])
    vertical_radius = radius * math.sqrt(max(0.0, 1.0 - axis_z * axis_z))
    return float(center[2] - half_length * axis_z - vertical_radius)


def placement_candidate(cfg, second_grasp, placement_index: int, rejection_index: int = 0) -> BPlacement:
    rng = np.random.default_rng(np.random.SeedSequence([second_grasp.seed, placement_index, rejection_index]))
    x = rng.uniform(*second_grasp.B_center_x_bounds_m)
    y = rng.uniform(*second_grasp.B_center_y_bounds_m)
    z = rng.uniform(*second_grasp.B_center_z_bounds_m)
    yaw = rng.uniform(*second_grasp.B_yaw_bounds_rad)
    quaternion = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    return BPlacement(placement_index, (x, y, z), quaternion, yaw)


def _set_b_pose(model, data, placement: BPlacement) -> tuple[int, int, int]:
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object_b")
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_b_geom")
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_b_free")
    qadr, vadr = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
    data.qpos[qadr:qadr + 3] = placement.position_m
    data.qpos[qadr + 3:qadr + 7] = placement.quaternion
    data.qvel[vadr:vadr + 6] = 0.0
    mujoco.mj_forward(model, data)
    return body, geom, joint


def placement_is_valid(record: dict, placement: BPlacement, penetration_tolerance_m: float) -> tuple[bool, str | None]:
    _, model, data, _ = reconstruct_grasp(record)
    _, b_geom, _ = _set_b_pose(model, data, placement)
    for geom_id in range(model.ngeom):
        if geom_id == b_geom:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name == "table":
            continue
        if not (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]):
            continue
        distance = mujoco.mj_geomDistance(model, data, b_geom, geom_id, 1.0, None)
        invalid_distance = 0.0 if name == "object_a_geom" else -penetration_tolerance_m
        if distance < invalid_distance:
            return False, f"initial_overlap:{name or geom_id}"
    return True, None


def valid_placement(record: dict, cfg, second_grasp, placement_index: int, maximum_rejections: int = 1000) -> BPlacement:
    for rejection_index in range(maximum_rejections):
        placement = placement_candidate(cfg, second_grasp, placement_index, rejection_index)
        valid, _ = placement_is_valid(record, placement, second_grasp.maximum_penetration_m)
        if valid:
            return placement
    raise RuntimeError(f"could not sample valid B placement {placement_index}")


def placement_is_approachable(record: dict, placement: BPlacement, occupied_mask, maximum_iterations: int = 120) -> bool:
    """Geometry-only DLS reach check using actual fingertip and B collision geoms."""

    cfg, model, data, indices = reconstruct_grasp(record)
    _, b_geom, _ = _set_b_pose(model, data, placement)
    free = ~np.asarray(occupied_mask, dtype=bool)
    finger_joints = _free_joint_indices(cfg, free)
    body_ids = {
        finger: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.hand.fingertip_bodies[index])
        for index, finger in enumerate(FINGER_ORDER) if free[index]
    }
    tip_geoms = {
        finger: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, cfg.hand.finger_geom_mapping[finger][0])
        for finger in body_ids
    }
    if not body_ids:
        return False
    centre = np.asarray(placement.position_m)
    object_b = next(obj for obj in cfg.scene.objects if obj.name == "object_b")
    radius, half_length = object_b.size[0], object_b.size[1]
    targets = {}
    for finger, body_id in body_ids.items():
        direction = data.xpos[body_id, :2] - centre[:2]
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 1e-9 else np.array([1.0, 0.0])
        tip_radius = model.geom_size[tip_geoms[finger], 0]
        contact_z = centre[2] + half_length - tip_radius
        targets[finger] = np.r_[centre[:2] + direction * radius, contact_z]
    desired = data.qpos[indices.qpos_addresses].copy()
    ranges = model.jnt_range[indices.joint_ids]
    for _ in range(maximum_iterations):
        _ik_update(model, data, indices, desired, targets, body_ids, finger_joints, 0.01, 0.25)
        desired = np.clip(desired, ranges[:, 0], ranges[:, 1])
        data.qpos[indices.qpos_addresses] = desired
        mujoco.mj_forward(model, data)
    return any(mujoco.mj_geomDistance(model, data, geom_id, b_geom, 1.0, None) <= 0.003 for geom_id in tip_geoms.values())


def _object_finger_state(grouped, object_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts, normal, tangential = [], [], []
    for finger in FINGER_ORDER:
        contacts = [row for row in grouped[finger] if object_name in {row.body1_name, row.body2_name}]
        counts.append(len(contacts))
        normal.append(sum(row.normal_force for row in contacts))
        tangential.append(sum(row.tangential_force for row in contacts))
    return np.asarray(counts), np.asarray(normal), np.asarray(tangential)


def _b_hand_state(contacts) -> tuple[int, float]:
    """Count all B contacts supported by the hand, including the palm."""

    rows = []
    for row in contacts:
        geoms = {row.geom1_name, row.geom2_name}
        bodies = {row.body1_name, row.body2_name}
        if "object_b_geom" not in geoms and "object_b" not in bodies:
            continue
        other_geoms = geoms - {"object_b_geom"}
        other_bodies = bodies - {"object_b"}
        if other_geoms & {"table", "object_a_geom"} or other_bodies & {"world", "object_a"}:
            continue
        rows.append(row)
    return len(rows), float(sum(row.normal_force for row in rows))


def classify_B_acquisition(
    *, fixture_released: bool, final_free_finger_contacts: int,
    final_hand_contacts: int, final_hand_normal_force_N: float,
    table_contact: bool, complete_hand_contact_loss: bool,
    maximum_penetration_m: float, maximum_translation_m: float,
    maximum_orientation_rad: float, numerically_stable: bool, criteria,
) -> bool:
    """Apply the frozen PI functional B-acquisition conjunction."""

    return bool(
        fixture_released
        and final_free_finger_contacts >= criteria.minimum_B_free_finger_contacts
        and final_hand_contacts >= criteria.minimum_B_hand_contacts
        and final_hand_normal_force_N > criteria.minimum_B_normal_force_N
        and not table_contact
        and not complete_hand_contact_loss
        and maximum_penetration_m <= criteria.maximum_penetration_m
        and maximum_translation_m <= criteria.maximum_B_translation_m
        and maximum_orientation_rad <= criteria.maximum_B_orientation_rad
        and numerically_stable
    )


def _rotation_change(quaternion, reference) -> float:
    return float(2 * np.arccos(np.clip(abs(float(np.dot(quaternion, reference))), 0, 1)))


def _free_joint_indices(cfg, free_mask: np.ndarray) -> dict[str, np.ndarray]:
    groups = load_search_config()["finger_groups"]
    by_name = {name: index for index, name in enumerate(cfg.hand.actuator_names)}
    return {
        finger: np.asarray([by_name[name] for name in groups[finger]], dtype=int)
        for index, finger in enumerate(FINGER_ORDER) if free_mask[index]
    }


def _ik_update(model, data, indices, desired_q, targets, body_ids, finger_joints, damping, step_size):
    for finger, target in targets.items():
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, body_ids[finger])
        local = finger_joints[finger]
        dofs = indices.qvel_addresses[local]
        jac = jacp[:, dofs]
        error = np.asarray(target) - data.xpos[body_ids[finger]]
        delta = jac.T @ np.linalg.solve(jac @ jac.T + damping ** 2 * np.eye(3), error)
        desired_q[local] += step_size * delta


def run_second_grasp_trial(
    record: dict,
    resource_record: dict,
    phase2,
    placement_index: int,
    tactile_output_path: Path,
) -> dict:
    cfg, model, data, indices = reconstruct_grasp(record)
    # The sampled pose is never conditioned on the A grasp. Invalid overlaps are
    # classified INVALID rather than rejection-resampled, preserving one global
    # distribution for every A grasp.
    placement = placement_candidate(cfg, phase2.second_grasp, placement_index)
    placement_valid, placement_invalid_reason = placement_is_valid(
        record, placement, phase2.second_grasp.maximum_penetration_m,
    )
    b_body, _, b_joint = _set_b_pose(model, data, placement)
    b_qadr, b_vadr = model.jnt_qposadr[b_joint], model.jnt_dofadr[b_joint]
    a_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object_a")
    a_reference_position = data.xpos[a_body].copy()
    a_reference_quaternion = data.xquat[a_body].copy()
    occupied = np.asarray(resource_record["occupied_finger_mask"], dtype=bool)
    free = ~occupied
    finger_joints = _free_joint_indices(cfg, free)
    body_ids = {
        finger: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.hand.fingertip_bodies[index])
        for index, finger in enumerate(FINGER_ORDER) if free[index]
    }
    initial_b = np.asarray(placement.position_m)
    object_b_cfg = next(obj for obj in cfg.scene.objects if obj.name == "object_b")
    radial = {}
    for finger, body_id in body_ids.items():
        direction = data.xpos[body_id, :2] - initial_b[:2]
        norm = np.linalg.norm(direction)
        radial[finger] = direction / norm if norm > 1e-9 else np.array([1.0, 0.0])
    tip_geoms = {
        finger: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, cfg.hand.finger_geom_mapping[finger][0])
        for finger in body_ids
    }
    contact_z = {
        finger: initial_b[2] + object_b_cfg.size[1] - model.geom_size[tip_geoms[finger], 0]
        for finger in body_ids
    }
    approach_targets = {
        finger: np.r_[initial_b[:2] + direction * (object_b_cfg.size[0] + 0.008), contact_z[finger]]
        for finger, direction in radial.items()
    }
    close_targets = {
        finger: np.r_[initial_b[:2] + direction * (object_b_cfg.size[0] - 0.001), contact_z[finger] + 0.008]
        for finger, direction in radial.items()
    }
    # Reconstruct the impedance set point used by the accepted Part B hold.
    # The measured final q is state, not the torque-producing controller target.
    desired_q = _joint_target(model, cfg, indices, record["hold_joint_fractions"])
    joint_ranges = model.jnt_range[indices.joint_ids]
    controller = JointImpedanceController(cfg.task.impedance_stiffness, cfg.task.impedance_damping, cfg.task.torque_limit)
    finger_mapping = cfg.hand.finger_geom_mapping
    phase_names, a_counts_ts, a_normal_ts, b_counts_ts, b_normal_ts = [], [], [], [], []
    tactile_binary_ts, tactile_normal_ts, tactile_ratio_ts, b_clearance_ts = [], [], [], []
    b_hand_contacts_ts, b_hand_force_ts, fixture_active_ts = [], [], []
    b_position_ts, b_quaternion_ts = [], []
    a_translation_ts, a_rotation_ts = [], []
    invalid_reason = None if placement_valid else placement_invalid_reason
    first_failure_timestep = None
    first_triggering_condition = None
    max_penetration = 0.0
    max_a_penetration = 0.0
    max_b_penetration = 0.0
    a_table_any = False
    a_zero_contact_any = False
    final_hold_start = phase2.second_grasp.approach_steps + phase2.second_grasp.close_steps
    fixture_release_timestep = final_hold_start
    fixture_released = False
    b_release_position = None
    b_release_quaternion = None
    total_steps = final_hold_start + phase2.second_grasp.final_hold_steps
    for step in range(total_steps):
        if step < phase2.second_grasp.approach_steps:
            phase = "approach"
            alpha = (step + 1) / phase2.second_grasp.approach_steps
            targets = {
                finger: (1 - alpha) * data.xpos[body_ids[finger]] + alpha * approach_targets[finger]
                for finger in body_ids
            }
        elif step < final_hold_start:
            phase = "close"
            alpha = (step - phase2.second_grasp.approach_steps + 1) / phase2.second_grasp.close_steps
            targets = {finger: (1 - alpha) * approach_targets[finger] + alpha * close_targets[finger] for finger in body_ids}
        else:
            phase = "final_hold"
            targets = close_targets
        if body_ids:
            _ik_update(model, data, indices, desired_q, targets, body_ids, finger_joints, phase2.second_grasp.ik_damping, phase2.second_grasp.ik_step_size)
        desired_q = np.clip(desired_q, joint_ranges[:, 0], joint_ranges[:, 1])
        q, qvel = hand_state(data, indices)
        data.ctrl[indices.actuator_ids] = controller.torque(desired_q, q, qvel)
        fixture_active = fixture_is_active(step, fixture_release_timestep)
        if fixture_active:
            data.qpos[b_qadr:b_qadr + 3] = placement.position_m
            data.qpos[b_qadr + 3:b_qadr + 7] = placement.quaternion
            data.qvel[b_vadr:b_vadr + 6] = 0.0
        elif not fixture_released:
            fixture_released = True
            b_release_position = data.xpos[b_body].copy()
            b_release_quaternion = data.xquat[b_body].copy()
        mujoco.mj_step(model, data)
        finite = all(np.all(np.isfinite(value)) for value in (data.qpos, data.qvel, data.ctrl))
        contacts = extract_contacts(model, data)
        grouped = group_contacts_by_finger(contacts, finger_mapping)
        a_counts, a_normal, _ = _object_finger_state(grouped, "object_a")
        b_counts, b_normal, _ = _object_finger_state(grouped, "object_b")
        b_hand_contacts, b_hand_force = _b_hand_state(contacts)
        b_grouped = {
            finger: [row for row in grouped[finger] if "object_b" in {row.body1_name, row.body2_name}]
            for finger in FINGER_ORDER
        }
        tactile = compute_phase2_tactile_features(
            b_grouped, FINGER_ORDER, phase2.tactile.binary_contact_threshold_N, phase2.tactile.zero_normal_epsilon_N,
        )
        a_table = any({row.geom1_name, row.geom2_name} == {"object_a_geom", "table"} for row in contacts)
        a_table_any |= a_table
        a_zero_contact_any |= int(np.sum(a_counts)) == 0
        relevant_distances = [
            row.distance for row in contacts
            if "object_a" in {row.body1_name, row.body2_name} or "object_b" in {row.body1_name, row.body2_name}
        ]
        penetration = max([0.0, *[-distance for distance in relevant_distances]])
        max_penetration = max(max_penetration, penetration)
        a_penetration = max([0.0, *[-row.distance for row in contacts if "object_a" in {row.body1_name, row.body2_name}]])
        b_penetration = max([0.0, *[-row.distance for row in contacts if "object_b" in {row.body1_name, row.body2_name}]])
        max_a_penetration = max(max_a_penetration, a_penetration)
        max_b_penetration = max(max_b_penetration, b_penetration)
        a_translation = float(np.linalg.norm(data.xpos[a_body] - a_reference_position))
        a_rotation = _rotation_change(data.xquat[a_body], a_reference_quaternion)
        a_translation_ts.append(a_translation)
        a_rotation_ts.append(a_rotation)
        a_loss_now = (
            a_table or int(np.sum(a_counts > 0)) < phase2.second_grasp.minimum_A_finger_contacts
            or float(np.sum(a_normal)) <= phase2.second_grasp.minimum_A_normal_force_N
            or a_translation > phase2.second_grasp.maximum_A_translation_m
            or a_rotation > phase2.second_grasp.maximum_A_orientation_rad
        )
        if a_loss_now and first_failure_timestep is None:
            first_failure_timestep = step
            first_triggering_condition = "A_retention_bound"
        if not finite:
            invalid_reason = "nonfinite_state"
        phase_names.append(phase)
        a_counts_ts.append(a_counts)
        a_normal_ts.append(a_normal)
        b_counts_ts.append(b_counts)
        b_normal_ts.append(b_normal)
        tactile_binary_ts.append(tactile["binary_contact"])
        tactile_normal_ts.append(tactile["normal_force_N"])
        tactile_ratio_ts.append(tactile["tangential_to_normal_ratio"])
        b_hand_contacts_ts.append(b_hand_contacts)
        b_hand_force_ts.append(b_hand_force)
        fixture_active_ts.append(fixture_active)
        b_position_ts.append(data.xpos[b_body].copy())
        b_quaternion_ts.append(data.xquat[b_body].copy())
        b_clearance_ts.append(cylinder_lowest_point_z(data.xpos[b_body], data.xquat[b_body], object_b_cfg.size[0], object_b_cfg.size[1]) - (cfg.scene.table_pos[2] + cfg.scene.table_size[2]))
        if invalid_reason is not None:
            if first_failure_timestep is None:
                first_failure_timestep, first_triggering_condition = step, invalid_reason
            break
    a_counts_array, a_normal_array = np.asarray(a_counts_ts), np.asarray(a_normal_ts)
    b_counts_array, b_normal_array = np.asarray(b_counts_ts), np.asarray(b_normal_ts)
    final_slice = slice(final_hold_start, len(phase_names))
    final_a_contacts = int(np.sum(a_counts_array[-1] > 0))
    final_a_force = float(np.sum(a_normal_array[-1]))
    final_b_contacts = int(np.sum((b_counts_array[-1] > 0) & free))
    final_b_force = float(b_hand_force_ts[-1])
    final_b_hand_contacts = int(b_hand_contacts_ts[-1])
    a_translation = float(np.linalg.norm(data.xpos[a_body] - a_reference_position))
    a_rotation = _rotation_change(data.xquat[a_body], a_reference_quaternion)
    a_retained = (
        invalid_reason is None and not a_table_any and not a_zero_contact_any
        and max_a_penetration <= phase2.second_grasp.maximum_penetration_m
        and a_translation <= phase2.second_grasp.maximum_A_translation_m
        and a_rotation <= phase2.second_grasp.maximum_A_orientation_rad
        and final_a_contacts >= phase2.second_grasp.minimum_A_finger_contacts
        and final_a_force > phase2.second_grasp.minimum_A_normal_force_N
    )
    complete_final_hold = len(phase_names) == total_steps
    b_contact_loss = not complete_final_hold or bool(np.any(np.asarray(b_hand_contacts_ts)[final_slice] == 0))
    b_table_contact = any(clearance <= 0.0 for clearance in np.asarray(b_clearance_ts)[final_slice]) if complete_final_hold else True
    if b_release_position is None:
        b_translation, b_rotation = math.inf, math.inf
    else:
        positions = np.asarray(b_position_ts)[final_slice]
        quaternions = np.asarray(b_quaternion_ts)[final_slice]
        b_translation = float(np.max(np.linalg.norm(positions - b_release_position, axis=1)))
        b_rotation = float(np.max([_rotation_change(quaternion, b_release_quaternion) for quaternion in quaternions]))
    b_acquired = classify_B_acquisition(
        fixture_released=fixture_released and complete_final_hold and not any(fixture_active_ts[final_slice]),
        final_free_finger_contacts=final_b_contacts,
        final_hand_contacts=final_b_hand_contacts,
        final_hand_normal_force_N=final_b_force,
        table_contact=b_table_contact,
        complete_hand_contact_loss=b_contact_loss,
        maximum_penetration_m=max_b_penetration,
        maximum_translation_m=b_translation,
        maximum_orientation_rad=b_rotation,
        numerically_stable=invalid_reason is None,
        criteria=phase2.second_grasp,
    )
    if invalid_reason is not None:
        outcome = "INVALID"
    elif a_retained and b_acquired:
        outcome = "BOTH_RETAINED"
    elif not a_retained and b_acquired:
        outcome = "A_DROPPED"
    elif a_retained and not b_acquired:
        outcome = "B_NOT_ACQUIRED"
    else:
        outcome = "BOTH_LOST"
    if first_failure_timestep is None and outcome != "BOTH_RETAINED":
        first_failure_timestep = len(phase_names) - 1
        first_triggering_condition = "B_not_acquired" if a_retained else "A_retention_bound"
    tactile_output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        tactile_output_path,
        phase=np.asarray(phase_names),
        A_finger_contact_count=a_counts_array,
        A_finger_normal_force_N=a_normal_array,
        B_finger_contact_count=b_counts_array,
        B_finger_normal_force_N=b_normal_array,
        B_tactile_binary_contact=np.asarray(tactile_binary_ts),
        B_tactile_normal_force_N=np.asarray(tactile_normal_ts),
        B_tactile_tangential_to_normal_ratio=np.asarray(tactile_ratio_ts),
        B_table_clearance_m=np.asarray(b_clearance_ts),
        B_hand_contact_count=np.asarray(b_hand_contacts_ts),
        B_hand_normal_force_N=np.asarray(b_hand_force_ts),
        B_fixture_active=np.asarray(fixture_active_ts),
        B_position_m=np.asarray(b_position_ts),
        B_quaternion=np.asarray(b_quaternion_ts),
        A_translation_m=np.asarray(a_translation_ts),
        A_orientation_rad=np.asarray(a_rotation_ts),
    )
    return {
        "outcome": outcome,
        "placement": {"index": placement.index, "position_m": list(placement.position_m), "quaternion": list(placement.quaternion), "yaw_rad": placement.yaw_rad},
        "first_failure_timestep": first_failure_timestep,
        "first_failure_phase": None if first_failure_timestep is None else phase_names[first_failure_timestep],
        "first_triggering_condition": first_triggering_condition,
        "invalid_reason": invalid_reason,
        "fixture": {"method": "kinematic_free_joint_pose_support", "release_timestep": fixture_release_timestep, "released": fixture_released, "active_during_final_hold": bool(any(fixture_active_ts[final_slice])) if complete_final_hold else None},
        "final_A_state": {"retained": a_retained, "finger_contacts": final_a_contacts, "normal_force_N": final_a_force, "per_finger_initial_normal_force_N": a_normal_array[0].tolist(), "per_finger_final_normal_force_N": a_normal_array[-1].tolist(), "per_finger_force_redistribution_N": (a_normal_array[-1] - a_normal_array[0]).tolist(), "translation_m": a_translation, "orientation_rad": a_rotation, "complete_contact_loss": a_zero_contact_any, "table_recontact": a_table_any},
        "final_B_state": {"acquired": b_acquired, "free_finger_contacts": final_b_contacts, "hand_supporting_contacts": final_b_hand_contacts, "normal_force_N": final_b_force, "complete_hand_contact_loss": b_contact_loss, "translation_after_release_m": b_translation if np.isfinite(b_translation) else None, "orientation_after_release_rad": b_rotation if np.isfinite(b_rotation) else None, "table_contact": b_table_contact, "minimum_final_hold_table_clearance_m": float(np.min(np.asarray(b_clearance_ts)[final_slice])) if complete_final_hold else None},
        "maximum_penetration_m": max_penetration,
        "maximum_A_penetration_m": max_a_penetration,
        "maximum_B_penetration_m": max_b_penetration,
        "resource_components": {key: resource_record[key] for key in ("occupied_finger_count", "occupied_finger_mask", "free_finger_workspace_vol_m3", "free_palm_volume_m3")},
        "tactile_time_series_path": tactile_output_path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "steps_completed": len(phase_names),
    }
