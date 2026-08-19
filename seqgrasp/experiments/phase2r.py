from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable

import mujoco
import numpy as np

from ..config import ConfigBundle
from ..control import JointImpedanceController, hand_state
from ..diagnostics.grasp_search import load_search_config
from ..sensing import extract_contacts
from .grasp_sampling import ferrari_canny_epsilon
from .resource_components import FINGER_ORDER, PALM_REFERENCE_TO_COMPILED
from .resumable import stable_trial_id


class GraspStateType(str, Enum):
    FINGERTIP = "FINGERTIP"
    PALMAR_SECURED = "PALMAR_SECURED"


PHASE2R_OUTCOMES = ("BOTH_RETAINED", "A_DROPPED", "B_NOT_ACQUIRED", "BOTH_LOST", "INVALID")
PHASE2R_EXPERIMENT_ID = "phase2R_palmar_vs_fingertip_formal"

REQUIRED_STATE_FIELDS = (
    "grasp_state_id", "grasp_state_type", "object_A_COM_palm_reference_m",
    "object_A_COM_palm_compiled_m", "COM_to_palm_origin_distance_m",
    "COM_to_palm_surface_distance_m", "palm_A_contact",
    "palm_A_contact_fraction", "palm_A_normal_force_N", "palm_A_contact_count",
    "per_finger_A_contact_flags", "per_finger_A_normal_force_N",
    "occupied_finger_count", "free_finger_count", "ferrari_canny_epsilon",
    "A_translation_drift_m", "A_rotation_drift_rad", "A_vertical_drift_m",
    "maximum_penetration_m", "minimum_joint_margin_rad",
    "maximum_actuator_utilization", "fixture_removed_before_validation",
    "equality_constraint_count", "final_joint_configuration_rad",
    "final_object_position_m", "final_object_quaternion",
)


def validate_grasp_state_schema(record: dict) -> None:
    missing = [key for key in REQUIRED_STATE_FIELDS if key not in record]
    if missing:
        raise ValueError(f"grasp state is missing required fields: {missing}")
    try:
        GraspStateType(record["grasp_state_type"])
    except ValueError as exc:
        raise ValueError("grasp_state_type must be FINGERTIP or PALMAR_SECURED") from exc
    for key in ("per_finger_A_contact_flags", "per_finger_A_normal_force_N"):
        if len(record[key]) != len(FINGER_ORDER):
            raise ValueError(f"{key} must follow the four-finger configured order")
    if int(record["occupied_finger_count"]) + int(record["free_finger_count"]) != len(FINGER_ORDER):
        raise ValueError("occupied and free finger counts must sum to four")


def second_grasp_digit_eligible(record: dict) -> bool:
    return int(record["free_finger_count"]) >= 2


def digit_precheck_outcome(record: dict) -> dict | None:
    if second_grasp_digit_eligible(record):
        return None
    return {
        "outcome": "B_NOT_ACQUIRED",
        "outcome_subreason": "INSUFFICIENT_FREE_DIGITS_PRECHECK",
        "dynamic_attempt_executed": False,
    }


def _finger_prefixes(cfg: ConfigBundle) -> dict[str, str]:
    return {
        finger: cfg.hand.fingertip_bodies[index].split("_", 1)[0]
        for index, finger in enumerate(FINGER_ORDER)
    }


def _object_hand_contacts(model, data, cfg: ConfigBundle, object_name: str = "object_a") -> dict:
    palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.hand.palm_body)
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
    prefixes = _finger_prefixes(cfg)
    finger_records: dict[str, list] = {finger: [] for finger in FINGER_ORDER}
    distal_records: dict[str, list] = {finger: [] for finger in FINGER_ORDER}
    palm_records = []
    hand_records = []
    for contact in extract_contacts(model, data):
        if object_name not in {contact.body1_name, contact.body2_name}:
            continue
        other_id = contact.body2_id if contact.body1_id == object_id else contact.body1_id
        other_name = contact.body2_name if contact.body1_id == object_id else contact.body1_name
        if other_id == palm_id:
            palm_records.append(contact)
            hand_records.append(contact)
            continue
        for finger, prefix in prefixes.items():
            if other_name == prefix or other_name.startswith(prefix + "_"):
                finger_records[finger].append(contact)
                hand_records.append(contact)
                if other_name.endswith("_distal") or other_name.endswith("_tip"):
                    distal_records[finger].append(contact)
                break
    center = np.asarray(data.xpos[object_id])
    positions, inward = [], []
    for record in hand_records:
        normal = np.asarray(record.normal).copy()
        if np.dot(normal, center - record.position) < 0:
            normal = -normal
        positions.append(np.asarray(record.position).copy())
        inward.append(normal)
    return {
        "palm_count": len(palm_records),
        "palm_force_N": float(sum(row.normal_force for row in palm_records)),
        "finger_counts": np.asarray([len(finger_records[f]) for f in FINGER_ORDER], dtype=int),
        "finger_force_N": np.asarray([sum(row.normal_force for row in finger_records[f]) for f in FINGER_ORDER], dtype=float),
        "distal_counts": np.asarray([len(distal_records[f]) for f in FINGER_ORDER], dtype=int),
        "hand_count": len(hand_records),
        "positions": np.asarray(positions, dtype=float).reshape(-1, 3),
        "inward_normals": np.asarray(inward, dtype=float).reshape(-1, 3),
        "maximum_penetration_m": float(max((max(0.0, -row.distance) for row in hand_records), default=0.0)),
    }


def _rotation_drift(quaternions: np.ndarray) -> float:
    if len(quaternions) == 0:
        return math.inf
    dots = np.abs(quaternions @ quaternions[0])
    return float(np.max(2.0 * np.arccos(np.clip(dots, 0.0, 1.0))))


def _palm_surface_distance(model, data, cfg: ConfigBundle) -> float:
    object_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_a_geom")
    palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.hand.palm_body)
    palm_geoms = [index for index in range(model.ngeom) if int(model.geom_bodyid[index]) == palm_id]
    return float(min(mujoco.mj_geomDistance(model, data, object_geom, geom, 1.0, None) for geom in palm_geoms))


def _inside_existing_palm_box(com_reference: np.ndarray, resources) -> bool:
    return bool(
        np.all(com_reference >= np.asarray(resources.free_palm_box_lower_m, dtype=float))
        and np.all(com_reference <= np.asarray(resources.free_palm_box_upper_m, dtype=float))
    )


def measure_stable_hold(
    cfg: ConfigBundle,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    indices,
    hold_joint_rad: np.ndarray,
    hold_steps: int,
    resources,
    friction_cone_edges: int,
    convex_hull_tolerance: float,
) -> dict:
    """Measure an unsupported endpoint state; no fixture operation occurs here."""

    controller = JointImpedanceController(
        cfg.task.impedance_stiffness, cfg.task.impedance_damping, cfg.task.torque_limit,
    )
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object_a")
    palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.hand.palm_body)
    table_top = cfg.scene.table_pos[2] + cfg.scene.table_size[2]
    object_cfg = next(item for item in cfg.scene.objects if item.name == "object_a")
    half_height = object_cfg.size[2]
    joint_limits = model.jnt_range[indices.joint_ids]
    rows = []
    for _ in range(hold_steps):
        q, qvel = hand_state(data, indices)
        data.ctrl[indices.actuator_ids] = controller.torque(hold_joint_rad, q, qvel)
        mujoco.mj_step(model, data)
        contact = _object_hand_contacts(model, data, cfg)
        q, _ = hand_state(data, indices)
        palm_rotation = data.xmat[palm_id].reshape(3, 3)
        com_compiled = (data.xpos[object_id] - data.xpos[palm_id]) @ palm_rotation
        com_reference = com_compiled @ PALM_REFERENCE_TO_COMPILED
        object_table_contact = any(
            {row.geom1_name, row.geom2_name} == {"object_a_geom", "table"}
            for row in extract_contacts(model, data)
        )
        rows.append({
            "position": data.xpos[object_id].copy(),
            "quaternion": data.xquat[object_id].copy(),
            "palm_count": contact["palm_count"],
            "palm_force_N": contact["palm_force_N"],
            "finger_counts": contact["finger_counts"],
            "finger_force_N": contact["finger_force_N"],
            "distal_counts": contact["distal_counts"],
            "hand_count": contact["hand_count"],
            "maximum_penetration_m": contact["maximum_penetration_m"],
            "contact_positions": contact["positions"],
            "contact_normals": contact["inward_normals"],
            "table_contact": object_table_contact,
            "table_clearance_m": float(data.xpos[object_id][2] - half_height - table_top),
            "com_compiled": com_compiled,
            "com_reference": com_reference,
            "palm_surface_distance_m": _palm_surface_distance(model, data, cfg),
            "joint_positions": q.copy(),
            "joint_margins": np.minimum(q - joint_limits[:, 0], joint_limits[:, 1] - q),
            "actuator_utilization": np.abs(data.ctrl[indices.actuator_ids]) / cfg.task.torque_limit,
            "numerical": all(np.all(np.isfinite(value)) for value in (data.qpos, data.qvel, data.ctrl)),
        })
    if not rows:
        raise ValueError("stable hold measurement requires at least one step")
    positions = np.asarray([row["position"] for row in rows])
    quaternions = np.asarray([row["quaternion"] for row in rows])
    finger_forces = np.asarray([row["finger_force_N"] for row in rows])
    finger_counts = np.asarray([row["finger_counts"] for row in rows])
    distal_counts = np.asarray([row["distal_counts"] for row in rows])
    mean_forces = np.mean(finger_forces, axis=0)
    occupied = mean_forces > resources.occupied_finger_normal_force_threshold_N
    final = rows[-1]
    epsilon = ferrari_canny_epsilon(
        final["contact_positions"], final["contact_normals"], final["position"],
        object_cfg.friction[0], friction_cone_edges, float(np.linalg.norm(object_cfg.size)),
        convex_hull_tolerance,
    )
    return {
        "object_A_COM_palm_reference_m": final["com_reference"].tolist(),
        "object_A_COM_palm_compiled_m": final["com_compiled"].tolist(),
        "COM_to_palm_origin_distance_m": float(np.linalg.norm(final["com_compiled"])),
        "COM_to_palm_surface_distance_m": final["palm_surface_distance_m"],
        "COM_inside_existing_palm_region": _inside_existing_palm_box(final["com_reference"], resources),
        "palm_A_contact": bool(final["palm_count"] > 0),
        "palm_A_contact_fraction": float(np.mean([row["palm_count"] > 0 for row in rows])),
        "palm_A_normal_force_N": float(np.mean([row["palm_force_N"] for row in rows])),
        "palm_A_contact_count": int(final["palm_count"]),
        "mean_palm_A_contact_count": float(np.mean([row["palm_count"] for row in rows])),
        "per_finger_A_contact_flags": (finger_counts[-1] > 0).tolist(),
        "per_finger_A_contact_fraction": np.mean(finger_counts > 0, axis=0).tolist(),
        "per_finger_A_normal_force_N": mean_forces.tolist(),
        "mean_per_finger_normal_force_N": mean_forces.tolist(),
        "per_finger_A_final_normal_force_N": finger_forces[-1].tolist(),
        "per_finger_distal_contact_fraction": np.mean(distal_counts > 0, axis=0).tolist(),
        "occupied_finger_count": int(np.sum(occupied)),
        "occupied_finger_mask": occupied.tolist(),
        "free_finger_count": int(len(FINGER_ORDER) - np.sum(occupied)),
        "free_finger_mask": (~occupied).tolist(),
        "ferrari_canny_epsilon": epsilon,
        "A_translation_drift_m": float(np.max(np.linalg.norm(positions - positions[0], axis=1))),
        "A_rotation_drift_rad": _rotation_drift(quaternions),
        "A_vertical_drift_m": float(np.max(np.abs(positions[:, 2] - positions[0, 2]))),
        "maximum_penetration_m": float(max(row["maximum_penetration_m"] for row in rows)),
        "minimum_joint_margin_rad": float(min(np.min(row["joint_margins"]) for row in rows)),
        "joint_margins_rad": final["joint_margins"].tolist(),
        "maximum_actuator_utilization": float(max(np.max(row["actuator_utilization"]) for row in rows)),
        "actuator_utilization": final["actuator_utilization"].tolist(),
        "total_A_normal_force_N": float(np.mean([row["palm_force_N"] for row in rows]) + np.sum(mean_forces)),
        "table_recontact": bool(any(row["table_contact"] for row in rows)),
        "minimum_table_clearance_m": float(min(row["table_clearance_m"] for row in rows)),
        "complete_hand_contact_loss": bool(any(row["hand_count"] == 0 for row in rows)),
        "numerically_stable": bool(all(row["numerical"] for row in rows)),
        "fixture_removed_before_validation": True,
        "fixture_active_during_validation": False,
        "equality_constraint_count": int(model.neq),
        "object_joint_type": "free",
        "stable_hold_steps": hold_steps,
        "final_joint_configuration_rad": final["joint_positions"].tolist(),
        "final_object_position_m": final["position"].tolist(),
        "final_object_quaternion": final["quaternion"].tolist(),
    }


def classify_grasp_state(record: dict, state_type: GraspStateType, state_cfg, numerical_tolerance: float) -> dict:
    common = {
        "fixture_removed": bool(record.get("fixture_removed_before_validation")) and not bool(record.get("fixture_active_during_validation")),
        "no_equality_constraint": int(record.get("equality_constraint_count", -1)) == 0,
        "free_object_joint": record.get("object_joint_type") == "free",
        "no_table_support": not bool(record.get("table_recontact")),
        "penetration": float(record.get("maximum_penetration_m", math.inf)) <= state_cfg.maximum_penetration_m,
        "translation": float(record.get("A_translation_drift_m", math.inf)) <= state_cfg.maximum_translation_drift_m,
        "orientation": float(record.get("A_rotation_drift_rad", math.inf)) <= state_cfg.maximum_orientation_drift_rad,
        "no_complete_hand_contact_loss": not bool(record.get("complete_hand_contact_loss", True)),
        "force_closure": float(record.get("ferrari_canny_epsilon", 0.0)) > numerical_tolerance,
        "numerical": bool(record.get("numerically_stable")),
    }
    if state_type is GraspStateType.FINGERTIP:
        distal = np.asarray(record.get("per_finger_distal_contact_fraction", [0.0] * 4), dtype=float)
        specific = {
            "no_persistent_palm_contact": float(record.get("palm_A_contact_fraction", 1.0)) == 0.0,
            "minimum_distal_finger_contacts": int(np.sum(distal > 0.0)) >= state_cfg.minimum_fingertip_contact_fingers,
        }
    else:
        occupied = int(record.get("occupied_finger_count", 0))
        specific = {
            "real_palm_contact": bool(record.get("palm_A_contact")),
            "persistent_palm_contact": float(record.get("palm_A_contact_fraction", 0.0)) >= state_cfg.palm_contact_fraction_minimum,
            "minimum_load_bearing_fingers": occupied >= state_cfg.minimum_palmar_load_bearing_fingers,
            "maximum_load_bearing_fingers": occupied <= state_cfg.maximum_palmar_load_bearing_fingers,
            "COM_inside_existing_palm_region": bool(record.get("COM_inside_existing_palm_region")),
        }
    checks = {**common, **specific}
    rejection = next((name for name, passed in checks.items() if not passed), None)
    return {
        **record,
        "grasp_state_type": state_type.value,
        "accepted": rejection is None,
        "rejection_reason": rejection,
        "checks": checks,
        "second_grasp_digit_eligible": second_grasp_digit_eligible(record),
    }


def split_calibration_states(states: Iterable[dict], count_per_group: int) -> tuple[list[dict], list[dict]]:
    ordered = sorted(states, key=lambda row: str(row["grasp_state_id"]))
    calibration, formal = [], []
    for state_type in GraspStateType:
        group = [row for row in ordered if row["grasp_state_type"] == state_type.value]
        calibration.extend(group[:count_per_group])
        formal.extend(group[count_per_group:])
    return calibration, formal


@dataclass(frozen=True)
class MatchedPair:
    matched_pair_id: str
    fingertip: dict
    palmar: dict
    standardized_distance: float


def nearest_neighbor_match(
    fingertip: list[dict], palmar: list[dict], covariates: list[str], target_pairs: int,
) -> list[MatchedPair]:
    if not fingertip or not palmar:
        return []
    left = sorted(fingertip, key=lambda row: str(row["grasp_state_id"]))
    right = sorted(palmar, key=lambda row: str(row["grasp_state_id"]))
    pooled = np.asarray([[float(row[key]) for key in covariates] for row in left + right], dtype=float)
    means = np.mean(pooled, axis=0)
    std = np.std(pooled, axis=0, ddof=1)
    std[std == 0.0] = 1.0
    left_z = (pooled[:len(left)] - means) / std
    right_z = (pooled[len(left):] - means) / std
    edges = []
    for i in range(len(left)):
        for j in range(len(right)):
            edges.append((float(np.linalg.norm(left_z[i] - right_z[j])), str(left[i]["grasp_state_id"]), str(right[j]["grasp_state_id"]), i, j))
    edges.sort()
    used_left, used_right, pairs = set(), set(), []
    for distance, _, _, i, j in edges:
        if i in used_left or j in used_right:
            continue
        pair_id = f"phase2R_pair_{len(pairs):03d}"
        pairs.append(MatchedPair(pair_id, left[i], right[j], distance))
        used_left.add(i)
        used_right.add(j)
        if len(pairs) >= target_pairs:
            break
    return pairs


def paired_formal_trial_id(matched_pair_id: str, state_type: str, b_seed_index: int, formal_seed: int) -> str:
    return stable_trial_id(PHASE2R_EXPERIMENT_ID, {
        "matched_pair_id": matched_pair_id,
        "grasp_state_type": state_type,
        "B_seed_index": int(b_seed_index),
        "formal_seed_namespace": int(formal_seed),
    })


def assert_formal_pairing(records: list[dict], seeds_per_state: int) -> None:
    by_pair: dict[tuple[str, int], set[str]] = {}
    for row in records:
        if row.get("pilot_only") or row.get("calibration_only"):
            raise ValueError("pilot/calibration records cannot enter the formal paired dataset")
        key = (str(row["matched_pair_id"]), int(row["B_seed_index"]))
        by_pair.setdefault(key, set()).add(str(row["grasp_state_type"]))
    expected = {item.value for item in GraspStateType}
    if any(types != expected for types in by_pair.values()):
        raise ValueError("every matched-pair/B-seed cell must contain both state types")
    per_pair: dict[str, set[int]] = {}
    for pair_id, seed in by_pair:
        per_pair.setdefault(pair_id, set()).add(seed)
    if any(seeds != set(range(seeds_per_state)) for seeds in per_pair.values()):
        raise ValueError("formal pairs must receive the exact same complete B seed set")
