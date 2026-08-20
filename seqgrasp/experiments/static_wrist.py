from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import itertools
import json
import math
from typing import Iterable

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from ..config import ConfigBundle
from ..control import resolve_hand_indices
from ..scene_builder import build_scene
from ..phase2tr_config import assert_index_thumb_free_topology
from .phase2r import GraspStateType, classify_grasp_state, measure_stable_hold
from .resource_components import FINGER_ORDER, _collision_geoms_for_prefix, _finger_prefixes, _too_close, reconstruct_grasp


WORLD_GRAVITY_M_PER_S2 = np.asarray([0.0, 0.0, -9.81])
PALM_NORMAL_COMPILED = np.asarray([1.0, 0.0, 0.0])


@dataclass(frozen=True)
class StaticWristPose:
    pose_id: str
    relative_rpy_deg: tuple[float, float, float]
    relative_quaternion_wxyz: tuple[float, float, float, float]
    source: str
    parent_pose_id: str | None = None


@dataclass(frozen=True)
class FrozenWristBRegion:
    wrist_pose_id: str
    wrist_relative_rpy_deg: tuple[float, float, float]
    wrist_quaternion_wxyz: tuple[float, float, float, float]
    B_xyz_bounds_m: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    B_yaw_bounds_rad: tuple[float, float]
    integrity_hash: str


def normalize_quaternion_wxyz(quaternion: Iterable[float]) -> np.ndarray:
    value = np.asarray(tuple(quaternion), dtype=float)
    if value.shape != (4,) or not np.all(np.isfinite(value)):
        raise ValueError("root quaternion must contain four finite values")
    norm = float(np.linalg.norm(value))
    if norm <= np.finfo(float).eps:
        raise ValueError("root quaternion norm must be positive")
    value = value / norm
    if value[0] < 0.0:
        value = -value
    return value


def _rotation_wxyz(quaternion: Iterable[float]) -> Rotation:
    return Rotation.from_quat(normalize_quaternion_wxyz(quaternion), scalar_first=True)


def relative_rpy_quaternion_wxyz(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    # Intrinsic xyz offsets are post-multiplied onto each accepted endpoint's
    # mount orientation. This convention is deterministic and changes no hand
    # internal transform.
    rotation = Rotation.from_euler("xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True)
    return normalize_quaternion_wxyz(rotation.as_quat(scalar_first=True))


def compose_mount_quaternion_wxyz(base_wxyz: Iterable[float], relative_wxyz: Iterable[float]) -> np.ndarray:
    composed = _rotation_wxyz(base_wxyz) * _rotation_wxyz(relative_wxyz)
    return normalize_quaternion_wxyz(composed.as_quat(scalar_first=True))


def transform_pose_preserving_palm_relative(
    old_palm_position_m: Iterable[float],
    old_palm_quaternion_wxyz: Iterable[float],
    new_palm_position_m: Iterable[float],
    new_palm_quaternion_wxyz: Iterable[float],
    object_position_m: Iterable[float],
    object_quaternion_wxyz: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    old_position = np.asarray(tuple(old_palm_position_m), dtype=float)
    new_position = np.asarray(tuple(new_palm_position_m), dtype=float)
    object_position = np.asarray(tuple(object_position_m), dtype=float)
    if any(value.shape != (3,) for value in (old_position, new_position, object_position)):
        raise ValueError("palm/object positions must be three-vectors")
    old_rotation = _rotation_wxyz(old_palm_quaternion_wxyz)
    new_rotation = _rotation_wxyz(new_palm_quaternion_wxyz)
    relative_position = old_rotation.inv().apply(object_position - old_position)
    relative_rotation = old_rotation.inv() * _rotation_wxyz(object_quaternion_wxyz)
    transformed_position = new_position + new_rotation.apply(relative_position)
    transformed_quaternion = normalize_quaternion_wxyz(
        (new_rotation * relative_rotation).as_quat(scalar_first=True),
    )
    return transformed_position, transformed_quaternion


def palm_normal_world(quaternion_wxyz: Iterable[float]) -> np.ndarray:
    return _rotation_wxyz(quaternion_wxyz).apply(PALM_NORMAL_COMPILED)


def gravity_in_palm_frame(quaternion_wxyz: Iterable[float]) -> np.ndarray:
    return _rotation_wxyz(quaternion_wxyz).inv().apply(WORLD_GRAVITY_M_PER_S2)


def palm_normal_gravity_angle_deg(quaternion_wxyz: Iterable[float]) -> float:
    normal = palm_normal_world(quaternion_wxyz)
    cosine = float(np.dot(normal, WORLD_GRAVITY_M_PER_S2) / (np.linalg.norm(normal) * np.linalg.norm(WORLD_GRAVITY_M_PER_S2)))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _pose_key(quaternion: Iterable[float]) -> tuple[float, ...]:
    return tuple(np.round(normalize_quaternion_wxyz(quaternion), 10))


def coarse_wrist_poses(rolls: list[float], pitches: list[float], yaws: list[float]) -> list[StaticWristPose]:
    unique: dict[tuple[float, ...], StaticWristPose] = {}
    for roll, pitch, yaw in itertools.product(rolls, pitches, yaws):
        quaternion = relative_rpy_quaternion_wxyz(roll, pitch, yaw)
        key = _pose_key(quaternion)
        if key not in unique:
            unique[key] = StaticWristPose(
                pose_id=f"coarse_r{roll:+g}_p{pitch:+g}_y{yaw:+g}",
                relative_rpy_deg=(float(roll), float(pitch), float(yaw)),
                relative_quaternion_wxyz=tuple(float(v) for v in quaternion),
                source="coarse",
            )
    poses = sorted(unique.values(), key=lambda pose: pose.pose_id)
    if not any(np.allclose(pose.relative_quaternion_wxyz, [1.0, 0.0, 0.0, 0.0]) for pose in poses):
        raise AssertionError("coarse grid omitted the original wrist orientation")
    return poses


def refined_wrist_poses(parents: Iterable[StaticWristPose], offsets: list[float]) -> list[StaticWristPose]:
    unique: dict[tuple[float, ...], StaticWristPose] = {}
    for parent in sorted(parents, key=lambda item: item.pose_id):
        for dr, dp, dy in itertools.product(offsets, offsets, offsets):
            rpy = tuple(float(a + b) for a, b in zip(parent.relative_rpy_deg, (dr, dp, dy)))
            quaternion = relative_rpy_quaternion_wxyz(*rpy)
            key = _pose_key(quaternion)
            candidate = StaticWristPose(
                pose_id=f"refined_{parent.pose_id}_dr{dr:+g}_dp{dp:+g}_dy{dy:+g}",
                relative_rpy_deg=rpy,
                relative_quaternion_wxyz=tuple(float(v) for v in quaternion),
                source="refined",
                parent_pose_id=parent.pose_id,
            )
            if key not in unique or candidate.pose_id < unique[key].pose_id:
                unique[key] = candidate
    return sorted(unique.values(), key=lambda pose: pose.pose_id)


def deterministic_screening_subset(records: list[dict], count: int) -> list[dict]:
    """Farthest-point coverage of the five authorized baseline covariates."""

    if len(records) < count:
        raise ValueError(f"need {count} endpoint records, found {len(records)}")
    keys = (
        "ferrari_canny_epsilon", "total_A_normal_force_N", "A_translation_drift_m",
        "A_rotation_drift_rad", "minimum_joint_margin_rad",
    )
    ordered = sorted(records, key=lambda row: str(row["grasp_state_id"]))
    values = np.asarray([[float(row[key]) for key in keys] for row in ordered])
    scale = np.std(values, axis=0, ddof=1)
    scale[scale == 0.0] = 1.0
    standardized = (values - np.mean(values, axis=0)) / scale
    centroid_distance = np.linalg.norm(standardized, axis=1)
    selected = [int(np.argmax(centroid_distance))]
    while len(selected) < count:
        distances = np.min(
            np.linalg.norm(standardized[:, None, :] - standardized[selected][None, :, :], axis=2),
            axis=1,
        )
        distances[selected] = -1.0
        best_distance = float(np.max(distances))
        candidates = np.flatnonzero(np.isclose(distances, best_distance))
        selected.append(min(candidates, key=lambda index: str(ordered[int(index)]["grasp_state_id"])))
    return [ordered[index] for index in selected]


def transformed_endpoint_record(record: dict, pose: StaticWristPose) -> dict:
    assert_index_thumb_free_topology(record)
    old_palm_position = np.asarray(record["initial_palm_position_m"], dtype=float)
    old_palm_quaternion = normalize_quaternion_wxyz(record["initial_palm_quaternion"])
    new_palm_quaternion = compose_mount_quaternion_wxyz(old_palm_quaternion, pose.relative_quaternion_wxyz)
    object_position, object_quaternion = transform_pose_preserving_palm_relative(
        old_palm_position, old_palm_quaternion,
        old_palm_position, new_palm_quaternion,
        record["final_object_position_m"], record["final_object_quaternion"],
    )
    transformed = dict(record)
    transformed.update({
        "source_phase2TR_grasp_state_id": record["grasp_state_id"],
        "grasp_state_id": f"phase2W_{pose.pose_id}_{record['grasp_state_id']}",
        "phase2W_wrist_pose_id": pose.pose_id,
        "phase2W_relative_rpy_deg": list(pose.relative_rpy_deg),
        "phase2W_relative_quaternion_wxyz": list(pose.relative_quaternion_wxyz),
        "initial_palm_position_m": old_palm_position.tolist(),
        "initial_palm_quaternion": new_palm_quaternion.tolist(),
        "initial_object_position_m": object_position.tolist(),
        "initial_object_quaternion": object_quaternion.tolist(),
        "final_object_position_m": object_position.tolist(),
        "final_object_quaternion": object_quaternion.tolist(),
        "gravity_world_m_per_s2": WORLD_GRAVITY_M_PER_S2.tolist(),
        "gravity_palm_m_per_s2": gravity_in_palm_frame(new_palm_quaternion).tolist(),
        "palm_normal_world": palm_normal_world(new_palm_quaternion).tolist(),
        "palm_normal_gravity_angle_deg": palm_normal_gravity_angle_deg(new_palm_quaternion),
    })
    return transformed


def _initial_invalid_reason(model: mujoco.MjModel, data: mujoco.MjData, tolerance_m: float) -> str | None:
    table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    object_a_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_a_geom")
    object_b_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_b_geom")
    object_body_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object_a"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object_b"),
    }
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if table_id in pair:
            other = next(iter(pair - {table_id}))
            if other == object_a_geom:
                return "initial_A_table_contact"
            if other != object_b_geom:
                return "initial_hand_table_contact"
        body1, body2 = int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])
        both_hand = body1 not in object_body_ids and body2 not in object_body_ids and body1 != 0 and body2 != 0
        if both_hand and float(contact.dist) < -tolerance_m:
            return "initial_invalid_hand_self_collision"
    if not all(np.all(np.isfinite(value)) for value in (data.qpos, data.qvel, data.xpos, data.xquat)):
        return "initial_nonfinite"
    return None


def revalidate_transformed_endpoint(
    record: dict,
    pose: StaticWristPose,
    base_cfg: ConfigBundle,
    phase2s,
    phase2,
) -> dict:
    transformed = transformed_endpoint_record(record, pose)
    cfg = replace(base_cfg, hand=replace(
        base_cfg.hand,
        mount_pos=list(transformed["initial_palm_position_m"]),
        mount_quat=list(transformed["initial_palm_quaternion"]),
    ))
    model, data = build_scene(cfg)
    # Guard the physics contract explicitly: gravity remains fixed in world.
    if not np.allclose(model.opt.gravity, WORLD_GRAVITY_M_PER_S2):
        raise ValueError("world gravity changed during static wrist transform")
    indices = resolve_hand_indices(model, cfg.hand)
    object_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_a_free")
    qadr, vadr = model.jnt_qposadr[object_joint], model.jnt_dofadr[object_joint]
    data.qpos[indices.qpos_addresses] = np.asarray(record["final_joint_configuration_rad"], dtype=float)
    data.qvel[indices.qvel_addresses] = 0.0
    data.qpos[qadr:qadr + 3] = transformed["initial_object_position_m"]
    data.qpos[qadr + 3:qadr + 7] = transformed["initial_object_quaternion"]
    data.qvel[vadr:vadr + 6] = 0.0
    # Park B away from the hand; Phase 2W endpoint screening contains no B outcome.
    b_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_b_free")
    b_qadr, b_vadr = model.jnt_qposadr[b_joint], model.jnt_dofadr[b_joint]
    data.qpos[b_qadr:b_qadr + 3] = [-0.35, 0.35, 0.10]
    data.qpos[b_qadr + 3:b_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[b_vadr:b_vadr + 6] = 0.0
    mujoco.mj_forward(model, data)
    invalid = _initial_invalid_reason(model, data, phase2s.state.maximum_penetration_m)
    if invalid is not None:
        return {
            **transformed,
            "accepted": False,
            "rejection_reason": invalid,
            "checks": {invalid: False},
            "initial_invalid_reason": invalid,
        }
    hold_target = np.asarray(record.get("retaining_joint_target_rad", record["final_joint_configuration_rad"]), dtype=float)
    measured = measure_stable_hold(
        cfg, model, data, indices, hold_target, phase2s.state.stable_hold_steps,
        phase2.resources, phase2.dataset.friction_cone_edges,
        phase2.dataset.convex_hull_tolerance,
    )
    combined = {**transformed, **measured, "initial_invalid_reason": None}
    state_type = GraspStateType(record["grasp_state_type"])
    classified = classify_grasp_state(
        combined, state_type, phase2s.state, phase2.dataset.convex_hull_tolerance,
    )
    exact = classified["occupied_finger_mask"] == [False, True, True, False]
    checks = {
        **classified["checks"],
        "exact_middle_ring_support": exact,
        "exact_index_thumb_free": exact,
        "world_fixed_gravity": np.allclose(model.opt.gravity, WORLD_GRAVITY_M_PER_S2),
    }
    rejection = next((name for name, passed in checks.items() if not passed), None)
    return {**classified, "checks": checks, "accepted": rejection is None, "rejection_reason": rejection}


def recompute_index_thumb_workspace(
    record: dict,
    resources,
    sample_count: int,
    seed: int,
    base_cfg: ConfigBundle,
) -> dict:
    """Recompute collision-filtered index/thumb tip clouds at a static wrist pose.

    Samples are evaluated in the compiled transformed scene. They are not a
    rigidly rotated copy of an earlier cloud; table, palm, retained A, occupied
    fingers, and free-finger self collisions are checked after every sample.
    """

    cfg, model, data, indices = reconstruct_grasp(record, base_cfg)
    if not np.allclose(model.opt.gravity, WORLD_GRAVITY_M_PER_S2):
        raise ValueError("workspace scene changed world gravity")
    ranges = model.jnt_range[indices.joint_ids]
    free_joint_indices = np.r_[0:4, 12:16]
    retained_q = data.qpos[indices.qpos_addresses].copy()
    prefixes = _finger_prefixes(cfg)
    index_geoms = _collision_geoms_for_prefix(model, prefixes["index"])
    thumb_geoms = _collision_geoms_for_prefix(model, prefixes["thumb"])
    occupied_geoms = [
        geom
        for finger in ("middle", "ring")
        for geom in _collision_geoms_for_prefix(model, prefixes[finger])
    ]
    palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.hand.palm_body)
    palm_geoms = [geom for geom in range(model.ngeom) if int(model.geom_bodyid[geom]) == palm_id]
    object_a_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_a_geom")
    table_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    tip_bodies = {
        "index": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.hand.fingertip_bodies[0]),
        "thumb": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.hand.fingertip_bodies[3]),
    }
    tip_geoms = {
        "index": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, cfg.hand.finger_geom_mapping["index"][0]),
        "thumb": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, cfg.hand.finger_geom_mapping["thumb"][0]),
    }
    radii = {finger: float(model.geom_size[geom, 0]) for finger, geom in tip_geoms.items()}
    rng = np.random.default_rng(seed)
    clouds = {"index": [], "thumb": []}
    margins = []
    free_joint_samples = []
    rejections = {
        "object_A": 0, "occupied_fingers": 0, "palm": 0,
        "table": 0, "free_finger_self_collision": 0,
    }
    tolerance = float(resources.workspace_collision_tolerance_m)
    for _ in range(sample_count):
        sampled = rng.uniform(ranges[free_joint_indices, 0], ranges[free_joint_indices, 1])
        data.qpos[indices.qpos_addresses] = retained_q
        data.qpos[indices.qpos_addresses[free_joint_indices]] = sampled
        mujoco.mj_forward(model, data)
        free_geoms = index_geoms + thumb_geoms
        if _too_close(model, data, free_geoms, [object_a_geom], tolerance):
            rejections["object_A"] += 1
            continue
        if _too_close(model, data, free_geoms, occupied_geoms, tolerance):
            rejections["occupied_fingers"] += 1
            continue
        free_set, index_set, thumb_set, palm_set = set(free_geoms), set(index_geoms), set(thumb_geoms), set(palm_geoms)
        palm_collision = table_collision = free_self_collision = False
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            left, right = int(contact.geom1), int(contact.geom2)
            pair = {left, right}
            if not pair.intersection(free_set):
                continue
            # Adjacent articulated links can be geometrically close by design.
            # Reject only compiled contacts exceeding the existing tolerance.
            penetrating = float(contact.dist) < -tolerance
            palm_collision |= penetrating and bool(pair.intersection(palm_set))
            table_collision |= penetrating and table_geom in pair
            free_self_collision |= penetrating and bool(pair.intersection(index_set)) and bool(pair.intersection(thumb_set))
        if palm_collision:
            rejections["palm"] += 1
            continue
        if table_collision:
            rejections["table"] += 1
            continue
        if free_self_collision:
            rejections["free_finger_self_collision"] += 1
            continue
        for finger, body_id in tip_bodies.items():
            clouds[finger].append(data.xpos[body_id].copy())
        margins.append(float(np.min(np.minimum(sampled - ranges[free_joint_indices, 0], ranges[free_joint_indices, 1] - sampled))))
        free_joint_samples.append(sampled.copy())
    voxel = float(resources.workspace_voxel_size_m)
    result = {
        "proposed_samples": int(sample_count),
        "valid_samples": len(margins),
        "rejections": rejections,
        "tip_radii_m": radii,
        "minimum_joint_margin_rad": float(min(margins)) if margins else 0.0,
        "mean_joint_margin_rad": float(np.mean(margins)) if margins else 0.0,
        "free_joint_samples_rad": np.asarray(free_joint_samples, dtype=float).reshape(-1, 8),
    }
    for finger in ("index", "thumb"):
        points = np.asarray(clouds[finger], dtype=float).reshape(-1, 3)
        result[f"{finger}_points_world_m"] = points
        if len(points):
            voxels = np.unique(np.floor(points / voxel).astype(np.int64), axis=0)
            result[f"{finger}_reachable_volume_m3"] = float(len(voxels) * voxel ** 3)
            result[f"{finger}_world_bounds_m"] = [points.min(axis=0).tolist(), points.max(axis=0).tolist()]
        else:
            result[f"{finger}_reachable_volume_m3"] = 0.0
            result[f"{finger}_world_bounds_m"] = None
    data.qpos[indices.qpos_addresses] = retained_q
    mujoco.mj_forward(model, data)
    result["cfg"] = cfg
    result["model"] = model
    result["data"] = data
    return result


def wrist_b_integrity_hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def freeze_wrist_b_region(
    pose: StaticWristPose,
    xyz_bounds_m: Iterable[Iterable[float]],
    yaw_bounds_rad: Iterable[float],
) -> FrozenWristBRegion:
    xyz = tuple(tuple(float(value) for value in pair) for pair in xyz_bounds_m)
    yaw = tuple(float(value) for value in yaw_bounds_rad)
    if len(xyz) != 3 or any(len(pair) != 2 or pair[0] >= pair[1] for pair in xyz):
        raise ValueError("B xyz bounds must be three nonempty intervals")
    if len(yaw) != 2 or yaw[0] >= yaw[1]:
        raise ValueError("B yaw bounds must be a nonempty interval")
    payload = {
        "wrist_pose_id": pose.pose_id,
        "wrist_relative_rpy_deg": list(pose.relative_rpy_deg),
        "wrist_quaternion_wxyz": list(normalize_quaternion_wxyz(pose.relative_quaternion_wxyz)),
        "B_xyz_bounds_m": [list(pair) for pair in xyz],
        "B_yaw_bounds_rad": list(yaw),
    }
    return FrozenWristBRegion(
        wrist_pose_id=pose.pose_id,
        wrist_relative_rpy_deg=pose.relative_rpy_deg,
        wrist_quaternion_wxyz=tuple(payload["wrist_quaternion_wxyz"]),
        B_xyz_bounds_m=xyz,
        B_yaw_bounds_rad=yaw,
        integrity_hash=wrist_b_integrity_hash(payload),
    )


def verify_wrist_b_freeze(frozen: FrozenWristBRegion) -> None:
    payload = asdict(frozen)
    claimed = payload.pop("integrity_hash")
    if wrist_b_integrity_hash(payload) != claimed:
        raise ValueError("frozen wrist/B distribution integrity check failed")


def formal_seed_id(formal_seed: int, matched_pair_id: str, state_type: str, B_seed_index: int) -> str:
    payload = ["phase2W-formal", int(formal_seed), matched_pair_id, state_type, int(B_seed_index)]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()
