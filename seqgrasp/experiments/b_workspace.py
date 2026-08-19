from __future__ import annotations

import hashlib
import math

import mujoco
import numpy as np

from ..diagnostics.grasp_search import load_search_config
from .resource_components import (
    FINGER_ORDER,
    PALM_REFERENCE_TO_COMPILED,
    _collision_geoms_for_prefix,
    _finger_prefixes,
    _too_close,
    reconstruct_grasp,
)
from .second_grasp import BPlacement, _set_b_pose


def _seed(base_seed: int, grasp_id: str) -> int:
    digest = hashlib.sha256(f"B-workspace:{base_seed}:{grasp_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def free_fingertip_workspace_clouds(
    record: dict, resources, sample_count: int, seed: int, base_cfg=None,
):
    """Collision-filtered free-tip Monte Carlo clouds in world coordinates."""

    cfg, model, data, indices = reconstruct_grasp(record, base_cfg)
    retained_q = data.qpos[indices.qpos_addresses].copy()
    retained_qvel = data.qvel[indices.qvel_addresses].copy()
    occupied = np.asarray(record["occupied_finger_mask"], dtype=bool)
    free = ~occupied
    groups = load_search_config()["finger_groups"]
    by_name = {name: index for index, name in enumerate(cfg.hand.actuator_names)}
    free_joints = np.asarray([
        by_name[name]
        for finger_index, finger in enumerate(FINGER_ORDER) if free[finger_index]
        for name in groups[finger]
    ], dtype=int)
    if not len(free_joints):
        return cfg, model, data, {finger: np.empty((0, 3)) for finger in FINGER_ORDER}, {}
    prefixes = _finger_prefixes(cfg)
    free_geoms = [
        geom for finger_index, finger in enumerate(FINGER_ORDER) if free[finger_index]
        for geom in _collision_geoms_for_prefix(model, prefixes[finger])
    ]
    occupied_geoms = [
        geom for finger_index, finger in enumerate(FINGER_ORDER) if occupied[finger_index]
        for geom in _collision_geoms_for_prefix(model, prefixes[finger])
    ]
    object_a_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_a_geom")
    body_ids = {
        finger: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.hand.fingertip_bodies[index])
        for index, finger in enumerate(FINGER_ORDER) if free[index]
    }
    tip_geoms = {
        finger: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, cfg.hand.finger_geom_mapping[finger][0])
        for finger in body_ids
    }
    radii = {finger: float(model.geom_size[geom, 0]) for finger, geom in tip_geoms.items()}
    ranges = model.jnt_range[indices.joint_ids]
    rng = np.random.default_rng(_seed(seed, str(record["grasp_id"])))
    clouds = {finger: [] for finger in FINGER_ORDER}
    for _ in range(sample_count):
        data.qpos[indices.qpos_addresses[free_joints]] = rng.uniform(
            ranges[free_joints, 0], ranges[free_joints, 1],
        )
        mujoco.mj_forward(model, data)
        if _too_close(model, data, free_geoms, [object_a_geom], resources.workspace_collision_tolerance_m):
            continue
        if occupied_geoms and _too_close(
            model, data, free_geoms, occupied_geoms, resources.workspace_collision_tolerance_m,
        ):
            continue
        for finger, body_id in body_ids.items():
            clouds[finger].append(data.xpos[body_id].copy())
    data.qpos[indices.qpos_addresses] = retained_q
    data.qvel[indices.qvel_addresses] = retained_qvel
    mujoco.mj_forward(model, data)
    return cfg, model, data, {
        finger: np.asarray(points, dtype=float).reshape((-1, 3)) for finger, points in clouds.items()
    }, radii


def point_to_vertical_cylinder_surface_distance(points, center, radius: float, half_length: float):
    points = np.asarray(points, dtype=float)
    center = np.asarray(center, dtype=float)
    radial = np.linalg.norm(points[:, :2] - center[:2], axis=1)
    radial_excess = np.maximum(radial - radius, 0.0)
    axial_excess = np.maximum(np.abs(points[:, 2] - center[2]) - half_length, 0.0)
    outside = np.hypot(radial_excess, axial_excess)
    inside = (radial <= radius) & (np.abs(points[:, 2] - center[2]) <= half_length)
    if np.any(inside):
        inside_distance = np.minimum(radius - radial, half_length - np.abs(points[:, 2] - center[2]))
        outside[inside] = inside_distance[inside]
    return outside


def analyze_B_geometry_state(cfg, model, data, clouds, tip_radii, resources, placement: BPlacement) -> dict:
    """Geometry-only accessibility and overlap metrics for one retained A grasp."""

    _, b_geom, _ = _set_b_pose(model, data, placement)
    object_b = next(obj for obj in cfg.scene.objects if obj.name == "object_b")
    per_finger = {}
    for finger in FINGER_ORDER:
        points = clouds.get(finger, np.empty((0, 3)))
        if not len(points):
            per_finger[finger] = math.inf
            continue
        distance = point_to_vertical_cylinder_surface_distance(
            points, placement.position_m, object_b.size[0], object_b.size[1],
        )
        per_finger[finger] = float(np.min(distance))
    accessible = [
        finger for finger, distance in per_finger.items()
        if distance <= tip_radii.get(finger, 0.0) + resources.workspace_collision_tolerance_m
    ]
    collision_a = False
    collision_hand = False
    for geom_id in range(model.ngeom):
        if geom_id == b_geom or not (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name == "table":
            continue
        distance = mujoco.mj_geomDistance(model, data, b_geom, geom_id, 1.0, None)
        if distance < 0.0:
            collision_a |= name == "object_a_geom"
            collision_hand |= name != "object_a_geom"
    palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.hand.palm_body)
    palm_rotation = data.xmat[palm_id].reshape(3, 3)
    compiled = (np.asarray(placement.position_m) - data.xpos[palm_id]) @ palm_rotation
    reference = compiled @ PALM_REFERENCE_TO_COMPILED
    inside_palm = bool(np.all(reference >= resources.free_palm_box_lower_m) and np.all(reference <= resources.free_palm_box_upper_m))
    return {
        "reachable": bool(accessible) and not collision_a and not collision_hand,
        "minimum_free_fingertip_to_B_m": min(per_finger.values()),
        "reachable_free_finger_count": len(accessible),
        "reachable_free_fingers": accessible,
        "initial_collision_A": bool(collision_a),
        "initial_collision_hand": bool(collision_hand),
        "inside_measured_free_palm_region": inside_palm,
    }


def stratified_representative_ids(accepted: list[dict], resources: list[dict], count: int) -> list[str]:
    """Deterministic coverage of counts and marginal resource/epsilon quantiles."""

    by_resource = {row["grasp_id"]: row for row in resources}
    rows = [{**row, **by_resource[row["grasp_id"]]} for row in accepted if row["grasp_id"] in by_resource]
    selected: list[str] = []
    for occupied in sorted({int(row["occupied_finger_count"]) for row in rows}):
        group = sorted((row for row in rows if int(row["occupied_finger_count"]) == occupied), key=lambda row: row["grasp_id"])
        for index in np.linspace(0, len(group) - 1, min(3, len(group))).round().astype(int):
            if group[index]["grasp_id"] not in selected:
                selected.append(group[index]["grasp_id"])
    for key in ("free_finger_workspace_vol_m3", "free_palm_volume_m3", "ferrari_canny_epsilon"):
        ordered = sorted(rows, key=lambda row: (float(row[key]), row["grasp_id"]))
        for index in np.linspace(0, len(ordered) - 1, 9).round().astype(int):
            if ordered[index]["grasp_id"] not in selected:
                selected.append(ordered[index]["grasp_id"])
    if len(selected) < count:
        for row in sorted(rows, key=lambda row: row["grasp_id"]):
            if row["grasp_id"] not in selected:
                selected.append(row["grasp_id"])
            if len(selected) >= count:
                break
    return selected[:count]
