from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib

import mujoco
import numpy as np

from ..config import ConfigBundle, load_configs
from ..control import resolve_hand_indices
from ..diagnostics.grasp_search import load_search_config
from ..phase2_config import ResourceExperimentConfig
from ..scene_builder import build_scene


FINGER_ORDER = ("index", "middle", "ring", "thumb")
RESOURCE_METHOD_ID = "allegro_palm_axis_transform_v1"
RESOURCE_RECORDS_FILENAME = f"resource_components_{RESOURCE_METHOD_ID}.jsonl"
# PI reference -z points palm-to-fingers; compiled Allegro palm +x does.
# This proper rotation maps [x_ref,y_ref,z_ref] -> [-z_ref,y_ref,x_ref].
PALM_REFERENCE_TO_COMPILED = np.asarray([
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
])


@dataclass(frozen=True)
class ResourceComponents:
    """The three raw Phase 2 components; intentionally no scalar J."""

    occupied_finger_count: int
    occupied_finger_mask: tuple[bool, bool, bool, bool]
    free_finger_workspace_vol_m3: float
    free_palm_volume_m3: float


def occupied_fingers(normal_force_N, threshold_N: float) -> tuple[int, np.ndarray]:
    mask = np.asarray(normal_force_N, dtype=float) > threshold_N
    if mask.shape != (len(FINGER_ORDER),):
        raise ValueError("normal-force vector must follow the configured four-finger order")
    return int(mask.sum()), mask


def _stable_seed(seed: int, grasp_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{grasp_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def reconstruct_grasp(
    record: dict, base_cfg: ConfigBundle | None = None,
) -> tuple[ConfigBundle, mujoco.MjModel, mujoco.MjData, object]:
    cfg = base_cfg or load_configs()
    hand = replace(
        cfg.hand,
        mount_pos=list(record["initial_palm_position_m"]),
        mount_quat=list(record["initial_palm_quaternion"]),
    )
    cfg = replace(cfg, hand=hand)
    model, data = build_scene(cfg)
    indices = resolve_hand_indices(model, cfg.hand)
    data.qpos[indices.qpos_addresses] = np.asarray(record["final_joint_configuration_rad"])
    data.qvel[indices.qvel_addresses] = 0.0
    object_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_a_free")
    qadr = model.jnt_qposadr[object_joint]
    vadr = model.jnt_dofadr[object_joint]
    data.qpos[qadr:qadr + 3] = record["final_object_position_m"]
    data.qpos[qadr + 3:qadr + 7] = record["final_object_quaternion"]
    data.qvel[vadr:vadr + 6] = 0.0
    mujoco.mj_forward(model, data)
    return cfg, model, data, indices


def _finger_prefixes(cfg: ConfigBundle) -> dict[str, str]:
    return {
        finger: cfg.hand.fingertip_bodies[index].split("_", 1)[0]
        for index, finger in enumerate(cfg.hand.finger_geom_mapping)
    }


def _collision_geoms_for_prefix(model: mujoco.MjModel, prefix: str) -> list[int]:
    result = []
    for geom_id in range(model.ngeom):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])) or ""
        if body_name.startswith(prefix + "_") and (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]):
            result.append(geom_id)
    return result


def _too_close(model, data, left: list[int], right: list[int], tolerance_m: float) -> bool:
    for geom_a in left:
        for geom_b in right:
            if geom_a == geom_b:
                continue
            distance = mujoco.mj_geomDistance(model, data, geom_a, geom_b, 1.0, None)
            if distance < tolerance_m:
                return True
    return False


def free_finger_workspace_volume(
    record: dict,
    resources: ResourceExperimentConfig,
    sample_count: int,
    seed: int,
    base_cfg: ConfigBundle | None = None,
) -> float:
    """Monte Carlo joint sampling followed by 5-mm fingertip voxel occupancy."""

    cfg, model, data, indices = reconstruct_grasp(record, base_cfg)
    occupied = np.asarray(record["occupied_finger_mask"], dtype=bool)
    free = ~occupied
    if not np.any(free):
        return 0.0
    groups = load_search_config()["finger_groups"]
    actuator_index = {name: index for index, name in enumerate(cfg.hand.actuator_names)}
    free_joint_indices = np.asarray([
        actuator_index[name]
        for finger_index, finger in enumerate(FINGER_ORDER) if free[finger_index]
        for name in groups[finger]
    ], dtype=int)
    ranges = model.jnt_range[indices.joint_ids]
    prefixes = _finger_prefixes(cfg)
    free_geoms = [
        geom
        for finger_index, finger in enumerate(FINGER_ORDER) if free[finger_index]
        for geom in _collision_geoms_for_prefix(model, prefixes[finger])
    ]
    occupied_geoms = [
        geom
        for finger_index, finger in enumerate(FINGER_ORDER) if occupied[finger_index]
        for geom in _collision_geoms_for_prefix(model, prefixes[finger])
    ]
    object_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_a_geom")
    tip_body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.hand.fingertip_bodies[index])
        for index in np.flatnonzero(free)
    ]
    rng = np.random.default_rng(seed)
    voxels: set[tuple[int, int, int]] = set()
    for _ in range(sample_count):
        sampled = rng.uniform(ranges[free_joint_indices, 0], ranges[free_joint_indices, 1])
        data.qpos[indices.qpos_addresses[free_joint_indices]] = sampled
        mujoco.mj_forward(model, data)
        if _too_close(model, data, free_geoms, [object_geom], resources.workspace_collision_tolerance_m):
            continue
        if occupied_geoms and _too_close(model, data, free_geoms, occupied_geoms, resources.workspace_collision_tolerance_m):
            continue
        for body_id in tip_body_ids:
            voxel = tuple(np.floor(data.xpos[body_id] / resources.workspace_voxel_size_m).astype(np.int64))
            voxels.add(voxel)
    return len(voxels) * resources.workspace_voxel_size_m ** 3


def _point_inside_geom(model, data, geom_id: int, points_world: np.ndarray) -> np.ndarray:
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    local = (points_world - data.geom_xpos[geom_id]) @ rotation
    geom_type = int(model.geom_type[geom_id])
    size = model.geom_size[geom_id]
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        return np.all(np.abs(local) <= size[:3], axis=1)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        closest_z = np.clip(local[:, 2], -size[1], size[1])
        radial = local - np.c_[np.zeros(len(local)), np.zeros(len(local)), closest_z]
        return np.linalg.norm(radial, axis=1) <= size[0]
    raise ValueError(f"unsupported collision geom type {geom_type}; Phase 2 expects boxes/capsules")


def free_palm_volume(
    record: dict, resources: ResourceExperimentConfig, base_cfg: ConfigBundle | None = None,
) -> float:
    """Count palm-frame voxel centres not occupied by A or finger collision geoms."""

    cfg, model, data, _ = reconstruct_grasp(record, base_cfg)
    low = np.asarray(resources.free_palm_box_lower_m, dtype=float)
    high = np.asarray(resources.free_palm_box_upper_m, dtype=float)
    step = resources.free_palm_voxel_size_m
    axes = [np.arange(low[i] + step / 2, high[i], step) for i in range(3)]
    points_reference = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    points_palm = points_reference @ PALM_REFERENCE_TO_COMPILED.T
    palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.hand.palm_body)
    palm_rotation = data.xmat[palm_id].reshape(3, 3)
    points_world = data.xpos[palm_id] + points_palm @ palm_rotation.T
    finger_prefixes = set(_finger_prefixes(cfg).values())
    relevant = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_a_geom")]
    for geom_id in range(model.ngeom):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])) or ""
        if any(body_name.startswith(prefix + "_") for prefix in finger_prefixes):
            if model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]:
                relevant.append(geom_id)
    occupied = np.zeros(len(points_world), dtype=bool)
    for geom_id in relevant:
        occupied |= _point_inside_geom(model, data, geom_id, points_world)
    return float(np.sum(~occupied) * step ** 3)


def compute_resource_components(
    record: dict,
    resources: ResourceExperimentConfig,
    seed: int,
    base_cfg: ConfigBundle | None = None,
) -> ResourceComponents:
    count, mask = occupied_fingers(
        record["mean_per_finger_normal_force_N"],
        resources.occupied_finger_normal_force_threshold_N,
    )
    # Recompute and persist the PI threshold result to avoid trusting stale data.
    enriched = dict(record)
    enriched["occupied_finger_count"] = count
    enriched["occupied_finger_mask"] = mask.tolist()
    sample_seed = _stable_seed(seed, str(record["grasp_id"]))
    return ResourceComponents(
        occupied_finger_count=count,
        occupied_finger_mask=tuple(bool(value) for value in mask),
        free_finger_workspace_vol_m3=free_finger_workspace_volume(
            enriched, resources, resources.workspace_samples, sample_seed, base_cfg,
        ),
        free_palm_volume_m3=free_palm_volume(enriched, resources, base_cfg),
    )
