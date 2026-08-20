from __future__ import annotations

import mujoco
import numpy as np
from scipy import ndimage

from ..config import ConfigBundle
from .resource_components import (
    PALM_REFERENCE_TO_COMPILED, _finger_prefixes, _point_inside_geom,
    reconstruct_grasp,
)


def palm_space_diagnostics(record: dict, resources, base_cfg: ConfigBundle) -> dict:
    """Exploratory spatial diagnostics for the existing palm measurement box."""

    cfg, model, data, _ = reconstruct_grasp(record, base_cfg)
    low = np.asarray(resources.free_palm_box_lower_m, dtype=float)
    high = np.asarray(resources.free_palm_box_upper_m, dtype=float)
    step = float(resources.free_palm_voxel_size_m)
    axes = [np.arange(low[index] + step / 2.0, high[index], step) for index in range(3)]
    shape = tuple(len(axis) for axis in axes)
    points_reference = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    points_compiled = points_reference @ PALM_REFERENCE_TO_COMPILED.T
    palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.hand.palm_body)
    palm_rotation = data.xmat[palm_id].reshape(3, 3)
    points_world = data.xpos[palm_id] + points_compiled @ palm_rotation.T
    prefixes = set(_finger_prefixes(cfg).values())
    object_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_a_geom")
    relevant = [object_geom]
    for geom_id in range(model.ngeom):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])) or ""
        if any(body_name.startswith(prefix + "_") for prefix in prefixes):
            if model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]:
                relevant.append(geom_id)
    occupied = np.zeros(len(points_world), dtype=bool)
    for geom_id in relevant:
        occupied |= _point_inside_geom(model, data, geom_id, points_world)
    free_grid = (~occupied).reshape(shape)
    labels, component_count = ndimage.label(free_grid, structure=ndimage.generate_binary_structure(3, 1))
    component_sizes = np.bincount(labels.ravel())[1:] if component_count else np.asarray([], dtype=int)
    padded = np.pad(free_grid, 1, constant_values=False)
    inscribed = ndimage.distance_transform_edt(padded, sampling=step)[1:-1, 1:-1, 1:-1]
    object_rotation = data.geom_xmat[object_geom].reshape(3, 3)
    object_size = model.geom_size[object_geom, :3]
    corners_local = np.asarray([
        [x, y, z]
        for x in (-object_size[0], object_size[0])
        for y in (-object_size[1], object_size[1])
        for z in (-object_size[2], object_size[2])
    ])
    corners_world = data.geom_xpos[object_geom] + corners_local @ object_rotation.T
    corners_compiled = (corners_world - data.xpos[palm_id]) @ palm_rotation
    corners_reference = corners_compiled @ PALM_REFERENCE_TO_COMPILED
    boundary_margin = float(np.min(np.c_[corners_reference - low, high - corners_reference]))
    return {
        "free_palm_volume_m3": float(np.sum(free_grid) * step ** 3),
        "occupied_palm_voxel_fraction": float(np.mean(occupied)),
        "minimum_object_to_palm_boundary_margin_m": boundary_margin,
        "largest_connected_free_palm_component_m3": float((np.max(component_sizes) if len(component_sizes) else 0) * step ** 3),
        "largest_inscribed_free_space_radius_m": float(np.max(inscribed)) if inscribed.size else 0.0,
        "palm_voxel_shape": list(shape),
        "palm_voxel_size_m": step,
    }
