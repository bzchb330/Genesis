from __future__ import annotations

import mujoco
import numpy as np


def physical_dimensions(shape: str, size) -> tuple[float, float, float]:
    """Return full physical xyz dimensions under MuJoCo geom-size semantics."""

    values = np.asarray(size, dtype=float)
    if shape == "cube":
        if values.shape != (3,):
            raise ValueError("box size must contain three half-extents")
        return tuple(float(value) for value in 2.0 * values)
    if shape == "cylinder":
        if values.shape != (2,):
            raise ValueError("cylinder size must contain radius and half-height")
        return float(2.0 * values[0]), float(2.0 * values[0]), float(2.0 * values[1])
    raise ValueError(f"unsupported Phase 2S object shape {shape}")


def vertical_half_extent(shape: str, size) -> float:
    values = np.asarray(size, dtype=float)
    return float(values[2] if shape == "cube" else values[1])


def compiled_object_geometry(model: mujoco.MjModel, name: str) -> dict:
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_geom")
    if geom < 0:
        raise ValueError(f"missing object geom {name}")
    geom_type = int(model.geom_type[geom])
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        shape, size = "cube", model.geom_size[geom, :3].copy()
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        shape, size = "cylinder", model.geom_size[geom, :2].copy()
    else:
        raise ValueError(f"unsupported compiled object geom type {geom_type}")
    body = int(model.geom_bodyid[geom])
    return {
        "shape": shape,
        "size": [float(value) for value in size],
        "physical_dimensions_m": list(physical_dimensions(shape, size)),
        "mass_kg": float(model.body_mass[body]),
    }
