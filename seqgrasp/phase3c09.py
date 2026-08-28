"""Phase 3C-0.9 static/kinematic storage reachability diagnostics.

This module never calls ``mujoco.mj_step``. Existing Phase 3C-0.8 state
trajectories are reconstructed with qpos assignment and ``mj_forward``;
all new searches are geometry or smooth kinematic-model calculations.
"""
from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
import json
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np
from scipy import ndimage
import yaml

from .config import ROOT
from .phase3c0 import palm_transform, world_to_palm
from .phase3c07 import SPHERE_RADIUS_M, _geom_ids, _object_geom_id, pocket_volume_from_audit
from .phase3c08 import FOREARM_JOINT_NAME, build_forearm_scene


OUTPUT = ROOT / "outputs/phase3C09"
SURFACES = ("palm", "thumb", "index", "middle", "ring", "little")


def load_phase3c09_config(path: Path | None = None) -> dict[str, Any]:
    source = path or ROOT / "configs/phase3C09_storage_reachability.yaml"
    return yaml.safe_load(source.read_text(encoding="utf-8"))


def load_phase3c08_results() -> dict[str, Any]:
    return json.loads((ROOT / "outputs/phase3C08/targeted_dynamics_results.json").read_text(encoding="utf-8"))


def select_top_trajectories(result: dict[str, Any], count: int = 5) -> list[dict[str, Any]]:
    return sorted(result["rows"], key=lambda row: (row["closest_pocket_distance_m"], row["trial_id"]))[:count]


def finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.gradient(values, dt, axis=0, edge_order=1)


def pocket_distance(center: Iterable[float], feasible_voxels: np.ndarray) -> float:
    return float(np.min(np.linalg.norm(np.asarray(feasible_voxels) - np.asarray(tuple(center)), axis=1)))


def decompose_contact_force(normal_force_n: float, tangential_force_n: float | None, friction_coefficient: float) -> dict[str, float | None]:
    utilization = None if tangential_force_n is None or normal_force_n <= 0 or friction_coefficient <= 0 else float(tangential_force_n / (friction_coefficient * normal_force_n))
    return {"normal_force_n": float(normal_force_n), "tangential_force_n": None if tangential_force_n is None else float(tangential_force_n), "friction_utilization": utilization}


def classify_multiresolution(connectivity: Iterable[bool]) -> str:
    values = tuple(bool(value) for value in connectivity)
    return "CS-A" if all(values) else ("CS-B" if values and values[0] and not values[-1] else "CS-C")


def bottleneck_from_path(clearance: np.ndarray, path: list[tuple[int, int, int]]) -> tuple[tuple[int, int, int] | None, float | None]:
    if not path: return None, None
    node = min(path, key=lambda index: clearance[index]); return node, float(clearance[node])


def collision_free_storage_candidate(clearance_m: float, near_surface_count: int, tolerance_m: float = 1e-9) -> bool:
    return bool(clearance_m >= -tolerance_m and near_surface_count >= 2)


def cluster_storage_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    return ndimage.label(np.asarray(mask, dtype=bool), structure=np.ones((3, 3, 3), dtype=np.int8))


def _surface_records(contact: dict[str, Any], surface: str) -> list[dict[str, Any]]:
    return [record for record in contact["records"] if record["surface"] == surface]


def _normal_force(contact: dict[str, Any], surface: str) -> float:
    return float(sum(record["normal_force_n"] for record in _surface_records(contact, surface)))


def reconstruct_trajectory(row: dict[str, Any]) -> dict[str, Any]:
    series = dict(np.load(row["timeseries_path"], allow_pickle=False))
    wrapper = build_forearm_scene(with_actuator=True); scene = wrapper.scene
    dt = float(scene.model.opt.timestep)
    centers = np.asarray(series["sphere_center_palm_m"])
    contacts = [json.loads(str(value)) for value in series["contact_geometry_json"]]
    qpos = np.asarray(series["qpos"])
    velocities = finite_difference(centers, dt)
    speed = np.linalg.norm(velocities, axis=1)
    distance_rate = finite_difference(np.asarray(series["pocket_distance_m"]), dt)
    wrist = {}
    for name in ("rh_WRJ1", "rh_WRJ2"):
        joint = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        wrist[name] = qpos[:, scene.model.jnt_qposadr[joint]]
    object_address = scene.model.jnt_qposadr[scene.object_joint_id]
    quaternions = qpos[:, object_address + 3:object_address + 7]
    angular_speed = np.zeros(len(qpos))
    for index in range(1, len(qpos)):
        dot = abs(float(np.dot(quaternions[index - 1], quaternions[index])))
        angular_speed[index] = 2.0 * np.arccos(np.clip(dot, -1.0, 1.0)) / dt
    angular_speed[0] = angular_speed[1]
    forces = {surface: np.asarray([_normal_force(contact, surface) for contact in contacts]) for surface in SURFACES}
    contact_points = {surface: np.full((len(qpos), 3), np.nan) for surface in ("thumb", "index")}
    contact_normals = {surface: np.full((len(qpos), 3), np.nan) for surface in ("thumb", "index")}
    for index, contact in enumerate(contacts):
        scene.data.qpos[:] = qpos[index]; mujoco.mj_forward(scene.model, scene.data)
        origin, rotation = palm_transform(scene)
        for surface in ("thumb", "index"):
            records = _surface_records(contact, surface)
            if records:
                contact_points[surface][index] = rotation.T @ (np.mean([r["position_world_m"] for r in records], axis=0) - origin)
                contact_normals[surface][index] = rotation.T @ np.mean([r["normal_world"] for r in records], axis=0)
    minimum_index = int(np.argmin(series["pocket_distance_m"]))
    window = int(load_phase3c09_config()["trajectory_diagnosis"]["descriptive_near_minimum_window_samples"])
    lo, hi = max(0, minimum_index - window), min(len(qpos), minimum_index + window + 1)
    final = "DROP_OR_ESCAPE" if row["sphere_loss"] else (
        "CONTACT_LOSS" if not contacts[minimum_index]["contact_topology"] else "OTHER"
    )
    return {
        "trial_id": row["trial_id"], "state_id": row["state_id"], "mode": row["mode"],
        "configuration_label": row["configuration_label"], "minimum_index": minimum_index,
        "minimum_time_s": float(minimum_index * dt), "minimum_pocket_distance_m": float(series["pocket_distance_m"][minimum_index]),
        "final_outcome": final, "failure_interpretation": final,
        "jamming_consistent": False,
        "near_minimum": {
            "window_samples": [lo, hi - 1],
            "distance_rate_mps_median": float(np.median(distance_rate[lo:hi])),
            "speed_mps_median": float(np.median(speed[lo:hi])),
            "thumb_normal_force_n_median": float(np.median(forces["thumb"][lo:hi])),
            "index_normal_force_n_median": float(np.median(forces["index"][lo:hi])),
            "curve_description": "minimum occurs during measured drop/escape; it is not a stationary loaded plateau" if row["sphere_loss"] else "descriptive minimum without a publication threshold",
        },
        "availability": {
            "sphere_center_distance": True, "pocket_boundary_distance": True,
            "sphere_position": True, "sphere_linear_velocity_finite_difference": True,
            "sphere_angular_speed_finite_difference": True, "normal_contact_force": True,
            "tangential_contact_force": False, "friction_utilization": False,
            "contact_position": True, "contact_slip_velocity": False,
            "unavailable_reason": "Phase 3C-0.8 stored qpos and normal-force records but not qvel, tangential forces, or contact slip velocity; no dynamics rerun is authorized.",
        },
        "series": {
            "step": series["step"].tolist(), "time_s": (series["step"] * dt).tolist(),
            "pocket_distance_m": series["pocket_distance_m"].tolist(),
            "pocket_boundary_distance_m": series["pocket_distance_m"].tolist(),
            "sphere_center_palm_m": centers.tolist(), "sphere_linear_velocity_palm_mps": velocities.tolist(),
            "sphere_speed_mps": speed.tolist(), "sphere_angular_speed_radps": angular_speed.tolist(),
            "distance_rate_mps": distance_rate.tolist(),
            "normal_force_n": {surface: values.tolist() for surface, values in forces.items()},
            "thumb_contact_point_palm_m": contact_points["thumb"].tolist(),
            "index_contact_point_palm_m": contact_points["index"].tolist(),
            "thumb_contact_normal_palm": contact_normals["thumb"].tolist(),
            "index_contact_normal_palm": contact_normals["index"].tolist(),
            "forearm_angle_rad": series["actual_forearm_rad"].tolist(),
            "WRJ1_rad": wrist["rh_WRJ1"].tolist(), "WRJ2_rad": wrist["rh_WRJ2"].tolist(),
            "gravity_in_palm_mps2": series["gravity_in_palm_mps2"].tolist(),
            "qpos": qpos.tolist(),
        },
    }


def trajectory_failure_audit() -> dict[str, Any]:
    result = load_phase3c08_results()
    count = int(load_phase3c09_config()["trajectory_diagnosis"]["top_count"])
    selected = select_top_trajectories(result, count)
    rows = [reconstruct_trajectory(row) for row in selected]
    return {"selection": "smallest recorded minimum pocket distance, then trial_id", "N": len(rows), "rows": rows}


def _hand_geom_ids(scene) -> tuple[int, ...]:
    ids = []
    for surface in SURFACES:
        ids.extend(_geom_ids(scene, surface))
    return tuple(sorted(set(ids)))


def exact_clearance_grid(scene, qpos: np.ndarray, points_palm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact compiled sphere-to-hand geom distance at assigned qpos; no stepping."""
    scene.data.qpos[:] = qpos; mujoco.mj_forward(scene.model, scene.data)
    origin, rotation = palm_transform(scene)
    world = origin + np.asarray(points_palm) @ rotation.T
    object_geom = _object_geom_id(scene); hand_geoms = _hand_geom_ids(scene)
    original = scene.data.geom_xpos[object_geom].copy()
    clearance = np.full(len(world), np.inf); limiting = np.full(len(world), -1, dtype=np.int32)
    for index, center in enumerate(world):
        scene.data.geom_xpos[object_geom] = center
        for geom in hand_geoms:
            distance = float(mujoco.mj_geomDistance(scene.model, scene.data, object_geom, geom, 0.25, None))
            if distance < clearance[index]:
                clearance[index] = distance; limiting[index] = geom
    scene.data.geom_xpos[object_geom] = original
    return clearance, limiting


def cspace_axes(start: np.ndarray, pocket_points: np.ndarray, resolution: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    margin = float(load_phase3c09_config()["cspace"]["domain_margin_m"])
    lower = np.minimum(start, pocket_points.min(axis=0)) - margin
    upper = np.maximum(start, pocket_points.max(axis=0)) + margin
    return tuple(np.arange(lower[i], upper[i] + resolution * 0.5, resolution) for i in range(3))


def _nearest_grid_index(point: np.ndarray, axes: tuple[np.ndarray, ...]) -> tuple[int, int, int]:
    return tuple(int(np.argmin(np.abs(axis - point[index]))) for index, axis in enumerate(axes))


_NEIGHBORS = tuple(
    (dx, dy, dz, float(np.sqrt(dx * dx + dy * dy + dz * dz)))
    for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
)


def astar_path(free: np.ndarray, start: tuple[int, int, int], goals: set[tuple[int, int, int]], resolution: float) -> list[tuple[int, int, int]]:
    if not free[start] or not goals:
        return []
    shape = free.shape; goal_array = np.asarray(tuple(goals), dtype=float)
    def heuristic(node):
        return float(np.min(np.linalg.norm(goal_array - np.asarray(node), axis=1))) * resolution
    queue = [(heuristic(start), 0.0, start)]; parent = {}; cost = {start: 0.0}; closed = set()
    while queue:
        _, value, node = heappop(queue)
        if node in closed: continue
        closed.add(node)
        if node in goals:
            path = [node]
            while node in parent: node = parent[node]; path.append(node)
            return path[::-1]
        for dx, dy, dz, scale in _NEIGHBORS:
            nxt = (node[0] + dx, node[1] + dy, node[2] + dz)
            if not (0 <= nxt[0] < shape[0] and 0 <= nxt[1] < shape[1] and 0 <= nxt[2] < shape[2]) or not free[nxt]: continue
            candidate = value + resolution * scale
            if candidate < cost.get(nxt, np.inf):
                cost[nxt] = candidate; parent[nxt] = node; heappush(queue, (candidate + heuristic(nxt), candidate, nxt))
    return []


def cspace_connectivity(qpos: np.ndarray, start: np.ndarray, pocket_points: np.ndarray, resolution: float, label: str) -> dict[str, Any]:
    wrapper = build_forearm_scene(with_actuator=True); scene = wrapper.scene
    axes = cspace_axes(start, pocket_points, resolution)
    mesh = np.meshgrid(*axes, indexing="ij"); points = np.column_stack([value.ravel() for value in mesh])
    clearance, limiting = exact_clearance_grid(scene, qpos, points)
    tolerance = float(load_phase3c09_config()["cspace"]["occupancy_numerical_tolerance_m"])
    free = (clearance >= -tolerance).reshape(tuple(len(axis) for axis in axes))
    structure = np.ones((3, 3, 3), dtype=np.int8)
    components, _ = ndimage.label(free, structure=structure)
    start_index = _nearest_grid_index(start, axes); component_id = int(components[start_index])
    pocket_indices = {_nearest_grid_index(point, axes) for point in pocket_points}
    connected_goals = {index for index in pocket_indices if component_id > 0 and int(components[index]) == component_id}
    path = astar_path(free, start_index, connected_goals, resolution)
    flat_limiting = limiting.reshape(free.shape); clearance_grid = clearance.reshape(free.shape)
    path_points = [[float(axes[axis][node[axis]]) for axis in range(3)] for node in path]
    min_clearance = float(min((clearance_grid[node] for node in path), default=np.nan))
    bottleneck = min(path, key=lambda node: clearance_grid[node]) if path else None
    limiting_name = None if bottleneck is None else mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_GEOM, int(flat_limiting[bottleneck]))
    path_length = float(sum(np.linalg.norm(np.asarray(path_points[i]) - np.asarray(path_points[i - 1])) for i in range(1, len(path_points))))
    destination = OUTPUT / "cspace_grids"; destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / f"{label}_{resolution * 1000:g}mm.npz", axes_x=axes[0], axes_y=axes[1], axes_z=axes[2], free=free, clearance_m=clearance_grid, path=np.asarray(path_points))
    return {
        "resolution_m": resolution, "domain_bounds_palm_m": [list(map(float, (axis[0] for axis in axes))), list(map(float, (axis[-1] for axis in axes)))],
        "shape": list(free.shape), "free_voxel_count": int(np.count_nonzero(free)), "occupied_voxel_count": int(free.size - np.count_nonzero(free)),
        "start_index": list(start_index), "start_free": bool(free[start_index]),
        "start_component_size": int(np.count_nonzero(components == component_id)) if component_id else 0,
        "pocket_voxels_connected": len(connected_goals), "pocket_voxel_count": len(pocket_indices),
        "path_exists": bool(path), "shortest_path_length_m": path_length if path else None,
        "minimum_path_clearance_m": min_clearance if path else None,
        "bottleneck_palm_m": None if bottleneck is None else path_points[path.index(bottleneck)],
        "bottleneck_opening_width_m": None if bottleneck is None else float(2 * (SPHERE_RADIUS_M + min_clearance)),
        "limiting_geom": limiting_name, "path_palm_m": path_points,
    }


def cspace_audit(trajectory: dict[str, Any]) -> dict[str, Any]:
    pocket_audit = json.loads((ROOT / "outputs/phase3C07/static_reachability.json").read_text(encoding="utf-8"))
    pocket_points = np.asarray(pocket_audit["pocket_volume"]["feasible_centers_palm_m"])
    resolutions = [float(value) for value in load_phase3c09_config()["cspace"]["resolutions_m"]]
    states = []
    for row in trajectory["rows"]:
        index = row["minimum_index"]; start = np.asarray(row["series"]["sphere_center_palm_m"][index]); qpos = np.asarray(row["series"]["qpos"][index])
        grids = [cspace_connectivity(qpos, start, pocket_points, resolution, row["trial_id"]) for resolution in resolutions]
        connected = [grid["path_exists"] for grid in grids]
        classification = classify_multiresolution(connected)
        states.append({"trial_id": row["trial_id"], "frozen_step": index, "start_palm_m": start.tolist(), "grids": grids, "classification": classification})
    return {"sphere_radius_m": SPHERE_RADIUS_M, "sphere_diameter_m": 2 * SPHERE_RADIUS_M,
            "fixture_and_table_excluded": True, "states": states,
            "all_representative_states_CS_C": all(row["classification"] == "CS-C" for row in states)}


def numerical_lie_bracket(field_i, field_j, x: np.ndarray, step: float) -> np.ndarray:
    derivative_j_i = (field_j(x + step * field_i(x)) - field_j(x - step * field_i(x))) / (2 * step)
    derivative_i_j = (field_i(x + step * field_j(x)) - field_i(x - step * field_j(x))) / (2 * step)
    return derivative_j_i - derivative_i_j


def _tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = normal / np.linalg.norm(normal); reference = np.asarray([0.0, 0.0, 1.0])
    if abs(normal @ reference) > 0.9: reference = np.asarray([0.0, 1.0, 0.0])
    first = np.cross(normal, reference); first /= np.linalg.norm(first)
    return first, np.cross(normal, first)


def smooth_contact_fields(normal: np.ndarray, mode: str) -> list:
    """Standard smooth rolling-on-a-local-sphere chart; not Coulomb switching."""
    normal = np.asarray(normal, dtype=float); normal /= np.linalg.norm(normal)
    basis = _tangent_basis(normal)
    fields = []
    count = 1 if mode == "M0_DUAL_ROLLING" else 2
    for control in range(count):
        def field(x, control=control):
            chart = x[6:8]
            local_normal = normal + chart[0] * basis[0] + chart[1] * basis[1]; local_normal /= np.linalg.norm(local_normal)
            tangents = _tangent_basis(local_normal); tangent = tangents[control]
            value = np.zeros(8); value[:3] = tangent; value[3:6] = np.cross(local_normal, tangent) / SPHERE_RADIUS_M; value[6 + control] = 1.0
            return value
        fields.append(field)
    return fields


def _matrix_rank(matrix: np.ndarray, relative_tolerance: float) -> int:
    singular = np.linalg.svd(matrix, compute_uv=False)
    return int(np.sum(singular > singular[0] * relative_tolerance)) if len(singular) and singular[0] else 0


def cyclic_bracket_check(field_i, field_j, x: np.ndarray, epsilon: float) -> np.ndarray:
    state = x.copy()
    for field, sign in ((field_i, 1), (field_j, 1), (field_i, -1), (field_j, -1)):
        state = state + sign * epsilon * field(state)
    return state - x


def contact_accessibility_audit(trajectory: dict[str, Any], cspace: dict[str, Any]) -> dict[str, Any]:
    representative = trajectory["rows"][0]; contacts_t = np.asarray(representative["series"]["thumb_contact_normal_palm"]); contacts_i = np.asarray(representative["series"]["index_contact_normal_palm"])
    valid = np.flatnonzero(np.all(np.isfinite(contacts_t), axis=1) & np.all(np.isfinite(contacts_i), axis=1))
    if not len(valid):
        return {"classification": "CT-D", "reason": "no stored dual-contact sample", "modes": []}
    sample = int(valid[-1]); thumb, index = contacts_t[sample], contacts_i[sample]
    target = np.asarray(representative["series"]["sphere_center_palm_m"][representative["minimum_index"]]) - np.asarray(representative["series"]["sphere_center_palm_m"][sample]); target /= np.linalg.norm(target)
    cfg = load_phase3c09_config()["contact_accessibility"]; tolerance = float(cfg["numerical_rank_relative_tolerance"]); x = np.zeros(8)
    modes = []
    mode_normals = {"M0_DUAL_ROLLING": thumb + index, "M1_INDEX_GUIDE_THUMB_MIGRATION": index,
                    "M2_THUMB_GUIDE_INDEX_MIGRATION": thumb, "M3_UNLOADED_SINGLE_GUIDE_GRAVITY": index}
    gravity = np.asarray(representative["series"]["gravity_in_palm_mps2"][sample]); gravity /= np.linalg.norm(gravity)
    for mode, normal in mode_normals.items():
        if mode == "M0_DUAL_ROLLING":
            direction = np.cross(thumb, index); direction /= np.linalg.norm(direction)
            def dual_field(state, direction=direction):
                value = np.zeros(8); value[:3] = direction; value[3:6] = np.cross((thumb + index) / np.linalg.norm(thumb + index), direction) / SPHERE_RADIUS_M; return value
            fields = [dual_field]
        elif mode == "M3_UNLOADED_SINGLE_GUIDE_GRAVITY":
            fields = smooth_contact_fields(normal, mode)[:1]
            guide_normal = normal / np.linalg.norm(normal); gravity_tangent = gravity - (gravity @ guide_normal) * guide_normal
            gravity_tangent /= np.linalg.norm(gravity_tangent)
            def gravity_field(state, direction=gravity_tangent):
                value = np.zeros(8); value[:3] = direction; value[3:6] = np.cross(guide_normal, direction) / SPHERE_RADIUS_M; return value
            fields.append(gravity_field)
        else:
            fields = smooth_contact_fields(normal, mode)
        first = np.column_stack([field(x) for field in fields])
        brackets_by_step = []
        for step in cfg["finite_difference_steps"]:
            brackets_by_step.append([numerical_lie_bracket(fields[i], fields[j], x, float(step)) for i in range(len(fields)) for j in range(i + 1, len(fields))])
        stable_brackets = brackets_by_step[-1]
        second = np.column_stack([first] + ([np.column_stack(stable_brackets)] if stable_brackets else []))
        first_translation = first[:3]; second_translation = second[:3]
        projection_first = np.linalg.lstsq(first_translation, target, rcond=None)[0]; residual_first = target - first_translation @ projection_first
        projection_second = np.linalg.lstsq(second_translation, target, rcond=None)[0]; residual_second = target - second_translation @ projection_second
        convergence = None
        if stable_brackets:
            norms = [float(np.linalg.norm(values[0] - stable_brackets[0])) for values in brackets_by_step[:-1]]
            convergence = {"differences_to_finest": norms, "stable": bool(norms[-1] <= norms[0] + 1e-12)}
        cycles = []
        if len(fields) >= 2:
            for epsilon in cfg["cyclic_epsilons"]:
                displacement = cyclic_bracket_check(fields[0], fields[1], x, float(epsilon))
                cycles.append({"epsilon": epsilon, "translation": displacement[:3].tolist(), "translation_over_epsilon_squared": (displacement[:3] / float(epsilon) ** 2).tolist()})
        first_accessible = np.linalg.norm(residual_first) <= tolerance
        second_accessible = np.linalg.norm(residual_second) <= tolerance
        classification = "CT-A" if first_accessible else ("CT-B" if second_accessible and convergence and convergence["stable"] else "CT-C")
        modes.append({"mode": mode, "smooth_model": "local rolling sphere chart with fixed smooth contact mode; no Coulomb/contact switching claim",
                      "state_dimension": 8, "control_field_count": len(fields), "first_order_rank": _matrix_rank(first, tolerance),
                      "control_vector_fields_at_analysis_state": first.T.tolist(),
                      "first_order_translation_rank": _matrix_rank(first_translation, tolerance), "first_order_target_projection": float(np.linalg.norm(target - residual_first)),
                      "first_order_target_residual": float(np.linalg.norm(residual_first)), "second_order_rank": _matrix_rank(second, tolerance),
                      "second_order_translation_rank": _matrix_rank(second_translation, tolerance), "second_order_target_residual": float(np.linalg.norm(residual_second)),
                      "second_order_bracket_vectors_at_analysis_state": [value.tolist() for value in stable_brackets],
                      "bracket_convergence": convergence, "cyclic_validation": cycles, "classification": classification,
                      "gravity_is_external_drift": mode == "M3_UNLOADED_SINGLE_GUIDE_GRAVITY"})
    return {"representative_trial": representative["trial_id"], "representative_stored_step": sample, "analysis_state": "sphere position (3), sphere local rotation (3), smooth contact chart (2)",
            "state_dimension": 8, "target_direction_palm": target.tolist(), "modes": modes,
            "cspace_gate_downstream_noncausal": bool(cspace["all_representative_states_CS_C"]),
            "global_nonsmooth_LARC_claimed": False}


def _finger_interpolation(scene, base_qpos: np.ndarray, surface: str, fraction: float) -> None:
    for joint in scene.joint_ids[surface]:
        address = scene.model.jnt_qposadr[joint]; lower, upper = scene.model.jnt_range[joint]
        scene.data.qpos[address] = (1.0 - fraction) * lower + fraction * upper


def storage_manifold_audit(trajectory: dict[str, Any]) -> dict[str, Any]:
    cfg = load_phase3c09_config()["storage_manifold"]; wrapper = build_forearm_scene(with_actuator=True); scene = wrapper.scene
    representative = trajectory["rows"][0]; source_index = max(0, representative["minimum_index"] - 20); base_qpos = np.asarray(representative["series"]["qpos"][source_index])
    axes = tuple(np.arange(cfg["center_bounds_palm_m"][axis][0], cfg["center_bounds_palm_m"][axis][1] + cfg["center_spacing_m"] * .5, cfg["center_spacing_m"]) for axis in ("x", "y", "z"))
    mesh = np.meshgrid(*axes, indexing="ij"); points = np.column_stack([value.ravel() for value in mesh]); shape = tuple(len(axis) for axis in axes)
    configurations = []; configuration_qpos = []; location_count = np.zeros(len(points), dtype=np.int32); best_near = np.zeros(len(points), dtype=np.int8); support_masks = np.zeros(len(points), dtype=np.int32); first_config = np.full(len(points), -1, dtype=np.int32)
    config_index = 0
    for middle in cfg["middle_flexion_fractions"]:
      for ring in cfg["ring_flexion_fractions"]:
       for little in cfg["little_flexion_fractions"]:
        for wrist_offset in cfg["wrist2_offsets_deg"]:
         for forearm in cfg["forearm_PS_deg"]:
          scene.data.qpos[:] = base_qpos
          _finger_interpolation(scene, base_qpos, "middle", float(middle)); _finger_interpolation(scene, base_qpos, "ring", float(ring)); _finger_interpolation(scene, base_qpos, "little", float(little))
          for name, value in (("rh_WRJ2", scene.data.qpos[scene.model.jnt_qposadr[mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, "rh_WRJ2")]] + np.deg2rad(wrist_offset)), (FOREARM_JOINT_NAME, np.deg2rad(forearm))):
              joint = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, name); lo, hi = scene.model.jnt_range[joint]; scene.data.qpos[scene.model.jnt_qposadr[joint]] = np.clip(value, lo, hi)
          qpos = scene.data.qpos.copy(); clearance, _ = exact_clearance_grid(scene, qpos, points)
          surface_clearance = []
          for surface in SURFACES:
              original = scene.collision_geoms; selected = tuple(_geom_ids(scene, surface))
              scene.data.qpos[:] = qpos; mujoco.mj_forward(scene.model, scene.data); origin, rotation = palm_transform(scene); world = origin + points @ rotation.T; object_geom = _object_geom_id(scene); old = scene.data.geom_xpos[object_geom].copy(); values = np.full(len(points), np.inf)
              for pi, center in enumerate(world):
                  scene.data.geom_xpos[object_geom] = center
                  values[pi] = min(float(mujoco.mj_geomDistance(scene.model, scene.data, object_geom, geom, .25, None)) for geom in selected)
              scene.data.geom_xpos[object_geom] = old; surface_clearance.append(values)
          surface_clearance = np.asarray(surface_clearance)
          near = surface_clearance <= float(cfg["near_surface_numerical_tolerance_m"])
          valid = (clearance >= -float(cfg["occupancy_numerical_tolerance_m"])) & (np.sum(near, axis=0) >= 2)
          first_config[valid & (first_config < 0)] = config_index
          location_count[valid] += 1; best_near[valid] = np.maximum(best_near[valid], np.sum(near[:, valid], axis=0))
          for surface_index in range(len(SURFACES)): support_masks[valid & near[surface_index]] |= 1 << surface_index
          configurations.append({"configuration_id": config_index, "middle_fraction": middle, "ring_fraction": ring, "little_fraction": little, "WRJ2_offset_deg": wrist_offset, "forearm_PS_deg": forearm, "valid_center_count": int(np.count_nonzero(valid))})
          configuration_qpos.append(qpos)
          config_index += 1
    valid_locations = location_count > 0; labels, basin_count = ndimage.label(valid_locations.reshape(shape), structure=np.ones((3, 3, 3), dtype=np.int8))
    boundary_distance = ndimage.distance_transform_edt(valid_locations.reshape(shape), sampling=float(cfg["center_spacing_m"]))
    basins = []
    old_pocket = np.asarray(json.loads((ROOT / "outputs/phase3C07/static_reachability.json").read_text(encoding="utf-8"))["pocket_volume"]["feasible_centers_palm_m"])
    for label in range(1, basin_count + 1):
        indices = np.flatnonzero(labels.ravel() == label); basin_points = points[indices]; masks = support_masks[indices]; dominant = []
        for surface_index, surface in enumerate(SURFACES):
            if np.mean((masks & (1 << surface_index)) > 0) >= .5: dominant.append(surface)
        centroid = np.mean(basin_points, axis=0); distance_old = float(np.min(np.linalg.norm(old_pocket - centroid, axis=1)))
        old_indices = [_nearest_grid_index(point, axes) for point in old_pocket]
        old_members = sum(int(labels[index] == label) for index in old_indices)
        representative_flat = int(indices[np.argmax(boundary_distance.ravel()[indices])]); representative_center = points[representative_flat]
        representative_config_id = int(first_config[representative_flat]); representative_qpos = configuration_qpos[representative_config_id]
        direct = cspace_connectivity(base_qpos, np.asarray(representative["series"]["sphere_center_palm_m"][source_index]), np.asarray([representative_center]), 0.002, f"storage_direct_BASIN_{label:02d}")
        reconfigured = cspace_connectivity(representative_qpos, np.asarray(representative["series"]["sphere_center_palm_m"][source_index]), np.asarray([representative_center]), 0.002, f"storage_reconfigured_BASIN_{label:02d}")
        interpolation_clear = True; object_address = scene.model.jnt_qposadr[scene.object_joint_id]
        for alpha in np.linspace(0.0, 1.0, 11):
            interpolated = (1.0 - alpha) * base_qpos + alpha * representative_qpos
            interpolated[object_address:object_address + 7] = base_qpos[object_address:object_address + 7]
            value, _ = exact_clearance_grid(scene, interpolated, np.asarray([representative["series"]["sphere_center_palm_m"][source_index]]))
            interpolation_clear &= bool(value[0] >= -float(cfg["occupancy_numerical_tolerance_m"]))
        reachability = "DIRECTLY_GEOMETRICALLY_CONNECTED" if direct["path_exists"] else ("CONNECTED_ONLY_WITH_HAND_RECONFIGURATION" if reconfigured["path_exists"] and interpolation_clear else "NOT_CONNECTED_UNDER_TESTED_GEOMETRY")
        max_near = int(np.max(best_near[indices])); cage = "GEOMETRIC_CAGE_CANDIDATE" if max_near >= 4 else ("PARTIAL_CAGE" if max_near >= 2 else "OPEN_SUPPORT")
        basins.append({"basin_id": f"BASIN_{label:02d}", "voxel_count": len(indices), "volume_m3": float(len(indices) * float(cfg["center_spacing_m"]) ** 3),
                       "centroid_palm_m": centroid.tolist(), "bounds_palm_m": [basin_points.min(axis=0).tolist(), basin_points.max(axis=0).tolist()],
                       "dominant_support_surfaces": dominant, "maximum_near_surface_count": max_near, "confinement": cage,
                       "maximum_distance_to_sampled_boundary_m": float(np.max(boundary_distance.ravel()[indices])),
                       "distance_to_previous_ulnar_pocket_m": distance_old,
                       "previous_ulnar_pocket_voxels_in_basin": old_members,
                       "representative_center_palm_m": representative_center.tolist(), "representative_configuration_id": representative_config_id,
                       "aperture_bottleneck_width_m": direct["bottleneck_opening_width_m"] if direct["path_exists"] else reconfigured["bottleneck_opening_width_m"],
                       "direct_cspace_path_length_m": direct["shortest_path_length_m"], "reconfigured_cspace_path_length_m": reconfigured["shortest_path_length_m"],
                       "joint_interpolation_clear": interpolation_clear,
                       "resource_availability": {surface: bool(np.mean((masks & (1 << surface_index)) == 0) >= .5) for surface_index, surface in enumerate(SURFACES) if surface != "palm"},
                       "reachability": reachability})
    basins.sort(key=lambda basin: basin["basin_id"])
    return {"reduced_search_dimension": 5, "variables": ["middle flexion", "ring flexion", "little flexion", "WRJ2 offset", "forearm_PS"],
            "candidate_configurations_evaluated": len(configurations), "center_voxels_per_configuration": len(points),
            "valid_configuration_center_pairs": int(sum(row["valid_center_count"] for row in configurations)),
            "valid_unique_storage_centers": int(np.count_nonzero(valid_locations)), "basin_count": len(basins), "basins": basins,
            "grid_bounds_palm_m": [points.min(axis=0).tolist(), points.max(axis=0).tolist()], "grid_spacing_m": cfg["center_spacing_m"],
            "near_surface_numerical_tolerance_m": cfg["near_surface_numerical_tolerance_m"], "configurations": configurations,
            "human_ulnar_pocket_assumed_optimal": False}


def phase3c09_contract() -> dict[str, Any]:
    return {"new_dynamic_rollout_steps": 0, "mj_step_called": False, "rl_training": False, "object_B": False,
            "thumb_release": False, "friction_changed": False, "contact_changed": False, "skin_added": False,
            "joint_limits_changed": False, "actuator_limits_changed": False, "object_size_changed": False}


def run_phase3c09() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    trajectory = trajectory_failure_audit(); (OUTPUT / "trajectory_failure_audit.json").write_text(json.dumps(trajectory, indent=2), encoding="utf-8")
    cspace = cspace_audit(trajectory); (OUTPUT / "cspace_connectivity_audit.json").write_text(json.dumps(cspace, indent=2), encoding="utf-8")
    contact = contact_accessibility_audit(trajectory, cspace); (OUTPUT / "contact_accessibility_audit.json").write_text(json.dumps(contact, indent=2), encoding="utf-8")
    storage = storage_manifold_audit(trajectory); (OUTPUT / "storage_manifold_audit.json").write_text(json.dumps(storage, indent=2), encoding="utf-8")
    result = {"phase": "3C-0.9", "branch": "codex/phase3C09-storage-reachability-contact-migration",
              "base_commit": "db31ac6321dbf241c0c52990d83df8926bdd56d8", "trajectory": trajectory,
              "cspace": cspace, "contact_accessibility": contact, "storage_manifold": storage,
              "contract": phase3c09_contract()}
    (OUTPUT / "phase3c09_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
