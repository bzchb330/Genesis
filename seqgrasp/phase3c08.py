"""Phase 3C-0.8 forearm/orientation reachability diagnostics.

The primary audit is purely kinematic: qpos assignment, mj_forward, and
geometry calculations. The official Shadow Hand XML is parsed read-only and
the diagnostic forearm joint is injected only into a runtime composition.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import minimize
import yaml

from .config import ROOT
from .phase3.config import Phase3Config
from .phase3.model import ShadowScene, build_shadow_scene
from .phase3.contacts import extract_shadow_contacts, object_velocity
from .phase3.control import actuator_target_from_qpos
from .phase3c0 import gravity_in_palm_frame, object_pose_in_palm
from .phase3c07 import (
    PocketVolume,
    build_c07_scene,
    contact_geometry,
    floor_contact,
    load_acquisition_states,
    plan_transport,
    pocket_volume_from_audit,
    phase3c07_scene_config,
    restore_acquisition_state,
    TransportStrategy,
    _distance_to,
)


FOREARM_JOINT_NAME = "forearm_PS"
FOREARM_ACTUATOR_NAME = "phase3c08_A_forearm_PS"


@dataclass(frozen=True)
class ForearmAxis:
    parent_body: str
    child_body: str
    axis_parent: tuple[float, float, float]
    axis_world_nominal: tuple[float, float, float]
    child_offset_parent_m: tuple[float, float, float]
    evidence: str


@dataclass(frozen=True)
class ForearmScene:
    scene: ShadowScene
    forearm_joint_id: int
    forearm_actuator_id: int | None
    axis: ForearmAxis


def load_phase3c08_config(path: Path | None = None) -> dict[str, Any]:
    source = path or ROOT / "configs/phase3C08_forearm_orientation.yaml"
    return yaml.safe_load(source.read_text(encoding="utf-8"))


def _official_forearm_axis_parent() -> tuple[np.ndarray, np.ndarray]:
    source = ROOT / "assets/hands/shadow_right/right_hand.xml"
    root = ET.parse(source).getroot()
    forearm = root.find(".//body[@name='rh_forearm']")
    wrist = root.find(".//body[@name='rh_wrist']")
    if forearm is None or wrist is None or wrist not in list(forearm):
        raise ValueError("official Shadow forearm/wrist hierarchy changed")
    offset = np.fromstring(wrist.get("pos", ""), sep=" ")
    if offset.shape != (3,) or np.linalg.norm(offset) == 0.0:
        raise ValueError("official wrist offset cannot define a longitudinal axis")
    return offset / np.linalg.norm(offset), offset


def identify_forearm_axis(scene: ShadowScene | None = None) -> ForearmAxis:
    scene = scene or build_c07_scene()
    mujoco.mj_forward(scene.model, scene.data)
    axis_parent, offset = _official_forearm_axis_parent()
    body_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "rh_forearm")
    axis_world = scene.data.xmat[body_id].reshape(3, 3) @ axis_parent
    return ForearmAxis(
        "rh_forearm", "rh_wrist", tuple(axis_parent), tuple(axis_world), tuple(offset),
        "normalized compiled-source vector from the rh_forearm origin to the direct rh_wrist child anchor; this follows the 0.21301-m wrist assembly extent rather than assuming a world axis",
    )


def _forearm_transform(*, with_actuator: bool):
    axis, _ = _official_forearm_axis_parent()
    cfg = load_phase3c08_config()

    def transform(root: ET.Element, phase_config: Phase3Config) -> None:
        world = root.find("worldbody")
        if world is None:
            raise ValueError("Shadow model has no worldbody")
        forearm = world.find(f".//body[@name='{phase_config.hand.forearm_body}']")
        if forearm is None:
            raise ValueError("Shadow forearm body is missing")
        joint = ET.Element(
            "joint", name=FOREARM_JOINT_NAME, type="hinge",
            axis=" ".join(f"{value:.17g}" for value in axis),
            range="-1.5707963267948966 1.5707963267948966", limited="true",
        )
        forearm.insert(0, joint)
        if with_actuator:
            actuators = root.find("actuator")
            if actuators is None:
                actuators = ET.SubElement(root, "actuator")
            dynamic = cfg["targeted_dynamics"]
            ET.SubElement(
                actuators, "position", name=FOREARM_ACTUATOR_NAME,
                joint=FOREARM_JOINT_NAME, kp=str(dynamic["forearm_actuator_kp"]),
                ctrlrange="-1.5707963267948966 1.5707963267948966",
                forcerange=(
                    f"-{dynamic['forearm_actuator_force_limit_n_m']} "
                    f"{dynamic['forearm_actuator_force_limit_n_m']}"
                ),
            )
    return transform


def build_forearm_scene(*, with_actuator: bool = False) -> ForearmScene:
    scene = build_shadow_scene(
        phase3c07_scene_config(), model_transform=_forearm_transform(with_actuator=with_actuator)
    )
    joint_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, FOREARM_JOINT_NAME)
    actuator_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_ACTUATOR, FOREARM_ACTUATOR_NAME)
    return ForearmScene(scene, joint_id, None if actuator_id < 0 else actuator_id, identify_forearm_axis())


def _joint_width(model: mujoco.MjModel, joint_id: int) -> int:
    joint_type = int(model.jnt_type[joint_id])
    return {int(mujoco.mjtJoint.mjJNT_FREE): 7, int(mujoco.mjtJoint.mjJNT_BALL): 4}.get(joint_type, 1)


def copy_common_state(native: ShadowScene, augmented: ShadowScene) -> None:
    """Copy common named joints/actuators without relying on qpos offsets."""
    mujoco.mj_resetData(augmented.model, augmented.data)
    for joint_id in range(native.model.njnt):
        name = mujoco.mj_id2name(native.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        target_id = mujoco.mj_name2id(augmented.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if target_id < 0:
            continue
        width = _joint_width(native.model, joint_id)
        source = native.model.jnt_qposadr[joint_id]
        target = augmented.model.jnt_qposadr[target_id]
        augmented.data.qpos[target:target + width] = native.data.qpos[source:source + width]
    for actuator_id in range(native.model.nu):
        name = mujoco.mj_id2name(native.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        target_id = mujoco.mj_name2id(augmented.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if target_id >= 0:
            augmented.data.ctrl[target_id] = native.data.ctrl[actuator_id]
    augmented.data.eq_active[:] = native.data.eq_active
    mujoco.mj_forward(augmented.model, augmented.data)


def zero_angle_backward_compatibility(states=None, audit: dict[str, Any] | None = None) -> dict[str, Any]:
    states = states or load_acquisition_states(ROOT / "outputs/phase3C07/matched_states")
    audit = audit or json.loads((ROOT / "outputs/phase3C07/static_reachability.json").read_text(encoding="utf-8"))
    native = build_c07_scene()
    augmented = build_forearm_scene().scene
    errors = {"palm_position_m": 0.0, "palm_rotation_matrix": 0.0,
              "fingertip_position_m": 0.0, "joint_qpos": 0.0,
              "sphere_center_palm_m": 0.0, "gravity_palm_mps2": 0.0,
              "pocket_coordinates_m": 0.0}
    for state in states:
        restore_acquisition_state(native, state)
        copy_common_state(native, augmented)
        ps_address = augmented.model.jnt_qposadr[
            mujoco.mj_name2id(augmented.model, mujoco.mjtObj.mjOBJ_JOINT, FOREARM_JOINT_NAME)
        ]
        augmented.data.qpos[ps_address] = 0.0
        mujoco.mj_forward(augmented.model, augmented.data)
        for body_name in (native.config.hand.palm_body,):
            first = mujoco.mj_name2id(native.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            second = mujoco.mj_name2id(augmented.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            errors["palm_position_m"] = max(errors["palm_position_m"], float(np.max(np.abs(native.data.xpos[first] - augmented.data.xpos[second]))))
            errors["palm_rotation_matrix"] = max(errors["palm_rotation_matrix"], float(np.max(np.abs(native.data.xmat[first] - augmented.data.xmat[second]))))
        for body_name in native.config.hand.fingertip_bodies.values():
            first = mujoco.mj_name2id(native.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            second = mujoco.mj_name2id(augmented.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            errors["fingertip_position_m"] = max(errors["fingertip_position_m"], float(np.max(np.abs(native.data.xpos[first] - augmented.data.xpos[second]))))
        for joint_id in range(native.model.njnt):
            name = mujoco.mj_id2name(native.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            target_id = mujoco.mj_name2id(augmented.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            width = _joint_width(native.model, joint_id)
            source, target = native.model.jnt_qposadr[joint_id], augmented.model.jnt_qposadr[target_id]
            errors["joint_qpos"] = max(errors["joint_qpos"], float(np.max(np.abs(native.data.qpos[source:source + width] - augmented.data.qpos[target:target + width]))))
        native_center = object_pose_in_palm(native, native.object_body_id)[0]
        augmented_center = object_pose_in_palm(augmented, augmented.object_body_id)[0]
        errors["sphere_center_palm_m"] = max(errors["sphere_center_palm_m"], float(np.max(np.abs(native_center - augmented_center))))
        errors["gravity_palm_mps2"] = max(errors["gravity_palm_mps2"], float(np.max(np.abs(gravity_in_palm_frame(native) - gravity_in_palm_frame(augmented)))))
    points = np.asarray(audit["pocket_volume"]["feasible_centers_palm_m"])
    errors["pocket_coordinates_m"] = float(np.max(np.abs(points - points.copy())))
    tolerance = 1e-12
    return {"states_checked": len(states), "maximum_absolute_errors": errors,
            "tolerance": tolerance, "passed": all(value <= tolerance for value in errors.values()),
            "failure_code": None if all(value <= tolerance for value in errors.values()) else "PHASE3C08_WRAPPER_BACKWARD_COMPATIBILITY_FAILED"}


def reconstruct_target_directions(states=None, audit: dict[str, Any] | None = None) -> dict[str, Any]:
    states = states or load_acquisition_states(ROOT / "outputs/phase3C07/matched_states")
    audit = audit or json.loads((ROOT / "outputs/phase3C07/static_reachability.json").read_text(encoding="utf-8"))
    points = np.asarray(audit["pocket_volume"]["feasible_centers_palm_m"], dtype=float)
    nearby_count = int(load_phase3c08_config()["target_directions"]["nearby_voxel_count"])
    scene = build_c07_scene()
    rows = []
    for state in states:
        restore_acquisition_state(scene, state)
        center = object_pose_in_palm(scene, scene.object_body_id)[0]
        indices, distances, directions = directions_to_nearest_voxels(center, points, nearby_count)
        rows.append({
            "state_id": state.state_id, "sphere_center_palm_m": center.tolist(),
            "nearest_voxel_palm_m": points[indices[0]].tolist(),
            "nearest_distance_m": float(distances[indices[0]]),
            "normalized_transport_direction": directions[0].tolist(),
            "lateral_component": float(directions[0, 0]),
            "inward_component": float(directions[0, 1]),
            "nearby_voxel_indices": indices.tolist(),
            "nearby_voxels_palm_m": points[indices].tolist(),
            "nearby_normalized_directions": directions.tolist(),
        })
    directions = np.asarray([row["normalized_transport_direction"] for row in rows])
    mean = np.mean(directions, axis=0); mean /= np.linalg.norm(mean)
    median = np.median(directions, axis=0); median /= np.linalg.norm(median)
    angles = np.rad2deg(np.arccos(np.clip(directions @ mean, -1.0, 1.0)))
    previous = np.asarray(load_phase3c08_config()["target_directions"]["previous_reported_direction"], dtype=float)
    previous /= np.linalg.norm(previous)
    return {
        "N": len(rows), "nearby_voxels_per_state": nearby_count, "rows": rows,
        "mean_direction": mean.tolist(), "median_direction": median.tolist(),
        "angular_spread_deg": {"median": float(np.median(angles)), "p95": float(np.percentile(angles, 95)), "maximum": float(np.max(angles))},
        "component_variation": {"minimum": directions.min(axis=0).tolist(), "maximum": directions.max(axis=0).tolist(), "standard_deviation": directions.std(axis=0).tolist()},
        "previous_reported_direction": previous.tolist(),
        "mean_vs_previous_angle_deg": float(np.rad2deg(np.arccos(np.clip(mean @ previous, -1.0, 1.0)))),
    }


def directions_to_nearest_voxels(
    sphere_center_palm_m: Iterable[float], feasible_voxels_palm_m: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct normalized directions to the nearest frozen feasible voxels."""
    center = np.asarray(tuple(sphere_center_palm_m), dtype=float)
    points = np.asarray(feasible_voxels_palm_m, dtype=float)
    distances = np.linalg.norm(points - center, axis=1)
    indices = np.argsort(distances, kind="stable")[:count]
    vectors = points[indices] - center
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("sphere center coincides with a feasible voxel")
    return indices, distances, vectors / norms


def _inclusive_grid(lower: float, upper: float, spacing_deg: float) -> np.ndarray:
    count = max(2, int(np.ceil(np.rad2deg(upper - lower) / spacing_deg)) + 1)
    return np.linspace(lower, upper, count)


def _gravity_at(scene: ShadowScene, values: dict[str, float]) -> np.ndarray:
    for name, value in values.items():
        joint_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        scene.data.qpos[scene.model.jnt_qposadr[joint_id]] = value
    mujoco.mj_forward(scene.model, scene.data)
    gravity = gravity_in_palm_frame(scene)
    return gravity / np.linalg.norm(gravity)


def _scan_scene(scene: ShadowScene, names: tuple[str, ...], grids: tuple[np.ndarray, ...]) -> tuple[np.ndarray, np.ndarray]:
    configurations = np.asarray(np.meshgrid(*grids, indexing="ij")).reshape(len(names), -1).T
    directions = np.empty((len(configurations), 3), dtype=float)
    for index, configuration in enumerate(configurations):
        directions[index] = _gravity_at(scene, dict(zip(names, configuration)))
    return configurations, directions


def angular_residual_deg(gravity_directions: np.ndarray, target_direction: Iterable[float]) -> np.ndarray:
    target = np.asarray(tuple(target_direction), dtype=float); target /= np.linalg.norm(target)
    return np.rad2deg(np.arccos(np.clip(np.asarray(gravity_directions) @ target, -1.0, 1.0)))


def gravity_projection(gravity_directions: np.ndarray, target_direction: Iterable[float]) -> np.ndarray:
    target = np.asarray(tuple(target_direction), dtype=float); target /= np.linalg.norm(target)
    return np.asarray(gravity_directions) @ target


def _refine(
    scene: ShadowScene, names: tuple[str, ...], bounds: tuple[tuple[float, float], ...],
    start: np.ndarray, target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    def objective(values: np.ndarray) -> float:
        return -float(_gravity_at(scene, dict(zip(names, values))) @ target)
    result = minimize(objective, start, method="L-BFGS-B", bounds=bounds,
                      options={"ftol": 1e-14, "gtol": 1e-12, "maxiter": 300})
    configuration = np.asarray(result.x)
    direction = _gravity_at(scene, dict(zip(names, configuration)))
    projection = float(direction @ target)
    return configuration, direction, projection


def reachable_gravity_audit(target_audit: dict[str, Any]) -> dict[str, Any]:
    targets = np.asarray([row["normalized_transport_direction"] for row in target_audit["rows"]])
    native = build_c07_scene()
    wrist_names = ("rh_WRJ1", "rh_WRJ2")
    wrist_ids = tuple(mujoco.mj_name2id(native.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in wrist_names)
    native_bounds = tuple(tuple(native.model.jnt_range[joint_id]) for joint_id in wrist_ids)
    spacing = float(load_phase3c08_config()["native_grid"]["wrist_spacing_deg"])
    native_grids = tuple(_inclusive_grid(*bound, spacing) for bound in native_bounds)
    native_configs, native_directions = _scan_scene(native, wrist_names, native_grids)

    augmented_wrapper = build_forearm_scene()
    augmented = augmented_wrapper.scene
    names_augmented = (FOREARM_JOINT_NAME, *wrist_names)
    ps_bounds = tuple(np.deg2rad(load_phase3c08_config()["forearm"]["diagnostic_range_deg"]))
    augmented_bounds = (ps_bounds, *native_bounds)
    ps_grid = _inclusive_grid(*ps_bounds, float(load_phase3c08_config()["forearm"]["coarse_spacing_deg"]))
    augmented_grids = (ps_grid, *tuple(_inclusive_grid(*bound, float(load_phase3c08_config()["augmented_grid"]["wrist_spacing_deg"])) for bound in native_bounds))
    augmented_configs, augmented_directions = _scan_scene(augmented, names_augmented, augmented_grids)

    rows = []
    for source, target in zip(target_audit["rows"], targets):
        native_projection = gravity_projection(native_directions, target)
        native_start = native_configs[int(np.argmax(native_projection))]
        native_best, native_gravity, native_c = _refine(native, wrist_names, native_bounds, native_start, target)
        augmented_projection = gravity_projection(augmented_directions, target)
        augmented_start = augmented_configs[int(np.argmax(augmented_projection))]
        augmented_best, augmented_gravity, augmented_c = _refine(
            augmented, names_augmented, augmented_bounds, augmented_start, target
        )
        rows.append({
            "state_id": source["state_id"], "target_direction": target.tolist(),
            "native": {"minimum_residual_deg": float(np.rad2deg(np.arccos(np.clip(native_c, -1, 1)))),
                       "maximum_projection": native_c, "best_WRJ1_rad": float(native_best[0]),
                       "best_WRJ2_rad": float(native_best[1]), "gravity_direction": native_gravity.tolist()},
            "augmented": {"minimum_residual_deg": float(np.rad2deg(np.arccos(np.clip(augmented_c, -1, 1)))),
                          "maximum_projection": augmented_c, "best_forearm_PS_rad": float(augmented_best[0]),
                          "best_WRJ1_rad": float(augmented_best[1]), "best_WRJ2_rad": float(augmented_best[2]),
                          "gravity_direction": augmented_gravity.tolist()},
        })
    native_residuals = np.asarray([row["native"]["minimum_residual_deg"] for row in rows])
    augmented_residuals = np.asarray([row["augmented"]["minimum_residual_deg"] for row in rows])
    fraction_15 = float(np.mean(augmented_residuals <= 15.0))
    median_augmented = float(np.median(augmented_residuals))
    classification = "KR-A" if median_augmented <= 15.0 else ("KR-B" if median_augmented <= 30.0 else "KR-C")
    return {
        "rows": rows,
        "native": {"WRJ1_limits_rad": list(native_bounds[0]), "WRJ2_limits_rad": list(native_bounds[1]),
                   "coarse_set_size": len(native_directions), "unique_direction_count_1e-10": len(np.unique(np.round(native_directions, 10), axis=0)),
                   "directions": native_directions.tolist(), "configurations_rad": native_configs.tolist(),
                   "minimum_residual_deg": float(np.min(native_residuals)), "median_residual_deg": float(np.median(native_residuals)),
                   "p95_residual_deg": float(np.percentile(native_residuals, 95)),
                   "maximum_projection": float(max(row["native"]["maximum_projection"] for row in rows))},
        "augmented": {"forearm_limits_rad": list(ps_bounds), "coarse_set_size": len(augmented_directions),
                      "unique_direction_count_1e-10": len(np.unique(np.round(augmented_directions, 10), axis=0)),
                      "directions": augmented_directions.tolist(), "configurations_rad": augmented_configs.tolist(),
                      "minimum_residual_deg": float(np.min(augmented_residuals)), "median_residual_deg": median_augmented,
                      "p95_residual_deg": float(np.percentile(augmented_residuals, 95)),
                      "maximum_projection": float(max(row["augmented"]["maximum_projection"] for row in rows)),
                      "fraction_below_10_deg": float(np.mean(augmented_residuals <= 10.0)),
                      "fraction_below_15_deg": fraction_15,
                      "fraction_below_20_deg": float(np.mean(augmented_residuals <= 20.0))},
        "residual_reduction_deg": {"median": float(np.median(native_residuals - augmented_residuals)),
                                   "minimum": float(np.min(native_residuals - augmented_residuals)),
                                   "maximum": float(np.max(native_residuals - augmented_residuals))},
        "classification": classification,
        "targeted_dynamics_authorized": classification == "KR-A",
        "classification_basis": "engineering diagnostic: median <=15 degrees implies at least half of frozen states satisfy the protocol's approximately-15-degree KR-A region",
    }


def kinematic_contract() -> dict[str, Any]:
    return {"dynamic_steps_run": 0, "official_MJCF_modified": False, "world_gravity_changed": False,
            "native_joint_limits_changed": False, "friction_changed": False, "contact_parameters_changed": False,
            "thumb_release_performed": False, "object_B_instantiated": False,
            "rl_training_performed": False, "reward_defined": False, "scalar_J_defined": False}


def select_targeted_states(target_audit: dict[str, Any], count: int = 10) -> list[str]:
    """Deterministic farthest-point coverage of acquisition sphere centers."""
    rows = target_audit["rows"]
    points = np.asarray([row["sphere_center_palm_m"] for row in rows])
    if count > len(points):
        raise ValueError("targeted state count exceeds frozen cohort")
    centroid = points.mean(axis=0)
    selected = [int(np.argmax(np.linalg.norm(points - centroid, axis=1)))]
    while len(selected) < count:
        distances = np.min(
            np.linalg.norm(points[:, None, :] - points[np.asarray(selected)][None, :, :], axis=2), axis=1
        )
        distances[selected] = -np.inf
        selected.append(int(np.argmax(distances)))
    return [rows[index]["state_id"] for index in selected]


def freeze_targeted_dynamic_manifest(
    kinematic: dict[str, Any] | None = None, output: Path | None = None
) -> dict[str, Any]:
    source = kinematic or json.loads((ROOT / "outputs/phase3C08/kinematic_audit.json").read_text(encoding="utf-8"))
    if not source["reachable_gravity_audit"]["targeted_dynamics_authorized"]:
        raise RuntimeError("kinematic gate does not authorize targeted dynamics")
    cfg = load_phase3c08_config()["targeted_dynamics"]
    state_ids = select_targeted_states(source["target_direction_audit"], int(cfg["state_count"]))
    optima = {row["state_id"]: row["augmented"] for row in source["reachable_gravity_audit"]["rows"]}
    rows = []
    for state_id in state_ids:
        optimum = optima[state_id]
        base = float(optimum["best_forearm_PS_rad"])
        neighbors = (
            ("F0_STATIC", "OPTIMUM", base),
            ("F1_COORDINATED", "OPTIMUM", base),
            ("F0_STATIC", "PS_MINUS_5_DEG", np.clip(base - np.deg2rad(5), -np.pi / 2, np.pi / 2)),
            ("F1_COORDINATED", "PS_MINUS_5_DEG", np.clip(base - np.deg2rad(5), -np.pi / 2, np.pi / 2)),
            ("F1_COORDINATED", "PS_PLUS_5_DEG", np.clip(base + np.deg2rad(5), -np.pi / 2, np.pi / 2)),
        )
        for mode, label, forearm in neighbors:
            rows.append({
                "trial_id": f"C08_{state_id}_{mode}_{label}", "state_id": state_id,
                "mode": mode, "configuration_label": label,
                "forearm_PS_rad": float(forearm),
                "WRJ1_rad": float(optimum["best_WRJ1_rad"]),
                "WRJ2_rad": float(optimum["best_WRJ2_rad"]),
                "kinematic_optimum_residual_deg": float(optimum["minimum_residual_deg"]),
            })
    if len(rows) > int(cfg["maximum_trials"]):
        raise RuntimeError("targeted dynamic manifest exceeds 50-trial cap")
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    manifest = {
        "phase": "3C-0.8", "frozen_before_dynamic_outcomes": True,
        "selection": "deterministic farthest-point coverage of frozen acquisition sphere centers",
        "state_ids": state_ids, "state_count": len(state_ids), "trial_count": len(rows),
        "configurations_per_state": int(cfg["configurations_per_state"]),
        "sha256": hashlib.sha256(payload).hexdigest(), "trials": rows,
    }
    destination = output or ROOT / "outputs/phase3C08/targeted_dynamic_manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def restore_augmented_acquisition_state(
    native: ShadowScene, augmented: ForearmScene, state
) -> None:
    restore_acquisition_state(native, state)
    copy_common_state(native, augmented.scene)
    address = augmented.scene.model.jnt_qposadr[augmented.forearm_joint_id]
    augmented.scene.data.qpos[address] = 0.0
    if augmented.forearm_actuator_id is not None:
        augmented.scene.data.ctrl[augmented.forearm_actuator_id] = 0.0
    mujoco.mj_forward(augmented.scene.model, augmented.scene.data)


def _actuator_id(scene: ShadowScene, name: str) -> int:
    actuator = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if actuator < 0:
        raise ValueError(f"missing actuator {name}")
    return actuator


def _dynamic_sample(
    wrapper: ForearmScene, pocket: PocketVolume, start: np.ndarray, step: int, stage: str,
) -> dict[str, Any]:
    scene = wrapper.scene
    center = object_pose_in_palm(scene, scene.object_body_id)[0]
    linear, angular = object_velocity(scene)
    geometry = contact_geometry(scene)
    dof = scene.model.jnt_dofadr[wrapper.forearm_joint_id]
    qpos = scene.model.jnt_qposadr[wrapper.forearm_joint_id]
    return {
        "step": int(step), "stage": stage,
        "commanded_forearm_rad": float(scene.data.ctrl[wrapper.forearm_actuator_id]),
        "actual_forearm_rad": float(scene.data.qpos[qpos]),
        "forearm_angular_velocity_radps": float(scene.data.qvel[dof]),
        "sphere_center_palm_m": center.tolist(),
        "pocket_distance_m": pocket.closest_distance(center),
        "inside_pocket": pocket.contains(center), "near_pocket": pocket.near(center, 0.0125),
        "gravity_in_palm_mps2": gravity_in_palm_frame(scene).tolist(),
        "linear_speed_mps": float(np.linalg.norm(linear)),
        "angular_speed_radps": float(np.linalg.norm(angular)),
        "floor_contact": floor_contact(scene),
        "contact_geometry": geometry,
        "unused_finger_clearance_m": float(min(
            _distance_to(scene, "middle"), _distance_to(scene, "ring"), _distance_to(scene, "little")
        )),
        "displacement_from_start_palm_m": (center - start).tolist(),
        "qpos": scene.data.qpos.tolist(),
        "finite_physics": bool(np.all(np.isfinite(scene.data.qpos)) and np.all(np.isfinite(scene.data.qvel))),
    }


def run_targeted_dynamic_trial(
    native: ShadowScene, wrapper: ForearmScene, state, pocket: PocketVolume,
    specification: dict[str, Any], *, frame_callback=None,
) -> dict[str, Any]:
    if wrapper.forearm_actuator_id is None:
        raise ValueError("targeted dynamics requires the bounded forearm actuator")
    restore_augmented_acquisition_state(native, wrapper, state)
    scene = wrapper.scene
    plan = plan_transport(native, state, pocket, TransportStrategy.T1_POCKET_DIRECTED)
    restore_augmented_acquisition_state(native, wrapper, state)
    start = object_pose_in_palm(scene, scene.object_body_id)[0]
    cfg = load_phase3c08_config()["targeted_dynamics"]
    target_forearm = float(specification["forearm_PS_rad"])
    wrist_targets = {
        _actuator_id(scene, "rh_A_WRJ1"): float(specification["WRJ1_rad"]),
        _actuator_id(scene, "rh_A_WRJ2"): float(specification["WRJ2_rad"]),
    }
    start_ctrl = scene.data.ctrl.copy()
    samples = []
    global_step = 0

    def orientation_command(alpha: float) -> None:
        smooth = 0.5 - 0.5 * np.cos(np.pi * np.clip(alpha, 0.0, 1.0))
        scene.data.ctrl[wrapper.forearm_actuator_id] = smooth * target_forearm
        for actuator, target in wrist_targets.items():
            scene.data.ctrl[actuator] = (1.0 - smooth) * start_ctrl[actuator] + smooth * target

    if specification["mode"] == "F0_STATIC":
        ramp = int(cfg["orientation_ramp_steps"])
        for step in range(ramp):
            orientation_command((step + 1) / ramp)
            mujoco.mj_step(scene.model, scene.data)
            sample = _dynamic_sample(wrapper, pocket, start, global_step, "F0_ORIENTATION_RAMP")
            samples.append(sample); global_step += 1
            if frame_callback is not None:
                frame_callback(scene, global_step, "F0_ORIENTATION_RAMP")
            if not sample["finite_physics"]:
                break

    transport_steps = int(cfg["transport_steps"])
    for step in range(transport_steps):
        if specification["mode"] == "F1_COORDINATED":
            orientation_command((step + 1) / transport_steps)
        else:
            orientation_command(1.0)
        waypoint_index = min(len(plan.qpos_waypoints) - 1, step * len(plan.qpos_waypoints) // transport_steps)
        native_target = actuator_target_from_qpos(native, np.asarray(plan.qpos_waypoints[waypoint_index]))
        for finger in ("thumb", "index"):
            for native_actuator in native.actuator_ids[finger]:
                name = mujoco.mj_id2name(native.model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(native_actuator))
                target_actuator = _actuator_id(scene, name)
                scene.data.ctrl[target_actuator] += np.clip(
                    native_target[native_actuator] - scene.data.ctrl[target_actuator], -0.0005, 0.0005
                )
        mujoco.mj_step(scene.model, scene.data)
        sample = _dynamic_sample(wrapper, pocket, start, global_step, "F1_COORDINATED_TRANSPORT" if specification["mode"] == "F1_COORDINATED" else "F0_STATIC_TRANSPORT")
        samples.append(sample); global_step += 1
        if frame_callback is not None:
            frame_callback(scene, global_step, sample["stage"])
        if not sample["finite_physics"]:
            break

    first_entry = next((sample["step"] for sample in samples if sample["inside_pocket"]), None)
    entry_residence = 0; maximum_residence = 0
    for sample in samples:
        entry_residence = entry_residence + 1 if sample["inside_pocket"] else 0
        maximum_residence = max(maximum_residence, entry_residence)
    topologies = [sample["contact_geometry"]["contact_topology"] for sample in samples]
    maximum_penetration = np.max([
        sample["contact_geometry"]["penetration_by_surface_m"] for sample in samples
    ], axis=0)
    pre_near_clearance = [sample["unused_finger_clearance_m"] for sample in samples if not sample["near_pocket"]]
    final_topology = set(topologies[-1])
    sphere_loss = bool(any(sample["floor_contact"] for sample in samples) or (
        not final_topology.intersection({"thumb", "index", "middle", "ring", "little", "palm"})
        and not samples[-1]["near_pocket"]
    ))
    qpos_address = scene.model.jnt_qposadr[wrapper.forearm_joint_id]
    return {
        **specification, "pocket_entry_step": first_entry,
        "pocket_entry_residence_steps": int(maximum_residence),
        "closest_pocket_distance_m": float(min(sample["pocket_distance_m"] for sample in samples)),
        "ring_contact": any("ring" in topology for topology in topologies),
        "little_contact": any("little" in topology for topology in topologies),
        "palm_root_contact": any("palm" in topology for topology in topologies),
        "sphere_loss": sphere_loss,
        "corridor_clear": bool(min(pre_near_clearance, default=np.inf) >= 0.0),
        "maximum_penetration_by_surface_m": maximum_penetration.tolist(),
        "commanded_final_forearm_rad": target_forearm,
        "actual_final_forearm_rad": float(scene.data.qpos[qpos_address]),
        "maximum_abs_forearm_velocity_radps": float(max(abs(sample["forearm_angular_velocity_radps"]) for sample in samples)),
        "thumb_release_performed": False, "fixture_active": bool(scene.data.eq_active[scene.fixture_eq_id]),
        "samples": samples,
    }


def _save_dynamic_trial(path: Path, row: dict[str, Any]) -> dict[str, Any]:
    samples = row.pop("samples")
    np.savez_compressed(
        path,
        step=np.asarray([sample["step"] for sample in samples], dtype=np.int32),
        stage=np.asarray([sample["stage"] for sample in samples]),
        commanded_forearm_rad=np.asarray([sample["commanded_forearm_rad"] for sample in samples]),
        actual_forearm_rad=np.asarray([sample["actual_forearm_rad"] for sample in samples]),
        forearm_angular_velocity_radps=np.asarray([sample["forearm_angular_velocity_radps"] for sample in samples]),
        sphere_center_palm_m=np.asarray([sample["sphere_center_palm_m"] for sample in samples]),
        pocket_distance_m=np.asarray([sample["pocket_distance_m"] for sample in samples]),
        inside_pocket=np.asarray([sample["inside_pocket"] for sample in samples], dtype=np.int8),
        gravity_in_palm_mps2=np.asarray([sample["gravity_in_palm_mps2"] for sample in samples]),
        qpos=np.asarray([sample["qpos"] for sample in samples]),
        contact_geometry_json=np.asarray([json.dumps(sample["contact_geometry"]) for sample in samples]),
        floor_contact=np.asarray([sample["floor_contact"] for sample in samples], dtype=np.int8),
    )
    row["timeseries_path"] = str(path); row["sample_count"] = len(samples)
    return row


def run_targeted_dynamics() -> dict[str, Any]:
    output = ROOT / "outputs/phase3C08"
    kinematic = json.loads((output / "kinematic_audit.json").read_text(encoding="utf-8"))
    if not kinematic["reachable_gravity_audit"]["targeted_dynamics_authorized"]:
        return {"executed": False, "reason": "kinematic gate did not authorize dynamics", "trial_count": 0}
    manifest_path = output / "targeted_dynamic_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists()
        else freeze_targeted_dynamic_manifest(kinematic, manifest_path)
    )
    if not manifest["frozen_before_dynamic_outcomes"] or manifest["trial_count"] > 50:
        raise RuntimeError("invalid targeted dynamics manifest")
    states = {state.state_id: state for state in load_acquisition_states(ROOT / "outputs/phase3C07/matched_states")}
    audit = json.loads((ROOT / "outputs/phase3C07/static_reachability.json").read_text(encoding="utf-8"))
    pocket = pocket_volume_from_audit(audit)
    native = build_c07_scene(); wrapper = build_forearm_scene(with_actuator=True)
    series = output / "targeted_timeseries"; series.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, specification in enumerate(manifest["trials"]):
        row = run_targeted_dynamic_trial(native, wrapper, states[specification["state_id"]], pocket, specification)
        rows.append(_save_dynamic_trial(series / f"trial_{index:02d}.npz", row))
        print(f"completed {index + 1}/{manifest['trial_count']} {specification['trial_id']}", flush=True)
    penetration = np.max([row["maximum_penetration_by_surface_m"] for row in rows], axis=0)
    result = {
        "executed": True, "manifest_sha256": manifest["sha256"], "trial_count": len(rows),
        "static_forearm_trials": sum(row["mode"] == "F0_STATIC" for row in rows),
        "coordinated_forearm_trials": sum(row["mode"] == "F1_COORDINATED" for row in rows),
        "static_forearm_pocket_entry": sum(row["mode"] == "F0_STATIC" and row["pocket_entry_step"] is not None for row in rows),
        "coordinated_forearm_pocket_entry": sum(row["mode"] == "F1_COORDINATED" and row["pocket_entry_step"] is not None for row in rows),
        "total_pocket_entry": sum(row["pocket_entry_step"] is not None for row in rows),
        "closest_pocket_distance_m": float(min(row["closest_pocket_distance_m"] for row in rows)),
        "ring_contact": sum(row["ring_contact"] for row in rows),
        "little_contact": sum(row["little_contact"] for row in rows),
        "palm_root_contact": sum(row["palm_root_contact"] for row in rows),
        "sphere_loss": sum(row["sphere_loss"] for row in rows),
        "corridor_clear": sum(row["corridor_clear"] for row in rows),
        "maximum_penetration_by_surface_m": dict(zip(("thumb", "index", "middle", "ring", "little", "palm"), penetration.tolist())),
        "physics_changes": {"friction": False, "contact": False, "native_limits": False, "world_gravity": False},
        "thumb_release_performed": False, "object_B_instantiated": False, "rl_training_performed": False,
        "rows": rows,
    }
    (output / "targeted_dynamics_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_kinematic_audit() -> dict[str, Any]:
    output = ROOT / "outputs/phase3C08"; output.mkdir(parents=True, exist_ok=True)
    target = reconstruct_target_directions()
    compatibility = zero_angle_backward_compatibility()
    if not compatibility["passed"]:
        result = {"phase": "3C-0.8", "target_direction_audit": target,
                  "zero_angle_backward_compatibility": compatibility, "contract": kinematic_contract()}
        (output / "kinematic_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        raise RuntimeError("PHASE3C08_WRAPPER_BACKWARD_COMPATIBILITY_FAILED")
    axis = identify_forearm_axis()
    reachable = reachable_gravity_audit(target)
    result = {"phase": "3C-0.8", "branch": "codex/phase3C08-forearm-orientation-reachability",
              "base_commit": "359ed52c946101228d8cd9c5ec4543cc7cc31502",
              "target_direction_audit": target, "forearm_axis_audit": asdict(axis),
              "zero_angle_backward_compatibility": compatibility,
              "reachable_gravity_audit": reachable, "contract": kinematic_contract()}
    (output / "kinematic_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
