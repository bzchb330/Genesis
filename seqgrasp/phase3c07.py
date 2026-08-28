"""Phase 3C-0.7: 25-mm pocket reachability and cage diagnostics.

The implementation changes object scale and scripted transport only. It does
not modify contact physics, release either acquisition digit, instantiate a
second object, train a policy, or define a scalar scientific objective.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import StrEnum
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np
from scipy.optimize import least_squares
import yaml

from .config import ROOT
from .phase3.config import SUPPORT_SURFACES, Phase3Config, load_phase3_config
from .phase3.contacts import extract_shadow_contacts, object_velocity
from .phase3.control import actuator_target_from_qpos
from .phase3.model import ShadowScene, build_shadow_scene, set_fixture
from .phase3c0 import gravity_in_palm_frame, object_pose_in_palm, palm_transform, world_to_palm
from .phase3c06 import (
    SphereAcquisitionState,
    _project,
    acquire_sphere_state,
    audit_non_thumb_link_lengths,
    floor_contact,
)


SPHERE_DIAMETER_M = 0.025
SPHERE_RADIUS_M = 0.0125
SPHERE_DENSITY_KG_M3 = 1000.0


class TransportStrategy(StrEnum):
    T0_OLD_DIRECT = "T0_OLD_DIRECT"
    T1_POCKET_DIRECTED = "T1_POCKET_DIRECTED"
    T2_WRIST_ASSISTED = "T2_WRIST_ASSISTED"


class PreshapeCondition(StrEnum):
    P0_AFTER_ENTRY = "P0_AFTER_ENTRY"
    P1_GEOMETRIC_APPROACH = "P1_GEOMETRIC_APPROACH"


def preshape_gate(
    condition: PreshapeCondition,
    *,
    inside_pocket: bool,
    near_pocket: bool,
    sweep_clearance_m: float,
) -> bool:
    """Protocol-defined geometric gate; it contains no time trigger."""
    if condition == PreshapeCondition.P0_AFTER_ENTRY:
        return bool(inside_pocket)
    return bool(near_pocket and sweep_clearance_m >= 0.0)


FAILURE_TAXONOMY = (
    "ACQUISITION_FAILED", "TRANSFER_CORRIDOR_BLOCKED",
    "POCKET_LATERAL_TRANSPORT_FAILED", "POCKET_INWARD_TRANSPORT_FAILED",
    "POCKET_NOT_REACHED", "WRIST_DIRECTION_UNFAVORABLE", "WRIST_DOF_LIMIT",
    "PRESHAPE_BLOCKED_TRANSFER", "PRESHAPE_MISALIGNED",
    "RING_CONTACT_NOT_ESTABLISHED", "LITTLE_CONTACT_NOT_ESTABLISHED",
    "PALM_ROOT_CONTACT_NOT_ESTABLISHED", "TRANSIENT_CONTACT_ONLY",
    "NO_LOAD_BEARING_CAGE", "SPHERE_ROLLED_OUT", "SPHERE_SLID_OUT",
    "EXCESSIVE_PENETRATION", "JOINT_BOUNDARY_LIMITED_TRANSPORT",
    "CAGE_HOLD_FAILED", "OTHER",
)


@dataclass(frozen=True)
class PocketVolume:
    lower_palm_m: tuple[float, float, float]
    upper_palm_m: tuple[float, float, float]
    voxel_half_width_m: tuple[float, float, float]
    feasible_centers_palm_m: tuple[tuple[float, float, float], ...]
    construction: str

    @property
    def center_palm_m(self) -> tuple[float, float, float]:
        return tuple((np.asarray(self.lower_palm_m) + np.asarray(self.upper_palm_m)) / 2.0)

    @property
    def volume_m3(self) -> float:
        cell = 2.0 * np.asarray(self.voxel_half_width_m)
        return float(len(self.feasible_centers_palm_m) * np.prod(cell))

    def contains(self, center_palm_m: Iterable[float]) -> bool:
        center = np.asarray(tuple(center_palm_m), dtype=float)
        points = np.asarray(self.feasible_centers_palm_m, dtype=float)
        if not len(points):
            return False
        half = np.asarray(self.voxel_half_width_m)
        return bool(np.any(np.all(np.abs(points - center) <= half + 1e-12, axis=1)))

    def closest_distance(self, center_palm_m: Iterable[float]) -> float:
        center = np.asarray(tuple(center_palm_m), dtype=float)
        points = np.asarray(self.feasible_centers_palm_m, dtype=float)
        return float(np.min(np.linalg.norm(points - center, axis=1))) if len(points) else float("inf")

    def near(self, center_palm_m: Iterable[float], expansion_m: float) -> bool:
        return self.closest_distance(center_palm_m) <= float(expansion_m)


@dataclass(frozen=True)
class TransportPlan:
    strategy: str
    desired_path_palm_m: tuple[tuple[float, float, float], ...]
    achievable_grasp_centers_palm_m: tuple[tuple[float, float, float], ...]
    qpos_waypoints: tuple[tuple[float, ...], ...]
    ik_residual_m: tuple[float, ...]
    straight_reference_palm_m: tuple[tuple[float, float, float], ...]


def load_phase3c07_config(path: Path | None = None) -> dict[str, Any]:
    source = path or ROOT / "configs/phase3C07_pocket_reachability_cage.yaml"
    with source.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def sphere_mass_kg() -> float:
    return float(SPHERE_DENSITY_KG_M3 * (4.0 / 3.0) * np.pi * SPHERE_RADIUS_M ** 3)


def phase3c07_scene_config() -> Phase3Config:
    base = load_phase3_config()
    cfg = load_phase3c07_config()["object"]
    raw = dict(base.raw)
    raw["object"] = {
        "name": cfg["name"], "shape": "sphere", "size": [SPHERE_RADIUS_M],
        "density": SPHERE_DENSITY_KG_M3, "friction": list(cfg["friction"]),
        "rgba": list(cfg["rgba"]), "initial_pos": list(base.object["initial_pos"]),
        "initial_quat": [1.0, 0.0, 0.0, 0.0], "fixture_name": cfg["fixture_name"],
    }
    return replace(base, raw=raw)


def build_c07_scene() -> ShadowScene:
    return build_shadow_scene(phase3c07_scene_config())


def _geom_ids(scene: ShadowScene, semantic: str) -> tuple[int, ...]:
    return tuple(
        mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in scene.collision_geoms[semantic]
    )


def _object_geom_id(scene: ShadowScene) -> int:
    return mujoco.mj_name2id(
        scene.model, mujoco.mjtObj.mjOBJ_GEOM, f"{scene.config.object['name']}_geom"
    )


def _distance_to(scene: ShadowScene, semantic: str) -> float:
    object_geom = _object_geom_id(scene)
    return float(min(
        mujoco.mj_geomDistance(scene.model, scene.data, object_geom, geom, 0.25, None)
        for geom in _geom_ids(scene, semantic)
    ))


def _set_object_palm(scene: ShadowScene, center_palm: np.ndarray) -> None:
    origin, rotation = palm_transform(scene)
    address = scene.model.jnt_qposadr[scene.object_joint_id]
    scene.data.qpos[address:address + 3] = origin + rotation @ np.asarray(center_palm)
    scene.data.qpos[address + 3:address + 7] = [1.0, 0.0, 0.0, 0.0]
    scene.data.mocap_pos[scene.fixture_mocap_id] = scene.data.qpos[address:address + 3]
    scene.data.mocap_quat[scene.fixture_mocap_id] = [1.0, 0.0, 0.0, 0.0]
    dof_address = scene.model.jnt_dofadr[scene.object_joint_id]
    scene.data.qvel[dof_address:dof_address + 6] = 0.0
    mujoco.mj_forward(scene.model, scene.data)


def _pocket_reference_geometry(scene: ShadowScene) -> dict[str, Any]:
    open_qpos = _project(scene, "open hand")
    flexed = open_qpos.copy()
    grasp = _project(scene, "grasp hard")
    for finger in ("ring", "little"):
        addresses = scene.model.jnt_qposadr[scene.joint_ids[finger]]
        flexed[addresses] = grasp[addresses]
    scene.data.qpos[:24] = open_qpos
    mujoco.mj_forward(scene.model, scene.data)
    refs = {}
    for label, body in {
        "ring_root": "rh_rfknuckle", "little_root": "rh_lfknuckle",
        "ring_proximal": "rh_rfproximal", "little_proximal": "rh_lfproximal",
        "palm_root": scene.config.hand.palm_body,
    }.items():
        body_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, body)
        refs[label] = world_to_palm(scene, scene.data.xpos[body_id]).tolist()
    ring_geoms, little_geoms = _geom_ids(scene, "ring"), _geom_ids(scene, "little")
    scene.data.qpos[:24] = flexed
    mujoco.mj_forward(scene.model, scene.data)
    gaps = []
    for ring_geom in ring_geoms:
        for little_geom in little_geoms:
            gaps.append(mujoco.mj_geomDistance(
                scene.model, scene.data, ring_geom, little_geom, 0.25, None
            ))
    refs["ring_little_minimum_opening_m"] = float(min(gaps))
    refs["open_qpos"] = open_qpos.tolist()
    refs["ring_little_flexed_qpos"] = flexed.tolist()
    return refs


def _incoming_path_clearance(scene: ShadowScene, center: np.ndarray, open_qpos: np.ndarray) -> float:
    samples = int(load_phase3c07_config()["static_map"]["incoming_path_samples"])
    # A palm-normal entry segment is only a static accessibility diagnostic;
    # dynamic transport remains curved and is measured separately.
    start = center.copy()
    start[1] = min(center[1] - 2.5 * SPHERE_RADIUS_M, -0.055)
    minimum = float("inf")
    scene.data.qpos[:24] = open_qpos
    for alpha in np.linspace(0.0, 1.0, samples):
        _set_object_palm(scene, (1.0 - alpha) * start + alpha * center)
        minimum = min(minimum, *(_distance_to(scene, finger) for finger in ("middle", "ring", "little")))
    return float(minimum)


def build_static_reachability_map() -> dict[str, Any]:
    """Outcome-independent voxel map from exact compiled geometry distances."""
    scene = build_c07_scene()
    refs = _pocket_reference_geometry(scene)
    open_qpos = np.asarray(refs.pop("open_qpos"))
    flexed_qpos = np.asarray(refs.pop("ring_little_flexed_qpos"))
    pinch_qpos = _project(scene, "two finger pinch")
    cfg = load_phase3c07_config()["static_map"]
    axes = [
        np.linspace(low, high, int(count))
        for (low, high), count in zip(
            (cfg["x_bounds_m"], cfg["y_bounds_m"], cfg["z_bounds_m"]), cfg["samples_xyz"]
        )
    ]
    steps = tuple(float(axis[1] - axis[0]) for axis in axes)
    rows = []
    feasible = []
    for x in axes[0]:
        for y in axes[1]:
            for z in axes[2]:
                center = np.asarray([x, y, z], dtype=float)
                scene.data.qpos[:24] = open_qpos
                _set_object_palm(scene, center)
                open_clearance = {
                    surface: _distance_to(scene, surface)
                    for surface in ("palm", "middle", "ring", "little")
                }
                scene.data.qpos[:24] = pinch_qpos
                mujoco.mj_forward(scene.model, scene.data)
                acquisition_reach = {
                    surface: _distance_to(scene, surface) for surface in ("thumb", "index")
                }
                scene.data.qpos[:24] = flexed_qpos
                mujoco.mj_forward(scene.model, scene.data)
                storage_reach = {
                    surface: _distance_to(scene, surface) for surface in ("ring", "little")
                }
                path_clearance = _incoming_path_clearance(scene, center, open_qpos)
                fits = bool(
                    min(open_clearance.values()) >= 0.0
                    and storage_reach["ring"] <= 0.0
                    and storage_reach["little"] <= 0.0
                    and open_clearance["palm"] <= SPHERE_RADIUS_M
                    and path_clearance >= 0.0
                )
                vectors = []
                for surface in ("palm", "ring", "little"):
                    geom = min(
                        _geom_ids(scene, surface),
                        key=lambda gid: np.linalg.norm(world_to_palm(scene, scene.data.geom_xpos[gid]) - center),
                    )
                    vector = center - world_to_palm(scene, scene.data.geom_xpos[geom])
                    vectors.append((vector / max(np.linalg.norm(vector), 1e-12)).tolist())
                row = {
                    "center_palm_m": center.tolist(), "palm_clearance_m": open_clearance["palm"],
                    "ring_link_clearance_m": open_clearance["ring"],
                    "little_link_clearance_m": open_clearance["little"],
                    "middle_link_clearance_m": open_clearance["middle"],
                    "thumb_reachability_gap_m": acquisition_reach["thumb"],
                    "index_reachability_gap_m": acquisition_reach["index"],
                    "ring_flexion_reachability_gap_m": storage_reach["ring"],
                    "little_flexion_reachability_gap_m": storage_reach["little"],
                    "incoming_path_minimum_clearance_m": path_clearance,
                    "candidate_escape_directions_palm": vectors,
                    "local_opening_width_m": refs["ring_little_minimum_opening_m"],
                    "sphere_geometrically_fits": fits,
                }
                rows.append(row)
                if fits:
                    feasible.append(tuple(float(value) for value in center))
    if not feasible:
        raise RuntimeError("static audit found no geometrically feasible 25-mm pocket voxels")
    points = np.asarray(feasible)
    half = tuple(value / 2.0 for value in steps)
    volume = PocketVolume(
        tuple((points.min(axis=0) - half).tolist()), tuple((points.max(axis=0) + half).tolist()),
        half, tuple(feasible),
        "union of outcome-independent grid voxels with nonnegative open-hand clearance, exact ring/little flexion reach, palm proximity within one sphere radius, and a nonintersecting palm-normal incoming segment",
    )
    return {
        "sphere": {"diameter_m": SPHERE_DIAMETER_M, "radius_m": SPHERE_RADIUS_M,
                   "density_kg_m3": SPHERE_DENSITY_KG_M3, "analytic_mass_kg": sphere_mass_kg(),
                   "compiled_mass_kg": float(scene.model.body_mass[scene.object_body_id])},
        "reference_geometry": refs, "grid_axes_m": [axis.tolist() for axis in axes],
        "grid_steps_m": list(steps), "candidate_count": len(rows),
        "feasible_count": len(feasible), "pocket_volume": asdict(volume), "candidates": rows,
        "constructed_before_dynamic_outcomes": True,
    }


def pocket_volume_from_audit(audit: dict[str, Any]) -> PocketVolume:
    return PocketVolume(**audit["pocket_volume"])


def acquire_c07_state(scene: ShadowScene, candidate_id: int) -> SphereAcquisitionState | None:
    state = acquire_sphere_state(scene, candidate_id)
    if state is None:
        return None
    return replace(state, state_id=f"C07_STATE_{candidate_id:05d}")


def freeze_acquisition_states(output_dir: Path | None = None) -> dict[str, Any]:
    output = output_dir or ROOT / "outputs/phase3C07/matched_states"
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_phase3c07_config()["matched_states"]
    scene = build_c07_scene()
    states = []
    attempted = 0
    for candidate_id in range(int(cfg["maximum_candidates"])):
        attempted += 1
        state = acquire_c07_state(scene, candidate_id)
        if state is not None:
            states.append(state)
        if len(states) == int(cfg["count"]):
            break
    if len(states) != int(cfg["count"]):
        raise RuntimeError(f"only {len(states)} valid 25-mm acquisition states found")
    for state in states:
        np.savez_compressed(output / f"{state.state_id}.npz", qpos=state.qpos, qvel=state.qvel, ctrl=state.ctrl)
    manifest = {
        "count": len(states), "candidates_attempted": attempted,
        "frozen_before_transport_outcomes": True,
        "state_ids": [state.state_id for state in states],
        "states": [asdict(state) | {"qpos": None, "qvel": None, "ctrl": None} for state in states],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"manifest": manifest, "states": states}


def load_acquisition_states(output_dir: Path | None = None) -> list[SphereAcquisitionState]:
    output = output_dir or ROOT / "outputs/phase3C07/matched_states"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    states = []
    for row in manifest["states"]:
        arrays = np.load(output / f"{row['state_id']}.npz")
        states.append(SphereAcquisitionState(
            **{key: value for key, value in row.items() if key not in {"qpos", "qvel", "ctrl"}},
            qpos=tuple(arrays["qpos"]), qvel=tuple(arrays["qvel"]), ctrl=tuple(arrays["ctrl"]),
        ))
    return states


def restore_acquisition_state(scene: ShadowScene, state: SphereAcquisitionState) -> None:
    mujoco.mj_resetData(scene.model, scene.data)
    scene.data.qpos[:] = state.qpos
    scene.data.qvel[:] = state.qvel
    scene.data.ctrl[:] = state.ctrl
    set_fixture(scene, False)
    mujoco.mj_forward(scene.model, scene.data)


def transport_components(start: Iterable[float], current: Iterable[float], pocket: PocketVolume) -> dict[str, float]:
    start = np.asarray(tuple(start), dtype=float)
    current = np.asarray(tuple(current), dtype=float)
    target = np.asarray(pocket.center_palm_m)
    delta = current - start
    desired = target - start
    return {
        "inward_progress_m": float(np.sign(desired[1]) * delta[1]),
        "lateral_ulnar_progress_m": float(-delta[0]),
        "pocket_distance_m": pocket.closest_distance(current),
        "ring_root_distance_m": float(np.linalg.norm(current - np.asarray([-0.011, 0.0, 0.095]))),
        "little_root_distance_m": float(np.linalg.norm(current - np.asarray([-0.033, 0.0, 0.0865]))),
    }


def _tip_geom_ids(scene: ShadowScene) -> tuple[int, int]:
    return tuple(
        mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, scene.fingertip_geoms[finger][0])
        for finger in ("thumb", "index")
    )


def _solve_grasp_waypoint(
    scene: ShadowScene, base_qpos: np.ndarray, desired_center: np.ndarray, offsets: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    joint_ids = np.r_[scene.joint_ids["thumb"], scene.joint_ids["index"]]
    addresses = scene.model.jnt_qposadr[joint_ids]
    geom_ids = _tip_geom_ids(scene)
    lower, upper = scene.model.jnt_range[joint_ids, 0], scene.model.jnt_range[joint_ids, 1]

    def residual(values: np.ndarray) -> np.ndarray:
        scene.data.qpos[:24] = base_qpos
        scene.data.qpos[addresses] = values
        mujoco.mj_forward(scene.model, scene.data)
        points = np.asarray([world_to_palm(scene, scene.data.geom_xpos[geom]) for geom in geom_ids])
        return (points - (desired_center + offsets)).ravel()

    result = least_squares(
        residual, np.clip(base_qpos[addresses], lower, upper), bounds=(lower, upper),
        ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=800,
    )
    solved = base_qpos.copy()
    solved[addresses] = result.x
    scene.data.qpos[:24] = solved
    mujoco.mj_forward(scene.model, scene.data)
    achieved = np.mean([world_to_palm(scene, scene.data.geom_xpos[geom]) for geom in geom_ids], axis=0)
    return solved, achieved, float(np.linalg.norm(result.fun))


def plan_transport(scene: ShadowScene, state: SphereAcquisitionState, pocket: PocketVolume, strategy: TransportStrategy) -> TransportPlan:
    restore_acquisition_state(scene, state)
    start = object_pose_in_palm(scene, scene.object_body_id)[0]
    goal = np.asarray(pocket.center_palm_m)
    count = int(load_phase3c07_config()["transport"]["waypoints"])
    if strategy == TransportStrategy.T0_OLD_DIRECT:
        desired = np.linspace(start, np.asarray(load_phase3c07_config()["transport"]["old_target_palm_m"]), count)
    else:
        # Curved palm-frame path: first move inward, then laterally/settle.
        inward = start.copy()
        inward[1] = goal[1]
        inward[2] = 0.5 * (start[2] + goal[2])
        first = np.linspace(start, inward, count // 2, endpoint=False)
        second = np.linspace(inward, goal, count - len(first))
        desired = np.vstack((first, second))
    straight = np.linspace(start, goal, count)
    geom_ids = _tip_geom_ids(scene)
    offsets = np.asarray([
        world_to_palm(scene, scene.data.geom_xpos[geom]) - start for geom in geom_ids
    ])
    qpos = np.asarray(state.qpos[:24], dtype=float)
    qpos_rows, achieved, residuals = [], [], []
    for waypoint in desired:
        qpos, actual, residual = _solve_grasp_waypoint(scene, qpos, waypoint, offsets)
        qpos_rows.append(tuple(float(value) for value in qpos))
        achieved.append(tuple(float(value) for value in actual))
        residuals.append(residual)
    return TransportPlan(
        strategy.value, tuple(tuple(float(v) for v in row) for row in desired), tuple(achieved),
        tuple(qpos_rows), tuple(residuals), tuple(tuple(float(v) for v in row) for row in straight),
    )


def wrist_commands(level: str) -> tuple[tuple[float, float], ...]:
    cfg = load_phase3c07_config()["wrist"]
    magnitude = float(cfg[f"{level}_deg"])
    if level == "W0":
        return ((0.0, 0.0),)
    return tuple(tuple(magnitude * float(value) for value in direction) for direction in cfg["structured_directions"])


def forearm_dof_audit(scene: ShadowScene, start_palm: np.ndarray, pocket: PocketVolume) -> dict[str, Any]:
    desired = np.asarray(pocket.center_palm_m) - np.asarray(start_palm)
    desired /= np.linalg.norm(desired)
    wrist_ids = np.asarray([
        mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in scene.config.hand.wrist_joints
    ])
    ranges = scene.model.jnt_range[wrist_ids]
    base = scene.data.qpos[:24].copy()
    candidates = []
    for wrj2 in np.linspace(*ranges[0], 41):
        for wrj1 in np.linspace(*ranges[1], 41):
            scene.data.qpos[:24] = base
            scene.data.qpos[scene.model.jnt_qposadr[wrist_ids]] = [wrj2, wrj1]
            mujoco.mj_forward(scene.model, scene.data)
            gravity = gravity_in_palm_frame(scene)
            unit = gravity / np.linalg.norm(gravity)
            candidates.append((float(np.dot(unit, desired)), wrj2, wrj1, unit.copy()))
    best = max(candidates, key=lambda row: row[0])
    missing_angle = float(np.rad2deg(np.arccos(np.clip(best[0], -1.0, 1.0))))
    lateral_unreachable = bool(abs(best[3][0] - desired[0]) > 1e-6 and max(abs(row[3][0]) for row in candidates) < 1e-6)
    return {
        "desired_transport_direction_palm": desired.tolist(),
        "best_reachable_gravity_direction_palm": best[3].tolist(),
        "best_wrist_qpos_rad": [float(best[1]), float(best[2])],
        "residual_orientation_angle_deg": missing_angle,
        "missing_lateral_gravity_component": lateral_unreachable,
        "code": "PHASE3C07_FOREARM_DOF_LIMIT" if lateral_unreachable else None,
        "stop_wrist_expansion": lateral_unreachable,
    }


def joint_boundary_events(scene: ShadowScene, step: int, stage: str) -> list[dict[str, Any]]:
    events = []
    for joint_id in range(24):
        address = scene.model.jnt_qposadr[joint_id]
        value = float(scene.data.qpos[address])
        lower, upper = scene.model.jnt_range[joint_id]
        if value <= lower or value >= upper:
            name = mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            group = next((finger for finger, ids in scene.joint_ids.items() if joint_id in ids), "wrist")
            events.append({
                "joint": name, "group": group, "step": int(step), "stage": stage,
                "qpos_rad": value, "lower_rad": float(lower), "upper_rad": float(upper),
                "signed_margin_rad": float(min(value - lower, upper - value)),
            })
    return events


def contact_geometry(scene: ShadowScene) -> dict[str, Any]:
    contacts = extract_shadow_contacts(scene)
    records = []
    normals = []
    for surface in SUPPORT_SURFACES:
        for record in contacts.records_by_surface[surface]:
            records.append({
                "surface": surface, "geom_pair": [record.geom1_name, record.geom2_name],
                "body_pair": [record.body1_name, record.body2_name],
                "position_world_m": record.position.tolist(), "normal_world": record.normal.tolist(),
                "penetration_m": float(max(0.0, -record.distance)),
                "normal_force_n": float(record.normal_force),
            })
            normals.append(record.normal)
    angles = []
    for index, first in enumerate(normals):
        for second in normals[index + 1:]:
            angles.append(float(np.rad2deg(np.arccos(np.clip(abs(np.dot(first, second)), 0.0, 1.0)))))
    rank = int(np.linalg.matrix_rank(np.asarray(normals))) if normals else 0
    acquisition = float(contacts.normal_forces[:2].sum())
    storage = float(contacts.normal_forces[2:].sum())
    total = acquisition + storage
    return {
        "records": records, "pairwise_normal_angles_deg": angles, "contact_normal_rank": rank,
        "contact_topology": [surface for index, surface in enumerate(SUPPORT_SURFACES) if contacts.contact_flags[index]],
        "load_bearing_topology": [surface for index, surface in enumerate(SUPPORT_SURFACES) if contacts.normal_forces[index] > 0.0],
        "acquisition_force_n": acquisition, "storage_force_n": storage,
        "lambda_storage": storage / total if total > 0.0 else 0.0,
        "penetration_by_surface_m": contacts.penetration_by_surface.tolist(),
        "penetration_by_surface_over_radius": (contacts.penetration_by_surface / SPHERE_RADIUS_M).tolist(),
        "maximum_penetration_m": contacts.maximum_penetration,
    }


def _sample(
    scene: ShadowScene,
    step: int,
    stage: str,
    start: np.ndarray,
    pocket: PocketVolume,
    previous_motion_speed: float | None = None,
) -> dict[str, Any]:
    center, _ = object_pose_in_palm(scene, scene.object_body_id)
    linear, angular = object_velocity(scene)
    geometry = contact_geometry(scene)
    motion_speed = float(np.linalg.norm(linear) + SPHERE_RADIUS_M * np.linalg.norm(angular))
    return {
        "step": int(step), "stage": stage, "sphere_center_palm_m": center.tolist(),
        "transport": transport_components(start, center, pocket),
        "inside_pocket": pocket.contains(center), "near_pocket": pocket.near(center, SPHERE_RADIUS_M),
        "unused_finger_clearance_m": float(min(
            _distance_to(scene, "middle"), _distance_to(scene, "ring"), _distance_to(scene, "little")
        )),
        "floor_contact": floor_contact(scene), "gravity_in_palm_mps2": gravity_in_palm_frame(scene).tolist(),
        "linear_speed_mps": float(np.linalg.norm(linear)), "angular_speed_radps": float(np.linalg.norm(angular)),
        "radius_scaled_motion_speed_mps": motion_speed,
        "motion_settling": bool(previous_motion_speed is not None and motion_speed <= previous_motion_speed + 1e-12),
        "contact_geometry": geometry, "joint_boundary_events": joint_boundary_events(scene, step, stage),
        "finite_physics": bool(np.all(np.isfinite(scene.data.qpos)) and np.all(np.isfinite(scene.data.qvel))),
    }


def _candidate_cage(sample: dict[str, Any]) -> bool:
    geometry = sample["contact_geometry"]
    storage_contact = any(surface in {"middle", "ring", "little", "palm"} for surface in geometry["load_bearing_topology"])
    return bool(
        sample["inside_pocket"] and storage_contact and not sample["floor_contact"]
        and geometry["contact_normal_rank"] >= 2 and sample["motion_settling"] and sample["finite_physics"]
    )


def _hold_valid(sample: dict[str, Any]) -> bool:
    geometry = sample["contact_geometry"]
    storage_contact = any(
        surface in {"middle", "ring", "little", "palm"}
        for surface in geometry["load_bearing_topology"]
    )
    return bool(
        (sample["inside_pocket"] or sample["near_pocket"])
        and storage_contact and not sample["floor_contact"]
        and geometry["contact_normal_rank"] >= 2
        and sample["motion_settling"] and sample["finite_physics"]
    )


def run_transport_trial(
    scene: ShadowScene,
    state: SphereAcquisitionState,
    pocket: PocketVolume,
    strategy: TransportStrategy,
    *,
    wrist_delta_deg: tuple[float, float] = (0.0, 0.0),
    preshape: PreshapeCondition | None = None,
    transport_plan: TransportPlan | None = None,
    frame_callback: Any | None = None,
) -> dict[str, Any]:
    restore_acquisition_state(scene, state)
    cfg = load_phase3c07_config()
    plan = transport_plan or plan_transport(scene, state, pocket, strategy)
    restore_acquisition_state(scene, state)
    start = object_pose_in_palm(scene, scene.object_body_id)[0]
    initial_distance = pocket.closest_distance(start)
    initial_components = transport_components(start, start, pocket)
    open_target = actuator_target_from_qpos(scene, _project(scene, "open hand"))
    grasp_target = actuator_target_from_qpos(scene, _project(scene, "grasp hard"))
    wrist_ids = scene.actuator_ids["wrist"]
    wrist_target = scene.data.ctrl.copy()
    wrist_target[wrist_ids] = np.clip(
        wrist_target[wrist_ids] + np.deg2rad(wrist_delta_deg),
        scene.model.actuator_ctrlrange[wrist_ids, 0], scene.model.actuator_ctrlrange[wrist_ids, 1],
    )
    steps = int(cfg["transport"]["steps"])
    samples = []
    first_entry = None
    residence = 0
    maximum_residence = 0
    preshape_step = None
    latches: dict[str, np.ndarray] = {}
    previous_motion_speed = None
    first_candidate_cage = None
    for step in range(steps):
        waypoint_index = min(len(plan.qpos_waypoints) - 1, step * len(plan.qpos_waypoints) // steps)
        target = actuator_target_from_qpos(scene, np.asarray(plan.qpos_waypoints[waypoint_index]))
        for finger in ("thumb", "index"):
            ids = scene.actuator_ids[finger]
            scene.data.ctrl[ids] += np.clip(
                target[ids] - scene.data.ctrl[ids],
                -float(cfg["transport"]["command_increment"]), float(cfg["transport"]["command_increment"]),
            )
        if strategy == TransportStrategy.T2_WRIST_ASSISTED:
            scene.data.ctrl[wrist_ids] += np.clip(
                wrist_target[wrist_ids] - scene.data.ctrl[wrist_ids],
                -np.deg2rad(float(cfg["transport"]["wrist_increment_deg"])),
                np.deg2rad(float(cfg["transport"]["wrist_increment_deg"])),
            )
        center = object_pose_in_palm(scene, scene.object_body_id)[0]
        near = pocket.near(center, SPHERE_RADIUS_M)
        inside = pocket.contains(center)
        sweep_clear = min(_distance_to(scene, "ring"), _distance_to(scene, "little"))
        trigger = bool(preshape is not None and preshape_gate(
            preshape, inside_pocket=inside, near_pocket=near, sweep_clearance_m=sweep_clear
        ))
        if trigger and preshape_step is None:
            preshape_step = step
        for finger in ("ring", "little"):
            ids = scene.actuator_ids[finger]
            if preshape_step is None:
                scene.data.ctrl[ids] = open_target[ids]
            else:
                contacts = extract_shadow_contacts(scene)
                surface = SUPPORT_SURFACES.index(finger)
                if finger not in latches and contacts.normal_forces[surface] >= float(cfg["preshape"]["contact_force_n"]):
                    latches[finger] = scene.data.ctrl[ids].copy()
                destination = latches.get(finger, grasp_target[ids])
                scene.data.ctrl[ids] += np.clip(
                    destination - scene.data.ctrl[ids],
                    -float(cfg["preshape"]["closure_increment"]), float(cfg["preshape"]["closure_increment"]),
                )
        mujoco.mj_step(scene.model, scene.data)
        stage = "PRESHAPE" if preshape_step is not None else "TRANSPORT"
        sample = _sample(scene, step, stage, start, pocket, previous_motion_speed)
        samples.append(sample)
        if frame_callback is not None:
            frame_callback(scene, step, stage)
        previous_motion_speed = sample["radius_scaled_motion_speed_mps"]
        if sample["inside_pocket"]:
            first_entry = step if first_entry is None else first_entry
            residence += 1
            maximum_residence = max(maximum_residence, residence)
        else:
            residence = 0
        if _candidate_cage(sample) and first_candidate_cage is None:
            first_candidate_cage = step
        if not sample["finite_physics"]:
            break
    closest = min(sample["transport"]["pocket_distance_m"] for sample in samples)
    end_center = np.asarray(samples[-1]["sphere_center_palm_m"])
    exit = end_center - np.asarray(pocket.center_palm_m)
    exit /= max(np.linalg.norm(exit), 1e-12)
    # Cage persistence uses the protocol-provided first 10-step hold checkpoint,
    # not a newly tuned publication threshold.
    candidate_runs = 0
    cage_formed = False
    for sample in samples:
        candidate_runs = candidate_runs + 1 if _candidate_cage(sample) else 0
        cage_formed |= candidate_runs >= 10
    hold = {str(step): False for step in cfg["cage_hold"]["checkpoints"]}
    hold_samples = []
    if cage_formed:
        retained = True
        checkpoints = set(int(value) for value in cfg["cage_hold"]["checkpoints"])
        for post in range(1, max(checkpoints) + 1):
            mujoco.mj_step(scene.model, scene.data)
            sample = _sample(scene, steps + post, "CAGE_HOLD", start, pocket, previous_motion_speed)
            previous_motion_speed = sample["radius_scaled_motion_speed_mps"]
            if frame_callback is not None:
                frame_callback(scene, steps + post, "CAGE_HOLD")
            retained &= _hold_valid(sample)
            if post in checkpoints:
                hold[str(post)] = retained
                hold_samples.append(sample)
    all_samples = samples + hold_samples
    maximum_penetration = np.max([
        sample["contact_geometry"]["penetration_by_surface_m"] for sample in all_samples
    ], axis=0)
    boundary_events = [event for sample in all_samples for event in sample["joint_boundary_events"]]
    minimum_unused_clearance = min(sample["unused_finger_clearance_m"] for sample in samples)
    final_components = samples[-1]["transport"]
    lateral_improved = final_components["lateral_ulnar_progress_m"] > initial_components["lateral_ulnar_progress_m"]
    inward_improved = final_components["inward_progress_m"] > initial_components["inward_progress_m"]
    failures = []
    if minimum_unused_clearance < 0.0 and first_entry is None:
        failures.append("TRANSFER_CORRIDOR_BLOCKED")
    if first_entry is None:
        failures.append("POCKET_NOT_REACHED")
    elif not cage_formed:
        failures.append("NO_LOAD_BEARING_CAGE")
    if not lateral_improved:
        failures.append("POCKET_LATERAL_TRANSPORT_FAILED")
    if not inward_improved:
        failures.append("POCKET_INWARD_TRANSPORT_FAILED")
    if not any("ring" in sample["contact_geometry"]["contact_topology"] for sample in samples):
        failures.append("RING_CONTACT_NOT_ESTABLISHED")
    if not any("little" in sample["contact_geometry"]["contact_topology"] for sample in samples):
        failures.append("LITTLE_CONTACT_NOT_ESTABLISHED")
    if not any("palm" in sample["contact_geometry"]["contact_topology"] for sample in samples):
        failures.append("PALM_ROOT_CONTACT_NOT_ESTABLISHED")
    transport_boundary_limited = bool(
        any(event["group"] in {"thumb", "index"} for event in boundary_events)
        and (not lateral_improved or not inward_improved)
    )
    if transport_boundary_limited:
        failures.append("JOINT_BOUNDARY_LIMITED_TRANSPORT")
    if cage_formed and not hold.get("1000", False):
        failures.append("CAGE_HOLD_FAILED")
    if first_entry is not None and not samples[-1]["inside_pocket"]:
        final_linear = samples[-1]["linear_speed_mps"]
        final_rolling = SPHERE_RADIUS_M * samples[-1]["angular_speed_radps"]
        failures.append("SPHERE_ROLLED_OUT" if final_rolling > final_linear else "SPHERE_SLID_OUT")
    final_topology = samples[-1]["contact_geometry"]["contact_topology"]
    return {
        "state_id": state.state_id, "strategy": strategy.value,
        "wrist_delta_command_deg": list(wrist_delta_deg),
        "actual_wrist_motion_deg": np.rad2deg(scene.data.qpos[:2] - np.asarray(state.qpos[:2])).tolist(),
        "preshape": None if preshape is None else preshape.value,
        "plan": asdict(plan), "first_pocket_entry_step": first_entry,
        "pocket_residence_steps": int(maximum_residence), "closest_approach_m": float(closest),
        "exit_direction_palm": exit.tolist(), "lateral_transport_success": bool(lateral_improved),
        "inward_transport_success": bool(inward_improved), "preshape_trigger_step": preshape_step,
        "first_candidate_cage_step": first_candidate_cage, "cage_formed": bool(cage_formed),
        "hold_survival": hold, "maximum_penetration_by_surface_m": maximum_penetration.tolist(),
        "maximum_penetration_by_surface_over_radius": (maximum_penetration / SPHERE_RADIUS_M).tolist(),
        "penetration_acceptability": "TODO(PI): no Phase 3C-0.7 threshold is frozen",
        "joint_boundary_events": boundary_events, "failures": failures,
        "joint_boundary_limited_transport": transport_boundary_limited,
        "minimum_unused_finger_clearance_m": float(minimum_unused_clearance),
        "corridor_clear": bool(minimum_unused_clearance >= 0.0),
        "thumb_contact_retained": "thumb" in final_topology,
        "index_contact_retained": "index" in final_topology,
        "fixture_active": bool(scene.data.eq_active[scene.fixture_eq_id]),
        "thumb_release_performed": False, "index_release_performed": False,
        "samples": all_samples,
    }


def contract() -> dict[str, Any]:
    scene = build_c07_scene()
    return {
        "sphere_diameter_m": 2.0 * scene.model.geom_size[_object_geom_id(scene), 0],
        "compiled_mass_kg": float(scene.model.body_mass[scene.object_body_id]),
        "world_gravity": scene.model.opt.gravity.tolist(), "thumb_release_performed": False,
        "index_release_performed": False,
        "object_B_instantiated": mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "object_B") >= 0,
        "rl_training_performed": False, "reward_defined": False, "scalar_J_defined": False,
        "compliant_skin_added": False, "official_MJCF_modified": False,
    }
