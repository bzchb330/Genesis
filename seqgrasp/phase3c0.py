"""Phase 3C-0 geometry and scripted-control interfaces.

This module deliberately contains no reward, policy, or training code.  It
exposes raw geometry and contact topology so later scientific criteria can be
chosen without baking those decisions into a scalar objective.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml

from .config import ROOT
from .phase3.config import FINGERS, Phase3Config, load_phase3_config
from .phase3.control import actuator_target_from_qpos
from .phase3.env import ObservationComponent, load_keyframe_qpos
from .phase3.model import (
    ShadowScene,
    _name_runtime_collision_geoms,
    _vec,
)
from .phase3b1a import project_feasible_hand_qpos


class Phase3CFingerRole(StrEnum):
    FREE = "FREE"
    PROBING = "PROBING"
    ACQUIRING = "ACQUIRING"
    TRANSFERRING = "TRANSFERRING"
    CLEARING_CORRIDOR = "CLEARING_CORRIDOR"
    SECURING_STORAGE = "SECURING_STORAGE"
    RELAXING_APERTURE = "RELAXING_APERTURE"
    RESECURING = "RESECURING"
    RELEASING = "RELEASING"


class Phase3CState(StrEnum):
    OPEN_HAND = "OPEN_HAND"
    MINIMAL_ACQUIRE_A = "MINIMAL_ACQUIRE_A"
    CLEAR_TRANSFER_CORRIDOR = "CLEAR_TRANSFER_CORRIDOR"
    TRANSFER_A_TO_PALM = "TRANSFER_A_TO_PALM"
    A_IN_STORAGE_REGION = "A_IN_STORAGE_REGION"
    SECURE_A = "SECURE_A"
    RELEASE_ACQUISITION_DIGITS = "RELEASE_ACQUISITION_DIGITS"
    ACQUISITION_RESOURCES_RECOVERED = "ACQUISITION_RESOURCES_RECOVERED"
    REORIENT_HAND_IF_USEFUL = "REORIENT_HAND_IF_USEFUL"
    MINIMAL_ACQUIRE_B = "MINIMAL_ACQUIRE_B"
    FIND_RETENTION_PRESERVING_INSERTION_CORRIDOR = "FIND_RETENTION_PRESERVING_INSERTION_CORRIDOR"
    RELAX_STORAGE_APERTURE_IF_NEEDED = "RELAX_STORAGE_APERTURE_IF_NEEDED"
    TRANSFER_B_INTO_STORAGE = "TRANSFER_B_INTO_STORAGE"
    RESECURE_A_AND_B = "RESECURE_A_AND_B"
    RESOURCES_RECOVERED_AGAIN = "RESOURCES_RECOVERED_AGAIN"


class Phase3CFailure(StrEnum):
    TRANSFER_CORRIDOR_BLOCKED = "TRANSFER_CORRIDOR_BLOCKED"
    UNUSED_FINGER_OBSTRUCTION = "UNUSED_FINGER_OBSTRUCTION"
    A_NOT_SECURED_BEFORE_RESOURCE_RELEASE = "A_NOT_SECURED_BEFORE_RESOURCE_RELEASE"
    A_LOST_DURING_WRIST_REORIENTATION = "A_LOST_DURING_WRIST_REORIENTATION"
    NO_FEASIBLE_INSERTION_CORRIDOR = "NO_FEASIBLE_INSERTION_CORRIDOR"
    APERTURE_TOO_SMALL = "APERTURE_TOO_SMALL"
    APERTURE_RELAXATION_LOST_A = "APERTURE_RELAXATION_LOST_A"
    B_COLLISION_WITH_A = "B_COLLISION_WITH_A"
    B_COLLISION_WITH_STORAGE_FINGERS = "B_COLLISION_WITH_STORAGE_FINGERS"
    B_INSERTION_FAILED = "B_INSERTION_FAILED"
    RESECURE_FAILED = "RESECURE_FAILED"
    A_LOST_DURING_B_INSERTION = "A_LOST_DURING_B_INSERTION"
    B_LOST_AFTER_INSERTION = "B_LOST_AFTER_INSERTION"
    BOTH_LOST = "BOTH_LOST"
    MULTI_OBJECT_STORAGE_SUCCESS = "MULTI_OBJECT_STORAGE_SUCCESS"
    OTHER = "OTHER"


@dataclass
class Phase3CRoles:
    state: Phase3CState = Phase3CState.OPEN_HAND
    fingers: dict[str, Phase3CFingerRole] = field(
        default_factory=lambda: {finger: Phase3CFingerRole.FREE for finger in FINGERS}
    )
    history: list[dict[str, Any]] = field(default_factory=list)

    def transition(
        self,
        state: Phase3CState,
        assignments: dict[str, Phase3CFingerRole],
        *,
        step: int,
        reason: str,
    ) -> None:
        unknown = set(assignments) - set(FINGERS)
        if unknown:
            raise ValueError(f"unknown fingers: {sorted(unknown)}")
        self.state = state
        self.fingers.update(assignments)
        self.history.append(
            {"step": int(step), "state": state.value, "reason": reason,
             "roles": {name: role.value for name, role in self.fingers.items()}}
        )

    def begin_minimal_acquisition(self, object_name: str, step: int) -> None:
        state = Phase3CState.MINIMAL_ACQUIRE_A if object_name == "A" else Phase3CState.MINIMAL_ACQUIRE_B
        self.transition(
            state,
            {"thumb": Phase3CFingerRole.ACQUIRING, "index": Phase3CFingerRole.ACQUIRING,
             "middle": Phase3CFingerRole.CLEARING_CORRIDOR,
             "ring": Phase3CFingerRole.CLEARING_CORRIDOR,
             "little": Phase3CFingerRole.CLEARING_CORRIDOR},
            step=step,
            reason="minimum useful acquisition set; unused digits preserve corridor",
        )

    def begin_transfer(self, step: int) -> None:
        self.transition(
            Phase3CState.TRANSFER_A_TO_PALM,
            {"thumb": Phase3CFingerRole.TRANSFERRING, "index": Phase3CFingerRole.TRANSFERRING},
            step=step,
            reason="fixture released; object motion is dynamic",
        )

    def storage_entry(self, securing: Iterable[str], step: int) -> None:
        assignments = {finger: Phase3CFingerRole.SECURING_STORAGE for finger in securing}
        self.transition(
            Phase3CState.A_IN_STORAGE_REGION,
            assignments,
            step=step,
            reason="object extent intersects configured palm-frame storage volume",
        )

    def relax_aperture(self, securing: Iterable[str], step: int) -> None:
        self.transition(
            Phase3CState.RELAX_STORAGE_APERTURE_IF_NEEDED,
            {finger: Phase3CFingerRole.RELAXING_APERTURE for finger in securing},
            step=step,
            reason="controlled storage aperture relaxation",
        )

    def resecure(self, securing: Iterable[str], step: int) -> None:
        self.transition(
            Phase3CState.RESECURE_A_AND_B,
            {finger: Phase3CFingerRole.RESECURING for finger in securing},
            step=step,
            reason="close after insertion attempt",
        )


@dataclass(frozen=True)
class StorageRegion:
    center_palm_m: tuple[float, float, float]
    half_extents_m: tuple[float, float, float]

    def measure(self, center_palm: np.ndarray, half_extents: np.ndarray) -> dict[str, Any]:
        center = np.asarray(center_palm, dtype=float)
        extent = np.asarray(half_extents, dtype=float)
        low = center - extent
        high = center + extent
        region_low = np.asarray(self.center_palm_m) - np.asarray(self.half_extents_m)
        region_high = np.asarray(self.center_palm_m) + np.asarray(self.half_extents_m)
        overlap = np.maximum(0.0, np.minimum(high, region_high) - np.maximum(low, region_low))
        volume = float(np.prod(2.0 * extent))
        overlap_volume = float(np.prod(overlap))
        signed_clearance = np.minimum(high - region_low, region_high - low)
        return {
            "object_center_palm_m": center.tolist(),
            "object_half_extents_m": extent.tolist(),
            "overlap_extents_m": overlap.tolist(),
            "occupancy_fraction": overlap_volume / volume if volume > 0 else 0.0,
            "center_inside": bool(np.all(center >= region_low) and np.all(center <= region_high)),
            "extent_fully_inside": bool(np.all(low >= region_low) and np.all(high <= region_high)),
            "boundary_clearance_m": signed_clearance.tolist(),
        }


@dataclass(frozen=True)
class Phase3CMultiScene:
    model: mujoco.MjModel
    data: mujoco.MjData
    config: Phase3Config
    collision_geoms: dict[str, tuple[str, ...]]
    fingertip_geoms: dict[str, tuple[str, ...]]
    actuator_ids: dict[str, np.ndarray]
    joint_ids: dict[str, np.ndarray]
    object_body_ids: dict[str, int]
    object_joint_ids: dict[str, int]
    object_geom_ids: dict[str, int]
    fixture_eq_ids: dict[str, int]
    fixture_mocap_ids: dict[str, int]


def load_phase3c0_config(path: Path | None = None) -> dict[str, Any]:
    source = path or ROOT / "configs/phase3C0_open_corridor.yaml"
    with source.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def open_hand_configuration(scene: ShadowScene | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    scene = scene or _single_scene()
    result = project_feasible_hand_qpos(load_keyframe_qpos("open hand"), scene)
    return np.asarray(result.projected_qpos), asdict(result)


def _single_scene() -> ShadowScene:
    from .phase3.model import build_shadow_scene
    return build_shadow_scene()


def palm_transform(scene: ShadowScene | Phase3CMultiScene) -> tuple[np.ndarray, np.ndarray]:
    palm = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, scene.config.hand.palm_body)
    return scene.data.xpos[palm].copy(), scene.data.xmat[palm].reshape(3, 3).copy()


def world_to_palm(scene: ShadowScene | Phase3CMultiScene, point: np.ndarray) -> np.ndarray:
    origin, rotation = palm_transform(scene)
    return rotation.T @ (np.asarray(point) - origin)


def gravity_in_palm_frame(scene: ShadowScene | Phase3CMultiScene) -> np.ndarray:
    _, rotation = palm_transform(scene)
    return rotation.T @ scene.model.opt.gravity


def object_pose_in_palm(
    scene: ShadowScene | Phase3CMultiScene, object_body_id: int
) -> tuple[np.ndarray, np.ndarray]:
    position = world_to_palm(scene, scene.data.xpos[object_body_id])
    _, palm_rotation = palm_transform(scene)
    object_rotation = scene.data.xmat[object_body_id].reshape(3, 3)
    return position, palm_rotation.T @ object_rotation


def configured_storage_region(config: dict[str, Any] | None = None) -> StorageRegion:
    raw = (config or load_phase3c0_config())["storage_region"]
    return StorageRegion(tuple(raw["center_m"]), tuple(raw["half_extents_m"]))


def storage_measurement(
    scene: ShadowScene | Phase3CMultiScene,
    object_body_id: int,
    object_half_extents: np.ndarray,
    region: StorageRegion | None = None,
) -> dict[str, Any]:
    position, rotation = object_pose_in_palm(scene, object_body_id)
    # AABB extent of the oriented ellipsoid in the palm frame.
    extent = np.abs(rotation) @ np.asarray(object_half_extents, dtype=float)
    result = (region or configured_storage_region()).measure(position, extent)
    palm_geoms = [mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                  for name in scene.collision_geoms["palm"]]
    result.update({
        "orientation_matrix_palm": rotation.tolist(),
        "distance_to_palm_support_surfaces_m": float(min(
            np.linalg.norm(scene.data.xpos[object_body_id] - scene.data.geom_xpos[gid])
            - scene.model.geom_rbound[gid] for gid in palm_geoms
        )),
    })
    return result


def _semantic_for_geom(scene: ShadowScene | Phase3CMultiScene, geom_id: int) -> str:
    name = mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
    for semantic, names in scene.collision_geoms.items():
        if name in names:
            return semantic
    return name


def transfer_corridor(
    scene: ShadowScene | Phase3CMultiScene,
    start_world: np.ndarray,
    end_world: np.ndarray,
    *,
    object_radius_m: float,
    excluded_surfaces: Iterable[str] = ("thumb", "index"),
    samples: int = 41,
    stored_object_geoms: Iterable[int] = (),
) -> dict[str, Any]:
    """Conservative swept-sphere corridor against compiled collision geoms."""
    alpha = np.linspace(0.0, 1.0, int(samples))[:, None]
    path = (1.0 - alpha) * np.asarray(start_world) + alpha * np.asarray(end_world)
    excluded = set(excluded_surfaces)
    candidates: list[tuple[int, str]] = []
    for semantic, names in scene.collision_geoms.items():
        if semantic in excluded or semantic == "palm":
            continue
        candidates.extend(
            (mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, name), semantic)
            for name in names
        )
    candidates.extend((int(gid), f"stored_object:{int(gid)}") for gid in stored_object_geoms)
    clearances = np.full(len(path), np.inf)
    nearest = [None] * len(path)
    for geom_id, semantic in candidates:
        values = np.linalg.norm(path - scene.data.geom_xpos[geom_id], axis=1)
        values -= float(scene.model.geom_rbound[geom_id]) + float(object_radius_m)
        improve = values < clearances
        clearances[improve] = values[improve]
        for index in np.flatnonzero(improve):
            nearest[int(index)] = semantic
    if not candidates:
        clearances[:] = np.inf
    bottleneck = int(np.argmin(clearances))
    obstructing = sorted({nearest[i] for i in np.flatnonzero(clearances < 0.0) if nearest[i]})
    return {
        "path_world_m": path.tolist(),
        "object_orientation_path": "held constant; raw orientation supplied by caller",
        "clearance_m": clearances.tolist(),
        "minimum_clearance_m": float(clearances[bottleneck]),
        "collision_free_fraction": float(np.mean(clearances >= 0.0)),
        "bottleneck_fraction": float(alpha[bottleneck, 0]),
        "bottleneck_world_m": path[bottleneck].tolist(),
        "obstructing_links": obstructing,
        "nearest_link_by_sample": nearest,
    }


def storage_aperture(
    scene: ShadowScene | Phase3CMultiScene,
    securing_fingers: Iterable[str] = ("middle", "ring", "little"),
) -> dict[str, Any]:
    """Raw, palm-frame opening geometry inferred from current support layout."""
    points: list[np.ndarray] = []
    names: list[str] = []
    for finger in securing_fingers:
        body_id = mujoco.mj_name2id(
            scene.model, mujoco.mjtObj.mjOBJ_BODY, scene.config.hand.fingertip_bodies[finger]
        )
        points.append(world_to_palm(scene, scene.data.xpos[body_id]))
        names.append(finger)
    palm_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, scene.config.hand.palm_body)
    points.append(world_to_palm(scene, scene.data.xpos[palm_id]))
    names.append("palm")
    cloud = np.asarray(points)
    centroid = cloud.mean(axis=0)
    centered = cloud - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=True)
    normal = vh[-1]
    gravity = gravity_in_palm_frame(scene)
    # Resolve sign from gravity only; no world-axis insertion direction is imposed.
    if np.dot(normal, gravity) > 0:
        normal = -normal
    axis_u, axis_v = vh[0], vh[1]
    u = centered @ axis_u
    v = centered @ axis_v
    pair_distances = [np.linalg.norm(a - b) for i, a in enumerate(cloud) for b in cloud[i + 1:]]
    width = float(np.ptp(u))
    height = float(np.ptp(v))
    return {
        "support_nodes": names,
        "centroid_palm_m": centroid.tolist(),
        "normal_palm": normal.tolist(),
        "basis_u_palm": axis_u.tolist(),
        "basis_v_palm": axis_v.tolist(),
        "effective_width_m": width,
        "effective_height_m": height,
        "minimum_node_clearance_m": float(min(pair_distances, default=0.0)),
        "available_insertion_depth_m": float(max(width, height)),
    }


def multi_object_support_graph(scene: Phase3CMultiScene) -> dict[str, Any]:
    edges: list[dict[str, Any]] = []
    object_by_geom = {geom_id: name for name, geom_id in scene.object_geom_ids.items()}
    force = np.zeros(6, dtype=float)
    for index in range(scene.data.ncon):
        contact = scene.data.contact[index]
        object_name = object_by_geom.get(int(contact.geom1), object_by_geom.get(int(contact.geom2)))
        if object_name is None:
            continue
        hand_geom = int(contact.geom2) if int(contact.geom1) in object_by_geom else int(contact.geom1)
        support = _semantic_for_geom(scene, hand_geom)
        if support not in (*FINGERS, "palm"):
            continue
        mujoco.mj_contactForce(scene.model, scene.data, index, force)
        edges.append({
            "support": support,
            "object": object_name,
            "normal_force_n": float(abs(force[0])),
            "tangential_force_n": float(np.linalg.norm(force[1:3])),
            "penetration_m": float(max(0.0, -contact.dist)),
        })
    return {"hand_nodes": [*FINGERS, "palm"], "object_nodes": ["A", "B"], "edges": edges}


def build_phase3c_multiscene(config: Phase3Config | None = None) -> Phase3CMultiScene:
    """Compile the unchanged official hand with two free diagnostic objects."""
    cfg = config or load_phase3_config()
    model_path = ROOT / cfg.hand.model_path
    root = ET.parse(model_path).getroot()
    world = root.find("worldbody")
    if world is None:
        raise ValueError("Shadow Hand MJCF has no worldbody")
    forearm = world.find(f".//body[@name='{cfg.hand.forearm_body}']")
    if forearm is None:
        raise ValueError("configured Shadow forearm body is missing")
    forearm.set("pos", _vec(cfg.hand.mount_pos))
    forearm.set("quat", _vec(cfg.hand.mount_quat))
    collision, fingertips = _name_runtime_collision_geoms(root, cfg)
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", str(cfg.raw["timestep"]))
    option.set("gravity", "0 0 -9.81")
    floor = cfg.raw["floor"]
    ET.SubElement(world, "geom", name=floor["name"], type="plane", size="1 1 0.05",
                  pos=f"0 0 {floor['z']}", rgba="0.35 0.35 0.38 1")
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    base = cfg.object
    starts = {"A": np.asarray(base["initial_pos"]), "B": np.asarray(base["initial_pos"]) + [0.0, 0.09, 0.0]}
    colors = {"A": base["rgba"], "B": [0.45, 0.55, 0.90, 1.0]}
    for label in ("A", "B"):
        body_name = f"phase3c_object_{label}"
        body = ET.SubElement(world, "body", name=body_name, pos=_vec(starts[label]), quat=_vec(base["initial_quat"]))
        ET.SubElement(body, "freejoint", name=f"{body_name}_free")
        ET.SubElement(body, "geom", name=f"{body_name}_geom", type=base["shape"], size=_vec(base["size"]),
                      friction=_vec(base["friction"]), rgba=_vec(colors[label]), condim="6", priority="1")
        anchor = f"phase3c_fixture_{label}_anchor"
        ET.SubElement(world, "body", name=anchor, mocap="true", pos=_vec(starts[label]), quat=_vec(base["initial_quat"]))
        ET.SubElement(equality, "weld", name=f"phase3c_fixture_{label}", body1=body_name,
                      body2=anchor, relpose="0 0 0 1 0 0 0")
    assets = {str(path.relative_to(model_path.parent)).replace("\\", "/"): path.read_bytes()
              for path in (model_path.parent / "assets").rglob("*") if path.is_file()}
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), assets)
    data = mujoco.MjData(model)
    actuator_ids = {group: np.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                                      for name in names], dtype=int)
                    for group, names in cfg.hand.actuator_groups.items()}
    joint_ids = {finger: np.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                                     for name in names], dtype=int)
                 for finger, names in cfg.hand.finger_joints.items()}
    body_ids = {label: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"phase3c_object_{label}")
                for label in ("A", "B")}
    joint_object_ids = {label: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"phase3c_object_{label}_free")
                        for label in ("A", "B")}
    geom_ids = {label: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"phase3c_object_{label}_geom")
                for label in ("A", "B")}
    eq_ids = {label: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"phase3c_fixture_{label}")
              for label in ("A", "B")}
    mocap_ids = {label: int(model.body_mocapid[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, f"phase3c_fixture_{label}_anchor")]) for label in ("A", "B")}
    return Phase3CMultiScene(model, data, cfg, collision, fingertips, actuator_ids, joint_ids,
                            body_ids, joint_object_ids, geom_ids, eq_ids, mocap_ids)


def set_phase3c_object_pose(
    scene: Phase3CMultiScene,
    label: str,
    position: Iterable[float],
    quaternion: Iterable[float] = (1.0, 0.0, 0.0, 0.0),
) -> None:
    if not bool(scene.data.eq_active[scene.fixture_eq_ids[label]]):
        raise RuntimeError(f"cannot set object {label} pose after its fixture has been released")
    address = scene.model.jnt_qposadr[scene.object_joint_ids[label]]
    dof = scene.model.jnt_dofadr[scene.object_joint_ids[label]]
    scene.data.qpos[address:address + 3] = position
    scene.data.qpos[address + 3:address + 7] = quaternion
    scene.data.qvel[dof:dof + 6] = 0.0
    scene.data.mocap_pos[scene.fixture_mocap_ids[label]] = position
    scene.data.mocap_quat[scene.fixture_mocap_ids[label]] = quaternion
    mujoco.mj_forward(scene.model, scene.data)


def release_phase3c_fixture(scene: Phase3CMultiScene, label: str) -> None:
    scene.data.eq_active[scene.fixture_eq_ids[label]] = 0


def phase3c_observation_contract(action_dimension: int = 26) -> tuple[ObservationComponent, ...]:
    roles = len(FINGERS) * len(Phase3CFingerRole)
    return (
        ObservationComponent("joint_positions", 24, True),
        ObservationComponent("joint_velocities", 24, True),
        ObservationComponent("wrist_state", 4, True),
        ObservationComponent("fingertip_contacts", 10, True),
        ObservationComponent("palm_contacts", 2, True),
        ObservationComponent("support_force_estimates", 12, True),
        ObservationComponent("stored_object_contact_history", 12, True),
        ObservationComponent("acquisition_object_contact_state", 6, True),
        ObservationComponent("finger_roles_one_hot", roles, True),
        ObservationComponent("previous_action", action_dimension, True),
        ObservationComponent("objects_pose_in_palm", 14, False),
        ObservationComponent("objects_velocity", 12, False),
        ObservationComponent("storage_aperture_geometry", 15, False),
        ObservationComponent("insertion_corridor_geometry", 8, False),
        ObservationComponent("gravity_in_palm_frame", 3, False),
        ObservationComponent("contact_graph", 24, False),
        ObservationComponent("exact_collision_clearance", 1, False),
    )


def phase3c_action_contract(scene: ShadowScene | Phase3CMultiScene) -> dict[str, Any]:
    return {
        "finger_target_increments": int(scene.model.nu - len(scene.actuator_ids["wrist"])),
        "wrist_target_increments": int(len(scene.actuator_ids["wrist"])),
        "semantic_group_stiffness": ["wrist", *FINGERS],
        "world_gravity_controlled": False,
        "reward_weights_defined": False,
    }
