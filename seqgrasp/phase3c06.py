"""Phase 3C-0.6 rigid-contact sphere and palmodigital-pocket diagnostics.

This module defines geometry, deterministic scripted probes, and raw physical
measurements. It does not define a reward, learned policy, scalar objective,
second object, or a new scientific acceptance threshold.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import StrEnum
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml

from .config import ROOT
from .phase3.config import SUPPORT_SURFACES, Phase3Config, load_phase3_config
from .phase3.contacts import extract_shadow_contacts, object_velocity
from .phase3.control import ContactAwareCloser, actuator_target_from_qpos
from .phase3.env import load_keyframe_qpos
from .phase3.model import ShadowScene, build_shadow_scene, set_fixture, set_object_pose
from .phase3b1a import project_feasible_hand_qpos
from .phase3c0 import StorageRegion, gravity_in_palm_frame, object_pose_in_palm, world_to_palm
from .phase3c05 import released_finger_available_motion


NON_THUMB_FINGERS = ("index", "middle", "ring", "little")
POCKET_NAMES = ("old_palm_center", "middle_ring", "ring_little", "ulnar_palmodigital")
PRESHAPE_CONDITIONS = ("NO_PRESHAPE", "PRESHAPE")
FAILURE_TAXONOMY = (
    "ACQUISITION_FAILED", "TRANSFER_CORRIDOR_BLOCKED", "PRESHAPE_TOO_EARLY",
    "PRESHAPE_TOO_LATE", "POCKET_NOT_REACHED", "POCKET_GEOMETRY_MISALIGNED",
    "NO_STORAGE_FINGER_CONTACT", "NO_LOAD_BEARING_SUPPORT", "SPHERE_ROLLED_OUT",
    "SPHERE_SLID_OUT", "WRIST_DIRECTION_UNFAVORABLE", "EXCESSIVE_PENETRATION",
    "JOINT_BOUNDARY_LIMIT", "WRIST_DOF_LIMIT", "LOSS_DURING_THUMB_RELEASE",
    "LOSS_AFTER_THUMB_RELEASE", "OTHER",
)


class SphereOutcome(StrEnum):
    SPHERE_ACQUIRED = "SPHERE_ACQUIRED"
    CORRIDOR_CLEARED = "CORRIDOR_CLEARED"
    PRESHAPE_STARTED = "PRESHAPE_STARTED"
    POCKET_ENTRY = "POCKET_ENTRY"
    PALM_OR_ROOT_CONTACT = "PALM_OR_ROOT_CONTACT"
    RING_CONTACT = "RING_CONTACT"
    LITTLE_CONTACT = "LITTLE_CONTACT"
    ALTERNATE_SUPPORT = "ALTERNATE_SUPPORT"
    THUMB_RELEASED = "THUMB_RELEASED"
    THUMB_RECOVERED = "THUMB_RECOVERED"
    INDEX_RELEASED = "INDEX_RELEASED"
    INDEX_RECOVERED = "INDEX_RECOVERED"
    SPHERE_RETAINED = "SPHERE_RETAINED"
    SPHERE_ESCAPED = "SPHERE_ESCAPED"
    TABLE_OR_FLOOR_CONTACT = "TABLE_OR_FLOOR_CONTACT"


@dataclass(frozen=True)
class LinkMeasurement:
    finger: str
    segment: str
    parent_body: str
    child_body: str
    vector_m: tuple[float, float, float]
    length_m: float


@dataclass(frozen=True)
class SphereScale:
    scale_id: str
    diameter_m: float
    radius_m: float
    density_kg_m3: float
    mass_kg: float


@dataclass(frozen=True)
class PocketRegion:
    name: str
    center_palm_m: tuple[float, float, float]
    half_extents_m: tuple[float, float, float]
    support_surfaces: tuple[str, ...]
    construction: str
    local_aperture_m: float

    def storage_region(self) -> StorageRegion:
        return StorageRegion(self.center_palm_m, self.half_extents_m)


@dataclass(frozen=True)
class SphereAcquisitionState:
    state_id: str
    candidate_id: int
    initial_position_m: tuple[float, float, float]
    qpos: tuple[float, ...]
    qvel: tuple[float, ...]
    ctrl: tuple[float, ...]
    contact_flags: tuple[float, ...]
    penetration_by_surface_m: tuple[float, ...]
    sha256: str


def load_phase3c06_config(path: Path | None = None) -> dict[str, Any]:
    source = path or ROOT / "configs/phase3C06_sphere_palmodigital.yaml"
    with source.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def audit_non_thumb_link_lengths(model_path: Path | None = None) -> tuple[LinkMeasurement, ...]:
    """Audit joint-to-joint proximal/intermediate lengths from official MJCF."""
    source = model_path or ROOT / "assets/hands/shadow_right/right_hand.xml"
    root = ET.parse(source).getroot()
    prefixes = {"index": "ff", "middle": "mf", "ring": "rf", "little": "lf"}
    rows: list[LinkMeasurement] = []
    for finger, prefix in prefixes.items():
        proximal = root.find(f".//body[@name='rh_{prefix}proximal']")
        middle = root.find(f".//body[@name='rh_{prefix}middle']")
        distal = root.find(f".//body[@name='rh_{prefix}distal']")
        if proximal is None or middle is None or distal is None:
            raise ValueError(f"official Shadow XML is missing audited {finger} bodies")
        for segment, parent, child in (
            ("proximal", proximal, middle), ("intermediate", middle, distal)
        ):
            vector = np.fromstring(child.get("pos", "0 0 0"), sep=" ")
            rows.append(LinkMeasurement(
                finger, segment, parent.get("name", ""), child.get("name", ""),
                tuple(float(value) for value in vector), float(np.linalg.norm(vector)),
            ))
    return tuple(rows)


def reference_link_length(measurements: Iterable[LinkMeasurement] | None = None) -> float:
    rows = tuple(measurements or audit_non_thumb_link_lengths())
    if not rows:
        raise ValueError("at least one official link measurement is required")
    return float(np.median([row.length_m for row in rows]))


def sphere_scale(scale: float = 1.0, density_kg_m3: float | None = None) -> SphereScale:
    cfg = load_phase3c06_config()
    density = float(density_kg_m3 or cfg["object"]["density_kg_m3"])
    diameter = reference_link_length() * float(scale)
    radius = diameter / 2.0
    mass = density * (4.0 / 3.0) * np.pi * radius ** 3
    label = {1.0: "D0", 1.25: "D1", 1.5: "D2", 1.75: "D3", 2.0: "D4"}.get(float(scale), f"D{scale:g}")
    return SphereScale(label, diameter, radius, density, float(mass))


def phase3c06_scene_config(scale: float = 1.0) -> Phase3Config:
    base = load_phase3_config()
    cfg = load_phase3c06_config()
    size = sphere_scale(scale)
    raw = dict(base.raw)
    raw["object"] = {
        "name": cfg["object"]["name"], "shape": "sphere", "size": [size.radius_m],
        "density": size.density_kg_m3, "friction": list(cfg["object"]["friction"]),
        "rgba": list(cfg["object"]["rgba"]),
        "initial_pos": list(base.object["initial_pos"]), "initial_quat": [1.0, 0.0, 0.0, 0.0],
        "fixture_name": cfg["object"]["fixture_name"],
    }
    return replace(base, raw=raw)


def build_sphere_scene(scale: float = 1.0) -> ShadowScene:
    return build_shadow_scene(phase3c06_scene_config(scale))


def _project(scene: ShadowScene, keyframe: str) -> np.ndarray:
    return np.asarray(project_feasible_hand_qpos(load_keyframe_qpos(keyframe), scene).projected_qpos)


def _body_palm(scene: ShadowScene, name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, name)
    return world_to_palm(scene, scene.data.xpos[body_id])


def construct_palmodigital_pockets(scene: ShadowScene | None = None, radius_m: float | None = None) -> dict[str, PocketRegion]:
    """Construct volumes from compiled open-hand finger roots and prior control."""
    scene = scene or build_sphere_scene()
    radius = float(radius_m or scene.config.object["size"][0])
    qpos = _project(scene, "open hand")
    scene.data.qpos[:24] = qpos
    scene.data.ctrl[:] = actuator_target_from_qpos(scene, qpos)
    mujoco.mj_forward(scene.model, scene.data)
    roots = {
        "middle": _body_palm(scene, "rh_mfknuckle"),
        "ring": _body_palm(scene, "rh_rfknuckle"),
        "little": _body_palm(scene, "rh_lfknuckle"),
    }
    old_raw = yaml.safe_load((ROOT / "configs/phase3C0_open_corridor.yaml").read_text(encoding="utf-8"))["storage_region"]
    old = np.asarray(old_raw["center_m"], dtype=float)
    middle_ring_anchor = (roots["middle"] + roots["ring"]) / 2.0
    ring_little_anchor = (roots["ring"] + roots["little"]) / 2.0
    # The open-hand palm surface is on negative palm-y. A one-radius inward
    # offset places the sphere center adjacent to, rather than inside, the root
    # axes. The z offset centers the volume just proximal to the joints.
    root_offset = np.asarray([0.0, -radius, -0.25 * radius])
    middle_ring = middle_ring_anchor + root_offset
    ring_little = ring_little_anchor + root_offset
    # Adjacent ulnar volume is derived between the ring/little root midpoint
    # and the previously configured palmar z plane, retaining ulnar x.
    ulnar = np.asarray([
        ring_little_anchor[0] - 0.15 * radius,
        -1.25 * radius,
        0.5 * (ring_little_anchor[2] + old[2]),
    ])
    half = (0.75 * radius, 0.55 * radius, 0.75 * radius)
    aperture_mr = float(np.linalg.norm(roots["middle"] - roots["ring"]))
    aperture_rl = float(np.linalg.norm(roots["ring"] - roots["little"]))
    return {
        "old_palm_center": PocketRegion(
            "old_palm_center", tuple(old), tuple(float(v) for v in old_raw["half_extents_m"]),
            ("palm", "middle", "ring", "little"), "Phase 3C-0 center preserved unchanged as control",
            float(min(2.0 * np.asarray(old_raw["half_extents_m"]))),
        ),
        "middle_ring": PocketRegion(
            "middle_ring", tuple(middle_ring), half, ("palm", "middle", "ring"),
            "midpoint of compiled open-hand middle/ring roots plus radius-derived palmar/proximal offset",
            aperture_mr,
        ),
        "ring_little": PocketRegion(
            "ring_little", tuple(ring_little), half, ("palm", "ring", "little"),
            "midpoint of compiled open-hand ring/little roots plus radius-derived palmar/proximal offset",
            aperture_rl,
        ),
        "ulnar_palmodigital": PocketRegion(
            "ulnar_palmodigital", tuple(ulnar), half, ("palm", "ring", "little"),
            "adjacent ulnar volume between ring/little root midpoint and prior palmar z plane",
            aperture_rl,
        ),
    }


def pocket_geometry(scene: ShadowScene, pocket: PocketRegion, radius_m: float) -> dict[str, Any]:
    center = np.asarray(pocket.center_palm_m)
    root_points = {finger: _body_palm(scene, {
        "middle": "rh_mfknuckle", "ring": "rh_rfknuckle", "little": "rh_lfknuckle"
    }[finger]) for finger in ("middle", "ring", "little")}
    clearances = {finger: float(np.linalg.norm(center - point) - radius_m) for finger, point in root_points.items()}
    directions = {
        "radial_from_palm_origin": (center / max(np.linalg.norm(center), 1e-12)).tolist(),
        "gravity_in_palm": (gravity_in_palm_frame(scene) / np.linalg.norm(gravity_in_palm_frame(scene))).tolist(),
    }
    return {
        "sphere_center_feasible_region": asdict(pocket),
        "palm_clearance_proxy_m": float(abs(center[1]) - radius_m),
        "proximal_finger_clearance_m": clearances,
        "ring_little_reachable_enclosure_m": {
            "ring_to_center": float(np.linalg.norm(center - root_points["ring"])),
            "little_to_center": float(np.linalg.norm(center - root_points["little"])),
        },
        "sphere_escape_directions": directions,
        "local_aperture_m": pocket.local_aperture_m,
        "support_surface_geometry": list(pocket.support_surfaces),
    }


def preshape_trigger(
    progress: float,
    corridor_bottleneck_fraction: float,
    sphere_storage_finger_clearance_m: float,
    predicted_sweep_clearance_m: float,
) -> bool:
    """Pure geometry event: bottleneck passed and current/future paths clear."""
    return bool(
        progress > corridor_bottleneck_fraction
        and sphere_storage_finger_clearance_m >= 0.0
        and predicted_sweep_clearance_m >= 0.0
    )


def normalized_penetration(penetration_m: Iterable[float], radius_m: float) -> np.ndarray:
    if radius_m <= 0.0:
        raise ValueError("sphere radius must be positive")
    return np.asarray(tuple(penetration_m), dtype=float) / float(radius_m)


def floor_contact(scene: ShadowScene) -> bool:
    floor_geom = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, scene.config.raw["floor"]["name"])
    for index in range(scene.data.ncon):
        contact = scene.data.contact[index]
        if floor_geom in {int(contact.geom1), int(contact.geom2)} and scene.object_body_id in {
            int(scene.model.geom_bodyid[contact.geom1]), int(scene.model.geom_bodyid[contact.geom2])
        }:
            return True
    return False


def storage_state(scene: ShadowScene, pocket: PocketRegion) -> dict[str, Any]:
    contacts = extract_shadow_contacts(scene)
    center, _ = object_pose_in_palm(scene, scene.object_body_id)
    inside = pocket.storage_region().measure(center, np.zeros(3))["center_inside"]
    contact_surfaces = tuple(
        surface for index, surface in enumerate(SUPPORT_SURFACES) if contacts.contact_flags[index]
    )
    alternate = tuple(surface for surface in contact_surfaces if surface not in {"thumb", "index"})
    physically_stored = bool(inside and not floor_contact(scene) and alternate)
    return {
        "center_inside": bool(inside), "floor_contact": floor_contact(scene),
        "contact_topology": list(contact_surfaces),
        "load_bearing_topology": [
            surface for index, surface in enumerate(SUPPORT_SURFACES) if contacts.normal_forces[index] > 0.0
        ],
        "alternate_support": bool(alternate), "physically_stored": physically_stored,
    }


def size_curriculum() -> tuple[SphereScale, ...]:
    return tuple(sphere_scale(float(value)) for value in load_phase3c06_config()["size_curriculum"]["scale"])


def wrist_commands(level: str) -> tuple[tuple[float, float], ...]:
    cfg = load_phase3c06_config()["wrist"]
    if level in {"W0", "W1"}:
        return tuple(tuple(float(v) for v in row) for row in cfg[f"{level}_commands_deg"])
    magnitude = float(cfg[f"{level}_magnitude_deg"])
    return tuple((a * magnitude, b * magnitude) for a, b in ((-1, -1), (-1, 1), (1, -1), (1, 1)))


def progression_allowed(rows: Iterable[dict[str, Any]]) -> bool:
    """Structural gate only: 'multiple distinct' means at least two IDs."""
    valid = {
        row["state_id"] for row in rows
        if row.get("thumb_recovered") and row.get("survival", {}).get("1000")
        and row.get("penetration_valid_for_progression") is True
    }
    return len(valid) >= 2


def deterministic_acquisition_position(scene: ShadowScene, candidate_id: int) -> np.ndarray:
    # Preserve the already validated Phase 3A thumb/index acquisition center;
    # only the object geometry and density-consistent mass change here.
    center = np.asarray(load_phase3_config().object["initial_pos"], dtype=float)
    half = np.asarray(load_phase3c06_config()["matched_states"]["position_half_width_m"])
    phases = np.mod((candidate_id + 0.5) * np.asarray([0.61803398875, 0.41421356237, 0.73205080757]), 1.0)
    return center + (2.0 * phases - 1.0) * half


def acquire_sphere_state(scene: ShadowScene, candidate_id: int) -> SphereAcquisitionState | None:
    mujoco.mj_resetData(scene.model, scene.data)
    open_qpos, pre_qpos, pinch_qpos = (_project(scene, name) for name in ("open hand", "pre grasp", "two finger pinch"))
    open_target = actuator_target_from_qpos(scene, open_qpos)
    pre_target = actuator_target_from_qpos(scene, pre_qpos)
    pinch_target = actuator_target_from_qpos(scene, pinch_qpos)
    scene.data.qpos[:24] = open_qpos
    scene.data.ctrl[:] = open_target
    position = deterministic_acquisition_position(scene, candidate_id)
    set_object_pose(scene, position)
    set_fixture(scene, True)
    mujoco.mj_forward(scene.model, scene.data)
    if np.any(extract_shadow_contacts(scene).contact_flags[2:5]):
        return None
    approach_ids = np.r_[scene.actuator_ids["wrist"], scene.actuator_ids["thumb"], scene.actuator_ids["index"]]
    for step in range(80):
        alpha = (step + 1) / 80.0
        scene.data.ctrl[approach_ids] = (1.0 - alpha) * open_target[approach_ids] + alpha * pre_target[approach_ids]
        mujoco.mj_step(scene.model, scene.data)
    closer = ContactAwareCloser(scene, float(load_phase3c06_config()["diagnostic"]["contact_force_n"]))
    ids = np.r_[scene.actuator_ids["thumb"], scene.actuator_ids["index"]]
    for step in range(180):
        alpha = (step + 1) / 180.0
        proposed = scene.data.ctrl.copy()
        proposed[ids] = (1.0 - alpha) * pre_target[ids] + alpha * pinch_target[ids]
        scene.data.ctrl[:] = closer.limit_target(proposed)
        mujoco.mj_step(scene.model, scene.data)
    for _ in range(50):
        mujoco.mj_step(scene.model, scene.data)
    contacts = extract_shadow_contacts(scene)
    if not bool(contacts.contact_flags[0] and contacts.contact_flags[1]):
        return None
    payload = np.r_[scene.data.qpos, scene.data.qvel, scene.data.ctrl]
    return SphereAcquisitionState(
        f"C06_D0_STATE_{candidate_id:05d}", candidate_id, tuple(float(v) for v in position),
        tuple(float(v) for v in scene.data.qpos), tuple(float(v) for v in scene.data.qvel),
        tuple(float(v) for v in scene.data.ctrl), tuple(float(v) for v in contacts.contact_flags),
        tuple(float(v) for v in contacts.penetration_by_surface), hashlib.sha256(payload.tobytes()).hexdigest(),
    )


def freeze_acquisition_states(output_dir: Path | None = None) -> dict[str, Any]:
    output = output_dir or ROOT / "outputs/phase3C06/matched_states"
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_phase3c06_config()["matched_states"]
    scene = build_sphere_scene()
    states: list[SphereAcquisitionState] = []
    for candidate_id in range(int(cfg["maximum_candidates"])):
        state = acquire_sphere_state(scene, candidate_id)
        if state is not None:
            states.append(state)
        if len(states) == int(cfg["count"]):
            break
    if len(states) != int(cfg["count"]):
        raise RuntimeError(f"only {len(states)} physically valid thumb/index sphere states found")
    for state in states:
        np.savez_compressed(output / f"{state.state_id}.npz", qpos=state.qpos, qvel=state.qvel, ctrl=state.ctrl)
    manifest = {
        "phase": "3C-0.6", "count": len(states), "frozen_before_outcomes": True,
        "state_ids": [state.state_id for state in states],
        "states": [asdict(state) | {"qpos": None, "qvel": None, "ctrl": None} for state in states],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"manifest": manifest, "states": states}


def load_acquisition_states(output_dir: Path | None = None) -> list[SphereAcquisitionState]:
    output = output_dir or ROOT / "outputs/phase3C06/matched_states"
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


def exact_unused_clearance(scene: ShadowScene, fingers: Iterable[str]) -> float:
    object_geom = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, f"{scene.config.object['name']}_geom")
    values = []
    for finger in fingers:
        for geom_name in scene.collision_geoms[finger]:
            geom = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            values.append(mujoco.mj_geomDistance(scene.model, scene.data, object_geom, geom, 0.25, None))
    return float(min(values, default=np.inf))


def _joint_margin(scene: ShadowScene) -> float:
    joint_ids = np.concatenate(tuple(scene.joint_ids.values()))
    qpos = scene.data.qpos[scene.model.jnt_qposadr[joint_ids]]
    limits = scene.model.jnt_range[joint_ids]
    return float(np.min(np.minimum(qpos - limits[:, 0], limits[:, 1] - qpos)))


def _sample(scene: ShadowScene, pocket: PocketRegion, radius: float, step: int, stage: str) -> dict[str, Any]:
    contacts = extract_shadow_contacts(scene)
    center, _ = object_pose_in_palm(scene, scene.object_body_id)
    linear, angular = object_velocity(scene)
    storage = storage_state(scene, pocket)
    contact_pairs = [{
        "geom_pair": [record.geom1_name, record.geom2_name],
        "body_pair": [record.body1_name, record.body2_name],
        "position_world_m": record.position.tolist(), "normal_world": record.normal.tolist(),
        "distance_m": float(record.distance), "penetration_m": float(max(0.0, -record.distance)),
        "normal_force_n": float(record.normal_force),
        "tangential_force_n": float(record.tangential_force),
    } for record in contacts.object_records]
    return {
        "step": int(step), "stage": stage, "center_palm_m": center.tolist(),
        "contact_flags": contacts.contact_flags.tolist(), "normal_forces_n": contacts.normal_forces.tolist(),
        "penetration_by_surface_m": contacts.penetration_by_surface.tolist(),
        "penetration_by_surface_over_radius": normalized_penetration(contacts.penetration_by_surface, radius).tolist(),
        "maximum_penetration_m": contacts.maximum_penetration,
        "maximum_penetration_over_radius": float(contacts.maximum_penetration / radius),
        "maximum_penetration_pair": contacts.maximum_penetration_pair,
        "gravity_in_palm_mps2": gravity_in_palm_frame(scene).tolist(),
        "linear_speed_mps": float(np.linalg.norm(linear)), "angular_speed_radps": float(np.linalg.norm(angular)),
        "unused_finger_clearance_m": exact_unused_clearance(scene, ("middle", "ring", "little")),
        "floor_contact": floor_contact(scene), "storage": storage, "minimum_joint_margin_rad": _joint_margin(scene),
        "contact_pairs": contact_pairs,
    }


def run_storage_trial(
    scene: ShadowScene,
    state: SphereAcquisitionState,
    pocket: PocketRegion,
    preshape: str,
    wrist_delta_deg: tuple[float, float],
    *,
    scale_id: str = "D0",
    frame_callback: Any | None = None,
) -> dict[str, Any]:
    cfg = load_phase3c06_config()
    restore_acquisition_state(scene, state)
    radius = float(scene.config.object["size"][0])
    open_target = actuator_target_from_qpos(scene, _project(scene, "open hand"))
    transfer_target = actuator_target_from_qpos(scene, _project(scene, "three finger pinch"))
    storage_target = actuator_target_from_qpos(scene, _project(scene, "grasp hard"))
    subset = tuple(cfg["experiment"]["storage_fingers"][pocket.name])
    wrist_target = scene.data.ctrl.copy()
    wrist_ids = scene.actuator_ids["wrist"]
    wrist_target[wrist_ids] = np.clip(
        wrist_target[wrist_ids] + np.deg2rad(wrist_delta_deg),
        scene.model.actuator_ctrlrange[wrist_ids, 0], scene.model.actuator_ctrlrange[wrist_ids, 1],
    )
    start_center = object_pose_in_palm(scene, scene.object_body_id)[0]
    target_center = np.asarray(pocket.center_palm_m)
    bottleneck_fraction = 0.50
    preshape_step = None
    preshape_trigger_state = None
    first_contact = {finger: None for finger in ("middle", "ring", "little", "palm")}
    samples: list[dict[str, Any]] = []
    pocket_entry_step = None
    corridor_clear = True
    previous_center = start_center.copy()
    escape_direction = np.zeros(3)
    total_steps = int(cfg["experiment"]["clearing_steps"]) + int(cfg["experiment"]["transfer_steps"])
    for step in range(total_steps):
        transfer_local = max(0, step - int(cfg["experiment"]["clearing_steps"]))
        progress = min(1.0, transfer_local / max(1, int(cfg["experiment"]["transfer_steps"]) - 1))
        for finger in ("middle", "ring", "little"):
            scene.data.ctrl[scene.actuator_ids[finger]] = open_target[scene.actuator_ids[finger]]
        if step >= int(cfg["experiment"]["clearing_steps"]):
            for finger in ("thumb", "index"):
                ids = scene.actuator_ids[finger]
                scene.data.ctrl[ids] += np.clip(transfer_target[ids] - scene.data.ctrl[ids], -0.0005, 0.0005)
            scene.data.ctrl[wrist_ids] += np.clip(wrist_target[wrist_ids] - scene.data.ctrl[wrist_ids], -np.deg2rad(0.1), np.deg2rad(0.1))
        center = object_pose_in_palm(scene, scene.object_body_id)[0]
        clearance = exact_unused_clearance(scene, subset)
        predicted_sweep = clearance - float(np.linalg.norm(center - previous_center))
        geometric_trigger = preshape_trigger(progress, bottleneck_fraction, clearance, predicted_sweep)
        inside = pocket.storage_region().measure(center, np.zeros(3))["center_inside"]
        if inside and pocket_entry_step is None:
            pocket_entry_step = step
        start_storage = bool(
            (preshape == "PRESHAPE" and geometric_trigger)
            or (preshape == "NO_PRESHAPE" and inside)
        )
        if start_storage and preshape_step is None:
            preshape_step = step
            preshape_trigger_state = {
                "step": int(step), "sphere_progress": float(progress),
                "corridor_bottleneck_fraction": float(bottleneck_fraction),
                "sphere_storage_finger_clearance_m": float(clearance),
                "predicted_finger_sweep_clearance_m": float(predicted_sweep),
                "remaining_distance_to_pocket_m": float(np.linalg.norm(center - target_center)),
                "sphere_center_palm_m": center.tolist(),
            }
        if preshape_step is not None:
            for finger in subset:
                ids = scene.actuator_ids[finger]
                scene.data.ctrl[ids] += np.clip(storage_target[ids] - scene.data.ctrl[ids], -0.002, 0.002)
        mujoco.mj_step(scene.model, scene.data)
        if frame_callback is not None:
            frame_callback(scene, step, "PRESHAPING_STORAGE" if preshape_step is not None else "CLEARING_CORRIDOR")
        sample = _sample(scene, pocket, radius, step, "PRESHAPING_STORAGE" if preshape_step is not None else "CLEARING_CORRIDOR")
        samples.append(sample)
        corridor_clear &= bool(sample["unused_finger_clearance_m"] >= 0.0 or preshape_step is not None)
        for finger in first_contact:
            index = SUPPORT_SURFACES.index(finger)
            if first_contact[finger] is None and sample["contact_flags"][index]:
                first_contact[finger] = step
        previous_center = center
        escape_direction = np.asarray(sample["center_palm_m"]) - target_center
    final_storage = storage_state(scene, pocket)
    thumb_release_attempted = bool(final_storage["physically_stored"])
    survival: dict[str, bool] = {}
    thumb_contact_free = False
    retained_during_release = final_storage["physically_stored"]
    if thumb_release_attempted:
        ids = scene.actuator_ids["thumb"]
        start = scene.data.ctrl.copy()
        ramp = int(cfg["experiment"]["release_ramp_steps"])
        for local in range(ramp):
            alpha = (local + 1) / ramp
            scene.data.ctrl[ids] = (1.0 - alpha) * start[ids] + alpha * open_target[ids]
            mujoco.mj_step(scene.model, scene.data)
            if frame_callback is not None:
                frame_callback(scene, total_steps + local, "THUMB_RELEASE")
            retained_during_release &= storage_state(scene, pocket)["physically_stored"]
            samples.append(_sample(scene, pocket, radius, total_steps + local, "THUMB_RELEASE"))
        thumb_contact_free = True
        retained = retained_during_release
        checkpoints = set(int(v) for v in cfg["experiment"]["survival_checkpoints"])
        for post in range(1, max(checkpoints) + 1):
            mujoco.mj_step(scene.model, scene.data)
            if frame_callback is not None:
                frame_callback(scene, total_steps + ramp + post, "POST_RELEASE")
            contacts = extract_shadow_contacts(scene)
            thumb_contact_free &= not bool(contacts.contact_flags[0])
            retained &= storage_state(scene, pocket)["physically_stored"]
            if post in checkpoints:
                survival[str(post)] = bool(retained and thumb_contact_free)
            if post in checkpoints or post == 1:
                samples.append(_sample(scene, pocket, radius, total_steps + ramp + post, "POST_RELEASE"))
    max_pen = max(sample["maximum_penetration_m"] for sample in samples)
    max_ratio = max(sample["maximum_penetration_over_radius"] for sample in samples)
    storage_contact = any(any(sample["contact_flags"][SUPPORT_SURFACES.index(f)] for f in subset) for sample in samples)
    palm_contact = any(sample["contact_flags"][5] for sample in samples)
    alternate = any(any(sample["contact_flags"][2:]) for sample in samples)
    thumb_recovered = bool(thumb_release_attempted and survival.get("1000") and thumb_contact_free)
    failures = []
    if not corridor_clear: failures.append("TRANSFER_CORRIDOR_BLOCKED")
    if pocket_entry_step is None: failures.append("POCKET_NOT_REACHED")
    if not storage_contact: failures.append("NO_STORAGE_FINGER_CONTACT")
    if not alternate: failures.append("NO_LOAD_BEARING_SUPPORT")
    if thumb_release_attempted and not retained_during_release: failures.append("LOSS_DURING_THUMB_RELEASE")
    if thumb_release_attempted and retained_during_release and not survival.get("1000", False): failures.append("LOSS_AFTER_THUMB_RELEASE")
    if min(sample["minimum_joint_margin_rad"] for sample in samples) <= 0.0: failures.append("JOINT_BOUNDARY_LIMIT")
    if not failures and not thumb_recovered: failures.append("OTHER")
    outcomes = [SphereOutcome.SPHERE_ACQUIRED.value]
    if corridor_clear: outcomes.append(SphereOutcome.CORRIDOR_CLEARED.value)
    if preshape_step is not None: outcomes.append(SphereOutcome.PRESHAPE_STARTED.value)
    if pocket_entry_step is not None: outcomes.append(SphereOutcome.POCKET_ENTRY.value)
    if palm_contact: outcomes.append(SphereOutcome.PALM_OR_ROOT_CONTACT.value)
    if first_contact["ring"] is not None: outcomes.append(SphereOutcome.RING_CONTACT.value)
    if first_contact["little"] is not None: outcomes.append(SphereOutcome.LITTLE_CONTACT.value)
    if alternate: outcomes.append(SphereOutcome.ALTERNATE_SUPPORT.value)
    if thumb_release_attempted: outcomes.append(SphereOutcome.THUMB_RELEASED.value)
    if thumb_recovered: outcomes.extend((SphereOutcome.THUMB_RECOVERED.value, SphereOutcome.SPHERE_RETAINED.value))
    elif not final_storage["physically_stored"]: outcomes.append(SphereOutcome.SPHERE_ESCAPED.value)
    if any(sample["floor_contact"] for sample in samples): outcomes.append(SphereOutcome.TABLE_OR_FLOOR_CONTACT.value)
    return {
        "state_id": state.state_id, "scale_id": scale_id, "pocket": pocket.name,
        "preshape": preshape, "storage_fingers": list(subset),
        "wrist_delta_command_deg": list(wrist_delta_deg),
        "actual_wrist_motion_deg": np.rad2deg(scene.data.qpos[:2] - np.asarray(state.qpos[:2])).tolist(),
        "gravity_in_palm_final_mps2": gravity_in_palm_frame(scene).tolist(),
        "outcomes": outcomes, "failures": failures, "corridor_cleared": corridor_clear,
        "preshape_trigger_step": preshape_step, "pocket_entry_step": pocket_entry_step,
        "preshape_trigger_state": preshape_trigger_state,
        "first_storage_finger_contact_step": min((v for k, v in first_contact.items() if k != "palm" and v is not None), default=None),
        "first_contact_step": first_contact, "palm_contact": palm_contact,
        "ring_contact": first_contact["ring"] is not None, "little_contact": first_contact["little"] is not None,
        "alternate_support": alternate, "stable_capture": final_storage["physically_stored"],
        "thumb_release_attempted": thumb_release_attempted, "thumb_recovered": thumb_recovered,
        "index_recovered": False, "survival": survival,
        "maximum_penetration_m": float(max_pen), "maximum_penetration_over_radius": float(max_ratio),
        "maximum_penetration_by_surface_m": np.max([s["penetration_by_surface_m"] for s in samples], axis=0).tolist(),
        "maximum_penetration_by_surface_over_radius": np.max([s["penetration_by_surface_over_radius"] for s in samples], axis=0).tolist(),
        "gross_overlap_warning": None,
        "penetration_valid_for_progression": None,
        "penetration_acceptability": "TODO(PI): no Phase 3C-0.6 threshold is frozen",
        "escape_direction_palm": (escape_direction / max(np.linalg.norm(escape_direction), 1e-12)).tolist(),
        "minimum_joint_margin_rad": float(min(s["minimum_joint_margin_rad"] for s in samples)),
        "samples": samples,
    }


def no_object_b_or_rl_contract() -> dict[str, Any]:
    scene = build_sphere_scene()
    return {
        "object_B_instantiated": mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "object_B") >= 0,
        "rl_training_performed": False, "reward_defined": False, "scalar_J_defined": False,
    }
