"""Phase 3C-0.5 coordinated palmar-capture mechanics.

No reinforcement learning, reward weights, scalar objective, second object, or
physics override is defined here.  All gates are labeled engineering probes and
their raw measurements are retained.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np
import yaml

from .config import ROOT
from .phase3.config import FINGERS, SUPPORT_SURFACES
from .phase3.contacts import extract_shadow_contacts, object_velocity
from .phase3.control import ContactAwareCloser, actuator_target_from_qpos
from .phase3.env import load_keyframe_qpos
from .phase3.model import ShadowScene, build_shadow_scene, set_fixture, set_object_pose
from .phase3b0 import _joint_margins
from .phase3b1a import project_feasible_hand_qpos
from .phase3c0 import (
    Phase3CFingerRole,
    Phase3CRoles,
    Phase3CState,
    configured_storage_region,
    gravity_in_palm_frame,
    load_phase3c0_config,
    object_pose_in_palm,
    palm_transform,
    storage_measurement,
    transfer_corridor,
)


class CaptureStrategy(StrEnum):
    SERIAL = "C05-SERIAL"
    SIMULTANEOUS = "C05-SIMULTANEOUS"
    WRIST = "C05-WRIST"
    WRIST_LOAD_TRANSFER = "C05-WRIST-LOAD-TRANSFER"


class CaptureOutcome(StrEnum):
    A_STORAGE_ENTRY = "A_STORAGE_ENTRY"
    STORAGE_FINGER_CONTACT = "STORAGE_FINGER_CONTACT"
    PALM_CONTACT = "PALM_CONTACT"
    ALTERNATE_SUPPORT_ESTABLISHED = "ALTERNATE_SUPPORT_ESTABLISHED"
    COORDINATED_CAPTURE_ESTABLISHED = "COORDINATED_CAPTURE_ESTABLISHED"
    THUMB_RELEASED = "THUMB_RELEASED"
    INDEX_RELEASED = "INDEX_RELEASED"
    ONE_RESOURCE_RECOVERED = "ONE_RESOURCE_RECOVERED"
    BOTH_RESOURCES_RECOVERED = "BOTH_RESOURCES_RECOVERED"
    A_RETAINED = "A_RETAINED"
    A_LOST = "A_LOST"
    SUPPORT_GATE_NOT_REACHED = "SUPPORT_GATE_NOT_REACHED"


@dataclass(frozen=True)
class MatchedCaptureState:
    state_id: str
    source_candidate: int
    initial_position_m: tuple[float, float, float]
    initial_quaternion_wxyz: tuple[float, float, float, float]
    storage_entry_step: int
    approach_condition: str
    qpos: tuple[float, ...]
    qvel: tuple[float, ...]
    ctrl: tuple[float, ...]
    palm_frame_position_m: tuple[float, float, float]
    hand_object_contact_flags: tuple[float, ...]
    maximum_penetration_m: float
    sha256: str


@dataclass(frozen=True)
class LoadShare:
    acquisition_force_n: float
    alternate_force_n: float
    total_force_n: float
    alternate_fraction: float


@dataclass
class SupportPersistence:
    gates: tuple[float, ...] = (0.10, 0.25, 0.50)
    durations: tuple[int, ...] = (10, 25, 50)
    runs: dict[float, int] = field(default_factory=dict)
    first_reached: dict[str, int | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.runs = {gate: 0 for gate in self.gates}
        self.first_reached = {
            f"{gate:.2f}/{duration}": None for gate in self.gates for duration in self.durations
        }

    def update(self, fraction: float, step: int) -> None:
        for gate in self.gates:
            self.runs[gate] = self.runs[gate] + 1 if fraction >= gate else 0
            for duration in self.durations:
                key = f"{gate:.2f}/{duration}"
                if self.first_reached[key] is None and self.runs[gate] >= duration:
                    self.first_reached[key] = int(step - duration + 1)

    def reached(self, gate: float, duration: int) -> bool:
        return self.first_reached[f"{gate:.2f}/{duration}"] is not None


@dataclass(frozen=True)
class ConcurrentCaptureCommand:
    acquisition_target: np.ndarray
    storage_target: np.ndarray
    wrist_target: np.ndarray

    def apply(
        self,
        scene: ShadowScene,
        storage_subset: tuple[str, ...],
        storage_latches: dict[str, np.ndarray],
        *,
        acquisition_enabled: bool,
        wrist_enabled: bool,
        acquisition_increment: float,
        storage_increment: float,
        wrist_increment: float,
        contact_force_n: float,
    ) -> dict[str, np.ndarray]:
        """Apply three independent command channels in one control step."""
        contacts = extract_shadow_contacts(scene)
        if acquisition_enabled:
            reference_penetration = 0.003
            for surface, finger in enumerate(("thumb", "index")):
                ids = scene.actuator_ids[finger]
                # Hold the current target while contact is load bearing.  Advance
                # only when force is weak and penetration is still within the
                # inherited safety reference.
                if (contacts.normal_forces[surface] < contact_force_n
                        and contacts.penetration_by_surface[surface] < reference_penetration):
                    scene.data.ctrl[ids] += np.clip(
                        self.acquisition_target[ids] - scene.data.ctrl[ids],
                        -acquisition_increment,
                        acquisition_increment,
                    )
        for finger in storage_subset:
            ids = scene.actuator_ids[finger]
            surface = SUPPORT_SURFACES.index(finger)
            if finger not in storage_latches and contacts.normal_forces[surface] >= contact_force_n:
                storage_latches[finger] = scene.data.ctrl[ids].copy()
            target = storage_latches.get(finger, self.storage_target[ids])
            scene.data.ctrl[ids] += np.clip(target - scene.data.ctrl[ids], -storage_increment, storage_increment)
        if wrist_enabled:
            ids = scene.actuator_ids["wrist"]
            scene.data.ctrl[ids] += np.clip(
                self.wrist_target[ids] - scene.data.ctrl[ids], -wrist_increment, wrist_increment
            )
        return {name: value.copy() for name, value in storage_latches.items()}


def load_phase3c05_config(path: Path | None = None) -> dict[str, Any]:
    source = path or ROOT / "configs/phase3C05_coordinated_capture.yaml"
    with source.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def alternate_load(normal_forces: Iterable[float]) -> LoadShare:
    values = np.asarray(tuple(normal_forces), dtype=float)
    if values.shape != (6,):
        raise ValueError("support force vector must be thumb,index,middle,ring,little,palm")
    acquisition = float(values[:2].sum())
    alternate = float(values[2:].sum())
    total = acquisition + alternate
    return LoadShare(acquisition, alternate, total, alternate / total if total > 0.0 else 0.0)


def _project(name: str, scene: ShadowScene) -> np.ndarray:
    return np.asarray(project_feasible_hand_qpos(load_keyframe_qpos(name), scene).projected_qpos)


def _move(scene: ShadowScene, group: str, target: np.ndarray, increment: float) -> None:
    ids = scene.actuator_ids[group]
    scene.data.ctrl[ids] += np.clip(target[ids] - scene.data.ctrl[ids], -increment, increment)


def _floor_contact(scene: ShadowScene) -> bool:
    object_body = scene.object_body_id
    floor_geom = mujoco.mj_name2id(
        scene.model, mujoco.mjtObj.mjOBJ_GEOM, scene.config.raw["floor"]["name"]
    )
    for index in range(scene.data.ncon):
        contact = scene.data.contact[index]
        bodies = {int(scene.model.geom_bodyid[contact.geom1]), int(scene.model.geom_bodyid[contact.geom2])}
        if object_body in bodies and floor_geom in {int(contact.geom1), int(contact.geom2)}:
            return True
    return False


def object_retained_in_hand(scene: ShadowScene) -> bool:
    contacts = extract_shadow_contacts(scene)
    storage = storage_measurement(
        scene, scene.object_body_id, np.asarray(scene.config.object["size"])
    )
    return bool(not _floor_contact(scene) and (np.any(contacts.contact_flags) or storage["center_inside"]))


def _quaternion_from_rotvec(rotation: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotation))
    if angle == 0.0:
        return np.asarray([1.0, 0.0, 0.0, 0.0])
    axis = rotation / angle
    return np.r_[np.cos(angle / 2.0), axis * np.sin(angle / 2.0)]


def _multiply_quaternion(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    output = np.empty(4)
    mujoco.mju_mulQuat(output, left, right)
    return output / np.linalg.norm(output)


def deterministic_candidate(candidate_id: int) -> tuple[np.ndarray, np.ndarray]:
    cfg0 = load_phase3c0_config()
    cfg = load_phase3c05_config()
    base_position = np.asarray(cfg0["diagnostic"]["open_corridor_initial_pos_m"], dtype=float)
    base_quaternion = np.asarray(cfg0["diagnostic"]["open_corridor_initial_quat_wxyz"], dtype=float)
    half = np.asarray(cfg["matched_states"]["position_half_width_m"], dtype=float)
    angle_half = np.deg2rad(cfg["matched_states"]["orientation_half_width_deg"])
    # Irrational rotations give deterministic, non-grid-clustered coverage.
    phases = np.mod((candidate_id + 0.5) * np.asarray([0.61803398875, 0.41421356237, 0.73205080757]), 1.0)
    position = base_position + (2.0 * phases - 1.0) * half
    angle_phases = np.mod((candidate_id + 0.5) * np.asarray([0.27182818285, 0.14142135623, 0.17320508076]), 1.0)
    delta = (2.0 * angle_phases - 1.0) * angle_half
    quaternion = _multiply_quaternion(_quaternion_from_rotvec(delta), base_quaternion)
    return position, quaternion


def prepare_storage_entry_state(
    scene: ShadowScene,
    candidate_id: int,
    *,
    maximum_transfer_steps: int = 450,
) -> MatchedCaptureState | None:
    position, quaternion = deterministic_candidate(candidate_id)
    mujoco.mj_resetData(scene.model, scene.data)
    open_qpos = _project("open hand", scene)
    pre_qpos = _project("pre grasp", scene)
    pinch_qpos = _project("two finger pinch", scene)
    support_qpos = _project("three finger pinch", scene)
    open_target = actuator_target_from_qpos(scene, open_qpos)
    pre_target = actuator_target_from_qpos(scene, pre_qpos)
    pinch_target = actuator_target_from_qpos(scene, pinch_qpos)
    support_target = actuator_target_from_qpos(scene, support_qpos)
    scene.data.qpos[:24] = open_qpos
    scene.data.qvel[:] = 0.0
    set_object_pose(scene, position, quaternion)
    set_fixture(scene, True)
    scene.data.ctrl[:] = open_target
    mujoco.mj_forward(scene.model, scene.data)
    if any(extract_shadow_contacts(scene).contact_flags[2:5]):
        return None
    step = 0
    for _ in range(20):
        mujoco.mj_step(scene.model, scene.data); step += 1
    approach_ids = np.r_[scene.actuator_ids["wrist"], scene.actuator_ids["thumb"], scene.actuator_ids["index"]]
    approach_steps = int(scene.config.diagnostic["approach_steps"])
    for local in range(approach_steps):
        alpha = (local + 1) / approach_steps
        scene.data.ctrl[approach_ids] = (1.0 - alpha) * open_target[approach_ids] + alpha * pre_target[approach_ids]
        mujoco.mj_step(scene.model, scene.data); step += 1
    closer = ContactAwareCloser(scene, float(scene.config.diagnostic["contact_force_n"]))
    ids = np.r_[scene.actuator_ids["thumb"], scene.actuator_ids["index"]]
    close_steps = int(scene.config.diagnostic["close_steps"])
    for local in range(close_steps):
        alpha = (local + 1) / close_steps
        proposed = scene.data.ctrl.copy()
        proposed[ids] = (1.0 - alpha) * pre_target[ids] + alpha * pinch_target[ids]
        scene.data.ctrl[:] = closer.limit_target(proposed)
        mujoco.mj_step(scene.model, scene.data); step += 1
    for _ in range(int(scene.config.diagnostic["settle_steps"])):
        mujoco.mj_step(scene.model, scene.data); step += 1
    contacts = extract_shadow_contacts(scene)
    if not bool(contacts.contact_flags[0] and contacts.contact_flags[1]):
        return None
    set_fixture(scene, False)
    reference = float(load_phase3c05_config()["diagnostic"]["reference_penetration_m"])
    for _ in range(maximum_transfer_steps):
        contacts = extract_shadow_contacts(scene)
        for surface_index, finger in enumerate(("thumb", "index")):
            if contacts.penetration_by_surface[surface_index] < reference:
                _move(scene, finger, support_target, 0.0005)
        _move(scene, "wrist", support_target, 0.0002)
        for finger in ("middle", "ring", "little"):
            scene.data.ctrl[scene.actuator_ids[finger]] = open_target[scene.actuator_ids[finger]]
        mujoco.mj_step(scene.model, scene.data); step += 1
        if _floor_contact(scene):
            return None
        measurement = storage_measurement(
            scene, scene.object_body_id, np.asarray(scene.config.object["size"])
        )
        contacts = extract_shadow_contacts(scene)
        region = configured_storage_region()
        center = np.asarray(measurement["object_center_palm_m"])
        region_center = np.asarray(region.center_palm_m)
        expanded = np.asarray(region.half_extents_m) + float(
            load_phase3c0_config()["storage_region"]["entry_margin_m"]
        )
        approaches_storage = bool(np.all(np.abs(center - region_center) <= expanded))
        dual_retained = bool(contacts.contact_flags[0] and contacts.contact_flags[1])
        if approaches_storage and dual_retained:
            payload = np.r_[scene.data.qpos, scene.data.qvel, scene.data.ctrl]
            digest = hashlib.sha256(payload.tobytes()).hexdigest()
            return MatchedCaptureState(
                state_id=f"C05_STATE_{candidate_id:05d}",
                source_candidate=candidate_id,
                initial_position_m=tuple(float(v) for v in position),
                initial_quaternion_wxyz=tuple(float(v) for v in quaternion),
                storage_entry_step=step,
                approach_condition="A_APPROACHES_STORAGE_WITH_DUAL_ACQUISITION_CONTACT",
                qpos=tuple(float(v) for v in scene.data.qpos),
                qvel=tuple(float(v) for v in scene.data.qvel),
                ctrl=tuple(float(v) for v in scene.data.ctrl),
                palm_frame_position_m=tuple(measurement["object_center_palm_m"]),
                hand_object_contact_flags=tuple(float(v) for v in contacts.contact_flags),
                maximum_penetration_m=float(contacts.maximum_penetration),
                sha256=digest,
            )
    return None


def freeze_matched_states(output_dir: Path | None = None) -> dict[str, Any]:
    output_dir = output_dir or ROOT / "outputs/phase3C05/matched_states"
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_phase3c05_config()
    required = int(cfg["matched_states"]["count"])
    scene = build_shadow_scene()
    states: list[MatchedCaptureState] = []
    candidate_id = 0
    while len(states) < required and candidate_id < required * 20:
        state = prepare_storage_entry_state(scene, candidate_id)
        if state is not None:
            states.append(state)
        candidate_id += 1
    if len(states) != required:
        raise RuntimeError(f"only {len(states)} physically valid storage-entry states found")
    for state in states:
        path = output_dir / f"{state.state_id}.npz"
        np.savez_compressed(path, qpos=state.qpos, qvel=state.qvel, ctrl=state.ctrl)
    manifest = {
        "seed": int(cfg["seed"]), "count": len(states),
        "frozen_before_capture_conditions": True,
        "states": [asdict(state) | {"qpos": None, "qvel": None, "ctrl": None} for state in states],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"manifest": manifest, "states": states}


def restore_state(scene: ShadowScene, state: MatchedCaptureState) -> None:
    mujoco.mj_resetData(scene.model, scene.data)
    scene.data.qpos[:] = state.qpos
    scene.data.qvel[:] = state.qvel
    scene.data.ctrl[:] = state.ctrl
    set_fixture(scene, False)
    mujoco.mj_forward(scene.model, scene.data)


def load_frozen_states(output_dir: Path | None = None) -> list[MatchedCaptureState]:
    output_dir = output_dir or ROOT / "outputs/phase3C05/matched_states"
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    states = []
    for row in manifest["states"]:
        arrays = np.load(output_dir / f"{row['state_id']}.npz")
        states.append(MatchedCaptureState(
            **{key: value for key, value in row.items() if key not in {"qpos", "qvel", "ctrl"}},
            qpos=tuple(arrays["qpos"]), qvel=tuple(arrays["qvel"]), ctrl=tuple(arrays["ctrl"]),
        ))
    return states


def phase3c0_failure_audit(result_path: Path | None = None) -> dict[str, Any]:
    result_path = result_path or ROOT / "outputs/phase3C0/phase3c0_results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    rows = []
    for trial in result["C0_A_and_B"]["trials"]["open_corridor"]:
        samples = trial["samples"]
        def first(predicate):
            match = next((sample for sample in samples if predicate(sample)), None)
            return None if match is None else int(match["step"])
        alternate = lambda sample: float(sum(sample["normal_forces_n"][2:])) > 0.0
        release_start = first(lambda sample: sample["stage"] == "RELEASE_ACQUISITION_DIGITS")
        first_alt = first(alternate)
        first_palm = first(lambda sample: sample["normal_forces_n"][5] > 0.0)
        first_storage = first(lambda sample: any(value > 0.0 for value in sample["normal_forces_n"][2:5]))
        loss = first(lambda sample: sample["floor_contact"])
        acquisition_support_loss = first(
            lambda sample: sample["step"] >= trial["storage_entry_step"]
            and float(sum(sample["normal_forces_n"][:2])) == 0.0
        )
        entry_sample = next(sample for sample in samples if sample["step"] >= trial["storage_entry_step"])
        rows.append({
            "initial_position_m": trial["object_initial_position_m"],
            "storage_entry_step": trial["storage_entry_step"],
            "storage_motion_start_step": trial["recruitment_step"],
            "first_storage_finger_contact_step": first_storage,
            "first_palm_contact_step": first_palm,
            "acquisition_unloading_start_step": release_start,
            "first_alternate_support_step": first_alt,
            "first_acquisition_support_loss_step": acquisition_support_loss,
            "A_loss_step": loss,
            "entry_wrist_qpos": entry_sample["hand_qpos"][:2],
            "acquisition_unloading_before_alternate_support": bool(
                release_start is not None and (first_alt is None or release_start < first_alt)
            ),
            "controlled_release_started_before_A_loss": bool(
                release_start is not None and loss is not None and release_start < loss
            ),
            "acquisition_support_lost_before_storage_contact": bool(
                acquisition_support_loss is not None
                and (first_storage is None or acquisition_support_loss < first_storage)
            ),
            "alternate_support_lost_before_release": bool(
                first_alt is not None and release_start is not None
                and not any(alternate(sample) for sample in samples if sample["step"] >= release_start)
            ),
            "timeseries": [{
                "step": sample["step"], "stage": sample["stage"],
                "normal_forces_n": sample["normal_forces_n"],
                "A_position_palm_m": sample["object_position_palm_m"],
                "A_orientation_palm": sample["storage"]["orientation_matrix_palm"],
                "A_linear_velocity_world_mps": sample["object_qvel"][:3],
                "A_angular_velocity_world_radps": sample["object_qvel"][3:],
                "gravity_in_palm_frame": sample["gravity_in_palm_frame"],
                "support_topology": [SUPPORT_SURFACES[i] for i, force in enumerate(sample["normal_forces_n"]) if force > 0.0],
            } for sample in samples],
        })
    return {"trials": rows, "all_release_after_storage_entry": all(
        row["acquisition_unloading_start_step"] > row["storage_entry_step"] for row in rows
    )}


def exact_object_unused_finger_clearance(scene: ShadowScene) -> float:
    """Return exact compiled-geom clearance from A to middle/ring/little."""
    object_geom = mujoco.mj_name2id(
        scene.model, mujoco.mjtObj.mjOBJ_GEOM, f"{scene.config.object['name']}_geom"
    )
    distances = []
    for finger in ("middle", "ring", "little"):
        for name in scene.collision_geoms[finger]:
            geom_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            distances.append(mujoco.mj_geomDistance(
                scene.model, scene.data, object_geom, geom_id, 0.25, None
            ))
    return float(min(distances))


def exact_corridor_metric_audit(result_path: Path | None = None) -> dict[str, Any]:
    result_path = result_path or ROOT / "outputs/phase3C0/phase3c0_results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    scene = build_shadow_scene()
    rows = []
    for trial in result["C0_A_and_B"]["trials"]["open_corridor"]:
        transfer_samples = [sample for sample in trial["samples"]
                            if sample["stage"] in {"MINIMAL_UNSUPPORTED_HOLD", "TRANSFER_A_TO_PALM"}]
        exact_actual = []
        old_actual = []
        actual_contact = []
        for sample in transfer_samples:
            scene.data.qpos[:24] = sample["hand_qpos"]
            address = scene.model.jnt_qposadr[scene.object_joint_id]
            scene.data.qpos[address:address + 7] = sample["object_qpos"]
            mujoco.mj_forward(scene.model, scene.data)
            exact_actual.append(exact_object_unused_finger_clearance(scene))
            old = transfer_corridor(
                scene, scene.data.xpos[scene.object_body_id], scene.data.xpos[scene.object_body_id],
                object_radius_m=float(max(scene.config.object["size"])), samples=1,
            )
            old_actual.append(float(old["minimum_clearance_m"]))
            actual_contact.append(bool(any(sample["contact_flags"][2:5])))
        rows.append({
            "initial_position_m": trial["object_initial_position_m"],
            "old_sphere_minimum_actual_path_clearance_m": float(min(old_actual)),
            "exact_geom_minimum_actual_path_clearance_m": float(min(exact_actual)),
            "observed_unused_finger_contact_steps": int(sum(actual_contact)),
            "old_predicted_obstruction": bool(min(old_actual) < 0.0),
            "exact_predicted_obstruction_on_actual_path": bool(min(exact_actual) < 0.0),
        })
    return {
        "coordinate_frames_consistent": True,
        "old_method": "object bounding sphere versus collision-geom bounding spheres on a straight candidate path",
        "exact_audit_method": "MuJoCo mj_geomDistance using oriented ellipsoid and compiled finger geoms on actual dynamic samples",
        "implementation_bug": False,
        "conservatism_sources": ["bounding spheres", "straight candidate path differs from actual dynamic path"],
        "retain_both_metrics": True,
        "trials": rows,
    }


def _sample_capture(
    scene: ShadowScene,
    step: int,
    persistence: SupportPersistence,
    subset: tuple[str, ...],
    storage_latches: dict[str, np.ndarray],
    initial_object_position: np.ndarray,
    *,
    record_state: bool = False,
) -> dict[str, Any]:
    contacts = extract_shadow_contacts(scene)
    load = alternate_load(contacts.normal_forces)
    linear, angular = object_velocity(scene)
    position, orientation = object_pose_in_palm(scene, scene.object_body_id)
    joint_margin, _ = _joint_margins(scene)
    clipped = np.isclose(scene.data.ctrl, scene.model.actuator_ctrlrange[:, 0]) | np.isclose(
        scene.data.ctrl, scene.model.actuator_ctrlrange[:, 1]
    )
    persistence.update(load.alternate_fraction, step)
    ratios = np.divide(
        contacts.tangential_forces,
        contacts.normal_forces,
        out=np.zeros_like(contacts.tangential_forces),
        where=contacts.normal_forces > 0.0,
    )
    contact_details = {
        surface: [{
            "geom_pair": [record.geom1_name, record.geom2_name],
            "body_pair": [record.body1_name, record.body2_name],
            "position_world_m": record.position.tolist(),
            "normal_world": record.normal.tolist(),
            "penetration_m": max(0.0, -record.distance),
            "normal_force_n": record.normal_force,
            "tangential_force_n": record.tangential_force,
            "tangential_normal_ratio": (
                record.tangential_force / record.normal_force if record.normal_force > 0.0 else 0.0
            ),
        } for record in contacts.records_by_surface[surface]]
        for surface in SUPPORT_SURFACES
    }
    sample = {
        "step": int(step), "normal_forces_n": contacts.normal_forces.tolist(),
        "tangential_forces_n": contacts.tangential_forces.tolist(),
        "tangential_normal_ratio_by_surface": ratios.tolist(),
        "penetration_by_surface_m": contacts.penetration_by_surface.tolist(),
        "contact_details": contact_details,
        "acquisition_force_n": load.acquisition_force_n,
        "alternate_force_n": load.alternate_force_n,
        "total_force_n": load.total_force_n,
        "alternate_fraction": load.alternate_fraction,
        "contact_flags": contacts.contact_flags.tolist(),
        "maximum_penetration_m": contacts.maximum_penetration,
        "maximum_penetration_pair": contacts.maximum_penetration_pair,
        "A_position_palm_m": position.tolist(), "A_orientation_palm": orientation.tolist(),
        "A_displacement_from_capture_start_m": float(
            np.linalg.norm(scene.data.xpos[scene.object_body_id] - initial_object_position)
        ),
        "A_linear_speed_mps": float(np.linalg.norm(linear)),
        "A_angular_speed_radps": float(np.linalg.norm(angular)),
        "gravity_in_palm_frame": gravity_in_palm_frame(scene).tolist(),
        "support_topology": [SUPPORT_SURFACES[i] for i, force in enumerate(contacts.normal_forces) if force > 0.0],
        "storage_subset": list(subset), "storage_latched": sorted(storage_latches),
        "wrist_qpos": scene.data.qpos[:2].tolist(),
        "minimum_joint_margin_rad": float(joint_margin.min()),
        "actuator_clipping_count": int(clipped.sum()),
        "floor_contact": _floor_contact(scene),
    }
    if record_state:
        sample["qpos"] = scene.data.qpos.copy().tolist()
        sample["qvel"] = scene.data.qvel.copy().tolist()
    return sample


def capture_trial(
    scene: ShadowScene,
    state: MatchedCaptureState,
    subset: tuple[str, ...],
    strategy: CaptureStrategy,
    *,
    wrist_delta_deg: tuple[float, float] = (0.0, 0.0),
    capture_steps: int | None = None,
    acquisition_keyframe: str = "three finger pinch",
    storage_keyframe: str = "grasp hard",
    record_state: bool = False,
) -> dict[str, Any]:
    cfg = load_phase3c05_config()
    restore_state(scene, state)
    capture_steps = capture_steps or int(cfg["capture"]["steps"])
    acquisition_target = actuator_target_from_qpos(scene, _project(acquisition_keyframe, scene))
    storage_target = actuator_target_from_qpos(scene, _project(storage_keyframe, scene))
    wrist_target = scene.data.ctrl.copy()
    wrist_target[scene.actuator_ids["wrist"]] += np.deg2rad(wrist_delta_deg)
    wrist_target = np.clip(wrist_target, scene.model.actuator_ctrlrange[:, 0], scene.model.actuator_ctrlrange[:, 1])
    command = ConcurrentCaptureCommand(acquisition_target, storage_target, wrist_target)
    persistence = SupportPersistence(
        tuple(float(value) for value in cfg["load_share"]["diagnostic_gates"]),
        tuple(int(value) for value in cfg["load_share"]["persistence_steps"]),
    )
    storage_latches: dict[str, np.ndarray] = {}
    samples = []
    initial_position = scene.data.xpos[scene.object_body_id].copy()
    initial_wrist = scene.data.qpos[:2].copy()
    for step in range(capture_steps):
        simultaneous = strategy != CaptureStrategy.SERIAL
        wrist_enabled = strategy in {CaptureStrategy.WRIST, CaptureStrategy.WRIST_LOAD_TRANSFER}
        command.apply(
            scene, subset, storage_latches,
            acquisition_enabled=simultaneous,
            wrist_enabled=wrist_enabled,
            acquisition_increment=float(cfg["capture"]["acquisition_increment"]),
            storage_increment=float(cfg["capture"]["command_increment"]),
            wrist_increment=np.deg2rad(0.1),
            contact_force_n=float(cfg["capture"]["contact_force_n"]),
        )
        mujoco.mj_step(scene.model, scene.data)
        samples.append(_sample_capture(
            scene, step, persistence, subset, storage_latches, initial_position,
            record_state=record_state
        ))
    maximum_fraction = max(sample["alternate_fraction"] for sample in samples)
    outcomes = [CaptureOutcome.A_STORAGE_ENTRY.value]
    if any(any(sample["normal_forces_n"][SUPPORT_SURFACES.index(f)] > 0 for f in subset) for sample in samples):
        outcomes.append(CaptureOutcome.STORAGE_FINGER_CONTACT.value)
    if any(sample["normal_forces_n"][5] > 0 for sample in samples):
        outcomes.append(CaptureOutcome.PALM_CONTACT.value)
    if any(value is not None for value in persistence.first_reached.values()):
        outcomes.append(CaptureOutcome.ALTERNATE_SUPPORT_ESTABLISHED.value)
    retained = object_retained_in_hand(scene)
    if retained: outcomes.append(CaptureOutcome.A_RETAINED.value)
    else: outcomes.append(CaptureOutcome.A_LOST.value)
    if retained and persistence.reached(0.10, 25):
        outcomes.append(CaptureOutcome.COORDINATED_CAPTURE_ESTABLISHED.value)
    elif not persistence.reached(0.10, 25):
        outcomes.append(CaptureOutcome.SUPPORT_GATE_NOT_REACHED.value)
    final_snapshot = {
        "qpos": scene.data.qpos.copy(), "qvel": scene.data.qvel.copy(), "ctrl": scene.data.ctrl.copy()
    }
    return {
        "state_id": state.state_id, "subset": list(subset), "strategy": strategy.value,
        "acquisition_keyframe": acquisition_keyframe, "storage_keyframe": storage_keyframe,
        "wrist_delta_command_deg": list(wrist_delta_deg),
        "actual_wrist_motion_deg": np.rad2deg(scene.data.qpos[:2] - initial_wrist).tolist(),
        "storage_contacts_latched": sorted(storage_latches),
        "outcomes": outcomes, "A_retained": retained,
        "maximum_alternate_fraction": maximum_fraction,
        "persistence_first_reached": persistence.first_reached,
        "A_displacement_m": float(np.linalg.norm(scene.data.xpos[scene.object_body_id] - initial_position)),
        "samples": samples, "final_snapshot": final_snapshot,
    }


def released_finger_available_motion(scene: ShadowScene, finger: str) -> float:
    ids = scene.joint_ids[finger]
    qpos = scene.data.qpos[scene.model.jnt_qposadr[ids]]
    limits = scene.model.jnt_range[ids]
    return float(np.sum(np.minimum(qpos - limits[:, 0], limits[:, 1] - qpos)))


def release_trial(
    scene: ShadowScene,
    capture: dict[str, Any],
    finger: str,
    ramp_steps: int,
    *,
    post_steps: int = 1000,
    record_state: bool = False,
) -> dict[str, Any]:
    cfg = load_phase3c05_config()
    snap = capture["final_snapshot"]
    mujoco.mj_resetData(scene.model, scene.data)
    scene.data.qpos[:] = snap["qpos"]
    scene.data.qvel[:] = snap["qvel"]
    scene.data.ctrl[:] = snap["ctrl"]
    set_fixture(scene, False)
    mujoco.mj_forward(scene.model, scene.data)
    gate = float(cfg["release"]["default_gate"])
    duration = int(cfg["release"]["default_persistence_steps"])
    qualified = capture["persistence_first_reached"][f"{gate:.2f}/{duration}"] is not None
    if not qualified:
        return {"executed": False, "reason": CaptureOutcome.SUPPORT_GATE_NOT_REACHED.value,
                "finger": finger, "ramp_steps": ramp_steps}
    open_target = actuator_target_from_qpos(scene, _project("open hand", scene))
    ids = scene.actuator_ids[finger]
    start = scene.data.ctrl.copy()
    samples = []
    retained_during_ramp = object_retained_in_hand(scene)
    floor_free_during_ramp = not _floor_contact(scene)
    for step in range(ramp_steps):
        alpha = (step + 1) / ramp_steps
        scene.data.ctrl[ids] = (1.0 - alpha) * start[ids] + alpha * open_target[ids]
        mujoco.mj_step(scene.model, scene.data)
        contacts = extract_shadow_contacts(scene)
        retained_during_ramp &= object_retained_in_hand(scene)
        floor_free_during_ramp &= not _floor_contact(scene)
        sample = {"step": step, "phase": "release_ramp",
                        "normal_forces_n": contacts.normal_forces.tolist(),
                        "alternate_fraction": alternate_load(contacts.normal_forces).alternate_fraction,
                        "floor_contact": _floor_contact(scene),
                        "retained_A": object_retained_in_hand(scene),
                        "released_finger_contact": bool(contacts.contact_flags[SUPPORT_SURFACES.index(finger)]),
                        "A_position_palm_m": object_pose_in_palm(scene, scene.object_body_id)[0].tolist(),
                        "wrist_qpos": scene.data.qpos[:2].tolist()}
        if record_state:
            sample["qpos"] = scene.data.qpos.copy().tolist()
            sample["qvel"] = scene.data.qvel.copy().tolist()
        samples.append(sample)
    survival = {}
    contact_free = True
    retained = True
    finger_index = SUPPORT_SURFACES.index(finger)
    checkpoints = set(int(value) for value in cfg["release"]["post_release_steps"])
    for step in range(1, post_steps + 1):
        mujoco.mj_step(scene.model, scene.data)
        contacts = extract_shadow_contacts(scene)
        contact_free &= not bool(contacts.contact_flags[finger_index])
        retained &= object_retained_in_hand(scene)
        if record_state:
            samples.append({
                "step": int(ramp_steps + step), "phase": "post_release",
                "normal_forces_n": contacts.normal_forces.tolist(),
                "alternate_fraction": alternate_load(contacts.normal_forces).alternate_fraction,
                "floor_contact": _floor_contact(scene), "retained_A": object_retained_in_hand(scene),
                "released_finger_contact": bool(contacts.contact_flags[finger_index]),
                "A_position_palm_m": object_pose_in_palm(scene, scene.object_body_id)[0].tolist(),
                "wrist_qpos": scene.data.qpos[:2].tolist(),
                "qpos": scene.data.qpos.copy().tolist(), "qvel": scene.data.qvel.copy().tolist(),
            })
        if step in checkpoints:
            survival[str(step)] = bool(retained and contact_free)
    available = released_finger_available_motion(scene, finger)
    fixture_active = bool(scene.data.eq_active[scene.fixture_eq_id])
    valid = bool(
        retained_during_ramp and floor_free_during_ramp and retained
        and contact_free and available > 0.0 and not fixture_active
    )
    return {
        "executed": True, "finger": finger, "ramp_steps": int(ramp_steps),
        "retained_A_during_ramp": retained_during_ramp,
        "floor_free_during_ramp": floor_free_during_ramp,
        "retained_A": retained, "released_finger_contact_free": contact_free,
        "released_finger_available_motion_raw": available,
        "one_resource_recovered": valid, "survival": survival, "samples": samples,
        "fixture_active": fixture_active,
    }
