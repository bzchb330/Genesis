from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np
from scipy.stats import qmc

from .config import ROOT
from .phase3.config import FINGERS, SUPPORT_SURFACES, Phase3Config, load_phase3_config
from .phase3.contacts import ShadowContactState, extract_shadow_contacts, object_velocity
from .phase3.control import ContactAwareCloser, actuator_target_from_qpos
from .phase3.env import load_keyframe_qpos
from .phase3.model import ShadowScene, build_shadow_scene, set_fixture, set_object_pose
from .phase3.resource import compute_resource_snapshot
from .phase3.roles import FingerRole, RoleState


HORIZONS = (1, 5, 10, 25, 50, 100, 200, 300, 500, 750, 1000)
PHASE3B0_SEED = 330
DATA_SCHEMA_VERSION = 3
EXERCISED_RADIUS_M = 0.004
EXERCISED_OFFSETS_M = np.asarray(
    (
        (-0.004, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.004, 0.0, 0.0),
        (0.0, -0.004, 0.0),
        (0.0, 0.004, 0.0),
        (0.0, 0.0, -0.004),
        (0.0, 0.0, 0.004),
    ),
    dtype=np.float64,
)


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: int
    sampling_seed: int
    candidate_seed: int
    object_position_m: tuple[float, float, float]
    object_quaternion_wxyz: tuple[float, float, float, float]
    offset_m: tuple[float, float, float]
    sampling_domain: str = "phase3A_seven_position_convex_hull"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _artifact_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        resolved = resolved.relative_to(ROOT.resolve())
    except ValueError:
        pass
    return str(resolved).replace("\\", "/")


def _candidate_seed(seed: int, candidate_id: int) -> int:
    return int(np.random.SeedSequence([seed, candidate_id]).generate_state(1, dtype=np.uint32)[0])


def _sobol_point(candidate_id: int, seed: int) -> np.ndarray:
    sampler = qmc.Sobol(d=7, scramble=True, seed=seed)
    if candidate_id:
        sampler.fast_forward(candidate_id)
    return sampler.random(1)[0]


def sample_candidate(
    candidate_id: int,
    *,
    seed: int = PHASE3B0_SEED,
    config: Phase3Config | None = None,
) -> CandidateSpec:
    """Deterministically sample inside the convex hull exercised in Phase 3A.

    The first seven candidates exactly reproduce the Phase 3A cohort. Later
    candidates uniformly fill the L1 ball without expanding its support.
    """

    if candidate_id < 0:
        raise ValueError("candidate_id must be nonnegative")
    cfg = config or load_phase3_config()
    if candidate_id < len(EXERCISED_OFFSETS_M):
        offset = EXERCISED_OFFSETS_M[candidate_id].copy()
    else:
        unit = _sobol_point(candidate_id - len(EXERCISED_OFFSETS_M), seed)
        weights = -np.log(np.clip(unit[:3], np.finfo(np.float64).tiny, 1.0))
        weights /= weights.sum()
        radius = EXERCISED_RADIUS_M * float(unit[3]) ** (1.0 / 3.0)
        signs = np.where(unit[4:] < 0.5, -1.0, 1.0)
        offset = radius * weights * signs
    if float(np.abs(offset).sum()) > EXERCISED_RADIUS_M + 1e-15:
        raise AssertionError("candidate escaped the exercised Phase 3A convex hull")
    position = np.asarray(cfg.object["initial_pos"], dtype=np.float64) + offset
    quaternion = np.asarray(cfg.object["initial_quat"], dtype=np.float64)
    return CandidateSpec(
        candidate_id=candidate_id,
        sampling_seed=seed,
        candidate_seed=_candidate_seed(seed, candidate_id),
        object_position_m=tuple(float(value) for value in position),
        object_quaternion_wxyz=tuple(float(value) for value in quaternion),
        offset_m=tuple(float(value) for value in offset),
    )


def _prepare_acquisition(scene: ShadowScene, spec: CandidateSpec) -> ContactAwareCloser:
    mujoco.mj_resetData(scene.model, scene.data)
    pre_qpos = load_keyframe_qpos("pre grasp")
    scene.data.qpos[:24] = pre_qpos
    scene.data.qvel[:] = 0.0
    set_object_pose(scene, spec.object_position_m, spec.object_quaternion_wxyz)
    set_fixture(scene, True)
    pre_target = actuator_target_from_qpos(scene, pre_qpos)
    pinch_target = actuator_target_from_qpos(scene, load_keyframe_qpos("two finger pinch"))
    scene.data.ctrl[:] = pre_target
    mujoco.mj_forward(scene.model, scene.data)
    for _ in range(20):
        mujoco.mj_step(scene.model, scene.data)

    closer = ContactAwareCloser(scene, float(scene.config.diagnostic["contact_force_n"]))
    acquisition_ids = np.r_[scene.actuator_ids["thumb"], scene.actuator_ids["index"]]
    close_steps = int(scene.config.diagnostic["close_steps"])
    for close_step in range(close_steps):
        alpha = (close_step + 1) / close_steps
        proposed = pre_target.copy()
        proposed[acquisition_ids] = (
            (1.0 - alpha) * pre_target[acquisition_ids]
            + alpha * pinch_target[acquisition_ids]
        )
        scene.data.ctrl[:] = closer.limit_target(proposed)
        mujoco.mj_step(scene.model, scene.data)
    for _ in range(int(scene.config.diagnostic["settle_steps"])):
        mujoco.mj_step(scene.model, scene.data)
    # mj_step integrates qpos after its position-dependent pipeline. Synchronize
    # derived kinematics/contact data so the serialized release state reconstructs
    # exactly from qpos/qvel instead of preserving a one-step-old xpos/xquat view.
    mujoco.mj_forward(scene.model, scene.data)
    return closer


def _surface_for_record(scene: ShadowScene, geom1: str, geom2: str) -> str:
    names = {geom1, geom2}
    for finger in FINGERS:
        if names.intersection(scene.fingertip_geoms[finger]):
            return f"{finger}_tip"
    for surface in SUPPORT_SURFACES:
        if names.intersection(scene.collision_geoms[surface]):
            return surface
    floor_name = str(scene.config.raw["floor"]["name"])
    if floor_name in names:
        return "table"
    return "other"


def contact_records(scene: ShadowScene, contacts: ShadowContactState | None = None) -> list[dict[str, Any]]:
    contacts = contacts or extract_shadow_contacts(scene)
    output: list[dict[str, Any]] = []
    object_name = str(scene.config.object["name"])
    for contact_index, record in enumerate(contacts.records):
        contact = scene.data.contact[contact_index]
        if object_name not in {record.body1_name, record.body2_name}:
            continue
        surface = _surface_for_record(scene, record.geom1_name, record.geom2_name)
        output.append(
            {
                "surface": surface,
                "geom_pair": [record.geom1_name, record.geom2_name],
                "body_pair": [record.body1_name, record.body2_name],
                "position_m": record.position.copy(),
                "normal_world": record.normal.copy(),
                "distance_m": float(record.distance),
                "penetration_m": float(max(0.0, -record.distance)),
                "normal_force_n": float(record.normal_force),
                "tangential_force_n": float(record.tangential_force),
                "tangential_normal_ratio": (
                    float(record.tangential_force / record.normal_force)
                    if record.normal_force > 0.0
                    else 0.0
                ),
                "contact_dim": int(contact.dim),
                "friction": np.asarray(contact.friction, dtype=np.float64).copy(),
            }
        )
    return output


def pair_aware_penetration(records: Iterable[dict[str, Any]]) -> dict[str, float]:
    categories = {
        "thumb_object": 0.0,
        "index_object": 0.0,
        "palm_object": 0.0,
        "other_finger_object": 0.0,
        "table_object": 0.0,
        "other_object": 0.0,
    }
    for record in records:
        surface = str(record["surface"])
        penetration = float(record["penetration_m"])
        if surface == "thumb_tip":
            key = "thumb_object"
        elif surface == "index_tip":
            key = "index_object"
        elif surface == "palm":
            key = "palm_object"
        elif surface in {"middle_tip", "ring_tip", "little_tip", "middle", "ring", "little"}:
            key = "other_finger_object"
        elif surface == "table":
            key = "table_object"
        else:
            key = "other_object"
        categories[key] = max(categories[key], penetration)
    categories["maximum_intended_grip"] = max(
        categories["thumb_object"], categories["index_object"]
    )
    categories["maximum_gross_non_grip"] = max(
        value
        for key, value in categories.items()
        if key not in {"thumb_object", "index_object", "maximum_intended_grip"}
    )
    return categories


def _joint_margins(scene: ShadowScene) -> tuple[np.ndarray, np.ndarray]:
    ranges = scene.model.jnt_range[:24]
    qpos = scene.data.qpos[:24]
    absolute = np.minimum(qpos - ranges[:, 0], ranges[:, 1] - qpos)
    normalized = absolute / (ranges[:, 1] - ranges[:, 0])
    return absolute, normalized


def _quat_conjugate(quaternion: np.ndarray) -> np.ndarray:
    output = np.asarray(quaternion, dtype=np.float64).copy()
    output[1:] *= -1.0
    return output


def _quat_multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    output = np.empty(4, dtype=np.float64)
    mujoco.mju_mulQuat(output, np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64))
    return output


def palm_relative_pose(scene: ShadowScene) -> tuple[np.ndarray, np.ndarray]:
    palm_id = mujoco.mj_name2id(
        scene.model, mujoco.mjtObj.mjOBJ_BODY, scene.config.hand.palm_body
    )
    palm_position = scene.data.xpos[palm_id]
    palm_quaternion = scene.data.xquat[palm_id]
    rotation = np.asarray(scene.data.xmat[palm_id], dtype=np.float64).reshape(3, 3)
    relative_position = rotation.T @ (scene.data.xpos[scene.object_body_id] - palm_position)
    relative_quaternion = _quat_multiply(
        _quat_conjugate(palm_quaternion), scene.data.xquat[scene.object_body_id]
    )
    if relative_quaternion[0] < 0.0:
        relative_quaternion *= -1.0
    return relative_position, relative_quaternion


def quaternion_distance(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    cosine = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return float(2.0 * np.arccos(cosine))


def detect_contact_gaps(
    contact_flags: np.ndarray,
    world_positions: np.ndarray,
    palm_relative_positions: np.ndarray,
    linear_velocities: np.ndarray,
    timestep: float,
) -> list[dict[str, Any]]:
    """Return every complete hand-object contact gap without classifying it as failure."""

    flags = np.asarray(contact_flags)
    complete_gap = ~np.any(flags > 0.0, axis=1)
    gaps: list[dict[str, Any]] = []
    start: int | None = None
    for index, is_gap in enumerate(complete_gap):
        if is_gap and start is None:
            start = index
        if start is not None and (not is_gap or index == len(complete_gap) - 1):
            end = index if not is_gap else index + 1
            reestablished = end < len(complete_gap) and not bool(complete_gap[end])
            reestablishing = (
                [surface for surface, active in zip(SUPPORT_SURFACES, flags[end]) if active > 0.0]
                if reestablished
                else []
            )
            velocity_slice = np.linalg.norm(linear_velocities[start:end], axis=1)
            gaps.append(
                {
                    "start_step": start,
                    "end_step_exclusive": end,
                    "duration_steps": end - start,
                    "duration_s": (end - start) * float(timestep),
                    "reestablished": reestablished,
                    "reestablished_by": reestablishing,
                    "world_displacement_m": float(
                        np.linalg.norm(world_positions[end - 1] - world_positions[start])
                    ),
                    "palm_relative_displacement_m": float(
                        np.linalg.norm(
                            palm_relative_positions[end - 1] - palm_relative_positions[start]
                        )
                    ),
                    "maximum_object_speed_m_s": float(velocity_slice.max()) if len(velocity_slice) else 0.0,
                }
            )
            start = None
    return gaps


def _state_arrays(scene: ShadowScene) -> dict[str, np.ndarray]:
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(scene.model, state_spec), dtype=np.float64)
    mujoco.mj_getState(scene.model, scene.data, state, state_spec)
    return {
        "integration_state": state,
        "qpos": scene.data.qpos.copy(),
        "qvel": scene.data.qvel.copy(),
        "act": scene.data.act.copy(),
        "qacc_warmstart": scene.data.qacc_warmstart.copy(),
        "ctrl": scene.data.ctrl.copy(),
        "actuator_targets": scene.data.ctrl.copy(),
        "stiffness_scales": np.ones(len(SUPPORT_SURFACES), dtype=np.float64),
        "mocap_pos": scene.data.mocap_pos.copy(),
        "mocap_quat": scene.data.mocap_quat.copy(),
        "eq_active": scene.data.eq_active.copy(),
        "time": np.asarray([scene.data.time], dtype=np.float64),
    }


def state_hash(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        array = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def restore_release_state(scene: ShadowScene, path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        arrays = {key: stored[key].copy() for key in stored.files}
    mujoco.mj_resetData(scene.model, scene.data)
    mujoco.mj_setState(
        scene.model,
        scene.data,
        arrays["integration_state"],
        mujoco.mjtState.mjSTATE_INTEGRATION,
    )
    scene.data.ctrl[:] = arrays["ctrl"]
    scene.data.mocap_pos[:] = arrays["mocap_pos"]
    scene.data.mocap_quat[:] = arrays["mocap_quat"]
    scene.data.eq_active[:] = arrays["eq_active"]
    mujoco.mj_forward(scene.model, scene.data)
    return arrays


def _release_snapshot(
    scene: ShadowScene,
    spec: CandidateSpec,
    closer: ContactAwareCloser,
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
    contacts = extract_shadow_contacts(scene)
    records = contact_records(scene, contacts)
    penetration = pair_aware_penetration(records)
    absolute_margin, normalized_margin = _joint_margins(scene)
    relative_position, relative_quaternion = palm_relative_pose(scene)
    linear_velocity, angular_velocity = object_velocity(scene)
    palm_id = mujoco.mj_name2id(
        scene.model, mujoco.mjtObj.mjOBJ_BODY, scene.config.hand.palm_body
    )
    ctrl_lower = scene.model.actuator_ctrlrange[:, 0]
    ctrl_upper = scene.model.actuator_ctrlrange[:, 1]
    saturated = np.isclose(scene.data.ctrl, ctrl_lower) | np.isclose(scene.data.ctrl, ctrl_upper)
    state = _state_arrays(scene)
    intended_surfaces = {"thumb_tip", "index_tip"}
    positive_surfaces = {
        str(record["surface"])
        for record in records
        if float(record["normal_force_n"]) > 0.0
    }
    thumb_count = sum(record["surface"] == "thumb_tip" for record in records)
    index_count = sum(record["surface"] == "index_tip" for record in records)
    nonminimal = sorted(positive_surfaces - intended_surfaces)
    finite = all(np.all(np.isfinite(array)) for array in state.values())
    accepted = bool(thumb_count and index_count and not nonminimal and finite)
    if not finite:
        rejection = "NUMERIC_INVALID"
    elif not thumb_count or not index_count:
        rejection = "ACQUISITION_NOT_ESTABLISHED"
    elif nonminimal:
        rejection = "INVALID_INITIAL_COLLISION"
    else:
        rejection = None
    metadata = {
        "candidate": asdict(spec),
        "accepted_raw_release": accepted,
        "rejection_reason": rejection,
        "contact_latches": dict(closer.latched),
        "thumb_object_contact_count": int(thumb_count),
        "index_object_contact_count": int(index_count),
        "other_finger_object_contact_counts": {
            finger: int(sum(record["surface"] in {finger, f"{finger}_tip"} for record in records))
            for finger in ("middle", "ring", "little")
        },
        "palm_object_contact": bool(any(record["surface"] == "palm" for record in records)),
        "table_object_contact": bool(any(record["surface"] == "table" for record in records)),
        "gross_contact_surfaces": nonminimal,
        "penetration_m": penetration,
        "object_position_m": scene.data.xpos[scene.object_body_id].copy(),
        "object_quaternion_wxyz": scene.data.xquat[scene.object_body_id].copy(),
        "object_linear_velocity_m_s": linear_velocity,
        "object_angular_velocity_rad_s": angular_velocity,
        "object_palm_relative_position_m": relative_position,
        "object_palm_relative_quaternion_wxyz": relative_quaternion,
        "palm_position_m": scene.data.xpos[palm_id].copy(),
        "palm_quaternion_wxyz": scene.data.xquat[palm_id].copy(),
        "finger_joint_states_rad": {
            finger: scene.data.qpos[scene.model.jnt_qposadr[ids]].copy()
            for finger, ids in scene.joint_ids.items()
        },
        "wrist_state": {
            "qpos_rad": scene.data.qpos[
                scene.model.jnt_qposadr[
                    np.asarray(
                        [
                            mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                            for name in scene.config.hand.wrist_joints
                        ],
                        dtype=int,
                    )
                ]
            ].copy(),
            "qvel_rad_s": scene.data.qvel[
                scene.model.jnt_dofadr[
                    np.asarray(
                        [
                            mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                            for name in scene.config.hand.wrist_joints
                        ],
                        dtype=int,
                    )
                ]
            ].copy(),
        },
        "joint_margin_rad": absolute_margin,
        "joint_margin_normalized": normalized_margin,
        "minimum_joint_margin_rad": float(absolute_margin.min()),
        "minimum_joint_margin_normalized": float(normalized_margin.min()),
        "actuator_command": scene.data.ctrl.copy(),
        "actuator_command_magnitude": float(np.linalg.norm(scene.data.ctrl)),
        "actuator_command_rate": np.zeros(scene.model.nu, dtype=np.float64),
        "actuator_saturated": saturated,
        "actuator_saturation_count": int(saturated.sum()),
        "stiffness_scales": np.ones(len(SUPPORT_SURFACES), dtype=np.float64),
        "stiffness_scale_rate": np.zeros(len(SUPPORT_SURFACES), dtype=np.float64),
        "contacts": records,
        "state_hash": state_hash(state),
    }
    return metadata, state, records


def _retention_sample(scene: ShadowScene) -> dict[str, Any]:
    contacts = extract_shadow_contacts(scene)
    records = contact_records(scene, contacts)
    penetration = pair_aware_penetration(records)
    world_position = scene.data.xpos[scene.object_body_id].copy()
    world_quaternion = scene.data.xquat[scene.object_body_id].copy()
    relative_position, relative_quaternion = palm_relative_pose(scene)
    linear, angular = object_velocity(scene)
    absolute_margin, normalized_margin = _joint_margins(scene)
    ctrl_lower = scene.model.actuator_ctrlrange[:, 0]
    ctrl_upper = scene.model.actuator_ctrlrange[:, 1]
    saturation = np.isclose(scene.data.ctrl, ctrl_lower) | np.isclose(scene.data.ctrl, ctrl_upper)
    return {
        "world_position": world_position,
        "world_quaternion": world_quaternion,
        "palm_relative_position": relative_position,
        "palm_relative_quaternion": relative_quaternion,
        "linear_velocity": linear,
        "angular_velocity": angular,
        "contact_flags": contacts.contact_flags.copy(),
        "normal_forces": contacts.normal_forces.copy(),
        "tangential_forces": contacts.tangential_forces.copy(),
        "penetration": np.asarray(
            [
                penetration["thumb_object"],
                penetration["index_object"],
                penetration["palm_object"],
                penetration["other_finger_object"],
                penetration["table_object"],
                penetration["other_object"],
                penetration["maximum_intended_grip"],
                penetration["maximum_gross_non_grip"],
            ],
            dtype=np.float64,
        ),
        "joint_margin_rad": absolute_margin,
        "joint_margin_normalized": normalized_margin,
        "actuator_command": scene.data.ctrl.copy(),
        "actuator_saturation": saturation.astype(np.int8),
        "stiffness_scales": np.ones(len(SUPPORT_SURFACES), dtype=np.float64),
        "table_contact": np.asarray(
            [any(record["surface"] == "table" for record in records)], dtype=np.int8
        ),
        "finite": np.asarray(
            [
                np.all(np.isfinite(scene.data.qpos))
                and np.all(np.isfinite(scene.data.qvel))
                and np.all(np.isfinite(scene.data.ctrl))
            ],
            dtype=np.int8,
        ),
    }


def characterize_retention(scene: ShadowScene, steps: int = 1000) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    set_fixture(scene, False)
    samples = [_retention_sample(scene)]
    first_table_contact: int | None = None
    first_numeric_invalidity: int | None = None
    for step in range(1, steps + 1):
        mujoco.mj_step(scene.model, scene.data)
        mujoco.mj_forward(scene.model, scene.data)
        sample = _retention_sample(scene)
        samples.append(sample)
        if first_table_contact is None and bool(sample["table_contact"][0]):
            first_table_contact = step
        if first_numeric_invalidity is None and not bool(sample["finite"][0]):
            first_numeric_invalidity = step
        if first_table_contact is not None or first_numeric_invalidity is not None:
            break

    arrays = {
        key: np.stack([sample[key] for sample in samples])
        for key in samples[0]
    }
    count = len(samples) - 1
    arrays["step"] = np.arange(count + 1, dtype=np.int32)
    arrays["time_s"] = arrays["step"].astype(np.float64) * float(scene.model.opt.timestep)
    release_world = arrays["world_position"][0]
    release_relative = arrays["palm_relative_position"][0]
    release_world_quat = arrays["world_quaternion"][0]
    release_relative_quat = arrays["palm_relative_quaternion"][0]
    arrays["world_translation_from_release"] = np.linalg.norm(
        arrays["world_position"] - release_world, axis=1
    )
    arrays["palm_relative_translation_from_release"] = np.linalg.norm(
        arrays["palm_relative_position"] - release_relative, axis=1
    )
    arrays["world_rotation_from_release"] = np.asarray(
        [quaternion_distance(release_world_quat, value) for value in arrays["world_quaternion"]]
    )
    arrays["palm_relative_rotation_from_release"] = np.asarray(
        [quaternion_distance(release_relative_quat, value) for value in arrays["palm_relative_quaternion"]]
    )
    normal = arrays["normal_forces"]
    tangential = arrays["tangential_forces"]
    arrays["tangential_normal_ratio"] = np.divide(
        tangential,
        normal,
        out=np.zeros_like(tangential),
        where=normal > 0.0,
    )
    arrays["total_acquisition_normal_force"] = normal[:, :2].sum(axis=1)
    arrays["total_hand_normal_force"] = normal.sum(axis=1)
    arrays["actuator_command_rate"] = np.vstack(
        (np.zeros((1, scene.model.nu)), np.diff(arrays["actuator_command"], axis=0))
    )
    arrays["stiffness_scale_rate"] = np.vstack(
        (np.zeros((1, len(SUPPORT_SURFACES))), np.diff(arrays["stiffness_scales"], axis=0))
    )

    gaps = detect_contact_gaps(
        arrays["contact_flags"],
        arrays["world_position"],
        arrays["palm_relative_position"],
        arrays["linear_velocity"],
        float(scene.model.opt.timestep),
    )

    first_thumb_loss = next(
        (index for index in range(1, len(arrays["contact_flags"])) if arrays["contact_flags"][index, 0] == 0.0),
        None,
    )
    first_index_loss = next(
        (index for index in range(1, len(arrays["contact_flags"])) if arrays["contact_flags"][index, 1] == 0.0),
        None,
    )
    first_complete_gap = gaps[0]["start_step"] if gaps else None
    first_complete_loss = next(
        (gap["start_step"] for gap in gaps if not gap["reestablished"]),
        None,
    )
    saturation_any = np.any(arrays["actuator_saturation"] > 0, axis=1)
    maximum_saturation_run = 0
    current_saturation_run = 0
    for saturated in saturation_any:
        current_saturation_run = current_saturation_run + 1 if saturated else 0
        maximum_saturation_run = max(maximum_saturation_run, current_saturation_run)
    survived = {
        str(horizon): bool(
            count >= horizon
            and (first_table_contact is None or first_table_contact > horizon)
            and (first_numeric_invalidity is None or first_numeric_invalidity > horizon)
        )
        for horizon in HORIZONS
    }
    failure_labels = []
    if gaps:
        failure_labels.append("CONTACT_LOST_AND_RECOVERED" if any(gap["reestablished"] for gap in gaps) else "CONTACT_LOST")
    if first_table_contact is not None:
        failure_labels.extend(("OBJECT_SLIPPED", "OBJECT_TABLE_CONTACT"))
    if first_numeric_invalidity is not None:
        failure_labels.append("NUMERIC_INVALID")
    first_joint_limit = next(
        (
            index
            for index in range(len(arrays["joint_margin_rad"]))
            if np.any(arrays["joint_margin_rad"][index] < 0.0)
        ),
        None,
    )
    if first_joint_limit is not None:
        failure_labels.append("JOINT_LIMIT")
    first_saturation = next(
        (index for index, saturated in enumerate(saturation_any) if saturated),
        None,
    )
    if first_saturation is not None:
        failure_labels.append("ACTUATOR_SATURATION")
    if survived["1000"]:
        failure_labels.append("SURVIVED_1000")
    if not failure_labels:
        failure_labels.append("OTHER")
    summary = {
        "simulated_steps": count,
        "horizon_survival": survived,
        "first_thumb_contact_loss_step": first_thumb_loss,
        "first_index_contact_loss_step": first_index_loss,
        "first_complete_hand_object_gap_step": first_complete_gap,
        "contact_gaps": gaps,
        "maximum_contact_gap_steps": max((gap["duration_steps"] for gap in gaps), default=0),
        "contact_reestablished": bool(any(gap["reestablished"] for gap in gaps)),
        "first_table_contact_step": first_table_contact,
        "first_complete_object_loss_step": first_complete_loss,
        "first_workspace_exit_step": None,
        "workspace_exit_definition": "UNDEFINED_PI_THRESHOLD",
        "first_numeric_invalidity_step": first_numeric_invalidity,
        "first_joint_limit_step": first_joint_limit,
        "first_actuator_saturation_step": first_saturation,
        "large_translation_label": "UNDEFINED_PI_THRESHOLD",
        "large_rotation_label": "UNDEFINED_PI_THRESHOLD",
        "failure_labels": failure_labels,
        "maximum_consecutive_saturation_steps": maximum_saturation_run,
    }
    return summary, arrays


def _resource_metadata(scene: ShadowScene) -> dict[str, Any]:
    contacts = extract_shadow_contacts(scene)
    roles = RoleState()
    roles.begin_probe()
    roles.acquisition_contact()
    snapshot = compute_resource_snapshot(scene, contacts, roles)
    return {
        "free_finger_identity": [
            finger for finger, free in zip(FINGERS, snapshot.free_finger_mask) if free
        ],
        "free_finger_mask": snapshot.free_finger_mask,
        "free_finger_count": snapshot.n_free,
        "thumb_index_occupied": [
            bool(snapshot.fingers["thumb"].contact),
            bool(snapshot.fingers["index"].contact),
        ],
        "available_motion_range_rad": {
            finger: snapshot.fingers[finger].available_motion_range for finger in FINGERS
        },
        "local_workspace_m": {
            finger: snapshot.fingers[finger].local_reachable_workspace for finger in FINGERS
        },
        "joint_margin_normalized": {
            finger: snapshot.fingers[finger].joint_margin for finger in FINGERS
        },
    }


def evaluate_attempt(
    candidate_id: int,
    output_directory: str | Path,
    *,
    seed: int = PHASE3B0_SEED,
    retention_steps: int = 1000,
) -> dict[str, Any]:
    output = Path(output_directory)
    attempt_path = output / "sampling" / f"attempt_{candidate_id:05d}.json"
    if attempt_path.exists():
        existing = json.loads(attempt_path.read_text(encoding="utf-8"))
        if existing.get("data_schema_version") == DATA_SCHEMA_VERSION:
            return existing
    spec = sample_candidate(candidate_id, seed=seed)
    scene = build_shadow_scene()
    closer = _prepare_acquisition(scene, spec)
    release, state, _ = _release_snapshot(scene, spec, closer)
    payload: dict[str, Any] = {
        "data_schema_version": DATA_SCHEMA_VERSION,
        "candidate": asdict(spec),
        "accepted_raw_release": bool(release["accepted_raw_release"]),
        "rejection_reason": release["rejection_reason"],
        "release": release,
        "release_state_path": None,
        "retention_timeseries_path": None,
        "retention": None,
    }
    if release["accepted_raw_release"]:
        state_path = output / "release_states" / f"state_{candidate_id:05d}.npz"
        _atomic_npz(state_path, state)
        payload["release_state_path"] = _artifact_path(state_path)
        payload["release"]["resource"] = _resource_metadata(scene)
        retention, arrays = characterize_retention(scene, retention_steps)
        timeseries_path = output / "retention_timeseries" / f"retention_{candidate_id:05d}.npz"
        _atomic_npz(timeseries_path, arrays)
        payload["retention_timeseries_path"] = _artifact_path(timeseries_path)
        payload["retention"] = retention
    _atomic_json(attempt_path, payload)
    return _jsonable(payload)


def _evaluate_worker(arguments: tuple[int, str, int, int]) -> dict[str, Any]:
    candidate_id, output_directory, seed, retention_steps = arguments
    return evaluate_attempt(
        candidate_id,
        output_directory,
        seed=seed,
        retention_steps=retention_steps,
    )


def load_attempts(
    output_directory: str | Path,
    *,
    schema_version: int | None = DATA_SCHEMA_VERSION,
) -> list[dict[str, Any]]:
    sampling = Path(output_directory) / "sampling"
    if not sampling.exists():
        return []
    attempts = [json.loads(path.read_text(encoding="utf-8")) for path in sampling.glob("attempt_*.json")]
    if schema_version is not None:
        attempts = [row for row in attempts if row.get("data_schema_version") == schema_version]
    return sorted(attempts, key=lambda row: int(row["candidate"]["candidate_id"]))


def generate_dataset(
    output_directory: str | Path = ROOT / "outputs/phase3B0",
    *,
    target: int = 500,
    attempt_cap: int = 20_000,
    workers: int = 1,
    batch_size: int = 32,
    seed: int = PHASE3B0_SEED,
    retention_steps: int = 1000,
) -> dict[str, Any]:
    if not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    attempts = load_attempts(output)
    accepted = sum(bool(row["accepted_raw_release"]) for row in attempts)
    next_id = max((int(row["candidate"]["candidate_id"]) for row in attempts), default=-1) + 1
    batch_index = len(list((output / "sampling").glob("batch_*.json"))) if (output / "sampling").exists() else 0
    while accepted < target and next_id < attempt_cap:
        remaining = target - accepted
        count = min(batch_size, remaining, attempt_cap - next_id)
        identifiers = list(range(next_id, next_id + count))
        arguments = [(identifier, str(output), seed, retention_steps) for identifier in identifiers]
        if workers == 1:
            batch = [_evaluate_worker(argument) for argument in arguments]
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                batch = list(executor.map(_evaluate_worker, arguments))
        accepted += sum(bool(row["accepted_raw_release"]) for row in batch)
        next_id += count
        _atomic_json(
            output / "sampling" / f"batch_{batch_index:05d}.json",
            {
                "batch_index": batch_index,
                "candidate_ids": identifiers,
                "accepted_in_batch": sum(bool(row["accepted_raw_release"]) for row in batch),
                "accepted_total": accepted,
                "attempts_total": next_id,
            },
        )
        batch_index += 1
    attempts = load_attempts(output)
    raw = [row for row in attempts if row["accepted_raw_release"]]
    cohort = raw[:target]
    manifest = {
        "phase": "Phase 3B-0",
        "data_schema_version": DATA_SCHEMA_VERSION,
        "sampling_seed": seed,
        "attempt_cap": attempt_cap,
        "target": target,
        "total_attempts": len(attempts),
        "raw_valid_release_states": len(raw),
        "target_reached": len(cohort) >= target,
        "cohort_candidate_ids": [row["candidate"]["candidate_id"] for row in cohort],
        "sampling_domain": "convex hull of the seven physically exercised Phase 3A positions",
        "sampled_dimensions": ["object_world_x", "object_world_y", "object_world_z"],
        "fixed_dimensions": [
            "object_orientation",
            "wrist_configuration",
            "thumb_approach",
            "index_approach",
            "controller_timing",
            "physics",
        ],
    }
    _atomic_json(output / "raw_manifest.json", manifest)
    return manifest
