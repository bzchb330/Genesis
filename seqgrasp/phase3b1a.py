from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from .config import ROOT
from .phase3.config import FINGERS, SUPPORT_SURFACES, load_phase3_config
from .phase3.contacts import extract_shadow_contacts, fingertip_object_penetration, object_velocity
from .phase3.control import ContactAwareCloser, actuator_target_from_qpos
from .phase3.env import load_keyframe_qpos
from .phase3.experiments import run_handoff_diagnostic
from .phase3.model import ShadowScene, build_shadow_scene, set_fixture, set_object_pose
from .phase3b0 import (
    CandidateSpec,
    _atomic_json,
    _atomic_npz,
    _jsonable,
    _joint_margins,
    _release_snapshot,
    characterize_retention,
)
from .phase3b0_analysis import assign_pose_regions, frozen_split, release_descriptor
from .phase3b05 import FeasibilityCandidate, sample_feasibility_candidate


PHASE3B1A_SEED = 331_000
DATA_SCHEMA_VERSION = 1
PILOT_PENETRATION_CEILING_M = 0.003
PILOT_STIFFNESS_RANGE = (0.75, 1.0)
PILOT_TARGET_DELTA_RAD = 0.0005
PILOT_RATE_CAP_RAD_PER_CONTROL_STEP = 0.0005
PILOT_EPISODE_STEPS = 1000
PILOT_FRAME_SKIP = 1
PILOT_TRAINING_SEEDS = (33101, 33102, 33103)


@dataclass(frozen=True)
class ProjectionResult:
    requested_qpos: tuple[float, ...]
    projected_qpos: tuple[float, ...]
    difference_rad: tuple[float, ...]
    l2_magnitude_rad: float
    maximum_absolute_change_rad: float
    original_violating_joints: tuple[str, ...]
    projected_minimum_joint_margin_rad: float
    tendon_constraint_margins_rad: dict[str, tuple[float, float]]
    optimizer_success: bool
    optimizer_message: str


def _joint_names(scene: ShadowScene) -> list[str]:
    return [
        mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_JOINT, index) or f"joint_{index}"
        for index in range(24)
    ]


def _tendon_constraint_matrix(scene: ShadowScene) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    rows, lower, upper, names = [], [], [], []
    for actuator_id in range(scene.model.nu):
        if int(scene.model.actuator_trntype[actuator_id]) != int(mujoco.mjtTrn.mjTRN_TENDON):
            continue
        tendon_id = int(scene.model.actuator_trnid[actuator_id, 0])
        tendon_name = mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_TENDON, tendon_id)
        if tendon_name is None:
            raise ValueError(f"missing name for tendon {tendon_id}")
        prefix = tendon_name[:-1]
        row = np.zeros(24, dtype=np.float64)
        for suffix in ("2", "1"):
            joint_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{suffix}")
            if joint_id < 0:
                raise ValueError(f"cannot resolve fixed-tendon joint {prefix}{suffix}")
            row[joint_id] = 1.0
        rows.append(row)
        lower.append(float(scene.model.actuator_ctrlrange[actuator_id, 0]))
        upper.append(float(scene.model.actuator_ctrlrange[actuator_id, 1]))
        names.append(tendon_name)
    return np.asarray(rows), np.asarray(lower), np.asarray(upper), names


def project_feasible_hand_qpos(
    requested_qpos: np.ndarray,
    scene: ShadowScene | None = None,
) -> ProjectionResult:
    """Minimum-L2 projection onto joint and fixed-tendon feasible constraints."""

    scene = scene or build_shadow_scene()
    requested = np.asarray(requested_qpos, dtype=np.float64)
    if requested.shape != (24,):
        raise ValueError("Shadow feasible projection requires exactly 24 hand coordinates")
    ranges = scene.model.jnt_range[:24].copy()
    matrix, tendon_lower, tendon_upper, tendon_names = _tendon_constraint_matrix(scene)
    initial = np.clip(requested, ranges[:, 0], ranges[:, 1])
    constraints = [LinearConstraint(matrix, tendon_lower, tendon_upper)] if len(matrix) else []
    result = minimize(
        lambda value: 0.5 * float(np.dot(value - requested, value - requested)),
        initial,
        jac=lambda value: value - requested,
        method="SLSQP",
        bounds=Bounds(ranges[:, 0], ranges[:, 1]),
        constraints=constraints,
        options={"ftol": 1e-14, "maxiter": 500, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"feasible pre-grasp projection failed: {result.message}")
    projected = np.asarray(result.x, dtype=np.float64)
    joint_margin = np.minimum(projected - ranges[:, 0], ranges[:, 1] - projected)
    tendon_values = matrix @ projected if len(matrix) else np.empty(0)
    tendon_margins = {
        name: (float(value - low), float(high - value))
        for name, value, low, high in zip(tendon_names, tendon_values, tendon_lower, tendon_upper)
    }
    if float(joint_margin.min()) < -1e-10 or any(min(value) < -1e-10 for value in tendon_margins.values()):
        raise RuntimeError("optimizer returned a state outside compiled joint/tendon constraints")
    names = _joint_names(scene)
    original_margin = np.minimum(requested - ranges[:, 0], ranges[:, 1] - requested)
    difference = projected - requested
    return ProjectionResult(
        requested_qpos=tuple(float(value) for value in requested),
        projected_qpos=tuple(float(value) for value in projected),
        difference_rad=tuple(float(value) for value in difference),
        l2_magnitude_rad=float(np.linalg.norm(difference)),
        maximum_absolute_change_rad=float(np.max(np.abs(difference))),
        original_violating_joints=tuple(names[index] for index in np.flatnonzero(original_margin < 0.0)),
        projected_minimum_joint_margin_rad=float(joint_margin.min()),
        tendon_constraint_margins_rad=tendon_margins,
        optimizer_success=True,
        optimizer_message=str(result.message),
    )


def projected_keyframe(name: str, scene: ShadowScene | None = None) -> tuple[np.ndarray, ProjectionResult]:
    scene = scene or build_shadow_scene()
    result = project_feasible_hand_qpos(load_keyframe_qpos(name), scene)
    return np.asarray(result.projected_qpos, dtype=np.float64), result


def _project_with_wrist(
    scene: ShadowScene,
    keyframe_name: str,
    wrist_qpos: tuple[float, float],
) -> tuple[np.ndarray, ProjectionResult]:
    requested = load_keyframe_qpos(keyframe_name).copy()
    wrist_ids = np.asarray(
        [
            mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in scene.config.hand.wrist_joints
        ],
        dtype=int,
    )
    requested[wrist_ids] = wrist_qpos
    result = project_feasible_hand_qpos(requested, scene)
    return np.asarray(result.projected_qpos), result


def prepare_projected_acquisition(
    scene: ShadowScene,
    candidate: FeasibilityCandidate,
) -> tuple[ContactAwareCloser, dict[str, Any]]:
    mujoco.mj_resetData(scene.model, scene.data)
    pre_qpos, pre_projection = _project_with_wrist(
        scene, "pre grasp", candidate.wrist_initial_qpos_rad
    )
    pinch_qpos, pinch_projection = _project_with_wrist(
        scene, "two finger pinch", candidate.wrist_initial_qpos_rad
    )
    scene.data.qpos[:24] = pre_qpos
    scene.data.qvel[:] = 0.0
    set_object_pose(scene, candidate.object_position_m, candidate.object_quaternion_wxyz)
    set_fixture(scene, True)
    pre_target = actuator_target_from_qpos(scene, pre_qpos)
    pinch_target = actuator_target_from_qpos(scene, pinch_qpos)
    scene.data.ctrl[:] = pre_target
    mujoco.mj_forward(scene.model, scene.data)
    initial_margin, _ = _joint_margins(scene)
    initial_target_bound = np.isclose(pre_target, scene.model.actuator_ctrlrange[:, 0]) | np.isclose(
        pre_target, scene.model.actuator_ctrlrange[:, 1]
    )
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
    mujoco.mj_forward(scene.model, scene.data)
    return closer, {
        "pregrasp_projection": asdict(pre_projection),
        "pinch_projection": asdict(pinch_projection),
        "initial_minimum_joint_margin_rad": float(initial_margin.min()),
        "initial_command_bound_actuators": [
            mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
            for index in np.flatnonzero(initial_target_bound)
        ],
    }


def evaluate_sanitized_candidate(
    candidate_id: int,
    output_directory: str | Path = ROOT / "outputs/phase3B1A",
    *,
    seed: int = PHASE3B1A_SEED,
    retention_steps: int = 250,
) -> dict[str, Any]:
    output = Path(output_directory)
    path = output / "resets" / "attempts" / f"attempt_{candidate_id:05d}.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("data_schema_version") == DATA_SCHEMA_VERSION:
            return existing
    candidate = sample_feasibility_candidate(1, candidate_id, seed=seed)
    scene = build_shadow_scene()
    closer, projection = prepare_projected_acquisition(scene, candidate)
    compatibility = CandidateSpec(
        candidate_id=candidate_id,
        sampling_seed=seed,
        candidate_seed=candidate.candidate_seed,
        object_position_m=candidate.object_position_m,
        object_quaternion_wxyz=candidate.object_quaternion_wxyz,
        offset_m=candidate.object_offset_m,
        sampling_domain="PHASE 3B-1A CONSERVATIVE PILOT ENGINEERING BOUNDS",
    )
    release, state, _ = _release_snapshot(scene, compatibility, closer)
    release["candidate"] = asdict(candidate)
    intended = float(release["penetration_m"]["maximum_intended_grip"])
    gross = float(release["penetration_m"]["maximum_gross_non_grip"])
    valid = bool(
        release["accepted_raw_release"]
        and intended <= PILOT_PENETRATION_CEILING_M
        and gross == 0.0
        and projection["initial_minimum_joint_margin_rad"] >= -1e-10
    )
    rejection = release["rejection_reason"]
    if release["accepted_raw_release"] and intended > PILOT_PENETRATION_CEILING_M:
        rejection = "INTENDED_PENETRATION_PILOT_CEILING"
    elif release["accepted_raw_release"] and gross > 0.0:
        rejection = "GROSS_NON_GRIP_COLLISION"
    elif release["accepted_raw_release"] and projection["initial_minimum_joint_margin_rad"] < -1e-10:
        rejection = "INFEASIBLE_PROJECTED_INITIALIZATION"
    payload: dict[str, Any] = {
        "data_schema_version": DATA_SCHEMA_VERSION,
        "candidate": asdict(candidate),
        "valid_sanitized_release": valid,
        "rejection_reason": rejection,
        "projection": projection,
        "release": release,
        "release_state_path": None,
        "retention": None,
        "retained_250": False,
    }
    if valid:
        state_path = output / "resets" / "states" / f"state_{candidate_id:05d}.npz"
        _atomic_npz(state_path, state)
        payload["release_state_path"] = str(state_path.relative_to(ROOT)).replace("\\", "/")
        retention, arrays = characterize_retention(scene, retention_steps)
        series_path = output / "resets" / "retention" / f"retention_{candidate_id:05d}.npz"
        _atomic_npz(series_path, arrays)
        payload["retention_timeseries_path"] = str(series_path.relative_to(ROOT)).replace("\\", "/")
        payload["retention"] = retention
        table = retention["first_table_contact_step"]
        numeric = retention["first_numeric_invalidity_step"]
        payload["retained_250"] = bool(
            retention["simulated_steps"] >= retention_steps
            and (table is None or table > retention_steps)
            and (numeric is None or numeric > retention_steps)
        )
    _atomic_json(path, payload)
    return _jsonable(payload)


def _sanitized_worker(arguments: tuple[int, str, int, int]) -> dict[str, Any]:
    return evaluate_sanitized_candidate(
        arguments[0], arguments[1], seed=arguments[2], retention_steps=arguments[3]
    )


def generate_sanitized_dataset(
    target: int = 500,
    *,
    attempt_cap: int = 1000,
    workers: int = 1,
    output_directory: str | Path = ROOT / "outputs/phase3B1A",
) -> dict[str, Any]:
    output = Path(output_directory)
    valid: list[dict[str, Any]] = []
    attempted = 0
    batch_size = max(16, workers * 4)
    while len(valid) < target and attempted < attempt_cap:
        end = min(attempted + batch_size, attempt_cap)
        arguments = [
            (candidate_id, str(output), PHASE3B1A_SEED, 250)
            for candidate_id in range(attempted, end)
        ]
        if workers == 1:
            rows = [_sanitized_worker(value) for value in arguments]
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                rows = list(executor.map(_sanitized_worker, arguments, chunksize=4))
        valid.extend(row for row in rows if row["valid_sanitized_release"])
        attempted = end
    valid = valid[:target]
    if len(valid) < target:
        raise RuntimeError(f"sanitized reset target not reached: {len(valid)}/{target}")
    descriptors, labels = release_descriptor(
        [
            {
                "candidate": row["candidate"],
                "release": row["release"],
                "accepted_raw_release": True,
            }
            for row in valid
        ]
    )
    regions, region_metadata = assign_pose_regions(descriptors, count=10)
    compatible = [
        {
            "candidate": row["candidate"],
            "release": row["release"],
            "accepted_raw_release": True,
        }
        for row in valid
    ]
    split = frozen_split(compatible, descriptors, regions)
    lookup = {int(row["candidate"]["candidate_id"]): row for row in valid}
    split_payload = {
        "status": "FROZEN BEFORE PPO TRAINING",
        "sampling_seed": PHASE3B1A_SEED,
        "descriptor_labels": labels,
        "pose_regions": region_metadata,
        "train": split["train"],
        "validation": split["validation"],
        "test": split["test"],
        "zero_id_overlap": split["zero_id_overlap"],
        "zero_state_hash_overlap": split["zero_state_hash_overlap"],
        "state_paths": {
            name: [lookup[candidate_id]["release_state_path"] for candidate_id in split[name]]
            for name in ("train", "validation", "test")
        },
    }
    _atomic_json(output / "resets" / "split.json", split_payload)
    manifest = {
        "data_schema_version": DATA_SCHEMA_VERSION,
        "target": target,
        "attempted": attempted,
        "valid_count": len(valid),
        "candidate_ids": [int(row["candidate"]["candidate_id"]) for row in valid],
        "counts": {name: len(split_payload[name]) for name in ("train", "validation", "test")},
        "split_path": "outputs/phase3B1A/resets/split.json",
        "bounds_status": "PILOT ENGINEERING BOUNDS",
    }
    _atomic_json(output / "resets" / "manifest.json", manifest)
    return manifest


def _projected_handoff_once() -> dict[str, Any]:
    cfg = load_phase3_config()
    scene = build_shadow_scene(cfg)
    nominal = sample_feasibility_candidate(0, 1, seed=PHASE3B1A_SEED)
    nominal = FeasibilityCandidate(
        **{
            **asdict(nominal),
            "object_position_m": tuple(float(value) for value in cfg.diagnostic["handoff_initial_pos"]),
            "object_offset_m": tuple(
                float(value)
                for value in np.asarray(cfg.diagnostic["handoff_initial_pos"]) - np.asarray(cfg.object["initial_pos"])
            ),
        }
    )
    closer, projection = prepare_projected_acquisition(scene, nominal)
    del closer
    set_fixture(scene, False)
    for _ in range(50):
        mujoco.mj_step(scene.model, scene.data)
    support_target, _ = projected_keyframe(str(cfg.diagnostic["handoff_support_keyframe"]), scene)
    support_target = actuator_target_from_qpos(scene, support_target)

    def move(group: str, target: np.ndarray, step: float) -> None:
        ids = scene.actuator_ids[group]
        scene.data.ctrl[ids] += np.clip(target[ids] - scene.data.ctrl[ids], -step, step)

    reference_penetration = float(cfg.diagnostic["reference_penetration_m"])
    for _ in range(350):
        for finger in ("thumb", "index"):
            if fingertip_object_penetration(scene, finger) < reference_penetration:
                move(finger, support_target, 0.0005)
        move("wrist", support_target, 0.0002)
        move("middle", support_target, 0.0005)
        mujoco.mj_step(scene.model, scene.data)
    for _ in range(350):
        for finger in ("thumb", "index"):
            if fingertip_object_penetration(scene, finger) < reference_penetration:
                move(finger, support_target, 0.0005)
        move("wrist", support_target, 0.0002)
        move("ring", support_target, 0.0005)
        move("little", support_target, 0.0005)
        mujoco.mj_step(scene.model, scene.data)
    open_qpos, _ = projected_keyframe("open hand", scene)
    open_target = actuator_target_from_qpos(scene, open_qpos)
    release_ids = scene.actuator_ids["thumb"]
    start = scene.data.ctrl.copy()
    for index in range(300):
        alpha = (index + 1) / 300
        scene.data.ctrl[release_ids] = (1.0 - alpha) * start[release_ids] + alpha * open_target[release_ids]
        mujoco.mj_step(scene.model, scene.data)
    for _ in range(150):
        mujoco.mj_step(scene.model, scene.data)
    contacts = extract_shadow_contacts(scene)
    object_name = str(cfg.object["name"])
    floor_name = str(cfg.raw["floor"]["name"])
    floor = any(
        object_name in {record.body1_name, record.body2_name}
        and floor_name in {record.geom1_name, record.geom2_name}
        for record in contacts.records
    )
    alternate = bool(np.any(contacts.contact_flags[2:]))
    thumb_free = not bool(contacts.contact_flags[0])
    ids = scene.joint_ids["thumb"]
    qpos = scene.data.qpos[scene.model.jnt_qposadr[ids]]
    limits = scene.model.jnt_range[ids]
    motion = float(np.sum(np.maximum(0.0, np.minimum(qpos - limits[:, 0], limits[:, 1] - qpos))))
    margin, _ = _joint_margins(scene)
    return {
        "resource_recovered_diagnostic": bool(thumb_free and alternate and not floor and motion > 0.0),
        "thumb_released": thumb_free,
        "alternate_support": alternate,
        "floor_contact": floor,
        "available_motion_rad": motion,
        "minimum_final_joint_margin_rad": float(margin.min()),
        "projection": projection,
    }


def run_projection_revalidation(
    output_directory: str | Path = ROOT / "outputs/phase3B1A",
) -> dict[str, Any]:
    output = Path(output_directory)
    scene = build_shadow_scene()
    _, projection = projected_keyframe("pre grasp", scene)
    original_handoff = run_handoff_diagnostic()["summary"]
    projected_handoff = _projected_handoff_once()
    baseline_paths = sorted((ROOT / "outputs/phase3B05/feasibility/level_1").glob("candidate_*.json"))[:50]
    baseline = [json.loads(path.read_text(encoding="utf-8")) for path in baseline_paths]
    if len(baseline) != 50:
        raise RuntimeError("Phase 3B-0.5 conservative baseline does not contain 50 states")
    projected = [evaluate_sanitized_candidate(index, output, retention_steps=250) for index in range(50)]

    def summarize(rows: list[dict[str, Any]], validity_key: str) -> dict[str, Any]:
        valid = [row for row in rows if row[validity_key]]
        penetrations = np.asarray(
            [row["release"]["penetration_m"]["maximum_intended_grip"] for row in valid], dtype=np.float64
        )
        margins = np.asarray([row["release"]["minimum_joint_margin_rad"] for row in valid], dtype=np.float64)
        positions = np.asarray([row["release"]["object_position_m"] for row in valid], dtype=np.float64)
        return {
            "attempted": len(rows),
            "valid_count": len(valid),
            "valid_fraction": len(valid) / len(rows),
            "retained_250_count": int(sum(bool(row["retained_250"]) for row in valid)),
            "retained_250_fraction": float(np.mean([row["retained_250"] for row in valid])) if valid else 0.0,
            "release_penetration_median": float(np.median(penetrations)) if len(penetrations) else float("nan"),
            "release_penetration_maximum": float(np.max(penetrations)) if len(penetrations) else float("nan"),
            "minimum_release_joint_margin_rad": float(np.min(margins)) if len(margins) else float("nan"),
            "release_contact_topologies": {
                "thumb_index_only": int(
                    sum(not row["release"]["gross_contact_surfaces"] for row in valid)
                )
            },
            "object_position_mean_m": positions.mean(axis=0).tolist() if len(positions) else [],
        }

    before = summarize(baseline, "accepted_raw_release")
    after = summarize(projected, "valid_sanitized_release")
    criteria = {
        "projected_phase3a_resource_recovered": bool(projected_handoff["resource_recovered_diagnostic"]),
        "acquisition_fraction_drop_at_most_0.10": after["valid_fraction"] >= before["valid_fraction"] - 0.10,
        "retention_fraction_drop_at_most_0.15": after["retained_250_fraction"] >= before["retained_250_fraction"] - 0.15,
        "maximum_intended_penetration_below_3mm": after["release_penetration_maximum"] <= PILOT_PENETRATION_CEILING_M,
        "all_projected_initializations_feasible": all(
            row["projection"]["initial_minimum_joint_margin_rad"] >= -1e-10 for row in projected
        ),
    }
    payload = {
        "pregrasp_projection": asdict(projection),
        "original_phase3a": {
            "resource_recovered_diagnostic": original_handoff["resource_recovered_diagnostic"],
            "alternate_support_present": original_handoff["alternate_support_present"],
            "configured_release_fingers_released": original_handoff["configured_release_fingers_released"],
        },
        "projected_phase3a": projected_handoff,
        "before_projection_50": before,
        "after_projection_50": after,
        "engineering_revalidation_criteria": criteria,
        "passed": all(criteria.values()),
        "failure_token": None if all(criteria.values()) else "PHASE3B1A_PROJECTION_REVALIDATION_FAILED",
        "physics_modified": False,
        "vendored_assets_modified": False,
    }
    _atomic_json(output / "projection" / "revalidation.json", payload)
    return payload
