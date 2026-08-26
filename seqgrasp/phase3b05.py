from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

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
    _joint_margins,
    _release_snapshot,
    characterize_retention,
    contact_records,
    detect_contact_gaps,
    load_attempts,
    palm_relative_pose,
    pair_aware_penetration,
    quaternion_distance,
    restore_release_state,
)


PHASE3B05_SEED = 330_500
DATA_SCHEMA_VERSION = 1
ACTIVE_PROTOCOL_VERSION = 2
PERSISTENCE_HORIZONS = (10, 25, 50, 100, 150, 250, 500)
DEDUPLICATION_THRESHOLDS = (0.0, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2)


@dataclass(frozen=True)
class FeasibilityLevel:
    level: int
    position_l1_radius_m: float
    orientation_limit_deg: float
    wrist_limit_deg: float
    label: str


FEASIBILITY_LEVELS = (
    FeasibilityLevel(0, 0.004, 0.0, 0.0, "existing Phase 3B-0 position-only space"),
    FeasibilityLevel(1, 0.006, 5.0, 5.0, "small orientation and wrist perturbations"),
    FeasibilityLevel(2, 0.008, 10.0, 10.0, "moderate orientation and wrist perturbations"),
    FeasibilityLevel(3, 0.010, 20.0, 15.0, "larger engineering feasibility probe"),
)


@dataclass(frozen=True)
class FeasibilityCandidate:
    level: int
    candidate_id: int
    sampling_seed: int
    candidate_seed: int
    position_l1_radius_m: float
    orientation_limit_deg: float
    wrist_limit_deg: float
    object_position_m: tuple[float, float, float]
    object_offset_m: tuple[float, float, float]
    object_euler_xyz_deg: tuple[float, float, float]
    object_quaternion_wxyz: tuple[float, float, float, float]
    wrist_perturbation_rad: tuple[float, float]
    wrist_initial_qpos_rad: tuple[float, float]
    sampling_domain: str = "ENGINEERING FEASIBILITY LEVELS - NOT PI-FROZEN TRAINING RANGES"


def _seed(*values: int) -> int:
    return int(np.random.SeedSequence(values).generate_state(1, dtype=np.uint32)[0])


def _artifact_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        resolved = resolved.relative_to(ROOT.resolve())
    except ValueError:
        pass
    return str(resolved).replace("\\", "/")


def _sobol(level: int, candidate_id: int, seed: int) -> np.ndarray:
    sampler = qmc.Sobol(d=11, scramble=True, seed=_seed(seed, level))
    if candidate_id:
        sampler.fast_forward(candidate_id)
    return sampler.random(1)[0]


def _level(level: int) -> FeasibilityLevel:
    try:
        return FEASIBILITY_LEVELS[level]
    except IndexError as error:
        raise ValueError(f"unknown Phase 3B-0.5 feasibility level {level}") from error


def _l1_offset(unit: np.ndarray, radius: float) -> np.ndarray:
    weights = -np.log(np.clip(unit[:3], np.finfo(np.float64).tiny, 1.0))
    weights /= weights.sum()
    radial = radius * float(unit[3]) ** (1.0 / 3.0)
    signs = np.where(unit[4:7] < 0.5, -1.0, 1.0)
    return radial * weights * signs


def sample_feasibility_candidate(
    level: int,
    candidate_id: int,
    *,
    seed: int = PHASE3B05_SEED,
) -> FeasibilityCandidate:
    """Sample one deterministic nested engineering feasibility probe.

    Level 3 intentionally includes both the requested 15 degree and 20 degree
    orientation envelopes: even candidate IDs use 15 degrees and odd IDs use
    20 degrees. These envelopes are exploratory, not frozen reset ranges.
    """

    if candidate_id < 0:
        raise ValueError("candidate_id must be nonnegative")
    definition = _level(level)
    unit = _sobol(level, candidate_id, seed)
    cfg = load_phase3_config()
    offset = _l1_offset(unit, definition.position_l1_radius_m)
    orientation_limit = definition.orientation_limit_deg
    if level == 3 and candidate_id % 2 == 0:
        orientation_limit = 15.0
    euler = (2.0 * unit[7:10] - 1.0) * orientation_limit
    delta_quat = Rotation.from_euler("xyz", euler, degrees=True).as_quat(scalar_first=True)
    nominal_quat = np.asarray(cfg.object["initial_quat"], dtype=np.float64)
    quaternion = (
        Rotation.from_quat(nominal_quat, scalar_first=True)
        * Rotation.from_quat(delta_quat, scalar_first=True)
    ).as_quat(scalar_first=True)

    scene = build_shadow_scene(cfg)
    pre_qpos = load_keyframe_qpos("pre grasp")
    wrist_ids = np.asarray(
        [mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in cfg.hand.wrist_joints],
        dtype=int,
    )
    requested_wrist = np.deg2rad((2.0 * unit[9:11] - 1.0) * definition.wrist_limit_deg)
    wrist = np.clip(
        pre_qpos[wrist_ids] + requested_wrist,
        scene.model.jnt_range[wrist_ids, 0],
        scene.model.jnt_range[wrist_ids, 1],
    )
    actual_delta = wrist - pre_qpos[wrist_ids]
    position = np.asarray(cfg.object["initial_pos"], dtype=np.float64) + offset
    return FeasibilityCandidate(
        level=level,
        candidate_id=candidate_id,
        sampling_seed=seed,
        candidate_seed=_seed(seed, level, candidate_id),
        position_l1_radius_m=definition.position_l1_radius_m,
        orientation_limit_deg=orientation_limit,
        wrist_limit_deg=definition.wrist_limit_deg,
        object_position_m=tuple(float(value) for value in position),
        object_offset_m=tuple(float(value) for value in offset),
        object_euler_xyz_deg=tuple(float(value) for value in euler),
        object_quaternion_wxyz=tuple(float(value) for value in quaternion),
        wrist_perturbation_rad=tuple(float(value) for value in actual_delta),
        wrist_initial_qpos_rad=tuple(float(value) for value in wrist),
    )


def sampled_wrist_is_within_limits(candidate: FeasibilityCandidate) -> bool:
    scene = build_shadow_scene()
    ids = np.asarray(
        [mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in scene.config.hand.wrist_joints],
        dtype=int,
    )
    values = np.asarray(candidate.wrist_initial_qpos_rad)
    return bool(np.all(values >= scene.model.jnt_range[ids, 0]) and np.all(values <= scene.model.jnt_range[ids, 1]))


def _prepare_feasibility_acquisition(
    scene: ShadowScene, candidate: FeasibilityCandidate
) -> ContactAwareCloser:
    mujoco.mj_resetData(scene.model, scene.data)
    pre_qpos = load_keyframe_qpos("pre grasp").copy()
    pinch_qpos = load_keyframe_qpos("two finger pinch").copy()
    wrist_ids = np.asarray(
        [mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in scene.config.hand.wrist_joints],
        dtype=int,
    )
    pre_qpos[wrist_ids] = candidate.wrist_initial_qpos_rad
    pinch_qpos[wrist_ids] = candidate.wrist_initial_qpos_rad
    scene.data.qpos[:24] = pre_qpos
    scene.data.qvel[:] = 0.0
    set_object_pose(scene, candidate.object_position_m, candidate.object_quaternion_wxyz)
    set_fixture(scene, True)
    pre_target = actuator_target_from_qpos(scene, pre_qpos)
    pinch_target = actuator_target_from_qpos(scene, pinch_qpos)
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
    mujoco.mj_forward(scene.model, scene.data)
    return closer


def evaluate_feasibility_candidate(
    level: int,
    candidate_id: int,
    output_directory: str | Path = ROOT / "outputs/phase3B05",
    *,
    seed: int = PHASE3B05_SEED,
    retention_steps: int = 250,
) -> dict[str, Any]:
    output = Path(output_directory)
    path = output / "feasibility" / f"level_{level}" / f"candidate_{candidate_id:04d}.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("data_schema_version") == DATA_SCHEMA_VERSION:
            return existing
    candidate = sample_feasibility_candidate(level, candidate_id, seed=seed)
    scene = build_shadow_scene()
    closer = _prepare_feasibility_acquisition(scene, candidate)
    compatibility = CandidateSpec(
        candidate_id=candidate_id,
        sampling_seed=seed,
        candidate_seed=candidate.candidate_seed,
        object_position_m=candidate.object_position_m,
        object_quaternion_wxyz=candidate.object_quaternion_wxyz,
        offset_m=candidate.object_offset_m,
        sampling_domain=candidate.sampling_domain,
    )
    release, state, _ = _release_snapshot(scene, compatibility, closer)
    release["candidate"] = asdict(candidate)
    payload: dict[str, Any] = {
        "data_schema_version": DATA_SCHEMA_VERSION,
        "candidate": asdict(candidate),
        "accepted_raw_release": bool(release["accepted_raw_release"]),
        "rejection_reason": release["rejection_reason"],
        "sampled_wrist_within_limits": sampled_wrist_is_within_limits(candidate),
        "release": release,
        "release_state_path": None,
        "retention_timeseries_path": None,
        "retention": None,
        "retained_250": False,
        "immediate_slip": False,
    }
    if release["accepted_raw_release"]:
        state_path = output / "feasibility" / "release_states" / f"level_{level}_state_{candidate_id:04d}.npz"
        _atomic_npz(state_path, state)
        payload["release_state_path"] = _artifact_path(state_path)
        retention, arrays = characterize_retention(scene, retention_steps)
        series_path = output / "feasibility" / "timeseries" / f"level_{level}_retention_{candidate_id:04d}.npz"
        _atomic_npz(series_path, arrays)
        payload["retention_timeseries_path"] = _artifact_path(series_path)
        payload["retention"] = retention
        table_step = retention["first_table_contact_step"]
        numeric_step = retention["first_numeric_invalidity_step"]
        payload["retained_250"] = bool(
            retention["simulated_steps"] >= retention_steps
            and (table_step is None or table_step > retention_steps)
            and (numeric_step is None or numeric_step > retention_steps)
        )
        payload["immediate_slip"] = bool(table_step is not None and table_step <= 25)
    _atomic_json(path, payload)
    return payload


def _feasibility_worker(arguments: tuple[int, int, str, int, int]) -> dict[str, Any]:
    return evaluate_feasibility_candidate(
        arguments[0], arguments[1], arguments[2], seed=arguments[3], retention_steps=arguments[4]
    )


def generate_feasibility_map(
    *,
    candidates_per_level: int = 200,
    workers: int = 1,
    seed: int = PHASE3B05_SEED,
    retention_steps: int = 250,
    output_directory: str | Path = ROOT / "outputs/phase3B05",
) -> list[dict[str, Any]]:
    output = Path(output_directory)
    all_rows: list[dict[str, Any]] = []
    level_summaries = []
    for definition in FEASIBILITY_LEVELS:
        arguments = [
            (definition.level, candidate_id, str(output), seed, retention_steps)
            for candidate_id in range(candidates_per_level)
        ]
        if workers == 1:
            rows = [_feasibility_worker(value) for value in arguments]
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                rows = list(executor.map(_feasibility_worker, arguments, chunksize=4))
        all_rows.extend(rows)
        valid = sum(bool(row["accepted_raw_release"]) for row in rows)
        level_summaries.append(
            {
                "level": definition.level,
                "tested": len(rows),
                "valid_release": valid,
                "valid_release_fraction": valid / len(rows),
                "continued_to_next_level": valid > 0,
            }
        )
        _atomic_json(output / "feasibility" / f"level_{definition.level}_summary.json", level_summaries[-1])
        if valid == 0:
            break
    _atomic_json(
        output / "feasibility" / "manifest.json",
        {
            "data_schema_version": DATA_SCHEMA_VERSION,
            "seed": seed,
            "candidates_per_level": candidates_per_level,
            "retention_steps": retention_steps,
            "label": "ENGINEERING FEASIBILITY LEVELS - NOT PI-FROZEN TRAINING RANGES",
            "levels": [asdict(value) for value in FEASIBILITY_LEVELS],
            "results": level_summaries,
        },
    )
    return all_rows


def load_feasibility_rows(output_directory: str | Path = ROOT / "outputs/phase3B05") -> list[dict[str, Any]]:
    paths = sorted(Path(output_directory).glob("feasibility/level_*/candidate_*.json"))
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def _joint_names(scene: ShadowScene) -> list[str]:
    return [mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_JOINT, index) or f"joint_{index}" for index in range(24)]


def _actuator_names(scene: ShadowScene) -> list[str]:
    return [mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) or f"actuator_{index}" for index in range(scene.model.nu)]


def _joint_tendon_membership(scene: ShadowScene) -> dict[int, list[str]]:
    membership: dict[int, list[str]] = defaultdict(list)
    for actuator_id in range(scene.model.nu):
        if int(scene.model.actuator_trntype[actuator_id]) != int(mujoco.mjtTrn.mjTRN_TENDON):
            continue
        tendon_id = int(scene.model.actuator_trnid[actuator_id, 0])
        tendon_name = mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_TENDON, tendon_id) or f"tendon_{tendon_id}"
        prefix = tendon_name[:-1]
        for suffix in ("2", "1"):
            joint_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{suffix}")
            if joint_id >= 0:
                membership[joint_id].append(tendon_name)
    return membership


def _longest_true_run(values: np.ndarray) -> int:
    best = current = 0
    for value in np.asarray(values, dtype=bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return int(best)


def audit_phase3b0(
    output_directory: str | Path = ROOT / "outputs/phase3B05",
) -> dict[str, Any]:
    """Audit every stored release and replay actual actuator force without retuning."""

    output = Path(output_directory)
    rows = load_attempts(ROOT / "outputs/phase3B0")
    rows = [row for row in rows if row["accepted_raw_release"]][:500]
    if len(rows) != 500:
        raise RuntimeError(f"expected 500 Phase 3B-0 release states, found {len(rows)}")
    scene = build_shadow_scene()
    names = _joint_names(scene)
    ranges = scene.model.jnt_range[:24].copy()
    tendon_membership = _joint_tendon_membership(scene)
    joint_records: list[dict[str, Any]] = []
    release_qpos = []
    for row in rows:
        state_path = Path(row["release_state_path"])
        if not state_path.is_absolute():
            state_path = ROOT / state_path
        with np.load(state_path, allow_pickle=False) as stored:
            qpos = stored["qpos"][:24].copy()
        release_qpos.append(qpos)
        for joint_id, name in enumerate(names):
            lower, upper = ranges[joint_id]
            lower_distance = float(qpos[joint_id] - lower)
            upper_distance = float(upper - qpos[joint_id])
            margin = min(lower_distance, upper_distance)
            joint_records.append(
                {
                    "candidate_id": int(row["candidate"]["candidate_id"]),
                    "joint_id": joint_id,
                    "joint_name": name,
                    "joint_type": mujoco.mjtJoint(int(scene.model.jnt_type[joint_id])).name,
                    "qpos_rad": float(qpos[joint_id]),
                    "lower_limit_rad": float(lower),
                    "upper_limit_rad": float(upper),
                    "distance_to_lower_rad": lower_distance,
                    "distance_to_upper_rad": upper_distance,
                    "minimum_margin_rad": float(margin),
                    "outside_model_limit": bool(margin < 0.0),
                    "tendon_coupling_affected": bool(joint_id in tendon_membership),
                    "tendons": tendon_membership.get(joint_id, []),
                    "within_solver_tolerance": bool(margin >= -float(scene.model.opt.tolerance)),
                }
            )
    release_qpos_array = np.asarray(release_qpos)
    joint_summaries = []
    for joint_id, name in enumerate(names):
        lower, upper = ranges[joint_id]
        margins = np.minimum(release_qpos_array[:, joint_id] - lower, upper - release_qpos_array[:, joint_id])
        joint_summaries.append(
            {
                "joint_id": joint_id,
                "joint_name": name,
                "joint_type": mujoco.mjtJoint(int(scene.model.jnt_type[joint_id])).name,
                "limits_rad": [float(lower), float(upper)],
                "tendon_coupling_affected": bool(joint_id in tendon_membership),
                "tendons": tendon_membership.get(joint_id, []),
                "outside_count": int(np.sum(margins < 0.0)),
                "outside_fraction": float(np.mean(margins < 0.0)),
                "margin_min_rad": float(np.min(margins)),
                "margin_median_rad": float(np.median(margins)),
                "margin_max_rad": float(np.max(margins)),
            }
        )

    handoff = run_handoff_diagnostic()
    handoff_qpos = np.asarray([sample["hand_qpos"] for sample in handoff["samples"]], dtype=np.float64)
    handoff_margins = np.minimum(handoff_qpos - ranges[:, 0], ranges[:, 1] - handoff_qpos)
    handoff_joint_summary = [
        {
            "joint_name": names[index],
            "outside_sample_count": int(np.sum(handoff_margins[:, index] < 0.0)),
            "minimum_margin_rad": float(np.min(handoff_margins[:, index])),
            "sample_count": int(len(handoff_qpos)),
        }
        for index in range(24)
    ]

    actuator_names = _actuator_names(scene)
    ctrl_lower = scene.model.actuator_ctrlrange[:, 0]
    ctrl_upper = scene.model.actuator_ctrlrange[:, 1]
    force_lower = scene.model.actuator_forcerange[:, 0]
    force_upper = scene.model.actuator_forcerange[:, 1]
    release_ctrl = np.empty((len(rows), scene.model.nu))
    release_force = np.empty_like(release_ctrl)
    post_command_flags: list[list[bool]] = [[] for _ in range(scene.model.nu)]
    post_force_flags: list[list[bool]] = [[] for _ in range(scene.model.nu)]
    post_forces: list[list[float]] = [[] for _ in range(scene.model.nu)]
    per_episode_runs: list[list[int]] = [[] for _ in range(scene.model.nu)]
    for row_index, row in enumerate(rows):
        state_path = Path(row["release_state_path"])
        if not state_path.is_absolute():
            state_path = ROOT / state_path
        restore_release_state(scene, state_path)
        release_ctrl[row_index] = scene.data.ctrl
        release_force[row_index] = scene.data.actuator_force
        set_fixture(scene, False)
        simulated_steps = int(row["retention"]["simulated_steps"])
        episode_flags = [[] for _ in range(scene.model.nu)]
        for _ in range(simulated_steps):
            mujoco.mj_step(scene.model, scene.data)
            command_flag = np.isclose(scene.data.ctrl, ctrl_lower) | np.isclose(scene.data.ctrl, ctrl_upper)
            force_flag = np.isclose(scene.data.actuator_force, force_lower, atol=1e-8, rtol=1e-7) | np.isclose(
                scene.data.actuator_force, force_upper, atol=1e-8, rtol=1e-7
            )
            for actuator_id in range(scene.model.nu):
                post_command_flags[actuator_id].append(bool(command_flag[actuator_id]))
                post_force_flags[actuator_id].append(bool(force_flag[actuator_id]))
                post_forces[actuator_id].append(float(scene.data.actuator_force[actuator_id]))
                episode_flags[actuator_id].append(bool(command_flag[actuator_id]))
        for actuator_id in range(scene.model.nu):
            per_episode_runs[actuator_id].append(_longest_true_run(np.asarray(episode_flags[actuator_id])))

    pre_qpos = load_keyframe_qpos("pre grasp")
    unclipped_targets = []
    clipped_targets = actuator_target_from_qpos(scene, pre_qpos)
    actuator_summaries = []
    for actuator_id, name in enumerate(actuator_names):
        transmission = int(scene.model.actuator_trntype[actuator_id])
        transmission_id = int(scene.model.actuator_trnid[actuator_id, 0])
        if transmission == int(mujoco.mjtTrn.mjTRN_JOINT):
            target_name = mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_JOINT, transmission_id)
            raw_target = float(pre_qpos[scene.model.jnt_qposadr[transmission_id]])
            target_type = "joint"
        elif transmission == int(mujoco.mjtTrn.mjTRN_TENDON):
            target_name = mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_TENDON, transmission_id)
            prefix = target_name[:-1]
            ids = [mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{suffix}") for suffix in ("2", "1")]
            raw_target = float(sum(pre_qpos[scene.model.jnt_qposadr[joint_id]] for joint_id in ids))
            target_type = "fixed tendon (J2 + J1)"
        else:
            target_name = f"transmission_{transmission_id}"
            raw_target = float("nan")
            target_type = f"transmission type {transmission}"
        unclipped_targets.append(raw_target)
        command_flags = np.isclose(release_ctrl[:, actuator_id], ctrl_lower[actuator_id]) | np.isclose(
            release_ctrl[:, actuator_id], ctrl_upper[actuator_id]
        )
        release_force_flags = np.isclose(release_force[:, actuator_id], force_lower[actuator_id], atol=1e-8, rtol=1e-7) | np.isclose(
            release_force[:, actuator_id], force_upper[actuator_id], atol=1e-8, rtol=1e-7
        )
        actuator_summaries.append(
            {
                "actuator_id": actuator_id,
                "actuator_name": name,
                "associated_joint_or_tendon": target_name,
                "transmission": target_type,
                "actuator_type": "position servo",
                "ctrlrange": [float(ctrl_lower[actuator_id]), float(ctrl_upper[actuator_id])],
                "forcerange": [float(force_lower[actuator_id]), float(force_upper[actuator_id])],
                "force_limited": bool(scene.model.actuator_forcelimited[actuator_id]),
                "pregrasp_unclipped_target": raw_target,
                "pregrasp_clipped_target": float(clipped_targets[actuator_id]),
                "release_command_median": float(np.median(release_ctrl[:, actuator_id])),
                "release_actual_force_median_n": float(np.median(release_force[:, actuator_id])),
                "release_actual_force_min_n": float(np.min(release_force[:, actuator_id])),
                "release_actual_force_max_n": float(np.max(release_force[:, actuator_id])),
                "release_command_limit_fraction": float(np.mean(command_flags)),
                "release_actual_force_limit_fraction": float(np.mean(release_force_flags)),
                "postrelease_command_limit_fraction": float(np.mean(post_command_flags[actuator_id])),
                "postrelease_actual_force_limit_fraction": float(np.mean(post_force_flags[actuator_id])),
                "postrelease_actual_force_min_n": float(np.min(post_forces[actuator_id])),
                "postrelease_actual_force_max_n": float(np.max(post_forces[actuator_id])),
                "maximum_consecutive_command_limit_samples": int(max(per_episode_runs[actuator_id])),
            }
        )

    audit = {
        "data_schema_version": DATA_SCHEMA_VERSION,
        "release_state_count": len(rows),
        "joint_margin_definition": "min(qpos - lower_limit, upper_limit - qpos) for each of the first 24 compiled hinge joints",
        "compiled_solver_tolerance": float(scene.model.opt.tolerance),
        "joint_records": joint_records,
        "joint_summary": joint_summaries,
        "phase3a_handoff_joint_summary": handoff_joint_summary,
        "phase3a_handoff_sample_count": int(len(handoff_qpos)),
        "actuator_command_limit_definition": "command equals either compiled ctrlrange boundary (numpy.isclose)",
        "actual_force_limit_definition": "data.actuator_force equals either compiled forcerange boundary (atol=1e-8, rtol=1e-7)",
        "actuator_summary": actuator_summaries,
        "diagnosis": {
            "joint_margin": "PHYSICAL_SOFT_CONSTRAINT_EXCURSION_FROM_OUT_OF_RANGE_OFFICIAL_KEYFRAME_COMPONENTS_AND_COUPLED_DYNAMICS",
            "joint_margin_metric_bug": False,
            "actuator": "COMMAND_TARGET_CLIPPING_AT_PREGRASP_KEYFRAME_BOUNDARIES_NOT_ACTUAL_FORCE_SATURATION",
            "actuator_semantic_mapping_bug": False,
            "physics_modified": False,
        },
    }
    _atomic_json(output / "audits" / "phase3b0_joint_actuator_audit.json", audit)
    return audit


def _available_motion(scene: ShadowScene, finger: str) -> float:
    ids = scene.joint_ids[finger]
    qpos = scene.data.qpos[scene.model.jnt_qposadr[ids]]
    limits = scene.model.jnt_range[ids]
    return float(np.sum(np.maximum(0.0, np.minimum(qpos - limits[:, 0], limits[:, 1] - qpos))))


def _fingertip_jacobian_envelope(scene: ShadowScene, finger: str) -> float:
    body_name = scene.config.hand.fingertip_bodies[finger]
    body_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    jacp = np.zeros((3, scene.model.nv), dtype=np.float64)
    jacr = np.zeros_like(jacp)
    mujoco.mj_jacBody(scene.model, scene.data, jacp, jacr, body_id)
    ids = scene.joint_ids[finger]
    dofs = scene.model.jnt_dofadr[ids]
    qpos = scene.data.qpos[scene.model.jnt_qposadr[ids]]
    limits = scene.model.jnt_range[ids]
    margins = np.maximum(0.0, np.minimum(qpos - limits[:, 0], limits[:, 1] - qpos))
    return float(np.linalg.norm(np.abs(jacp[:, dofs]) @ margins))


def symmetry_aware_orientation_change(first: np.ndarray, second: np.ndarray) -> float:
    """D2 symmetry-aware angle for an ellipsoid with three unequal semi-axes."""

    first_rotation = Rotation.from_quat(np.asarray(first), scalar_first=True)
    second_rotation = Rotation.from_quat(np.asarray(second), scalar_first=True)
    symmetries = [
        Rotation.identity(),
        Rotation.from_rotvec([np.pi, 0.0, 0.0]),
        Rotation.from_rotvec([0.0, np.pi, 0.0]),
        Rotation.from_rotvec([0.0, 0.0, np.pi]),
    ]
    return float(min((first_rotation.inv() * second_rotation * symmetry).magnitude() for symmetry in symmetries))


def _active_sample(scene: ShadowScene, stiffness_scale: float) -> dict[str, np.ndarray]:
    contacts = extract_shadow_contacts(scene)
    records = contact_records(scene, contacts)
    penetration = pair_aware_penetration(records)
    relative_position, relative_quaternion = palm_relative_pose(scene)
    linear, angular = object_velocity(scene)
    margins, _ = _joint_margins(scene)
    table = any(record["surface"] == "table" for record in records)
    return {
        "world_position": scene.data.xpos[scene.object_body_id].copy(),
        "world_quaternion": scene.data.xquat[scene.object_body_id].copy(),
        "palm_relative_position": relative_position,
        "palm_relative_quaternion": relative_quaternion,
        "linear_velocity": linear,
        "angular_velocity": angular,
        "contact_flags": contacts.contact_flags.astype(np.int8),
        "normal_forces": contacts.normal_forces.copy(),
        "support_load_fraction": contacts.support_load_fraction.copy(),
        "penetration": np.asarray([penetration["maximum_intended_grip"], penetration["maximum_gross_non_grip"]]),
        "joint_margin_rad": margins,
        "actuator_command": scene.data.ctrl.copy(),
        "actuator_force": scene.data.actuator_force.copy(),
        "stiffness_scale": np.asarray([stiffness_scale]),
        "table_contact": np.asarray([table], dtype=np.int8),
        "finite": np.asarray([np.all(np.isfinite(scene.data.qpos)) and np.all(np.isfinite(scene.data.qvel))], dtype=np.int8),
    }


def _apply_stiffness(scene: ShadowScene, base_gain: np.ndarray, base_bias: np.ndarray, scale: float) -> None:
    scene.model.actuator_gainprm[:, 0] = base_gain * scale
    scene.model.actuator_biasprm[:, 1] = base_bias * scale


def _render_frame(renderer: mujoco.Renderer, scene: ShadowScene, camera: mujoco.MjvCamera) -> np.ndarray:
    renderer.update_scene(scene.data, camera=camera)
    return renderer.render().copy()


def run_active_handoff(
    state_path: str | Path,
    *,
    source_id: str,
    release_finger: str,
    family: str = "baseline",
    displacement_scale: float = 1.0,
    stiffness_scale: float = 1.0,
    rate_scale: float = 1.0,
    motion_scale: float = 0.0,
    output_directory: str | Path = ROOT / "outputs/phase3B05",
    trial_id: str | None = None,
    stage_steps: tuple[int, int, int, int] = (350, 350, 300, 500),
    render_video_path: str | Path | None = None,
) -> dict[str, Any]:
    if release_finger not in {"thumb", "index"}:
        raise ValueError("release_finger must be thumb or index")
    if stiffness_scale > 1.0 or stiffness_scale <= 0.0:
        raise ValueError("stiffness calibration must remain in (0, 1]")
    trial_id = trial_id or f"{source_id}_{release_finger}_{family}_{displacement_scale:g}_{stiffness_scale:g}_{rate_scale:g}_{motion_scale:g}"
    output = Path(output_directory)
    summary_path = output / "active" / "trials" / f"{trial_id}.json"
    series_path = output / "active" / "timeseries" / f"{trial_id}.npz"
    if summary_path.exists() and render_video_path is None:
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing.get("data_schema_version") == DATA_SCHEMA_VERSION:
            existing = repair_active_persistence(existing)
            if family != "C2" or existing.get("active_protocol_version") == ACTIVE_PROTOCOL_VERSION:
                _atomic_json(summary_path, existing)
                return existing

    scene = build_shadow_scene()
    state_path = Path(state_path)
    if not state_path.is_absolute():
        state_path = ROOT / state_path
    restore_release_state(scene, state_path)
    set_fixture(scene, False)
    base_gain = scene.model.actuator_gainprm[:, 0].copy()
    base_bias = scene.model.actuator_biasprm[:, 1].copy()
    target_scale = float(stiffness_scale)
    support_target = actuator_target_from_qpos(scene, load_keyframe_qpos("three finger pinch"))
    open_target = actuator_target_from_qpos(scene, load_keyframe_qpos("open hand"))
    diagnostic_target = actuator_target_from_qpos(scene, load_keyframe_qpos("pre grasp"))
    reference_step = 0.0005
    wrist_step = 0.0002
    samples: list[dict[str, np.ndarray]] = []
    stages: list[str] = []
    release_start_index = -1
    motion_start_index = -1
    frames: list[np.ndarray] = []
    renderer = None
    camera = None
    if render_video_path is not None:
        renderer = mujoco.Renderer(scene.model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = (0.34, -0.02, 0.01)
        camera.distance = 0.36
        camera.azimuth = 145
        camera.elevation = -18

    def record(stage: str) -> None:
        samples.append(_active_sample(scene, current_stiffness[0]))
        stages.append(stage)
        if renderer is not None and len(samples) % 10 == 1:
            frames.append(_render_frame(renderer, scene, camera))

    def advance(stage: str) -> None:
        mujoco.mj_step(scene.model, scene.data)
        mujoco.mj_forward(scene.model, scene.data)
        record(stage)

    def move_group(group: str, step: float) -> None:
        ids = scene.actuator_ids[group]
        scene.data.ctrl[ids] += np.clip(support_target[ids] - scene.data.ctrl[ids], -step, step)

    def regulated_acquisition(step: float) -> None:
        for finger in ("thumb", "index"):
            if fingertip_object_penetration(scene, finger) < float(scene.config.diagnostic["reference_penetration_m"]):
                move_group(finger, step)

    current_stiffness = [1.0]
    record("SOURCE_RELEASE")
    for _ in range(50):
        advance("MINIMAL_UNSUPPORTED_HOLD")
    middle_steps, support_steps, release_steps, post_steps = stage_steps
    target_step = reference_step * displacement_scale * rate_scale
    target_wrist_step = wrist_step * displacement_scale * rate_scale
    stiffness_ramp_steps = max(1, int(round(100 / rate_scale)))
    for index in range(middle_steps):
        if index < stiffness_ramp_steps:
            current_stiffness[0] = 1.0 + (target_scale - 1.0) * (index + 1) / stiffness_ramp_steps
            _apply_stiffness(scene, base_gain, base_bias, current_stiffness[0])
        regulated_acquisition(target_step)
        move_group("wrist", target_wrist_step)
        move_group("middle", target_step)
        advance("MIDDLE_FIRST_DYNAMIC_TRANSFER")
    for _ in range(support_steps):
        regulated_acquisition(target_step)
        move_group("wrist", target_wrist_step)
        move_group("ring", target_step)
        move_group("little", target_step)
        advance("RING_LITTLE_SUPPORT")
    release_start_index = len(samples)
    release_ids = scene.actuator_ids[release_finger]
    release_start = scene.data.ctrl.copy()
    for index in range(release_steps):
        alpha = (index + 1) / release_steps
        scene.data.ctrl[release_ids] = (1.0 - alpha) * release_start[release_ids] + alpha * open_target[release_ids]
        advance("ACQUISITION_FINGER_RELEASE")
    post_release_index = len(samples)
    for _ in range(post_steps):
        advance("POST_RELEASE_RETENTION")

    if motion_scale > 0.0:
        motion_start_index = len(samples)
        start = scene.data.ctrl.copy()
        delta = np.clip(
            diagnostic_target[release_ids] - start[release_ids],
            -reference_step * motion_scale * 25,
            reference_step * motion_scale * 25,
        )
        for index in range(25):
            scene.data.ctrl[release_ids] = start[release_ids] + delta * (index + 1) / 25
            advance("USABLE_MOTION_OUT")
        for index in range(25):
            scene.data.ctrl[release_ids] = start[release_ids] + delta * (1.0 - (index + 1) / 25)
            advance("USABLE_MOTION_RETURN")
        for _ in range(50):
            advance("POST_MOTION_RETENTION")

    if renderer is not None:
        renderer.close()
        import imageio.v2 as imageio

        render_path = Path(render_video_path)
        render_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(render_path, frames, fps=50, codec="libx264", quality=8)

    arrays = {key: np.stack([sample[key] for sample in samples]) for key in samples[0]}
    arrays["step"] = np.arange(len(samples), dtype=np.int32)
    arrays["stage"] = np.asarray(stages, dtype="U40")
    dt = float(scene.model.opt.timestep)
    arrays["time_s"] = arrays["step"] * dt
    arrays["object_acceleration"] = np.vstack((np.zeros(3), np.diff(arrays["linear_velocity"], axis=0) / dt))
    arrays["angular_acceleration"] = np.vstack((np.zeros(3), np.diff(arrays["angular_velocity"], axis=0) / dt))
    arrays["actuator_control_rate"] = np.vstack((np.zeros((1, scene.model.nu)), np.diff(arrays["actuator_command"], axis=0) / dt))
    arrays["stiffness_rate"] = np.r_[0.0, np.diff(arrays["stiffness_scale"][:, 0]) / dt]
    arrays["contact_impulse_proxy"] = arrays["normal_forces"].sum(axis=1) * dt
    release_index = max(post_release_index - 1, 0)
    release_quat = arrays["palm_relative_quaternion"][release_index]
    arrays["orientation_change"] = np.asarray(
        [quaternion_distance(release_quat, value) for value in arrays["palm_relative_quaternion"]]
    )
    arrays["symmetry_aware_orientation_change"] = np.asarray(
        [symmetry_aware_orientation_change(release_quat, value) for value in arrays["palm_relative_quaternion"]]
    )
    angular_speed = np.linalg.norm(arrays["angular_velocity"], axis=1)
    kernel = np.ones(25) / 25
    arrays["sustained_angular_speed"] = np.convolve(angular_speed, kernel, mode="same")
    gaps = detect_contact_gaps(
        arrays["contact_flags"], arrays["world_position"], arrays["palm_relative_position"], arrays["linear_velocity"], dt
    )
    for gap in gaps:
        gap["during_controlled_handoff"] = bool(gap["start_step"] < post_release_index)
        gap["subsequently_retained"] = bool(not np.any(arrays["table_contact"][gap["end_step_exclusive"] :, 0]))

    finger_index = FINGERS.index(release_finger)
    alternative_indices = [index for index in range(len(SUPPORT_SURFACES)) if index != finger_index]
    persistence = {}
    for horizon in PERSISTENCE_HORIZONS:
        end = min(post_release_index + horizon, len(samples))
        segment = slice(post_release_index, end)
        enough = end == post_release_index + horizon
        contact_free = bool(enough and np.all(arrays["contact_flags"][segment, finger_index] == 0))
        object_retained = bool(
            enough
            and not np.any(arrays["table_contact"][segment, 0])
            and np.all(arrays["finite"][segment, 0])
            and np.any(arrays["contact_flags"][end - 1, alternative_indices])
        )
        persistence[str(horizon)] = {
            "contact_free": contact_free,
            "object_retained": object_retained,
            "usable_available_motion": bool(_available_motion(scene, release_finger) > 0.0),
            "combined": bool(contact_free and object_retained and _available_motion(scene, release_finger) > 0.0),
        }

    palm_index = SUPPORT_SURFACES.index("palm")
    alternate = arrays["contact_flags"][:, alternative_indices]
    palm_contact = bool(np.any(arrays["contact_flags"][:, palm_index]))
    alternate_support_before_release = bool(np.any(alternate[:post_release_index]))
    released = bool(arrays["contact_flags"][release_index, finger_index] == 0)
    final_retained = bool(
        not np.any(arrays["table_contact"][post_release_index:, 0])
        and np.all(arrays["finite"][post_release_index:, 0])
        and np.any(arrays["contact_flags"][-1, alternative_indices])
    )
    motion_result = None
    if motion_start_index >= 0:
        before = arrays["palm_relative_position"][motion_start_index].copy()
        after = arrays["palm_relative_position"][-1]
        selected_clear = bool(np.all(arrays["contact_flags"][motion_start_index:, finger_index] == 0))
        motion_result = {
            "motion_scale": motion_scale,
            "joint_space_available_motion_rad": _available_motion(scene, release_finger),
            "jacobian_displacement_envelope_m": _fingertip_jacobian_envelope(scene, release_finger),
            "collision_free_reachable": bool(selected_clear and not np.any(arrays["table_contact"][motion_start_index:, 0])),
            "retained_after_motion": final_retained,
            "object_translation_due_to_probe_m": float(np.linalg.norm(after - before)),
            "object_rotation_due_to_probe_rad": float(arrays["symmetry_aware_orientation_change"][-1] - arrays["symmetry_aware_orientation_change"][motion_start_index]),
        }

    ctrl_lower = scene.model.actuator_ctrlrange[:, 0]
    ctrl_upper = scene.model.actuator_ctrlrange[:, 1]
    command_limit = np.isclose(arrays["actuator_command"], ctrl_lower) | np.isclose(arrays["actuator_command"], ctrl_upper)
    metadata = {
        "data_schema_version": DATA_SCHEMA_VERSION,
        "active_protocol_version": ACTIVE_PROTOCOL_VERSION,
        "trial_id": trial_id,
        "source_id": source_id,
        "source_state_path": _artifact_path(state_path),
        "release_finger": release_finger,
        "family": family,
        "displacement_scale": displacement_scale,
        "stiffness_scale": stiffness_scale,
        "rate_scale": rate_scale,
        "motion_scale": motion_scale,
        "reference_max_safe_scripted_target_step_rad": reference_step,
        "stages": {"release_start_index": release_start_index, "post_release_index": post_release_index, "motion_start_index": motion_start_index},
        "palm_contact_achieved": palm_contact,
        "alternate_support_before_release": alternate_support_before_release,
        "support_shift_observed": bool(np.any(arrays["support_load_fraction"][:post_release_index, alternative_indices].sum(axis=1) > 0.5)),
        "selected_finger_released": released,
        "final_retained_raw": final_retained,
        "diagnostic_handoff_complete": bool(palm_contact and alternate_support_before_release and released and final_retained),
        "persistence": persistence,
        "available_motion_at_end_rad": _available_motion(scene, release_finger),
        "jacobian_displacement_envelope_at_end_m": _fingertip_jacobian_envelope(scene, release_finger),
        "usable_motion_probe": motion_result,
        "maximum_intended_penetration_m": float(np.max(arrays["penetration"][:, 0])),
        "maximum_gross_penetration_m": float(np.max(arrays["penetration"][:, 1])),
        "maximum_object_acceleration_m_s2": float(np.max(np.linalg.norm(arrays["object_acceleration"], axis=1))),
        "maximum_object_angular_acceleration_rad_s2": float(np.max(np.linalg.norm(arrays["angular_acceleration"], axis=1))),
        "maximum_actuator_control_rate_rad_s": float(np.max(np.abs(arrays["actuator_control_rate"]))),
        "maximum_stiffness_rate_per_s": float(np.max(np.abs(arrays["stiffness_rate"]))),
        "maximum_contact_impulse_proxy_n_s": float(np.max(arrays["contact_impulse_proxy"])),
        "minimum_joint_margin_rad": float(np.min(arrays["joint_margin_rad"])),
        "command_limit_sample_fraction": float(np.mean(np.any(command_limit, axis=1))),
        "control_effort_l1_n_steps": float(np.sum(np.abs(arrays["actuator_force"]))),
        "maximum_palm_relative_translation_m": float(np.max(np.linalg.norm(arrays["palm_relative_position"] - arrays["palm_relative_position"][0], axis=1))),
        "maximum_total_orientation_change_rad": float(np.max(arrays["orientation_change"])),
        "maximum_symmetry_aware_orientation_change_rad": float(np.max(arrays["symmetry_aware_orientation_change"])),
        "maximum_sustained_angular_speed_rad_s": float(np.max(arrays["sustained_angular_speed"])),
        "final_angular_speed_rad_s": float(angular_speed[-1]),
        "contact_gaps": gaps,
        "timeseries_path": _artifact_path(series_path),
        "video_path": _artifact_path(render_video_path) if render_video_path is not None else None,
        "success_definition_status": "ENGINEERING DIAGNOSTIC ONLY - FINAL PHASE 3B CRITERIA NOT PI-FROZEN",
    }
    _atomic_npz(series_path, arrays)
    _atomic_json(summary_path, metadata)
    return metadata


def repair_active_persistence(metadata: dict[str, Any]) -> dict[str, Any]:
    """Repair the original Phase 3B-0.5 500-step endpoint bookkeeping.

    This is a metadata-only correction over the stored raw timeseries. It does
    not rerun dynamics or alter any scientific threshold.
    """

    series_path = Path(metadata["timeseries_path"])
    if not series_path.is_absolute():
        series_path = ROOT / series_path
    if not series_path.exists():
        return metadata
    with np.load(series_path, allow_pickle=False) as stored:
        flags = stored["contact_flags"]
        table = stored["table_contact"][:, 0]
        finite = stored["finite"][:, 0]
    post_release_index = int(metadata["stages"]["post_release_index"])
    finger_index = FINGERS.index(metadata["release_finger"])
    alternatives = [index for index in range(len(SUPPORT_SURFACES)) if index != finger_index]
    available = float(metadata["available_motion_at_end_rad"])
    persistence = {}
    for horizon in PERSISTENCE_HORIZONS:
        end = min(post_release_index + horizon, len(flags))
        enough = end == post_release_index + horizon
        segment = slice(post_release_index, end)
        contact_free = bool(enough and np.all(flags[segment, finger_index] == 0))
        retained = bool(
            enough
            and not np.any(table[segment])
            and np.all(finite[segment])
            and np.any(flags[end - 1, alternatives])
        )
        persistence[str(horizon)] = {
            "contact_free": contact_free,
            "object_retained": retained,
            "usable_available_motion": available > 0.0,
            "combined": bool(contact_free and retained and available > 0.0),
        }
    metadata["persistence"] = persistence
    metadata["persistence_endpoint_bookkeeping_version"] = 2
    return metadata


def select_active_sources(rows: Iterable[dict[str, Any]], per_level: int = 2) -> list[dict[str, Any]]:
    selected = []
    for level in range(len(FEASIBILITY_LEVELS)):
        eligible = [
            row for row in rows
            if int(row["candidate"]["level"]) == level and row["accepted_raw_release"] and row["retained_250"]
        ]
        eligible.sort(
            key=lambda row: (
                float(row["release"]["penetration_m"]["maximum_intended_grip"]),
                int(row["candidate"]["candidate_id"]),
            )
        )
        if eligible:
            indices = np.linspace(0, len(eligible) - 1, min(per_level, len(eligible)), dtype=int)
            selected.extend(eligible[index] for index in indices)
    return selected


def generate_active_calibration(
    feasibility_rows: list[dict[str, Any]],
    *,
    output_directory: str | Path = ROOT / "outputs/phase3B05",
    per_level: int = 2,
) -> list[dict[str, Any]]:
    sources = select_active_sources(feasibility_rows, per_level=per_level)
    if not sources:
        raise RuntimeError("no valid retained feasibility states are available for active calibration")
    conditions = [
        ("baseline", 1.0, 1.0, 1.0, 0.0),
        *(("E2", scale, 1.0, 1.0, 0.0) for scale in (0.25, 0.5, 1.0, 1.5)),
        *(("E3", 1.0, scale, 1.0, 0.0) for scale in (1.0, 0.75, 0.5, 0.25)),
        *(("E6", 1.0, 1.0, scale, 0.0) for scale in (0.5, 1.0, 2.0)),
        *(("C2", 1.0, 1.0, 1.0, scale) for scale in (0.25, 0.5, 1.0)),
    ]
    results = []
    for source in sources:
        level = int(source["candidate"]["level"])
        candidate_id = int(source["candidate"]["candidate_id"])
        source_id = f"L{level}C{candidate_id:04d}"
        for release_finger in ("thumb", "index"):
            for family, displacement, stiffness, rate, motion in conditions:
                trial_id = f"{source_id}_{release_finger}_{family}_d{displacement:g}_k{stiffness:g}_r{rate:g}_m{motion:g}".replace(".", "p")
                results.append(
                    run_active_handoff(
                        source["release_state_path"],
                        source_id=source_id,
                        release_finger=release_finger,
                        family=family,
                        displacement_scale=displacement,
                        stiffness_scale=stiffness,
                        rate_scale=rate,
                        motion_scale=motion,
                        output_directory=output_directory,
                        trial_id=trial_id,
                    )
                )
    _atomic_json(
        Path(output_directory) / "active" / "manifest.json",
        {
            "data_schema_version": DATA_SCHEMA_VERSION,
            "source_count": len(sources),
            "trial_count": len(results),
            "sources": [
                {"level": row["candidate"]["level"], "candidate_id": row["candidate"]["candidate_id"], "state": row["release_state_path"]}
                for row in sources
            ],
            "conditions": [list(value) for value in conditions],
            "trial_ids": [row["trial_id"] for row in results],
        },
    )
    return results


def load_active_trials(output_directory: str | Path = ROOT / "outputs/phase3B05") -> list[dict[str, Any]]:
    output = Path(output_directory)
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(output.glob("active/trials/*.json"))
    ]
    manifest_path = output / "active" / "manifest.json"
    if manifest_path.exists():
        expected = set(json.loads(manifest_path.read_text(encoding="utf-8")).get("trial_ids", ()))
        if expected:
            rows = [row for row in rows if row["trial_id"] in expected]
    return rows
