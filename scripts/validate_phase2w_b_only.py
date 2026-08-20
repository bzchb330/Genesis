#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from seqgrasp.config import ROOT, load_configs
from seqgrasp.control import resolve_hand_indices
from seqgrasp.experiments.metadata import config_hash, git_commit_sha
from seqgrasp.experiments.phase2_5_trajectory import BAcquisitionTrajectory, run_b_acquisition_trajectory
from seqgrasp.experiments.resumable import IncrementalJsonlStore, stable_trial_id
from seqgrasp.experiments.second_grasp import BPlacement
from seqgrasp.experiments.static_wrist import StaticWristPose, compose_mount_quaternion_wxyz
from seqgrasp.phase2_5_config import load_phase2_5_config
from seqgrasp.phase2w_config import load_phase2w_config
from seqgrasp.scene_builder import build_scene


GROUPS = ("FINGERTIP", "PALMAR_SECURED")
PAIR_INDICES = (0, 3)
OTHER_INDICES = (1, 2)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _latest_geometry(stage: str) -> Path:
    paths = list((ROOT / "outputs" / "phase2W" / "wrist_geometry" / stage).rglob("summary.json"))
    if not paths:
        raise FileNotFoundError(f"no Phase 2W {stage} geometry summary")
    eligible = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "collision_free_opposition_orientation_count" in payload:
            eligible.append(path)
    if not eligible:
        raise FileNotFoundError(f"no revised Phase 2W {stage} geometry summary")
    return max(eligible, key=lambda path: path.stat().st_mtime_ns)


def _geometry_rank(row: dict) -> tuple:
    access = min(row["common_access"][group]["access_fraction"] for group in GROUPS)
    hand = max(row["common_access"][group]["initial_hand_overlap_fraction"] for group in GROUPS)
    A_overlap = max(row["common_access"][group]["initial_A_overlap_fraction"] for group in GROUPS)
    top = row.get("top_geometry_candidates") or []
    epsilon = top[0]["minimum_epsilon"] if top else 0.0
    margin = min(row["workspace"][group]["mean_joint_margin_rad"] for group in GROUPS)
    endpoint = row["endpoint_summary"]
    return (
        int(endpoint["minimum_group_survival"]),
        int(row["collision_free_opposition_count"] > 0),
        float(access), -float(hand), -float(A_overlap),
        int(epsilon > 0.0), float(margin), int(endpoint["total_survival"]),
    )


def _selected_geometry(coarse_path: Path, refined_path: Path, limit: int) -> list[dict]:
    rows = []
    for path in (coarse_path, refined_path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(
            row for row in payload["poses"]
            if row.get("collision_free_opposition_count", 0) > 0
            and row.get("common_access") is not None
            and all(row["common_access"][group]["access_fraction"] > 0 for group in GROUPS)
        )
    rows.sort(key=lambda row: (_geometry_rank(row), row["pose"]["pose_id"]), reverse=True)
    selected, quaternions = [], set()
    for row in rows:
        key = tuple(np.round(row["pose"]["relative_quaternion_wxyz"], 10))
        if key in quaternions:
            continue
        selected.append(row)
        quaternions.add(key)
        if len(selected) >= limit:
            break
    return selected


def _native_trajectory_source() -> tuple[Path, dict]:
    paths = list((ROOT / "outputs" / "phase2TR" / "b_only_index_thumb").rglob("summary.json"))
    if not paths:
        raise FileNotFoundError("Phase 2T-R B-only summary unavailable")
    candidates = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("best_success") and payload.get("strict_success_count", 0) >= 3:
            candidates.append((path.stat().st_mtime_ns, path, payload["best_success"]))
    if not candidates:
        raise RuntimeError("Phase 2T-R has no strict index+thumb proposal source")
    _, path, source = max(candidates)
    return path, source


def _pose(payload: dict) -> StaticWristPose:
    return StaticWristPose(
        pose_id=payload["pose_id"],
        relative_rpy_deg=tuple(payload["relative_rpy_deg"]),
        relative_quaternion_wxyz=tuple(payload["relative_quaternion_wxyz"]),
        source=payload["source"],
        parent_pose_id=payload.get("parent_pose_id"),
    )


def _scene_for_pose(base_cfg, pose_payload: dict):
    pose = _pose(pose_payload)
    quaternion = compose_mount_quaternion_wxyz(base_cfg.hand.mount_quat, pose.relative_quaternion_wxyz)
    return replace(base_cfg, hand=replace(base_cfg.hand, mount_quat=quaternion.tolist()))


def _unit_samples(seed: int, pose_id: str, count: int) -> np.ndarray:
    derived = int(np.random.SeedSequence([seed, sum(pose_id.encode())]).generate_state(1)[0])
    return qmc.LatinHypercube(40, seed=derived).random(count)


def _trajectory(
    source: dict,
    geometry: dict,
    candidate_index: int,
    unit: np.ndarray,
    cfg25,
    joint_ranges: np.ndarray,
) -> tuple[BAcquisitionTrajectory, str]:
    payload = source["trajectory"]
    active = np.repeat([True, False, False, True], 4)
    if candidate_index == 0:
        return BAcquisitionTrajectory(
            candidate_index=0,
            approach_joint_rad=tuple(payload["approach_joint_rad"]),
            precontact_joint_rad=tuple(payload["precontact_joint_rad"]),
            closing_joint_rad=tuple(payload["closing_joint_rad"]),
            hold_joint_rad=tuple(payload["hold_joint_rad"]),
            close_steps=int(payload["close_steps"]),
            per_finger_close_delay_steps=tuple(payload["per_finger_close_delay_steps"]),
            fixture_release_delay_steps=int(payload["fixture_release_delay_steps"]),
        ), "native_phase2TR_success"
    anchor_kind = ("native_phase2TR_success", "FINGERTIP_geometry", "PALMAR_SECURED_geometry")[candidate_index % 3]
    native_pre = np.asarray(payload["precontact_joint_rad"], dtype=float)
    anchor = native_pre.copy()
    if anchor_kind != "native_phase2TR_success":
        group = anchor_kind.removesuffix("_geometry")
        candidate = geometry["top_geometry_candidates"][0]
        representative = candidate["groups"][group]["representative_joint_rad"]
        anchor[:4] = representative["index"]
        anchor[12:16] = representative["thumb"]
    cursor = 4
    stages = []
    for name, bounds in (
        ("approach_joint_rad", cfg25.trajectory_search.joint_approach_offset_rad),
        ("precontact_joint_rad", cfg25.trajectory_search.joint_precontact_offset_rad),
        ("closing_joint_rad", cfg25.trajectory_search.joint_closing_offset_rad),
        ("hold_joint_rad", cfg25.trajectory_search.joint_hold_offset_rad),
    ):
        values = anchor + (np.asarray(payload[name], dtype=float) - native_pre)
        offsets = np.interp(unit[cursor:cursor + 8], [0.0, 1.0], bounds)
        cursor += 8
        values[active] += offsets
        stages.append(np.clip(values, joint_ranges[:, 0], joint_ranges[:, 1]))
    close_steps = int(round(np.interp(unit[cursor], [0.0, 1.0], cfg25.timing.close_steps_bounds)))
    cursor += 1
    delay_bounds = cfg25.trajectory_search.per_finger_close_delay_steps
    index_delay = int(round(np.interp(unit[cursor], [0.0, 1.0], delay_bounds)))
    thumb_delay = int(round(np.interp(unit[cursor + 1], [0.0, 1.0], delay_bounds)))
    cursor += 2
    release_delay = int(round(np.interp(unit[cursor], [0.0, 1.0], cfg25.timing.fixture_release_delay_steps_bounds)))
    return BAcquisitionTrajectory(
        candidate_index=candidate_index,
        approach_joint_rad=tuple(float(v) for v in stages[0]),
        precontact_joint_rad=tuple(float(v) for v in stages[1]),
        closing_joint_rad=tuple(float(v) for v in stages[2]),
        hold_joint_rad=tuple(float(v) for v in stages[3]),
        close_steps=close_steps,
        per_finger_close_delay_steps=(index_delay, 0, 0, thumb_delay),
        fixture_release_delay_steps=release_delay,
    ), anchor_kind


def _placement(region: dict, candidate_index: int, unit: np.ndarray) -> BPlacement:
    bounds = region["center_bounds_m"]
    low = np.asarray([bounds[axis][0] for axis in "xyz"] + [region["yaw_bounds_rad"][0]], dtype=float)
    high = np.asarray([bounds[axis][1] for axis in "xyz"] + [region["yaw_bounds_rad"][1]], dtype=float)
    row = (low + high) / 2.0 if candidate_index == 0 else low + unit[:4] * (high - low)
    yaw = float(row[3])
    return BPlacement(
        index=candidate_index,
        position_m=tuple(float(value) for value in row[:3]),
        quaternion=tuple(float(value) for value in Rotation.from_euler("z", yaw).as_quat(scalar_first=True)),
        yaw_rad=yaw,
    )


def _strict(summary: dict, arrays: dict[str, np.ndarray], cfg25) -> tuple[bool, bool, bool]:
    release = int(summary["fixture_release_timestep"])
    flags = np.asarray(arrays["B_per_finger_contact_flag"], dtype=int)
    pre = flags[:release]
    both_before_release = bool(len(pre) and np.any(np.all(pre[:, PAIR_INDICES] > 0, axis=1)))
    no_assist = bool(not np.any(flags[:, OTHER_INDICES] > 0))
    strict = bool(
        summary["B_acquired"]
        and both_before_release
        and no_assist
        and summary["unsupported_contact_steps"] == cfg25.timing.unsupported_hold_steps
        and not summary["B_table_contact_after_release"]
        and summary["first_post_release_contact_loss_step"] is None
        and summary["numerically_valid"]
    )
    return strict, both_before_release, no_assist


def _search_task(payload):
    cfg25, scene_cfg, geometry, source, index, unit, joint_ranges, cfg_hash = payload
    trajectory, anchor_kind = _trajectory(source, geometry, index, unit, cfg25, joint_ranges)
    placement = _placement(geometry["candidate_region"], index, unit)
    summary, arrays = run_b_acquisition_trajectory(
        cfg25, trajectory, placement=placement, scene_cfg=scene_cfg, collect_timeseries=True,
    )
    strict, both, no_assist = _strict(summary, arrays, cfg25)
    identity = {"wrist_pose_id": geometry["pose"]["pose_id"], "candidate_index": index, "config_hash": cfg_hash}
    return {
        **summary,
        "trial_id": stable_trial_id("phase2W-static-wrist-B-only", identity),
        "wrist_pose_id": geometry["pose"]["pose_id"],
        "wrist_pose": geometry["pose"],
        "candidate_region": geometry["candidate_region"],
        "geometry_rank_evidence": {
            "endpoint_minimum_group_survival": geometry["endpoint_summary"]["minimum_group_survival"],
            "minimum_group_access": min(geometry["common_access"][group]["access_fraction"] for group in GROUPS),
            "collision_free_opposition_count": geometry["collision_free_opposition_count"],
            "positive_ferrari_canny_count": geometry["positive_ferrari_canny_count"],
        },
        "candidate_index": int(index),
        "permitted_acquisition_pair": ["index", "thumb"],
        "both_index_thumb_contact_before_release": both,
        "middle_ring_assist": not no_assist,
        "strict_index_thumb_success": strict,
        "trajectory_proposal_center": anchor_kind,
        "config_hash": cfg_hash,
    }


def _robust_task(payload):
    cfg25, scene_cfg, source_result, index, seed, cfg_hash = payload
    rng = np.random.default_rng(np.random.SeedSequence([seed, sum(source_result["wrist_pose_id"].encode()), index]))
    trajectory_payload = source_result["trajectory"]
    trajectory = BAcquisitionTrajectory(
        candidate_index=index,
        approach_joint_rad=tuple(trajectory_payload["approach_joint_rad"]),
        precontact_joint_rad=tuple(trajectory_payload["precontact_joint_rad"]),
        closing_joint_rad=tuple(trajectory_payload["closing_joint_rad"]),
        hold_joint_rad=tuple(trajectory_payload["hold_joint_rad"]),
        close_steps=int(trajectory_payload["close_steps"]),
        per_finger_close_delay_steps=tuple(trajectory_payload["per_finger_close_delay_steps"]),
        fixture_release_delay_steps=int(trajectory_payload["fixture_release_delay_steps"]),
    )
    region = source_result["candidate_region"]
    bounds = region["center_bounds_m"]
    position = np.asarray(source_result["placement"]["position_m"], dtype=float)
    local_half = np.minimum(
        0.0005,
        np.asarray([(bounds[axis][1] - bounds[axis][0]) / 2.0 for axis in "xyz"]),
    )
    position = position + rng.uniform(-local_half, local_half)
    for axis_index, axis in enumerate("xyz"):
        position[axis_index] = np.clip(position[axis_index], bounds[axis][0], bounds[axis][1])
    yaw_half = min(0.012, (region["yaw_bounds_rad"][1] - region["yaw_bounds_rad"][0]) / 2.0)
    yaw = float(np.clip(
        source_result["placement"]["yaw_rad"] + rng.uniform(-yaw_half, yaw_half),
        region["yaw_bounds_rad"][0], region["yaw_bounds_rad"][1],
    ))
    placement = BPlacement(
        index=index, position_m=tuple(float(value) for value in position),
        quaternion=tuple(float(value) for value in Rotation.from_euler("z", yaw).as_quat(scalar_first=True)),
        yaw_rad=yaw,
    )
    summary, arrays = run_b_acquisition_trajectory(
        cfg25, trajectory, placement=placement, scene_cfg=scene_cfg, collect_timeseries=True,
    )
    strict, both, no_assist = _strict(summary, arrays, cfg25)
    identity = {"wrist_pose_id": source_result["wrist_pose_id"], "robustness_index": index, "config_hash": cfg_hash}
    return {
        **summary,
        "trial_id": stable_trial_id("phase2W-static-wrist-B-only-robustness", identity),
        "wrist_pose_id": source_result["wrist_pose_id"],
        "robustness_index": int(index),
        "candidate_region": region,
        "both_index_thumb_contact_before_release": both,
        "middle_ring_assist": not no_assist,
        "strict_index_thumb_success": strict,
        "config_hash": cfg_hash,
    }


def _run_search(store, geometries, cfg25, base_cfg, source, counts, seed, joint_ranges, cfg_hash, workers):
    completed = store.completed_ids()
    payloads = []
    for geometry in geometries:
        pose_id = geometry["pose"]["pose_id"]
        scene_cfg = _scene_for_pose(base_cfg, geometry["pose"])
        units = _unit_samples(seed, pose_id, counts[pose_id])
        for index in range(counts[pose_id]):
            identity = {"wrist_pose_id": pose_id, "candidate_index": index, "config_hash": cfg_hash}
            if stable_trial_id("phase2W-static-wrist-B-only", identity) not in completed:
                payloads.append((cfg25, scene_cfg, geometry, source, index, units[index], joint_ranges, cfg_hash))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        buffer = []
        for count, result in enumerate(executor.map(_search_task, payloads), start=1):
            buffer.append(result)
            if len(buffer) >= workers or count == len(payloads):
                store.append_many(buffer)
                buffer.clear()
            if count % (workers * 8) == 0 or count == len(payloads):
                print(f"Phase 2W B-only search: {len(completed) + count}/{sum(counts.values())}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate index+thumb B-only control at pre-outcome selected static wrist poses")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--initial-only", action="store_true")
    args = parser.parse_args()
    cfg_w, cfg_w_path = load_phase2w_config()
    cfg25, cfg25_path = load_phase2_5_config()
    coarse_path, refined_path = _latest_geometry("coarse"), _latest_geometry("refined")
    geometries = _selected_geometry(coarse_path, refined_path, cfg_w.wrist_search.top_dynamic_candidates)
    if not geometries:
        print("PHASE2W_NO_STATIC_WRIST_GEOMETRIC_REGION")
        return 3
    source_path, source = _native_trajectory_source()
    base_cfg = load_configs(scene_filename=cfg_w.scene_filename)
    model, _ = build_scene(base_cfg)
    indices = resolve_hand_indices(model, base_cfg.hand)
    joint_ranges = model.jnt_range[indices.joint_ids].copy()
    cfg_hash = config_hash([
        cfg_w_path, cfg25_path, coarse_path, refined_path, source_path,
        ROOT / "seqgrasp" / "experiments" / "static_wrist.py",
        ROOT / "scripts" / "validate_phase2w_b_only.py",
    ])
    output = ROOT / cfg_w.output_dir / "b_only_dynamic" / cfg_hash[:12]
    output.mkdir(parents=True, exist_ok=True)
    selection = {
        "selection_basis": "Phase 2W Section 14 pre-outcome lexicographic ranking; no A+B outcomes",
        "candidate_count": len(geometries),
        "candidates": [
            {
                "pre_outcome_rank": rank + 1,
                "pose": geometry["pose"],
                "candidate_region": geometry["candidate_region"],
                "endpoint_summary": geometry["endpoint_summary"],
                "workspace": geometry["workspace"],
                "common_access": geometry["common_access"],
                "collision_free_opposition_count": geometry["collision_free_opposition_count"],
                "positive_ferrari_canny_count": geometry["positive_ferrari_canny_count"],
                "top_geometry_candidates": geometry["top_geometry_candidates"],
            }
            for rank, geometry in enumerate(geometries)
        ],
    }
    (output / "selected_wrist_candidates.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    workers = min(args.workers or max(1, (os.cpu_count() or 1) // 2), cfg_w.wrist_search.maximum_workers)
    store = IncrementalJsonlStore(output / "candidate_results.jsonl", 30.0, 0.05)
    counts = {geometry["pose"]["pose_id"]: cfg_w.second_grasp.initial_candidates_per_wrist for geometry in geometries}
    _run_search(store, geometries, cfg25, base_cfg, source, counts, cfg_w.seeds.B_only, joint_ranges, cfg_hash, workers)
    rows = store.records()
    by_pose = {
        geometry["pose"]["pose_id"]: [row for row in rows if row["wrist_pose_id"] == geometry["pose"]["pose_id"] and row["candidate_index"] < counts[geometry["pose"]["pose_id"]]]
        for geometry in geometries
    }
    if not args.initial_only:
        remaining = cfg_w.second_grasp.global_candidate_cap - sum(counts.values())
        for geometry in geometries:
            pose_id = geometry["pose"]["pose_id"]
            if any(row["strict_index_thumb_success"] for row in by_pose[pose_id]):
                continue
            added = cfg_w.second_grasp.expanded_candidates_per_wrist - counts[pose_id]
            if added <= remaining:
                counts[pose_id] = cfg_w.second_grasp.expanded_candidates_per_wrist
                remaining -= added
            if remaining < added:
                break
        _run_search(store, geometries, cfg25, base_cfg, source, counts, cfg_w.seeds.B_only, joint_ranges, cfg_hash, workers)
        rows = store.records()
        by_pose = {
            geometry["pose"]["pose_id"]: [row for row in rows if row["wrist_pose_id"] == geometry["pose"]["pose_id"] and row["candidate_index"] < counts[geometry["pose"]["pose_id"]]]
            for geometry in geometries
        }
    pose_summaries = []
    for rank, geometry in enumerate(geometries, start=1):
        pose_id = geometry["pose"]["pose_id"]
        pose_rows = by_pose[pose_id]
        successes = [row for row in pose_rows if row["strict_index_thumb_success"]]
        pose_summaries.append({
            "pre_outcome_rank": rank,
            "wrist_pose_id": pose_id,
            "pose": geometry["pose"],
            "candidate_region": geometry["candidate_region"],
            "candidate_count": len(pose_rows),
            "strict_success_count": len(successes),
            "strict_success_indices": [row["candidate_index"] for row in successes],
            "failure_mechanisms": dict(Counter(row["failure_mechanism"] for row in pose_rows if not row["strict_index_thumb_success"])),
            "best_success": min(
                successes,
                key=lambda row: (row["maximum_B_translation_after_release_m"], row["maximum_B_orientation_after_release_rad"], row["candidate_index"]),
            ) if successes else None,
            "geometry": {
                "endpoint_summary": geometry["endpoint_summary"],
                "workspace": geometry["workspace"],
                "common_access": geometry["common_access"],
                "collision_free_opposition_count": geometry["collision_free_opposition_count"],
                "positive_ferrari_canny_count": geometry["positive_ferrari_canny_count"],
            },
        })
    passing = [row for row in pose_summaries if row["strict_success_count"] >= cfg_w.second_grasp.hard_minimum_successes]
    robust_store = IncrementalJsonlStore(output / "robustness_results.jsonl", 30.0, 0.05)
    robustness_summaries = []
    if passing:
        robust_candidates = sorted(
            passing,
            key=lambda row: (-row["strict_success_count"], row["pre_outcome_rank"]),
        )[:cfg_w.second_grasp.robustness_configuration_cap]
        completed = robust_store.completed_ids()
        payloads = []
        for row in robust_candidates:
            scene_cfg = _scene_for_pose(base_cfg, row["pose"])
            for index in range(cfg_w.second_grasp.robustness_trials_per_configuration):
                identity = {"wrist_pose_id": row["wrist_pose_id"], "robustness_index": index, "config_hash": cfg_hash}
                if stable_trial_id("phase2W-static-wrist-B-only-robustness", identity) not in completed:
                    payloads.append((cfg25, scene_cfg, row["best_success"], index, cfg_w.seeds.robustness, cfg_hash))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            buffer = []
            for count, result in enumerate(executor.map(_robust_task, payloads), start=1):
                buffer.append(result)
                if len(buffer) >= workers or count == len(payloads):
                    robust_store.append_many(buffer)
                    buffer.clear()
                if count % (workers * 4) == 0 or count == len(payloads):
                    print(f"Phase 2W B-only robustness: {len(completed) + count}/{len(robust_candidates) * cfg_w.second_grasp.robustness_trials_per_configuration}", flush=True)
        robust_rows = robust_store.records()
        for candidate in robust_candidates:
            subset = [row for row in robust_rows if row["wrist_pose_id"] == candidate["wrist_pose_id"]]
            robustness_summaries.append({
                "wrist_pose_id": candidate["wrist_pose_id"],
                "trial_count": len(subset),
                "strict_success_count": sum(row["strict_index_thumb_success"] for row in subset),
                "strict_success_fraction": sum(row["strict_index_thumb_success"] for row in subset) / len(subset) if subset else 0.0,
                "failure_mechanisms": dict(Counter(row["failure_mechanism"] for row in subset if not row["strict_index_thumb_success"])),
                "contact_loss_count": sum(row["first_post_release_contact_loss_step"] is not None for row in subset),
                "table_contact_count": sum(row["B_table_contact_after_release"] for row in subset),
                "rotation_failure_count": sum(row["maximum_B_orientation_after_release_rad"] > cfg25.criteria.maximum_B_orientation_rad for row in subset),
                "penetration_failure_count": sum(row["maximum_B_penetration_m"] > cfg25.criteria.maximum_penetration_m for row in subset),
                "translation_m": [row["maximum_B_translation_after_release_m"] for row in subset],
                "rotation_rad": [row["maximum_B_orientation_after_release_rad"] for row in subset],
            })
    status = "B_CONTROL_PASS" if passing else (
        "PHASE2W_NO_STATIC_WRIST_B_CONTROL"
        if sum(len(rows) for rows in by_pose.values()) >= cfg_w.second_grasp.global_candidate_cap
        else "SEARCH_INCOMPLETE"
    )
    summary = {
        "status": status,
        "selected_wrist_candidate_count": len(geometries),
        "total_B_only_candidates": sum(len(rows) for rows in by_pose.values()),
        "global_candidate_cap": cfg_w.second_grasp.global_candidate_cap,
        "poses": pose_summaries,
        "passing_pose_count": len(passing),
        "robustness": robustness_summaries,
        "native_phase2TR_proposal_source": {
            "path": str(source_path.relative_to(ROOT)),
            "candidate_index": source["candidate_index"],
            "source_phase2S_candidate_index": source["source_phase2S_candidate_index"],
        },
        "config_hash": cfg_hash,
        "git_commit_sha": git_commit_sha(ROOT),
        "resume_command": f"python scripts/validate_phase2w_b_only.py --workers {workers}",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output), "status": status,
        "total_B_only_candidates": summary["total_B_only_candidates"],
        "strict_successes_by_pose": {row["wrist_pose_id"]: row["strict_success_count"] for row in pose_summaries},
        "robustness": robustness_summaries,
    }, indent=2))
    return 3 if status == "PHASE2W_NO_STATIC_WRIST_B_CONTROL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
