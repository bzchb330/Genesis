#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path
import statistics

import numpy as np

from seqgrasp.config import ROOT, load_configs
from seqgrasp.experiments.metadata import config_hash, git_commit_sha
from seqgrasp.experiments.resumable import IncrementalJsonlStore, stable_trial_id
from seqgrasp.experiments.static_wrist import (
    StaticWristPose,
    coarse_wrist_poses,
    compose_mount_quaternion_wxyz,
    deterministic_screening_subset,
    gravity_in_palm_frame,
    palm_normal_gravity_angle_deg,
    palm_normal_world,
    refined_wrist_poses,
    revalidate_transformed_endpoint,
)
from seqgrasp.phase2_config import load_phase2_config
from seqgrasp.phase2s_config import load_phase2s_config
from seqgrasp.phase2w_config import load_phase2w_config


METRIC_KEYS = (
    "A_translation_drift_m", "A_rotation_drift_rad", "table_recontact",
    "complete_hand_contact_loss", "palm_A_contact_fraction",
    "mean_per_finger_normal_force_N", "occupied_finger_mask",
    "free_finger_mask", "ferrari_canny_epsilon", "total_A_normal_force_N",
    "minimum_joint_margin_rad", "maximum_penetration_m",
    "COM_to_palm_surface_distance_m", "object_A_COM_palm_reference_m",
    "final_object_position_m", "final_object_quaternion",
    "final_joint_configuration_rad",
)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _largest(root: Path) -> tuple[Path, list[dict]]:
    paths = list(root.rglob("accepted_states.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no accepted_states.jsonl under {root}")
    candidates = [(len(_jsonl(path)), path.stat().st_mtime_ns, path) for path in paths]
    path = max(candidates)[2]
    return path, _jsonl(path)


def _pose(payload: dict) -> StaticWristPose:
    return StaticWristPose(
        pose_id=payload["pose_id"],
        relative_rpy_deg=tuple(payload["relative_rpy_deg"]),
        relative_quaternion_wxyz=tuple(payload["relative_quaternion_wxyz"]),
        source=payload["source"],
        parent_pose_id=payload.get("parent_pose_id"),
    )


def _task(payload):
    group, record, pose_payload, base_cfg, phase2s, phase2 = payload
    pose = _pose(pose_payload)
    result = revalidate_transformed_endpoint(record, pose, base_cfg, phase2s, phase2)
    identity = {"pose_id": pose.pose_id, "group": group, "source_state_id": record["grasp_state_id"]}
    compact = {
        "trial_id": stable_trial_id("phase2W-endpoint-screen", identity),
        **identity,
        "pose": pose_payload,
        "accepted": bool(result["accepted"]),
        "rejection_reason": result.get("rejection_reason"),
        "checks": result.get("checks", {}),
        "initial_palm_position_m": result["initial_palm_position_m"],
        "initial_palm_quaternion": result["initial_palm_quaternion"],
        "initial_object_position_m": result["initial_object_position_m"],
        "initial_object_quaternion": result["initial_object_quaternion"],
        "gravity_palm_m_per_s2": result["gravity_palm_m_per_s2"],
        "palm_normal_world": result["palm_normal_world"],
        "palm_normal_gravity_angle_deg": result["palm_normal_gravity_angle_deg"],
    }
    compact.update({key: result.get(key) for key in METRIC_KEYS})
    return compact


def _mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(statistics.fmean(values)) if values else None


def _summaries(poses: list[StaticWristPose], rows: list[dict], minimum: int, base_mount_quat: list[float]) -> list[dict]:
    result = []
    for pose in poses:
        by_pose = [row for row in rows if row["pose_id"] == pose.pose_id]
        groups = {}
        for group in ("FINGERTIP", "PALMAR_SECURED"):
            group_rows = [row for row in by_pose if row["group"] == group]
            valid = [row for row in group_rows if row["accepted"]]
            groups[group] = {
                "screened": len(group_rows),
                "valid": len(valid),
                "valid_fraction": len(valid) / len(group_rows) if group_rows else 0.0,
                "failure_mechanisms": dict(Counter(str(row["rejection_reason"]) for row in group_rows if not row["accepted"])),
                "translation_drift_m_mean": _mean(valid, "A_translation_drift_m"),
                "rotation_drift_rad_mean": _mean(valid, "A_rotation_drift_rad"),
                "palm_contact_fraction_mean": _mean(valid, "palm_A_contact_fraction"),
                "minimum_joint_margin_rad_mean": _mean(valid, "minimum_joint_margin_rad"),
                "valid_source_state_ids": [row["source_state_id"] for row in valid],
            }
        nominal_quaternion = compose_mount_quaternion_wxyz(base_mount_quat, pose.relative_quaternion_wxyz)
        eligible = all(groups[group]["valid"] >= minimum for group in groups)
        result.append({
            "pose": asdict(pose),
            "eligible_endpoint_gate": eligible,
            "groups": groups,
            "minimum_group_survival": min(groups[group]["valid"] for group in groups),
            "total_survival": sum(groups[group]["valid"] for group in groups),
            "nominal_mount_quaternion_wxyz": nominal_quaternion.tolist(),
            "nominal_palm_normal_world": palm_normal_world(nominal_quaternion).tolist(),
            "nominal_gravity_palm_m_per_s2": gravity_in_palm_frame(nominal_quaternion).tolist(),
            "nominal_palm_normal_gravity_angle_deg": palm_normal_gravity_angle_deg(nominal_quaternion),
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Resumable static-wrist endpoint stability screening")
    parser.add_argument("--stage", choices=("coarse", "refined"), default="coarse")
    parser.add_argument("--parent-poses", help="JSON file containing the selected coarse pose payloads for refinement")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--limit-poses", type=int)
    args = parser.parse_args()
    cfg_w, cfg_w_path = load_phase2w_config()
    phase2s, phase2s_path = load_phase2s_config()
    phase2, phase2_path = load_phase2_config()
    base_cfg = load_configs(scene_filename=cfg_w.scene_filename)
    fingertip_path, fingertip_all = _largest(ROOT / "outputs" / "phase2TR" / "fingertip_states")
    palmar_path, palmar_all = _largest(ROOT / "outputs" / "phase2TR" / "palmar_states")
    count = cfg_w.wrist_search.screening_states_per_group
    selected = {
        "FINGERTIP": deterministic_screening_subset(fingertip_all[:100], count),
        "PALMAR_SECURED": deterministic_screening_subset(palmar_all[:102], count),
    }
    if args.stage == "coarse":
        poses = coarse_wrist_poses(
            cfg_w.wrist_search.coarse_roll_deg,
            cfg_w.wrist_search.coarse_pitch_deg,
            cfg_w.wrist_search.coarse_yaw_deg,
        )
        parent_path = None
    else:
        if not args.parent_poses:
            raise ValueError("--parent-poses is required for refined screening")
        parent_path = Path(args.parent_poses).resolve()
        parent_payload = json.loads(parent_path.read_text(encoding="utf-8"))
        parents = [_pose(item) for item in parent_payload["poses"]]
        if len(parents) != cfg_w.wrist_search.top_coarse_for_refinement:
            raise ValueError("refinement requires exactly five coarse parent poses")
        poses = refined_wrist_poses(parents, cfg_w.wrist_search.refinement_offsets_deg)
    if args.limit_poses is not None:
        poses = poses[:args.limit_poses]
    hash_inputs = [
        cfg_w_path, phase2s_path, phase2_path, fingertip_path, palmar_path,
        ROOT / "seqgrasp" / "experiments" / "static_wrist.py",
        ROOT / "scripts" / "screen_phase2w_endpoints.py",
    ]
    if parent_path is not None:
        hash_inputs.append(parent_path)
    cfg_hash = config_hash(hash_inputs)
    output = ROOT / cfg_w.output_dir / "endpoint_screen" / args.stage / cfg_hash[:12]
    store = IncrementalJsonlStore(output / "endpoint_trials.jsonl", 30.0, 0.05)
    completed = store.completed_ids()
    payloads = []
    for pose in poses:
        pose_payload = asdict(pose)
        for group, records in selected.items():
            for record in records:
                identity = {"pose_id": pose.pose_id, "group": group, "source_state_id": record["grasp_state_id"]}
                if stable_trial_id("phase2W-endpoint-screen", identity) not in completed:
                    payloads.append((group, record, pose_payload, base_cfg, phase2s, phase2))
    workers = min(args.workers or max(1, (os.cpu_count() or 1) // 2), cfg_w.wrist_search.maximum_workers)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        buffer = []
        for index, result in enumerate(executor.map(_task, payloads), start=1):
            buffer.append(result)
            if len(buffer) >= workers or index == len(payloads):
                store.append_many(buffer)
                buffer.clear()
            if index % (workers * 5) == 0 or index == len(payloads):
                print(f"Phase 2W {args.stage} endpoint screening: {len(completed) + index}/{len(poses) * 2 * count}", flush=True)
    all_rows = store.records()
    pose_ids = {pose.pose_id for pose in poses}
    rows = [row for row in all_rows if row["pose_id"] in pose_ids]
    summaries = _summaries(
        poses, rows, cfg_w.wrist_search.minimum_valid_states_per_group,
        base_cfg.hand.mount_quat,
    )
    eligible = [row for row in summaries if row["eligible_endpoint_gate"]]
    summary = {
        "stage": args.stage,
        "orientation_count": len(poses),
        "orientation_pose_ids": [pose.pose_id for pose in poses],
        "deduplicated_from_count": 125 if args.stage == "coarse" else 27 * cfg_w.wrist_search.top_coarse_for_refinement,
        "screening_states_per_group": count,
        "selected_state_ids": {key: [row["grasp_state_id"] for row in value] for key, value in selected.items()},
        "eligible_orientation_count": len(eligible),
        "poses": summaries,
        "config_hash": cfg_hash,
        "git_commit_sha": git_commit_sha(ROOT),
        "resume_command": f"python scripts/screen_phase2w_endpoints.py --stage {args.stage} --workers {workers}" + (f" --parent-poses {parent_path}" if parent_path else ""),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "orientation_count": len(poses),
        "eligible_orientation_count": len(eligible),
        "top_endpoint_survival": [
            {"pose_id": row["pose"]["pose_id"], "minimum_group_survival": row["minimum_group_survival"], "total_survival": row["total_survival"]}
            for row in sorted(summaries, key=lambda item: (-item["minimum_group_survival"], -item["total_survival"], item["pose"]["pose_id"]))[:10]
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
