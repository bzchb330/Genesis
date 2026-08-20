#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import math
import os
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from seqgrasp.config import ROOT, load_configs
from seqgrasp.experiments.grasp_sampling import ferrari_canny_epsilon
from seqgrasp.experiments.metadata import config_hash, git_commit_sha
from seqgrasp.experiments.phase2_6_workspace import contact_opposition_angle_deg, cylinder_surface_geometry
from seqgrasp.experiments.resumable import IncrementalJsonlStore, stable_trial_id
from seqgrasp.experiments.resource_components import reconstruct_grasp
from seqgrasp.experiments.second_grasp import BPlacement, _set_b_pose
from seqgrasp.experiments.static_wrist import StaticWristPose, recompute_index_thumb_workspace
from seqgrasp.phase2_config import load_phase2_config
from seqgrasp.phase2w_config import load_phase2w_config


GROUPS = ("FINGERTIP", "PALMAR_SECURED")


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _latest_summary(stage: str) -> Path:
    paths = list((ROOT / "outputs" / "phase2W" / "endpoint_screen" / stage).rglob("summary.json"))
    if not paths:
        raise FileNotFoundError(f"no Phase 2W {stage} endpoint summary")
    return max(paths, key=lambda path: path.stat().st_mtime_ns)


def _endpoint_sources() -> tuple[dict[str, dict], list[Path]]:
    paths = []
    records = {}
    for group_dir in ("fingertip_states", "palmar_states"):
        candidates = list((ROOT / "outputs" / "phase2TR" / group_dir).rglob("accepted_states.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"no Phase 2T-R {group_dir}")
        path = max(candidates, key=lambda item: (len(_jsonl(item)), item.stat().st_mtime_ns))
        paths.append(path)
        for row in _jsonl(path):
            records[row["grasp_state_id"]] = row
    return records, paths


def _pose(payload: dict) -> StaticWristPose:
    return StaticWristPose(
        pose_id=payload["pose_id"],
        relative_rpy_deg=tuple(payload["relative_rpy_deg"]),
        relative_quaternion_wxyz=tuple(payload["relative_quaternion_wxyz"]),
        source=payload["source"],
        parent_pose_id=payload.get("parent_pose_id"),
    )


def _screen_record(source: dict, screened: dict) -> dict:
    record = dict(source)
    for key in (
        "initial_palm_position_m", "initial_palm_quaternion",
        "initial_object_position_m", "initial_object_quaternion",
        "final_object_position_m", "final_object_quaternion",
        "final_joint_configuration_rad", "occupied_finger_mask", "free_finger_mask",
        "A_translation_drift_m", "A_rotation_drift_rad", "ferrari_canny_epsilon",
        "total_A_normal_force_N", "minimum_joint_margin_rad", "palm_A_contact_fraction",
        "COM_to_palm_surface_distance_m", "object_A_COM_palm_reference_m",
    ):
        if screened.get(key) is not None:
            record[key] = screened[key]
    record["grasp_id"] = screened["source_state_id"]
    record["grasp_state_id"] = f"phase2W_{screened['pose_id']}_{screened['source_state_id']}"
    return record


def _surface_metrics(workspace: dict, center: np.ndarray, phase2, object_b) -> dict:
    contacts, normals, accessible = [], [], []
    representative_joint_rad = {}
    distances = {}
    for finger in ("index", "thumb"):
        points = workspace[f"{finger}_points_world_m"]
        if not len(points):
            distances[finger] = math.inf
            continue
        distance, contact_points, outward = cylinder_surface_geometry(
            points, center, object_b.size[0], object_b.size[1],
        )
        index = int(np.argmin(distance))
        distances[finger] = float(distance[index])
        if distance[index] <= workspace["tip_radii_m"][finger] + phase2.resources.workspace_collision_tolerance_m:
            accessible.append(finger)
            contacts.append(contact_points[index])
            normals.append(-outward[index])
            sample = workspace["free_joint_samples_rad"][index]
            representative_joint_rad[finger] = (
                sample[:4].tolist() if finger == "index" else sample[4:8].tolist()
            )
    opposition = contact_opposition_angle_deg(np.asarray(normals)) if len(normals) == 2 else 0.0
    epsilon = 0.0
    if len(normals) == 2:
        epsilon = ferrari_canny_epsilon(
            np.asarray(contacts), np.asarray(normals), center,
            object_b.friction[0], phase2.dataset.friction_cone_edges,
            float(np.linalg.norm(object_b.size)), phase2.dataset.convex_hull_tolerance,
        )
    return {
        "accessible_fingers": accessible,
        "index_thumb_access": set(accessible) == {"index", "thumb"},
        "opposition_angle_deg": opposition,
        "ferrari_canny_epsilon": float(epsilon),
        "surface_distances_m": distances,
        "representative_joint_rad": representative_joint_rad,
    }


def _overlap(model, data, placement: BPlacement, object_b) -> tuple[bool, bool, bool]:
    _, b_geom, _ = _set_b_pose(model, data, placement)
    table_top = float(model.geom_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table"), 2] + model.geom_size[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table"), 2])
    table_penetration = placement.position_m[2] - object_b.size[1] < table_top
    hand_overlap = False
    A_overlap = False
    for geom_id in range(model.ngeom):
        if geom_id == b_geom or not (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name == "table":
            continue
        distance = float(mujoco.mj_geomDistance(model, data, b_geom, geom_id, 1.0, None))
        if distance < 0.0:
            A_overlap |= name == "object_a_geom"
            hand_overlap |= name != "object_a_geom"
    return bool(hand_overlap), bool(A_overlap), bool(table_penetration)


def _candidate_bounds(workspaces: dict, object_b) -> tuple[np.ndarray, np.ndarray] | None:
    lows, highs = [], []
    for group in GROUPS:
        workspace = workspaces[group]
        for finger in ("index", "thumb"):
            points = workspace[f"{finger}_points_world_m"]
            if not len(points):
                return None
            extent = np.asarray([
                object_b.size[0] + workspace["tip_radii_m"][finger],
                object_b.size[0] + workspace["tip_radii_m"][finger],
                object_b.size[1] + workspace["tip_radii_m"][finger],
            ])
            lows.append(points.min(axis=0) - extent)
            highs.append(points.max(axis=0) + extent)
    low = np.max(np.asarray(lows), axis=0)
    high = np.min(np.asarray(highs), axis=0)
    low[2] = max(low[2], float(object_b.size[1]) + 0.001)
    if np.any(low >= high):
        return None
    return low, high


def _sample_region(center: np.ndarray, low: np.ndarray, high: np.ndarray, yaw_bounds: list[float], half_width: float = 0.001) -> dict:
    local_low = np.maximum(low, center - half_width)
    local_high = np.minimum(high, center + half_width)
    # Keep a nonempty, deterministic interval at an envelope edge.
    for axis in range(3):
        if local_high[axis] - local_low[axis] < 0.0002:
            local_low[axis] = max(low[axis], center[axis] - 0.0001)
            local_high[axis] = min(high[axis], center[axis] + 0.0001)
    return {
        "center_bounds_m": {
            "x": [float(local_low[0]), float(local_high[0])],
            "y": [float(local_low[1]), float(local_high[1])],
            "z": [float(local_low[2]), float(local_high[2])],
        },
        "yaw_bounds_rad": [float(yaw_bounds[0]), float(yaw_bounds[1])],
    }


def _placement(index: int, row: np.ndarray) -> BPlacement:
    yaw = float(row[3])
    return BPlacement(
        index=index,
        position_m=tuple(float(value) for value in row[:3]),
        quaternion=tuple(float(value) for value in Rotation.from_euler("z", yaw).as_quat(scalar_first=True)),
        yaw_rad=yaw,
    )


def _evaluate_pose(payload):
    pose_summary, screen_rows, sources, base_cfg, phase2, cfg_w = payload
    pose = _pose(pose_summary["pose"])
    accepted_by_group = {
        group: [row for row in screen_rows if row["pose_id"] == pose.pose_id and row["group"] == group and row["accepted"]]
        for group in GROUPS
    }
    representative_rows = {
        group: max(accepted_by_group[group], key=lambda row: (float(row.get("minimum_joint_margin_rad") or -math.inf), row["source_state_id"]))
        for group in GROUPS
    }
    state_records = {
        group: [_screen_record(sources[row["source_state_id"]], row) for row in accepted_by_group[group]]
        for group in GROUPS
    }
    workspaces = {}
    compact_workspaces = {}
    for group_index, group in enumerate(GROUPS):
        representative = _screen_record(sources[representative_rows[group]["source_state_id"]], representative_rows[group])
        workspace = recompute_index_thumb_workspace(
            representative, phase2.resources, cfg_w.geometry.workspace_samples_per_group,
            int(np.random.SeedSequence([cfg_w.seeds.geometry, group_index, sum(pose.pose_id.encode())]).generate_state(1)[0]),
            base_cfg,
        )
        workspaces[group] = workspace
        compact_workspaces[group] = {
            key: value for key, value in workspace.items()
            if key not in {
                "cfg", "model", "data", "index_points_world_m", "thumb_points_world_m",
                "free_joint_samples_rad",
            }
        }
        compact_workspaces[group]["representative_state_id"] = representative_rows[group]["source_state_id"]
    object_b = next(obj for obj in base_cfg.scene.objects if obj.name == "object_b")
    bounds = _candidate_bounds(workspaces, object_b)
    candidate_count = cfg_w.geometry.candidate_B_poses_per_orientation
    if bounds is None:
        return {
            "trial_id": stable_trial_id("phase2W-wrist-geometry", pose.pose_id),
            "pose": asdict(pose), "endpoint_summary": pose_summary,
            "workspace": compact_workspaces, "candidate_B_pose_count": 0,
            "candidate_envelope_m": None,
            "collision_free_opposition_count": 0,
            "positive_ferrari_canny_count": 0,
            "candidate_region": None, "common_access": None,
            "rejection_mechanisms": {"empty_common_envelope": 1},
        }
    low, high = bounds
    unit = qmc.LatinHypercube(
        d=4,
        seed=np.random.default_rng(np.random.SeedSequence([cfg_w.seeds.geometry, sum(pose.pose_id.encode())])),
    ).random(candidate_count)
    samples = qmc.scale(
        unit,
        np.r_[low, cfg_w.geometry.yaw_bounds_rad[0]],
        np.r_[high, cfg_w.geometry.yaw_bounds_rad[1]],
    )
    opposition_candidates = []
    rejections = Counter()
    for index, row in enumerate(samples):
        placement = _placement(index, row)
        group_metrics = {}
        rejected = None
        for group in GROUPS:
            metrics = _surface_metrics(workspaces[group], np.asarray(placement.position_m), phase2, object_b)
            hand_overlap, A_overlap, table_penetration = _overlap(
                workspaces[group]["model"], workspaces[group]["data"], placement, object_b,
            )
            metrics.update({
                "initial_hand_overlap": hand_overlap,
                "initial_A_overlap": A_overlap,
                "table_penetration": table_penetration,
            })
            group_metrics[group] = metrics
            if table_penetration:
                rejected = rejected or "table_penetration"
            elif hand_overlap:
                rejected = rejected or f"hand_overlap_{group}"
            elif A_overlap:
                rejected = rejected or f"A_overlap_{group}"
            elif not metrics["index_thumb_access"]:
                rejected = rejected or f"index_thumb_access_{group}"
            elif metrics["opposition_angle_deg"] < cfg_w.geometry.minimum_opposition_angle_deg:
                rejected = rejected or f"opposition_{group}"
        if rejected is None:
            opposition_candidates.append({
                "candidate_index": index,
                "position_m": list(placement.position_m),
                "yaw_rad": placement.yaw_rad,
                "groups": group_metrics,
                "minimum_epsilon": min(group_metrics[group]["ferrari_canny_epsilon"] for group in GROUPS),
                "minimum_opposition_deg": min(group_metrics[group]["opposition_angle_deg"] for group in GROUPS),
            })
        else:
            rejections[rejected] += 1
    opposition_candidates.sort(key=lambda row: (-row["minimum_epsilon"], -row["minimum_opposition_deg"], row["candidate_index"]))
    positive_epsilon_count = sum(
        row["minimum_epsilon"] > phase2.dataset.convex_hull_tolerance
        for row in opposition_candidates
    )
    if not opposition_candidates:
        return {
            "trial_id": stable_trial_id("phase2W-wrist-geometry", pose.pose_id),
            "pose": asdict(pose), "endpoint_summary": pose_summary,
            "workspace": compact_workspaces, "candidate_B_pose_count": candidate_count,
            "candidate_envelope_m": {"low": low.tolist(), "high": high.tolist()},
            "collision_free_opposition_count": 0,
            "positive_ferrari_canny_count": 0,
            "top_geometry_candidates": [],
            "candidate_region": None, "common_access": None,
            "rejection_mechanisms": dict(rejections),
        }
    best = opposition_candidates[0]
    region = _sample_region(np.asarray(best["position_m"]), low, high, cfg_w.geometry.yaw_bounds_rad)
    region_low = np.asarray([region["center_bounds_m"][axis][0] for axis in "xyz"] + [region["yaw_bounds_rad"][0]])
    region_high = np.asarray([region["center_bounds_m"][axis][1] for axis in "xyz"] + [region["yaw_bounds_rad"][1]])
    access_samples = qmc.scale(
        qmc.LatinHypercube(
            d=4,
            seed=np.random.default_rng(np.random.SeedSequence([cfg_w.seeds.geometry, 99, sum(pose.pose_id.encode())])),
        ).random(100),
        region_low, region_high,
    )
    group_access = {}
    for group in GROUPS:
        models = []
        for record in state_records[group]:
            cfg, model, data, _ = reconstruct_grasp(record, base_cfg)
            models.append((model, data))
        access_count = hand_overlap_count = A_overlap_count = table_count = 0
        total = len(models) * len(access_samples)
        for sample_index, row in enumerate(access_samples):
            placement = _placement(sample_index, row)
            reach = _surface_metrics(workspaces[group], np.asarray(placement.position_m), phase2, object_b)
            reach_ok = bool(
                reach["index_thumb_access"]
                and reach["opposition_angle_deg"] >= cfg_w.geometry.minimum_opposition_angle_deg
            )
            for model, data in models:
                hand_overlap, A_overlap, table_penetration = _overlap(model, data, placement, object_b)
                hand_overlap_count += int(hand_overlap)
                A_overlap_count += int(A_overlap)
                table_count += int(table_penetration)
                access_count += int(reach_ok and not hand_overlap and not A_overlap and not table_penetration)
        group_access[group] = {
            "state_count": len(models),
            "B_pose_count": len(access_samples),
            "state_pose_evaluations": total,
            "access_fraction": access_count / total if total else 0.0,
            "initial_hand_overlap_fraction": hand_overlap_count / total if total else 0.0,
            "initial_A_overlap_fraction": A_overlap_count / total if total else 0.0,
            "table_penetration_fraction": table_count / total if total else 0.0,
        }
    voxel = phase2.resources.workspace_voxel_size_m
    positive_centers = np.asarray([row["position_m"] for row in opposition_candidates])
    opposition_voxels = np.unique(np.floor(positive_centers / voxel).astype(np.int64), axis=0)
    return {
        "trial_id": stable_trial_id("phase2W-wrist-geometry", pose.pose_id),
        "pose": asdict(pose), "endpoint_summary": pose_summary,
        "workspace": compact_workspaces, "candidate_B_pose_count": candidate_count,
        "candidate_envelope_m": {"low": low.tolist(), "high": high.tolist()},
        "collision_free_opposition_count": len(opposition_candidates),
        "positive_ferrari_canny_count": int(positive_epsilon_count),
        "opposition_region_volume_m3": float(len(opposition_voxels) * voxel ** 3),
        "top_geometry_candidates": opposition_candidates[:10],
        "candidate_region": region, "common_access": group_access,
        "rejection_mechanisms": dict(rejections),
    }


def _ranking(row: dict) -> tuple:
    groups = row.get("common_access") or {}
    access = min((groups.get(group, {}).get("access_fraction", 0.0) for group in GROUPS), default=0.0)
    hand = max((groups.get(group, {}).get("initial_hand_overlap_fraction", 1.0) for group in GROUPS), default=1.0)
    A_overlap = max((groups.get(group, {}).get("initial_A_overlap_fraction", 1.0) for group in GROUPS), default=1.0)
    top = row.get("top_geometry_candidates") or []
    epsilon = top[0]["minimum_epsilon"] if top else 0.0
    margin = min(
        (row["workspace"][group].get("mean_joint_margin_rad", 0.0) for group in GROUPS),
        default=0.0,
    )
    endpoint = row["endpoint_summary"]
    return (
        int(endpoint["eligible_endpoint_gate"]),
        int(row.get("collision_free_opposition_count", 0) > 0),
        access,
        -hand,
        -A_overlap,
        int(epsilon > 0.0),
        margin,
        endpoint["minimum_group_survival"],
        endpoint["total_survival"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Map Phase 2W static-wrist index/thumb common geometry")
    parser.add_argument("--stage", choices=("coarse", "refined"), default="coarse")
    parser.add_argument("--endpoint-summary")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--limit-poses", type=int)
    args = parser.parse_args()
    cfg_w, cfg_w_path = load_phase2w_config()
    phase2, phase2_path = load_phase2_config()
    base_cfg = load_configs(scene_filename=cfg_w.scene_filename)
    endpoint_summary_path = Path(args.endpoint_summary).resolve() if args.endpoint_summary else _latest_summary(args.stage)
    endpoint_summary = json.loads(endpoint_summary_path.read_text(encoding="utf-8"))
    screen_trials_path = endpoint_summary_path.parent / "endpoint_trials.jsonl"
    screen_rows = _jsonl(screen_trials_path)
    eligible = [row for row in endpoint_summary["poses"] if row["eligible_endpoint_gate"]]
    if args.limit_poses is not None:
        eligible = eligible[:args.limit_poses]
    sources, source_paths = _endpoint_sources()
    cfg_hash = config_hash([
        cfg_w_path, phase2_path, endpoint_summary_path, screen_trials_path, *source_paths,
        ROOT / "seqgrasp" / "experiments" / "static_wrist.py",
        ROOT / "scripts" / "map_phase2w_wrist_geometry.py",
    ])
    output = ROOT / cfg_w.output_dir / "wrist_geometry" / args.stage / cfg_hash[:12]
    store = IncrementalJsonlStore(output / "pose_results.jsonl", 30.0, 0.05)
    completed = store.completed_ids()
    payloads = []
    for pose_summary in eligible:
        trial_id = stable_trial_id("phase2W-wrist-geometry", pose_summary["pose"]["pose_id"])
        if trial_id not in completed:
            payloads.append((pose_summary, screen_rows, sources, base_cfg, phase2, cfg_w))
    workers = min(args.workers or max(1, (os.cpu_count() or 1) // 2), cfg_w.wrist_search.maximum_workers)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, result in enumerate(executor.map(_evaluate_pose, payloads), start=1):
            store.append(result)
            print(
                f"Phase 2W {args.stage} wrist geometry: {len(completed) + index}/{len(eligible)} "
                f"pose={result['pose']['pose_id']} opposition={result['collision_free_opposition_count']} "
                f"positive_epsilon={result['positive_ferrari_canny_count']}",
                flush=True,
            )
    pose_ids = {row["pose"]["pose_id"] for row in eligible}
    results = [row for row in store.records() if row["pose"]["pose_id"] in pose_ids]
    ranked = sorted(results, key=lambda row: (_ranking(row), row["pose"]["pose_id"]), reverse=True)
    top_five = ranked[:cfg_w.wrist_search.top_coarse_for_refinement]
    parent_payload = {
        "selection_basis": "Phase 2W Section 14 pre-outcome lexicographic geometry ranking",
        "poses": [row["pose"] for row in top_five],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "refinement_parent_poses.json").write_text(json.dumps(parent_payload, indent=2), encoding="utf-8")
    summary = {
        "stage": args.stage,
        "endpoint_eligible_orientation_count": len(eligible),
        "evaluated_orientation_count": len(results),
        "collision_free_opposition_orientation_count": sum(row["collision_free_opposition_count"] > 0 for row in results),
        "positive_ferrari_canny_orientation_count": sum(row["positive_ferrari_canny_count"] > 0 for row in results),
        "common_nonzero_access_orientation_count": sum(
            row.get("common_access") is not None
            and all(row["common_access"][group]["access_fraction"] > 0 for group in GROUPS)
            for row in results
        ),
        "ranked_pose_ids": [row["pose"]["pose_id"] for row in ranked],
        "top_five_for_refinement": [row["pose"]["pose_id"] for row in top_five],
        "poses": results,
        "config_hash": cfg_hash,
        "git_commit_sha": git_commit_sha(ROOT),
        "resume_command": f"python scripts/map_phase2w_wrist_geometry.py --stage {args.stage} --workers {workers} --endpoint-summary {endpoint_summary_path}",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "evaluated_orientation_count": len(results),
        "collision_free_opposition_orientation_count": summary["collision_free_opposition_orientation_count"],
        "positive_ferrari_canny_orientation_count": summary["positive_ferrari_canny_orientation_count"],
        "common_nonzero_access_orientation_count": summary["common_nonzero_access_orientation_count"],
        "top_five_for_refinement": summary["top_five_for_refinement"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
