#!/usr/bin/env python
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from seqgrasp.config import ROOT, load_configs
from seqgrasp.experiments.b_workspace import analyze_B_geometry_state, free_fingertip_workspace_clouds
from seqgrasp.experiments.metadata import config_hash, git_commit_sha
from seqgrasp.experiments.second_grasp import BPlacement
from seqgrasp.phase2_config import load_phase2_config
from seqgrasp.phase2tr_config import assert_index_thumb_free_topology, load_phase2tr_config


PLACEMENTS_PER_STATE = 50


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _largest(root: Path, name: str) -> tuple[Path, list[dict]]:
    candidates = [(len(_jsonl(path)), path.stat().st_mtime_ns, path) for path in root.rglob(name)]
    if not candidates:
        raise FileNotFoundError(f"missing {name} under {root}")
    path = max(candidates)[2]
    return path, _jsonl(path)


def _placement(bounds, index, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    position = tuple(float(rng.uniform(*bounds[axis])) for axis in "xyz")
    yaw = float(rng.uniform(-0.05, 0.05))
    quaternion = tuple(float(v) for v in Rotation.from_euler("z", yaw).as_quat(scalar_first=True))
    return BPlacement(index, position, quaternion, yaw)


def _state_task(payload):
    row, phase2, phase2tr, bounds = payload
    assert_index_thumb_free_topology(row)
    cfg = load_configs(scene_filename=phase2tr.scene_filename)
    enriched = dict(row)
    enriched["grasp_id"] = row["grasp_state_id"]
    state = free_fingertip_workspace_clouds(
        enriched, phase2.resources, phase2.second_grasp.geometry_workspace_samples,
        phase2tr.second_grasp.geometry_seed, base_cfg=cfg,
    )
    clouds = state[3]
    voxel = phase2.resources.workspace_voxel_size_m
    cloud_voxels = {
        finger: {tuple(np.floor(point / voxel).astype(np.int64)) for point in clouds[finger]}
        for finger in ("index", "thumb")
    }
    placements = [
        analyze_B_geometry_state(
            *state, phase2.resources,
            _placement(bounds, index, phase2tr.second_grasp.geometry_seed),
        )
        for index in range(PLACEMENTS_PER_STATE)
    ]
    return {
        "grasp_state_id": row["grasp_state_id"],
        "grasp_state_type": row["grasp_state_type"],
        "index_workspace_m3": len(cloud_voxels["index"]) * voxel ** 3,
        "thumb_workspace_m3": len(cloud_voxels["thumb"]) * voxel ** 3,
        "joint_index_thumb_workspace_m3": len(cloud_voxels["index"] & cloud_voxels["thumb"]) * voxel ** 3,
        "placements": placements,
    }


def main() -> int:
    phase2tr, tr_path = load_phase2tr_config()
    phase2, p_path = load_phase2_config()
    f_path, fingertip = _largest(ROOT / phase2tr.output_dir / "fingertip_states", "accepted_states.jsonl")
    p_state_path, palmar = _largest(ROOT / phase2tr.output_dir / "palmar_states", "accepted_states.jsonl")
    b_path, b_rows = _largest(ROOT / phase2tr.output_dir / "b_only_index_thumb", "candidate_results.jsonl")
    strict = [row for row in b_rows if row.get("strict_index_thumb_success")]
    if len(strict) < phase2tr.second_grasp.b_only_hard_minimum:
        raise RuntimeError("Phase 2T-R B-only gate has not passed")
    best = min(strict, key=lambda row: (row["maximum_B_translation_after_release_m"], row["maximum_B_orientation_after_release_rad"]))
    center = np.asarray(best["placement"]["position_m"], dtype=float)
    bounds = {axis: [float(center[i] - 0.0005), float(center[i] + 0.0005)] for i, axis in enumerate("xyz")}
    states = fingertip[:phase2tr.state_search.fingertip_target] + palmar[:phase2tr.state_search.palmar_target]
    cfg_hash = config_hash([tr_path, p_path, f_path, p_state_path, b_path, ROOT / "scripts" / "select_phase2tr_B_region.py"])
    output = ROOT / phase2tr.output_dir / "geometry" / cfg_hash[:12]
    output.mkdir(parents=True, exist_ok=True)
    workers = min(max(1, (os.cpu_count() or 1) // 2), phase2tr.state_search.maximum_workers)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        rows = []
        for index, result in enumerate(executor.map(_state_task, ((row, phase2, phase2tr, bounds) for row in states)), start=1):
            rows.append(result)
            if index % workers == 0 or index == len(states):
                print(f"Phase 2T-R geometry states: {index}/{len(states)}", flush=True)
    (output / "geometry_access.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    groups = {}
    for state_type in ("FINGERTIP", "PALMAR_SECURED"):
        group = [row for row in rows if row["grasp_state_type"] == state_type]
        placements = [item for row in group for item in row["placements"]]
        groups[state_type] = {
            "states": len(group),
            "state_placement_pairs": len(placements),
            "reachable_pairs": sum(item["reachable"] for item in placements),
            "access_fraction": float(np.mean([item["reachable"] for item in placements])),
            "states_with_any_access": sum(any(item["reachable"] for item in row["placements"]) for row in group),
            "initial_A_overlap_pairs": sum(item["initial_collision_A"] for item in placements),
            "initial_hand_overlap_pairs": sum(item["initial_collision_hand"] for item in placements),
            "index_workspace_m3_mean": float(np.mean([row["index_workspace_m3"] for row in group])),
            "thumb_workspace_m3_mean": float(np.mean([row["thumb_workspace_m3"] for row in group])),
            "joint_index_thumb_workspace_m3_mean": float(np.mean([row["joint_index_thumb_workspace_m3"] for row in group])),
        }
    eligible = all(groups[g]["access_fraction"] > 0 for g in groups) and all(
        groups[g]["initial_A_overlap_pairs"] == 0 and groups[g]["initial_hand_overlap_pairs"] == 0 for g in groups
    )
    summary = {
        "status": "PASS" if eligible else "PHASE2TR_NO_COMMON_B_REGION",
        "selection_basis": "validated native index+thumb B-only region; maximize shared geometry access; zero A/hand overlap; no A+B dynamic outcomes",
        "center_bounds_m": bounds,
        "yaw_bounds_rad": [-0.05, 0.05],
        "source_B_only_candidate_index": int(best["candidate_index"]),
        "source_phase2S_candidate_index": int(best["source_phase2S_candidate_index"]),
        "B_only_strict_successes": len(strict),
        "groups": groups,
        "config_hash": cfg_hash,
        "git_commit_sha": git_commit_sha(ROOT),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not eligible:
        return 3
    robust_summary_path = next((ROOT / phase2tr.output_dir / "b_only_index_thumb").rglob("summary.json"))
    robust = json.loads(robust_summary_path.read_text(encoding="utf-8"))
    frozen = {
        "experiment_id": phase2tr.experiment_id,
        "selection_basis": summary["selection_basis"],
        "center_bounds_m": bounds,
        "yaw_bounds_rad": [-0.05, 0.05],
        "vertical_cylinder": True,
        "fixture_behavior": "kinematic free-joint pose support until scripted release; unsupported final hold",
        "index_thumb_B_only_candidate_count": robust["candidate_count"],
        "index_thumb_B_only_strict_success_count": robust["strict_success_count"],
        "B_only_robustness_trials": robust["robustness_trial_count"],
        "B_only_robustness_successes": robust["robustness_success_count"],
        "B_only_robustness_fraction": robust["robustness_success_fraction"],
        "geometry_access": groups,
        "geometry_seed_namespace": phase2tr.second_grasp.geometry_seed,
        "calibration_seed_namespace": phase2tr.second_grasp.calibration_seed,
        "formal_seed_namespace": phase2tr.second_grasp.formal_seed,
        "config_hash": cfg_hash,
        "git_commit_sha": git_commit_sha(ROOT),
    }
    (ROOT / "configs" / "phase2TR_frozen_B_distribution.yaml").write_text(yaml.safe_dump(frozen, sort_keys=False), encoding="utf-8")
    (ROOT / "docs" / "PHASE2TR_B_DISTRIBUTION_FREEZE.md").write_text(
        "# Phase 2T-R B distribution freeze\n\n"
        "This region was frozen before any A+B dynamic calibration. Selection used only native index+thumb B-only validation, the new targeted endpoint states, and geometry/collision checks.\n\n"
        f"- xyz bounds [m]: `{json.dumps(bounds, sort_keys=True)}`\n"
        "- yaw bounds [rad]: `[-0.05, 0.05]`\n"
        f"- Strict index+thumb B-only evidence: {robust['strict_success_count']}/{robust['candidate_count']} candidates\n"
        f"- Perturbation robustness: {robust['robustness_success_count']}/{robust['robustness_trial_count']} ({robust['robustness_success_fraction']:.3f})\n"
        f"- Geometry access/collisions: `{json.dumps(groups, sort_keys=True)}`\n"
        f"- Config hash: `{cfg_hash}`\n"
        f"- Git SHA at freeze: `{git_commit_sha(ROOT)}`\n\n"
        "The identical region and seed placements apply to both endpoint groups. It must not be changed in response to calibration or formal outcomes.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
