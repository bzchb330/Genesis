#!/usr/bin/env python
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
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
from seqgrasp.phase2r_config import load_phase2r_config


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _latest_matching() -> tuple[Path, list[dict]]:
    candidates = [(path.stat().st_mtime_ns, path) for path in (ROOT / "outputs" / "phase2R" / "matching").rglob("matched_states.jsonl")]
    if not candidates:
        raise FileNotFoundError("no Phase 2R matched states found")
    path = max(candidates)[1]
    return path, _read_jsonl(path)


def _placement(region: dict, region_index: int, placement_index: int, seed: int) -> BPlacement:
    rng = np.random.default_rng(np.random.SeedSequence([seed, region_index, placement_index]))
    bounds = region["center_bounds_m"]
    position = tuple(float(rng.uniform(*bounds[axis])) for axis in "xyz")
    yaw = float(rng.uniform(*region["yaw_bounds_rad"]))
    quaternion = tuple(float(value) for value in Rotation.from_euler("z", yaw).as_quat(scalar_first=True))
    return BPlacement(placement_index, position, quaternion, yaw)


def _state_task(payload):
    row, regions, phase2, phase2r = payload
    cfg = load_configs()
    enriched = dict(row)
    enriched["grasp_id"] = row["grasp_state_id"]
    enriched.setdefault("initial_palm_position_m", list(cfg.hand.mount_pos))
    enriched.setdefault("initial_palm_quaternion", list(cfg.hand.mount_quat))
    state = free_fingertip_workspace_clouds(
        enriched, phase2.resources, phase2.second_grasp.geometry_workspace_samples,
        phase2r.second_grasp.geometry_seed,
    )
    results = []
    for region_index, region in enumerate(regions):
        for placement_index in range(phase2r.second_grasp.geometry_placements_per_region):
            result = analyze_B_geometry_state(
                *state, phase2.resources,
                _placement(region, region_index, placement_index, phase2r.second_grasp.geometry_seed),
            )
            results.append({"region": region["name"], "placement_index": placement_index, **result})
    return {
        "grasp_state_id": row["grasp_state_id"],
        "grasp_state_type": row["grasp_state_type"],
        "matched_pair_id": row["matched_pair_id"],
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Select one common Phase 2R B region using geometry only")
    parser.add_argument("--config", default="configs/phase2R_palmar_vs_fingertip.yaml")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    phase2r, phase2r_path = load_phase2r_config(ROOT / args.config)
    phase2, phase2_path = load_phase2_config()
    matched_path, states = _latest_matching()
    regions_path = ROOT / "configs" / "phase2_6_b_only_graspable_regions.yaml"
    regions = yaml.safe_load(regions_path.read_text(encoding="utf-8"))["regions"]
    cfg_hash = config_hash([phase2r_path, phase2_path, matched_path, regions_path])
    output = ROOT / phase2r.output_dir / "diagnostics" / "B_region_selection" / cfg_hash[:12]
    output.mkdir(parents=True, exist_ok=True)
    workers = min(args.workers or max(1, (os.cpu_count() or 1) // 2), phase2r.state.maximum_workers)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        state_results = []
        for index, result in enumerate(executor.map(_state_task, ((row, regions, phase2, phase2r) for row in states)), start=1):
            state_results.append(result)
            if index % workers == 0 or index == len(states):
                print(f"B geometry states: {index}/{len(states)}", flush=True)
    (output / "geometry_access.json").write_text(json.dumps(state_results, indent=2), encoding="utf-8")
    summaries = []
    for region in regions:
        by_type = {}
        for state_type in ("FINGERTIP", "PALMAR_SECURED"):
            rows = [result for state in state_results if state["grasp_state_type"] == state_type for result in state["results"] if result["region"] == region["name"]]
            by_type[state_type] = {
                "state_placement_pairs": len(rows),
                "reachable_pairs": sum(row["reachable"] for row in rows),
                "access_fraction": float(np.mean([row["reachable"] for row in rows])),
                "states_with_any_access": sum(
                    any(item["reachable"] for item in state["results"] if item["region"] == region["name"])
                    for state in state_results if state["grasp_state_type"] == state_type
                ),
                "initial_A_overlap_pairs": sum(row["initial_collision_A"] for row in rows),
                "initial_hand_overlap_pairs": sum(row["initial_collision_hand"] for row in rows),
            }
        summaries.append({
            "region": region["name"],
            "B_only_robustness_fraction": region["B_only_robustness_fraction"],
            "groups": by_type,
            "minimum_group_access_fraction": min(by_type[group]["access_fraction"] for group in by_type),
            "zero_initial_A_overlap": all(by_type[group]["initial_A_overlap_pairs"] == 0 for group in by_type),
        })
    eligible = [
        row for row in summaries
        if row["zero_initial_A_overlap"]
        and all(row["groups"][group]["access_fraction"] > 0.0 for group in ("FINGERTIP", "PALMAR_SECURED"))
    ]
    selected = max(eligible, key=lambda row: (row["minimum_group_access_fraction"], row["B_only_robustness_fraction"], row["region"])) if eligible else None
    summary = {
        "status": "PASS" if selected is not None else "PHASE2R_NO_COMMON_B_REGION",
        "selection_basis": "maximize minimum matched-group geometric access; B-only evidence; zero A overlap; no dynamic outcomes",
        "regions": summaries,
        "selected_region": None if selected is None else selected["region"],
        "config_hash": cfg_hash,
        "git_commit_sha": git_commit_sha(ROOT),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if selected is None:
        return 3
    region = next(row for row in regions if row["name"] == selected["region"])
    frozen = {
        "experiment_id": phase2r.experiment_id,
        "selection_basis": summary["selection_basis"],
        "source_phase2_6_region": region["name"],
        "center_bounds_m": region["center_bounds_m"],
        "yaw_bounds_rad": region["yaw_bounds_rad"],
        "vertical_cylinder": True,
        "fixture_behavior": "kinematic free-joint pose support until scripted release; unsupported final hold",
        "geometry_seed_namespace": phase2r.second_grasp.geometry_seed,
        "calibration_seed_namespace": phase2r.second_grasp.calibration_seed,
        "formal_seed_namespace": phase2r.second_grasp.formal_seed,
        "B_only_robustness_fraction": region["B_only_robustness_fraction"],
        "geometry_access": selected["groups"],
        "config_hash": cfg_hash,
        "git_commit_sha": git_commit_sha(ROOT),
    }
    frozen_path = ROOT / "configs" / "phase2R_frozen_B_distribution.yaml"
    frozen_path.write_text(yaml.safe_dump(frozen, sort_keys=False), encoding="utf-8")
    report = ROOT / "docs" / "PHASE2R_B_DISTRIBUTION_FREEZE.md"
    report.write_text(
        "# Phase 2R common B distribution freeze\n\n"
        "The region was selected before any Phase 2R dynamic second-grasp outcomes. Selection used only the "
        "three Phase 2.6 B-only-positive regions, matched-state geometry, and initial-overlap checks.\n\n"
        f"- Selected source region: `{region['name']}`\n"
        f"- xyz bounds [m]: `{json.dumps(region['center_bounds_m'], sort_keys=True)}`\n"
        f"- yaw bounds [rad]: `{json.dumps(region['yaw_bounds_rad'])}`\n"
        f"- B-only robustness evidence: {region['B_only_robustness_fraction']}\n"
        f"- Geometry access: `{json.dumps(selected['groups'], sort_keys=True)}`\n"
        "- Fixture: kinematic free-joint pose support until scripted release; final hold unsupported\n"
        f"- Seed namespaces: geometry {phase2r.second_grasp.geometry_seed}, calibration {phase2r.second_grasp.calibration_seed}, formal {phase2r.second_grasp.formal_seed}\n"
        f"- Config hash: `{cfg_hash}`\n- Git SHA at freeze: `" + git_commit_sha(ROOT) + "`\n\n"
        "The exact same frozen distribution and seeds apply to both endpoint-state groups. It may not be changed based on dynamic outcomes.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
