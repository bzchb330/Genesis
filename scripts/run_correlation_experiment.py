#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "seqgrasp-matplotlib"))
import matplotlib.pyplot as plt
import numpy as np

from seqgrasp.config import ROOT, load_configs
from seqgrasp.experiments.metadata import git_commit_sha
from seqgrasp.experiments.b_workspace import (
    analyze_B_geometry_state,
    free_fingertip_workspace_clouds,
    stratified_representative_ids,
)
from seqgrasp.experiments.resource_components import RESOURCE_RECORDS_FILENAME
from seqgrasp.experiments.resumable import IncrementalJsonlStore
from seqgrasp.experiments.second_grasp import (
    OUTCOMES,
    correlation_trial_id,
    placement_candidate,
    run_second_grasp_trial,
)
from seqgrasp.phase2_config import load_phase2_config


def _dataset_dir(root: Path) -> Path:
    candidates = []
    for path in root.glob("*/accepted_grasps.jsonl"):
        candidates.append((len(path.read_text(encoding="utf-8").splitlines()), path.parent))
    if not candidates:
        raise FileNotFoundError("no first-grasp dataset found")
    return max(candidates, key=lambda item: item[0])[1]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _evaluate_trial_task(payload):
    trial_id, record, resource_record, phase2, placement_index, tactile_path, pilot_only, commit = payload
    started = time.perf_counter()
    result = run_second_grasp_trial(record, resource_record, phase2, placement_index, tactile_path)
    result.update({
        "trial_id": trial_id,
        "grasp_id": record["grasp_id"],
        "placement_index": placement_index,
        "ferrari_canny_epsilon": record["ferrari_canny_epsilon"],
        "commanded_finger_subset": record["commanded_finger_subset"],
        "runtime_seconds": time.perf_counter() - started,
        "git_commit_sha": commit,
        "config_hash": record["config_hash"],
        "pilot_only": bool(pilot_only),
    })
    return result


def geometry_preflight(accepted, resources, phase2, output: Path) -> dict:
    cfg = load_configs()
    by_resource = {row["grasp_id"]: row for row in resources}
    representative_ids = stratified_representative_ids(
        accepted, resources, phase2.second_grasp.representative_grasp_count,
    )
    representatives = [row for row in accepted if row["grasp_id"] in representative_ids]
    if not representatives:
        raise RuntimeError("accepted first grasps are required for B geometry preflight")
    states = {}
    for record in representatives:
        enriched = {**record, **by_resource[record["grasp_id"]]}
        states[record["grasp_id"]] = free_fingertip_workspace_clouds(
            enriched, phase2.resources, phase2.second_grasp.geometry_workspace_samples,
            phase2.second_grasp.seed,
        )
    results = []
    for placement_index in range(phase2.second_grasp.placement_preflight_count):
        placement = placement_candidate(cfg, phase2.second_grasp, placement_index)
        metrics = []
        for record in representatives:
            state = states[record["grasp_id"]]
            metrics.append(analyze_B_geometry_state(*state, phase2.resources, placement))
        results.append({
            "placement_index": placement_index,
            "position_m": list(placement.position_m),
            "yaw_rad": placement.yaw_rad,
            "reachable_representative_grasps": sum(row["reachable"] for row in metrics),
            "minimum_free_fingertip_to_B_m": min(row["minimum_free_fingertip_to_B_m"] for row in metrics),
            "median_minimum_free_fingertip_to_B_m": float(np.median([row["minimum_free_fingertip_to_B_m"] for row in metrics])),
            "maximum_reachable_free_finger_count": max(row["reachable_free_finger_count"] for row in metrics),
            "initial_collision_A_grasps": sum(row["initial_collision_A"] for row in metrics),
            "initial_collision_hand_grasps": sum(row["initial_collision_hand"] for row in metrics),
            "inside_measured_free_palm_grasps": sum(row["inside_measured_free_palm_region"] for row in metrics),
        })
    output.mkdir(parents=True, exist_ok=True)
    (output / "geometry_preflight.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    plt.style.use(ROOT / "configs" / "phase2_publication.mplstyle")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    xy = np.asarray([row["position_m"][:2] for row in results])
    for ax, key, label in (
        (axes[0], "reachable_representative_grasps", "reachable retained-grasp geometries"),
        (axes[1], "median_minimum_free_fingertip_to_B_m", "median minimum distance [m]"),
    ):
        values = np.asarray([row[key] for row in results])
        points = ax.scatter(xy[:, 0], xy[:, 1], c=values, cmap="viridis", s=18)
        ax.set(xlabel="B centre x [m]", ylabel="B centre y [m]")
        fig.colorbar(points, ax=ax, label=label)
    fig.savefig(output / "B_geometry_preflight_panel.pdf")
    publication = ROOT / "docs" / "figures" / "phase2" / "B_geometry_preflight_panel.pdf"
    publication.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(publication)
    plt.close(fig)
    summary = {
        "placements": len(results),
        "representative_grasps": len(representatives),
        "representative_grasp_ids": representative_ids,
        "reachable_placements_any_representative": sum(row["reachable_representative_grasps"] > 0 for row in results),
        "unreachable_placements_all_representatives": sum(row["reachable_representative_grasps"] == 0 for row in results),
        "reachable_grasp_pose_pairs": sum(row["reachable_representative_grasps"] for row in results),
        "total_grasp_pose_pairs": len(results) * len(representatives),
        "reachable_pair_fraction": float(np.mean([row["reachable_representative_grasps"] / len(representatives) for row in results])),
        "minimum_distance_m": {
            "min": float(np.min([row["minimum_free_fingertip_to_B_m"] for row in results])),
            "median": float(np.median([row["median_minimum_free_fingertip_to_B_m"] for row in results])),
            "max": float(np.max([row["median_minimum_free_fingertip_to_B_m"] for row in results])),
        },
        "reachable_free_finger_count_max": max(row["maximum_reachable_free_finger_count"] for row in results),
        "invalid_overlap_pair_fraction": float(sum(row["initial_collision_A_grasps"] + row["initial_collision_hand_grasps"] for row in results) / (len(results) * len(representatives))),
        "status": "PHASE2_B_WORKSPACE_GEOMETRY_BLOCKED" if not any(row["reachable_representative_grasps"] > 0 for row in results) else (
            "PASS" if any(row["reachable_representative_grasps"] == 0 for row in results) and sum(row["initial_collision_A_grasps"] + row["initial_collision_hand_grasps"] for row in results) < len(results) * len(representatives) / 2 else "FAIL_NONDEGENERATE_GATE"
        ),
    }
    (output / "geometry_preflight_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run resumable Phase 2 second-grasp correlation trials")
    parser.add_argument("--config", default="configs/phase2_physics_validation.yaml")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--pilot", action="store_true", help="run the excluded 20x3 engineering pilot")
    parser.add_argument("--limit", type=int, help="engineering validation limit; production omits this")
    args = parser.parse_args()
    phase2, _ = load_phase2_config(ROOT / args.config)
    workers = min(args.workers or max(1, (os.cpu_count() or 1) // 2), 8)
    dataset_dir = _dataset_dir(ROOT / phase2.persistence.output_dir / "grasp_dataset")
    accepted = _read_jsonl(dataset_dir / "accepted_grasps.jsonl")
    resources = _read_jsonl(dataset_dir / RESOURCE_RECORDS_FILENAME)
    planned = len(accepted) * phase2.second_grasp.trials_per_grasp
    correlation_dir = dataset_dir / "correlation"
    mode_dir = correlation_dir / ("pilot" if args.pilot else "formal")
    existing = _read_jsonl(mode_dir / "trials.jsonl")
    measured = [float(row["runtime_seconds"]) for row in existing if row.get("runtime_seconds")]
    seconds_per_trial = float(np.median(measured)) if measured else 10.0
    estimate = {
        "planned_trials": planned,
        "available_first_grasps": len(accepted),
        "available_resource_records": len(resources),
        "completed_trials": len(existing),
        "worker_count": workers,
        "simulation_steps_per_trial": phase2.second_grasp.approach_steps + phase2.second_grasp.close_steps + phase2.second_grasp.final_hold_steps,
        "estimated_seconds_per_trial": seconds_per_trial,
        "estimated_wall_time_hours": planned * seconds_per_trial / workers / 3600,
        "estimated_storage_GB": planned * 0.00015,
    }
    if args.dry_run:
        print(json.dumps(estimate, indent=2))
        return 0
    by_resource = {row["grasp_id"]: row for row in resources}
    preflight = geometry_preflight(accepted, resources, phase2, correlation_dir / "preflight")
    print(json.dumps(preflight, indent=2), flush=True)
    if preflight["status"] != "PASS":
        return 3
    if args.preflight_only:
        return 0
    if len(accepted) < phase2.required_for_later_parts.accepted_grasp_target or any(row["grasp_id"] not in by_resource for row in accepted):
        print(f"DATASET_INCOMPLETE accepted={len(accepted)} resources={len(resources)}")
        return 2
    if args.pilot:
        ids = set(stratified_representative_ids(accepted, resources, 20))
        accepted = [row for row in accepted if row["grasp_id"] in ids]
    if args.limit is not None:
        accepted = accepted[:args.limit]
    mode_planned = len(accepted) * (3 if args.pilot else phase2.second_grasp.trials_per_grasp)
    store = IncrementalJsonlStore(mode_dir / "trials.jsonl", phase2.persistence.lock_timeout_seconds, phase2.persistence.lock_poll_seconds)
    completed = store.completed_ids()
    tasks = []
    for record in accepted:
        placement_count = 3 if args.pilot else phase2.second_grasp.trials_per_grasp
        for placement_index in range(placement_count):
            trial_id = correlation_trial_id(record, placement_index, args.pilot)
            if trial_id not in completed:
                tasks.append((trial_id, record, placement_index))

    commit = git_commit_sha(ROOT)
    payloads = [
        (
            trial_id, record, by_resource[record["grasp_id"]], phase2, placement_index,
            mode_dir / "tactile" / f"{trial_id.split(':')[-1]}.npz", args.pilot, commit,
        )
        for trial_id, record, placement_index in tasks
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        buffered = []
        for index, result in enumerate(executor.map(_evaluate_trial_task, payloads), start=1):
            buffered.append(result)
            if len(buffered) == workers or index == len(tasks):
                store.append_many(buffered)
                buffered.clear()
            if index % workers == 0 or index == len(tasks):
                print(f"correlation trials: {len(completed) + index}/{len(accepted) * (3 if args.pilot else phase2.second_grasp.trials_per_grasp)}", flush=True)
    records = store.records()
    summary = {"completed": len(records), "planned": mode_planned, "pilot_only": bool(args.pilot), "outcomes": dict(Counter(row["outcome"] for row in records))}
    (mode_dir / "correlation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    assert set(summary["outcomes"]).issubset(OUTCOMES)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
