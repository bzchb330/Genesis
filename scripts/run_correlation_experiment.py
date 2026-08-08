#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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
from seqgrasp.experiments.resource_components import RESOURCE_RECORDS_FILENAME
from seqgrasp.experiments.resumable import IncrementalJsonlStore, stable_trial_id
from seqgrasp.experiments.second_grasp import (
    OUTCOMES,
    placement_candidate,
    placement_is_approachable,
    placement_is_valid,
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


def _representatives(accepted: list[dict]) -> list[dict]:
    selected = []
    masks = sorted({tuple(row["occupied_finger_mask"]) for row in accepted})
    for mask in masks:
        group = sorted(
            (row for row in accepted if tuple(row["occupied_finger_mask"]) == mask),
            key=lambda row: row["ferrari_canny_epsilon"],
        )
        selected.append(group[len(group) // 2])
    ordered = sorted(accepted, key=lambda row: row["ferrari_canny_epsilon"])
    for quantile in (0.0, 0.5, 1.0):
        row = ordered[round(quantile * (len(ordered) - 1))]
        if row["grasp_id"] not in {item["grasp_id"] for item in selected}:
            selected.append(row)
    return selected


def geometry_preflight(accepted, resources, phase2, output: Path) -> dict:
    cfg = load_configs()
    by_grasp = {row["grasp_id"]: row for row in accepted}
    by_grasp.update({row["grasp_id"]: {**by_grasp.get(row["grasp_id"], {}), **row} for row in resources})
    representatives = _representatives(accepted)
    if not representatives:
        raise RuntimeError("accepted first grasps are required for B geometry preflight")
    results = []
    for placement_index in range(phase2.second_grasp.placement_preflight_count):
        placement = placement_candidate(cfg, phase2.second_grasp, placement_index)
        valid_count = 0
        approachable_count = 0
        for record in representatives:
            valid, _ = placement_is_valid(record, placement, phase2.second_grasp.maximum_penetration_m)
            valid_count += int(valid)
            if valid and placement_is_approachable(record, placement, by_grasp[record["grasp_id"]]["occupied_finger_mask"]):
                approachable_count += 1
        results.append({
            "placement_index": placement_index,
            "position_m": list(placement.position_m),
            "yaw_rad": placement.yaw_rad,
            "valid_representative_grasps": valid_count,
            "approachable_representative_grasps": approachable_count,
        })
    output.mkdir(parents=True, exist_ok=True)
    (output / "geometry_preflight.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    plt.style.use(ROOT / "configs" / "phase2_publication.mplstyle")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharex=True, sharey=True)
    xy = np.asarray([row["position_m"][:2] for row in results])
    for ax, key, label in (
        (axes[0], "valid_representative_grasps", "valid retained-grasp geometries"),
        (axes[1], "approachable_representative_grasps", "DLS-reachable retained grasps"),
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
        "valid_placements_any_representative": sum(row["valid_representative_grasps"] > 0 for row in results),
        "approachable_placements_any_representative": sum(row["approachable_representative_grasps"] > 0 for row in results),
        "status": "PHASE2_B_WORKSPACE_GEOMETRY_BLOCKED" if not any(row["approachable_representative_grasps"] > 0 for row in results) else "PASS",
    }
    (output / "geometry_preflight_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run resumable Phase 2 second-grasp correlation trials")
    parser.add_argument("--config", default="configs/phase2_physics_validation.yaml")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--limit", type=int, help="engineering validation limit; production omits this")
    args = parser.parse_args()
    phase2, _ = load_phase2_config(ROOT / args.config)
    workers = min(args.workers or max(1, (os.cpu_count() or 1) // 2), 8)
    planned = phase2.required_for_later_parts.accepted_grasp_target * phase2.second_grasp.trials_per_grasp
    dataset_dir = _dataset_dir(ROOT / phase2.persistence.output_dir / "grasp_dataset")
    accepted = _read_jsonl(dataset_dir / "accepted_grasps.jsonl")
    resources = _read_jsonl(dataset_dir / RESOURCE_RECORDS_FILENAME)
    correlation_dir = dataset_dir / "correlation"
    existing = _read_jsonl(correlation_dir / "trials.jsonl")
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
    target_grasps = phase2.required_for_later_parts.accepted_grasp_target
    if len(accepted) < target_grasps or any(row["grasp_id"] not in by_resource for row in accepted[:target_grasps]):
        print(f"DATASET_INCOMPLETE accepted={len(accepted)}/{target_grasps} resources={len(resources)}/{target_grasps}")
        return 2
    accepted = accepted[:target_grasps]
    if args.limit is not None:
        accepted = accepted[:args.limit]
    store = IncrementalJsonlStore(correlation_dir / "trials.jsonl", phase2.persistence.lock_timeout_seconds, phase2.persistence.lock_poll_seconds)
    completed = store.completed_ids()
    tasks = []
    for record in accepted:
        for placement_index in range(phase2.second_grasp.trials_per_grasp):
            trial_id = stable_trial_id("phase2-second-grasp", {"grasp_id": record["grasp_id"], "placement_index": placement_index, "config_hash": record["config_hash"]})
            if trial_id not in completed:
                tasks.append((trial_id, record, placement_index))

    def evaluate(task):
        trial_id, record, placement_index = task
        started = time.perf_counter()
        tactile_path = correlation_dir / "tactile" / f"{trial_id.split(':')[-1]}.npz"
        result = run_second_grasp_trial(record, by_resource[record["grasp_id"]], phase2, placement_index, tactile_path)
        result.update({
            "trial_id": trial_id,
            "grasp_id": record["grasp_id"],
            "placement_index": placement_index,
            "ferrari_canny_epsilon": record["ferrari_canny_epsilon"],
            "commanded_finger_subset": record["commanded_finger_subset"],
            "runtime_seconds": time.perf_counter() - started,
            "git_commit_sha": git_commit_sha(ROOT),
            "config_hash": record["config_hash"],
        })
        return result

    with ThreadPoolExecutor(max_workers=workers) as executor:
        buffered = []
        for index, result in enumerate(executor.map(evaluate, tasks), start=1):
            buffered.append(result)
            if len(buffered) == workers or index == len(tasks):
                store.append_many(buffered)
                buffered.clear()
            if index % workers == 0 or index == len(tasks):
                print(f"correlation trials: {len(completed) + index}/{len(accepted) * phase2.second_grasp.trials_per_grasp}", flush=True)
    records = store.records()
    summary = {"completed": len(records), "planned": planned, "outcomes": dict(Counter(row["outcome"] for row in records))}
    (correlation_dir / "correlation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    assert set(summary["outcomes"]).issubset(OUTCOMES)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
