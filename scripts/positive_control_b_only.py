#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

import yaml

from seqgrasp.config import ROOT
from seqgrasp.experiments.metadata import config_hash, git_commit_sha
from seqgrasp.experiments.phase2_5_trajectory import (
    b_only_lexicographic_key,
    run_b_acquisition_trajectory,
    sample_b_only_trajectory,
    trajectory_profile_dict,
)
from seqgrasp.experiments.resumable import IncrementalJsonlStore, stable_trial_id
from seqgrasp.phase2_5_config import load_phase2_5_config


def _evaluate(payload):
    cfg, index, cfg_hash, commit = payload
    trajectory = sample_b_only_trajectory(cfg, index)
    summary, _ = run_b_acquisition_trajectory(cfg, trajectory)
    summary.update({
        "trial_id": stable_trial_id("phase2.5-b-only", {"candidate_index": index, "config_hash": cfg_hash}),
        "experiment_id": "phase2_5_calibration",
        "calibration_only": True,
        "config_hash": cfg_hash,
        "git_commit_sha": commit,
    })
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Resumable structured B-only positive-control search")
    parser.add_argument("--config", default="configs/phase2_5_second_grasp_calibration.yaml")
    parser.add_argument("--candidates", type=int)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    cfg, source = load_phase2_5_config(ROOT / args.config)
    count = args.candidates or cfg.trajectory_search.initial_candidate_count
    if count not in (cfg.trajectory_search.initial_candidate_count, cfg.trajectory_search.expanded_candidate_count):
        raise ValueError("candidate count must be the configured initial or expanded search budget")
    workers = min(args.workers or max(1, (os.cpu_count() or 1) // 2), cfg.maximum_workers)
    cfg_hash = config_hash([source, ROOT / cfg.frozen_phase2_config, ROOT / "configs/hand_allegro.yaml", ROOT / "configs/scene_two_object.yaml", ROOT / "configs/task_sequential.yaml"])
    output = ROOT / cfg.output_dir / "b_only_search" / cfg_hash[:12]
    store = IncrementalJsonlStore(output / "candidate_results.jsonl", 30.0, 0.05)
    completed = store.completed_ids()
    pending = [
        index for index in range(count)
        if stable_trial_id("phase2.5-b-only", {"candidate_index": index, "config_hash": cfg_hash}) not in completed
    ]
    commit = git_commit_sha(ROOT)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        buffered = []
        for completed_index, result in enumerate(executor.map(_evaluate, ((cfg, index, cfg_hash, commit) for index in pending)), start=1):
            buffered.append(result)
            if len(buffered) == workers or completed_index == len(pending):
                store.append_many(buffered); buffered.clear()
            if completed_index % (workers * 4) == 0 or completed_index == len(pending):
                print(f"B-only search: {len(completed) + completed_index}/{count}", flush=True)
    records = [
        row for row in store.records()
        if row.get("config_hash") == cfg_hash and int(row["candidate_index"]) < count
    ]
    ranked = sorted(records, key=b_only_lexicographic_key, reverse=True)
    successes = [row for row in ranked if row["B_acquired"]]
    profile_dir = ROOT / "configs" / "grasps"
    for profile_index, row in enumerate(successes[:cfg.trajectory_search.maximum_saved_profiles], start=1):
        target = profile_dir / f"b_only_candidate_{profile_index:02d}.yaml"
        payload = trajectory_profile_dict(sample_b_only_trajectory(cfg, int(row["candidate_index"])))
        payload["source_candidate_index"] = int(row["candidate_index"])
        payload["calibration_only"] = True
        target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    summary = {
        "status": "PASS" if successes else ("PHASE2_5_B_ONLY_CONTROL_FAILED" if count == cfg.trajectory_search.expanded_candidate_count else "EXPANSION_REQUIRED"),
        "candidate_count": count,
        "completed_candidates": len(records),
        "successful_B_only_trajectories": len(successes),
        "failure_mechanisms": dict(Counter(row["failure_mechanism"] for row in records if not row["B_acquired"])),
        "best": ranked[0] if ranked else None,
        "saved_profiles": min(len(successes), cfg.trajectory_search.maximum_saved_profiles),
        "config_hash": cfg_hash,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if successes else (3 if count == cfg.trajectory_search.expanded_candidate_count else 2)


if __name__ == "__main__":
    raise SystemExit(main())
