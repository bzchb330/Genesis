#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

from seqgrasp.config import ROOT
from seqgrasp.experiments.grasp_sampling import evaluate_extension_candidate
from seqgrasp.experiments.metadata import git_commit_sha
from seqgrasp.experiments.resumable import IncrementalJsonlStore, stable_trial_id
from seqgrasp.phase2_config import load_phase2_config


def _dataset_dir(root: Path) -> Path:
    candidates = [(len(path.read_text(encoding="utf-8").splitlines()), path.parent) for path in root.glob("*/accepted_grasps.jsonl")]
    return max(candidates, key=lambda item: item[0])[1]


def _evaluate(payload):
    phase2, index, anchors = payload
    return evaluate_extension_candidate(phase2, index, anchors)


def main() -> int:
    parser = argparse.ArgumentParser(description="Resumable targeted occupied-count-two Phase 2 dataset extension")
    parser.add_argument("--config", default="configs/phase2_physics_validation.yaml")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--maximum-attempts", type=int, default=20_000)
    parser.add_argument("--target-occupied-two", type=int, default=30)
    args = parser.parse_args()
    phase2, _ = load_phase2_config(ROOT / args.config)
    workers = min(args.workers or max(1, (os.cpu_count() or 1) // 2), 8)
    dataset = _dataset_dir(ROOT / phase2.persistence.output_dir / "grasp_dataset")
    accepted_store = IncrementalJsonlStore(dataset / "accepted_grasps.jsonl", phase2.persistence.lock_timeout_seconds, phase2.persistence.lock_poll_seconds)
    attempts_store = IncrementalJsonlStore(dataset / "extension_candidate_attempts.jsonl", phase2.persistence.lock_timeout_seconds, phase2.persistence.lock_poll_seconds)
    accepted = accepted_store.records()
    original_anchors = [row for row in accepted[:200] if row["occupied_finger_count"] == 2 and row["occupied_finger_mask"][-1]]
    if not original_anchors:
        raise RuntimeError("no opposed occupied-count-two anchors are available")
    attempts = attempts_store.records()
    attempted = {int(row["extension_attempt_index"]) for row in attempts}
    next_attempt = 0
    while next_attempt in attempted:
        next_attempt += 1
    commit = git_commit_sha(ROOT)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        while sum(row["occupied_finger_count"] == 2 for row in accepted) < args.target_occupied_two and next_attempt < args.maximum_attempts and len(accepted) < 500:
            batch = list(range(next_attempt, min(next_attempt + workers, args.maximum_attempts)))
            results = list(executor.map(_evaluate, ((phase2, index, original_anchors) for index in batch)))
            additions = []
            for result in results:
                identity = {"extension_attempt_index": result["extension_attempt_index"], "generation_seed": phase2.dataset.seed}
                result.update({
                    "trial_id": stable_trial_id("phase2-grasp-extension-attempt", identity),
                    "config_hash": accepted[0]["config_hash"],
                    "git_commit_sha": commit,
                })
                if result["accepted"] and result["occupied_finger_count"] == 2 and len(accepted) < 500:
                    addition = dict(result)
                    addition["grasp_id"] = f"phase2_grasp_{len(accepted):04d}"
                    addition["trial_id"] = stable_trial_id("phase2-accepted-grasp", {"source_trial_id": result["trial_id"]})
                    accepted.append(addition)
                    additions.append(addition)
            attempts_store.append_many(results)
            accepted_store.append_many(additions)
            next_attempt = batch[-1] + 1
            if next_attempt % (workers * 10) == 0 or additions:
                print(f"extension attempts={next_attempt}/{args.maximum_attempts} accepted={len(accepted)} occupied2={sum(row['occupied_finger_count'] == 2 for row in accepted)}/{args.target_occupied_two}", flush=True)
    attempts = attempts_store.records()
    summary = {
        "status": "TARGET_REACHED" if sum(row["occupied_finger_count"] == 2 for row in accepted) >= args.target_occupied_two else "ATTEMPT_CAP_REACHED",
        "original_grasps_preserved": 200,
        "final_accepted_grasps": len(accepted),
        "additional_candidate_attempts": len(attempts),
        "occupied_finger_count_distribution": dict(Counter(str(row["occupied_finger_count"]) for row in accepted)),
        "attempted_sampling_mode_distribution": dict(Counter(row["sampling_mode"] for row in attempts)),
        "accepted_extension_sampling_mode_distribution": dict(Counter(row["sampling_mode"] for row in accepted[200:])),
        "attempted_commanded_subset_distribution": dict(Counter("+".join(row["commanded_finger_subset"]) for row in attempts)),
        "acceptance_thresholds_relaxed": False,
        "resumable_attempt_store": str((dataset / "extension_candidate_attempts.jsonl").relative_to(ROOT)).replace("\\", "/"),
    }
    (dataset / "extension_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
