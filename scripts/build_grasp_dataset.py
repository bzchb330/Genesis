#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import time

import numpy as np

from seqgrasp.config import ROOT
from seqgrasp.experiments.grasp_sampling import evaluate_candidate
from seqgrasp.experiments.metadata import config_hash, git_commit_sha
from seqgrasp.experiments.resumable import IncrementalJsonlStore, stable_trial_id
from seqgrasp.phase2_config import load_phase2_config


def _evaluate_task(payload):
    phase2, index = payload
    return evaluate_candidate(phase2, index)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the resumable Phase 2 first-grasp dataset")
    parser.add_argument("--config", default="configs/phase2_physics_validation.yaml")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--target", type=int)
    args = parser.parse_args()
    phase2, config_path = load_phase2_config(ROOT / args.config)
    target = phase2.required_for_later_parts.accepted_grasp_target if args.target is None else args.target
    workers = min(args.workers or max(1, (os.cpu_count() or 1) // 2), phase2.dataset.maximum_workers)
    profile_paths = [ROOT / path for path in phase2.dataset.proposal_profile_paths]
    cfg_hash = config_hash([
        config_path,
        *profile_paths,
        ROOT / "configs" / "hand_allegro.yaml",
        ROOT / "configs" / "scene_two_object.yaml",
        ROOT / "configs" / "task_sequential.yaml",
        ROOT / "configs" / "diagnostic_grasp_a.yaml",
    ])
    output = ROOT / phase2.persistence.output_dir / "grasp_dataset" / cfg_hash[:12]
    attempts_store = IncrementalJsonlStore(output / "candidate_attempts.jsonl", phase2.persistence.lock_timeout_seconds, phase2.persistence.lock_poll_seconds)
    accepted_store = IncrementalJsonlStore(output / "accepted_grasps.jsonl", phase2.persistence.lock_timeout_seconds, phase2.persistence.lock_poll_seconds)
    commit = git_commit_sha(ROOT)
    attempts_by_index = {int(record["attempt_index"]): record for record in attempts_store.records()}
    accepted = accepted_store.records()
    next_attempt = 0
    while next_attempt in attempts_by_index:
        next_attempt += 1
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        while len(accepted) < target and next_attempt < phase2.dataset.maximum_candidate_attempts:
            batch = list(range(next_attempt, min(next_attempt + workers, phase2.dataset.maximum_candidate_attempts)))
            results = list(executor.map(_evaluate_task, ((phase2, index) for index in batch)))
            accepted_batch = []
            for result in results:
                identity = {"attempt_index": result["attempt_index"], "generation_seed": phase2.dataset.seed, "config_hash": cfg_hash}
                result.update({
                    "trial_id": stable_trial_id("phase2-grasp-attempt", identity),
                    "config_hash": cfg_hash,
                    "git_commit_sha": commit,
                })
                if result["accepted"] and len(accepted) < target:
                    accepted_record = dict(result)
                    accepted_record["grasp_id"] = f"phase2_grasp_{len(accepted):04d}"
                    accepted_record["trial_id"] = stable_trial_id("phase2-accepted-grasp", {"source_trial_id": result["trial_id"]})
                    accepted.append(accepted_record)
                    accepted_batch.append(accepted_record)
            attempts_store.append_many(results)
            accepted_store.append_many(accepted_batch)
            next_attempt = batch[-1] + 1
            if next_attempt % (workers * 10) == 0 or len(accepted) >= target:
                print(f"attempts={next_attempt} accepted={len(accepted)}/{target}", flush=True)
                output.mkdir(parents=True, exist_ok=True)
                (output / "progress.json").write_text(json.dumps({
                    "status": "RUNNING" if len(accepted) < target else "COMPLETE",
                    "accepted_grasps": len(accepted), "target": target,
                    "candidate_attempts": next_attempt, "workers": workers,
                    "config_hash": cfg_hash, "git_commit_sha": commit,
                }, indent=2), encoding="utf-8")
    attempts = attempts_store.records()
    elapsed = time.perf_counter() - started
    epsilon = np.asarray([row["ferrari_canny_epsilon"] for row in accepted], dtype=float)
    palm_translation = np.asarray([row["palm_translation_perturbation_m"] for row in accepted], dtype=float)
    palm_orientation = np.asarray([row["palm_orientation_perturbation_deg"] for row in accepted], dtype=float)
    object_yaw = np.asarray([row["object_yaw_rad"] for row in accepted], dtype=float)
    summary = {
        "status": "COMPLETE" if len(accepted) >= target else "SAMPLING_EXHAUSTED",
        "accepted_grasps": len(accepted),
        "target": target,
        "candidate_attempts": len(attempts),
        "workers": workers,
        "runtime_seconds_this_invocation": elapsed,
        "config_hash": cfg_hash,
        "git_commit_sha": commit,
        "commanded_subset_distribution": dict(Counter("+".join(row["commanded_finger_subset"]) for row in accepted)),
        "commanded_subset_size_distribution": dict(Counter(str(len(row["commanded_finger_subset"])) for row in accepted)),
        "attempted_subset_distribution": dict(Counter("+".join(row["commanded_finger_subset"]) for row in attempts)),
        "attempted_subset_size_distribution": dict(Counter(str(len(row["commanded_finger_subset"])) for row in attempts)),
        "accepted_sampling_mode_distribution": dict(Counter(row["sampling_mode"] for row in accepted)),
        "occupied_finger_count_distribution": dict(Counter(str(row["occupied_finger_count"]) for row in accepted)),
        "ferrari_canny_epsilon": {
            "min": float(np.min(epsilon)), "max": float(np.max(epsilon)),
            "mean": float(np.mean(epsilon)), "std": float(np.std(epsilon)),
        } if len(epsilon) else {},
        "accepted_palm_translation_perturbation_m": {
            axis: {"min": float(np.min(palm_translation[:, index])), "max": float(np.max(palm_translation[:, index])), "std": float(np.std(palm_translation[:, index]))}
            for index, axis in enumerate("xyz")
        } if len(palm_translation) else {},
        "accepted_palm_orientation_perturbation_deg": {
            axis: {"min": float(np.min(palm_orientation[:, index])), "max": float(np.max(palm_orientation[:, index])), "std": float(np.std(palm_orientation[:, index]))}
            for index, axis in enumerate(("roll", "pitch", "yaw"))
        } if len(palm_orientation) else {},
        "accepted_object_yaw_rad": {
            "min": float(np.min(object_yaw)), "max": float(np.max(object_yaw)), "std": float(np.std(object_yaw)),
        } if len(object_yaw) else {},
        "rejection_reasons": dict(Counter(str(row.get("rejection_reason")) for row in attempts if not row["accepted"])),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
