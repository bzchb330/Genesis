#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

from seqgrasp.config import ROOT
from seqgrasp.experiments.metadata import config_hash, git_commit_sha
from seqgrasp.experiments.phase2s_sampling import (
    evaluate_phase2r_fingertip_seed,
    evaluate_phase2s_palmar_candidate,
    evaluate_supplemental_fingertip_candidate,
)
from seqgrasp.experiments.resumable import IncrementalJsonlStore, stable_trial_id
from seqgrasp.phase2_config import load_phase2_config
from seqgrasp.phase2s_config import load_phase2s_config


FINGERTIP_REPLAY_METHOD = "phase2R_candidate_revalidation_v1"
FINGERTIP_SUPPLEMENT_CAP = 6000


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _source_dataset() -> Path:
    candidates = [(len(_jsonl(path)), path) for path in (ROOT / "outputs" / "phase2" / "grasp_dataset").glob("*/accepted_grasps.jsonl")]
    if not candidates:
        raise FileNotFoundError("no Phase 2 candidate dataset available as proposal seeds")
    return max(candidates)[1]


def _summary(accepted: list[dict], attempts: list[dict], state_type: str, target: int, cap: int) -> dict:
    return {
        "grasp_state_type": state_type,
        "accepted_states": len(accepted),
        "target": target,
        "attempts": len(attempts),
        "attempt_cap": cap,
        "rejection_reasons": dict(Counter(str(row.get("rejection_reason")) for row in attempts if not row["accepted"])),
        "occupied_finger_count_distribution": dict(Counter(str(row["occupied_finger_count"]) for row in accepted)),
        "contact_topology_distribution": dict(Counter(
            "+".join(finger for finger, flag in zip(("index", "middle", "ring", "thumb"), row["occupied_finger_mask"]) if flag)
            for row in accepted
        )),
    }


def _replay_task(payload):
    return evaluate_phase2r_fingertip_seed(*payload)


def _supplement_task(payload):
    return evaluate_supplemental_fingertip_candidate(*payload)


def _palmar_task(payload):
    return evaluate_phase2s_palmar_candidate(*payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build resumable Phase 2S half-scale endpoint states")
    parser.add_argument("--config", default="configs/phase2S_half_scale_objects.yaml")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--fingertip-only", action="store_true")
    parser.add_argument("--palmar-only", action="store_true")
    parser.add_argument("--attempt-limit", type=int)
    args = parser.parse_args()
    if args.fingertip_only and args.palmar_only:
        parser.error("choose at most one state-only mode")
    phase2s, phase2s_path = load_phase2s_config(ROOT / args.config)
    phase2, phase2_path = load_phase2_config()
    source_path = _source_dataset()
    source = _jsonl(source_path)[:500]
    workers = min(args.workers or max(1, (os.cpu_count() or 1) // 2), phase2s.state.maximum_workers)
    cfg_hash = config_hash([
        phase2s_path, phase2_path, source_path, ROOT / "configs" / phase2s.scene_filename,
        ROOT / "configs" / "hand_allegro.yaml", *[ROOT / path for path in phase2s.state.proposal_profile_paths],
        ROOT / "seqgrasp" / "experiments" / "phase2s_sampling.py",
    ])
    root = ROOT / phase2s.output_dir
    run_key, commit = cfg_hash[:12], git_commit_sha(ROOT)

    if not args.palmar_only:
        state_dir = root / "fingertip_states" / FINGERTIP_REPLAY_METHOD / run_key
        attempts = IncrementalJsonlStore(state_dir / "attempts.jsonl", 30.0, 0.05)
        accepted = IncrementalJsonlStore(state_dir / "accepted_states.jsonl", 30.0, 0.05)
        completed = attempts.completed_ids()
        pending = [row for row in source if stable_trial_id("phase2S-fingertip-replay", row["grasp_id"]) not in completed]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for index, result in enumerate(executor.map(_replay_task, ((phase2s, phase2, row) for row in pending)), start=1):
                result.update({
                    "trial_id": stable_trial_id("phase2S-fingertip-replay", result["source_grasp_id"]),
                    "experiment_id": phase2s.experiment_id, "config_hash": cfg_hash, "git_commit_sha": commit,
                })
                attempts.append(result)
                if result["accepted"]:
                    accepted.append({**result, "trial_id": stable_trial_id("phase2S-fingertip-state", result["grasp_state_id"])})
                if index % workers == 0 or index == len(pending):
                    print(f"fingertip seed replay: {len(completed) + index}/{len(source)} accepted={len(accepted.records())}", flush=True)
        if len(accepted.records()) < phase2s.state.fingertip_target:
            attempted_variants = {int(row["supplemental_variant_index"]) for row in attempts.records() if "supplemental_variant_index" in row}
            cap = FINGERTIP_SUPPLEMENT_CAP if args.attempt_limit is None else min(args.attempt_limit, FINGERTIP_SUPPLEMENT_CAP)
            next_variant = next((value for value in range(cap) if value not in attempted_variants), cap)
            with ProcessPoolExecutor(max_workers=workers) as executor:
                while len(accepted.records()) < phase2s.state.fingertip_target and next_variant < cap:
                    batch = [value for value in range(next_variant, min(next_variant + workers, cap)) if value not in attempted_variants]
                    focused_source = next(row for row in source if row["grasp_id"] == "phase2_grasp_0040")
                    payloads = ((
                        phase2s, phase2,
                        focused_source if value % 5 != 0 else source[value % len(source)], value,
                    ) for value in batch)
                    results = list(executor.map(_supplement_task, payloads))
                    accepted_batch = []
                    for result in results:
                        result.update({
                            "trial_id": stable_trial_id("phase2S-fingertip-supplement", {"variant": result["supplemental_variant_index"], "config": cfg_hash}),
                            "experiment_id": phase2s.experiment_id, "config_hash": cfg_hash, "git_commit_sha": commit,
                        })
                        if result["accepted"] and len(accepted.records()) + len(accepted_batch) < phase2s.state.maximum_states_per_group:
                            accepted_batch.append({**result, "trial_id": stable_trial_id("phase2S-fingertip-state", result["grasp_state_id"])})
                    attempts.append_many(results); accepted.append_many(accepted_batch)
                    attempted_variants.update(batch); next_variant = batch[-1] + 1
                    if accepted_batch or next_variant % (workers * 10) == 0:
                        print(f"fingertip supplement: variants={len(attempted_variants)}/{cap} accepted={len(accepted.records())}/{phase2s.state.fingertip_target}", flush=True)
        summary = _summary(accepted.records(), attempts.records(), "FINGERTIP", phase2s.state.fingertip_target, len(source) + FINGERTIP_SUPPLEMENT_CAP)
        summary["replayed_large_object_proposals"] = len(source)
        summary["status"] = "TARGET_REACHED" if len(accepted.records()) >= phase2s.state.fingertip_target else "SEARCH_INCOMPLETE"
        (state_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)

    if not args.fingertip_only:
        cap = phase2s.state.maximum_palmar_attempts if args.attempt_limit is None else min(args.attempt_limit, phase2s.state.maximum_palmar_attempts)
        state_dir = root / "palmar_states" / run_key
        attempts = IncrementalJsonlStore(state_dir / "attempts.jsonl", 30.0, 0.05)
        accepted = IncrementalJsonlStore(state_dir / "accepted_states.jsonl", 30.0, 0.05)
        attempted = {int(row["attempt_index"]) for row in attempts.records()}
        next_index = next((index for index in range(cap) if index not in attempted), cap)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            while len(accepted.records()) < phase2s.state.palmar_target and next_index < cap:
                batch = [index for index in range(next_index, min(next_index + workers, cap)) if index not in attempted]
                results = list(executor.map(_palmar_task, ((phase2s, phase2, index) for index in batch)))
                accepted_batch = []
                for result in results:
                    result.update({
                        "trial_id": stable_trial_id("phase2S-palmar-attempt", {"attempt_index": result["attempt_index"], "config": cfg_hash}),
                        "experiment_id": phase2s.experiment_id, "config_hash": cfg_hash, "git_commit_sha": commit,
                    })
                    if result["accepted"] and len(accepted.records()) + len(accepted_batch) < phase2s.state.maximum_states_per_group:
                        accepted_batch.append({**result, "trial_id": stable_trial_id("phase2S-palmar-state", result["grasp_state_id"])})
                attempts.append_many(results); accepted.append_many(accepted_batch)
                attempted.update(batch); next_index = batch[-1] + 1
                if accepted_batch or next_index % (workers * 10) == 0:
                    print(f"palmar search: attempts={len(attempted)}/{cap} accepted={len(accepted.records())}/{phase2s.state.palmar_target}", flush=True)
        summary = _summary(accepted.records(), attempts.records(), "PALMAR_SECURED", phase2s.state.palmar_target, cap)
        if len(accepted.records()) >= phase2s.state.palmar_target:
            summary["status"] = "TARGET_REACHED"
        elif len(attempts.records()) >= phase2s.state.maximum_palmar_attempts and len(accepted.records()) < phase2s.state.minimum_palmar_states:
            summary["status"] = "PHASE2S_INSUFFICIENT_PALMAR_STATES"
        else:
            summary["status"] = "SEARCH_INCOMPLETE"
        (state_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        if summary["status"] == "PHASE2S_INSUFFICIENT_PALMAR_STATES":
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
