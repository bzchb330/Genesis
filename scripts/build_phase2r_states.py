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
from seqgrasp.experiments.palmar_grasp_sampling import (
    evaluate_existing_fingertip_state,
    evaluate_palmar_candidate,
)
from seqgrasp.experiments.resumable import IncrementalJsonlStore, stable_trial_id
from seqgrasp.phase2_config import load_phase2_config
from seqgrasp.phase2r_config import load_phase2r_config


FINGERTIP_METHOD_ID = "impedance_hold_setpoint_v1"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _source_dataset() -> Path:
    candidates = []
    for path in (ROOT / "outputs" / "phase2" / "grasp_dataset").glob("*/accepted_grasps.jsonl"):
        candidates.append((len(path.read_text(encoding="utf-8").splitlines()), path))
    if not candidates:
        raise FileNotFoundError("no validated Phase 2 accepted-grasp dataset found")
    return max(candidates, key=lambda item: item[0])[1]


def _fingertip_task(payload):
    phase2r, phase2, row = payload
    return evaluate_existing_fingertip_state(phase2r, phase2, row)


def _palmar_task(payload):
    phase2r, phase2, index = payload
    return evaluate_palmar_candidate(phase2r, phase2, index)


def _summary(rows: list[dict], attempts: list[dict], state_type: str, target: int, attempt_cap: int | None) -> dict:
    accepted = [row for row in rows if row["accepted"]]
    return {
        "grasp_state_type": state_type,
        "accepted_states": len(accepted),
        "target": target,
        "attempts": len(attempts),
        "attempt_cap": attempt_cap,
        "rejection_reasons": dict(Counter(str(row.get("rejection_reason")) for row in attempts if not row["accepted"])),
        "occupied_finger_count_distribution": dict(Counter(str(row["occupied_finger_count"]) for row in accepted)),
        "contact_topology_distribution": dict(Counter(
            "+".join(finger for finger, flag in zip(("index", "middle", "ring", "thumb"), row["occupied_finger_mask"]) if flag)
            for row in accepted
        )),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build resumable Phase 2R endpoint-state populations")
    parser.add_argument("--config", default="configs/phase2R_palmar_vs_fingertip.yaml")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--fingertip-only", action="store_true")
    parser.add_argument("--palmar-only", action="store_true")
    parser.add_argument("--attempt-limit", type=int, help="engineering smoke limit; omit for the authorized full search")
    args = parser.parse_args()
    if args.fingertip_only and args.palmar_only:
        parser.error("choose at most one state-only mode")
    phase2r, phase2r_path = load_phase2r_config(ROOT / args.config)
    phase2, phase2_path = load_phase2_config()
    workers = min(args.workers or max(1, (os.cpu_count() or 1) // 2), phase2r.state.maximum_workers)
    cfg_hash = config_hash([
        phase2r_path, phase2_path, ROOT / "configs" / "hand_allegro.yaml",
        ROOT / "configs" / "scene_two_object.yaml", *[ROOT / path for path in phase2r.state.proposal_profile_paths],
    ])
    root = ROOT / phase2r.output_dir
    run_key = cfg_hash[:12]
    commit = git_commit_sha(ROOT)

    if not args.palmar_only:
        source = _read_jsonl(_source_dataset())[:phase2r.state.maximum_states_per_group]
        state_dir = root / "fingertip_states" / FINGERTIP_METHOD_ID / run_key
        attempts = IncrementalJsonlStore(state_dir / "attempts.jsonl", 30.0, 0.05)
        accepted = IncrementalJsonlStore(state_dir / "accepted_states.jsonl", 30.0, 0.05)
        completed = attempts.completed_ids()
        pending = [row for row in source if stable_trial_id("phase2R-fingertip-filter", row["grasp_id"]) not in completed]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for index, result in enumerate(executor.map(_fingertip_task, ((phase2r, phase2, row) for row in pending)), start=1):
                result.update({
                    "trial_id": stable_trial_id("phase2R-fingertip-filter", result["source_grasp_id"]),
                    "config_hash": cfg_hash, "git_commit_sha": commit,
                })
                attempts.append(result)
                if result["accepted"]:
                    accepted.append({**result, "trial_id": stable_trial_id("phase2R-fingertip-state", result["grasp_state_id"])})
                if index % workers == 0 or index == len(pending):
                    print(f"fingertip filtering: {len(completed) + index}/{len(source)}", flush=True)
        finger_summary = _summary(accepted.records(), attempts.records(), "FINGERTIP", phase2r.state.fingertip_target, len(source))
        (state_dir / "summary.json").write_text(json.dumps(finger_summary, indent=2), encoding="utf-8")
        print(json.dumps(finger_summary, indent=2), flush=True)

    if not args.fingertip_only:
        cap = phase2r.state.maximum_palmar_attempts if args.attempt_limit is None else min(args.attempt_limit, phase2r.state.maximum_palmar_attempts)
        state_dir = root / "palmar_states" / run_key
        attempts = IncrementalJsonlStore(state_dir / "attempts.jsonl", 30.0, 0.05)
        accepted = IncrementalJsonlStore(state_dir / "accepted_states.jsonl", 30.0, 0.05)
        attempted = {int(row["attempt_index"]) for row in attempts.records()}
        next_index = next((index for index in range(cap) if index not in attempted), cap)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            while len(accepted.records()) < phase2r.state.palmar_target and next_index < cap:
                batch = [index for index in range(next_index, min(next_index + workers, cap)) if index not in attempted]
                if not batch:
                    next_index += workers
                    continue
                results = list(executor.map(_palmar_task, ((phase2r, phase2, index) for index in batch)))
                accepted_batch = []
                for result in results:
                    result.update({
                        "trial_id": stable_trial_id("phase2R-palmar-attempt", {"attempt_index": result["attempt_index"], "config_hash": cfg_hash}),
                        "config_hash": cfg_hash, "git_commit_sha": commit,
                    })
                    if result["accepted"] and len(accepted.records()) + len(accepted_batch) < phase2r.state.maximum_states_per_group:
                        accepted_batch.append({**result, "trial_id": stable_trial_id("phase2R-palmar-state", result["grasp_state_id"])})
                attempts.append_many(results)
                accepted.append_many(accepted_batch)
                attempted.update(batch)
                next_index = batch[-1] + 1
                if next_index % (workers * 10) == 0 or accepted_batch:
                    print(f"palmar search: attempts={len(attempted)}/{cap} accepted={len(accepted.records())}/{phase2r.state.palmar_target}", flush=True)
        palmar_summary = _summary(accepted.records(), attempts.records(), "PALMAR_SECURED", phase2r.state.palmar_target, cap)
        full_search_complete = len(attempts.records()) >= phase2r.state.maximum_palmar_attempts
        if len(accepted.records()) >= phase2r.state.palmar_target:
            palmar_summary["status"] = "TARGET_REACHED"
        elif full_search_complete and len(accepted.records()) < phase2r.state.minimum_palmar_states:
            palmar_summary["status"] = "PHASE2R_INSUFFICIENT_PALMAR_STATES"
        else:
            palmar_summary["status"] = "SEARCH_INCOMPLETE"
        (state_dir / "summary.json").write_text(json.dumps(palmar_summary, indent=2), encoding="utf-8")
        print(json.dumps(palmar_summary, indent=2), flush=True)
        if palmar_summary["status"] == "PHASE2R_INSUFFICIENT_PALMAR_STATES":
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
