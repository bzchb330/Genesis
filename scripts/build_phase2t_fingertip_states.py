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
from seqgrasp.experiments.phase2t_sampling import evaluate_two_finger_fingertip_candidate
from seqgrasp.experiments.resumable import IncrementalJsonlStore, stable_trial_id
from seqgrasp.phase2_config import load_phase2_config
from seqgrasp.phase2s_config import load_phase2s_config
from seqgrasp.phase2t_config import load_phase2t_config


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _phase2s_sources() -> tuple[Path, list[dict]]:
    candidates = [
        (len(_jsonl(path)), path.stat().st_mtime_ns, path)
        for path in (ROOT / "outputs" / "phase2S" / "fingertip_states").rglob("accepted_states.jsonl")
    ]
    if not candidates:
        raise FileNotFoundError("Phase 2S fingertip endpoint proposals are unavailable")
    path = max(candidates)[2]
    return path, _jsonl(path)


def _task(payload):
    return evaluate_two_finger_fingertip_candidate(*payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Search all six exact two-finger Phase 2T fingertip topologies")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--attempt-limit", type=int)
    args = parser.parse_args()
    phase2t, phase2t_path = load_phase2t_config()
    phase2s, phase2s_path = load_phase2s_config()
    phase2, phase2_path = load_phase2_config()
    source_path, sources = _phase2s_sources()
    cfg_hash = config_hash([
        phase2t_path, phase2s_path, phase2_path, source_path,
        ROOT / "configs" / phase2t.scene_filename,
        ROOT / "seqgrasp" / "experiments" / "phase2t_sampling.py",
    ])
    output = ROOT / phase2t.output_dir / "fingertip_states" / cfg_hash[:12]
    attempts = IncrementalJsonlStore(output / "attempts.jsonl", 30.0, 0.05)
    accepted = IncrementalJsonlStore(output / "accepted_states.jsonl", 30.0, 0.05)
    cap = min(args.attempt_limit or phase2t.state_search.fingertip_attempt_cap, phase2t.state_search.fingertip_attempt_cap)
    completed = {int(row["targeted_attempt_index"]) for row in attempts.records()}
    workers = min(args.workers or max(1, (os.cpu_count() or 1) // 2), phase2t.state_search.maximum_workers)
    support_pairs = [tuple(pair) for pair in phase2t.state_search.support_pairs]
    source_by_pair = {}
    pilot_ids = {
        "phase2S_fingertip_supplement_000011",
        "phase2S_fingertip_supplement_000031",
    }
    for pair in support_pairs:
        compatible = [
            row for row in sources
            if all(row["occupied_finger_mask"][("index", "middle", "ring", "thumb").index(finger)] for finger in pair)
        ]
        source_by_pair[pair] = compatible or sources
    commit = git_commit_sha(ROOT)
    pending = [index for index in range(cap) if index not in completed]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for offset in range(0, len(pending), workers):
            batch = pending[offset:offset + workers]
            payloads = []
            for index in batch:
                pair = support_pairs[index % len(support_pairs)]
                pool = source_by_pair[pair]
                pilots = [row for row in pool if row["grasp_state_id"] in pilot_ids]
                local_index = index // len(support_pairs)
                source = pilots[local_index % len(pilots)] if pilots and local_index % 5 != 0 else pool[local_index % len(pool)]
                payloads.append((phase2s, phase2, source, pair, index, phase2t.state_search.seed))
            results = list(executor.map(_task, payloads))
            accepted_rows = []
            for result in results:
                result.update({
                    "trial_id": stable_trial_id("phase2T-fingertip-attempt", result["targeted_attempt_index"]),
                    "experiment_id": phase2t.experiment_id,
                    "config_hash": cfg_hash,
                    "git_commit_sha": commit,
                })
                if result["accepted"]:
                    accepted_rows.append({
                        **result,
                        "trial_id": stable_trial_id("phase2T-fingertip-state", result["grasp_state_id"]),
                    })
            attempts.append_many(results)
            accepted.append_many(accepted_rows)
            done = len(completed) + offset + len(batch)
            if accepted_rows or done % (workers * 25) == 0 or done == cap:
                print(f"Phase 2T fingertip: attempts={done}/{cap} accepted={len(accepted.records())}", flush=True)
            if len(accepted.records()) >= phase2t.state_search.fingertip_target:
                break
    rows = attempts.records()
    valid = accepted.records()
    by_support = Counter("+".join(row["support_pair"]) for row in valid)
    by_free = Counter("+".join(row["free_finger_set"]) for row in valid)
    attempted_support = Counter("+".join(row["support_pair"]) for row in rows)
    summary = {
        "experiment_id": phase2t.experiment_id,
        "status": (
            "TARGET_REACHED" if len(valid) >= phase2t.state_search.fingertip_target
            else "PHASE2T_TWO_FINGER_FINGERTIP_NOT_STABLE" if len(rows) >= phase2t.state_search.fingertip_attempt_cap and len(valid) < phase2t.state_search.fingertip_minimum
            else "SEARCH_INCOMPLETE"
        ),
        "attempts": len(rows),
        "valid_states": len(valid),
        "attempted_by_support_pair": dict(sorted(attempted_support.items())),
        "valid_by_occupied_pair": dict(sorted(by_support.items())),
        "valid_by_free_pair": dict(sorted(by_free.items())),
        "rejection_reasons": dict(Counter(str(row.get("rejection_reason")) for row in rows if not row["accepted"])),
        "config_hash": cfg_hash,
        "git_commit_sha": commit,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 3 if summary["status"] == "PHASE2T_TWO_FINGER_FINGERTIP_NOT_STABLE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
