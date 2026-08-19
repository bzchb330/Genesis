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
from seqgrasp.experiments.phase2t_sampling import evaluate_two_free_palmar_candidate
from seqgrasp.experiments.resumable import IncrementalJsonlStore, stable_trial_id
from seqgrasp.phase2_config import load_phase2_config
from seqgrasp.phase2s_config import load_phase2s_config
from seqgrasp.phase2t_config import load_phase2t_config


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _sources() -> tuple[Path, list[dict]]:
    candidates = [
        (len(_jsonl(path)), path.stat().st_mtime_ns, path)
        for path in (ROOT / "outputs" / "phase2S" / "palmar_states").rglob("accepted_states.jsonl")
    ]
    if not candidates:
        raise FileNotFoundError("Phase 2S palmar endpoints are unavailable")
    path = max(candidates)[2]
    rows = [row for row in _jsonl(path) if row["occupied_finger_count"] == 2]
    if not rows:
        raise RuntimeError("Phase 2S contains no two-finger palmar proposal basins")
    return path, rows


def _task(payload):
    return evaluate_two_free_palmar_candidate(*payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int)
    parser.add_argument("--attempt-limit", type=int)
    args = parser.parse_args()
    phase2t, phase2t_path = load_phase2t_config()
    phase2s, phase2s_path = load_phase2s_config()
    phase2, phase2_path = load_phase2_config()
    source_path, sources = _sources()
    cfg_hash = config_hash([
        phase2t_path, phase2s_path, phase2_path, source_path,
        ROOT / "configs" / phase2t.scene_filename,
        ROOT / "seqgrasp" / "experiments" / "phase2t_sampling.py",
    ])
    output = ROOT / phase2t.output_dir / "palmar_states" / cfg_hash[:12]
    attempts = IncrementalJsonlStore(output / "attempts.jsonl", 30.0, 0.05)
    accepted = IncrementalJsonlStore(output / "accepted_states.jsonl", 30.0, 0.05)
    cap = min(args.attempt_limit or phase2t.state_search.palmar_attempt_cap, phase2t.state_search.palmar_attempt_cap)
    completed = {int(row["targeted_attempt_index"]) for row in attempts.records()}
    pending = [index for index in range(cap) if index not in completed]
    workers = min(args.workers or max(1, (os.cpu_count() or 1) // 2), phase2t.state_search.maximum_workers)
    commit = git_commit_sha(ROOT)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for offset in range(0, len(pending), workers):
            batch = pending[offset:offset + workers]
            results = list(executor.map(_task, (
                (phase2s, phase2, sources[index % len(sources)], index, phase2t.state_search.seed)
                for index in batch
            )))
            accepted_rows = []
            for result in results:
                result.update({
                    "trial_id": stable_trial_id("phase2T-palmar-attempt", result["targeted_attempt_index"]),
                    "experiment_id": phase2t.experiment_id,
                    "config_hash": cfg_hash,
                    "git_commit_sha": commit,
                })
                if result["accepted"]:
                    accepted_rows.append({
                        **result,
                        "trial_id": stable_trial_id("phase2T-palmar-state", result["grasp_state_id"]),
                    })
            attempts.append_many(results)
            accepted.append_many(accepted_rows)
            done = len(completed) + offset + len(batch)
            if accepted_rows or done % (workers * 25) == 0 or done == cap:
                print(f"Phase 2T palmar: attempts={done}/{cap} accepted={len(accepted.records())}", flush=True)
            if len(accepted.records()) >= phase2t.state_search.palmar_target:
                break
    rows, valid = attempts.records(), accepted.records()
    summary = {
        "experiment_id": phase2t.experiment_id,
        "status": "TARGET_REACHED" if len(valid) >= phase2t.state_search.palmar_target else "SEARCH_INCOMPLETE",
        "attempts": len(rows),
        "valid_states": len(valid),
        "valid_by_occupied_pair": dict(sorted(Counter("+".join(row["support_pair"]) for row in valid).items())),
        "valid_by_free_pair": dict(sorted(Counter("+".join(row["free_finger_set"]) for row in valid).items())),
        "rejection_reasons": dict(Counter(str(row.get("rejection_reason")) for row in rows if not row["accepted"])),
        "config_hash": cfg_hash,
        "git_commit_sha": commit,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
