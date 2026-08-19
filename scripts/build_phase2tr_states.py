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
from seqgrasp.experiments.phase2t_sampling import (
    evaluate_phase2tr_fingertip_candidate,
    evaluate_phase2tr_palmar_candidate,
)
from seqgrasp.experiments.resumable import IncrementalJsonlStore, stable_trial_id
from seqgrasp.phase2_config import load_phase2_config
from seqgrasp.phase2s_config import load_phase2s_config
from seqgrasp.phase2tr_config import assert_index_thumb_free_topology, load_phase2tr_config


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _largest(root: Path) -> tuple[Path, list[dict]]:
    candidates = [(len(_jsonl(path)), path.stat().st_mtime_ns, path) for path in root.rglob("accepted_states.jsonl")]
    if not candidates:
        raise FileNotFoundError(f"no accepted endpoint source under {root}")
    path = max(candidates)[2]
    return path, _jsonl(path)


def _task(payload):
    group, args = payload
    fn = evaluate_phase2tr_fingertip_candidate if group == "fingertip" else evaluate_phase2tr_palmar_candidate
    return fn(*args)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build exact middle+ring/index+thumb Phase 2T-R endpoint states")
    parser.add_argument("group", choices=("fingertip", "palmar"))
    parser.add_argument("--workers", type=int)
    parser.add_argument("--attempt-limit", type=int)
    args = parser.parse_args()
    phase2tr, tr_path = load_phase2tr_config()
    phase2s, s_path = load_phase2s_config()
    phase2, p_path = load_phase2_config()
    if args.group == "fingertip":
        source_path, all_sources = _largest(ROOT / "outputs" / "phase2T" / "fingertip_states")
        sources = [r for r in all_sources if r.get("occupied_finger_mask") == [False, True, True, False]]
        cap, target, minimum = (
            phase2tr.state_search.fingertip_attempt_cap,
            phase2tr.state_search.fingertip_target,
            phase2tr.state_search.fingertip_minimum,
        )
    else:
        source_path, all_sources = _largest(ROOT / "outputs" / "phase2T" / "palmar_states")
        sources = [r for r in all_sources if r.get("occupied_finger_mask") == [False, False, True, True]]
        cap, target, minimum = (
            phase2tr.state_search.palmar_attempt_cap,
            phase2tr.state_search.palmar_target,
            phase2tr.state_search.palmar_minimum,
        )
    if args.group == "fingertip" and len(sources) != 2:
        raise RuntimeError(f"Phase 2T must supply exactly two middle+ring fingertip proposal centers, found {len(sources)}")
    if not sources:
        raise RuntimeError(f"no compatible {args.group} proposal centers")
    cfg_hash = config_hash([tr_path, s_path, p_path, source_path, ROOT / "seqgrasp" / "experiments" / "phase2t_sampling.py"])
    output = ROOT / phase2tr.output_dir / f"{args.group}_states" / cfg_hash[:12]
    attempts = IncrementalJsonlStore(output / "attempts.jsonl", 30.0, 0.05)
    accepted = IncrementalJsonlStore(output / "accepted_states.jsonl", 30.0, 0.05)
    limit = min(args.attempt_limit or cap, cap)
    completed = {int(row["targeted_attempt_index"] if "targeted_attempt_index" in row else row["attempt_index"]) for row in attempts.records()}
    pending = [i for i in range(limit) if i not in completed]
    workers = min(args.workers or max(1, (os.cpu_count() or 1) // 2), phase2tr.state_search.maximum_workers)
    commit = git_commit_sha(ROOT)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for offset in range(0, len(pending), workers):
            batch = pending[offset:offset + workers]
            payloads = [(args.group, (phase2s, phase2, sources[i % len(sources)], i, phase2tr.state_search.seed)) for i in batch]
            results = list(executor.map(_task, payloads))
            valid = []
            for result in results:
                index = int(result.get("targeted_attempt_index", result["attempt_index"]))
                result.update({
                    "targeted_attempt_index": index,
                    "trial_id": stable_trial_id(f"phase2TR-{args.group}-attempt", index),
                    "experiment_id": phase2tr.experiment_id,
                    "config_hash": cfg_hash,
                    "git_commit_sha": commit,
                })
                if result["accepted"]:
                    assert_index_thumb_free_topology(result)
                    valid.append({**result, "trial_id": stable_trial_id(f"phase2TR-{args.group}-state", result["grasp_state_id"])})
            attempts.append_many(results)
            accepted.append_many(valid)
            done = len(completed) + offset + len(batch)
            if valid or done % (workers * 25) == 0 or done == limit:
                print(f"Phase 2T-R {args.group}: attempts={done}/{limit} accepted={len(accepted.records())}", flush=True)
            if len(accepted.records()) >= target:
                break
    rows, valid = attempts.records(), accepted.records()
    cap_exhausted = len(rows) >= cap
    stop = len(valid) < minimum and cap_exhausted
    stop_code = f"PHASE2TR_INSUFFICIENT_INDEX_THUMB_FREE_{args.group.upper()}"
    summary = {
        "experiment_id": phase2tr.experiment_id,
        "group": args.group.upper(),
        "status": "TARGET_REACHED" if len(valid) >= target else stop_code if stop else "SEARCH_INCOMPLETE",
        "attempts": len(rows),
        "valid_states": len(valid),
        "acceptance_rate": len(valid) / len(rows) if rows else 0.0,
        "local_seed_derived_states": sum(r.get("sampling_mode", "").startswith("local") for r in valid),
        "global_search_derived_states": sum(not r.get("sampling_mode", "").startswith("local") for r in valid),
        "failure_mechanisms": dict(Counter(str(r.get("rejection_reason")) for r in rows if not r["accepted"])),
        "config_hash": cfg_hash,
        "git_commit_sha": commit,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 3 if stop else 0


if __name__ == "__main__":
    raise SystemExit(main())
