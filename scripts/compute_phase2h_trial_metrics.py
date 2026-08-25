#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

from seqgrasp.config import ROOT, load_configs
from seqgrasp.experiments.phase2h_visuals import assert_replay_matches, replay_trial, trial_metrics
from seqgrasp.experiments.resumable import IncrementalJsonlStore
from seqgrasp.phase2_5_config import load_phase2_5_config


METRIC_METHOD_ID = "phase2h_existing_strict_gate_prefix_v3"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _source_results() -> Path:
    evidence = json.loads(
        (ROOT / "outputs" / "phase2W" / "analysis" / "evidence.json").read_text(encoding="utf-8")
    )
    expected_failures = evidence["B_only_failure_mechanisms"]
    matches = []
    for summary_path in (ROOT / "outputs" / "phase2W" / "b_only_dynamic").rglob("summary.json"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("status") == "PHASE2W_NO_STATIC_WRIST_B_CONTROL"
            and summary.get("total_B_only_candidates") == 8192
        ):
            failures = Counter()
            for pose in summary["poses"]:
                failures.update(pose["failure_mechanisms"])
            candidate_path = summary_path.with_name("candidate_results.jsonl")
            if candidate_path.exists() and dict(failures) == expected_failures:
                matches.append(candidate_path)
    if len(matches) != 1:
        raise RuntimeError(f"expected one final 8192-trial Phase 2W result, found {len(matches)}")
    return matches[0]


def _task(row: dict) -> dict:
    cfg25, _ = load_phase2_5_config()
    base_cfg = load_configs(scene_filename="scene_two_object_half_scale.yaml")
    summary, arrays = replay_trial(row, cfg25, base_cfg)
    assert_replay_matches(row, summary)
    result = trial_metrics(row, summary, arrays, cfg25)
    result["metric_method_id"] = METRIC_METHOD_ID
    return result


def _seed_precondition_failures(store: IncrementalJsonlStore, source_hash: str) -> int:
    """Reuse the complete v1 replay only where the strict precondition is false.

    A trial without pre-release dual index+thumb contact (and no assist) has
    strict survival zero by the existing gate, independent of later force or
    motion. Its previously replayed contact/motion metrics remain exact.
    """

    legacy = ROOT / "outputs" / "phase2H" / "trial_metrics" / source_hash[:12] / "metrics.jsonl"
    if not legacy.exists():
        return 0
    seeded = []
    for row in _jsonl(legacy):
        if row["dual_contact_pre_release"]:
            continue
        copied = dict(row)
        copied.update({
            "strict_survival_steps": 0,
            "first_strict_failure_timestep": int(row["fixture_release_timestep"]),
            "first_strict_failure_relative_step": 0,
            "first_strict_failure_mechanism": "PRE_RELEASE_DUAL_CONTACT_NOT_ESTABLISHED",
            "metric_method_id": METRIC_METHOD_ID,
        })
        seeded.append(copied)
    count = 0
    for start in range(0, len(seeded), 64):
        count += store.append_many(seeded[start:start + 64])
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay existing Phase 2W trials for Phase 2H visual metrics")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="development-only prefix; omit for the required full replay")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be in [1, 8]")
    source = _source_results()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    output = ROOT / "outputs" / "phase2H" / "trial_metrics" / f"{source_hash[:12]}_{METRIC_METHOD_ID}"
    store = IncrementalJsonlStore(output / "metrics.jsonl", 30.0, 0.05)
    seeded = _seed_precondition_failures(store, source_hash)
    completed = store.completed_ids()
    rows = []
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["trial_id"] not in completed:
                rows.append(row)
            if args.limit is not None and len(rows) >= args.limit:
                break
    print(f"Phase 2H metrics: {len(completed)} complete ({seeded} newly seeded strict-precondition failures), {len(rows)} pending")
    batch = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, result in enumerate(executor.map(_task, rows, chunksize=4), start=1):
            batch.append(result)
            if len(batch) >= 16:
                store.append_many(batch)
                batch.clear()
            if index % 64 == 0 or index == len(rows):
                print(f"Phase 2H metrics replay: {index}/{len(rows)}")
        store.append_many(batch)
    records = store.records()
    if args.limit is None and len(records) != 8192:
        raise RuntimeError(f"expected 8192 complete Phase 2W trials, found {len(records)}")
    summary = {
        "status": "complete" if len(records) == 8192 else "development_partial",
        "metric_method_id": METRIC_METHOD_ID,
        "source_candidate_results": str(source.relative_to(ROOT)),
        "source_sha256": source_hash,
        "trial_count": len(records),
        "failure_mechanisms": dict(Counter(row["failure_mechanism"] for row in records)),
        "maximum_strict_survival_steps": max((row["strict_survival_steps"] for row in records), default=0),
        "maximum_dual_contact_survival_steps": max((row["dual_contact_survival_steps"] for row in records), default=0),
        "metric_definitions": {
            "dual_contact_survival_steps": "consecutive post-release prefix with index+thumb contact and no middle/ring assist",
            "strict_survival_steps": "consecutive post-release prefix satisfying the existing Phase 2W pre-release dual-contact, no-assist, minimum-contact, normal-force, cumulative penetration, table, translation, rotation, and numerical gates",
        },
        "resume_command": "python scripts/compute_phase2h_trial_metrics.py --workers 8",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
