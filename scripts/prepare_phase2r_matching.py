#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import math
import os
from pathlib import Path

import numpy as np

from seqgrasp.config import ROOT, load_configs
from seqgrasp.experiments.metadata import config_hash, git_commit_sha
from seqgrasp.experiments.phase2r import nearest_neighbor_match, split_calibration_states, validate_grasp_state_schema
from seqgrasp.experiments.resource_components import compute_resource_components
from seqgrasp.experiments.resumable import IncrementalJsonlStore, stable_trial_id
from seqgrasp.phase2_config import load_phase2_config
from seqgrasp.phase2r_config import load_phase2r_config


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _largest_state_file(group: str) -> Path:
    candidates = []
    for path in (ROOT / "outputs" / "phase2R" / f"{group}_states").rglob("accepted_states.jsonl"):
        candidates.append((len(_read_jsonl(path)), path.stat().st_mtime_ns, path))
    if not candidates:
        raise FileNotFoundError(f"no Phase 2R {group} state file found")
    return max(candidates)[2]


def _enrich_for_resource(row: dict) -> dict:
    cfg = load_configs()
    result = dict(row)
    result["grasp_id"] = result["grasp_state_id"]
    result.setdefault("initial_palm_position_m", list(cfg.hand.mount_pos))
    result.setdefault("initial_palm_quaternion", list(cfg.hand.mount_quat))
    result.setdefault("mean_per_finger_normal_force_N", result["per_finger_A_normal_force_N"])
    return result


def _resource_task(payload):
    row, resources, seed, commit, cfg_hash = payload
    enriched = _enrich_for_resource(row)
    components = compute_resource_components(enriched, resources, seed)
    return {
        "trial_id": stable_trial_id("phase2R-resource-state", row["grasp_state_id"]),
        "grasp_state_id": row["grasp_state_id"],
        **asdict(components),
        "free_finger_count": 4 - components.occupied_finger_count,
        "git_commit_sha": commit,
        "config_hash": cfg_hash,
    }


def _moments(rows, key):
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    return {"mean": float(np.mean(values)), "standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}


def _smd(left, right, key):
    x = np.asarray([float(row[key]) for row in left], dtype=float)
    y = np.asarray([float(row[key]) for row in right], dtype=float)
    pooled = math.sqrt((float(np.var(x, ddof=1)) + float(np.var(y, ddof=1))) / 2.0) if len(x) > 1 and len(y) > 1 else 0.0
    return 0.0 if pooled == 0.0 else float((np.mean(y) - np.mean(x)) / pooled)


def _balance(fingertip, palmar, covariates):
    return {
        key: {
            "FINGERTIP": _moments(fingertip, key),
            "PALMAR_SECURED": _moments(palmar, key),
            "standardized_mean_difference_palmar_minus_fingertip": _smd(fingertip, palmar, key),
        }
        for key in covariates
    }


def _distribution(rows, key):
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    return {
        "count": len(values), "minimum": float(np.min(values)), "maximum": float(np.max(values)),
        "mean": float(np.mean(values)), "standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute Phase 2R pre-B resources and deterministic matching")
    parser.add_argument("--config", default="configs/phase2R_palmar_vs_fingertip.yaml")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    phase2r, phase2r_path = load_phase2r_config(ROOT / args.config)
    phase2, phase2_path = load_phase2_config()
    fingertip_path, palmar_path = _largest_state_file("fingertip"), _largest_state_file("palmar")
    fingertip, palmar = _read_jsonl(fingertip_path), _read_jsonl(palmar_path)
    if len(fingertip) < phase2r.state.fingertip_target or len(palmar) < phase2r.state.palmar_target:
        raise RuntimeError(f"endpoint populations incomplete: FINGERTIP={len(fingertip)}, PALMAR_SECURED={len(palmar)}")
    for row in fingertip + palmar:
        validate_grasp_state_schema(row)
    cfg_hash = config_hash([phase2r_path, phase2_path, fingertip_path, palmar_path])
    output = ROOT / phase2r.output_dir / "matching" / cfg_hash[:12]
    store = IncrementalJsonlStore(output / "resource_states.jsonl", 30.0, 0.05)
    completed = store.completed_ids()
    states = fingertip + palmar
    pending = [row for row in states if stable_trial_id("phase2R-resource-state", row["grasp_state_id"]) not in completed]
    workers = min(args.workers or max(1, (os.cpu_count() or 1) // 2), phase2r.state.maximum_workers)
    commit = git_commit_sha(ROOT)
    payloads = ((row, phase2.resources, phase2r.state.seed, commit, cfg_hash) for row in pending)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        buffer = []
        for index, result in enumerate(executor.map(_resource_task, payloads), start=1):
            buffer.append(result)
            if len(buffer) >= workers or index == len(pending):
                store.append_many(buffer)
                buffer.clear()
            if index % workers == 0 or index == len(pending):
                print(f"pre-B resources: {len(completed) + index}/{len(states)}", flush=True)
    resources = {row["grasp_state_id"]: row for row in store.records()}
    enriched = [{**row, **resources[row["grasp_state_id"]]} for row in states]
    calibration, formal_pool = split_calibration_states(enriched, phase2r.matching.calibration_per_group)
    finger_pool = [row for row in formal_pool if row["grasp_state_type"] == "FINGERTIP"]
    palm_pool = [row for row in formal_pool if row["grasp_state_type"] == "PALMAR_SECURED"]
    pairs = nearest_neighbor_match(finger_pool, palm_pool, phase2r.matching.covariates, phase2r.matching.target_pairs)
    if len(pairs) < phase2r.matching.target_pairs:
        raise RuntimeError(f"only {len(pairs)} non-reused matched pairs available")
    matched_rows = []
    for pair in pairs:
        for row in (pair.fingertip, pair.palmar):
            matched_rows.append({
                **row, "matched_pair_id": pair.matched_pair_id,
                "matching_standardized_distance": pair.standardized_distance,
                "trial_id": stable_trial_id("phase2R-matched-state", {"pair": pair.matched_pair_id, "state": row["grasp_state_id"]}),
            })
    matched_store = IncrementalJsonlStore(output / "matched_states.jsonl", 30.0, 0.05)
    matched_store.append_many(matched_rows)
    calibration_store = IncrementalJsonlStore(output / "calibration_states.jsonl", 30.0, 0.05)
    calibration_store.append_many([
        {**row, "calibration_only": True, "trial_id": stable_trial_id("phase2R-calibration-state", row["grasp_state_id"])}
        for row in calibration
    ])
    matched_finger = [row for row in matched_rows if row["grasp_state_type"] == "FINGERTIP"]
    matched_palm = [row for row in matched_rows if row["grasp_state_type"] == "PALMAR_SECURED"]
    distances = np.asarray([pair.standardized_distance for pair in pairs])
    summary = {
        "matched_pair_count": len(pairs),
        "group_sample_sizes": {"FINGERTIP": len(fingertip), "PALMAR_SECURED": len(palmar)},
        "calibration_state_counts": dict(Counter(row["grasp_state_type"] for row in calibration)),
        "formal_pool_counts": {"FINGERTIP": len(finger_pool), "PALMAR_SECURED": len(palm_pool)},
        "discarded_formal_pool_states": {"FINGERTIP": len(finger_pool) - len(pairs), "PALMAR_SECURED": len(palm_pool) - len(pairs)},
        "matching_covariates": phase2r.matching.covariates,
        "balance_before_matching": _balance(finger_pool, palm_pool, phase2r.matching.covariates),
        "balance_after_matching": _balance(matched_finger, matched_palm, phase2r.matching.covariates),
        "matching_distance": _distribution([{"distance": value} for value in distances], "distance"),
        "resource_distributions": {
            state_type: {
                key: _distribution([row for row in enriched if row["grasp_state_type"] == state_type], key)
                for key in ("occupied_finger_count", "free_finger_count", "free_finger_workspace_vol_m3", "free_palm_volume_m3", "COM_to_palm_origin_distance_m", "palm_A_contact_fraction")
            }
            for state_type in ("FINGERTIP", "PALMAR_SECURED")
        },
        "config_hash": cfg_hash,
        "git_commit_sha": commit,
    }
    (output / "matching_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = ROOT / "docs" / "PHASE2R_MATCHING_REPORT.md"
    report.write_text(
        "# Phase 2R matching report\n\n"
        "Matching used nearest neighbours without replacement on standardized baseline stability covariates. "
        "No B outcomes or hypothesized resource variables were used. Calibration states were removed before matching.\n\n"
        f"- Endpoint populations: FINGERTIP {len(fingertip)}; PALMAR_SECURED {len(palmar)}\n"
        f"- Calibration reserve: 20 per group\n- Formal matched pairs: {len(pairs)}\n"
        f"- Discarded formal-pool states: `{json.dumps(summary['discarded_formal_pool_states'], sort_keys=True)}`\n"
        f"- Matching-distance distribution: `{json.dumps(summary['matching_distance'], sort_keys=True)}`\n\n"
        "## Balance before matching\n\n```json\n" + json.dumps(summary["balance_before_matching"], indent=2) + "\n```\n\n"
        "## Balance after matching\n\n```json\n" + json.dumps(summary["balance_after_matching"], indent=2) + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
