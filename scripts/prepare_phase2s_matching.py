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
from seqgrasp.phase2s_config import load_phase2s_config, validate_phase2s_state_record


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _largest(group: str) -> Path:
    candidates = [(len(_jsonl(path)), path.stat().st_mtime_ns, path) for path in (ROOT / "outputs" / "phase2S" / f"{group}_states").rglob("accepted_states.jsonl")]
    if not candidates:
        raise FileNotFoundError(group)
    return max(candidates)[2]


def _enrich(row: dict, base_cfg) -> dict:
    result = dict(row)
    result["grasp_id"] = result["grasp_state_id"]
    result.setdefault("initial_palm_position_m", list(base_cfg.hand.mount_pos))
    result.setdefault("initial_palm_quaternion", list(base_cfg.hand.mount_quat))
    result.setdefault("mean_per_finger_normal_force_N", result["per_finger_A_normal_force_N"])
    return result


def _resource_task(payload):
    row, resources, seed, commit, cfg_hash, base_cfg = payload
    components = compute_resource_components(_enrich(row, base_cfg), resources, seed, base_cfg)
    return {
        "trial_id": stable_trial_id("phase2S-resource-state", row["grasp_state_id"]),
        "grasp_state_id": row["grasp_state_id"],
        **asdict(components),
        "free_finger_count": 4 - components.occupied_finger_count,
        "git_commit_sha": commit, "config_hash": cfg_hash,
    }


def _moments(rows, key):
    values = np.asarray([float(row[key]) for row in rows])
    return {"mean": float(np.mean(values)), "standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}


def _smd(left, right, key):
    x, y = np.asarray([float(row[key]) for row in left]), np.asarray([float(row[key]) for row in right])
    pooled = math.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2.0)
    return None if pooled == 0 else float((np.mean(y) - np.mean(x)) / pooled)


def _balance(left, right, covariates):
    return {key: {"FINGERTIP": _moments(left, key), "PALMAR_SECURED": _moments(right, key), "standardized_mean_difference_palmar_minus_fingertip": _smd(left, right, key)} for key in covariates}


def _distribution(rows, key):
    values = np.asarray([float(row[key]) for row in rows])
    return {"count": len(values), "minimum": float(np.min(values)), "maximum": float(np.max(values)), "mean": float(np.mean(values)), "standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, "median": float(np.median(values))}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute Phase 2S resources and fresh matching")
    parser.add_argument("--config", default="configs/phase2S_half_scale_objects.yaml")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    phase2s, phase2s_path = load_phase2s_config(ROOT / args.config)
    phase2, phase2_path = load_phase2_config()
    base_cfg = load_configs(scene_filename=phase2s.scene_filename)
    fingertip_path, palmar_path = _largest("fingertip"), _largest("palmar")
    fingertip, palmar = _jsonl(fingertip_path), _jsonl(palmar_path)
    if len(fingertip) < phase2s.state.fingertip_target or len(palmar) < phase2s.state.palmar_target:
        raise RuntimeError(f"Phase 2S populations incomplete: {len(fingertip)}, {len(palmar)}")
    for row in fingertip + palmar:
        validate_grasp_state_schema(row)
        validate_phase2s_state_record(row)
    cfg_hash = config_hash([phase2s_path, phase2_path, fingertip_path, palmar_path, ROOT / "configs" / phase2s.scene_filename])
    output = ROOT / phase2s.output_dir / "matching" / cfg_hash[:12]
    store = IncrementalJsonlStore(output / "resource_states.jsonl", 30.0, 0.05)
    completed = store.completed_ids(); states = fingertip + palmar
    pending = [row for row in states if stable_trial_id("phase2S-resource-state", row["grasp_state_id"]) not in completed]
    workers = min(args.workers or max(1, (os.cpu_count() or 1) // 2), phase2s.state.maximum_workers)
    commit = git_commit_sha(ROOT)
    payloads = ((row, phase2.resources, phase2s.state.seed, commit, cfg_hash, base_cfg) for row in pending)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        buffer = []
        for index, result in enumerate(executor.map(_resource_task, payloads), start=1):
            buffer.append(result)
            if len(buffer) >= workers or index == len(pending):
                store.append_many(buffer); buffer.clear()
            if index % workers == 0 or index == len(pending):
                print(f"Phase 2S resources: {len(completed) + index}/{len(states)}", flush=True)
    resources = {row["grasp_state_id"]: row for row in store.records()}
    enriched = [{**row, **resources[row["grasp_state_id"]]} for row in states]
    low, high = np.asarray(phase2.resources.free_palm_box_lower_m), np.asarray(phase2.resources.free_palm_box_upper_m)
    for row in enriched:
        com = np.asarray(row["object_A_COM_palm_reference_m"])
        row["nearest_palm_storage_boundary_m"] = float(np.min(np.r_[com - low, high - com]))
    calibration, formal_pool = split_calibration_states(enriched, phase2s.matching.calibration_per_group)
    finger_pool = [row for row in formal_pool if row["grasp_state_type"] == "FINGERTIP"]
    palm_pool = [row for row in formal_pool if row["grasp_state_type"] == "PALMAR_SECURED"]
    pairs = nearest_neighbor_match(finger_pool, palm_pool, phase2s.matching.covariates, phase2s.matching.target_pairs)
    if len(pairs) != phase2s.matching.target_pairs:
        raise RuntimeError(f"only {len(pairs)} Phase 2S matched pairs")
    matched = []
    for pair in pairs:
        for row in (pair.fingertip, pair.palmar):
            matched.append({**row, "matched_pair_id": pair.matched_pair_id, "matching_standardized_distance": pair.standardized_distance, "trial_id": stable_trial_id("phase2S-matched-state", {"pair": pair.matched_pair_id, "state": row["grasp_state_id"]})})
    IncrementalJsonlStore(output / "matched_states.jsonl", 30.0, 0.05).append_many(matched)
    IncrementalJsonlStore(output / "calibration_states.jsonl", 30.0, 0.05).append_many([
        {**row, "calibration_only": True, "trial_id": stable_trial_id("phase2S-calibration-state", row["grasp_state_id"])} for row in calibration
    ])
    matched_f = [row for row in matched if row["grasp_state_type"] == "FINGERTIP"]
    matched_p = [row for row in matched if row["grasp_state_type"] == "PALMAR_SECURED"]
    object_a = next(row for row in base_cfg.scene.objects if row.name == "object_a")
    palm_dims = high - low; object_dims = 2.0 * np.asarray(object_a.size)
    packing = {
        "method": "axis-aligned bounding-box grid count inside configured palm measurement volume",
        "palm_box_dimensions_m": palm_dims.tolist(), "object_A_bounding_dimensions_m": object_dims.tolist(),
        "copies_per_axis": np.floor(palm_dims / object_dims).astype(int).tolist(),
        "maximum_non_overlapping_axis_aligned_copies": int(np.prod(np.floor(palm_dims / object_dims))),
        "interpretation": "geometry-only descriptive upper-bound construction; not a manipulation success metric",
    }
    distances = np.asarray([pair.standardized_distance for pair in pairs])
    summary = {
        "matched_pair_count": len(pairs), "group_sample_sizes": {"FINGERTIP": len(fingertip), "PALMAR_SECURED": len(palmar)},
        "calibration_state_counts": dict(Counter(row["grasp_state_type"] for row in calibration)),
        "balance_before_matching": _balance(finger_pool, palm_pool, phase2s.matching.covariates),
        "balance_after_matching": _balance(matched_f, matched_p, phase2s.matching.covariates),
        "matching_distance": _distribution([{"value": value} for value in distances], "value"),
        "resource_distributions": {group: {key: _distribution([row for row in enriched if row["grasp_state_type"] == group], key) for key in ("occupied_finger_count", "free_finger_count", "free_finger_workspace_vol_m3", "free_palm_volume_m3", "COM_to_palm_origin_distance_m", "palm_A_contact_fraction", "nearest_palm_storage_boundary_m")} for group in ("FINGERTIP", "PALMAR_SECURED")},
        "palm_packing_estimate": packing, "config_hash": cfg_hash, "git_commit_sha": commit,
    }
    (output / "matching_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (ROOT / "docs" / "PHASE2S_MATCHING_REPORT.md").write_text(
        "# Phase 2S matching and resource report\n\nNo Phase 2R labels or resource values enter this dataset. All states were revalidated and all resource components recomputed with half-scale geometry.\n\n"
        f"- Populations: FINGERTIP {len(fingertip)}, PALMAR_SECURED {len(palmar)}\n- Calibration: 20+20\n- Matched pairs: {len(pairs)}\n\n"
        "## Balance before\n\n```json\n" + json.dumps(summary["balance_before_matching"], indent=2) + "\n```\n\n## Balance after\n\n```json\n" + json.dumps(summary["balance_after_matching"], indent=2) + "\n```\n\n## Resource distributions\n\n```json\n" + json.dumps(summary["resource_distributions"], indent=2) + "\n```\n\n## Packing description\n\n```json\n" + json.dumps(packing, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
