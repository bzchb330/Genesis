#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from seqgrasp.config import ROOT, load_configs
from seqgrasp.experiments.metadata import config_hash, git_commit_sha
from seqgrasp.experiments.phase2_5_trajectory import BAcquisitionTrajectory, run_b_acquisition_trajectory
from seqgrasp.experiments.resumable import IncrementalJsonlStore, stable_trial_id
from seqgrasp.experiments.second_grasp import BPlacement
from seqgrasp.phase2_5_config import load_phase2_5_config
from seqgrasp.phase2tr_config import FREE_FINGERS, load_phase2tr_config


FINGER_ORDER = ("index", "middle", "ring", "thumb")
PAIR_INDICES = (0, 3)
OTHER_INDICES = (1, 2)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _sources() -> tuple[Path, list[dict]]:
    candidates = [(len(_jsonl(path)), path.stat().st_mtime_ns, path) for path in (ROOT / "outputs" / "phase2S" / "b_only_dynamic").rglob("candidate_results.jsonl")]
    if not candidates:
        raise FileNotFoundError("Phase 2S B-only dynamic results are unavailable")
    path = max(candidates)[2]
    rows = [row for row in _jsonl(path) if row.get("B_acquired") and row.get("geometry_topology") == "index+thumb"]
    if len(rows) < 3:
        raise RuntimeError(f"Phase 2S must supply at least three index+thumb successes, found {len(rows)}")
    return path, sorted(rows, key=lambda row: int(row["candidate_index"]))


def _trajectory(payload: dict, index: int, rng: np.random.Generator, perturb: bool) -> BAcquisitionTrajectory:
    active = np.repeat([True, False, False, True], 4)

    def stage(name: str) -> tuple[float, ...]:
        values = np.asarray(payload[name], dtype=float)
        if perturb:
            values[active] += rng.uniform(-0.004, 0.004, int(np.sum(active)))
        return tuple(float(value) for value in values)

    return BAcquisitionTrajectory(
        candidate_index=index,
        approach_joint_rad=stage("approach_joint_rad"),
        precontact_joint_rad=stage("precontact_joint_rad"),
        closing_joint_rad=stage("closing_joint_rad"),
        hold_joint_rad=stage("hold_joint_rad"),
        close_steps=int(payload["close_steps"]),
        per_finger_close_delay_steps=tuple(payload["per_finger_close_delay_steps"]),
        fixture_release_delay_steps=int(payload["fixture_release_delay_steps"]),
    )


def _evaluate(payload):
    cfg25, scene_cfg, source, index, seed, robustness = payload
    rng = np.random.default_rng(np.random.SeedSequence([seed, 31 if robustness else 30, index]))
    exact_revalidation = not robustness and index < 3
    span = 0.0 if exact_revalidation else (0.0005 if robustness else 0.0012)
    yaw_span = 0.0 if exact_revalidation else (0.012 if robustness else 0.035)
    position = np.asarray(source["placement"]["position_m"], dtype=float) + rng.uniform(-span, span, 3)
    yaw = float(source["placement"].get("yaw_rad", 0.0) + rng.uniform(-yaw_span, yaw_span))
    placement = BPlacement(
        index,
        tuple(float(v) for v in position),
        tuple(float(v) for v in Rotation.from_euler("z", yaw).as_quat(scalar_first=True)),
        yaw,
    )
    trajectory = _trajectory(source["trajectory"], index, rng, perturb=not exact_revalidation)
    summary, arrays = run_b_acquisition_trajectory(
        cfg25, trajectory, placement=placement, scene_cfg=scene_cfg, collect_timeseries=True,
    )
    release = int(summary["fixture_release_timestep"])
    flags = np.asarray(arrays["B_per_finger_contact_flag"], dtype=int)
    pre = flags[:release]
    both_before_release = bool(len(pre) and np.any(np.all(pre[:, PAIR_INDICES] > 0, axis=1)))
    no_middle_ring_assist = bool(not np.any(flags[:, OTHER_INDICES] > 0))
    strict = bool(
        summary["B_acquired"]
        and both_before_release
        and no_middle_ring_assist
        and summary["unsupported_contact_steps"] == cfg25.timing.unsupported_hold_steps
        and not summary["B_table_contact_after_release"]
        and summary["first_post_release_contact_loss_step"] is None
    )
    return {
        **summary,
        "trial_id": stable_trial_id("phase2TR-index-thumb-b-only-robust" if robustness else "phase2TR-index-thumb-b-only", index),
        "candidate_index": int(index),
        "source_phase2S_candidate_index": int(source["candidate_index"]),
        "permitted_acquisition_pair": list(FREE_FINGERS),
        "both_index_thumb_contact_before_release": both_before_release,
        "middle_ring_assist": not no_middle_ring_assist,
        "strict_index_thumb_success": strict,
        "robustness_trial": bool(robustness),
    }


def _run(rows, cfg25, scene_cfg, sources, indices, seed, workers, robustness):
    completed = rows.completed_ids()
    pending = [i for i in indices if stable_trial_id("phase2TR-index-thumb-b-only-robust" if robustness else "phase2TR-index-thumb-b-only", i) not in completed]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        buffer = []
        payloads = (
            (cfg25, scene_cfg, sources[i % len(sources)], i, seed, robustness)
            for i in pending
        )
        for count, result in enumerate(executor.map(_evaluate, payloads), start=1):
            buffer.append(result)
            if len(buffer) >= workers or count == len(pending):
                rows.append_many(buffer)
                buffer.clear()
            if count % (workers * 4) == 0 or count == len(pending):
                print(f"Phase 2T-R {'robustness' if robustness else 'B-only'}: {len(completed) + count}/{len(indices)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate native index+thumb B-only control")
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--robustness", type=int, default=0)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    phase2tr, tr_path = load_phase2tr_config()
    cfg25, cfg25_path = load_phase2_5_config()
    if args.candidates > phase2tr.second_grasp.b_only_candidate_cap:
        raise ValueError("Phase 2T-R B-only candidate cap exceeded")
    if args.robustness and args.robustness < phase2tr.second_grasp.robustness_trials:
        raise ValueError("Phase 2T-R robustness requires at least 100 trials")
    source_path, sources = _sources()
    scene_cfg = load_configs(scene_filename=phase2tr.scene_filename)
    cfg_hash = config_hash([tr_path, cfg25_path, source_path, ROOT / "scripts" / "validate_phase2tr_b_only.py"])
    output = ROOT / phase2tr.output_dir / "b_only_index_thumb" / cfg_hash[:12]
    store = IncrementalJsonlStore(output / "candidate_results.jsonl", 30.0, 0.05)
    workers = min(args.workers or max(1, (os.cpu_count() or 1) // 2), phase2tr.state_search.maximum_workers)
    _run(store, cfg25, scene_cfg, sources, range(args.candidates), phase2tr.second_grasp.b_only_seed, workers, False)
    candidates = [row for row in store.records() if not row["robustness_trial"] and int(row["candidate_index"]) < args.candidates]
    successes = [row for row in candidates if row["strict_index_thumb_success"]]
    if args.robustness and successes:
        robust_store = IncrementalJsonlStore(output / "robustness_trials.jsonl", 30.0, 0.05)
        best = sorted(successes, key=lambda row: (row["maximum_B_translation_after_release_m"], row["maximum_B_orientation_after_release_rad"]))[0]
        robust_source = next(row for row in sources if int(row["candidate_index"]) == int(best["source_phase2S_candidate_index"]))
        _run(robust_store, cfg25, scene_cfg, [robust_source], range(args.robustness), phase2tr.second_grasp.b_only_seed, workers, True)
        robust = robust_store.records()[:args.robustness]
    else:
        robust = []
    status = (
        "PASS_TARGET" if len(successes) >= phase2tr.second_grasp.b_only_success_target
        else "PASS_MINIMUM" if len(successes) >= phase2tr.second_grasp.b_only_hard_minimum
        else "PHASE2TR_INDEX_THUMB_B_CONTROL_FAILED" if len(candidates) >= phase2tr.second_grasp.b_only_candidate_cap
        else "SEARCH_INCOMPLETE"
    )
    summary = {
        "status": status,
        "candidate_count": len(candidates),
        "strict_success_count": len(successes),
        "success_candidate_indices": [int(row["candidate_index"]) for row in successes],
        "failure_mechanisms": dict(Counter(row["failure_mechanism"] for row in candidates if not row["strict_index_thumb_success"])),
        "robustness_trial_count": len(robust),
        "robustness_success_count": sum(row["strict_index_thumb_success"] for row in robust),
        "robustness_success_fraction": sum(row["strict_index_thumb_success"] for row in robust) / len(robust) if robust else None,
        "robustness_failure_mechanisms": dict(Counter(row["failure_mechanism"] for row in robust if not row["strict_index_thumb_success"])),
        "robustness_translation_m": [row["maximum_B_translation_after_release_m"] for row in robust],
        "robustness_rotation_rad": [row["maximum_B_orientation_after_release_rad"] for row in robust],
        "best_success": successes[0] if successes else None,
        "config_hash": cfg_hash,
        "git_commit_sha": git_commit_sha(ROOT),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key not in {"best_success", "robustness_translation_m", "robustness_rotation_rad"}}, indent=2))
    return 3 if status == "PHASE2TR_INDEX_THUMB_B_CONTROL_FAILED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
