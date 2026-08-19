#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from seqgrasp.config import ROOT, load_configs
from seqgrasp.experiments.metadata import config_hash, git_commit_sha
from seqgrasp.experiments.phase2_5_trajectory import run_b_acquisition_trajectory
from seqgrasp.experiments.phase2_6_dynamic import trajectory_from_unit
from seqgrasp.experiments.resumable import IncrementalJsonlStore, stable_trial_id
from seqgrasp.experiments.second_grasp import BPlacement
from seqgrasp.phase2_5_config import load_phase2_5_config
from seqgrasp.phase2tr_config import load_phase2tr_config


PAIR = (0, 3)
OTHERS = (1, 2)


def _pose(index: int) -> tuple[Path, dict]:
    path = ROOT / "outputs" / "phase2S" / "b_pose_graspability" / "candidate_poses.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if int(row["candidate_index"]) == index:
            if not {"index", "thumb"}.issubset(row["accessible_fingers"]):
                raise RuntimeError("mapper pose lacks index+thumb access")
            row["accessible_fingers"] = ["index", "thumb"]
            return path, row
    raise KeyError(index)


def _evaluate(payload):
    cfg25, scene_cfg, pose, index, unit, cfg_hash = payload
    trajectory = trajectory_from_unit(cfg25, pose, index, unit, base_cfg=scene_cfg)
    placement = BPlacement(int(pose["candidate_index"]), tuple(pose["position_m"]), (1.0, 0.0, 0.0, 0.0), 0.0)
    summary, arrays = run_b_acquisition_trajectory(cfg25, trajectory, placement=placement, scene_cfg=scene_cfg, collect_timeseries=True)
    release = int(summary["fixture_release_timestep"])
    flags = np.asarray(arrays["B_per_finger_contact_flag"], dtype=int)
    pre = flags[:release]
    both = bool(len(pre) and np.any(np.all(pre[:, PAIR] > 0, axis=1)))
    no_assist = bool(not np.any(flags[:, OTHERS] > 0))
    strict = bool(summary["B_acquired"] and both and no_assist)
    return {
        **summary,
        "trial_id": stable_trial_id("phase2TR-common-index-thumb-b-only", {"pose": pose["candidate_index"], "index": index, "hash": cfg_hash}),
        "candidate_index": int(index),
        "mapper_pose_candidate_index": int(pose["candidate_index"]),
        "mapper_ferrari_canny_epsilon": float(pose["ferrari_canny_epsilon"]),
        "mapper_opposition_angle_deg": float(pose["maximum_opposition_angle_deg"]),
        "permitted_acquisition_pair": ["index", "thumb"],
        "both_index_thumb_contact_before_release": both,
        "middle_ring_assist": not no_assist,
        "strict_index_thumb_success": strict,
        "config_hash": cfg_hash,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-index", type=int, default=4168)
    parser.add_argument("--candidates", type=int, default=512)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    phase2tr, tr_path = load_phase2tr_config()
    cfg25, cfg25_path = load_phase2_5_config()
    if args.candidates > phase2tr.second_grasp.b_only_candidate_cap:
        raise ValueError("Phase 2T-R B-only candidate cap exceeded")
    pose_path, pose = _pose(args.pose_index)
    scene_cfg = load_configs(scene_filename=phase2tr.scene_filename)
    cfg_hash = config_hash([tr_path, cfg25_path, pose_path, ROOT / "scripts" / "search_phase2tr_common_b_control.py"])
    output = ROOT / phase2tr.output_dir / "b_only_common_regions" / str(args.pose_index) / cfg_hash[:12]
    store = IncrementalJsonlStore(output / "candidate_results.jsonl", 30.0, 0.05)
    completed = store.completed_ids()
    lhs_seed = int(np.random.SeedSequence([phase2tr.second_grasp.b_only_seed, args.pose_index]).generate_state(1)[0])
    unit = qmc.LatinHypercube(70, seed=lhs_seed).random(args.candidates)
    pending = [i for i in range(args.candidates) if stable_trial_id("phase2TR-common-index-thumb-b-only", {"pose": pose["candidate_index"], "index": i, "hash": cfg_hash}) not in completed]
    workers = min(args.workers or max(1, (os.cpu_count() or 1) // 2), phase2tr.state_search.maximum_workers)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        buffer = []
        payloads = ((cfg25, scene_cfg, pose, i, unit[i], cfg_hash) for i in pending)
        for count, result in enumerate(executor.map(_evaluate, payloads), start=1):
            buffer.append(result)
            if len(buffer) >= workers or count == len(pending):
                store.append_many(buffer); buffer.clear()
            if count % (workers * 8) == 0 or count == len(pending):
                print(f"Phase 2T-R common-region B-only: {len(completed) + count}/{args.candidates}", flush=True)
    rows = [row for row in store.records() if int(row["candidate_index"]) < args.candidates]
    successes = [row for row in rows if row["strict_index_thumb_success"]]
    summary = {
        "status": "PASS" if len(successes) >= 3 else "SEARCH_INCOMPLETE" if args.candidates < phase2tr.second_grasp.b_only_candidate_cap else "PHASE2TR_INDEX_THUMB_B_CONTROL_FAILED",
        "mapper_pose_candidate_index": args.pose_index,
        "position_m": pose["position_m"],
        "mapper_ferrari_canny_epsilon": pose["ferrari_canny_epsilon"],
        "mapper_opposition_angle_deg": pose["maximum_opposition_angle_deg"],
        "candidate_count": len(rows),
        "strict_success_count": len(successes),
        "failure_mechanisms": dict(Counter(row["failure_mechanism"] for row in rows if not row["strict_index_thumb_success"])),
        "best_success": successes[0] if successes else None,
        "config_hash": cfg_hash,
        "git_commit_sha": git_commit_sha(ROOT),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "best_success"}, indent=2))
    return 0 if successes else 2


if __name__ == "__main__":
    raise SystemExit(main())
