#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from seqgrasp.config import ROOT, load_configs
from seqgrasp.experiments.b_workspace import analyze_B_geometry_state, free_fingertip_workspace_clouds
from seqgrasp.experiments.metadata import config_hash, git_commit_sha
from seqgrasp.experiments.phase2_5_trajectory import BAcquisitionTrajectory
from seqgrasp.experiments.phase2r import PHASE2R_EXPERIMENT_ID, assert_formal_pairing, paired_formal_trial_id
from seqgrasp.experiments.phase2r_second_grasp import run_phase2r_second_grasp_trial
from seqgrasp.experiments.resumable import IncrementalJsonlStore, stable_trial_id
from seqgrasp.experiments.second_grasp import BPlacement
from seqgrasp.phase2_5_config import load_phase2_5_config
from seqgrasp.phase2_config import load_phase2_config
from seqgrasp.phase2r_config import load_phase2r_config


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _latest_matched() -> tuple[Path, list[dict]]:
    candidates = [(path.stat().st_mtime_ns, path) for path in (ROOT / "outputs" / "phase2R" / "matching").rglob("matched_states.jsonl")]
    if not candidates:
        raise FileNotFoundError("no Phase 2R matched states found")
    path = max(candidates)[1]
    return path, _read_jsonl(path)


def _placement(frozen: dict, seed_index: int, seed: int) -> BPlacement:
    rng = np.random.default_rng(np.random.SeedSequence([seed, seed_index]))
    bounds = frozen["center_bounds_m"]
    position = tuple(float(rng.uniform(*bounds[axis])) for axis in "xyz")
    yaw = float(rng.uniform(*frozen["yaw_bounds_rad"]))
    quaternion = tuple(float(value) for value in Rotation.from_euler("z", yaw).as_quat(scalar_first=True))
    return BPlacement(seed_index, position, quaternion, yaw)


def _trajectory(payload: dict) -> BAcquisitionTrajectory:
    return BAcquisitionTrajectory(
        candidate_index=int(payload["candidate_index"]),
        approach_joint_rad=tuple(payload["approach_joint_rad"]),
        precontact_joint_rad=tuple(payload["precontact_joint_rad"]),
        closing_joint_rad=tuple(payload["closing_joint_rad"]),
        hold_joint_rad=tuple(payload["hold_joint_rad"]),
        close_steps=int(payload["close_steps"]),
        per_finger_close_delay_steps=tuple(payload["per_finger_close_delay_steps"]),
        fixture_release_delay_steps=int(payload["fixture_release_delay_steps"]),
    )


def _with_reconstruction_metadata(row: dict) -> dict:
    cfg = load_configs()
    return {
        **row,
        "initial_palm_position_m": row.get("initial_palm_position_m", list(cfg.hand.mount_pos)),
        "initial_palm_quaternion": row.get("initial_palm_quaternion", list(cfg.hand.mount_quat)),
    }


def _geometry_task(payload):
    state, phase2, phase2r, frozen = payload
    state = _with_reconstruction_metadata(state)
    state["grasp_id"] = state["grasp_state_id"]
    workspace = free_fingertip_workspace_clouds(
        state, phase2.resources, phase2.second_grasp.geometry_workspace_samples,
        phase2r.second_grasp.formal_seed,
    )
    rows = []
    for seed_index in range(phase2r.second_grasp.formal_B_seeds_per_state):
        placement = _placement(frozen, seed_index, phase2r.second_grasp.formal_seed)
        analysis = analyze_B_geometry_state(*workspace, phase2.resources, placement)
        if not np.isfinite(analysis["minimum_free_fingertip_to_B_m"]):
            analysis["minimum_free_fingertip_to_B_m"] = None
        rows.append({
            "trial_id": stable_trial_id("phase2R-formal-geometry", {"state": state["grasp_state_id"], "seed": seed_index}),
            "grasp_state_id": state["grasp_state_id"],
            "B_seed_index": seed_index,
            **analysis,
        })
    return rows


def _formal_task(payload):
    cfg25, state_cfg, trajectory, state, placement, seed_index, geometry, cfg_hash, commit = payload
    started = time.perf_counter()
    state = _with_reconstruction_metadata(state)
    result = run_phase2r_second_grasp_trial(cfg25, state_cfg, trajectory, state, placement)
    result.update({
        "trial_id": paired_formal_trial_id(
            state["matched_pair_id"], state["grasp_state_type"], seed_index,
            int(state["formal_seed_namespace"]),
        ),
        "experiment_id": PHASE2R_EXPERIMENT_ID,
        "matched_pair_id": state["matched_pair_id"],
        "grasp_state_id": state["grasp_state_id"],
        "grasp_state_type": state["grasp_state_type"],
        "B_seed_index": seed_index,
        "B_seed_namespace": int(state["formal_seed_namespace"]),
        "B_geometrically_reachable": bool(geometry["reachable"]),
        "B_initial_collision_A": bool(geometry["initial_collision_A"]),
        "B_initial_collision_hand": bool(geometry["initial_collision_hand"]),
        "second_grasp_digit_eligible": int(state["free_finger_count"]) >= 2,
        "resource_components": {
            key: state[key] for key in (
                "occupied_finger_count", "free_finger_count", "free_finger_workspace_vol_m3", "free_palm_volume_m3",
                "palm_A_contact_fraction", "COM_to_palm_origin_distance_m",
            )
        },
        "ferrari_canny_epsilon": state["ferrari_canny_epsilon"],
        "A_translation_drift_m": state["A_translation_drift_m"],
        "A_rotation_drift_rad": state["A_rotation_drift_rad"],
        "minimum_joint_margin_rad": state["minimum_joint_margin_rad"],
        "pilot_only": False,
        "calibration_only": False,
        "runtime_seconds": time.perf_counter() - started,
        "config_hash": cfg_hash,
        "git_commit_sha": commit,
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen Phase 2R formal matched experiment")
    parser.add_argument("--config", default="configs/phase2R_palmar_vs_fingertip.yaml")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-pairs", type=int, help="engineering dry-run only; formal execution omits this")
    args = parser.parse_args()
    phase2r, phase2r_path = load_phase2r_config(ROOT / args.config)
    phase2, phase2_path = load_phase2_config()
    cfg25, cfg25_path = load_phase2_5_config()
    matched_path, states = _latest_matched()
    frozen_B_path = ROOT / "configs" / "phase2R_frozen_B_distribution.yaml"
    frozen_controller_path = ROOT / "configs" / "phase2R_frozen_controller.yaml"
    frozen_B = yaml.safe_load(frozen_B_path.read_text(encoding="utf-8"))
    frozen_controller = yaml.safe_load(frozen_controller_path.read_text(encoding="utf-8"))
    trajectory = _trajectory(frozen_controller["trajectory"])
    pair_ids = sorted({row["matched_pair_id"] for row in states})
    if len(pair_ids) != phase2r.matching.target_pairs:
        raise RuntimeError(f"expected 100 matched pairs, found {len(pair_ids)}")
    if args.limit_pairs is not None:
        pair_ids = pair_ids[:args.limit_pairs]
        states = [row for row in states if row["matched_pair_id"] in pair_ids]
    planned = len(pair_ids) * 2 * phase2r.second_grasp.formal_B_seeds_per_state
    eligible_dynamic = sum(int(row["free_finger_count"] >= 2) for row in states) * phase2r.second_grasp.formal_B_seeds_per_state
    estimate = {
        "experiment_id": PHASE2R_EXPERIMENT_ID,
        "matched_pairs": len(pair_ids),
        "planned_trials": planned,
        "precheck_dynamic_attempts": eligible_dynamic,
        "precheck_skips": planned - eligible_dynamic,
        "workers": min(args.workers or max(1, (os.cpu_count() or 1) // 2), phase2r.state.maximum_workers),
        "simulation_steps_per_dynamic_trial": cfg25.timing.approach_steps + cfg25.timing.precontact_steps + max(trajectory.close_steps, 1) + trajectory.fixture_release_delay_steps + cfg25.timing.unsupported_hold_steps,
    }
    if args.dry_run:
        print(json.dumps(estimate, indent=2))
        return 0
    cfg_hash = config_hash([phase2r_path, phase2_path, cfg25_path, matched_path, frozen_B_path, frozen_controller_path])
    output = ROOT / phase2r.output_dir / "formal" / cfg_hash[:12]
    workers = estimate["workers"]
    geometry_store = IncrementalJsonlStore(output / "formal_geometry.jsonl", 30.0, 0.05)
    geometry_completed = {(row["grasp_state_id"], int(row["B_seed_index"])) for row in geometry_store.records()}
    geometry_pending = [row for row in states if any((row["grasp_state_id"], seed) not in geometry_completed for seed in range(phase2r.second_grasp.formal_B_seeds_per_state))]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, rows in enumerate(executor.map(_geometry_task, ((row, phase2, phase2r, frozen_B) for row in geometry_pending)), start=1):
            geometry_store.append_many([row for row in rows if (row["grasp_state_id"], int(row["B_seed_index"])) not in geometry_completed])
            if index % workers == 0 or index == len(geometry_pending):
                print(f"formal geometry states: {len(states) - len(geometry_pending) + index}/{len(states)}", flush=True)
    geometry = {(row["grasp_state_id"], int(row["B_seed_index"])): row for row in geometry_store.records()}
    store = IncrementalJsonlStore(output / "trials.jsonl", 30.0, 0.05)
    completed = store.completed_ids()
    commit = git_commit_sha(ROOT)
    tasks = []
    for state in states:
        state = {**state, "formal_seed_namespace": phase2r.second_grasp.formal_seed}
        for seed_index in range(phase2r.second_grasp.formal_B_seeds_per_state):
            trial_id = paired_formal_trial_id(state["matched_pair_id"], state["grasp_state_type"], seed_index, phase2r.second_grasp.formal_seed)
            if trial_id not in completed:
                tasks.append((
                    cfg25, phase2r.state, trajectory, state,
                    _placement(frozen_B, seed_index, phase2r.second_grasp.formal_seed), seed_index,
                    geometry[(state["grasp_state_id"], seed_index)], cfg_hash, commit,
                ))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        buffer = []
        for index, result in enumerate(executor.map(_formal_task, tasks), start=1):
            buffer.append(result)
            if len(buffer) >= workers or index == len(tasks):
                store.append_many(buffer)
                buffer.clear()
            if index % (workers * 10) == 0 or index == len(tasks):
                print(f"formal trials: {len(completed) + index}/{planned}", flush=True)
    records = store.records()
    if args.limit_pairs is None:
        if len(records) != planned:
            raise RuntimeError(f"formal record count {len(records)} != planned {planned}")
        assert_formal_pairing(records, phase2r.second_grasp.formal_B_seeds_per_state)
    summary = {
        **estimate,
        "completed_trials": len(records),
        "outcomes_by_group": {
            group: dict(Counter(row["outcome"] for row in records if row["grasp_state_type"] == group))
            for group in ("FINGERTIP", "PALMAR_SECURED")
        },
        "subreasons_by_group": {
            group: dict(Counter(str(row.get("outcome_subreason")) for row in records if row["grasp_state_type"] == group))
            for group in ("FINGERTIP", "PALMAR_SECURED")
        },
        "BOTH_RETAINED_total": sum(row["outcome"] == "BOTH_RETAINED" for row in records),
        "status": "PHASE2R_FORMAL_ZERO_POSITIVE_CLASS" if records and not any(row["outcome"] == "BOTH_RETAINED" for row in records) else "COMPLETE",
        "config_hash": cfg_hash,
        "git_commit_sha": commit,
    }
    (output / "formal_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
