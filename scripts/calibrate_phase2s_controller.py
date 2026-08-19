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
from scipy.spatial.transform import Rotation
import yaml

from seqgrasp.config import ROOT, load_configs
from seqgrasp.experiments.metadata import config_hash, git_commit_sha
from seqgrasp.experiments.phase2_5_trajectory import BAcquisitionTrajectory
from seqgrasp.experiments.phase2r_second_grasp import (
    ACQUISITION_PAIR_PRIORITY,
    run_phase2r_second_grasp_trial,
    select_static_acquisition_pair,
)
from seqgrasp.experiments.resumable import IncrementalJsonlStore, stable_trial_id
from seqgrasp.experiments.second_grasp import BPlacement
from seqgrasp.phase2_5_config import load_phase2_5_config
from seqgrasp.phase2s_config import load_phase2s_config


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _latest_calibration_states():
    candidates = [
        (path.stat().st_mtime_ns, path)
        for path in (ROOT / "outputs" / "phase2S" / "matching").rglob("calibration_states.jsonl")
    ]
    if not candidates:
        raise FileNotFoundError("no Phase 2S calibration split")
    path = max(candidates)[1]
    return path, _read_jsonl(path)


def _trajectory(payload):
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


def _source_pair(pose, demonstrated_topologies=None):
    accessible = set(pose["accessible_fingers"])
    if demonstrated_topologies:
        ranked = sorted(demonstrated_topologies.items(), key=lambda item: (item[1], item[0]), reverse=True)
        for topology, _ in ranked:
            demonstrated = set(filter(None, topology.split("+")))
            pair = next((pair for pair in ACQUISITION_PAIR_PRIORITY if set(pair).issubset(demonstrated)), None)
            if pair is not None:
                return pair
    pair = next((pair for pair in ACQUISITION_PAIR_PRIORITY if set(pair).issubset(accessible)), None)
    if pair is None:
        raise RuntimeError(f"profile lacks a two-finger source pair: {sorted(accessible)}")
    return pair


def _placement(frozen, seed_index, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, seed_index]))
    bounds = frozen["center_bounds_m"]
    position = tuple(float(rng.uniform(*bounds[axis])) for axis in "xyz")
    yaw = float(rng.uniform(*frozen["yaw_bounds_rad"]))
    quaternion = tuple(float(value) for value in Rotation.from_euler("z", yaw).as_quat(scalar_first=True))
    return BPlacement(seed_index, position, quaternion, yaw)


def _task(payload):
    cfg25, state_cfg, scene_cfg, controller_id, trajectory, source_pair, source_fingers, state, placement, seed_index, cfg_hash, commit = payload
    result = run_phase2r_second_grasp_trial(
        cfg25,
        state_cfg,
        trajectory,
        state,
        placement,
        scene_cfg=scene_cfg,
        controller_source_pair=source_pair,
        controller_source_fingers=source_fingers,
    )
    result.update({
        "trial_id": stable_trial_id("phase2S-controller-calibration", {
            "controller_id": controller_id,
            "grasp_state_id": state["grasp_state_id"],
            "B_seed_index": seed_index,
            "config_hash": cfg_hash,
        }),
        "controller_id": controller_id,
        "controller_source_pair": list(source_pair),
        "controller_source_fingers": list(source_fingers),
        "grasp_state_id": state["grasp_state_id"],
        "grasp_state_type": state["grasp_state_type"],
        "B_seed_index": seed_index,
        "calibration_only": True,
        "pilot_only": False,
        "config_hash": cfg_hash,
        "git_commit_sha": commit,
    })
    return result


def _summary(controller_id, rows):
    dynamic = [row for row in rows if row.get("dynamic_attempt_executed")]
    successes = [row for row in rows if row["outcome"] == "BOTH_RETAINED"]
    return {
        "controller_id": controller_id,
        "planned_trials": len(rows),
        "dynamic_trials": len(dynamic),
        "precheck_skips": len(rows) - len(dynamic),
        "outcome_counts": dict(Counter(row["outcome"] for row in rows)),
        "numerically_valid_trials": sum(
            row.get("numerically_valid", True) and row["outcome"] != "INVALID" for row in rows
        ),
        "B_contact_before_release_trials": sum(
            row.get("maximum_B_hand_contacts_before_release", 0) > 0 for row in dynamic
        ),
        "A_retained_trials": sum(row.get("A_retained", False) for row in rows),
        "fixture_released_trials": sum(row.get("fixture_released", False) for row in dynamic),
        "B_retained_unsupported_trials": sum(row.get("B_acquired", False) for row in dynamic),
        "BOTH_RETAINED_trials": len(successes),
        "successful_A_states": len({row["grasp_state_id"] for row in successes}),
        "successful_B_seeds": len({row["B_seed_index"] for row in successes}),
        "maximum_penetration_m": max(
            (float(row.get("maximum_B_penetration_m", 0.0)) for row in dynamic), default=0.0
        ),
    }


def _selection_key(row):
    return (
        row["BOTH_RETAINED_trials"],
        row["successful_A_states"],
        row["successful_B_seeds"],
        row["numerically_valid_trials"],
        row["B_contact_before_release_trials"],
        row["A_retained_trials"],
        row["fixture_released_trials"],
        row["B_retained_unsupported_trials"],
        -row["maximum_penetration_m"],
        row["controller_id"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    phase2s, phase2s_path = load_phase2s_config()
    cfg25, cfg25_path = load_phase2_5_config()
    states_path, states = _latest_calibration_states()
    scene_cfg = load_configs(scene_filename=phase2s.scene_filename)
    states = [{
        **row,
        "initial_palm_position_m": row.get("initial_palm_position_m", list(scene_cfg.hand.mount_pos)),
        "initial_palm_quaternion": row.get("initial_palm_quaternion", list(scene_cfg.hand.mount_quat)),
    } for row in states]
    if Counter(row["grasp_state_type"] for row in states) != Counter({"FINGERTIP": 20, "PALMAR_SECURED": 20}):
        raise RuntimeError("controller calibration requires exactly 20+20 reserved states")
    frozen_path = ROOT / "configs" / "phase2S_frozen_B_distribution.yaml"
    frozen = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
    profile_paths = [ROOT / "configs" / "grasps" / frozen["source_B_only_profile"]]
    regions_path = ROOT / "configs" / "phase2S_b_only_graspable_regions.yaml"
    regions = yaml.safe_load(regions_path.read_text(encoding="utf-8"))["regions"]
    topology_by_profile = {
        row["source_profile"]: row.get("demonstrated_contact_topologies", {}) for row in regions
    }
    controllers = []
    for path in profile_paths:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        controllers.append((
            path.stem,
            _trajectory(payload["trajectory"]),
            _source_pair(payload["pose"], topology_by_profile.get(path.name)),
            tuple(payload["pose"]["accessible_fingers"]),
        ))
    if not controllers or len(controllers) > phase2s.second_grasp.maximum_controller_candidates:
        raise RuntimeError("invalid Phase 2S controller candidate count")
    cfg_hash = config_hash([
        phase2s_path,
        cfg25_path,
        states_path,
        frozen_path,
        regions_path,
        ROOT / "seqgrasp" / "experiments" / "phase2r_second_grasp.py",
        *profile_paths,
    ])
    output = ROOT / phase2s.output_dir / "calibration" / cfg_hash[:12]
    store = IncrementalJsonlStore(output / "trials.jsonl", 30.0, 0.05)
    completed = store.completed_ids()
    commit = git_commit_sha(ROOT)
    tasks = []
    for controller_id, trajectory, source_pair, source_fingers in controllers:
        for state in states:
            for seed_index in range(phase2s.second_grasp.calibration_B_seeds_per_state):
                trial_id = stable_trial_id("phase2S-controller-calibration", {
                    "controller_id": controller_id,
                    "grasp_state_id": state["grasp_state_id"],
                    "B_seed_index": seed_index,
                    "config_hash": cfg_hash,
                })
                if trial_id not in completed:
                    tasks.append((
                        cfg25,
                        phase2s.state,
                        scene_cfg,
                        controller_id,
                        trajectory,
                        source_pair,
                        source_fingers,
                        state,
                        _placement(frozen, seed_index, phase2s.second_grasp.calibration_seed),
                        seed_index,
                        cfg_hash,
                        commit,
                    ))
    workers = min(args.workers or max(1, (os.cpu_count() or 1) // 2), phase2s.state.maximum_workers)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        buffer = []
        for index, result in enumerate(executor.map(_task, tasks), start=1):
            buffer.append(result)
            if len(buffer) >= workers or index == len(tasks):
                store.append_many(buffer)
                buffer.clear()
            if index % (workers * 4) == 0 or index == len(tasks):
                print(
                    f"Phase 2S controller calibration: {len(completed) + index}/{len(controllers) * len(states) * phase2s.second_grasp.calibration_B_seeds_per_state}",
                    flush=True,
                )
    rows = store.records()
    candidates = [
        _summary(controller_id, [row for row in rows if row["controller_id"] == controller_id])
        for controller_id, _, _, _ in controllers
    ]
    successful = [row for row in candidates if row["BOTH_RETAINED_trials"] > 0]
    selected = max(successful, key=_selection_key) if successful else None
    summary = {
        "status": "PASS" if selected else "EXPANSION_REQUIRED",
        "selection_rule": "pooled BOTH_RETAINED count and state/seed coverage, then validity/contact/A-retention/release/penetration; no group-effect optimization",
        "group_rate_difference_used_for_selection": False,
        "controller_candidates": len(controllers),
        "candidate_summaries": candidates,
        "selected_controller_id": None if selected is None else selected["controller_id"],
        "config_hash": cfg_hash,
        "git_commit_sha": commit,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if selected is None:
        return 2
    controller_id, trajectory, source_pair, source_fingers = next(
        item for item in controllers if item[0] == selected["controller_id"]
    )
    freeze = {
        "experiment_id": phase2s.experiment_id,
        "controller_id": controller_id,
        "algorithm": "static source finger subset used when all are free; otherwise the demonstrated source pair is remapped once to a static free pair before motion",
        "source_pair": list(source_pair),
        "source_fingers": list(source_fingers),
        "acquisition_pair_priority": [list(pair) for pair in ACQUISITION_PAIR_PRIORITY],
        "trajectory": asdict(trajectory),
        "timing": asdict(cfg25.timing),
        "fixture_release": "after approach, precontact, close, and frozen release delay; final 500 steps unsupported",
        "thresholds": asdict(cfg25.criteria),
        "calibration_state_ids": [row["grasp_state_id"] for row in states],
        "calibration_B_seed_indices": list(range(phase2s.second_grasp.calibration_B_seeds_per_state)),
        "formal_seed_namespace": phase2s.second_grasp.formal_seed,
        "calibration_outcome": selected,
        "config_hash": cfg_hash,
        "git_commit_sha": commit,
    }
    (ROOT / "configs" / "phase2S_frozen_controller.yaml").write_text(
        yaml.safe_dump(freeze, sort_keys=False), encoding="utf-8"
    )
    (ROOT / "docs" / "PHASE2S_CONTROLLER_FREEZE.md").write_text(
        "# Phase 2S generic controller freeze\n\n"
        "One controller was selected from pooled 20+20 calibration data without using the formal group-rate difference. "
        "The free-finger pair is assigned once before motion and never changes during a trial.\n\n"
        f"- Selected controller: `{controller_id}`\n"
        f"- Controller logic: {freeze['algorithm']}\n"
        f"- Source pair: `{json.dumps(freeze['source_pair'])}`\n"
        f"- Source finger subset: `{json.dumps(freeze['source_fingers'])}`\n"
        f"- Acquisition-pair priority: `{json.dumps(freeze['acquisition_pair_priority'])}`\n"
        f"- Trajectory: `{json.dumps(freeze['trajectory'], sort_keys=True)}`\n"
        f"- Timing: `{json.dumps(freeze['timing'], sort_keys=True)}`\n"
        f"- Fixture release: {freeze['fixture_release']}\n"
        f"- Thresholds: `{json.dumps(freeze['thresholds'], sort_keys=True)}`\n"
        f"- Calibration IDs: `{json.dumps(freeze['calibration_state_ids'])}`\n"
        f"- Calibration B seeds: `{json.dumps(freeze['calibration_B_seed_indices'])}`\n"
        f"- Formal seed namespace: {freeze['formal_seed_namespace']}\n"
        f"- Calibration outcome: `{json.dumps(selected, sort_keys=True)}`\n"
        f"- Config hash: `{cfg_hash}`\n- Git SHA at freeze: `{commit}`\n\n"
        "The controller and B distribution may not be changed based on formal outcomes.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
