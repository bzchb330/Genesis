#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import os

import numpy as np
from scipy.stats import qmc
import yaml

from seqgrasp.config import ROOT, load_configs
from seqgrasp.experiments.metadata import config_hash, git_commit_sha
from seqgrasp.experiments.phase2_5_trajectory import (
    BAcquisitionTrajectory,
    b_only_lexicographic_key,
    run_b_acquisition_trajectory,
)
from seqgrasp.experiments.phase2_6_dynamic import trajectory_from_unit
from seqgrasp.experiments.resumable import IncrementalJsonlStore, stable_trial_id
from seqgrasp.experiments.second_grasp import BPlacement
from seqgrasp.phase2_5_config import load_phase2_5_config
from seqgrasp.phase2_6_config import load_phase2_6_config
from seqgrasp.phase2s_config import load_phase2s_config


def _from_payload(payload: dict, candidate_index: int) -> BAcquisitionTrajectory:
    return BAcquisitionTrajectory(
        candidate_index=candidate_index,
        approach_joint_rad=tuple(payload["approach_joint_rad"]),
        precontact_joint_rad=tuple(payload["precontact_joint_rad"]),
        closing_joint_rad=tuple(payload["closing_joint_rad"]),
        hold_joint_rad=tuple(payload["hold_joint_rad"]),
        close_steps=int(payload["close_steps"]),
        per_finger_close_delay_steps=tuple(payload["per_finger_close_delay_steps"]),
        fixture_release_delay_steps=int(payload["fixture_release_delay_steps"]),
    )


def _refined_trajectory(cfg25, pose, candidate_index, unit, source):
    """Deterministically refine near-positive broad-search trajectories."""

    base = _from_payload(source["trajectory"], candidate_index)
    active = np.repeat(
        np.asarray([name in pose["accessible_fingers"] for name in ("index", "middle", "ring", "thumb")]),
        4,
    )
    anchor = np.asarray(base.precontact_joint_rad, dtype=float)
    for finger, values in pose["representative_joint_rad"].items():
        finger_index = ("index", "middle", "ring", "thumb").index(finger)
        anchor[4 * finger_index:4 * finger_index + 4] = values
    cursor = 0
    stages = []
    scale_bounds = ((0.75, 1.00), (0.75, 1.00), (0.45, 1.00), (0.45, 1.00))
    for values, bounds in zip(
        (base.approach_joint_rad, base.precontact_joint_rad, base.closing_joint_rad, base.hold_joint_rad),
        scale_bounds,
    ):
        scale = float(np.interp(unit[cursor], [0.0, 1.0], bounds))
        cursor += 1
        perturbation = np.interp(unit[cursor:cursor + 16], [0.0, 1.0], [-0.015, 0.015])
        cursor += 16
        result = anchor + scale * (np.asarray(values) - anchor)
        result[active] += perturbation[active]
        result[~active] = np.asarray(values)[~active]
        stages.append(tuple(float(value) for value in result))
    close_steps = int(np.clip(
        round(base.close_steps + np.interp(unit[cursor], [0.0, 1.0], [-40, 40])),
        *cfg25.timing.close_steps_bounds,
    ))
    cursor += 1
    release_delay = int(np.clip(
        round(base.fixture_release_delay_steps + np.interp(unit[cursor], [0.0, 1.0], [-40, 40])),
        *cfg25.timing.fixture_release_delay_steps_bounds,
    ))
    return BAcquisitionTrajectory(
        candidate_index,
        *stages,
        close_steps,
        base.per_finger_close_delay_steps,
        release_delay,
    )


def _refinement_key(row, cfg25):
    criteria = cfg25.criteria
    violation = (
        max(0.0, row["maximum_B_penetration_m"] / criteria.maximum_penetration_m - 1.0)
        + max(0.0, row["maximum_B_translation_after_release_m"] / criteria.maximum_B_translation_m - 1.0)
        + max(0.0, row["maximum_B_orientation_after_release_rad"] / criteria.maximum_B_orientation_rad - 1.0)
    )
    return (
        int(row["B_acquired"]),
        int(not row["B_table_contact_after_release"]),
        int(row["first_post_release_contact_loss_step"] is None),
        row["unsupported_contact_steps"],
        -violation,
        -row["maximum_B_penetration_m"],
        -row["candidate_index"],
    )


def evaluate(payload):
    cfg25, base_cfg, pose, index, unit, cfg_hash, commit, refinement_source = payload
    trajectory = (
        trajectory_from_unit(cfg25, pose, index, unit, base_cfg=base_cfg)
        if refinement_source is None
        else _refined_trajectory(cfg25, pose, index, unit, refinement_source)
    )
    placement = BPlacement(
        int(pose["candidate_index"]), tuple(pose["position_m"]), (1.0, 0.0, 0.0, 0.0), 0.0
    )
    summary, _ = run_b_acquisition_trajectory(
        cfg25, trajectory, placement=placement, scene_cfg=base_cfg
    )
    summary.update({
        "pose_candidate_index": int(pose["candidate_index"]),
        "geometry_accessible_fingers": pose["accessible_fingers"],
        "geometry_topology": "+".join(pose["accessible_fingers"]),
        "trial_id": stable_trial_id(
            "phase2S-b-only", {"candidate_index": index, "config_hash": cfg_hash}
        ),
        "experiment_id": "phase2S_b_only_calibration",
        "calibration_only": True,
        "config_hash": cfg_hash,
        "git_commit_sha": commit,
    })
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=int, choices=(4096, 8192), default=4096)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    phase2s, source = load_phase2s_config()
    phase26, phase26_source = load_phase2_6_config()
    cfg25, cfg25_source = load_phase2_5_config(ROOT / phase26.frozen_phase2_5_config)
    base_cfg = load_configs(scene_filename=phase2s.scene_filename)
    if args.candidates > phase2s.second_grasp.B_only_candidate_cap:
        raise ValueError("candidate count exceeds the Phase 2S B-only cap")
    pose_path = ROOT / phase2s.output_dir / "b_pose_graspability" / "selected_poses.yaml"
    poses = yaml.safe_load(pose_path.read_text(encoding="utf-8"))["selected_poses"]
    if len(poses) != 50:
        raise RuntimeError("small-B geometry stage did not select exactly 50 poses")
    cfg_hash = config_hash([
        source,
        phase26_source,
        cfg25_source,
        pose_path,
        ROOT / "configs" / "hand_allegro.yaml",
        ROOT / "configs" / phase2s.scene_filename,
        ROOT / "configs" / "task_sequential.yaml",
        ROOT / "seqgrasp" / "experiments" / "phase2_6_dynamic.py",
    ])
    output = ROOT / phase2s.output_dir / "b_only_dynamic" / cfg_hash[:12]
    store = IncrementalJsonlStore(output / "candidate_results.jsonl", 30.0, 0.05)
    completed = store.completed_ids()
    unit = qmc.LatinHypercube(
        70, seed=phase2s.second_grasp.B_only_seed
    ).random(phase2s.second_grasp.B_only_candidate_cap)
    pending = [
        index for index in range(args.candidates)
        if stable_trial_id(
            "phase2S-b-only", {"candidate_index": index, "config_hash": cfg_hash}
        ) not in completed
    ]
    broad_rows = [
        row for row in store.records()
        if row.get("config_hash") == cfg_hash and int(row["candidate_index"]) < 4096
    ]
    existing_successes = sum(
        row.get("B_acquired", False)
        for row in store.records()
        if row.get("config_hash") == cfg_hash and int(row["candidate_index"]) < args.candidates
    )
    if existing_successes >= phase2s.second_grasp.B_only_success_target:
        pending = []
    refinement_sources = []
    if args.candidates == phase2s.second_grasp.B_only_candidate_cap:
        if len(broad_rows) != 4096:
            raise RuntimeError("the 4096-candidate broad search must finish before deterministic refinement")
        ranked_sources = sorted(broad_rows, key=lambda row: _refinement_key(row, cfg25), reverse=True)
        seen_trajectories = set()
        for row in ranked_sources:
            if row["unsupported_contact_steps"] != cfg25.timing.unsupported_hold_steps:
                continue
            key = int(row["candidate_index"])
            if key in seen_trajectories:
                continue
            refinement_sources.append(row)
            seen_trajectories.add(key)
            if len(refinement_sources) == 64:
                break
        if not refinement_sources:
            raise RuntimeError("broad search produced no contact-persistent refinement source")
    workers = min(
        args.workers or max(1, (os.cpu_count() or 1) // 2),
        phase2s.state.maximum_workers,
    )
    commit = git_commit_sha(ROOT)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        buffered = []
        def payload_for(index):
            source_row = None
            pose = poses[index % len(poses)]
            if index >= 4096:
                source_row = refinement_sources[(index - 4096) % len(refinement_sources)]
                pose = poses[int(source_row["candidate_index"]) % len(poses)]
            return (cfg25, base_cfg, pose, index, unit[index], cfg_hash, commit, source_row)

        payloads = (payload_for(index) for index in pending)
        for completed_index, result in enumerate(executor.map(evaluate, payloads), start=1):
            buffered.append(result)
            if len(buffered) == workers or completed_index == len(pending):
                store.append_many(buffered)
                buffered.clear()
            if completed_index % (workers * 4) == 0 or completed_index == len(pending):
                print(
                    f"Phase 2S B-only: {len(completed) + completed_index}/{args.candidates}",
                    flush=True,
                )
    rows = [
        row for row in store.records()
        if row.get("config_hash") == cfg_hash and int(row["candidate_index"]) < args.candidates
    ]
    ranked = sorted(rows, key=b_only_lexicographic_key, reverse=True)
    successes = [row for row in ranked if row["B_acquired"]]
    target = phase2s.second_grasp.B_only_success_target
    minimum = phase2s.second_grasp.B_only_minimum_successes
    if len(successes) >= target:
        status = "PASS_TARGET"
        exit_code = 0
    elif args.candidates < phase2s.second_grasp.B_only_candidate_cap:
        status = "EXPANSION_REQUIRED"
        exit_code = 2
    elif len(successes) >= minimum:
        status = "PASS_MINIMUM_AFTER_CAP"
        exit_code = 0
    else:
        status = "PHASE2S_B_ONLY_CONTROL_FAILED"
        exit_code = 3
    summary = {
        "status": status,
        "candidate_budget": args.candidates,
        "candidate_count": len(rows),
        "completed_candidates": len(rows),
        "successful_B_only_trajectories": len(successes),
        "target_success_count": target,
        "minimum_success_count": minimum,
        "successful_topologies": dict(Counter(row["geometry_topology"] for row in successes)),
        "failure_mechanisms": dict(
            Counter(row["failure_mechanism"] for row in rows if not row["B_acquired"])
        ),
        "best": ranked[0] if ranked else None,
        "config_hash": cfg_hash,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for profile_index, row in enumerate(successes[:20], start=1):
        pose = next(
            item for item in poses if int(item["candidate_index"]) == int(row["pose_candidate_index"])
        )
        trajectory = _from_payload(row["trajectory"], int(row["candidate_index"]))
        (ROOT / "configs" / "grasps" / f"phase2S_b_only_{profile_index:02d}.yaml").write_text(
            yaml.safe_dump({
                "pose": pose,
                "trajectory": trajectory.__dict__,
                "source_candidate_index": int(row["candidate_index"]),
                "calibration_only": True,
                "scene_filename": phase2s.scene_filename,
            }, sort_keys=False),
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
