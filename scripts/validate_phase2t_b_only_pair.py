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
import yaml

from seqgrasp.config import ROOT, load_configs
from seqgrasp.control import resolve_hand_indices
from seqgrasp.experiments.metadata import config_hash, git_commit_sha
from seqgrasp.experiments.phase2_5_trajectory import BAcquisitionTrajectory, run_b_acquisition_trajectory
from seqgrasp.experiments.phase2r_second_grasp import remap_pair_trajectory
from seqgrasp.experiments.resumable import IncrementalJsonlStore, stable_trial_id
from seqgrasp.experiments.second_grasp import BPlacement
from seqgrasp.phase2_5_config import load_phase2_5_config
from seqgrasp.phase2s_config import load_phase2s_config
from seqgrasp.phase2t_config import load_phase2t_config
from seqgrasp.scene_builder import build_scene


PAIR = ("index", "middle")
FINGER_ORDER = ("index", "middle", "ring", "thumb")


def _trajectory(payload: dict, index: int) -> BAcquisitionTrajectory:
    return BAcquisitionTrajectory(
        candidate_index=index,
        approach_joint_rad=tuple(payload["approach_joint_rad"]),
        precontact_joint_rad=tuple(payload["precontact_joint_rad"]),
        closing_joint_rad=tuple(payload["closing_joint_rad"]),
        hold_joint_rad=tuple(payload["hold_joint_rad"]),
        close_steps=int(payload["close_steps"]),
        per_finger_close_delay_steps=tuple(payload["per_finger_close_delay_steps"]),
        fixture_release_delay_steps=int(payload["fixture_release_delay_steps"]),
    )


def _placement(frozen: dict, profile: dict, index: int, seed: int, refinement_source=None) -> BPlacement:
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    if refinement_source is not None:
        center = np.asarray(refinement_source["placement"]["position_m"], dtype=float)
        position = tuple(float(value) for value in center + rng.uniform(-0.003, 0.003, 3))
        yaw = float(refinement_source["placement"]["yaw_rad"] + rng.uniform(-0.08, 0.08))
    elif index < 420:
        bounds = frozen["center_bounds_m"]
        position = tuple(float(rng.uniform(*bounds[axis])) for axis in "xyz")
        yaw = float(rng.uniform(*frozen["yaw_bounds_rad"]))
    else:
        center = np.asarray(profile["pose"]["position_m"], dtype=float)
        position = tuple(float(value) for value in center + rng.uniform(
            [-0.015, -0.040, -0.020], [0.015, 0.040, 0.020],
        ))
        yaw = float(rng.uniform(-0.2, 0.2))
    quaternion = tuple(float(value) for value in Rotation.from_euler("z", yaw).as_quat(scalar_first=True))
    return BPlacement(index, position, quaternion, yaw)


def _source_pair(profile: dict) -> tuple[str, str]:
    accessible = tuple(profile["pose"]["accessible_fingers"])
    preferred = (("middle", "ring"), ("ring", "thumb"), ("index", "thumb"), ("middle", "thumb"))
    return next((pair for pair in preferred if set(pair).issubset(accessible)), tuple(accessible[:2]))


def _evaluate(payload):
    cfg25, scene_cfg, profile_path, profile, baseline, frozen, index, seed, refinement_source = payload
    if refinement_source is None:
        source = _trajectory(profile["trajectory"], index)
        source_pair = _source_pair(profile)
        adapted = remap_pair_trajectory(source, source_pair, PAIR, baseline)
    else:
        source_pair = tuple(refinement_source["source_pair"])
        adapted = _trajectory(refinement_source["trajectory"], index)
    if index >= 420:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 1, index]))
        active = np.repeat([finger in PAIR for finger in FINGER_ORDER], 4)

        def perturbed(values, scale_bounds):
            values = np.asarray(values, dtype=float)
            result = baseline + rng.uniform(*scale_bounds) * (values - baseline)
            noise = 0.025 if refinement_source is not None else 0.07
            result[active] += rng.uniform(-noise, noise, int(np.sum(active)))
            result[~active] = baseline[~active]
            return tuple(float(value) for value in result)

        adapted = replace(
            adapted,
            approach_joint_rad=perturbed(adapted.approach_joint_rad, (0.97, 1.03) if refinement_source else (0.80, 1.10)),
            precontact_joint_rad=perturbed(adapted.precontact_joint_rad, (0.97, 1.03) if refinement_source else (0.85, 1.15)),
            closing_joint_rad=perturbed(adapted.closing_joint_rad, (0.97, 1.03) if refinement_source else (0.90, 1.20)),
            hold_joint_rad=perturbed(adapted.hold_joint_rad, (0.97, 1.03) if refinement_source else (0.90, 1.20)),
            close_steps=int(np.clip(adapted.close_steps + rng.integers(-30 if refinement_source else -60, 31 if refinement_source else 61), 250, 500)),
            fixture_release_delay_steps=int(np.clip(adapted.fixture_release_delay_steps + rng.integers(-25 if refinement_source else -50, 26 if refinement_source else 51), 0, 200)),
        )
    summary, arrays = run_b_acquisition_trajectory(
        cfg25, adapted, placement=_placement(frozen, profile, index, seed, refinement_source),
        scene_cfg=scene_cfg, collect_timeseries=True,
    )
    release = int(summary["fixture_release_timestep"])
    flags = arrays["B_per_finger_contact_flag"]
    pair_indices = [FINGER_ORDER.index(finger) for finger in PAIR]
    other_indices = [index for index in range(4) if index not in pair_indices]
    both_pair_at_release = bool(np.all(flags[release:, pair_indices][-1] > 0))
    no_extra_fingers = bool(not np.any(flags[:, other_indices] > 0))
    strict = bool(summary["B_acquired"] and both_pair_at_release and no_extra_fingers)
    return {
        **summary,
        "trial_id": stable_trial_id("phase2T-pair-specific-b-only", index),
        "candidate_index": index,
        "source_profile": profile_path.name,
        "refinement_source_candidate_index": None if refinement_source is None else int(refinement_source["candidate_index"]),
        "source_pair": list(source_pair),
        "permitted_acquisition_pair": list(PAIR),
        "both_selected_digits_contact_at_final": both_pair_at_release,
        "extra_acquisition_finger_contact": not no_extra_fingers,
        "strict_pair_specific_success": strict,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=int, default=420)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    phase2t, phase2t_path = load_phase2t_config()
    phase2s, phase2s_path = load_phase2s_config()
    cfg25, cfg25_path = load_phase2_5_config()
    if args.candidates > phase2t.second_grasp.b_only_candidate_cap:
        raise ValueError("candidate cap exceeded")
    scene_cfg = load_configs(scene_filename=phase2t.scene_filename)
    model, data = build_scene(scene_cfg)
    indices = resolve_hand_indices(model, scene_cfg.hand)
    baseline = np.asarray(data.qpos[indices.qpos_addresses], dtype=float)
    profile_paths = sorted((ROOT / "configs" / "grasps").glob("phase2S_b_only_*.yaml"))
    profiles = [(path, yaml.safe_load(path.read_text(encoding="utf-8"))) for path in profile_paths]
    frozen_path = ROOT / "configs" / "phase2S_frozen_B_distribution.yaml"
    frozen = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
    cfg_hash = config_hash([
        phase2t_path, phase2s_path, cfg25_path, frozen_path, *profile_paths,
        ROOT / "scripts" / "validate_phase2t_b_only_pair.py",
    ])
    output = ROOT / phase2t.output_dir / "b_only_pair" / cfg_hash[:12]
    store = IncrementalJsonlStore(output / "candidate_results.jsonl", 30.0, 0.05)
    completed = store.completed_ids()
    pending = [
        index for index in range(args.candidates)
        if stable_trial_id("phase2T-pair-specific-b-only", index) not in completed
    ]
    workers = min(args.workers or max(1, (os.cpu_count() or 1) // 2), phase2t.state_search.maximum_workers)
    broad_rows = [row for row in store.records() if int(row["candidate_index"]) < 1024]
    refinement_sources = []
    if args.candidates > 1024:
        if len(broad_rows) != 1024:
            raise RuntimeError("complete the 1024-candidate broad pair search before refinement")

        def refinement_key(row):
            violation = (
                max(0.0, row["maximum_B_penetration_m"] / cfg25.criteria.maximum_penetration_m - 1.0)
                + max(0.0, row["maximum_B_translation_after_release_m"] / cfg25.criteria.maximum_B_translation_m - 1.0)
                + max(0.0, row["maximum_B_orientation_after_release_rad"] / cfg25.criteria.maximum_B_orientation_rad - 1.0)
            )
            return (
                int(row["invalid_reason"] is None), row["unsupported_contact_steps"],
                int(row["first_post_release_contact_loss_step"] is None), -violation,
                -row["maximum_B_translation_after_release_m"], -row["candidate_index"],
            )

        refinement_sources = sorted(broad_rows, key=refinement_key, reverse=True)[:64]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        def payload_for(index):
            refinement = refinement_sources[(index - 1024) % len(refinement_sources)] if index >= 1024 else None
            profile_index = index % len(profiles)
            if refinement is not None:
                profile_index = next(
                    i for i, (path, _) in enumerate(profiles) if path.name == refinement["source_profile"]
                )
            return (
                cfg25, scene_cfg, profiles[profile_index][0], profiles[profile_index][1], baseline,
                frozen, index, phase2t.second_grasp.b_only_seed, refinement,
            )

        payloads = (payload_for(index) for index in pending)
        buffer = []
        for count, result in enumerate(executor.map(_evaluate, payloads), start=1):
            result.update({"experiment_id": phase2t.experiment_id, "config_hash": cfg_hash, "git_commit_sha": git_commit_sha(ROOT)})
            buffer.append(result)
            if len(buffer) >= workers or count == len(pending):
                store.append_many(buffer); buffer.clear()
            if count % (workers * 4) == 0 or count == len(pending):
                print(f"Phase 2T pair-specific B-only: {len(completed) + count}/{args.candidates}", flush=True)
    rows = [row for row in store.records() if int(row["candidate_index"]) < args.candidates]
    successes = [row for row in rows if row["strict_pair_specific_success"]]
    status = "PASS" if len(successes) >= phase2t.second_grasp.b_only_hard_minimum else "PHASE2T_NO_PAIR_SPECIFIC_B_CONTROL"
    summary = {
        "status": status,
        "free_finger_topology": list(PAIR),
        "candidate_count": len(rows),
        "strict_success_count": len(successes),
        "target_success_count": phase2t.second_grasp.b_only_success_target,
        "source_profile_successes": dict(Counter(row["source_profile"] for row in successes)),
        "failure_mechanisms": dict(Counter(row["failure_mechanism"] for row in rows if not row["strict_pair_specific_success"])),
        "best_successes": successes[:10],
        "config_hash": cfg_hash,
        "git_commit_sha": git_commit_sha(ROOT),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if successes else 3


if __name__ == "__main__":
    raise SystemExit(main())
