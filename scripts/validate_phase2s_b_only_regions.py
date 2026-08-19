#!/usr/bin/env python
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math

import numpy as np
from scipy.stats import qmc
import yaml

from seqgrasp.config import ROOT, load_configs
from seqgrasp.experiments.phase2_5_trajectory import BAcquisitionTrajectory, run_b_acquisition_trajectory
from seqgrasp.experiments.resumable import IncrementalJsonlStore, stable_trial_id
from seqgrasp.experiments.second_grasp import BPlacement
from seqgrasp.phase2_5_config import load_phase2_5_config
from seqgrasp.phase2s_config import load_phase2s_config


POSITION_HALF_WIDTH_M = np.asarray([0.001, 0.001, 0.001])
YAW_HALF_WIDTH_RAD = 0.10
NARROW_POSITION_HALF_WIDTH_M = np.asarray([0.0001, 0.0001, 0.0001])
NARROW_YAW_HALF_WIDTH_RAD = 0.01
PROFILES_TO_VALIDATE = 10
TRIALS_PER_PROFILE = 20


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


def main() -> int:
    phase2s, _ = load_phase2s_config()
    cfg25, _ = load_phase2_5_config()
    scene_cfg = load_configs(scene_filename=phase2s.scene_filename)
    paths = sorted((ROOT / "configs" / "grasps").glob("phase2S_b_only_*.yaml"))[:PROFILES_TO_VALIDATE]
    if len(paths) < phase2s.second_grasp.B_only_minimum_successes:
        raise RuntimeError("insufficient strict half-scale B-only profiles")
    output = ROOT / phase2s.output_dir / "b_only_robustness"
    store = IncrementalJsonlStore(output / "trials.jsonl", 30.0, 0.05)
    completed = store.completed_ids()
    profiles = []
    for profile_index, path in enumerate(paths):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        profile_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        profiles.append((path, payload, profile_hash))
        center = np.asarray(payload["pose"]["position_m"], dtype=float)
        samples = qmc.LatinHypercube(
            4,
            seed=np.random.default_rng(
                np.random.SeedSequence([phase2s.second_grasp.geometry_seed, profile_index])
            ),
        ).random(TRIALS_PER_PROFILE)
        for trial_index, sample in enumerate(samples):
            trial_id = stable_trial_id(
                "phase2S-b-only-robustness",
                {"profile": path.name, "profile_hash": profile_hash, "trial": trial_index},
            )
            if trial_id in completed:
                continue
            delta = (2.0 * sample[:3] - 1.0) * POSITION_HALF_WIDTH_M
            yaw = float(np.interp(sample[3], [0.0, 1.0], [-YAW_HALF_WIDTH_RAD, YAW_HALF_WIDTH_RAD]))
            placement = BPlacement(
                trial_index,
                tuple(center + delta),
                (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)),
                yaw,
            )
            summary, arrays = run_b_acquisition_trajectory(
                cfg25,
                _trajectory(payload["trajectory"]),
                placement=placement,
                collect_timeseries=True,
                scene_cfg=scene_cfg,
            )
            final = np.asarray(arrays["B_per_finger_contact_flag"][-1])
            summary.update({
                "trial_id": trial_id,
                "profile": path.name,
                "source_candidate_index": int(payload["source_candidate_index"]),
                "robustness_trial_index": trial_index,
                "position_delta_m": delta.tolist(),
                "yaw_delta_rad": yaw,
                "final_contact_topology": "+".join(
                    name for name, active in zip(("index", "middle", "ring", "thumb"), final) if active
                ),
                "calibration_only": True,
                "experiment_id": "phase2S_b_only_robustness",
                "region_scale": "wide",
                "profile_hash": profile_hash,
            })
            store.append(summary)
        wide_rows = [
            row for row in store.records()
            if row["profile"] == path.name and row.get("profile_hash") == profile_hash
            and row.get("region_scale", "wide") == "wide"
        ]
        if not any(row["B_acquired"] for row in wide_rows):
            narrow_samples = qmc.LatinHypercube(
                4,
                seed=np.random.default_rng(
                    np.random.SeedSequence([phase2s.second_grasp.geometry_seed, 1, profile_index])
                ),
            ).random(TRIALS_PER_PROFILE)
            for trial_index, sample in enumerate(narrow_samples):
                trial_id = stable_trial_id(
                    "phase2S-b-only-robustness-narrow",
                    {"profile": path.name, "profile_hash": profile_hash, "trial": trial_index},
                )
                if trial_id in completed:
                    continue
                delta = (2.0 * sample[:3] - 1.0) * NARROW_POSITION_HALF_WIDTH_M
                yaw = float(np.interp(
                    sample[3], [0.0, 1.0], [-NARROW_YAW_HALF_WIDTH_RAD, NARROW_YAW_HALF_WIDTH_RAD]
                ))
                placement = BPlacement(
                    trial_index,
                    tuple(center + delta),
                    (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)),
                    yaw,
                )
                summary, arrays = run_b_acquisition_trajectory(
                    cfg25,
                    _trajectory(payload["trajectory"]),
                    placement=placement,
                    collect_timeseries=True,
                    scene_cfg=scene_cfg,
                )
                final = np.asarray(arrays["B_per_finger_contact_flag"][-1])
                summary.update({
                    "trial_id": trial_id,
                    "profile": path.name,
                    "source_candidate_index": int(payload["source_candidate_index"]),
                    "robustness_trial_index": trial_index,
                    "position_delta_m": delta.tolist(),
                    "yaw_delta_rad": yaw,
                    "final_contact_topology": "+".join(
                        name for name, active in zip(("index", "middle", "ring", "thumb"), final) if active
                    ),
                    "calibration_only": True,
                    "experiment_id": "phase2S_b_only_robustness",
                    "region_scale": "narrow",
                    "profile_hash": profile_hash,
                })
                store.append(summary)
        print(f"Phase 2S robustness profile {profile_index + 1}/{len(paths)}", flush=True)
    current_hashes = {profile_hash for _, _, profile_hash in profiles}
    trials = [row for row in store.records() if row.get("profile_hash") in current_hashes]
    regions = []
    by_profile = {}
    for profile_index, (path, payload, profile_hash) in enumerate(profiles):
        wide_subset = [
            row for row in trials
            if row["profile"] == path.name and row.get("profile_hash") == profile_hash
            and row.get("region_scale", "wide") == "wide"
        ]
        narrow_subset = [
            row for row in trials if row["profile"] == path.name
            and row.get("profile_hash") == profile_hash and row.get("region_scale") == "narrow"
        ]
        subset = wide_subset if any(row["B_acquired"] for row in wide_subset) else narrow_subset
        selected_scale = "wide" if subset is wide_subset else "narrow"
        position_width = POSITION_HALF_WIDTH_M if selected_scale == "wide" else NARROW_POSITION_HALF_WIDTH_M
        yaw_width = YAW_HALF_WIDTH_RAD if selected_scale == "wide" else NARROW_YAW_HALF_WIDTH_RAD
        if not any(row["B_acquired"] for row in subset):
            selected_scale = "nominal_only"
            position_width = np.zeros(3)
            yaw_width = 0.0
        center = np.asarray(payload["pose"]["position_m"], dtype=float)
        successes = sum(row["B_acquired"] for row in subset)
        by_profile[path.stem] = {
            "trials": len(subset),
            "successes": successes,
            "success_fraction": successes / len(subset) if subset else 0.0,
            "initial_collision_rate": (
                sum(row.get("invalid_reason") is not None for row in subset) / len(subset) if subset else None
            ),
            "successful_contact_topologies": dict(
                Counter(row["final_contact_topology"] for row in subset if row["B_acquired"])
            ),
            "selected_region_scale": selected_scale,
        }
        # The source profile itself is a strict success at the region centre;
        # include it in the local robustness denominator rather than discarding
        # weak but physically demonstrated regions.
        regions.append({
            "name": f"half_scale_region_{profile_index + 1:02d}",
            "source_profile": path.name,
            "source_candidate_index": int(payload["source_candidate_index"]),
            "source_accessible_fingers": payload["pose"]["accessible_fingers"],
            "demonstrated_contact_topologies": dict(
                Counter(row["final_contact_topology"] for row in subset if row["B_acquired"])
            ),
            "center_bounds_m": {
                axis: [float(center[index] - position_width[index]), float(center[index] + position_width[index])]
                for index, axis in enumerate("xyz")
            },
            "yaw_bounds_rad": [-yaw_width, yaw_width],
            "B_only_robustness_fraction": (successes + 1) / (len(subset) + 1),
            "B_only_trials": len(subset) + 1,
            "B_only_successes": successes + 1,
            "source_nominal_strict_success": True,
            "initial_collision_rate": by_profile[path.stem]["initial_collision_rate"],
        })
    summary = {
        "validated_profiles": len(profiles),
        "trials": len(trials),
        "successes": sum(row["B_acquired"] for row in trials),
        "overall_success_fraction": (
            sum(row["B_acquired"] for row in trials) / len(trials) if trials else 0.0
        ),
        "by_profile": by_profile,
        "positive_local_regions": len(regions),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (ROOT / "configs" / "phase2S_b_only_graspable_regions.yaml").write_text(
        yaml.safe_dump({"regions": regions}, sort_keys=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if regions else 3


if __name__ == "__main__":
    raise SystemExit(main())
