#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import yaml

from seqgrasp.config import ROOT, load_configs
from seqgrasp.experiments.phase2_5_trajectory import BAcquisitionTrajectory, run_b_acquisition_trajectory
from seqgrasp.experiments.second_grasp import BPlacement
from seqgrasp.phase2_5_config import load_phase2_5_config
from seqgrasp.phase2s_config import load_phase2s_config


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


def main() -> int:
    phase2s, _ = load_phase2s_config()
    cfg25, _ = load_phase2_5_config()
    scene_cfg = load_configs(scene_filename=phase2s.scene_filename)
    rows = []
    for profile_path in sorted((ROOT / "configs" / "grasps").glob("phase2_6_b_only_*.yaml")):
        payload = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
        pose = payload["pose"]
        placement = BPlacement(
            int(pose["candidate_index"]), tuple(pose["position_m"]), (1.0, 0.0, 0.0, 0.0), 0.0
        )
        summary, _ = run_b_acquisition_trajectory(
            cfg25, _trajectory(payload["trajectory"]), placement=placement, scene_cfg=scene_cfg
        )
        rows.append({
            "source_profile": str(profile_path.relative_to(ROOT)),
            "proposal_only": True,
            "replayed_with_half_scale_scene": True,
            **summary,
        })
    output = ROOT / phase2s.output_dir / "b_only_seed_replay"
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    summary = {
        "replayed_previous_successes": len(rows),
        "strict_half_scale_successes": sum(row["B_acquired"] for row in rows),
        "source_profiles": [row["source_profile"] for row in rows],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
