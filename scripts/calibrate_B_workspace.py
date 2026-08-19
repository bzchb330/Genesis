#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from seqgrasp.config import ROOT
from seqgrasp.experiments.b_workspace import analyze_B_geometry_state, free_fingertip_workspace_clouds
from seqgrasp.experiments.resource_components import RESOURCE_RECORDS_FILENAME
from seqgrasp.experiments.second_grasp import BPlacement
from seqgrasp.phase2_config import load_phase2_config


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _dataset_dir(root: Path) -> Path:
    candidates = [(len(path.read_text(encoding="utf-8").splitlines()), path.parent) for path in root.glob("*/accepted_grasps.jsonl")]
    return max(candidates, key=lambda item: item[0])[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Geometry-only calibration of the global fixture-presented B volume")
    parser.add_argument("--config", default="configs/phase2_physics_validation.yaml")
    parser.add_argument("--grid-step", type=float, default=0.02)
    args = parser.parse_args()
    phase2, _ = load_phase2_config(ROOT / args.config)
    dataset = _dataset_dir(ROOT / phase2.persistence.output_dir / "grasp_dataset")
    accepted = _jsonl(dataset / "accepted_grasps.jsonl")[:200]
    resources = {row["grasp_id"]: row for row in _jsonl(dataset / RESOURCE_RECORDS_FILENAME)}
    states = []
    for index, record in enumerate(accepted, start=1):
        enriched = {**record, **resources[record["grasp_id"]]}
        states.append((record["grasp_id"], *free_fingertip_workspace_clouds(
            enriched, phase2.resources, phase2.second_grasp.geometry_workspace_samples,
            phase2.second_grasp.seed,
        )))
        if index % 20 == 0:
            print(f"workspace clouds: {index}/{len(accepted)}", flush=True)
    step = args.grid_step
    axes = (
        np.arange(-0.02, 0.12001, step),
        np.arange(0.00, 0.12001, step),
        np.arange(0.10, 0.22001, step),
    )
    candidates = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    rows = []
    for index, center in enumerate(candidates):
        yaw = 0.0
        placement = BPlacement(index, tuple(float(x) for x in center), (1.0, 0.0, 0.0, 0.0), yaw)
        metrics = [
            analyze_B_geometry_state(cfg, model, data, clouds, radii, phase2.resources, placement)
            for _, cfg, model, data, clouds, radii in states
        ]
        rows.append({
            "candidate_index": index,
            "center_m": center.tolist(),
            "reachable_population_fraction": float(np.mean([row["reachable"] for row in metrics])),
            "initial_collision_A_fraction": float(np.mean([row["initial_collision_A"] for row in metrics])),
            "initial_collision_hand_fraction": float(np.mean([row["initial_collision_hand"] for row in metrics])),
            "inside_measured_free_palm_fraction": float(np.mean([row["inside_measured_free_palm_region"] for row in metrics])),
            "minimum_distance_median": float(np.median([row["minimum_free_fingertip_to_B_m"] for row in metrics])),
            "per_grasp": [{"grasp_id": state[0], **metric} for state, metric in zip(states, metrics)],
        })
        if (index + 1) % 50 == 0:
            print(f"candidate centers: {index + 1}/{len(candidates)}", flush=True)
    eligible = [row for row in rows if 0.20 <= row["reachable_population_fraction"] <= 0.80]
    if not eligible:
        raise RuntimeError("no geometry-only candidate meets the PI calibration target")
    best = min(eligible, key=lambda row: (
        row["initial_collision_A_fraction"] + row["initial_collision_hand_fraction"],
        abs(row["reachable_population_fraction"] - 0.5),
        -row["inside_measured_free_palm_fraction"],
        row["candidate_index"],
    ))
    half_width = step / 4.0
    recommended = {
        "source_A_grasps": len(accepted),
        "candidate_centers": len(rows),
        "PI_calibration_target_fraction": [0.20, 0.80],
        "selected_center_m": best["center_m"],
        "selected_center_metrics": {key: value for key, value in best.items() if key != "per_grasp"},
        "recommended_axis_aligned_bounds_m": {
            axis: [float(value - half_width), float(value + half_width)]
            for axis, value in zip("xyz", best["center_m"])
        },
        "selection_used_dynamic_outcomes": False,
    }
    output = dataset / "correlation" / "calibration"
    output.mkdir(parents=True, exist_ok=True)
    (output / "geometry_reachability_map.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output / "geometry_calibration_recommendation.json").write_text(json.dumps(recommended, indent=2), encoding="utf-8")
    print(json.dumps(recommended, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
