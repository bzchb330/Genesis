#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import shutil
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "seqgrasp-matplotlib"))
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from seqgrasp.config import ROOT
from seqgrasp.experiments.metadata import git_commit_sha
from seqgrasp.experiments.resource_components import (
    PALM_REFERENCE_TO_COMPILED,
    RESOURCE_METHOD_ID,
    RESOURCE_RECORDS_FILENAME,
    compute_resource_components,
    free_finger_workspace_volume,
    reconstruct_grasp,
)
from seqgrasp.experiments.resumable import IncrementalJsonlStore, stable_trial_id
from seqgrasp.phase2_config import load_phase2_config


def _compute_task(payload):
    row, resources, seed, commit = payload
    values = compute_resource_components(row, resources, seed)
    return {
        "trial_id": stable_trial_id("phase2-resource-components", row["grasp_id"]),
        "grasp_id": row["grasp_id"],
        "occupied_finger_count": values.occupied_finger_count,
        "occupied_finger_mask": list(values.occupied_finger_mask),
        "free_finger_workspace_vol_m3": values.free_finger_workspace_vol_m3,
        "free_palm_volume_m3": values.free_palm_volume_m3,
        "units": {"occupied_finger_count": "count", "free_finger_workspace_vol_m3": "m^3", "free_palm_volume_m3": "m^3"},
        "git_commit_sha": commit,
        "source_config_hash": row["config_hash"],
        "resource_method_id": RESOURCE_METHOD_ID,
    }


def _dataset_dir(root: Path) -> Path:
    candidates = []
    for accepted_path in root.glob("*/accepted_grasps.jsonl"):
        candidates.append((len(accepted_path.read_text(encoding="utf-8").splitlines()), accepted_path.parent))
    if not candidates:
        raise FileNotFoundError("no grasp dataset found; run scripts/build_grasp_dataset.py")
    return max(candidates, key=lambda item: item[0])[1]


def _representatives(records: list[dict]) -> list[dict]:
    selected = []
    for occupied_count in sorted({int(row["occupied_finger_count"]) for row in records}):
        group = sorted(
            (row for row in records if int(row["occupied_finger_count"]) == occupied_count),
            key=lambda row: float(row["ferrari_canny_epsilon"]),
        )
        for quantile in (0.0, 0.5, 1.0):
            row = group[round(quantile * (len(group) - 1))]
            if row["grasp_id"] not in {item["grasp_id"] for item in selected}:
                selected.append(row)
    return selected


def _plot_palm_box(record: dict, lower, upper, output: Path) -> dict:
    cfg, model, data, _ = reconstruct_grasp(record)
    palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.hand.palm_body)
    rotation = data.xmat[palm_id].reshape(3, 3)
    origin = data.xpos[palm_id]
    corners_local = np.asarray([
        [x, y, z] for x in (lower[0], upper[0]) for y in (lower[1], upper[1]) for z in (lower[2], upper[2])
    ])
    corners_compiled = corners_local @ PALM_REFERENCE_TO_COMPILED.T
    corners = origin + corners_compiled @ rotation.T
    collision_centres = np.asarray([
        data.geom_xpos[index] for index in range(model.ngeom)
        if (model.geom_contype[index] or model.geom_conaffinity[index])
        and (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[index])) or "") != "world"
    ])
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.sum(corners_local[i] != corners_local[j]) == 1]
    plt.style.use(ROOT / "configs" / "phase2_publication.mplstyle")
    fig = plt.figure(figsize=(7.0, 3.3))
    for panel, view in enumerate(((20, -60), (90, -90)), start=1):
        ax = fig.add_subplot(1, 2, panel, projection="3d")
        ax.scatter(*collision_centres.T, s=10, color="#0072B2", label="collision geom centres")
        for i, j in edges:
            ax.plot(*np.stack([corners[i], corners[j]]).T, color="#D55E00", linewidth=1)
        ax.scatter(*origin, marker="x", color="black", s=25, label="palm origin")
        ax.set(xlabel="world x [m]", ylabel="world y [m]", zlabel="world z [m]")
        ax.view_init(*view)
        ax.set_box_aspect((1, 1, 1))
    fig.legend(loc="upper center", ncol=2, frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    fingertip_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in cfg.hand.fingertip_bodies]
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object_a")
    diagnostic_world = np.vstack([data.xpos[fingertip_ids], data.xpos[object_id]])
    diagnostic_compiled = (diagnostic_world - origin) @ rotation
    diagnostic_reference = diagnostic_compiled @ PALM_REFERENCE_TO_COMPILED
    inside = np.all((diagnostic_reference >= lower) & (diagnostic_reference <= upper), axis=1)
    return {
        "fingertip_centres_inside": int(np.sum(inside[:len(fingertip_ids)])),
        "fingertip_centres_total": len(fingertip_ids),
        "held_object_centre_inside": bool(inside[-1]),
        "reference_palm_frame_points_m": diagnostic_reference.tolist(),
        "orientation": "PI reference [x,y,z] maps to compiled Allegro palm [-z,y,x]",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute the three raw Phase 2 resource components")
    parser.add_argument("--config", default="configs/phase2_physics_validation.yaml")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    phase2, _ = load_phase2_config(ROOT / args.config)
    dataset_dir = _dataset_dir(ROOT / phase2.persistence.output_dir / "grasp_dataset")
    accepted = [json.loads(line) for line in (dataset_dir / "accepted_grasps.jsonl").read_text(encoding="utf-8").splitlines()]
    if args.limit is not None:
        accepted = accepted[:args.limit]
    workers = min(args.workers or max(1, (os.cpu_count() or 1) // 2), 8)
    store = IncrementalJsonlStore(dataset_dir / RESOURCE_RECORDS_FILENAME, phase2.persistence.lock_timeout_seconds, phase2.persistence.lock_poll_seconds)
    completed = store.completed_ids()
    pending = [row for row in accepted if stable_trial_id("phase2-resource-components", row["grasp_id"]) not in completed]

    commit = git_commit_sha(ROOT)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        buffered = []
        payloads = ((row, phase2.resources, phase2.dataset.seed, commit) for row in pending)
        for index, result in enumerate(executor.map(_compute_task, payloads), start=1):
            buffered.append(result)
            if len(buffered) == workers or index == len(pending):
                store.append_many(buffered)
                buffered.clear()
            if index % workers == 0 or index == len(pending):
                print(f"resource components: {len(completed) + index}/{len(accepted)}", flush=True)
    records = store.records()
    palm_box_diagnostics = {}
    if accepted:
        box_figure = dataset_dir / "figures" / f"free_palm_measurement_box_{RESOURCE_METHOD_ID}.pdf"
        palm_box_diagnostics = _plot_palm_box(
            accepted[0],
            np.asarray(phase2.resources.free_palm_box_lower_m),
            np.asarray(phase2.resources.free_palm_box_upper_m),
            box_figure,
        )
        publication_box = ROOT / "docs" / "figures" / "phase2" / "free_palm_measurement_box.pdf"
        publication_box.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(box_figure, publication_box)
    convergence = []
    for row in _representatives(accepted):
        previous = None
        for budget in phase2.resources.workspace_convergence_samples:
            volume = free_finger_workspace_volume(row, phase2.resources, budget, phase2.dataset.seed)
            convergence.append({
                "grasp_id": row["grasp_id"], "occupied_finger_count": row["occupied_finger_count"],
                "ferrari_canny_epsilon": row["ferrari_canny_epsilon"], "samples": budget,
                "volume_m3": volume,
                "relative_change": None if previous is None or previous == 0 else (volume - previous) / previous,
            })
            previous = volume
    (dataset_dir / f"workspace_convergence_{RESOURCE_METHOD_ID}.json").write_text(json.dumps(convergence, indent=2), encoding="utf-8")
    convergence_summary = {}
    for budget in phase2.resources.workspace_convergence_samples:
        group = [row for row in convergence if row["samples"] == budget]
        changes = [row["relative_change"] for row in group if row["relative_change"] is not None]
        convergence_summary[str(budget)] = {
            "representative_grasps": len(group),
            "mean_volume_m3": float(np.mean([row["volume_m3"] for row in group])) if group else None,
            "mean_relative_change_from_previous_budget": float(np.mean(changes)) if changes else None,
            "maximum_absolute_relative_change_from_previous_budget": float(np.max(np.abs(changes))) if changes else None,
        }
    summary = {
        "records": len(records),
        "occupied_finger_count_distribution": dict(Counter(str(row["occupied_finger_count"]) for row in records)),
        "components": {
            key: {stat: float(func([row[key] for row in records])) for stat, func in (("min", np.min), ("max", np.max), ("mean", np.mean), ("std", np.std))}
            for key in ("occupied_finger_count", "free_finger_workspace_vol_m3", "free_palm_volume_m3")
        } if records else {},
        "workspace_convergence_records": len(convergence),
        "workspace_convergence": convergence_summary,
        "free_palm_method": "palm-frame voxel-centre occupancy against actual box/capsule collision geometry",
        "palm_axes": "PI reference [x,y,z] maps to compiled Allegro palm [-z,y,x]",
        "resource_method_id": RESOURCE_METHOD_ID,
        "free_palm_box_debug": palm_box_diagnostics,
        "scalar_J": None,
    }
    (dataset_dir / f"resource_summary_{RESOURCE_METHOD_ID}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
