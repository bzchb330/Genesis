#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from seqgrasp.config import ROOT
from seqgrasp.experiments.metadata import config_hash
from seqgrasp.experiments.phase2_5_trajectory import (
    run_b_acquisition_trajectory,
    sample_b_only_trajectory,
)
from seqgrasp.phase2_5_config import load_phase2_5_config


def _write_flat_csv(path: Path, arrays: dict[str, np.ndarray]) -> None:
    keys = list(arrays)
    columns: list[str] = []
    for key in keys:
        shape = arrays[key].shape[1:]
        columns.extend([key] if not shape else [f"{key}_{'_'.join(map(str, index))}" for index in np.ndindex(shape)])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for row_index in range(len(arrays["timestep"])):
            row: list[object] = []
            for key in keys:
                value = np.asarray(arrays[key][row_index])
                row.extend([value.item()] if value.ndim == 0 else value.reshape(-1).tolist())
            writer.writerow(row)


def _plot_release_window(arrays: dict[str, np.ndarray], release: int, output: Path) -> None:
    relative = arrays["timestep"] - release
    finger_labels = ("index", "middle", "ring", "thumb")
    fig, axes = plt.subplots(3, 2, figsize=(9.0, 8.0), sharex=True)
    axes[0, 0].step(relative, arrays["fixture_active"].astype(int), where="post", label="fixture active")
    axes[0, 0].plot(relative, arrays["B_hand_contacts"], label="B-hand contacts")
    axes[0, 0].plot(relative, arrays["B_free_finger_contacts"], label="B-free-finger contacts")
    axes[0, 0].legend(fontsize=7)
    for finger_index, label in enumerate(finger_labels):
        axes[0, 1].plot(relative, arrays["B_per_finger_normal_force_N"][:, finger_index], label=label)
    axes[0, 1].plot(relative, arrays["B_hand_normal_force_N"], color="black", linewidth=1.0, label="all hand")
    axes[0, 1].set_ylabel("normal force [N]")
    axes[0, 1].legend(fontsize=7, ncol=2)
    axes[1, 0].plot(relative, arrays["B_vertical_position_m"], label="B center z")
    axes[1, 0].plot(relative, arrays["B_table_distance_m"], label="B-table distance")
    axes[1, 0].step(relative, arrays["B_table_contact"].astype(int), where="post", label="table contact")
    axes[1, 0].set_ylabel("distance [m]")
    axes[1, 0].legend(fontsize=7)
    axes[1, 1].plot(relative, np.linalg.norm(arrays["B_linear_velocity_m_per_s"], axis=1), label="linear speed [m/s]")
    axes[1, 1].plot(relative, np.linalg.norm(arrays["B_angular_velocity_rad_per_s"], axis=1), label="angular speed [rad/s]")
    axes[1, 1].legend(fontsize=7)
    axes[2, 0].plot(relative, np.max(arrays["B_penetration_depths_m"], axis=1), label="max fingertip penetration")
    axes[2, 0].plot(relative, arrays["A_displacement_m"], label="A displacement")
    axes[2, 0].set_ylabel("distance [m]")
    axes[2, 0].legend(fontsize=7)
    axes[2, 1].plot(relative, np.max(np.abs(arrays["free_finger_actuator_controls"]), axis=1), label="max |free-finger control|")
    axes[2, 1].step(relative, arrays["actuator_saturation_count"], where="post", label="saturated actuators")
    axes[2, 1].legend(fontsize=7)
    for ax in axes.flat:
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.grid(alpha=0.2)
        ax.set_xlabel("steps relative to fixture release")
    fig.tight_layout()
    fig.savefig(output / "fixture_release_diagnostic.pdf")
    fig.savefig(output / "fixture_release_diagnostic.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Log and render a selected Phase 2.5 B-acquisition failure")
    parser.add_argument("--config", default="configs/phase2_5_second_grasp_calibration.yaml")
    parser.add_argument("--diagnostic-config", default="configs/phase2_5_failure_diagnostic.yaml")
    parser.add_argument("--candidate-index", type=int)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    cfg, source = load_phase2_5_config(ROOT / args.config)
    diagnostic = yaml.safe_load((ROOT / args.diagnostic_config).read_text(encoding="utf-8"))
    candidate_index = int(diagnostic["selected_candidate_index"]) if args.candidate_index is None else args.candidate_index
    cfg_hash = config_hash([
        source,
        ROOT / cfg.frozen_phase2_config,
        ROOT / "configs/hand_allegro.yaml",
        ROOT / "configs/scene_two_object.yaml",
        ROOT / "configs/task_sequential.yaml",
    ])
    search_dir = ROOT / cfg.output_dir / "b_only_search" / cfg_hash[:12]
    records = [
        json.loads(line) for line in (search_dir / "candidate_results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = next((row for row in records if int(row["candidate_index"]) == candidate_index), None)
    if selected is None:
        raise RuntimeError(f"candidate {candidate_index} is absent from {search_dir}")
    output = ROOT / cfg.output_dir / "diagnostics" / f"b_only_candidate_{candidate_index:04d}"
    output.mkdir(parents=True, exist_ok=True)
    video_path = None if args.no_video else output / "representative_failure.mp4"
    trajectory = sample_b_only_trajectory(cfg, candidate_index)
    summary, arrays = run_b_acquisition_trajectory(
        cfg,
        trajectory,
        collect_timeseries=True,
        render_video_path=video_path,
        render_stride=int(diagnostic["render_stride"]),
        video_fps=int(diagnostic["video_fps"]),
    )
    assert arrays is not None
    release = int(summary["fixture_release_timestep"])
    start = release - cfg.timing.diagnostic_pre_release_steps
    stop = release + cfg.timing.diagnostic_post_release_steps
    focused = {key: value[start:stop] for key, value in arrays.items()}
    np.savez_compressed(output / "complete_timeseries.npz", **arrays)
    np.savez_compressed(output / "fixture_release_window.npz", **focused)
    _write_flat_csv(output / "fixture_release_window.csv", focused)
    plt.style.use(ROOT / "configs" / "phase2_publication.mplstyle")
    _plot_release_window(focused, release, output)
    metadata = {
        **summary,
        "search_record": selected,
        "config_hash": cfg_hash,
        "diagnostic_window": {
            "start_timestep": start,
            "release_timestep": release,
            "stop_timestep_exclusive": stop,
            "pre_release_steps": cfg.timing.diagnostic_pre_release_steps,
            "post_release_steps": cfg.timing.diagnostic_post_release_steps,
            "logged_steps": len(focused["timestep"]),
        },
        "artifacts": {
            "complete_timeseries": str(output / "complete_timeseries.npz"),
            "release_window_npz": str(output / "fixture_release_window.npz"),
            "release_window_csv": str(output / "fixture_release_window.csv"),
            "plot_pdf": str(output / "fixture_release_diagnostic.pdf"),
            "plot_png": str(output / "fixture_release_diagnostic.png"),
            "video": None if video_path is None else str(video_path),
        },
    }
    (output / "diagnostic_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
