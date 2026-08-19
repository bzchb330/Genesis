#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from seqgrasp.config import ROOT, load_configs
from seqgrasp.control import resolve_hand_indices
from seqgrasp.diagnostics.multi_grasp import load_grasp_profile
from seqgrasp.diagnostics.scripted_grasp import _joint_target
from seqgrasp.experiments.phase2_6_workspace import world_to_frame
from seqgrasp.experiments.resource_components import FINGER_ORDER, reconstruct_grasp
from seqgrasp.phase2_6_config import load_phase2_6_config
from seqgrasp.scene_builder import build_scene


def _hand_self_collision(model, data) -> bool:
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        names = []
        for geom_id in (contact.geom1, contact.geom2):
            body_id = int(model.geom_bodyid[geom_id])
            names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "")
        if all(name and not name.startswith("object_") and name != "world" for name in names):
            return True
    return False


def _box_edges(low: np.ndarray, high: np.ndarray):
    corners = np.asarray([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
    edges = []
    for left in range(len(corners)):
        for right in range(left + 1, len(corners)):
            if np.sum(corners[left] != corners[right]) == 1:
                edges.append((corners[left], corners[right]))
    return edges


def _plot(summary, clouds, old_low, old_high, output: Path, plot_count: int) -> None:
    colors = {"index": "#0072B2", "middle": "#E69F00", "ring": "#009E73", "thumb": "#CC79A7"}
    pairs = ((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ"))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (a, b, title) in zip(axes, pairs):
        for finger, points in clouds.items():
            stride = max(1, len(points) // plot_count)
            sample = points[::stride][:plot_count]
            ax.scatter(sample[:, a], sample[:, b], s=0.3, alpha=0.15, color=colors[finger], label=finger)
        for left, right in _box_edges(old_low, old_high):
            ax.plot([left[a], right[a]], [left[b], right[b]], color="black", linewidth=1.0)
        palm = np.asarray(summary["palm_origin_world_m"])
        bases = np.asarray(list(summary["finger_base_world_m"].values()))
        ax.scatter([palm[a]], [palm[b]], marker="s", color="black", label="palm")
        ax.scatter(bases[:, a], bases[:, b], marker="x", color="red", label="bases")
        ax.set(title=title, xlabel="xyz"[a] + " [m]", ylabel="xyz"[b] + " [m]", aspect="equal")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=7, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(output / "hand_workspace_orthographic.pdf")
    fig.savefig(output / "hand_workspace_orthographic.png", dpi=180)
    plt.close(fig)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    for finger, points in clouds.items():
        stride = max(1, len(points) // plot_count)
        sample = points[::stride][:plot_count]
        ax.scatter(*sample.T, s=0.25, alpha=0.12, color=colors[finger], label=finger)
    for left, right in _box_edges(old_low, old_high):
        ax.plot(*np.stack([left, right]).T, color="black", linewidth=1.0)
    ax.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "hand_workspace_3d.pdf")
    fig.savefig(output / "hand_workspace_3d.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Resumable dense fixed-base Allegro fingertip workspace map")
    parser.add_argument("--config", default="configs/phase2_6_b_graspable_workspace.yaml")
    args = parser.parse_args()
    phase26, _ = load_phase2_6_config(ROOT / args.config)
    cfg = load_configs()
    model, data = build_scene(cfg)
    indices = resolve_hand_indices(model, cfg.hand)
    _, profile = load_grasp_profile(ROOT / "configs" / "grasps" / "resource_grasp_A_02.yaml")
    open_q = _joint_target(model, cfg, indices, profile.open_joint_fractions)
    data.qpos[indices.qpos_addresses] = open_q
    for object_name, position in (("object_a", (-0.3, -0.3, 0.1)), ("object_b", (-0.3, 0.3, 0.1))):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{object_name}_free")
        data.qpos[model.jnt_qposadr[joint]:model.jnt_qposadr[joint] + 3] = position
    mujoco.mj_forward(model, data)
    palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.hand.palm_body)
    palm_origin = data.xpos[palm_id].copy()
    palm_rotation = data.xmat[palm_id].reshape(3, 3).copy()
    output = ROOT / phase26.output_dir / "workspace"
    batches = output / "batches"
    batches.mkdir(parents=True, exist_ok=True)
    ranges = model.jnt_range[indices.joint_ids]
    clouds_world = {}
    cloud_records = {}
    finger_radii = {}
    statistics = {}
    for finger_index, finger in enumerate(FINGER_ORDER):
        tip_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, cfg.hand.finger_geom_mapping[finger][0])
        finger_radii[finger] = float(model.geom_size[tip_geom, 0])
        valid_positions, valid_q, valid_margin = [], [], []
        proposed_total = rejected_total = batch_index = 0
        while sum(len(value) for value in valid_positions) < phase26.workspace.samples_per_finger:
            batch_path = batches / f"{finger}_{batch_index:04d}.npz"
            if batch_path.exists():
                loaded = np.load(batch_path)
                positions = loaded["positions_world_m"]
                joints = loaded["joint_rad"]
                margins = loaded["joint_margin_rad"]
                proposed = int(loaded["proposed_count"])
            else:
                rng = np.random.default_rng(np.random.SeedSequence([phase26.seeds.workspace, finger_index, batch_index]))
                sl = slice(4 * finger_index, 4 * finger_index + 4)
                samples = rng.uniform(ranges[sl, 0], ranges[sl, 1], size=(phase26.workspace.batch_size, 4))
                positions_list, joints_list, margins_list = [], [], []
                for joint_sample in samples:
                    data.qpos[indices.qpos_addresses] = open_q
                    data.qpos[indices.qpos_addresses[sl]] = joint_sample
                    mujoco.mj_forward(model, data)
                    if _hand_self_collision(model, data):
                        continue
                    positions_list.append(data.geom_xpos[tip_geom].copy())
                    joints_list.append(joint_sample.copy())
                    margins_list.append(float(np.min(np.minimum(joint_sample - ranges[sl, 0], ranges[sl, 1] - joint_sample))))
                positions = np.asarray(positions_list, dtype=float).reshape(-1, 3)
                joints = np.asarray(joints_list, dtype=float).reshape(-1, 4)
                margins = np.asarray(margins_list, dtype=float)
                proposed = len(samples)
                np.savez_compressed(batch_path, positions_world_m=positions, joint_rad=joints, joint_margin_rad=margins, proposed_count=proposed)
            valid_positions.append(positions)
            valid_q.append(joints)
            valid_margin.append(margins)
            proposed_total += proposed
            rejected_total += proposed - len(positions)
            batch_index += 1
            if batch_index % 10 == 0:
                print(f"{finger}: {sum(len(value) for value in valid_positions)}/{phase26.workspace.samples_per_finger} valid", flush=True)
        positions = np.concatenate(valid_positions)[:phase26.workspace.samples_per_finger]
        joints = np.concatenate(valid_q)[:phase26.workspace.samples_per_finger]
        margins = np.concatenate(valid_margin)[:phase26.workspace.samples_per_finger]
        palm_points = world_to_frame(positions, palm_origin, palm_rotation)
        aggregate = output / f"{finger}_workspace.npz"
        np.savez_compressed(aggregate, positions_world_m=positions, positions_palm_m=palm_points, joint_rad=joints, joint_margin_rad=margins)
        clouds_world[finger] = positions
        cloud_records[finger] = {"world": positions, "palm": palm_points}
        statistics[finger] = {
            "valid_samples": len(positions), "proposed_samples": proposed_total,
            "rejected_self_collision_samples": rejected_total,
            "world_bounds_m": [positions.min(axis=0).tolist(), positions.max(axis=0).tolist()],
            "palm_bounds_m": [palm_points.min(axis=0).tolist(), palm_points.max(axis=0).tolist()],
            "fingertip_collision_radius_m": finger_radii[finger],
        }
    data.qpos[indices.qpos_addresses] = open_q
    mujoco.mj_forward(model, data)
    base_world = {}
    open_tips = {}
    nominal_tips = {}
    for finger_index, finger in enumerate(FINGER_ORDER):
        joint_id = int(indices.joint_ids[4 * finger_index])
        base_world[finger] = data.xanchor[joint_id].tolist()
        tip_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.hand.fingertip_bodies[finger_index])
        open_tips[finger] = data.xpos[tip_body].tolist()
    data.qpos[indices.qpos_addresses] = model.qpos0[indices.qpos_addresses]
    mujoco.mj_forward(model, data)
    for finger_index, finger in enumerate(FINGER_ORDER):
        tip_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.hand.fingertip_bodies[finger_index])
        nominal_tips[finger] = data.xpos[tip_body].tolist()
    accepted_paths = sorted((ROOT / "outputs" / "phase2" / "grasp_dataset").glob("*/accepted_grasps.jsonl"), key=lambda path: path.stat().st_size)
    accepted = [] if not accepted_paths else [json.loads(line) for line in accepted_paths[-1].read_text(encoding="utf-8").splitlines() if line]
    A_tip_positions = {finger: [] for finger in FINGER_ORDER}
    for record in accepted:
        grasp_cfg, grasp_model, grasp_data, _ = reconstruct_grasp(record)
        for finger_index, finger in enumerate(FINGER_ORDER):
            body = mujoco.mj_name2id(grasp_model, mujoco.mjtObj.mjOBJ_BODY, grasp_cfg.hand.fingertip_bodies[finger_index])
            A_tip_positions[finger].append(grasp_data.xpos[body].tolist())
    summary = {
        "palm_origin_world_m": palm_origin.tolist(), "palm_rotation_world": palm_rotation.tolist(),
        "finger_base_world_m": base_world, "nominal_fingertip_world_m": nominal_tips,
        "open_fingertip_world_m": open_tips, "validated_A_grasp_count": len(accepted),
        "validated_A_grasp_fingertips_world_m": A_tip_positions, "fingers": statistics,
    }
    (output / "hand_workspace_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    old_low = np.asarray([phase26.object_B.old_center_x_bounds_m[0], phase26.object_B.old_center_y_bounds_m[0], phase26.object_B.old_center_z_bounds_m[0]])
    old_high = np.asarray([phase26.object_B.old_center_x_bounds_m[1], phase26.object_B.old_center_y_bounds_m[1], phase26.object_B.old_center_z_bounds_m[1]])
    figure_dir = ROOT / "docs" / "figures" / "phase2_6"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use(ROOT / "configs" / "phase2_publication.mplstyle")
    _plot(summary, clouds_world, old_low, old_high, figure_dir, phase26.workspace.plot_samples_per_finger)
    print(json.dumps({"output": str(output), "valid_samples": {key: value["valid_samples"] for key, value in statistics.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
