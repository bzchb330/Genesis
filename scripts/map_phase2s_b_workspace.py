#!/usr/bin/env python
from __future__ import annotations

import json
import math

import mujoco
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc
import yaml

from seqgrasp.config import ROOT, load_configs
from seqgrasp.control import resolve_hand_indices
from seqgrasp.diagnostics.multi_grasp import load_grasp_profile
from seqgrasp.diagnostics.scripted_grasp import _joint_target
from seqgrasp.experiments.grasp_sampling import ferrari_canny_epsilon
from seqgrasp.experiments.phase2_6_workspace import (
    accessible_surface_samples,
    contact_opposition_angle_deg,
    lexicographic_pose_key,
    pairwise_envelope_boxes,
)
from seqgrasp.experiments.resource_components import FINGER_ORDER
from seqgrasp.experiments.second_grasp import BPlacement, _set_b_pose
from seqgrasp.phase2_6_config import load_phase2_6_config
from seqgrasp.phase2_config import load_phase2_config
from seqgrasp.phase2s_config import load_phase2s_config
from seqgrasp.scene_builder import build_scene


def structured_centers(boxes, count, seed):
    names = sorted(boxes)
    per = math.ceil(count / len(names))
    rows = []
    for index, name in enumerate(names):
        low, high = boxes[name]
        unit = qmc.LatinHypercube(
            3, seed=np.random.default_rng(np.random.SeedSequence([seed, index]))
        ).random(per)
        rows.extend(qmc.scale(unit, low, high))
    rows.sort(key=lambda row: tuple(row))
    return np.asarray(rows[:count])


def select_diverse(records, count):
    promising = [
        row for row in records
        if row["valid_initial_geometry"]
        and row["accessible_finger_count"] >= 2
        and row["opposition_available"]
    ]
    promising.sort(key=lexicographic_pose_key, reverse=True)
    selected = []
    remaining = promising.copy()
    while remaining and len(selected) < count:
        seen = {tuple(row["accessible_fingers"]) for row in selected}
        unseen = [row for row in remaining if tuple(row["accessible_fingers"]) not in seen]
        pool = unseen or remaining
        if not selected:
            choice = pool[0]
        else:
            prior = np.asarray([row["position_m"] for row in selected])
            spans = np.maximum(
                np.ptp(np.asarray([row["position_m"] for row in promising]), axis=0), 1e-12
            )
            choice = max(
                pool,
                key=lambda row: float(
                    np.min(np.linalg.norm((np.asarray(row["position_m"]) - prior) / spans, axis=1))
                ),
            )
        selected.append(choice)
        remaining.remove(choice)
    return selected


def main() -> int:
    phase2s, _ = load_phase2s_config()
    phase26, _ = load_phase2_6_config()
    phase2, _ = load_phase2_config(ROOT / phase26.frozen_phase2_config)
    cfg = load_configs(scene_filename=phase2s.scene_filename)
    model, data = build_scene(cfg)
    indices = resolve_hand_indices(model, cfg.hand)
    _, profile = load_grasp_profile(ROOT / "configs" / "grasps" / "resource_grasp_A_02.yaml")
    data.qpos[indices.qpos_addresses] = _joint_target(model, cfg, indices, profile.open_joint_fractions)
    mujoco.mj_forward(model, data)

    # These clouds contain only hand kinematics/self-collision data. The hand is
    # unchanged in Phase 2S; all B-dependent envelopes below are regenerated.
    workspace_dir = ROOT / phase26.output_dir / "workspace"
    clouds, joints, margins, radii, trees = {}, {}, {}, {}, {}
    for finger in FINGER_ORDER:
        array = np.load(workspace_dir / f"{finger}_workspace.npz")
        clouds[finger] = array["positions_world_m"]
        joints[finger] = array["joint_rad"]
        margins[finger] = array["joint_margin_rad"]
        trees[finger] = cKDTree(clouds[finger])
        geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, cfg.hand.finger_geom_mapping[finger][0]
        )
        radii[finger] = float(model.geom_size[geom_id, 0])

    obj = next(item for item in cfg.scene.objects if item.name == "object_b")
    radius, half_height = obj.size[0], obj.size[1]
    table_top = cfg.scene.table_pos[2] + cfg.scene.table_size[2]
    boxes = pairwise_envelope_boxes(clouds, radii, radius, half_height)
    boxes = {
        name: (np.maximum(low, [-np.inf, -np.inf, table_top + half_height]), high)
        for name, (low, high) in boxes.items()
        if high[2] > table_top + half_height
    }
    centers = structured_centers(
        boxes, phase2s.second_grasp.workspace_candidate_poses, phase2s.second_grasp.workspace_seed
    )
    b_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_b_geom")
    palm_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.hand.palm_body)
    palm_geoms = [geom for geom in range(model.ngeom) if int(model.geom_bodyid[geom]) == palm_body]
    search_radius = max(
        math.sqrt(
            (radius + tip_radius + phase26.workspace.surface_access_tolerance_m) ** 2
            + (half_height + tip_radius + phase26.workspace.surface_access_tolerance_m) ** 2
        )
        for tip_radius in radii.values()
    )
    records = []
    for candidate_index, center in enumerate(centers):
        _set_b_pose(model, data, BPlacement(candidate_index, tuple(center), (1.0, 0.0, 0.0, 0.0), 0.0))
        minimum_distance = float("inf")
        valid = center[2] - half_height >= table_top
        for geom in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
            if geom == b_geom or name == "table":
                continue
            if model.geom_contype[geom] or model.geom_conaffinity[geom]:
                minimum_distance = min(
                    minimum_distance,
                    float(mujoco.mj_geomDistance(model, data, b_geom, geom, 1.0, None)),
                )
        valid &= minimum_distance >= -phase2.second_grasp.maximum_penetration_m
        contacts, normals, accessible, representative, joint_margins = [], [], [], {}, []
        predicted_penetration = 0.0
        for finger in FINGER_ORDER:
            nearby = trees[finger].query_ball_point(center, search_radius)
            if not nearby:
                continue
            points = clouds[finger][nearby]
            mask, surface_contacts, inward = accessible_surface_samples(
                points,
                center,
                radius,
                half_height,
                radii[finger],
                phase26.workspace.surface_access_tolerance_m,
            )
            if not np.any(mask):
                continue
            sources = np.asarray(nearby)[mask]
            distances = np.linalg.norm(points[mask] - surface_contacts, axis=1)
            local = int(np.argmin(np.abs(distances - radii[finger])))
            source_index = int(sources[local])
            accessible.append(finger)
            contacts.append(surface_contacts[local])
            normals.append(inward[local])
            representative[finger] = joints[finger][source_index].tolist()
            joint_margins.append(float(margins[finger][source_index]))
            predicted_penetration = max(
                predicted_penetration, max(0.0, radii[finger] - float(distances[local]))
            )
        angle = contact_opposition_angle_deg(np.asarray(normals))
        epsilon = ferrari_canny_epsilon(
            np.asarray(contacts),
            np.asarray(normals),
            center,
            obj.friction[0],
            phase2.dataset.friction_cone_edges,
            float(np.linalg.norm(obj.size)),
            phase2.dataset.convex_hull_tolerance,
        )
        palm_distance = min(
            (float(mujoco.mj_geomDistance(model, data, b_geom, geom, 1.0, None)) for geom in palm_geoms),
            default=float("inf"),
        )
        records.append({
            "candidate_index": candidate_index,
            "position_m": center.tolist(),
            "yaw_rad": 0.0,
            "valid_initial_geometry": bool(valid),
            "minimum_initial_distance_m": minimum_distance,
            "accessible_fingers": accessible,
            "accessible_finger_count": len(accessible),
            "representative_joint_rad": representative,
            "contact_positions_m": [value.tolist() for value in contacts],
            "inward_contact_normals": [value.tolist() for value in normals],
            "maximum_opposition_angle_deg": angle,
            "opposition_available": angle >= phase26.workspace.opposition_minimum_angle_deg,
            "ferrari_canny_epsilon": epsilon,
            "palm_support_available": abs(palm_distance) <= phase26.workspace.palm_support_tolerance_m,
            "predicted_penetration_m": predicted_penetration,
            "minimum_joint_margin_rad": min(joint_margins, default=0.0),
        })
        if (candidate_index + 1) % 1000 == 0:
            print(f"Phase 2S B-pose map: {candidate_index + 1}/{len(centers)}", flush=True)

    selected = select_diverse(records, 50)
    output = ROOT / phase2s.output_dir / "b_pose_graspability"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "candidate_poses.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    (output / "selected_poses.yaml").write_text(
        yaml.safe_dump({"selected_poses": selected}, sort_keys=False), encoding="utf-8"
    )
    summary = {
        "candidate_pose_count": len(records),
        "hand_workspace_source": str(workspace_dir.relative_to(ROOT)),
        "hand_workspace_reuse_reason": "hand geometry and kinematics are unchanged; B-dependent envelopes were regenerated",
        "half_scale_B_radius_m": radius,
        "half_scale_B_half_height_m": half_height,
        "derived_pairwise_envelope_boxes_m": {
            name: [low.tolist(), high.tolist()] for name, (low, high) in boxes.items()
        },
        "valid_initial_geometry_count": sum(row["valid_initial_geometry"] for row in records),
        "multi_finger_access_count": sum(row["accessible_finger_count"] >= 2 for row in records),
        "opposing_contact_count": sum(row["opposition_available"] for row in records),
        "positive_ferrari_canny_count": sum(
            row["ferrari_canny_epsilon"] > phase2.dataset.convex_hull_tolerance for row in records
        ),
        "selected_pose_count": len(selected),
        "selected_topologies": sorted({"+".join(row["accessible_fingers"]) for row in selected}),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if len(selected) == 50 else 2


if __name__ == "__main__":
    raise SystemExit(main())
