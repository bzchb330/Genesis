from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from itertools import combinations
import math
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation

from ..config import ROOT, ConfigBundle, DiagnosticProfile, load_configs
from ..control import JointImpedanceController, hand_state, resolve_hand_indices
from ..diagnostics.grasp_search import load_search_config
from ..diagnostics.multi_grasp import bundle_for_profile, load_grasp_profile
from ..diagnostics.scripted_grasp import _finger_object_contact_arrays, _joint_target, _object_addresses
from ..env.termination import Phase, failure_reason
from ..phase2_config import Phase2Config
from ..scene_builder import build_scene, randomize_objects
from ..sensing import extract_contacts, group_contacts_by_finger


FINGERS = ("index", "middle", "ring", "thumb")


def _run_candidate_probe(cfg: ConfigBundle, seed: int, short_hold_steps: int) -> dict:
    """Replay the exact diagnostic controller while retaining only screening data."""

    profile = cfg.diagnostic.profiles[cfg.diagnostic.active_profile]
    rng = np.random.default_rng(seed)
    model, data = build_scene(cfg)
    randomize_objects(model, data, cfg, rng)
    indices = resolve_hand_indices(model, cfg.hand)
    joint_limits = model.jnt_range[indices.joint_ids].copy()
    open_q = _joint_target(model, cfg, indices, profile.open_joint_fractions)
    closed_q = _joint_target(model, cfg, indices, profile.closed_joint_fractions)
    hold_q = _joint_target(model, cfg, indices, profile.hold_joint_fractions or profile.closed_joint_fractions)
    close_delays = np.asarray([
        0.0 if profile.actuator_close_delay_seconds is None else profile.actuator_close_delay_seconds.get(name, 0.0)
        for name in cfg.hand.actuator_names
    ])
    object_id, object_qadr, object_vadr = _object_addresses(model, cfg.diagnostic.object_name)
    fixture_pos = np.asarray(profile.object_fixture_pos, dtype=float).copy()
    fixture_pos[:2] += rng.uniform(-cfg.diagnostic.fixture_jitter_xy, cfg.diagnostic.fixture_jitter_xy, 2)
    fixture_quat = np.asarray(profile.object_fixture_quat, dtype=float)
    data.qpos[indices.qpos_addresses] = open_q
    data.qvel[indices.qvel_addresses] = 0.0
    data.qpos[object_qadr:object_qadr + 3] = fixture_pos
    data.qpos[object_qadr + 3:object_qadr + 7] = fixture_quat
    data.qvel[object_vadr:object_vadr + 6] = 0.0
    mujoco.mj_forward(model, data)
    controller = JointImpedanceController(cfg.task.impedance_stiffness, cfg.task.impedance_damping, cfg.task.torque_limit)
    fingers = list(cfg.hand.finger_geom_mapping)
    object_geom = f"{cfg.diagnostic.object_name}_geom"
    initial_penetration = 0.0
    hold_rows = []
    previous = open_q.copy()
    terminated_reason = None
    for stage, duration in profile.stage_durations_seconds.items():
        steps = max(1, round(duration / model.opt.timestep))
        start = previous.copy()
        target = open_q if stage == "open_pregrasp" else (closed_q if stage in {"close", "establish_contact"} else hold_q)
        phase = Phase(profile.episode_phase_by_stage[stage])
        for step in range(steps):
            if stage == "close":
                elapsed = (step + 1) * model.opt.timestep
                alpha = np.clip((elapsed - close_delays) / np.maximum(duration - close_delays, model.opt.timestep), 0.0, 1.0)
            else:
                alpha = (step + 1) / steps
            desired = np.clip((1 - alpha) * start + alpha * target, joint_limits[:, 0], joint_limits[:, 1])
            q, qvel = hand_state(data, indices)
            data.ctrl[indices.actuator_ids] = controller.torque(desired, q, qvel)
            if stage in profile.kinematic_fixture_stages:
                data.qpos[object_qadr:object_qadr + 3] = fixture_pos
                data.qpos[object_qadr + 3:object_qadr + 7] = fixture_quat
                data.qvel[object_vadr:object_vadr + 6] = 0.0
            mujoco.mj_step(model, data)
            if stage in {"open_pregrasp", "hold"}:
                contacts = extract_contacts(model, data)
                grouped = group_contacts_by_finger(contacts, cfg.hand.finger_geom_mapping)
                position = data.xpos[object_id].copy()
                counts, contact_positions, contact_normals, distances, forces = _finger_object_contact_arrays(
                    grouped, fingers, cfg.diagnostic.object_name, position,
                )
                penetration = float(np.max(np.where(counts > 0, np.maximum(0.0, -distances), 0.0)))
                if stage == "open_pregrasp":
                    initial_penetration = max(initial_penetration, penetration)
                else:
                    hold_rows.append({
                        "object_position": position,
                        "object_orientation": data.xquat[object_id].copy(),
                        "finger_object_contact_count": counts,
                        "finger_object_contact_position_world": contact_positions,
                        "finger_object_contact_normal_world": contact_normals,
                        "finger_object_contact_distance_m": distances,
                        "finger_object_normal_force_raw": forces,
                        "object_table_contact": int(any({row.geom1_name, row.geom2_name} == {object_geom, "table"} for row in contacts)),
                        "joint_positions": hand_state(data, indices)[0],
                        "actuator_controls": data.ctrl[indices.actuator_ids].copy(),
                        "numerical": int(all(np.all(np.isfinite(value)) for value in (data.qpos, data.qvel, data.ctrl))),
                    })
            terminated_reason = failure_reason(model, data, cfg, phase)
            if terminated_reason:
                break
        previous = target.copy()
        if terminated_reason:
            break
    selected = hold_rows[-short_hold_steps:]
    arrays = {
        key: np.asarray([row[key] for row in selected])
        for key in selected[0]
    } if selected else {}
    return {"arrays": arrays, "hold_steps": len(selected), "initial_penetration_m": initial_penetration, "termination_reason": terminated_reason}


@lru_cache(maxsize=1)
def _base_joint_ranges() -> np.ndarray:
    from ..scene_builder import build_scene
    from ..control import resolve_hand_indices
    cfg = load_configs()
    model, _ = build_scene(cfg)
    indices = resolve_hand_indices(model, cfg.hand)
    return model.jnt_range[indices.joint_ids].copy()


def ferrari_canny_epsilon(
    contact_positions_m: np.ndarray,
    inward_normals: np.ndarray,
    object_center_m: np.ndarray,
    friction_coefficient: float,
    friction_cone_edges: int,
    characteristic_length_m: float,
    numerical_tolerance: float,
) -> float:
    """Ferrari-Canny epsilon from the convex hull of unit primitive wrenches.

    Friction cones are polygonalized into ``friction_cone_edges`` unit force
    directions. Wrenches are ``[force, torque/characteristic_length]`` so the
    hull coordinates share force units. Epsilon is the radius of the largest
    origin-centered Euclidean ball inside that hull, or zero if the origin is
    not strictly interior above numerical tolerance.

    Reference: C. Ferrari and J. Canny, "Planning Optimal Grasps," IEEE ICRA,
    1992, pp. 2290-2295, doi:10.1109/ROBOT.1992.219918.
    """

    if len(contact_positions_m) < 2 or characteristic_length_m <= 0:
        return 0.0
    wrenches = []
    for point, normal in zip(contact_positions_m, inward_normals):
        normal = np.asarray(normal, dtype=float)
        norm = np.linalg.norm(normal)
        if norm <= numerical_tolerance:
            continue
        normal = normal / norm
        reference = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        tangent_1 = np.cross(normal, reference)
        tangent_1 /= np.linalg.norm(tangent_1)
        tangent_2 = np.cross(normal, tangent_1)
        for edge in range(friction_cone_edges):
            angle = 2.0 * math.pi * edge / friction_cone_edges
            force = normal + friction_coefficient * (math.cos(angle) * tangent_1 + math.sin(angle) * tangent_2)
            force /= np.linalg.norm(force)
            torque = np.cross(np.asarray(point) - object_center_m, force) / characteristic_length_m
            wrenches.append(np.r_[force, torque])
    points = np.asarray(wrenches)
    if len(points) < 7 or np.linalg.matrix_rank(points - points.mean(axis=0), tol=numerical_tolerance) < 6:
        return 0.0
    try:
        hull = ConvexHull(points)
    except QhullError:
        return 0.0
    offsets = hull.equations[:, -1]
    normals = hull.equations[:, :-1]
    if np.max(offsets) >= -numerical_tolerance:
        return 0.0
    return float(np.min(-offsets / np.linalg.norm(normals, axis=1)))


def deterministic_subset(attempt_index: int, subset_sizes: list[int]) -> tuple[str, ...]:
    size = subset_sizes[attempt_index % len(subset_sizes)]
    choices = list(combinations(FINGERS, size))
    cycle = attempt_index // len(subset_sizes)
    return choices[cycle % len(choices)]


def _sampled_profile(
    cfg: ConfigBundle,
    phase2: Phase2Config,
    proposal: DiagnosticProfile,
    subset: tuple[str, ...],
    rng: np.random.Generator,
    anchor: bool,
) -> tuple[DiagnosticProfile, dict]:
    groups = load_search_config()["finger_groups"]
    model_ranges_cfg = cfg.hand.actuator_names
    # Fractions are converted to radians by run_scripted_grasp. Convert the PI
    # +/- radian perturbation to a per-joint fraction using compiled ranges.
    ranges = _base_joint_ranges()
    closed = dict(proposal.closed_joint_fractions)
    base_hold = proposal.hold_joint_fractions or proposal.closed_joint_fractions
    hold = dict(base_hold)
    active_names = {name for finger in subset for name in groups[finger]}
    for joint_index, name in enumerate(model_ranges_cfg):
        if name in active_names:
            span = ranges[joint_index, 1] - ranges[joint_index, 0]
            joint_width = phase2.dataset.anchor_active_joint_perturbation_rad if anchor else phase2.dataset.active_joint_perturbation_rad
            delta_fraction = rng.uniform(-joint_width, joint_width) / span
            closed[name] = float(np.clip(closed[name] + delta_fraction, 0.0, 1.0))
            hold[name] = float(np.clip(hold[name] + delta_fraction, 0.0, 1.0))
        else:
            closed[name] = proposal.open_joint_fractions[name]
            hold[name] = proposal.open_joint_fractions[name]
    fixture_pos = np.asarray(proposal.object_fixture_pos, dtype=float)
    if anchor:
        jitter = phase2.dataset.anchor_object_jitter_half_width_m
        fixture_pos += [rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter), 0.0]
    else:
        fixture_pos += [rng.uniform(*phase2.dataset.object_a_jitter_x_m), rng.uniform(*phase2.dataset.object_a_jitter_y_m), 0.0]
    object_yaw = rng.uniform(*phase2.dataset.object_a_yaw_rad)
    fixture_quat = Rotation.from_euler("z", object_yaw).as_quat(scalar_first=True).tolist()
    durations = dict(proposal.stage_durations_seconds)
    durations["hold"] = phase2.dataset.short_hold_steps * cfg.scene.timestep
    profile = replace(
        proposal,
        object_fixture_pos=fixture_pos.tolist(),
        object_fixture_quat=fixture_quat,
        stage_durations_seconds=durations,
        closed_joint_fractions=closed,
        hold_joint_fractions=hold,
    )
    return profile, {"object_yaw_rad": object_yaw, "closed_joint_fractions": closed, "hold_joint_fractions": hold}


def sample_candidate(phase2: Phase2Config, attempt_index: int) -> tuple[ConfigBundle, DiagnosticProfile, dict]:
    base = load_configs()
    rng = np.random.default_rng(np.random.SeedSequence([phase2.dataset.seed, attempt_index]))
    proposal_index = (attempt_index // len(phase2.dataset.finger_subset_sizes)) % len(phase2.dataset.proposal_profile_paths)
    proposal_path = ROOT / phase2.dataset.proposal_profile_paths[proposal_index]
    _, proposal = load_grasp_profile(proposal_path)
    subset = deterministic_subset(attempt_index, phase2.dataset.finger_subset_sizes)
    anchor = (
        phase2.dataset.proposal_profile_paths[proposal_index] == phase2.dataset.anchor_profile_path
        and list(subset) in phase2.dataset.anchor_commanded_subsets
    )
    translation_bounds = (
        {axis: [-phase2.dataset.anchor_palm_translation_half_width_m, phase2.dataset.anchor_palm_translation_half_width_m] for axis in "xyz"}
        if anchor else phase2.dataset.palm_translation_bounds_m
    )
    translation = np.array([
        rng.uniform(*translation_bounds[axis]) for axis in "xyz"
    ])
    orientation_bounds = (
        {axis: [-phase2.dataset.anchor_palm_orientation_half_width_deg, phase2.dataset.anchor_palm_orientation_half_width_deg] for axis in ("roll", "pitch", "yaw")}
        if anchor else phase2.dataset.palm_orientation_bounds_deg
    )
    angles_deg = np.array([
        rng.uniform(*orientation_bounds[axis]) for axis in ("roll", "pitch", "yaw")
    ])
    baseline_rotation = Rotation.from_quat(base.hand.mount_quat, scalar_first=True)
    perturbation = Rotation.from_euler("xyz", np.deg2rad(angles_deg))
    mount_quat = (baseline_rotation * perturbation).as_quat(scalar_first=True)
    hand = replace(
        base.hand,
        mount_pos=(np.asarray(base.hand.mount_pos) + translation).tolist(),
        mount_quat=mount_quat.tolist(),
    )
    cfg = replace(base, hand=hand)
    profile, sampled = _sampled_profile(cfg, phase2, proposal, subset, rng, anchor)
    metadata = {
        "attempt_index": attempt_index,
        "generation_seed": phase2.dataset.seed,
        "proposal_profile_path": str(proposal_path.relative_to(ROOT)).replace("\\", "/"),
        "commanded_finger_subset": list(subset),
        "sampling_mode": "validated_proposal_anchor" if anchor else "full_PI_range",
        "palm_translation_perturbation_m": translation.tolist(),
        "palm_orientation_perturbation_deg": angles_deg.tolist(),
        "initial_palm_position_m": hand.mount_pos,
        "initial_palm_quaternion": hand.mount_quat,
        **sampled,
    }
    return bundle_for_profile(cfg, "phase2_dataset_candidate", profile), profile, metadata


def evaluate_candidate(phase2: Phase2Config, attempt_index: int) -> dict:
    cfg, profile, metadata = sample_candidate(phase2, attempt_index)
    probe = _run_candidate_probe(cfg, phase2.dataset.seed + attempt_index, phase2.dataset.short_hold_steps)
    arrays = probe["arrays"]
    if probe["hold_steps"] < phase2.dataset.short_hold_steps:
        return {**metadata, "accepted": False, "rejection_reason": "short_hold_incomplete"}
    counts = arrays["finger_object_contact_count"]
    distances = arrays["finger_object_contact_distance_m"]
    penetration = float(np.max(np.where(counts > 0, np.maximum(0.0, -distances), 0.0)))
    positions = arrays["object_position"]
    displacement = positions - positions[0]
    translation = float(np.max(np.linalg.norm(displacement, axis=1)))
    quaternions = arrays["object_orientation"]
    rotation = float(np.max(2.0 * np.arccos(np.clip(np.abs(quaternions @ quaternions[0]), 0.0, 1.0))))
    table = bool(np.any(arrays["object_table_contact"] > 0))
    complete_loss = bool(np.any(np.sum(counts, axis=1) == 0))
    initial_penetration = probe["initial_penetration_m"]
    final = -1
    active = arrays["finger_object_contact_count"][final] > 0
    contact_positions = arrays["finger_object_contact_position_world"][final][active]
    contact_normals = arrays["finger_object_contact_normal_world"][final][active]
    object_cfg = next(obj for obj in cfg.scene.objects if obj.name == cfg.diagnostic.object_name)
    characteristic_length = float(np.linalg.norm(object_cfg.size))
    epsilon = ferrari_canny_epsilon(
        contact_positions,
        contact_normals,
        arrays["object_position"][final],
        object_cfg.friction[0],
        phase2.dataset.friction_cone_edges,
        characteristic_length,
        phase2.dataset.convex_hull_tolerance,
    )
    mean_forces = np.mean(arrays["finger_object_normal_force_raw"], axis=0)
    occupied = mean_forces > phase2.resources.occupied_finger_normal_force_threshold_N
    checks = {
        "initial_penetration": initial_penetration <= phase2.dataset.maximum_penetration_m,
        "hold_penetration": penetration <= phase2.dataset.maximum_penetration_m,
        "translation": translation <= phase2.dataset.maximum_translation_drift_m,
        "orientation": rotation <= phase2.dataset.maximum_orientation_drift_rad,
        "table": phase2.dataset.allow_table_recontact or not table,
        "contact_loss": phase2.dataset.allow_complete_contact_loss or not complete_loss,
        "force_closure": epsilon > phase2.dataset.convex_hull_tolerance,
        "physical_closure": bool(np.any(active)),
        "numerical": bool(np.all(arrays["numerical"] > 0)),
    }
    rejection = next((name for name, passed in checks.items() if not passed), None)
    return {
        **metadata,
        "accepted": rejection is None,
        "rejection_reason": rejection,
        "checks": checks,
        "stability": {
            "maximum_penetration_m": penetration,
            "initial_penetration_m": initial_penetration,
            "maximum_translation_drift_m": translation,
            "maximum_orientation_drift_rad": rotation,
            "table_recontact": table,
            "complete_contact_loss": complete_loss,
        },
        "ferrari_canny_epsilon": epsilon,
        "characteristic_length_m": characteristic_length,
        "final_joint_configuration_rad": arrays["joint_positions"][final].tolist(),
        "initial_object_position_m": profile.object_fixture_pos,
        "initial_object_quaternion": profile.object_fixture_quat,
        "final_object_position_m": arrays["object_position"][final].tolist(),
        "final_object_quaternion": arrays["object_orientation"][final].tolist(),
        "occupied_finger_count": int(np.sum(occupied)),
        "occupied_finger_mask": occupied.tolist(),
        "mean_per_finger_normal_force_N": mean_forces.tolist(),
        "final_per_finger_contact_positions_m": arrays["finger_object_contact_position_world"][final].tolist(),
        "final_per_finger_contact_normals": arrays["finger_object_contact_normal_world"][final].tolist(),
        "final_per_finger_contact_counts": arrays["finger_object_contact_count"][final].tolist(),
    }
