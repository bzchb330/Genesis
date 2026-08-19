from __future__ import annotations

from dataclasses import replace

import mujoco
import numpy as np

from ..config import ConfigBundle
from ..control import JointImpedanceController, hand_state, resolve_hand_indices
from ..phase2_config import Phase2Config
from ..phase2s_config import Phase2SConfig
from ..scene_builder import build_scene
from .palmar_grasp_sampling import (
    evaluate_existing_fingertip_state,
    evaluate_palmar_candidate,
    evaluate_palmar_proposal,
    sample_palmar_candidate,
)
from .phase2r import GraspStateType, classify_grasp_state, measure_stable_hold


def phase2s_scene_config(phase2s: Phase2SConfig) -> ConfigBundle:
    from ..config import load_configs

    return load_configs(scene_filename=phase2s.scene_filename)


def evaluate_phase2r_fingertip_seed(
    phase2s: Phase2SConfig, phase2: Phase2Config, source: dict,
) -> dict:
    cfg = phase2s_scene_config(phase2s)
    result = evaluate_existing_fingertip_state(phase2s, phase2, source, cfg)
    result.update({
        "proposal_source_experiment": "Phase2R_large_object_candidate",
        "revalidated_with_half_scale_geometry": True,
        "grasp_state_id": f"phase2S_fingertip_replay_{source['grasp_id']}",
    })
    return result


def evaluate_phase2s_palmar_candidate(
    phase2s: Phase2SConfig, phase2: Phase2Config, attempt_index: int,
) -> dict:
    cfg = phase2s_scene_config(phase2s)
    # Four fifths of proposals perturb one of nine physically accepted pilot
    # basins; every fifth remains an independent broad/focused sampler draw.
    pilot_basins = (15, 269, 360, 373, 375, 547, 681, 694, 743)
    basin_focused = attempt_index % 5 != 0
    if basin_focused:
        basin_index = pilot_basins[(attempt_index // 5) % len(pilot_basins)]
        proposal = sample_palmar_candidate(phase2s, basin_index, cfg)
        rng = np.random.default_rng(np.random.SeedSequence([phase2s.state.seed, 2, attempt_index]))
        compiled = np.asarray(proposal["initial_object_COM_palm_compiled_m"], dtype=float)
        compiled += rng.uniform(-0.002, 0.002, 3)
        model, data = build_scene(cfg)
        mujoco.mj_forward(model, data)
        palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.hand.palm_body)
        palm_rotation = data.xmat[palm].reshape(3, 3)
        world = data.xpos[palm] + compiled @ palm_rotation.T
        target = np.asarray(proposal["retaining_joint_target_rad"], dtype=float)
        limits = model.jnt_range[resolve_hand_indices(model, cfg.hand).joint_ids]
        target = np.clip(target + rng.uniform(-0.018, 0.018, len(target)), limits[:, 0], limits[:, 1])
        proposal.update({
            "attempt_index": int(attempt_index),
            "sampling_mode": "focused_half_scale_palmar_pilot_basin",
            "pilot_basin_attempt_index": int(basin_index),
            "initial_object_COM_palm_compiled_m": compiled.tolist(),
            "initial_object_position_m": world.tolist(),
            "retaining_joint_target_rad": target.tolist(),
        })
        result = evaluate_palmar_proposal(phase2s, phase2, proposal, cfg)
    else:
        result = evaluate_palmar_candidate(phase2s, phase2, attempt_index, cfg)
    result.update({
        "proposal_source_experiment": "Phase2S_half_scale_palmar_search",
        "revalidated_with_half_scale_geometry": True,
        "grasp_state_id": f"phase2S_palmar_attempt_{attempt_index:05d}",
    })
    return result


def evaluate_supplemental_fingertip_candidate(
    phase2s: Phase2SConfig,
    phase2: Phase2Config,
    source: dict,
    variant_index: int,
) -> dict:
    """Tighten a large-object proposal around half-scale A, then validate unsupported."""

    base_cfg = phase2s_scene_config(phase2s)
    hand = replace(
        base_cfg.hand,
        mount_pos=list(source["initial_palm_position_m"]),
        mount_quat=list(source["initial_palm_quaternion"]),
    )
    cfg = replace(base_cfg, hand=hand)
    model, data = build_scene(cfg)
    indices = resolve_hand_indices(model, cfg.hand)
    rng = np.random.default_rng(np.random.SeedSequence([phase2s.state.seed, 1, variant_index]))
    focused = variant_index % 5 != 0
    source_q = np.asarray(source["final_joint_configuration_rad"], dtype=float)
    ranges = model.jnt_range[indices.joint_ids]
    closure_fraction = float(rng.uniform(0.275, 0.335) if focused else rng.uniform(0.04, 0.34))
    target_q = source_q + closure_fraction * (ranges[:, 1] - source_q)
    active_fingers = set(source.get("commanded_finger_subset", ("index", "middle", "ring", "thumb")))
    finger_order = ("index", "middle", "ring", "thumb")
    active_mask = np.repeat([finger in active_fingers for finger in finger_order], 4)
    target_q[~active_mask] = source_q[~active_mask]
    target_q += rng.uniform(-0.025, 0.025, len(target_q)) * active_mask
    target_q = np.clip(target_q, ranges[:, 0], ranges[:, 1])
    object_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_a_free")
    qadr, vadr = int(model.jnt_qposadr[object_joint]), int(model.jnt_dofadr[object_joint])
    if focused:
        successful_basin_offset = np.asarray([-0.00192, -0.00285, 0.00059])
        fixture_position = np.asarray(source["final_object_position_m"], dtype=float) + successful_basin_offset + rng.uniform(-0.0015, 0.0015, 3)
    else:
        fixture_position = np.asarray(source["final_object_position_m"], dtype=float) + rng.uniform(-0.006, 0.006, 3)
    fixture_quaternion = np.asarray(source["final_object_quaternion"], dtype=float)
    data.qpos[indices.qpos_addresses] = source_q
    data.qvel[indices.qvel_addresses] = 0.0
    data.qpos[qadr:qadr + 3] = fixture_position
    data.qpos[qadr + 3:qadr + 7] = fixture_quaternion
    data.qvel[vadr:vadr + 6] = 0.0
    mujoco.mj_forward(model, data)
    controller = JointImpedanceController(
        cfg.task.impedance_stiffness, cfg.task.impedance_damping, cfg.task.torque_limit,
    )
    close_steps = phase2s.state.fixture_close_steps
    for step in range(close_steps + phase2s.state.fixture_contact_steps):
        alpha = min(1.0, (step + 1) / close_steps)
        desired = (1.0 - alpha) * source_q + alpha * target_q
        q, qvel = hand_state(data, indices)
        data.ctrl[indices.actuator_ids] = controller.torque(desired, q, qvel)
        data.qpos[qadr:qadr + 3] = fixture_position
        data.qpos[qadr + 3:qadr + 7] = fixture_quaternion
        data.qvel[vadr:vadr + 6] = 0.0
        mujoco.mj_step(model, data)
    data.qpos[qadr:qadr + 3] = fixture_position
    data.qpos[qadr + 3:qadr + 7] = fixture_quaternion
    data.qvel[vadr:vadr + 6] = 0.0
    mujoco.mj_forward(model, data)
    measured = measure_stable_hold(
        cfg, model, data, indices, target_q, phase2s.state.stable_hold_steps,
        phase2.resources, phase2.dataset.friction_cone_edges, phase2.dataset.convex_hull_tolerance,
    )
    record = {
        **source,
        **measured,
        "source_grasp_id": source["grasp_id"],
        "grasp_state_id": f"phase2S_fingertip_supplement_{variant_index:06d}",
        "initial_palm_position_m": list(cfg.hand.mount_pos),
        "initial_palm_quaternion": list(cfg.hand.mount_quat),
        "initial_object_position_m": fixture_position.tolist(),
        "initial_object_quaternion": fixture_quaternion.tolist(),
        "retaining_joint_target_rad": target_q.tolist(),
        "fixture_method": "temporary_free_joint_pose_reset_during_half_scale_acquisition_search",
        "fixture_release_timestep": close_steps + phase2s.state.fixture_contact_steps,
        "proposal_source_experiment": "Phase2R_large_object_joint_pose_tightened_and_revalidated",
        "revalidated_with_half_scale_geometry": True,
        "supplemental_variant_index": int(variant_index),
        "supplemental_closure_fraction": closure_fraction,
        "sampling_mode": "focused_half_scale_fingertip_basin" if focused else "broad_half_scale_fingertip_search",
    }
    return classify_grasp_state(
        record, GraspStateType.FINGERTIP, phase2s.state,
        phase2.dataset.convex_hull_tolerance,
    )
