from __future__ import annotations

from dataclasses import replace
import math

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from ..config import ROOT, ConfigBundle, load_configs
from ..control import JointImpedanceController, hand_state, resolve_hand_indices
from ..diagnostics.grasp_search import load_search_config
from ..diagnostics.multi_grasp import load_grasp_profile
from ..diagnostics.scripted_grasp import _joint_target
from ..phase2_config import Phase2Config
from ..phase2r_config import Phase2RConfig
from ..scene_builder import build_scene
from .phase2r import GraspStateType, classify_grasp_state, measure_stable_hold
from .resource_components import reconstruct_grasp


def _object_addresses(model: mujoco.MjModel) -> tuple[int, int]:
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_a_free")
    return int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])


def _palm_box_front_surface_x(model: mujoco.MjModel, data: mujoco.MjData, cfg: ConfigBundle) -> float:
    palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.hand.palm_body)
    palm_rotation = data.xmat[palm].reshape(3, 3)
    candidates = []
    for geom in range(model.ngeom):
        if int(model.geom_bodyid[geom]) != palm or int(model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            continue
        center = (data.geom_xpos[geom] - data.xpos[palm]) @ palm_rotation
        rotation = palm_rotation.T @ data.geom_xmat[geom].reshape(3, 3)
        half_extent_x = float(np.sum(np.abs(rotation[0]) * model.geom_size[geom, :3]))
        candidates.append(float(center[0] + half_extent_x))
    if not candidates:
        raise RuntimeError("palmar initialization requires the existing palm collision box")
    return max(candidates)


def sample_palmar_candidate(phase2r: Phase2RConfig, attempt_index: int) -> dict:
    """Create one deterministic engineering proposal without evaluating outcomes."""

    cfg = load_configs()
    state_cfg = phase2r.state
    rng = np.random.default_rng(np.random.SeedSequence([state_cfg.seed, attempt_index]))
    focused = attempt_index % state_cfg.focused_candidate_stride != 0
    subsets = state_cfg.focused_retaining_finger_subsets if focused else state_cfg.retaining_finger_subsets
    profile_paths = state_cfg.focused_proposal_profile_paths if focused else state_cfg.proposal_profile_paths
    subset = tuple(subsets[attempt_index % len(subsets)])
    profile_index = (attempt_index // len(subsets)) % len(profile_paths)
    profile_path = ROOT / profile_paths[profile_index]
    _, profile = load_grasp_profile(profile_path)
    model, data = build_scene(cfg)
    indices = resolve_hand_indices(model, cfg.hand)
    mujoco.mj_forward(model, data)
    palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.hand.palm_body)
    palm_rotation = data.xmat[palm].reshape(3, 3)
    object_half_x = next(item for item in cfg.scene.objects if item.name == "object_a").size[0]
    compiled_position = np.asarray([
        _palm_box_front_surface_x(model, data, cfg) + object_half_x + rng.uniform(*state_cfg.palm_surface_offset_bounds_m),
        rng.uniform(*(state_cfg.focused_palm_tangent_y_bounds_m if focused else state_cfg.palm_tangent_y_bounds_m)),
        rng.uniform(*(state_cfg.focused_palm_tangent_z_bounds_m if focused else state_cfg.palm_tangent_z_bounds_m)),
    ])
    world_position = data.xpos[palm] + compiled_position @ palm_rotation.T
    palm_world_rotation = Rotation.from_matrix(palm_rotation)
    palm_relative_rotation = Rotation.from_euler("x", rng.uniform(-math.pi, math.pi))
    world_quaternion = (palm_world_rotation * palm_relative_rotation).as_quat(scalar_first=True)

    open_joint = _joint_target(model, cfg, indices, profile.open_joint_fractions)
    base_hold = _joint_target(model, cfg, indices, profile.hold_joint_fractions or profile.closed_joint_fractions)
    joint_limits = model.jnt_range[indices.joint_ids]
    groups = load_search_config()["finger_groups"]
    active = {name for finger in subset for name in groups[finger]}
    target_joint = open_joint.copy()
    closure_scale = rng.uniform(*(state_cfg.focused_closure_scale_bounds if focused else state_cfg.active_closure_scale_bounds))
    joint_perturbation = state_cfg.focused_joint_perturbation_rad if focused else state_cfg.active_joint_perturbation_rad
    for index, name in enumerate(cfg.hand.actuator_names):
        if name in active:
            target_joint[index] = np.clip(
                open_joint[index] + closure_scale * (base_hold[index] - open_joint[index])
                + rng.uniform(-joint_perturbation, joint_perturbation),
                joint_limits[index, 0], joint_limits[index, 1],
            )
    return {
        "attempt_index": int(attempt_index),
        "generation_seed": int(state_cfg.seed),
        "sampling_mode": "focused_endpoint_basin" if focused else "broad_palm_region",
        "retaining_finger_subset": list(subset),
        "proposal_profile_path": str(profile_path.relative_to(ROOT)).replace("\\", "/"),
        "initial_palm_position_m": list(cfg.hand.mount_pos),
        "initial_palm_quaternion": list(cfg.hand.mount_quat),
        "initial_object_position_m": world_position.tolist(),
        "initial_object_quaternion": world_quaternion.tolist(),
        "initial_object_COM_palm_compiled_m": compiled_position.tolist(),
        "active_closure_scale": float(closure_scale),
        "open_joint_configuration_rad": open_joint.tolist(),
        "retaining_joint_target_rad": target_joint.tolist(),
    }


def evaluate_palmar_candidate(
    phase2r: Phase2RConfig, phase2: Phase2Config, attempt_index: int,
) -> dict:
    proposal = sample_palmar_candidate(phase2r, attempt_index)
    cfg = load_configs()
    model, data = build_scene(cfg)
    indices = resolve_hand_indices(model, cfg.hand)
    qadr, vadr = _object_addresses(model)
    open_joint = np.asarray(proposal["open_joint_configuration_rad"], dtype=float)
    target_joint = np.asarray(proposal["retaining_joint_target_rad"], dtype=float)
    fixture_position = np.asarray(proposal["initial_object_position_m"], dtype=float)
    fixture_quaternion = np.asarray(proposal["initial_object_quaternion"], dtype=float)
    data.qpos[indices.qpos_addresses] = open_joint
    data.qvel[indices.qvel_addresses] = 0.0
    data.qpos[qadr:qadr + 3] = fixture_position
    data.qpos[qadr + 3:qadr + 7] = fixture_quaternion
    data.qvel[vadr:vadr + 6] = 0.0
    mujoco.mj_forward(model, data)
    controller = JointImpedanceController(
        cfg.task.impedance_stiffness, cfg.task.impedance_damping, cfg.task.torque_limit,
    )
    close_steps = phase2r.state.fixture_close_steps
    contact_steps = phase2r.state.fixture_contact_steps
    for step in range(close_steps + contact_steps):
        alpha = min(1.0, (step + 1) / close_steps)
        desired = (1.0 - alpha) * open_joint + alpha * target_joint
        q, qvel = hand_state(data, indices)
        data.ctrl[indices.actuator_ids] = controller.torque(desired, q, qvel)
        # Temporary fixture: only the free-joint state is reset during closure.
        data.qpos[qadr:qadr + 3] = fixture_position
        data.qpos[qadr + 3:qadr + 7] = fixture_quaternion
        data.qvel[vadr:vadr + 6] = 0.0
        mujoco.mj_step(model, data)
    # Release boundary: clear fixture-induced object velocity and forward once.
    # Every subsequent step is ordinary gravity/contact/friction dynamics.
    data.qpos[qadr:qadr + 3] = fixture_position
    data.qpos[qadr + 3:qadr + 7] = fixture_quaternion
    data.qvel[vadr:vadr + 6] = 0.0
    mujoco.mj_forward(model, data)
    measured = measure_stable_hold(
        cfg, model, data, indices, target_joint, phase2r.state.stable_hold_steps,
        phase2.resources, phase2.dataset.friction_cone_edges, phase2.dataset.convex_hull_tolerance,
    )
    record = {
        **proposal,
        **measured,
        "fixture_method": "temporary_free_joint_pose_reset_during_initialization_only",
        "fixture_release_timestep": close_steps + contact_steps,
        "grasp_state_id": f"phase2R_palmar_attempt_{attempt_index:05d}",
    }
    return classify_grasp_state(
        record, GraspStateType.PALMAR_SECURED, phase2r.state,
        phase2.dataset.convex_hull_tolerance,
    )


def evaluate_existing_fingertip_state(
    phase2r: Phase2RConfig, phase2: Phase2Config, source: dict,
) -> dict:
    cfg, model, data, indices = reconstruct_grasp(source)
    # The accepted dataset stores both measured q and the torque-producing
    # impedance set point. Reuse the latter, as the formal Phase 2 replay does.
    hold = _joint_target(model, cfg, indices, source["hold_joint_fractions"])
    measured = measure_stable_hold(
        cfg, model, data, indices, hold, phase2r.state.stable_hold_steps,
        phase2.resources, phase2.dataset.friction_cone_edges, phase2.dataset.convex_hull_tolerance,
    )
    record = {
        **source,
        **measured,
        "source_grasp_id": source["grasp_id"],
        "grasp_state_id": f"phase2R_fingertip_{source['grasp_id']}",
        "fixture_method": "none_during_endpoint_revalidation",
        "fixture_release_timestep": 0,
    }
    return classify_grasp_state(
        record, GraspStateType.FINGERTIP, phase2r.state,
        phase2.dataset.convex_hull_tolerance,
    )
