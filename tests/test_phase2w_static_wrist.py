from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import mujoco

from seqgrasp.config import load_configs
from seqgrasp.experiments.static_wrist import (
    WORLD_GRAVITY_M_PER_S2,
    StaticWristPose,
    _initial_invalid_reason,
    coarse_wrist_poses,
    compose_mount_quaternion_wxyz,
    formal_seed_id,
    freeze_wrist_b_region,
    gravity_in_palm_frame,
    normalize_quaternion_wxyz,
    recompute_index_thumb_workspace,
    refined_wrist_poses,
    transform_pose_preserving_palm_relative,
    verify_wrist_b_freeze,
)
from seqgrasp.experiments.phase2w_protocol import calibration_trial_id, freeze_controller_payload, verify_controller_freeze
from seqgrasp.phase2_config import load_phase2_config
from seqgrasp.control import resolve_hand_indices
from seqgrasp.phase2w_config import load_phase2w_config
from seqgrasp.scene_builder import build_scene


def test_phase2w_config_and_static_root_contract():
    cfg, _ = load_phase2w_config()
    assert cfg.topology.occupied_fingers == ["middle", "ring"]
    assert cfg.topology.free_fingers == ["index", "thumb"]
    assert cfg.second_grasp.global_candidate_cap == 8192
    assert cfg.calibration.formal_B_seeds_per_pair == 20
    scene = load_configs(scene_filename=cfg.scene_filename)
    model, _ = build_scene(scene)
    assert model.nu == scene.hand.dof_count
    assert np.allclose(model.opt.gravity, WORLD_GRAVITY_M_PER_S2)
    assert all("wrist" not in name.lower() for name in scene.hand.actuator_names)


def test_quaternion_normalization_and_deterministic_wrist_transform():
    assert np.allclose(normalize_quaternion_wxyz([2, 0, 0, 0]), [1, 0, 0, 0])
    with pytest.raises(ValueError):
        normalize_quaternion_wxyz([0, 0, 0, 0])
    relative = normalize_quaternion_wxyz([1, 0.2, -0.1, 0.3])
    first = compose_mount_quaternion_wxyz([1, 0, 0, 0], relative)
    second = compose_mount_quaternion_wxyz([1, 0, 0, 0], relative)
    assert np.array_equal(first, second)


def test_transform_preserves_palm_relative_A_pose_and_world_gravity():
    old_palm = np.array([0.0, 0.0, 0.18])
    old_quat = np.array([1.0, 0.0, 0.0, 0.0])
    new_quat = normalize_quaternion_wxyz([0.9238795325, 0.0, 0.3826834324, 0.0])
    A_position = np.array([0.04, -0.01, 0.17])
    A_quat = normalize_quaternion_wxyz([0.98, 0.1, 0.0, 0.0])
    transformed_position, transformed_quat = transform_pose_preserving_palm_relative(
        old_palm, old_quat, old_palm, new_quat, A_position, A_quat,
    )
    from scipy.spatial.transform import Rotation
    old_R = Rotation.from_quat(old_quat, scalar_first=True)
    new_R = Rotation.from_quat(new_quat, scalar_first=True)
    assert np.allclose(old_R.inv().apply(A_position - old_palm), new_R.inv().apply(transformed_position - old_palm))
    relative_old = old_R.inv() * Rotation.from_quat(A_quat, scalar_first=True)
    relative_new = new_R.inv() * Rotation.from_quat(transformed_quat, scalar_first=True)
    assert np.allclose(relative_old.as_matrix(), relative_new.as_matrix())
    assert np.allclose(gravity_in_palm_frame(new_quat), new_R.inv().apply(WORLD_GRAVITY_M_PER_S2))


def test_orientation_generation_deduplicates_and_keeps_baseline():
    cfg, _ = load_phase2w_config()
    poses = coarse_wrist_poses(
        cfg.wrist_search.coarse_roll_deg,
        cfg.wrist_search.coarse_pitch_deg,
        cfg.wrist_search.coarse_yaw_deg,
    )
    assert len(poses) <= 125
    assert len({tuple(np.round(p.relative_quaternion_wxyz, 10)) for p in poses}) == len(poses)
    assert any(np.allclose(p.relative_quaternion_wxyz, [1, 0, 0, 0]) for p in poses)
    refined = refined_wrist_poses(poses[:5], cfg.wrist_search.refinement_offsets_deg)
    assert len(refined) <= 135


def test_wrist_and_B_freeze_rejects_mutation():
    pose = StaticWristPose("test", (0.0, 45.0, 0.0), tuple(normalize_quaternion_wxyz([1, 0, 0.2, 0])), "test")
    frozen = freeze_wrist_b_region(pose, [[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]], [-0.1, 0.1])
    verify_wrist_b_freeze(frozen)
    with pytest.raises(ValueError):
        verify_wrist_b_freeze(replace(frozen, B_yaw_bounds_rad=(-0.2, 0.2)))
    controller = freeze_controller_payload(frozen, {"hold": [0.1, 0.2]}, ["cal-2", "cal-1"])
    verify_controller_freeze(controller)
    with pytest.raises(ValueError):
        verify_controller_freeze(replace(controller, controller_payload={"hold": [0.9]}))


def test_transformed_workspace_is_recomputed_in_compiled_scene():
    phase2, _ = load_phase2_config()
    cfg = load_configs(scene_filename="scene_two_object_half_scale.yaml")
    model, _ = build_scene(cfg)
    indices = resolve_hand_indices(model, cfg.hand)
    record = {
        "grasp_id": "synthetic-workspace-test",
        "initial_palm_position_m": cfg.hand.mount_pos,
        "initial_palm_quaternion": cfg.hand.mount_quat,
        "final_joint_configuration_rad": model.qpos0[indices.qpos_addresses].tolist(),
        "final_object_position_m": [0.30, 0.0, 0.10],
        "final_object_quaternion": [1.0, 0.0, 0.0, 0.0],
        "occupied_finger_mask": [False, True, True, False],
    }
    result = recompute_index_thumb_workspace(record, phase2.resources, 20, 1234, cfg)
    assert result["proposed_samples"] == 20
    assert result["index_points_world_m"].shape[1:] == (3,)
    assert result["thumb_points_world_m"].shape[1:] == (3,)
    assert result["free_joint_samples_rad"].shape[1:] == (8,)


def test_invalid_initial_hand_and_object_table_contacts_are_rejected():
    cfg = load_configs(scene_filename="scene_two_object_half_scale.yaml")
    low_hand_cfg = replace(cfg, hand=replace(cfg.hand, mount_pos=[0.0, 0.0, 0.0]))
    model, data = build_scene(low_hand_cfg)
    for joint_name, position in (
        ("object_a_free", [-0.4, 0.4, 0.2]),
        ("object_b_free", [-0.4, -0.4, 0.2]),
    ):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qpos_address = model.jnt_qposadr[joint_id]
        data.qpos[qpos_address:qpos_address + 3] = position
    mujoco.mj_forward(model, data)
    assert _initial_invalid_reason(model, data, 1e-4) == "initial_hand_table_contact"

    model, data = build_scene(cfg)
    for joint_name, position in (
        ("object_a_free", [0.3, 0.0, 0.0]),
        ("object_b_free", [-0.4, -0.4, 0.2]),
    ):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qpos_address = model.jnt_qposadr[joint_id]
        data.qpos[qpos_address:qpos_address + 3] = position
    mujoco.mj_forward(model, data)
    assert _initial_invalid_reason(model, data, 1e-4) == "initial_A_table_contact"


def test_phase2w_formal_seed_isolated_and_sources_define_no_J_RL_or_dynamic_wrist():
    assert formal_seed_id(20261009, "pair-1", "FINGERTIP", 0) != formal_seed_id(20261009, "pair-1", "PALMAR_SECURED", 0)
    assert calibration_trial_id(20261008, "state-1", 0, 0) != formal_seed_id(20261009, "pair-1", "FINGERTIP", 0)
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "configs" / "phase2W_static_wrist_feasibility.yaml",
        root / "seqgrasp" / "phase2w_config.py",
        root / "seqgrasp" / "experiments" / "static_wrist.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths).lower()
    forbidden = ("compute_" + "j", "train_" + "rl", "wrist_actuator", "wrist_velocity_target")
    assert all(token not in text for token in forbidden)
