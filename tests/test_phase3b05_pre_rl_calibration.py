from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from seqgrasp.phase3.control import actuator_target_from_qpos
from seqgrasp.phase3.env import load_keyframe_qpos
from seqgrasp.phase3.experiments import run_handoff_diagnostic
from seqgrasp.phase3.model import build_shadow_scene
from seqgrasp.phase3b0 import _joint_margins, sample_candidate
from seqgrasp.phase3b05 import (
    ACTIVE_PROTOCOL_VERSION,
    PERSISTENCE_HORIZONS,
    evaluate_feasibility_candidate,
    run_active_handoff,
    sample_feasibility_candidate,
    sampled_wrist_is_within_limits,
    symmetry_aware_orientation_change,
)


def test_shadow_joint_margin_uses_compiled_joint_order_and_signed_limits() -> None:
    scene = build_shadow_scene()
    ranges = scene.model.jnt_range[:24]
    scene.data.qpos[:24] = ranges.mean(axis=1)
    scene.data.qpos[3] = ranges[3, 0] - 0.002
    absolute, normalized = _joint_margins(scene)
    np.testing.assert_allclose(absolute[np.arange(24) != 3], (ranges[:, 1] - ranges[:, 0])[np.arange(24) != 3] / 2)
    assert np.isclose(absolute[3], -0.002)
    assert normalized[3] < 0.0


def test_tendon_joint_semantic_indexing_and_target_mapping() -> None:
    scene = build_shadow_scene()
    qpos = load_keyframe_qpos("pre grasp")
    target = actuator_target_from_qpos(scene, qpos)
    for finger, actuator_name in (("index", "rh_A_FFJ0"), ("middle", "rh_A_MFJ0"), ("ring", "rh_A_RFJ0"), ("little", "rh_A_LFJ0")):
        actuator_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        joint_ids = scene.joint_ids[finger][-2:]
        raw = float(sum(qpos[scene.model.jnt_qposadr[joint_id]] for joint_id in joint_ids))
        expected = np.clip(raw, *scene.model.actuator_ctrlrange[actuator_id])
        assert np.isclose(target[actuator_id], expected)


def test_release_command_limit_identifies_exact_two_actuators(tmp_path: Path) -> None:
    row = evaluate_feasibility_candidate(0, 0, tmp_path, retention_steps=2)
    scene = build_shadow_scene()
    names = [
        mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        for index, value in enumerate(row["release"]["actuator_saturated"])
        if value
    ]
    assert names == ["rh_A_LFJ4", "rh_A_LFJ0"]


def test_expanded_sampling_is_deterministic_and_exercises_orientation_and_wrist() -> None:
    first = sample_feasibility_candidate(3, 17)
    second = sample_feasibility_candidate(3, 17)
    assert first == second
    assert np.linalg.norm(first.object_euler_xyz_deg) > 0.0
    assert np.linalg.norm(first.wrist_perturbation_rad) > 0.0
    assert np.sum(np.abs(first.object_offset_m)) <= 0.010 + 1e-15
    assert first.orientation_limit_deg == 20.0
    assert sample_feasibility_candidate(3, 16).orientation_limit_deg == 15.0


def test_all_sampled_initial_wrist_poses_respect_joint_limits() -> None:
    for level in range(4):
        for candidate_id in range(32):
            assert sampled_wrist_is_within_limits(sample_feasibility_candidate(level, candidate_id))


def test_active_handoff_is_reproducible_and_records_persistence(tmp_path: Path) -> None:
    source = evaluate_feasibility_candidate(0, 0, tmp_path, retention_steps=2)
    kwargs = dict(
        state_path=source["release_state_path"],
        source_id="test",
        release_finger="thumb",
        output_directory=tmp_path,
        stage_steps=(3, 3, 3, 12),
    )
    first = run_active_handoff(**kwargs, trial_id="first")
    second = run_active_handoff(**kwargs, trial_id="second")
    for key in (
        "selected_finger_released",
        "final_retained_raw",
        "maximum_intended_penetration_m",
        "minimum_joint_margin_rad",
    ):
        assert first[key] == second[key]
    assert first["active_protocol_version"] == ACTIVE_PROTOCOL_VERSION
    assert tuple(int(value) for value in first["persistence"]) == PERSISTENCE_HORIZONS


def test_usable_motion_displacement_stiffness_and_rate_calibration(tmp_path: Path) -> None:
    source = evaluate_feasibility_candidate(0, 0, tmp_path, retention_steps=2)
    common = dict(
        state_path=source["release_state_path"],
        source_id="test",
        release_finger="index",
        output_directory=tmp_path,
        stage_steps=(3, 3, 3, 12),
    )
    motion = run_active_handoff(**common, family="C2", motion_scale=0.5, trial_id="motion")
    assert motion["usable_motion_probe"] is not None
    assert motion["usable_motion_probe"]["joint_space_available_motion_rad"] > 0.0
    assert motion["usable_motion_probe"]["jacobian_displacement_envelope_m"] > 0.0
    with np.load(tmp_path / "active/timeseries/motion.npz", allow_pickle=False) as stored:
        stages = stored["stage"]
        commands = stored["actuator_command"]
        motion_rows = np.flatnonzero(stages == "USABLE_MOTION_OUT")
        assert len(motion_rows) == 25
        assert np.max(np.abs(np.diff(commands[motion_rows], axis=0))) > 0.0
    displacement = run_active_handoff(**common, family="E2", displacement_scale=0.5, trial_id="e2")
    stiffness = run_active_handoff(**common, family="E3", stiffness_scale=0.5, trial_id="e3")
    rate = run_active_handoff(**common, family="E6", rate_scale=2.0, trial_id="e6")
    assert displacement["displacement_scale"] == 0.5
    assert stiffness["stiffness_scale"] == 0.5
    assert rate["rate_scale"] == 2.0
    assert stiffness["maximum_stiffness_rate_per_s"] > 0.0


def test_symmetry_aware_orientation_and_backward_compatibility() -> None:
    identity = np.asarray([1.0, 0.0, 0.0, 0.0])
    half_turn_x = np.asarray([0.0, 1.0, 0.0, 0.0])
    assert symmetry_aware_orientation_change(identity, half_turn_x) < 1e-12
    assert sample_candidate(11) == sample_candidate(11)
    handoff = run_handoff_diagnostic()
    assert handoff["summary"]["resource_recovered_diagnostic"]
    assert handoff["summary"]["post_release_object_qpos_was_never_set"]


def test_phase3b05_has_no_rl_dependency_or_reward_definition() -> None:
    source = (Path(__file__).parents[1] / "seqgrasp/phase3b05.py").read_text(encoding="utf-8")
    analysis = (Path(__file__).parents[1] / "seqgrasp/phase3b05_analysis.py").read_text(encoding="utf-8")
    combined = (source + analysis).lower()
    assert "stable_baselines" not in combined
    assert "import torch" not in combined
    assert "scalar_j" not in combined
    assert "reward_weights" not in combined
