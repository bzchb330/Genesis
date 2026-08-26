from __future__ import annotations

import ast
import inspect

import mujoco
import numpy as np
import pytest

from seqgrasp.phase3.model import build_shadow_scene, set_fixture, set_object_pose
from seqgrasp.phase3c0 import Phase3CState, gravity_in_palm_frame
from seqgrasp.phase3c05 import (
    CaptureStrategy,
    ConcurrentCaptureCommand,
    SupportPersistence,
    alternate_load,
    deterministic_candidate,
    exact_object_unused_finger_clearance,
    load_phase3c05_config,
    release_trial,
    released_finger_available_motion,
    capture_trial,
    MatchedCaptureState,
)


def test_coordinated_capture_phase_is_between_storage_and_release():
    members = list(Phase3CState)
    assert members.index(Phase3CState.A_IN_STORAGE_REGION) < members.index(Phase3CState.COORDINATED_CAPTURE)
    assert members.index(Phase3CState.COORDINATED_CAPTURE) < members.index(Phase3CState.RELEASE_ACQUISITION_DIGITS)


def test_concurrent_command_exposes_independent_acquisition_storage_and_wrist_channels():
    scene = build_shadow_scene()
    set_object_pose(scene, (1.0, 1.0, 1.0))
    mujoco.mj_forward(scene.model, scene.data)
    command = ConcurrentCaptureCommand(
        np.full(scene.model.nu, 0.1), np.full(scene.model.nu, 0.2), np.full(scene.model.nu, 0.3)
    )
    command.apply(
        scene, ("middle",), {}, acquisition_enabled=True, wrist_enabled=True,
        acquisition_increment=0.001, storage_increment=0.002, wrist_increment=0.003,
        contact_force_n=1e9,
    )
    np.testing.assert_allclose(scene.data.ctrl[scene.actuator_ids["thumb"]], 0.001)
    np.testing.assert_allclose(scene.data.ctrl[scene.actuator_ids["index"]], 0.001)
    np.testing.assert_allclose(scene.data.ctrl[scene.actuator_ids["middle"]], 0.002)
    np.testing.assert_allclose(scene.data.ctrl[scene.actuator_ids["wrist"]], 0.003)


def test_wrist_control_is_optional_and_world_gravity_is_unchanged():
    scene = build_shadow_scene()
    mujoco.mj_forward(scene.model, scene.data)
    before = scene.model.opt.gravity.copy()
    command = ConcurrentCaptureCommand(np.zeros(scene.model.nu), np.zeros(scene.model.nu), np.ones(scene.model.nu))
    command.apply(scene, (), {}, acquisition_enabled=False, wrist_enabled=True,
                  acquisition_increment=.01, storage_increment=.01, wrist_increment=.01,
                  contact_force_n=.02)
    assert np.all(scene.data.ctrl[scene.actuator_ids["wrist"]] > 0)
    np.testing.assert_array_equal(scene.model.opt.gravity, before)


def test_gravity_transform_preserves_magnitude_and_responds_to_wrist_pose():
    scene = build_shadow_scene(); mujoco.mj_forward(scene.model, scene.data)
    before = gravity_in_palm_frame(scene)
    joint = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, "rh_WRJ1")
    scene.data.qpos[scene.model.jnt_qposadr[joint]] += 0.1
    mujoco.mj_forward(scene.model, scene.data)
    after = gravity_in_palm_frame(scene)
    assert np.linalg.norm(after) == pytest.approx(9.81)
    assert not np.allclose(before, after)


def test_acquisition_release_is_withheld_before_support_gate():
    scene = build_shadow_scene(); mujoco.mj_forward(scene.model, scene.data)
    snap = {"qpos": scene.data.qpos.copy(), "qvel": scene.data.qvel.copy(), "ctrl": scene.data.ctrl.copy()}
    result = release_trial(scene, {"final_snapshot": snap, "persistence_first_reached": {"0.10/25": None}}, "thumb", 25)
    assert not result["executed"] and result["reason"] == "SUPPORT_GATE_NOT_REACHED"


def test_all_seven_storage_finger_subsets_are_frozen():
    subsets = {tuple(value) for value in load_phase3c05_config()["capture"]["storage_subsets"]}
    assert subsets == {("middle",), ("ring",), ("little",), ("middle", "ring"),
                       ("middle", "little"), ("ring", "little"),
                       ("middle", "ring", "little")}


def test_alternate_load_excludes_thumb_and_index_and_handles_zero_force():
    share = alternate_load([1, 2, 3, 4, 5, 6])
    assert share.acquisition_force_n == 3 and share.alternate_force_n == 18
    assert share.alternate_fraction == pytest.approx(18 / 21)
    assert alternate_load(np.zeros(6)).alternate_fraction == 0.0


def test_support_persistence_requires_consecutive_samples():
    persistence = SupportPersistence((.10,), (3,))
    for step, value in enumerate((.2, .2, 0, .2, .2, .2)):
        persistence.update(value, step)
    assert persistence.first_reached["0.10/3"] == 3


@pytest.mark.parametrize("finger", ["thumb", "index"])
def test_gradual_one_finger_release_reaches_open_target_without_fixture(finger):
    scene = build_shadow_scene(); set_fixture(scene, False); mujoco.mj_forward(scene.model, scene.data)
    snap = {"qpos": scene.data.qpos.copy(), "qvel": scene.data.qvel.copy(), "ctrl": scene.data.ctrl.copy()}
    result = release_trial(scene, {"final_snapshot": snap, "persistence_first_reached": {"0.10/25": 0}},
                           finger, 25, post_steps=1, record_state=True)
    assert result["executed"] and len(result["samples"]) == 26
    assert not result["fixture_active"]


def test_resource_detector_reports_positive_joint_space_motion():
    scene = build_shadow_scene(); mujoco.mj_forward(scene.model, scene.data)
    assert released_finger_available_motion(scene, "thumb") > 0
    assert released_finger_available_motion(scene, "index") > 0


def test_matched_state_candidates_are_deterministic_and_nonidentical():
    first = deterministic_candidate(7); replay = deterministic_candidate(7); other = deterministic_candidate(8)
    np.testing.assert_array_equal(first[0], replay[0]); np.testing.assert_array_equal(first[1], replay[1])
    assert not np.array_equal(first[0], other[0])
    assert load_phase3c05_config()["matched_states"]["count"] == 50


def test_corridor_audit_uses_exact_compiled_geometry_distance():
    scene = build_shadow_scene(); mujoco.mj_resetData(scene.model, scene.data)
    set_object_pose(scene, (1.0, 1.0, 1.0)); mujoco.mj_forward(scene.model, scene.data)
    assert exact_object_unused_finger_clearance(scene) > 0.0


def test_phase3c05_has_no_object_B_and_no_rl_dependency():
    scene = build_shadow_scene()
    source = inspect.getsource(__import__("seqgrasp.phase3c05", fromlist=["phase3c05"]))
    imports = {
        alias.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "object_B") == -1
    assert not any(name.startswith("stable_baselines") for name in imports)
    assert "reward_weights" not in load_phase3c05_config()


def test_phase3c0_backward_compatible_scene_dimensions_and_states():
    scene = build_shadow_scene()
    assert scene.model.nq == 31 and scene.model.nu == 20
    assert Phase3CState.OPEN_HAND.value == "OPEN_HAND"
    assert Phase3CState.A_IN_STORAGE_REGION.value == "A_IN_STORAGE_REGION"


def test_capture_sample_records_contact_geometry_force_ratio_and_displacement():
    scene = build_shadow_scene(); mujoco.mj_forward(scene.model, scene.data)
    payload = np.r_[scene.data.qpos, scene.data.qvel, scene.data.ctrl]
    state = MatchedCaptureState(
        "test", 0, (0, 0, 0), (1, 0, 0, 0), 0, "test",
        tuple(scene.data.qpos), tuple(scene.data.qvel), tuple(scene.data.ctrl),
        (0, 0, 0), (0, 0, 0, 0, 0, 0), 0.0, str(payload.size),
    )
    trial = capture_trial(scene, state, ("middle",), CaptureStrategy.SIMULTANEOUS,
                          capture_steps=1)
    sample = trial["samples"][0]
    assert set(sample["contact_details"]) == {"thumb", "index", "middle", "ring", "little", "palm"}
    assert len(sample["tangential_normal_ratio_by_surface"]) == 6
    assert len(sample["penetration_by_surface_m"]) == 6
    assert sample["A_displacement_from_capture_start_m"] >= 0.0
