from __future__ import annotations

import mujoco
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from seqgrasp.phase3.env import Phase3ShadowHandEnv
from seqgrasp.phase3.model import build_shadow_scene
from seqgrasp.phase3b1a_env import Phase3B1APrivilegedEnv
from seqgrasp.phase3c0 import (
    Phase3CFingerRole,
    Phase3CFailure,
    Phase3CRoles,
    Phase3CState,
    StorageRegion,
    build_phase3c_multiscene,
    configured_storage_region,
    gravity_in_palm_frame,
    load_phase3c0_config,
    multi_object_support_graph,
    object_pose_in_palm,
    open_hand_configuration,
    phase3c_action_contract,
    phase3c_observation_contract,
    release_phase3c_fixture,
    set_phase3c_object_pose,
    storage_aperture,
    storage_measurement,
    transfer_corridor,
)
from seqgrasp.phase3c0_env import Phase3COpenCorridorEnv


def test_open_hand_initialization_is_explicit_feasible_and_all_zero():
    qpos, projection = open_hand_configuration()
    np.testing.assert_allclose(qpos, 0.0, atol=1e-12)
    assert projection["optimizer_success"]
    assert projection["projected_minimum_joint_margin_rad"] >= -1e-10


def test_phase3c_has_all_required_time_varying_finger_roles():
    assert {role.value for role in Phase3CFingerRole} == {
        "FREE", "PROBING", "ACQUIRING", "TRANSFERRING", "CLEARING_CORRIDOR",
        "SECURING_STORAGE", "RELAXING_APERTURE", "RESECURING", "RELEASING",
    }
    roles = Phase3CRoles()
    roles.begin_minimal_acquisition("A", 2)
    assert roles.fingers["middle"] == Phase3CFingerRole.CLEARING_CORRIDOR
    roles.begin_transfer(3)
    roles.storage_entry(("middle", "ring"), 7)
    roles.relax_aperture(("middle", "ring"), 9)
    roles.resecure(("middle", "ring"), 12)
    assert roles.fingers["middle"] == Phase3CFingerRole.RESECURING
    assert [item["step"] for item in roles.history] == [2, 3, 7, 9, 12]


def test_clearing_corridor_is_semantic_not_maximum_openness():
    roles = Phase3CRoles()
    roles.begin_minimal_acquisition("A", 0)
    assert all(roles.fingers[f] == Phase3CFingerRole.CLEARING_CORRIDOR
               for f in ("middle", "ring", "little"))
    assert "corridor" in roles.history[-1]["reason"]


def test_open_corridor_initial_pose_has_no_unused_finger_contact():
    cfg = load_phase3c0_config()
    scene = build_phase3c_multiscene()
    mujoco.mj_resetData(scene.model, scene.data)
    qpos, _ = open_hand_configuration()
    scene.data.qpos[:24] = qpos
    set_phase3c_object_pose(scene, "A", cfg["diagnostic"]["open_corridor_initial_pos_m"],
                            cfg["diagnostic"]["open_corridor_initial_quat_wxyz"])
    mujoco.mj_forward(scene.model, scene.data)
    graph = multi_object_support_graph(scene)
    unused = {edge["support"] for edge in graph["edges"] if edge["object"] == "A"}
    assert not unused.intersection({"middle", "ring", "little"})


def test_storage_region_reports_raw_extent_overlap_and_containment():
    region = StorageRegion((0.0, 0.0, 0.0), (0.05, 0.05, 0.05))
    inside = region.measure(np.zeros(3), np.full(3, 0.01))
    outside = region.measure(np.asarray([0.2, 0.0, 0.0]), np.full(3, 0.01))
    assert inside["extent_fully_inside"] and inside["occupancy_fraction"] == pytest.approx(1.0)
    assert not outside["center_inside"] and outside["occupancy_fraction"] == 0.0


def test_compiled_storage_measurement_is_in_palm_frame():
    scene = build_phase3c_multiscene()
    mujoco.mj_forward(scene.model, scene.data)
    measured = storage_measurement(scene, scene.object_body_ids["A"],
                                   np.asarray(scene.config.object["size"]))
    direct, rotation = object_pose_in_palm(scene, scene.object_body_ids["A"])
    np.testing.assert_allclose(measured["object_center_palm_m"], direct)
    assert np.asarray(rotation).shape == (3, 3)


def test_transfer_corridor_returns_clearance_obstructions_and_bottleneck():
    scene = build_shadow_scene()
    mujoco.mj_forward(scene.model, scene.data)
    result = transfer_corridor(scene, np.asarray([0.45, -0.1, 0.05]),
                               np.asarray([0.29, 0.0, 0.01]), object_radius_m=0.02, samples=17)
    assert len(result["clearance_m"]) == 17
    assert 0.0 <= result["bottleneck_fraction"] <= 1.0
    assert 0.0 <= result["collision_free_fraction"] <= 1.0
    assert isinstance(result["obstructing_links"], list)


def test_delayed_recruitment_requires_explicit_storage_entry_transition():
    roles = Phase3CRoles()
    roles.begin_minimal_acquisition("A", 1)
    roles.begin_transfer(2)
    assert roles.fingers["ring"] == Phase3CFingerRole.CLEARING_CORRIDOR
    roles.storage_entry(("ring", "little"), 11)
    assert roles.fingers["ring"] == Phase3CFingerRole.SECURING_STORAGE
    assert roles.history[-1]["state"] == Phase3CState.A_IN_STORAGE_REGION.value


def test_storage_aperture_is_state_dependent_palm_frame_geometry():
    scene = build_phase3c_multiscene()
    qpos, _ = open_hand_configuration()
    scene.data.qpos[:24] = qpos
    mujoco.mj_forward(scene.model, scene.data)
    aperture = storage_aperture(scene)
    assert np.linalg.norm(aperture["normal_palm"]) == pytest.approx(1.0)
    assert aperture["effective_width_m"] > 0.0
    assert aperture["effective_height_m"] > 0.0
    assert aperture["support_nodes"][-1] == "palm"


def test_wrist_orientation_changes_palm_frame_gravity_but_not_world_gravity():
    scene = build_phase3c_multiscene()
    mujoco.mj_forward(scene.model, scene.data)
    before = gravity_in_palm_frame(scene)
    world_before = scene.model.opt.gravity.copy()
    wrist = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, "rh_WRJ1")
    scene.data.qpos[scene.model.jnt_qposadr[wrist]] = 0.3
    mujoco.mj_forward(scene.model, scene.data)
    after = gravity_in_palm_frame(scene)
    assert not np.allclose(before, after)
    np.testing.assert_array_equal(scene.model.opt.gravity, world_before)
    assert np.linalg.norm(after) == pytest.approx(9.81)


def test_retention_wrist_search_has_compiled_feasible_domain():
    scene = build_phase3c_multiscene()
    for name in scene.config.hand.wrist_joints:
        jid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert scene.model.jnt_limited[jid]
        assert scene.model.jnt_range[jid, 1] > scene.model.jnt_range[jid, 0]


def test_two_object_support_graph_preserves_object_and_support_identity():
    scene = build_phase3c_multiscene()
    mujoco.mj_forward(scene.model, scene.data)
    graph = multi_object_support_graph(scene)
    assert graph["hand_nodes"] == ["thumb", "index", "middle", "ring", "little", "palm"]
    assert graph["object_nodes"] == ["A", "B"]
    assert isinstance(graph["edges"], list)


def test_A_is_dynamic_during_B_setup_and_cannot_be_reteleported():
    scene = build_phase3c_multiscene()
    release_phase3c_fixture(scene, "A")
    assert scene.data.eq_active[scene.fixture_eq_ids["A"]] == 0
    with pytest.raises(RuntimeError, match="after its fixture has been released"):
        set_phase3c_object_pose(scene, "A", (0.3, 0.0, 0.0))
    assert scene.model.jnt_type[scene.object_joint_ids["A"]] == mujoco.mjtJoint.mjJNT_FREE


def test_fixture_api_is_one_way_and_does_not_expose_reactivation():
    scene = build_phase3c_multiscene()
    release_phase3c_fixture(scene, "B")
    assert not bool(scene.data.eq_active[scene.fixture_eq_ids["B"]])
    assert "activate" not in release_phase3c_fixture.__name__


def test_aperture_relaxation_changes_geometry_without_gravity_control():
    scene = build_phase3c_multiscene()
    qpos, _ = open_hand_configuration()
    scene.data.qpos[:24] = qpos
    mujoco.mj_forward(scene.model, scene.data)
    before = storage_aperture(scene)
    middle = scene.joint_ids["middle"]
    scene.data.qpos[scene.model.jnt_qposadr[middle]] += 0.05
    mujoco.mj_forward(scene.model, scene.data)
    after = storage_aperture(scene)
    assert before != after
    np.testing.assert_array_equal(scene.model.opt.gravity, [0.0, 0.0, -9.81])


def test_resecure_and_failure_labels_do_not_rewrite_historical_labels():
    roles = Phase3CRoles()
    roles.relax_aperture(("middle", "ring"), 4)
    roles.resecure(("middle", "ring"), 8)
    assert roles.state == Phase3CState.RESECURE_A_AND_B
    assert Phase3CFailure.RESECURE_FAILED.value == "RESECURE_FAILED"
    assert len(Phase3CFailure) == 16


def test_phase3c_future_contract_has_wrist_action_and_no_reward_weights():
    scene = build_phase3c_multiscene()
    action = phase3c_action_contract(scene)
    assert action["wrist_target_increments"] == 2
    assert not action["world_gravity_controlled"]
    assert not action["reward_weights_defined"]
    assert not any(component.name == "scalar_J" for component in phase3c_observation_contract())


def test_phase3c_env_dimensions_checker_and_seeded_reset():
    env = Phase3COpenCorridorEnv()
    check_env(env, skip_render_check=True)
    first, info = env.reset(seed=37)
    second, _ = env.reset(seed=37)
    np.testing.assert_array_equal(first, second)
    assert first.size == sum(item.dimension for item in env.observation_metadata if item.actor_available)
    assert set(info["privileged_observation"]) == {
        item.name for item in env.observation_metadata if not item.actor_available
    }
    _, reward, terminated, _, info = env.step(np.zeros(env.action_space.shape, np.float32))
    assert reward == 0.0 and not terminated and not info["reward_defined"]


def test_phase3A_and_phase3B_interfaces_remain_backward_compatible():
    legacy = Phase3ShadowHandEnv()
    phase3b = Phase3B1APrivilegedEnv()
    assert legacy.action_space.shape == (26,)
    assert phase3b.action_space.shape == (26,)
    assert build_shadow_scene().model.nq == 31
    assert build_phase3c_multiscene().model.nq == 38
