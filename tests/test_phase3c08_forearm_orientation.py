from __future__ import annotations

import ast
import inspect

import mujoco
import numpy as np
import pytest

import seqgrasp.phase3c08 as c08
from seqgrasp.phase3c0 import gravity_in_palm_frame
from seqgrasp.phase3c07 import build_c07_scene


def test_target_direction_reconstruction_uses_nearest_voxels():
    points = np.asarray([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]])
    indices, distances, directions = c08.directions_to_nearest_voxels((0, 0, 0), points, 2)
    assert indices.tolist() == [1, 0]
    assert distances.tolist() == [2.0, 1.0, 3.0]
    np.testing.assert_allclose(directions, [[0, 1, 0], [1, 0, 0]])


def test_gravity_transform_preserves_world_gravity_and_normalizes_result():
    scene = build_c07_scene(); world = scene.model.opt.gravity.copy()
    direction = c08._gravity_at(scene, {"rh_WRJ2": 0.1})
    assert np.linalg.norm(direction) == pytest.approx(1.0)
    np.testing.assert_array_equal(scene.model.opt.gravity, world)
    np.testing.assert_allclose(direction, gravity_in_palm_frame(scene) / 9.81)


def test_native_reachable_set_generation_shape():
    scene = build_c07_scene()
    configurations, directions = c08._scan_scene(
        scene, ("rh_WRJ1", "rh_WRJ2"), (np.asarray([-0.1, 0.1]), np.asarray([-0.2, 0.0, 0.2]))
    )
    assert configurations.shape == (6, 2)
    assert directions.shape == (6, 3)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)


def test_forearm_axis_is_derived_from_official_hierarchy():
    axis = c08.identify_forearm_axis()
    assert (axis.parent_body, axis.child_body) == ("rh_forearm", "rh_wrist")
    assert np.linalg.norm(axis.axis_parent) == pytest.approx(1.0)
    assert abs(axis.axis_parent[2]) > 0.99
    assert "rh_wrist child anchor" in axis.evidence


def test_wrapper_zero_angle_backward_compatibility_for_nominal_state():
    native = build_c07_scene(); augmented = c08.build_forearm_scene().scene
    mujoco.mj_forward(native.model, native.data)
    c08.copy_common_state(native, augmented)
    palm_native = mujoco.mj_name2id(native.model, mujoco.mjtObj.mjOBJ_BODY, native.config.hand.palm_body)
    palm_augmented = mujoco.mj_name2id(augmented.model, mujoco.mjtObj.mjOBJ_BODY, augmented.config.hand.palm_body)
    np.testing.assert_allclose(native.data.xpos[palm_native], augmented.data.xpos[palm_augmented], atol=1e-12)
    np.testing.assert_allclose(native.data.xmat[palm_native], augmented.data.xmat[palm_augmented], atol=1e-12)


def test_forearm_qpos_is_bounded_and_assignable():
    wrapper = c08.build_forearm_scene()
    np.testing.assert_allclose(wrapper.scene.model.jnt_range[wrapper.forearm_joint_id], [-np.pi / 2, np.pi / 2])
    address = wrapper.scene.model.jnt_qposadr[wrapper.forearm_joint_id]
    wrapper.scene.data.qpos[address] = 0.25; mujoco.mj_forward(wrapper.scene.model, wrapper.scene.data)
    assert wrapper.scene.data.qpos[address] == pytest.approx(0.25)


def test_augmented_reachable_set_generation_includes_forearm_dimension():
    wrapper = c08.build_forearm_scene()
    configurations, directions = c08._scan_scene(
        wrapper.scene, (c08.FOREARM_JOINT_NAME, "rh_WRJ1", "rh_WRJ2"),
        (np.asarray([-0.2, 0.2]), np.asarray([0.0]), np.asarray([0.0])),
    )
    assert configurations.shape == (2, 3)
    assert not np.allclose(directions[0], directions[1])


def test_angular_residual_calculation():
    residual = c08.angular_residual_deg(np.eye(3), (1, 0, 0))
    np.testing.assert_allclose(residual, [0, 90, 90])


def test_gravity_projection_is_reported_separately():
    projection = c08.gravity_projection(np.eye(3), (1, 0, 0))
    np.testing.assert_allclose(projection, [1, 0, 0])
    assert not np.array_equal(projection, c08.angular_residual_deg(np.eye(3), (1, 0, 0)))


def test_coarse_to_fine_optimization_is_deterministic():
    wrapper = c08.build_forearm_scene()
    names = (c08.FOREARM_JOINT_NAME, "rh_WRJ1", "rh_WRJ2")
    bounds = ((-0.3, 0.3), (-0.2, 0.2), (-0.1, 0.1))
    target = np.asarray([-0.4, 0.8, -0.2]); target /= np.linalg.norm(target)
    first = c08._refine(wrapper.scene, names, bounds, np.zeros(3), target)
    second = c08._refine(wrapper.scene, names, bounds, np.zeros(3), target)
    np.testing.assert_allclose(first[0], second[0], atol=1e-12)
    assert first[2] == pytest.approx(second[2], abs=1e-12)


def test_native_and_forearm_joint_limits_are_respected():
    native = build_c07_scene(); cfg = c08.load_phase3c08_config()
    for name in ("rh_WRJ1", "rh_WRJ2"):
        joint = mujoco.mj_name2id(native.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert native.model.jnt_limited[joint]
    assert cfg["forearm"]["diagnostic_range_deg"] == [-90.0, 90.0]


def test_no_dynamics_before_kinematic_gate(tmp_path):
    source = {"reachable_gravity_audit": {"targeted_dynamics_authorized": False}}
    with pytest.raises(RuntimeError, match="does not authorize"):
        c08.freeze_targeted_dynamic_manifest(source, tmp_path / "manifest.json")


def test_targeted_dynamics_cap_is_exactly_50():
    cfg = c08.load_phase3c08_config()["targeted_dynamics"]
    assert cfg["maximum_trials"] == 50
    assert cfg["state_count"] * cfg["configurations_per_state"] == 50
    assert "exceeds 50-trial cap" in inspect.getsource(c08.freeze_targeted_dynamic_manifest)


def test_no_thumb_release_contract():
    assert c08.kinematic_contract()["thumb_release_performed"] is False
    assert '"thumb_release_performed": False' in inspect.getsource(c08.run_targeted_dynamic_trial)


def test_no_object_b_contract_or_model_name():
    assert c08.kinematic_contract()["object_B_instantiated"] is False
    wrapper = c08.build_forearm_scene()
    assert mujoco.mj_name2id(wrapper.scene.model, mujoco.mjtObj.mjOBJ_BODY, "object_B") < 0


def test_no_rl_reward_or_scalar_j():
    contract = c08.kinematic_contract()
    assert not contract["rl_training_performed"] and not contract["reward_defined"] and not contract["scalar_J_defined"]
    imports = {alias.name for node in ast.walk(ast.parse(inspect.getsource(c08)))
               if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    assert not any(name.startswith("stable_baselines") for name in imports)


def test_no_friction_or_contact_changes_in_wrapper():
    native = build_c07_scene(); wrapper = c08.build_forearm_scene().scene
    np.testing.assert_array_equal(native.model.geom_friction, wrapper.model.geom_friction)
    np.testing.assert_array_equal(native.model.geom_solref, wrapper.model.geom_solref)
    np.testing.assert_array_equal(native.model.geom_solimp, wrapper.model.geom_solimp)
    assert not c08.kinematic_contract()["friction_changed"]
    assert not c08.kinematic_contract()["contact_parameters_changed"]


def test_phase3c07_model_remains_backward_compatible():
    before = build_c07_scene(); after = build_c07_scene()
    assert (before.model.nq, before.model.nv, before.model.nu) == (after.model.nq, after.model.nv, after.model.nu)
    assert mujoco.mj_name2id(before.model, mujoco.mjtObj.mjOBJ_JOINT, c08.FOREARM_JOINT_NAME) < 0
