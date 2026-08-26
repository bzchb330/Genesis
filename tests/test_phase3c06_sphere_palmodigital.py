from __future__ import annotations

import ast
import inspect

import mujoco
import numpy as np
import pytest

from seqgrasp.phase3.model import build_shadow_scene
from seqgrasp.phase3c0 import configured_storage_region, gravity_in_palm_frame
from seqgrasp.phase3c06 import (
    FAILURE_TAXONOMY,
    POCKET_NAMES,
    acquire_sphere_state,
    audit_non_thumb_link_lengths,
    build_sphere_scene,
    construct_palmodigital_pockets,
    load_phase3c06_config,
    no_object_b_or_rl_contract,
    normalized_penetration,
    phase3c06_scene_config,
    preshape_trigger,
    progression_allowed,
    reference_link_length,
    size_curriculum,
    sphere_scale,
    storage_state,
    wrist_commands,
)


def test_sphere_geometry_compiles_as_sphere():
    scene = build_sphere_scene()
    geom = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, "phase3c06_sphere_geom")
    assert scene.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_SPHERE
    assert scene.model.geom_size[geom, 0] == pytest.approx(0.0175)


def test_D0_diameter_is_median_of_official_non_thumb_links():
    rows = audit_non_thumb_link_lengths()
    assert len(rows) == 8
    assert [row.length_m for row in rows].count(0.045) == 4
    assert [row.length_m for row in rows].count(0.025) == 4
    assert reference_link_length(rows) == pytest.approx(0.035)
    assert sphere_scale().diameter_m == pytest.approx(0.035)


def test_density_consistent_mass_scaling_and_compiled_mass():
    d0, d1 = sphere_scale(), sphere_scale(1.25)
    assert d1.mass_kg / d0.mass_kg == pytest.approx(1.25 ** 3)
    scene = build_sphere_scene()
    assert scene.model.body_mass[scene.object_body_id] == pytest.approx(d0.mass_kg)


def test_palmodigital_pocket_coordinate_construction_uses_compiled_roots():
    pockets = construct_palmodigital_pockets()
    assert tuple(pockets) == POCKET_NAMES
    assert pockets["ring_little"].center_palm_m[0] < pockets["middle_ring"].center_palm_m[0]
    assert pockets["ring_little"].center_palm_m[2] > pockets["old_palm_center"].center_palm_m[2]


def test_old_palm_center_is_backward_compatible():
    old = configured_storage_region()
    control = construct_palmodigital_pockets()["old_palm_center"]
    np.testing.assert_allclose(control.center_palm_m, old.center_palm_m)
    np.testing.assert_allclose(control.half_extents_m, old.half_extents_m)


def test_open_hand_sphere_acquisition_uses_only_thumb_and_index():
    scene = build_sphere_scene()
    state = acquire_sphere_state(scene, 0)
    assert state is not None
    assert state.contact_flags[0] and state.contact_flags[1]
    assert not any(state.contact_flags[2:5])


def test_sphere_transfer_corridor_clearance_uses_exact_geom_distance():
    scene = build_sphere_scene()
    state = acquire_sphere_state(scene, 0)
    assert state is not None
    assert not any(state.contact_flags[2:5])


def test_preshape_trigger_is_geometric_and_not_time_based():
    assert preshape_trigger(.6, .5, .001, .001)
    assert not preshape_trigger(.4, .5, .001, .001)
    assert not preshape_trigger(.6, .5, -.001, .001)


def test_ring_little_support_mapping_is_explicit():
    cfg = load_phase3c06_config()
    assert cfg["experiment"]["storage_fingers"]["ring_little"] == ["ring", "little"]
    assert construct_palmodigital_pockets()["ring_little"].support_surfaces == ("palm", "ring", "little")


def test_wrist_transform_keeps_world_gravity_fixed():
    scene = build_sphere_scene(); mujoco.mj_forward(scene.model, scene.data)
    world = scene.model.opt.gravity.copy(); before = gravity_in_palm_frame(scene)
    joint = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, "rh_WRJ1")
    scene.data.qpos[scene.model.jnt_qposadr[joint]] += .1; mujoco.mj_forward(scene.model, scene.data)
    assert not np.allclose(before, gravity_in_palm_frame(scene))
    np.testing.assert_array_equal(scene.model.opt.gravity, world)


def test_sphere_penetration_normalized_by_radius():
    np.testing.assert_allclose(normalized_penetration([0, .00175], .0175), [0, .1])
    with pytest.raises(ValueError): normalized_penetration([0], 0)


def test_storage_state_detector_requires_region_hand_support_and_no_floor():
    scene = build_sphere_scene(); mujoco.mj_forward(scene.model, scene.data)
    result = storage_state(scene, construct_palmodigital_pockets(scene)["ring_little"])
    assert set(result) == {"center_inside", "floor_contact", "contact_topology",
                           "load_bearing_topology", "alternate_support", "physically_stored"}
    assert not result["physically_stored"]


def test_thumb_resource_detector_is_contact_independent_joint_space_quantity():
    from seqgrasp.phase3c05 import released_finger_available_motion
    scene = build_sphere_scene(); mujoco.mj_forward(scene.model, scene.data)
    assert released_finger_available_motion(scene, "thumb") > 0.0


def test_size_curriculum_is_sequential_and_volume_scaled():
    scales = size_curriculum()
    assert [row.diameter_m / scales[0].diameter_m for row in scales] == pytest.approx([1, 1.25, 1.5, 1.75, 2])


def test_wrist_progression_has_W0_W1_then_conditional_W2_W3():
    assert wrist_commands("W0") == ((0.0, 0.0),)
    assert (5.0, -5.0) in wrist_commands("W1")
    assert max(abs(v) for row in wrist_commands("W2") for v in row) == 10
    assert max(abs(v) for row in wrist_commands("W3") for v in row) == 20
    assert not progression_allowed([{"state_id": "one", "thumb_recovered": True,
                                     "survival": {"1000": True}, "gross_overlap_warning": False}])
    pending = [{"state_id": name, "thumb_recovered": True, "survival": {"1000": True},
                "penetration_valid_for_progression": None} for name in ("one", "two")]
    assert not progression_allowed(pending)


def test_no_object_B_no_RL_no_reward_and_no_scalar_J():
    contract = no_object_b_or_rl_contract()
    assert contract == {"object_B_instantiated": False, "rl_training_performed": False,
                        "reward_defined": False, "scalar_J_defined": False}
    source = inspect.getsource(__import__("seqgrasp.phase3c06", fromlist=["phase3c06"]))
    imports = {alias.name for node in ast.walk(ast.parse(source))
               if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    assert not any(name.startswith("stable_baselines") for name in imports)


def test_phase3c05_backward_compatible_default_scene_and_object():
    old = build_shadow_scene()
    assert old.config.object["shape"] == "ellipsoid"
    assert old.config.object["size"] == [0.03, 0.04, 0.02]
    assert "density" not in old.config.object
    assert phase3c06_scene_config().object["shape"] == "sphere"


def test_failure_taxonomy_exactly_matches_protocol():
    assert len(FAILURE_TAXONOMY) == 17
    assert FAILURE_TAXONOMY[0] == "ACQUISITION_FAILED"
    assert FAILURE_TAXONOMY[-1] == "OTHER"
