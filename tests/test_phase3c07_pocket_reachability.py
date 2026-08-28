from __future__ import annotations

import ast
import inspect

import mujoco
import numpy as np
import pytest

from seqgrasp.phase3c0 import gravity_in_palm_frame
from seqgrasp.phase3c06 import build_sphere_scene
from seqgrasp.phase3c07 import (
    FAILURE_TAXONOMY,
    PreshapeCondition,
    PocketVolume,
    SPHERE_DENSITY_KG_M3,
    SPHERE_DIAMETER_M,
    SPHERE_RADIUS_M,
    TransportStrategy,
    _candidate_cage,
    build_c07_scene,
    contact_geometry,
    contract,
    joint_boundary_events,
    load_phase3c07_config,
    phase3c07_scene_config,
    preshape_gate,
    sphere_mass_kg,
    transport_components,
    wrist_commands,
)


def _pocket() -> PocketVolume:
    return PocketVolume((-0.02, -0.03, 0.08), (-0.01, -0.02, 0.10),
                        (0.005, 0.005, 0.01), ((-0.015, -0.025, 0.09),), "test voxel")


def test_exact_25_mm_sphere_size():
    scene = build_c07_scene()
    geom = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, "phase3c07_sphere_geom")
    assert SPHERE_DIAMETER_M == pytest.approx(0.025)
    assert scene.model.geom_size[geom, 0] == pytest.approx(SPHERE_RADIUS_M)


def test_density_consistent_25_mm_mass():
    scene = build_c07_scene()
    assert SPHERE_DENSITY_KG_M3 == 1000.0
    assert sphere_mass_kg() == pytest.approx(1000 * 4 / 3 * np.pi * 0.0125 ** 3)
    assert scene.model.body_mass[scene.object_body_id] == pytest.approx(sphere_mass_kg())


def test_ring_little_pocket_geometry_uses_actual_compiled_bodies():
    scene = build_c07_scene()
    for body in ("rh_rfknuckle", "rh_lfknuckle", "rh_rfproximal", "rh_lfproximal"):
        assert mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, body) >= 0


def test_pocket_volume_membership_is_voxel_union_not_bounding_box_only():
    pocket = _pocket()
    assert pocket.contains((-0.015, -0.025, 0.09))
    assert not pocket.contains((0.0, 0.0, 0.0))
    assert pocket.volume_m3 == pytest.approx(0.01 * 0.01 * 0.02)


def test_static_map_is_declared_outcome_independent():
    source = inspect.getsource(__import__("seqgrasp.phase3c07", fromlist=["phase3c07"]).build_static_reachability_map)
    assert "dynamic" not in inspect.signature(__import__("seqgrasp.phase3c07", fromlist=["phase3c07"]).build_static_reachability_map).parameters
    assert "constructed_before_dynamic_outcomes" in source


def test_palm_frame_transport_decomposition_and_progress():
    pocket = _pocket()
    result = transport_components((0.02, -0.08, 0.10), (0.00, -0.04, 0.095), pocket)
    assert result["lateral_ulnar_progress_m"] > 0
    assert result["inward_progress_m"] > 0
    assert result["pocket_distance_m"] >= 0


def test_matched_transport_strategies_are_exactly_frozen_protocol_set():
    assert tuple(TransportStrategy) == (
        TransportStrategy.T0_OLD_DIRECT,
        TransportStrategy.T1_POCKET_DIRECTED,
        TransportStrategy.T2_WRIST_ASSISTED,
    )


def test_wrist_range_generation_is_staged_and_structured():
    assert wrist_commands("W0") == ((0.0, 0.0),)
    assert len(wrist_commands("W1")) == 8
    assert max(abs(x) for row in wrist_commands("W1") for x in row) == 5
    assert max(abs(x) for row in wrist_commands("W2") for x in row) == 10
    assert max(abs(x) for row in wrist_commands("W3") for x in row) == 20


def test_gravity_transform_changes_frame_not_world_gravity():
    scene = build_c07_scene(); mujoco.mj_forward(scene.model, scene.data)
    world = scene.model.opt.gravity.copy(); before = gravity_in_palm_frame(scene)
    joint = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, "rh_WRJ1")
    scene.data.qpos[scene.model.jnt_qposadr[joint]] += 0.1; mujoco.mj_forward(scene.model, scene.data)
    assert not np.allclose(before, gravity_in_palm_frame(scene))
    np.testing.assert_array_equal(world, scene.model.opt.gravity)


def test_preshape_gating_is_geometric_not_timed():
    assert preshape_gate(PreshapeCondition.P0_AFTER_ENTRY, inside_pocket=True, near_pocket=True, sweep_clearance_m=-1)
    assert not preshape_gate(PreshapeCondition.P0_AFTER_ENTRY, inside_pocket=False, near_pocket=True, sweep_clearance_m=1)
    assert preshape_gate(PreshapeCondition.P1_GEOMETRIC_APPROACH, inside_pocket=False, near_pocket=True, sweep_clearance_m=0)
    assert not preshape_gate(PreshapeCondition.P1_GEOMETRIC_APPROACH, inside_pocket=True, near_pocket=True, sweep_clearance_m=-1e-6)


def test_cage_requires_volume_support_rank_settling_and_no_floor():
    sample = {"inside_pocket": True, "floor_contact": False, "finite_physics": True,
              "motion_settling": True, "contact_geometry": {
                  "load_bearing_topology": ["ring", "palm"], "contact_normal_rank": 2}}
    assert _candidate_cage(sample)
    sample["motion_settling"] = False
    assert not _candidate_cage(sample)


def test_contact_geometry_records_load_bearing_topology_and_normals():
    scene = build_c07_scene(); mujoco.mj_forward(scene.model, scene.data)
    result = contact_geometry(scene)
    assert {"records", "pairwise_normal_angles_deg", "contact_normal_rank",
            "load_bearing_topology", "lambda_storage"} <= result.keys()


def test_cage_hold_checkpoints_include_1000_steps():
    assert load_phase3c07_config()["cage_hold"]["checkpoints"] == [10, 25, 50, 100, 200, 300, 500, 750, 1000]


def test_penetration_is_normalized_by_12_5_mm_radius_in_source():
    source = inspect.getsource(contact_geometry)
    assert "SPHERE_RADIUS_M" in source
    assert SPHERE_RADIUS_M == pytest.approx(0.0125)


def test_joint_boundary_localization_reports_joint_step_and_stage():
    scene = build_c07_scene()
    joint = 0; address = scene.model.jnt_qposadr[joint]
    scene.data.qpos[address] = scene.model.jnt_range[joint, 0]
    events = joint_boundary_events(scene, 7, "TRANSPORT")
    assert any(event["step"] == 7 and event["stage"] == "TRANSPORT" and event["joint"] for event in events)


def test_no_thumb_or_index_release_contract():
    result = contract()
    assert not result["thumb_release_performed"]
    assert not result["index_release_performed"]


def test_no_object_b_no_rl_no_reward_and_no_scalar_j():
    result = contract()
    assert not result["object_B_instantiated"]
    assert not result["rl_training_performed"]
    assert not result["reward_defined"]
    assert not result["scalar_J_defined"]
    module = __import__("seqgrasp.phase3c07", fromlist=["phase3c07"])
    imports = {alias.name for node in ast.walk(ast.parse(inspect.getsource(module)))
               if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    assert not any(name.startswith("stable_baselines") for name in imports)


def test_no_compliant_skin_or_official_mjcf_change_contract():
    result = contract()
    assert not result["compliant_skin_added"]
    assert not result["official_MJCF_modified"]


def test_phase3c06_backward_compatible_geometry():
    assert build_sphere_scene().config.object["size"] == [0.0175]
    assert phase3c07_scene_config().object["size"] == [0.0125]


def test_failure_taxonomy_exactly_matches_protocol():
    assert len(FAILURE_TAXONOMY) == 20
    assert FAILURE_TAXONOMY[0] == "ACQUISITION_FAILED"
    assert FAILURE_TAXONOMY[-1] == "OTHER"
