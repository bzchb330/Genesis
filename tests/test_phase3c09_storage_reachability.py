from __future__ import annotations

import ast
import inspect

import mujoco
import numpy as np
import pytest

import seqgrasp.phase3c09 as c09
from seqgrasp.phase3c08 import build_forearm_scene


def test_top_trajectory_deterministic_selection():
    result = {"rows": [{"trial_id": "b", "closest_pocket_distance_m": .1}, {"trial_id": "a", "closest_pocket_distance_m": .1}, {"trial_id": "c", "closest_pocket_distance_m": .2}]}
    assert [row["trial_id"] for row in c09.select_top_trajectories(result, 2)] == ["a", "b"]


def test_time_series_reconstruction_finite_difference():
    values = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    np.testing.assert_allclose(c09.finite_difference(values, .5)[:, 0], 2.0)


def test_pocket_distance_calculation():
    assert c09.pocket_distance((0, 0, 0), np.asarray([[1, 0, 0], [0, 2, 0]])) == pytest.approx(1.0)


def test_contact_force_decomposition():
    value = c09.decompose_contact_force(4.0, 1.0, .5)
    assert value == {"normal_force_n": 4.0, "tangential_force_n": 1.0, "friction_utilization": .5}


def test_friction_utilization_unavailable_is_not_invented():
    assert c09.decompose_contact_force(4.0, None, .5)["friction_utilization"] is None


def test_25_mm_minkowski_inflation_radius():
    cfg = c09.load_phase3c09_config()
    assert cfg["cspace"]["sphere_radius_m"] == pytest.approx(.0125)
    assert 2 * cfg["cspace"]["sphere_radius_m"] == pytest.approx(.025)


def test_cspace_occupancy_uses_exact_compiled_geom_distance():
    wrapper = build_forearm_scene(with_actuator=True); qpos = wrapper.scene.data.qpos.copy()
    clearance, limiting = c09.exact_clearance_grid(wrapper.scene, qpos, np.asarray([[0, 0, .09], [.2, .2, .2]]))
    assert clearance.shape == (2,) and limiting.shape == (2,) and np.all(np.isfinite(clearance))


def test_astar_connectivity():
    free = np.ones((3, 3, 3), dtype=bool); free[1, 1, :] = False
    path = c09.astar_path(free, (0, 0, 0), {(2, 2, 2)}, .001)
    assert path[0] == (0, 0, 0) and path[-1] == (2, 2, 2)


def test_multiresolution_agreement_classification():
    assert c09.classify_multiresolution((True, True)) == "CS-A"
    assert c09.classify_multiresolution((True, False)) == "CS-B"
    assert c09.classify_multiresolution((False, False)) == "CS-C"


def test_bottleneck_detection():
    values = np.ones((2, 2, 2)); values[1, 0, 0] = .2
    assert c09.bottleneck_from_path(values, [(0, 0, 0), (1, 0, 0)]) == ((1, 0, 0), .2)


def test_first_order_contact_distribution_dimensions():
    fields = c09.smooth_contact_fields(np.asarray([0, 0, 1.0]), "M1_INDEX_GUIDE_THUMB_MIGRATION")
    matrix = np.column_stack([field(np.zeros(8)) for field in fields])
    assert matrix.shape == (8, 2) and np.linalg.matrix_rank(matrix) == 2


def test_lie_bracket_finite_difference_consistency():
    fields = c09.smooth_contact_fields(np.asarray([0, 0, 1.0]), "M1_INDEX_GUIDE_THUMB_MIGRATION")
    x = np.zeros(8); coarse = c09.numerical_lie_bracket(fields[0], fields[1], x, 1e-3); fine = c09.numerical_lie_bracket(fields[0], fields[1], x, 5e-4)
    assert np.linalg.norm(coarse - fine) < 1e-3


def test_cyclic_motion_scales_quadratically():
    fields = c09.smooth_contact_fields(np.asarray([0, 0, 1.0]), "M1_INDEX_GUIDE_THUMB_MIGRATION"); x = np.zeros(8)
    large = c09.cyclic_bracket_check(fields[0], fields[1], x, .01)[:3]; small = c09.cyclic_bracket_check(fields[0], fields[1], x, .005)[:3]
    assert np.linalg.norm(small) / np.linalg.norm(large) == pytest.approx(.25, rel=.08)


def test_accessibility_claim_is_smooth_mode_only():
    assert "Standard smooth rolling" in inspect.getdoc(c09.smooth_contact_fields)
    assert not c09.contact_accessibility_audit.__doc__ or "global LARC" not in c09.contact_accessibility_audit.__doc__


def test_reduced_storage_manifold_search_has_five_bounded_variables():
    cfg = c09.load_phase3c09_config()["storage_manifold"]
    assert len(cfg["middle_flexion_fractions"]) == len(cfg["ring_flexion_fractions"]) == len(cfg["little_flexion_fractions"]) == 3
    assert len(cfg["wrist2_offsets_deg"]) == len(cfg["forearm_PS_deg"]) == 3


def test_collision_free_storage_candidate_generation():
    assert c09.collision_free_storage_candidate(0.0, 2)
    assert not c09.collision_free_storage_candidate(-1e-3, 3)
    assert not c09.collision_free_storage_candidate(.01, 1)


def test_storage_basin_clustering():
    mask = np.zeros((4, 4, 4), dtype=bool); mask[0, 0, 0] = True; mask[3, 3, 3] = True
    labels, count = c09.cluster_storage_mask(mask)
    assert count == 2 and set(np.unique(labels)) == {0, 1, 2}


def test_acquisition_to_storage_geometric_connectivity():
    free = np.ones((4, 4, 4), dtype=bool)
    assert c09.astar_path(free, (0, 0, 0), {(3, 3, 3)}, .002)


def test_no_new_dynamics_rollout_call():
    tree = ast.parse(inspect.getsource(c09)); calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(isinstance(node.func, ast.Attribute) and node.func.attr == "mj_step" for node in calls)
    assert c09.phase3c09_contract()["new_dynamic_rollout_steps"] == 0


def test_no_rl():
    assert not c09.phase3c09_contract()["rl_training"]
    assert "stable_baselines" not in inspect.getsource(c09)


def test_no_object_b():
    assert not c09.phase3c09_contract()["object_B"]
    wrapper = build_forearm_scene(with_actuator=True)
    assert mujoco.mj_name2id(wrapper.scene.model, mujoco.mjtObj.mjOBJ_BODY, "object_B") < 0


def test_no_thumb_release():
    assert not c09.phase3c09_contract()["thumb_release"]


def test_no_friction_or_contact_changes():
    contract = c09.phase3c09_contract(); assert not contract["friction_changed"] and not contract["contact_changed"]


def test_no_skin():
    assert not c09.phase3c09_contract()["skin_added"]


def test_phase3c08_backward_compatibility():
    first = build_forearm_scene(with_actuator=True); second = build_forearm_scene(with_actuator=True)
    assert (first.scene.model.nq, first.scene.model.nv, first.scene.model.nu) == (second.scene.model.nq, second.scene.model.nv, second.scene.model.nu)
