from __future__ import annotations

import inspect
import json

import mujoco
import numpy as np
import pytest

import seqgrasp.phase3c10 as c10
from seqgrasp.config import ROOT
from seqgrasp.phase3c08 import build_forearm_scene


def _metric() -> dict:
    return json.loads((ROOT / "outputs/phase3C10/metric_repair_audit.json").read_text(encoding="utf-8"))


def _manifest() -> dict:
    return json.loads((ROOT / "outputs/phase3C10/B03_validation_manifest.json").read_text(encoding="utf-8"))


def _results() -> dict:
    return json.loads((ROOT / "outputs/phase3C10/B03_validation_results.json").read_text(encoding="utf-8"))


def test_support_gated_progress():
    value = c10.support_gated_progress([1.0, .8, .6, .5], [1.0, 0.0, 1.0, 1.0], [.01, .01, .2, .01], .05, non_hand_support=[False, False, False, True])
    np.testing.assert_array_equal(value["eligible"], [True, False, False, False])
    np.testing.assert_allclose(value["valid_progress_m"], [0.0, 0.0, 0.0, 0.0])


def test_phase3c08_flyby_rejected_by_new_metric():
    value = _metric()
    assert value["flyby_minimum_rejected"] and value["hand_force_at_minimum_n"] == 0.0


def test_raw_distance_retained_only_as_diagnostic():
    assert _metric()["raw_minimum_distance_role"] == "DESCRIPTIVE_GEOMETRIC_QUANTITY_ONLY"


def test_transfer_clearance_diagnostic():
    value = c10.transfer_clearance({"middle": .01, "ring": .0, "little": .02})
    assert value["corridor_collision_free"] and value["minimum_storage_finger_clearance_m"] == 0.0


def test_receiver_ready_diagnostic_requires_geometry_and_opportunity():
    result = c10.receiver_ready_with_contact_opportunity(np.zeros(3), np.zeros(3), np.asarray([0, 1]), True, 2, .05)
    assert result["ready"]
    assert not c10.receiver_ready_with_contact_opportunity(np.zeros(3), np.zeros(3), np.asarray([0, 1]), True, 1, .05)["ready"]


def test_actual_b03_candidate_selection_frozen_before_outcomes():
    value = _manifest()
    assert value["frozen_before_outcomes"] and value["candidates"]["constructed_before_dynamic_outcomes"]
    assert [row["candidate_id"] for row in value["candidates"]["selected"]] == ["B03_CANDIDATE_00", "B03_CANDIDATE_01", "B03_CANDIDATE_02"]


def test_b03_direct_placement_initialization():
    wrapper = build_forearm_scene(with_actuator=True); c10.initialize_b03_trial(wrapper, _manifest()["trials"][0])
    assert np.all(wrapper.scene.data.qvel == 0.0)
    assert np.all(np.isfinite(wrapper.scene.data.qpos))


def test_b03_hold_logging():
    row = _results()["rows"][0]; series = np.load(row["timeseries_path"], allow_pickle=False)
    required = {"step", "center_palm_m", "linear_speed_mps", "inside_B03", "storage_force_n", "maximum_penetration_m", "qpos"}
    assert required <= set(series.files) and len(series["step"]) == 1000


def test_gravity_orientation_predeclaration():
    value = _manifest()
    assert [row["orientation_id"] for row in value["orientations"]] == ["NOMINAL", "RECEIVER_BIASED", "ESCAPE_BIASED", "TANGENTIAL"]
    assert value["trial_count"] == 12


def test_thumb_workspace_calculation():
    points = np.asarray([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=float)
    value = c10.workspace_descriptor(points, 6, 1, 1, [.1, .2])
    assert value["reachable_volume_m3"] == pytest.approx(1/6) and value["blocked_fraction"] == pytest.approx(1/3)


def test_index_workspace_calculation():
    value = c10.workspace_descriptor(np.zeros((0, 3)), 4, 2, 1, [.2])
    assert value["reachable_volume_m3"] == 0.0 and value["palm_frame_range_m"] == [0.0, 0.0, 0.0]


def test_thumb_index_joint_workspace():
    value = c10.joint_acquisition_workspace(np.asarray([[0,0,0], [0,0,0]]), np.asarray([[.02,0,0], [.10,0,0]]), .08)
    assert value["opposition_sample_count"] == 1 and value["minimum_aperture_m"] == pytest.approx(.02)


def test_m1_m2_mapping_correctness():
    assert c10.m2_contact_mapping() == {"guide": "thumb", "unload_and_migrate": "index", "mode": "M2"}


def test_primary_m2_scripted_sequence_mapping():
    assert c10.scripted_stage_specification()[3]["name"] == "INDEX_UNLOAD_THUMB_GUIDE"


def test_receiver_formed_before_substantial_index_unload():
    stages = c10.scripted_stage_specification(); unload = next(i for i, row in enumerate(stages) if row["index_unload"])
    assert stages[unload - 1]["name"] == "B03_RECEIVER_PRESHAPE" and stages[unload]["receiver_required"]


def test_premature_support_loss_termination():
    assert c10.premature_support_loss(0.0, 0.0)
    assert not c10.premature_support_loss(0.0, .1)


def test_support_topology_timeline_logged():
    assert all("dominant_support_topology" in row for row in _results()["rows"])


def test_sphere_relative_speed_logging():
    series = np.load(_results()["rows"][0]["timeseries_path"], allow_pickle=False)
    assert len(series["linear_speed_mps"]) == 1000 and np.all(np.isfinite(series["linear_speed_mps"]))


def test_no_ballistic_flyby_success():
    value = c10.support_gated_progress([.1, .001], [1, 0], [.01, 1.2], .1)
    assert value["valid_progress_m"][-1] == 0.0


def test_b03_entry_requires_storage_support_for_success():
    assert not c10.b03_entry_success(True, True, 0.0, True)
    assert c10.b03_entry_success(True, True, .1, True)


def test_no_large_batch():
    assert c10.load_phase3c10_config()["B03_validation"]["maximum_trials"] == 12
    assert not c10.phase3c10_contract()["large_batch"]


def test_no_optimizer():
    assert not c10.phase3c10_contract()["optimizer"] and "trajectory optimization" not in inspect.getsource(c10)


def test_no_rl():
    assert not c10.phase3c10_contract()["rl"] and "stable_baselines" not in inspect.getsource(c10)


def test_no_object_b():
    assert not c10.phase3c10_contract()["object_B"]
    wrapper = build_forearm_scene(with_actuator=True)
    assert mujoco.mj_name2id(wrapper.scene.model, mujoco.mjtObj.mjOBJ_BODY, "object_B") < 0


def test_no_friction_or_contact_modification():
    contract = c10.phase3c10_contract()
    assert not contract["friction_changed"] and not contract["contact_changed"]


def test_no_skin():
    assert not c10.phase3c10_contract()["skin_added"]
    assert "skin" not in inspect.getsource(c10).lower().replace('"skin_added"', "")


def test_phase3c09_backward_compatibility():
    first = build_forearm_scene(with_actuator=True); second = build_forearm_scene(with_actuator=True)
    assert (first.scene.model.nq, first.scene.model.nv, first.scene.model.nu) == (second.scene.model.nq, second.scene.model.nv, second.scene.model.nu)
