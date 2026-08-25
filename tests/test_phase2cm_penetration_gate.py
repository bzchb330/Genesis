from __future__ import annotations

import mujoco
import numpy as np

from seqgrasp.config import load_configs
from seqgrasp.experiments.phase2cm import construct_contact_variant
from seqgrasp.experiments.phase2cm_penetration import (
    _code_definition,
    _distribution,
    classify_b_contact,
    measure_b_penetration,
)


TRIAL = {"wrist_pose": {"relative_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}}


def test_penetration_pair_classification_distinguishes_intended_tip_table_and_object():
    base = load_configs(scene_filename="scene_two_object_half_scale.yaml")
    cfg, model, _ = construct_contact_variant(base, TRIAL, 3)
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_b_geom")
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ff_tip_collision")
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    object_a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_a_geom")
    tip = classify_b_contact(model, cfg, index, b)
    assert tip["bucket"] == "index-B"
    assert tip["finger"] == "index"
    assert tip["is_configured_fingertip"]
    assert classify_b_contact(model, cfg, b, table)["bucket"] == "B-table"
    assert classify_b_contact(model, cfg, object_a, b)["bucket"] == "B-A"


def test_actual_intended_fingertip_overlap_is_included_in_measured_penetration():
    base = load_configs(scene_filename="scene_two_object_half_scale.yaml")
    cfg, model, data = construct_contact_variant(base, TRIAL, 3)
    tip_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ff_tip")
    b_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_b_free")
    qadr = model.jnt_qposadr[b_joint]
    mujoco.mj_forward(model, data)
    data.qpos[qadr:qadr + 3] = data.xpos[tip_body]
    data.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    measured = measure_b_penetration(model, data, cfg)
    assert measured["maximum_penetration_m"] > 0.003
    assert measured["per_pair_maximum_penetration_m"]["index-B"] > 0.003
    assert measured["responsible_contact"]["is_configured_fingertip"]


def test_phase2h_scope_and_penetration_bins_are_explicit_and_boundary_stable():
    definitions = _code_definition()
    assert definitions["threshold_m"] == 0.003
    assert definitions["phase2W"]["included_pairs"]["B-table"]
    assert definitions["phase2H"]["included_pairs"]["index-B"]
    assert not definitions["phase2H"]["included_pairs"]["palm-B"]
    assert not definitions["phase2H"]["pre_release_accumulated"]
    assert definitions["phase2CM_primary"]["initial_boundary_checked"] is False
    bins = _distribution([0.001, 0.002, 0.003, 0.004, 0.005, 0.006])["bins"]
    assert list(bins.values()) == [1, 1, 1, 1, 1, 1]
