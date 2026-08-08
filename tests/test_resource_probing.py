import mujoco
import numpy as np

from seqgrasp import load_configs
from seqgrasp.config import ROOT
from seqgrasp.control import resolve_hand_indices
from seqgrasp.diagnostics import (
    load_grasp_profile,
    reachability_cloud,
    resource_rows,
    run_b_probe,
    run_scripted_grasp,
)
from seqgrasp.env.resource import compute_resource_metric
from seqgrasp.scene_builder import build_scene
from seqgrasp.sensing.contact import extract_contacts, group_contacts_by_finger


PROFILE_PATH = ROOT / "configs" / "grasps" / "resource_grasp_A_01.yaml"


def _profile():
    payload, profile = load_grasp_profile(PROFILE_PATH)
    assert payload["engineering_only"] is True
    assert payload["scientific_success_assigned"] is False
    return profile


def test_selected_resource_profiles_load_and_leave_metric_undefined():
    for number in (1, 2, 3):
        path = ROOT / "configs" / "grasps" / f"resource_grasp_A_{number:02d}.yaml"
        payload, profile = load_grasp_profile(path)
        assert payload["engineering_only"] is True
        assert payload["selection_metadata"]["retained_all_20_engineering_windows"] is True
        assert set(profile.closed_joint_fractions) == set(load_configs().hand.actuator_names)
    assert compute_resource_metric(None, load_configs()) is None


def test_raw_resource_export_dimensions_order_and_units():
    cfg = load_configs()
    run = run_scripted_grasp(cfg, seed=3, save_outputs=False)
    rows = resource_rows("test_grasp", 3, run, stride=17)
    assert rows
    first = rows[0]
    assert list(first)[:4] == ["grasp", "seed", "time_s", "time_after_release_s"]
    assert sum(key.startswith("q_") and key.endswith("_rad") for key in first) == cfg.hand.dof_count
    assert sum(key.startswith("qdot_") and key.endswith("_rad_s") for key in first) == cfg.hand.dof_count
    assert sum(
        key.startswith("control_")
        and not key.startswith(("control_lower_", "control_upper_"))
        and key.endswith("_Nm")
        for key in first
    ) == cfg.hand.dof_count
    assert sum(key.startswith("A_normal_force_") and key.endswith("_N") for key in first) == len(cfg.hand.finger_geom_mapping)
    assert all(set(row) == set(first) for row in rows)


def test_reachability_cloud_is_seed_deterministic_and_well_shaped():
    cfg = load_configs()
    profile = _profile()
    first = reachability_cloud(cfg, profile, "index", samples=8, amplitude=0.12, seed=11)
    second = reachability_cloud(cfg, profile, "index", samples=8, amplitude=0.12, seed=11)
    assert first["columns"] == second["columns"]
    assert first["points"].shape == (8, 12)
    assert first["targets"].shape == (8, cfg.hand.dof_count)
    np.testing.assert_array_equal(first["points"], second["points"])
    np.testing.assert_array_equal(first["targets"], second["targets"])


def test_object_b_contact_detection_and_separation():
    cfg = load_configs()
    model, data = build_scene(cfg)
    profile = _profile()
    indices = resolve_hand_indices(model, cfg.hand)
    limits = model.jnt_range[indices.joint_ids]
    fractions = np.asarray([profile.closed_joint_fractions[name] for name in cfg.hand.actuator_names])
    data.qpos[indices.qpos_addresses] = limits[:, 0] + fractions * (limits[:, 1] - limits[:, 0])
    mujoco.mj_forward(model, data)

    finger = "index"
    tip_name = cfg.hand.finger_geom_mapping[finger][0]
    tip_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, tip_name)
    b_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_b_free")
    b_qpos = model.jnt_qposadr[b_joint]
    data.qpos[b_qpos:b_qpos + 3] = data.geom_xpos[tip_geom] + np.array([0.02, 0.0, 0.0])
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    grouped = group_contacts_by_finger(extract_contacts(model, data), cfg.hand.finger_geom_mapping)
    contacts = [record for record in grouped[finger] if "object_b" in {record.body1_name, record.body2_name}]
    assert contacts
    assert all(record.distance <= 0.0 for record in contacts)

    data.qpos[b_qpos:b_qpos + 3] = np.array([1.0, 1.0, 1.0])
    mujoco.mj_forward(model, data)
    grouped = group_contacts_by_finger(extract_contacts(model, data), cfg.hand.finger_geom_mapping)
    assert not [record for record in grouped[finger] if "object_b" in {record.body1_name, record.body2_name}]


def test_b_probe_logs_raw_a_disturbance_without_assigning_success():
    cfg = load_configs()
    run, metrics = run_b_probe(
        cfg,
        "resource_grasp_A_01",
        _profile(),
        "index",
        seed=2,
        samples=4,
        amplitude=0.12,
    )
    assert run.arrays["finger_b_contact_count"].shape[1] == len(cfg.hand.finger_geom_mapping)
    assert metrics["engineering_only"] is True
    assert metrics["scientific_success_assigned"] is False
    for key in (
        "A_maximum_translation_m",
        "A_maximum_rotation_rad",
        "A_vertical_displacement_m",
        "A_maximum_force_redistribution_N",
        "A_contact_pattern_changes",
        "A_complete_contact_loss_event",
        "A_table_contact_event",
    ):
        assert key in metrics
