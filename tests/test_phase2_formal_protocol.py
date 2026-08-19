from types import SimpleNamespace

from seqgrasp.config import load_configs
from seqgrasp.experiments.b_workspace import stratified_representative_ids
from seqgrasp.experiments.second_grasp import (
    classify_B_acquisition,
    correlation_trial_id,
    fixture_is_active,
    formal_nonpilot_records,
    placement_candidate,
)
from seqgrasp.phase2_config import load_phase2_config
from seqgrasp.experiments.resumable import IncrementalJsonlStore, stable_trial_id


def _criteria():
    return SimpleNamespace(
        minimum_B_free_finger_contacts=1,
        minimum_B_hand_contacts=1,
        minimum_B_normal_force_N=0.20,
        maximum_penetration_m=0.003,
        maximum_B_translation_m=0.005,
        maximum_B_orientation_rad=0.20,
    )


def test_B_fixture_activation_and_release_boundary():
    assert fixture_is_active(899, 900)
    assert not fixture_is_active(900, 900)
    assert not fixture_is_active(1399, 900)


def test_functional_B_acquisition_requires_free_finger_and_all_retention_terms():
    valid = dict(
        fixture_released=True, final_free_finger_contacts=1,
        final_hand_contacts=1, final_hand_normal_force_N=0.201,
        table_contact=False, complete_hand_contact_loss=False,
        maximum_penetration_m=0.003, maximum_translation_m=0.005,
        maximum_orientation_rad=0.20, numerically_stable=True,
        criteria=_criteria(),
    )
    assert classify_B_acquisition(**valid)
    for key, failing in (
        ("fixture_released", False), ("final_free_finger_contacts", 0),
        ("final_hand_contacts", 0), ("final_hand_normal_force_N", 0.20),
        ("table_contact", True), ("complete_hand_contact_loss", True),
        ("maximum_penetration_m", 0.00301), ("maximum_translation_m", 0.00501),
        ("maximum_orientation_rad", 0.2001), ("numerically_stable", False),
    ):
        changed = dict(valid); changed[key] = failing
        assert not classify_B_acquisition(**changed)


def test_global_B_distribution_and_sampling_are_fixed_and_deterministic():
    cfg = load_configs()
    phase2, _ = load_phase2_config()
    first = placement_candidate(cfg, phase2.second_grasp, 19)
    second = placement_candidate(cfg, phase2.second_grasp, 19)
    assert first == second
    assert phase2.second_grasp.B_center_x_bounds_m == [0.055, 0.065]
    assert phase2.second_grasp.B_center_y_bounds_m == [0.115, 0.125]
    assert phase2.second_grasp.B_center_z_bounds_m == [0.215, 0.225]


def test_pilot_formal_separation_and_trial_ids():
    rows = [{"trial_id": "p", "pilot_only": True}, {"trial_id": "f", "pilot_only": False}]
    assert formal_nonpilot_records(rows) == [rows[1]]
    record = {"grasp_id": "g", "config_hash": "h"}
    assert correlation_trial_id(record, 2, True) != correlation_trial_id(record, 2, False)
    assert correlation_trial_id(record, 2, False) == correlation_trial_id(record, 2, False)


def test_geometry_stratification_is_reproducible_and_covers_occupied_counts():
    accepted = [
        {"grasp_id": f"g{i:02d}", "occupied_finger_count": 2 + i % 3, "ferrari_canny_epsilon": i / 100}
        for i in range(30)
    ]
    resources = [
        {"grasp_id": f"g{i:02d}", "occupied_finger_count": 2 + i % 3,
         "free_finger_workspace_vol_m3": i * 1e-6, "free_palm_volume_m3": 0.003 + i * 1e-7}
        for i in range(30)
    ]
    first = stratified_representative_ids(accepted, resources, 20)
    assert first == stratified_representative_ids(accepted, resources, 20)
    selected_counts = {accepted[int(grasp_id[1:])]["occupied_finger_count"] for grasp_id in first}
    assert selected_counts == {2, 3, 4}


def test_dataset_extension_and_formal_stores_resume_without_duplicates(tmp_path):
    extension = IncrementalJsonlStore(tmp_path / "extension_candidate_attempts.jsonl", 1.0, 0.01)
    attempt_id = stable_trial_id("phase2-grasp-extension-attempt", {"extension_attempt_index": 4})
    extension.append({"trial_id": attempt_id, "extension_attempt_index": 4})
    extension.append({"trial_id": attempt_id, "extension_attempt_index": 4})
    assert extension.completed_ids() == {attempt_id}
    formal = IncrementalJsonlStore(tmp_path / "formal" / "trials.jsonl", 1.0, 0.01)
    record = {"grasp_id": "g", "config_hash": "h"}
    trial_id = correlation_trial_id(record, 0, False)
    formal.append({"trial_id": trial_id, "pilot_only": False})
    assert formal.completed_ids() == {trial_id}
