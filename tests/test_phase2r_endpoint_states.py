from types import SimpleNamespace

import pytest

from seqgrasp.experiments.palmar_grasp_sampling import sample_palmar_candidate
from seqgrasp.experiments.phase2r import (
    GraspStateType,
    PHASE2R_EXPERIMENT_ID,
    assert_formal_pairing,
    classify_grasp_state,
    digit_precheck_outcome,
    nearest_neighbor_match,
    paired_formal_trial_id,
    split_calibration_states,
    validate_grasp_state_schema,
)
from seqgrasp.experiments.resumable import IncrementalJsonlStore
from seqgrasp.experiments.phase2r_second_grasp import (
    remap_index_thumb_trajectory,
    select_static_acquisition_pair,
)
from seqgrasp.experiments.phase2_5_trajectory import BAcquisitionTrajectory
from seqgrasp.phase2r_config import load_phase2r_config


def _state_config():
    return SimpleNamespace(
        maximum_penetration_m=0.003,
        maximum_translation_drift_m=0.005,
        maximum_orientation_drift_rad=0.20,
        minimum_fingertip_contact_fingers=2,
        palm_contact_fraction_minimum=0.80,
        minimum_palmar_load_bearing_fingers=1,
        maximum_palmar_load_bearing_fingers=2,
    )


def _measurement(**updates):
    row = {
        "grasp_state_id": "state-0",
        "object_A_COM_palm_reference_m": [0.0, 0.0, -0.04],
        "object_A_COM_palm_compiled_m": [0.04, 0.0, 0.0],
        "COM_to_palm_origin_distance_m": 0.04,
        "COM_to_palm_surface_distance_m": 0.0,
        "COM_inside_existing_palm_region": True,
        "palm_A_contact": True,
        "palm_A_contact_fraction": 1.0,
        "palm_A_normal_force_N": 1.0,
        "palm_A_contact_count": 1,
        "per_finger_A_contact_flags": [False, True, False, False],
        "per_finger_A_contact_fraction": [0.0, 1.0, 0.0, 0.0],
        "per_finger_distal_contact_fraction": [1.0, 1.0, 0.0, 0.0],
        "per_finger_A_normal_force_N": [0.0, 1.0, 0.0, 0.0],
        "occupied_finger_count": 1,
        "occupied_finger_mask": [False, True, False, False],
        "free_finger_count": 3,
        "ferrari_canny_epsilon": 0.1,
        "A_translation_drift_m": 0.001,
        "A_rotation_drift_rad": 0.01,
        "A_vertical_drift_m": 0.001,
        "maximum_penetration_m": 0.001,
        "minimum_joint_margin_rad": 0.1,
        "maximum_actuator_utilization": 0.5,
        "fixture_removed_before_validation": True,
        "fixture_active_during_validation": False,
        "equality_constraint_count": 0,
        "object_joint_type": "free",
        "table_recontact": False,
        "complete_hand_contact_loss": False,
        "numerically_stable": True,
        "final_joint_configuration_rad": [0.0] * 16,
        "final_object_position_m": [0.04, 0.0, 0.15],
        "final_object_quaternion": [1.0, 0.0, 0.0, 0.0],
        "total_A_normal_force_N": 2.0,
    }
    row.update(updates)
    return row


def test_grasp_state_schema_and_explicit_allowed_types():
    row = classify_grasp_state(_measurement(), GraspStateType.PALMAR_SECURED, _state_config(), 1e-8)
    validate_grasp_state_schema(row)
    with pytest.raises(ValueError):
        validate_grasp_state_schema({**row, "grasp_state_type": "TRANSFERRED"})
    assert PHASE2R_EXPERIMENT_ID == "phase2R_palmar_vs_fingertip_formal"


def test_fingertip_and_palmar_classification_are_endpoint_specific():
    palmar = classify_grasp_state(_measurement(), GraspStateType.PALMAR_SECURED, _state_config(), 1e-8)
    assert palmar["accepted"] and palmar["free_finger_count"] == 3
    fingertip_source = _measurement(
        palm_A_contact=False, palm_A_contact_fraction=0.0, palm_A_contact_count=0,
        palm_A_normal_force_N=0.0, occupied_finger_count=2, free_finger_count=2,
        occupied_finger_mask=[True, True, False, False],
    )
    fingertip = classify_grasp_state(fingertip_source, GraspStateType.FINGERTIP, _state_config(), 1e-8)
    assert fingertip["accepted"]
    assert not classify_grasp_state(
        {**palmar, "palm_A_contact_fraction": 0.79}, GraspStateType.PALMAR_SECURED, _state_config(), 1e-8,
    )["accepted"]


def test_fixture_and_equality_constraints_are_rejected_before_palmar_acceptance():
    for update, reason in (
        ({"fixture_removed_before_validation": False}, "fixture_removed"),
        ({"fixture_active_during_validation": True}, "fixture_removed"),
        ({"equality_constraint_count": 1}, "no_equality_constraint"),
    ):
        row = classify_grasp_state(_measurement(**update), GraspStateType.PALMAR_SECURED, _state_config(), 1e-8)
        assert not row["accepted"] and row["rejection_reason"] == reason


def test_palmar_sampling_is_seed_and_attempt_deterministic():
    cfg, _ = load_phase2r_config()
    first = sample_palmar_candidate(cfg, 17)
    second = sample_palmar_candidate(cfg, 17)
    third = sample_palmar_candidate(cfg, 18)
    assert first == second
    assert first["initial_object_position_m"] != third["initial_object_position_m"]
    assert first["retaining_finger_subset"] in cfg.state.focused_retaining_finger_subsets


def _matching_state(group, index):
    return {
        "grasp_state_id": f"{group}-{index}", "grasp_state_type": group,
        "ferrari_canny_epsilon": index / 10 + (0.001 if group == "PALMAR_SECURED" else 0.0),
        "total_A_normal_force_N": 1.0 + index,
        "A_translation_drift_m": 0.001 + index * 1e-5,
        "A_rotation_drift_rad": 0.01 + index * 1e-4,
        "minimum_joint_margin_rad": 0.1 + index * 1e-3,
    }


def test_matching_is_deterministic_without_state_reuse_and_split_excludes_calibration():
    covariates = [
        "ferrari_canny_epsilon", "total_A_normal_force_N", "A_translation_drift_m",
        "A_rotation_drift_rad", "minimum_joint_margin_rad",
    ]
    finger = [_matching_state("FINGERTIP", i) for i in range(6)]
    palm = [_matching_state("PALMAR_SECURED", i) for i in range(6)]
    calibration, formal = split_calibration_states(finger + palm, 1)
    assert len(calibration) == 2 and not {row["grasp_state_id"] for row in calibration} & {row["grasp_state_id"] for row in formal}
    pairs = nearest_neighbor_match(
        [row for row in formal if row["grasp_state_type"] == "FINGERTIP"],
        [row for row in formal if row["grasp_state_type"] == "PALMAR_SECURED"], covariates, 4,
    )
    repeated = nearest_neighbor_match(
        list(reversed([row for row in formal if row["grasp_state_type"] == "FINGERTIP"])),
        list(reversed([row for row in formal if row["grasp_state_type"] == "PALMAR_SECURED"])), covariates, 4,
    )
    assert pairs == repeated
    ids = [pair.fingertip["grasp_state_id"] for pair in pairs] + [pair.palmar["grasp_state_id"] for pair in pairs]
    assert len(ids) == len(set(ids))


def test_digit_precheck_records_existing_outcome_and_required_subreason():
    row = _measurement(free_finger_count=1, occupied_finger_count=3)
    result = digit_precheck_outcome(row)
    assert result == {
        "outcome": "B_NOT_ACQUIRED",
        "outcome_subreason": "INSUFFICIENT_FREE_DIGITS_PRECHECK",
        "dynamic_attempt_executed": False,
    }
    assert digit_precheck_outcome(_measurement(free_finger_count=2, occupied_finger_count=2)) is None


def test_formal_B_seed_pairing_and_pilot_exclusion():
    rows = []
    for seed in range(2):
        for state_type in ("FINGERTIP", "PALMAR_SECURED"):
            rows.append({"matched_pair_id": "p0", "B_seed_index": seed, "grasp_state_type": state_type})
    assert_formal_pairing(rows, 2)
    with pytest.raises(ValueError):
        assert_formal_pairing([{**rows[0], "pilot_only": True}, *rows[1:]], 2)
    with pytest.raises(ValueError):
        assert_formal_pairing(rows[:-1], 2)
    assert paired_formal_trial_id("p0", "FINGERTIP", 0, 11) == paired_formal_trial_id("p0", "FINGERTIP", 0, 11)
    assert paired_formal_trial_id("p0", "FINGERTIP", 0, 11) != paired_formal_trial_id("p0", "PALMAR_SECURED", 0, 11)


def test_static_acquisition_pair_is_fixed_by_free_digit_geometry_priority():
    assert select_static_acquisition_pair([False, True, True, False]) == ("index", "thumb")
    assert select_static_acquisition_pair([True, False, True, False]) == ("middle", "thumb")
    assert select_static_acquisition_pair([False, False, True, True]) == ("index", "middle")
    assert select_static_acquisition_pair([True, True, False, False]) == ("ring", "thumb")
    assert select_static_acquisition_pair([True, True, True, False]) is None

    source = BAcquisitionTrajectory(1, tuple(range(16)), tuple(range(16)), tuple(range(16)), tuple(range(16)), 10, (1, 2, 3, 4), 5)
    mapped = remap_index_thumb_trajectory(source, ("middle", "thumb"), [-1.0] * 16)
    assert mapped.approach_joint_rad[4:8] == tuple(range(4))
    assert mapped.approach_joint_rad[12:16] == tuple(range(12, 16))
    assert mapped.approach_joint_rad[:4] == (-1.0,) * 4
    assert mapped.per_finger_close_delay_steps == (0, 1, 0, 4)


def test_phase2r_incremental_store_resumes_without_duplicate(tmp_path):
    store = IncrementalJsonlStore(tmp_path / "phase2r.jsonl", 1.0, 0.01)
    assert store.append({"trial_id": "one", "outcome": "B_NOT_ACQUIRED"})
    assert not store.append({"trial_id": "one", "outcome": "BOTH_RETAINED"})
    assert store.completed_ids() == {"one"}
