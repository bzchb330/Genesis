from dataclasses import replace

import numpy as np

from seqgrasp.experiments.phase2_5_trajectory import (
    _phase_target,
    classify_failure_mechanism,
    run_b_acquisition_trajectory,
    sample_b_only_trajectory,
)
from seqgrasp.phase2_5_config import load_phase2_5_config


def test_phase2_5_config_preserves_frozen_protocol_and_seed_namespaces():
    cfg, _ = load_phase2_5_config()
    assert cfg.frozen_B_distribution.center_x_bounds_m == [0.055, 0.065]
    assert cfg.frozen_B_distribution.center_y_bounds_m == [0.115, 0.125]
    assert cfg.frozen_B_distribution.center_z_bounds_m == [0.215, 0.225]
    assert cfg.timing.unsupported_hold_steps == 500
    assert cfg.trajectory_search.initial_candidate_count == 512
    assert cfg.trajectory_search.expanded_candidate_count == 2048
    assert len({cfg.seeds.b_only_search, cfg.seeds.calibration_B_namespace, cfg.seeds.formal_v2_B_namespace}) == 3


def test_phase2_5_trajectory_sampling_is_deterministic_and_has_all_targets():
    cfg, _ = load_phase2_5_config()
    first = sample_b_only_trajectory(cfg, 17)
    assert first == sample_b_only_trajectory(cfg, 17)
    assert len(first.approach_joint_rad) == len(first.precontact_joint_rad) == 16
    assert len(first.closing_joint_rad) == len(first.hold_joint_rad) == 16
    phase, _ = _phase_target(0, np.zeros(16), first, cfg.timing.approach_steps, cfg.timing.precontact_steps)
    assert phase == "approach"
    phase, _ = _phase_target(cfg.timing.approach_steps, np.zeros(16), first, cfg.timing.approach_steps, cfg.timing.precontact_steps)
    assert phase == "precontact"


def test_phase2_5_failure_labels_are_descriptive_and_deterministic():
    base = {
        "invalid_reason": None,
        "A_present": False,
        "A_retained": True,
        "maximum_B_free_finger_contacts_before_release": 1,
        "maximum_B_hand_contacts_before_release": 2,
        "maximum_B_hand_normal_force_before_release_N": 1.0,
        "first_post_release_contact_loss_step": None,
        "B_table_contact_after_release": False,
        "maximum_B_orientation_after_release_rad": 0.0,
    }
    assert classify_failure_mechanism({**base, "maximum_B_free_finger_contacts_before_release": 0}) == "NO_B_CONTACT_BEFORE_RELEASE"
    assert classify_failure_mechanism({**base, "first_post_release_contact_loss_step": 5}) == "CONTACT_LOST_IMMEDIATELY_AFTER_RELEASE"
    assert classify_failure_mechanism({**base, "B_table_contact_after_release": True}) == "B_SLIPPED_TO_TABLE"


def test_phase2_5_fixture_release_and_diagnostic_logging_smoke():
    cfg, _ = load_phase2_5_config()
    trajectory = replace(sample_b_only_trajectory(cfg, 0), close_steps=1, fixture_release_delay_steps=0)
    fast_cfg = replace(
        cfg,
        timing=replace(cfg.timing, approach_steps=1, precontact_steps=1, unsupported_hold_steps=3),
    )
    summary, arrays = run_b_acquisition_trajectory(fast_cfg, trajectory, collect_timeseries=True)
    assert summary["fixture_released"]
    assert summary["fixture_release_timestep"] == 3
    assert arrays is not None
    required = {
        "fixture_active", "B_position_m", "B_linear_velocity_m_per_s", "B_table_contact",
        "B_hand_contacts", "B_free_finger_contacts", "B_per_finger_normal_force_N",
        "B_per_finger_tangential_force_N", "B_contact_positions_m", "B_contact_normals",
        "A_position_m", "A_displacement_m", "commanded_free_finger_target_rad",
        "actual_free_finger_joint_rad", "free_finger_joint_velocity_rad_per_s",
        "free_finger_actuator_controls",
    }
    assert required <= set(arrays)
    assert arrays["fixture_active"].tolist() == [True, True, True, False, False, False]
