import numpy as np

from seqgrasp.experiments.phase2h_visuals import _maximum_run, _prefix_length, diagnostic_series
from seqgrasp.phase2_5_config import load_phase2_5_config


def test_phase2h_survival_helpers_are_prefix_and_run_specific():
    values = np.asarray([True, True, False, True, True, True, False])
    assert _prefix_length(values) == 2
    assert _maximum_run(values) == 3
    assert _prefix_length(np.zeros(4, dtype=bool)) == 0
    assert _maximum_run(np.zeros(4, dtype=bool)) == 0


def test_phase2h_strict_prefix_uses_existing_force_and_penetration_gates():
    cfg25, _ = load_phase2_5_config()
    length, release = 6, 2
    flags = np.zeros((length, 4), dtype=np.int8)
    flags[1:, 0] = 1
    flags[1:3, 3] = 1
    arrays = {
        "B_position_m": np.zeros((length, 3)),
        "B_quaternion": np.tile([1.0, 0.0, 0.0, 0.0], (length, 1)),
        "B_per_finger_contact_flag": flags,
        "B_penetration_depths_m": np.zeros((length, 4)),
        "B_table_contact": np.zeros(length, dtype=np.int8),
        "B_free_finger_contacts": np.asarray([0, 2, 2, 1, 1, 1]),
        "B_hand_contacts": np.asarray([0, 2, 2, 1, 1, 1]),
        "B_hand_normal_force_N": np.asarray([0.0, 0.3, 0.3, 0.3, 0.1, 0.3]),
    }
    series = diagnostic_series({"fixture_release_timestep": release, "numerically_valid": True}, arrays, cfg25)
    assert _prefix_length(series["strict_state"][release:]) == 2
    arrays["B_penetration_depths_m"][1, 0] = cfg25.criteria.maximum_penetration_m + 1e-4
    series = diagnostic_series({"fixture_release_timestep": release, "numerically_valid": True}, arrays, cfg25)
    assert _prefix_length(series["strict_state"][release:]) == 0
