import math

import numpy as np

from seqgrasp import load_configs
from seqgrasp.experiments.second_grasp import OUTCOMES, cylinder_lowest_point_z, placement_candidate
from seqgrasp.phase2_config import load_phase2_config


def test_vertical_and_horizontal_cylinder_lowest_point_are_orientation_aware():
    assert np.isclose(cylinder_lowest_point_z([0, 0, 1], [1, 0, 0, 0], 0.25, 0.4), 0.6)
    horizontal_x = [math.sqrt(0.5), math.sqrt(0.5), 0, 0]
    assert np.isclose(cylinder_lowest_point_z([0, 0, 1], horizontal_x, 0.25, 0.4), 0.75)


def test_B_placement_is_deterministic_bounded_and_fixture_presented():
    cfg = load_configs()
    phase2, _ = load_phase2_config()
    first = placement_candidate(cfg, phase2.second_grasp, 7)
    second = placement_candidate(cfg, phase2.second_grasp, 7)
    assert first == second
    assert phase2.second_grasp.B_center_x_bounds_m[0] <= first.position_m[0] <= phase2.second_grasp.B_center_x_bounds_m[1]
    assert phase2.second_grasp.B_center_y_bounds_m[0] <= first.position_m[1] <= phase2.second_grasp.B_center_y_bounds_m[1]
    assert phase2.second_grasp.B_center_z_bounds_m[0] <= first.position_m[2] <= phase2.second_grasp.B_center_z_bounds_m[1]
    assert first.quaternion[1:3] == (0.0, 0.0)
    assert len(OUTCOMES) == len(set(OUTCOMES)) == 5
