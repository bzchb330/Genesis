from collections import Counter

import numpy as np

from seqgrasp.experiments.grasp_sampling import deterministic_subset, evaluate_candidate, ferrari_canny_epsilon, sample_candidate
from seqgrasp.phase2_config import load_phase2_config


def test_finger_subset_sampling_is_deterministic_and_size_balanced():
    phase2, _ = load_phase2_config()
    first = [deterministic_subset(index, phase2.dataset.finger_subset_sizes) for index in range(30)]
    second = [deterministic_subset(index, phase2.dataset.finger_subset_sizes) for index in range(30)]
    assert first == second
    assert Counter(map(len, first)) == {2: 10, 3: 10, 4: 10}
    assert len(set(first)) > 3


def test_candidate_sampling_is_seed_deterministic_and_bounded():
    phase2, _ = load_phase2_config()
    _, profile_a, metadata_a = sample_candidate(phase2, 7)
    _, profile_b, metadata_b = sample_candidate(phase2, 7)
    assert metadata_a == metadata_b
    assert profile_a == profile_b
    assert all(0.0 <= value <= 1.0 for value in profile_a.closed_joint_fractions.values())
    assert len(metadata_a["commanded_finger_subset"]) in phase2.dataset.finger_subset_sizes
    _, _, anchor = sample_candidate(phase2, 23)
    assert metadata_a["sampling_mode"] == "full_PI_range"
    assert anchor["sampling_mode"] == "validated_proposal_anchor"


def test_ferrari_canny_epsilon_detects_symmetric_force_closure():
    points = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0],
        [0, -1, 0], [0, 0, 1], [0, 0, -1],
    ], dtype=float) * 0.025
    normals = -points / np.linalg.norm(points, axis=1, keepdims=True)
    epsilon = ferrari_canny_epsilon(points, normals, np.zeros(3), 0.8, 8, np.sqrt(3) * 0.025, 1e-8)
    assert epsilon > 1e-8
    assert ferrari_canny_epsilon(points[:1], normals[:1], np.zeros(3), 0.8, 8, 0.025, 1e-8) == 0.0


def test_streamlined_candidate_probe_is_seed_deterministic():
    phase2, _ = load_phase2_config()
    first = evaluate_candidate(phase2, 23)
    second = evaluate_candidate(phase2, 23)
    assert first["accepted"] is True
    assert first["accepted"] == second["accepted"]
    assert first["stability"] == second["stability"]
    assert first["ferrari_canny_epsilon"] == second["ferrari_canny_epsilon"]
