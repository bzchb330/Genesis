from __future__ import annotations

from pathlib import Path

import numpy as np

from seqgrasp.phase3.config import FINGERS
from seqgrasp.phase3.model import build_shadow_scene
from seqgrasp.phase3b0 import (
    EXERCISED_OFFSETS_M,
    EXERCISED_RADIUS_M,
    detect_contact_gaps,
    evaluate_attempt,
    pair_aware_penetration,
    palm_relative_pose,
    restore_release_state,
    sample_candidate,
)
from seqgrasp.phase3b0_analysis import (
    assign_pose_regions,
    frozen_split,
    greedy_deduplicate,
    reproduce_phase3a,
)


def test_phase3b0_sampling_is_deterministic_and_stays_in_exercised_convex_hull():
    first = [sample_candidate(index) for index in range(64)]
    second = [sample_candidate(index) for index in range(64)]
    assert first == second
    np.testing.assert_array_equal(
        np.asarray([candidate.offset_m for candidate in first[:7]]), EXERCISED_OFFSETS_M
    )
    assert all(sum(abs(value) for value in candidate.offset_m) <= EXERCISED_RADIUS_M + 1e-15 for candidate in first)
    assert all(candidate.object_quaternion_wxyz == (1.0, 0.0, 0.0, 0.0) for candidate in first)


def test_contact_aware_release_serialization_reconstruction_and_semantics(tmp_path: Path):
    result = evaluate_attempt(0, tmp_path / "phase3b0", retention_steps=10)
    assert result["accepted_raw_release"]
    assert result["release"]["contact_latches"] == {"thumb": True, "index": True}
    assert result["release"]["thumb_object_contact_count"] > 0
    assert result["release"]["index_object_contact_count"] > 0
    assert result["release"]["other_finger_object_contact_counts"] == {
        "middle": 0,
        "ring": 0,
        "little": 0,
    }
    assert not result["release"]["palm_object_contact"]
    assert tuple(FINGERS) == ("thumb", "index", "middle", "ring", "little")
    state_path = Path(result["release_state_path"])
    assert state_path.exists()
    scene = build_shadow_scene()
    arrays = restore_release_state(scene, state_path)
    np.testing.assert_array_equal(scene.data.qpos, arrays["qpos"])
    np.testing.assert_array_equal(scene.data.qvel, arrays["qvel"])
    np.testing.assert_array_equal(scene.data.ctrl, arrays["ctrl"])
    np.testing.assert_array_equal(scene.data.mocap_pos, arrays["mocap_pos"])
    np.testing.assert_array_equal(scene.data.mocap_quat, arrays["mocap_quat"])
    np.testing.assert_array_equal(scene.data.eq_active, arrays["eq_active"])
    relative_position, relative_quaternion = palm_relative_pose(scene)
    np.testing.assert_allclose(
        relative_position,
        result["release"]["object_palm_relative_position_m"],
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        relative_quaternion,
        result["release"]["object_palm_relative_quaternion_wxyz"],
        rtol=0.0,
        atol=1e-15,
    )
    assert result["retention"]["horizon_survival"]["1"]
    assert result["retention"]["horizon_survival"]["5"]
    assert result["retention"]["horizon_survival"]["10"]
    assert not result["retention"]["horizon_survival"]["25"]


def test_pair_aware_penetration_keeps_intended_and_gross_contacts_separate():
    records = [
        {"surface": "thumb_tip", "penetration_m": 0.0007},
        {"surface": "index_tip", "penetration_m": 0.0005},
        {"surface": "palm", "penetration_m": 0.0012},
        {"surface": "table", "penetration_m": 0.0002},
    ]
    penetration = pair_aware_penetration(records)
    assert penetration["thumb_object"] == 0.0007
    assert penetration["index_object"] == 0.0005
    assert penetration["palm_object"] == 0.0012
    assert penetration["table_object"] == 0.0002
    assert penetration["maximum_intended_grip"] == 0.0007
    assert penetration["maximum_gross_non_grip"] == 0.0012


def test_contact_gap_detection_records_reestablishment_and_open_gap():
    flags = np.zeros((7, 6), dtype=np.float64)
    flags[0, 0] = 1.0
    flags[1, 0] = 1.0
    flags[4, 1] = 1.0
    flags[5, 1] = 1.0
    positions = np.c_[np.arange(7, dtype=np.float64), np.zeros((7, 2))]
    velocities = np.ones((7, 3), dtype=np.float64)
    gaps = detect_contact_gaps(flags, positions, positions, velocities, 0.002)
    assert [(gap["start_step"], gap["duration_steps"]) for gap in gaps] == [(2, 2), (6, 1)]
    assert gaps[0]["reestablished"]
    assert gaps[0]["reestablished_by"] == ["index"]
    assert not gaps[1]["reestablished"]


def test_deduplication_pose_regions_and_split_are_deterministic():
    rng = np.random.default_rng(330)
    descriptors = rng.normal(size=(500, 24))
    descriptors[1] = descriptors[0]
    assert greedy_deduplicate(descriptors, 0.0) == greedy_deduplicate(descriptors, 0.0)
    assert len(greedy_deduplicate(descriptors, 0.0)) == 499
    regions_a, report_a = assign_pose_regions(descriptors)
    regions_b, report_b = assign_pose_regions(descriptors)
    np.testing.assert_array_equal(regions_a, regions_b)
    assert report_a == report_b
    rows = [
        {
            "candidate": {"candidate_id": index},
            "release": {"state_hash": f"hash-{index}"},
        }
        for index in range(500)
    ]
    first = frozen_split(rows, descriptors, regions_a)
    second = frozen_split(rows, descriptors, regions_a)
    assert first == second
    assert len(first["train"]) == 300
    assert len(first["validation"]) == 100
    assert len(first["test"]) == 100
    assert first["zero_id_overlap"]
    assert first["zero_state_hash_overlap"]


def test_phase3a_handoff_chain_remains_reproducible():
    reproduction = reproduce_phase3a()
    assert reproduction["passed"], reproduction
