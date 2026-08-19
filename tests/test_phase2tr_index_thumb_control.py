from pathlib import Path

import pytest

from seqgrasp.phase2tr_config import assert_index_thumb_free_topology, load_phase2tr_config


def test_phase2tr_protocol_fixes_identity_and_caps():
    cfg, _ = load_phase2tr_config()
    assert cfg.topology.occupied_fingers == ["middle", "ring"]
    assert cfg.topology.free_fingers == ["index", "thumb"]
    assert cfg.topology.load_bearing_threshold_N == 0.20
    assert cfg.state_search.fingertip_attempt_cap == 50_000
    assert cfg.state_search.palmar_attempt_cap == 30_000
    assert cfg.second_grasp.b_only_candidate_cap == 4096
    assert cfg.second_grasp.maximum_controller_candidates == 2048


def test_phase2tr_identity_assertion_rejects_remapping():
    assert_index_thumb_free_topology({"occupied_finger_mask": [False, True, True, False]})
    with pytest.raises(ValueError):
        assert_index_thumb_free_topology({"occupied_finger_mask": [False, False, True, True]})


def test_phase2tr_sources_do_not_define_scalar_resource_or_training():
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "configs" / "phase2TR_index_thumb_control.yaml",
        root / "seqgrasp" / "phase2tr_config.py",
        root / "scripts" / "build_phase2tr_states.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    forbidden = ("scalar_" + "J", "resource_" + "weights", "train_" + "rl")
    assert all(token not in text for token in forbidden)
