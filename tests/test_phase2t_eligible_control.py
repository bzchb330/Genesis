import re

from seqgrasp.phase2t_config import PHASE2T_EXPERIMENT_ID, load_phase2t_config


def test_phase2t_config_requires_all_pairs_and_preserves_half_scale_scene():
    cfg, _ = load_phase2t_config()
    assert cfg.experiment_id == PHASE2T_EXPERIMENT_ID
    assert cfg.scene_filename == "scene_two_object_half_scale.yaml"
    assert len({tuple(pair) for pair in cfg.state_search.support_pairs}) == 6
    assert cfg.state_search.fingertip_attempt_cap == 50_000
    assert cfg.state_search.palmar_attempt_cap == 30_000
    assert cfg.state_search.maximum_workers <= 8
    assert cfg.second_grasp.formal_B_seeds_per_state == 20


def test_phase2t_defines_no_scalar_J_or_training_configuration():
    cfg, path = load_phase2t_config()
    text = path.read_text(encoding="utf-8").lower()
    assert "compute_j" not in text
    assert "reward" not in text
    assert re.search(r"\bppo\b", text) is None
    assert re.search(r"\brl\b", text) is None
    assert cfg.matching.covariates == [
        "ferrari_canny_epsilon", "total_A_normal_force_N",
        "A_translation_drift_m", "A_rotation_drift_rad", "minimum_joint_margin_rad",
    ]
