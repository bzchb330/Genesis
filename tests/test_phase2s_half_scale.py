from pathlib import Path

import mujoco
import pytest

from seqgrasp.config import ROOT, load_configs
from seqgrasp.experiments.phase2r import paired_formal_trial_id
from seqgrasp.experiments.resumable import IncrementalJsonlStore
from seqgrasp.phase2s_config import (
    OBJECT_LINEAR_SCALE,
    PHASE2S_EXPERIMENT_ID,
    PHASE2S_FORMAL_EXPERIMENT_ID,
    load_phase2s_config,
    validate_phase2s_state_record,
)
from seqgrasp.scene_builder import build_scene


def _geom(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


def test_compiled_A_and_B_dimensions_are_exactly_half_and_masses_are_fixed():
    large = load_configs()
    small_cfg, _ = load_phase2s_config()
    small = load_configs(scene_filename=small_cfg.scene_filename)
    large_model, _ = build_scene(large)
    small_model, _ = build_scene(small)
    for geom_name in ("object_a_geom", "object_b_geom"):
        old_id, new_id = _geom(large_model, geom_name), _geom(small_model, geom_name)
        assert small_model.geom_size[new_id].tolist() == pytest.approx(
            (OBJECT_LINEAR_SCALE * large_model.geom_size[old_id]).tolist(), rel=0.0, abs=0.0
        )
        assert float(large_model.body_mass[large_model.geom_bodyid[old_id]]) == pytest.approx(0.08)
        assert float(small_model.body_mass[small_model.geom_bodyid[new_id]]) == pytest.approx(0.08)


def test_half_scale_defaults_preserve_one_millimetre_table_clearance():
    phase2s, _ = load_phase2s_config()
    cfg = load_configs(scene_filename=phase2s.scene_filename)
    table_top = cfg.scene.table_pos[2] + cfg.scene.table_size[2]
    by_name = {obj.name: obj for obj in cfg.scene.objects}
    assert by_name["object_a"].pos[2] - by_name["object_a"].size[2] - table_top == pytest.approx(0.001)
    assert by_name["object_b"].pos[2] - by_name["object_b"].size[1] - table_top == pytest.approx(0.001)


def test_phase2s_namespace_requires_dynamic_half_scale_revalidation():
    valid = {
        "grasp_state_id": "phase2S_fingertip_001",
        "experiment_id": PHASE2S_EXPERIMENT_ID,
        "revalidated_with_half_scale_geometry": True,
    }
    validate_phase2s_state_record(valid)
    for invalid in (
        {**valid, "grasp_state_id": "phase2R_fingertip_001"},
        {**valid, "experiment_id": "phase2R_palmar_vs_fingertip"},
        {**valid, "revalidated_with_half_scale_geometry": False},
    ):
        with pytest.raises(ValueError):
            validate_phase2s_state_record(invalid)


def test_phase2r_baseline_artifacts_remain_separate_and_present():
    required = (
        ROOT / "docs" / "PHASE2R_PALMAR_VS_FINGERTIP_RESULTS.md",
        ROOT / "docs" / "PHASE2R_PRELIMINARY_EVIDENCE.md",
        ROOT / "docs" / "figures" / "phase2R" / "phase2R_main_result.pdf",
        ROOT / "configs" / "phase2R_frozen_B_distribution.yaml",
        ROOT / "configs" / "phase2R_frozen_controller.yaml",
    )
    assert all(path.exists() for path in required)
    assert (ROOT / "outputs" / "phase2R").is_dir()


def test_small_B_geometry_freezes_and_seed_namespaces_are_new():
    phase2s, _ = load_phase2s_config()
    assert (ROOT / "configs" / "phase2S_frozen_B_distribution.yaml").exists()
    assert phase2s.second_grasp.calibration_seed != phase2s.second_grasp.formal_seed
    assert phase2s.second_grasp.formal_seed not in {20250901, 20251000, 20261200, 20261300}
    assert phase2s.formal_experiment_id == PHASE2S_FORMAL_EXPERIMENT_ID


def test_phase2s_paired_seed_ids_are_consistent_and_group_specific():
    namespace = load_phase2s_config()[0].second_grasp.formal_seed
    first = paired_formal_trial_id("pair-001", "FINGERTIP", 7, namespace)
    assert first == paired_formal_trial_id("pair-001", "FINGERTIP", 7, namespace)
    assert first != paired_formal_trial_id("pair-001", "PALMAR_SECURED", 7, namespace)


def test_phase2s_incremental_results_resume_without_duplicates(tmp_path):
    store = IncrementalJsonlStore(tmp_path / "phase2s.jsonl", 1.0, 0.01)
    assert store.append({"trial_id": "phase2S-one", "outcome": "B_NOT_ACQUIRED"})
    assert not store.append({"trial_id": "phase2S-one", "outcome": "BOTH_RETAINED"})
    assert store.completed_ids() == {"phase2S-one"}


def test_phase2s_sources_define_neither_scalar_J_nor_RL():
    paths = [
        *ROOT.glob("seqgrasp/**/*phase2s*.py"),
        *ROOT.glob("scripts/*phase2s*.py"),
        ROOT / "configs" / "phase2S_half_scale_objects.yaml",
    ]
    text = "\n".join(path.read_text(encoding="utf-8").lower() for path in paths if path.exists())
    assert "scalar_j" not in text
    assert "import gym" not in text
    assert "stable_baselines" not in text
