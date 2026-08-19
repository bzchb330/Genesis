from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import ROOT
from .phase2r_config import Phase2RMatchingConfig, Phase2RStateConfig


OBJECT_LINEAR_SCALE = 0.5
PHASE2S_EXPERIMENT_ID = "phase2S_half_scale_objects"
PHASE2S_FORMAL_EXPERIMENT_ID = "phase2S_half_scale_formal"


@dataclass(frozen=True)
class Phase2SSecondGraspConfig:
    workspace_seed: int
    B_only_seed: int
    geometry_seed: int
    calibration_seed: int
    formal_seed: int
    workspace_candidate_poses: int
    B_only_candidate_cap: int
    B_only_success_target: int
    B_only_minimum_successes: int
    geometry_placements_per_region: int
    calibration_B_seeds_per_state: int
    formal_B_seeds_per_state: int
    maximum_controller_candidates: int


@dataclass(frozen=True)
class Phase2SConfig:
    experiment_id: str
    formal_experiment_id: str
    output_dir: str
    scene_filename: str
    object_linear_scale: float
    state: Phase2RStateConfig
    matching: Phase2RMatchingConfig
    second_grasp: Phase2SSecondGraspConfig


def load_phase2s_config(path: str | Path | None = None) -> tuple[Phase2SConfig, Path]:
    source = Path(path) if path is not None else ROOT / "configs" / "phase2S_half_scale_objects.yaml"
    source = source.resolve()
    payload: dict[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8"))
    cfg = Phase2SConfig(
        experiment_id=str(payload["experiment_id"]),
        formal_experiment_id=str(payload["formal_experiment_id"]),
        output_dir=str(payload["output_dir"]),
        scene_filename=str(payload["scene_filename"]),
        object_linear_scale=float(payload["object_linear_scale"]),
        state=Phase2RStateConfig(**payload["state"]),
        matching=Phase2RMatchingConfig(**payload["matching"]),
        second_grasp=Phase2SSecondGraspConfig(**payload["second_grasp"]),
    )
    if cfg.experiment_id != PHASE2S_EXPERIMENT_ID or cfg.formal_experiment_id != PHASE2S_FORMAL_EXPERIMENT_ID:
        raise ValueError("Phase 2S experiment namespaces must remain separate")
    if cfg.object_linear_scale != OBJECT_LINEAR_SCALE:
        raise ValueError("Phase 2S object linear scale must be exactly 0.5")
    if cfg.scene_filename != "scene_two_object_half_scale.yaml":
        raise ValueError("Phase 2S must use its isolated half-scale scene")
    if cfg.state.maximum_workers > 8 or cfg.state.maximum_palmar_attempts != 30_000:
        raise ValueError("Phase 2S worker and palmar-search limits differ from the authorized protocol")
    if cfg.state.fingertip_target < 200 or cfg.state.palmar_target < 200 or cfg.state.minimum_palmar_states != 100:
        raise ValueError("Phase 2S endpoint-state targets differ from the authorized protocol")
    if cfg.state.palm_contact_fraction_minimum != 0.80 or cfg.state.load_bearing_force_threshold_N != 0.20:
        raise ValueError("Phase 2S must preserve the Phase 2R contact thresholds")
    if cfg.matching.target_pairs != 100 or cfg.matching.calibration_per_group != 20:
        raise ValueError("Phase 2S matching and calibration counts differ from the protocol")
    if cfg.second_grasp.B_only_candidate_cap != 8192 or cfg.second_grasp.maximum_controller_candidates != 4096:
        raise ValueError("Phase 2S search caps differ from the authorized protocol")
    if cfg.second_grasp.formal_B_seeds_per_state != 20 or cfg.second_grasp.calibration_B_seeds_per_state != 5:
        raise ValueError("Phase 2S B-seed counts differ from the protocol")
    seeds = {
        cfg.second_grasp.workspace_seed,
        cfg.second_grasp.B_only_seed,
        cfg.second_grasp.geometry_seed,
        cfg.second_grasp.calibration_seed,
        cfg.second_grasp.formal_seed,
    }
    if len(seeds) != 5:
        raise ValueError("Phase 2S seed namespaces must be isolated")
    return cfg, source


def validate_phase2s_state_record(record: dict) -> None:
    """Reject unrevalidated or cross-experiment state records."""

    if (
        not record.get("revalidated_with_half_scale_geometry")
        or not str(record.get("grasp_state_id", "")).startswith("phase2S_")
        or record.get("experiment_id") not in (None, PHASE2S_EXPERIMENT_ID)
    ):
        raise ValueError("Phase 2R or unrevalidated state attempted to enter Phase 2S")
