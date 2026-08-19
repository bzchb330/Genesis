from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import ROOT


@dataclass(frozen=True)
class Phase2RStateConfig:
    seed: int
    fingertip_target: int
    palmar_target: int
    minimum_palmar_states: int
    maximum_states_per_group: int
    maximum_palmar_attempts: int
    maximum_workers: int
    fixture_close_steps: int
    fixture_contact_steps: int
    stable_hold_steps: int
    palm_contact_fraction_minimum: float
    load_bearing_force_threshold_N: float
    minimum_palmar_load_bearing_fingers: int
    maximum_palmar_load_bearing_fingers: int
    minimum_fingertip_contact_fingers: int
    maximum_penetration_m: float
    maximum_translation_drift_m: float
    maximum_orientation_drift_rad: float
    palm_tangent_y_bounds_m: list[float]
    palm_tangent_z_bounds_m: list[float]
    palm_surface_offset_bounds_m: list[float]
    active_closure_scale_bounds: list[float]
    active_joint_perturbation_rad: float
    focused_candidate_stride: int
    focused_palm_tangent_y_bounds_m: list[float]
    focused_palm_tangent_z_bounds_m: list[float]
    focused_closure_scale_bounds: list[float]
    focused_joint_perturbation_rad: float
    focused_retaining_finger_subsets: list[list[str]]
    focused_proposal_profile_paths: list[str]
    retaining_finger_subsets: list[list[str]]
    proposal_profile_paths: list[str]


@dataclass(frozen=True)
class Phase2RMatchingConfig:
    target_pairs: int
    calibration_per_group: int
    covariates: list[str]


@dataclass(frozen=True)
class Phase2RBConfig:
    geometry_seed: int
    calibration_seed: int
    formal_seed: int
    geometry_placements_per_region: int
    calibration_B_seeds_per_state: int
    formal_B_seeds_per_state: int
    maximum_trajectory_candidates: int


@dataclass(frozen=True)
class Phase2RConfig:
    experiment_id: str
    output_dir: str
    state: Phase2RStateConfig
    matching: Phase2RMatchingConfig
    second_grasp: Phase2RBConfig


def load_phase2r_config(path: str | Path | None = None) -> tuple[Phase2RConfig, Path]:
    source = Path(path) if path is not None else ROOT / "configs" / "phase2R_palmar_vs_fingertip.yaml"
    source = source.resolve()
    payload: dict[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8"))
    cfg = Phase2RConfig(
        experiment_id=str(payload["experiment_id"]),
        output_dir=str(payload["output_dir"]),
        state=Phase2RStateConfig(**payload["state"]),
        matching=Phase2RMatchingConfig(**payload["matching"]),
        second_grasp=Phase2RBConfig(**payload["second_grasp"]),
    )
    if cfg.experiment_id != "phase2R_palmar_vs_fingertip_formal":
        raise ValueError("Phase 2R experiment ID must remain separate from every prior dataset")
    if cfg.state.maximum_workers > 8:
        raise ValueError("Phase 2R may use at most eight workers")
    if cfg.state.maximum_palmar_attempts != 30_000:
        raise ValueError("the authorized palmar search cap is 30,000 deterministic candidates")
    if cfg.state.minimum_palmar_states != 100 or cfg.state.palmar_target < 150:
        raise ValueError("Phase 2R requires the supplied palmar stop and target counts")
    if cfg.state.fingertip_target < 150 or cfg.state.maximum_states_per_group > 500:
        raise ValueError("Phase 2R group targets must remain within the supplied bounds")
    if cfg.state.palm_contact_fraction_minimum != 0.80:
        raise ValueError("palmar contact must persist for 80% of the stable window")
    if cfg.state.load_bearing_force_threshold_N != 0.20:
        raise ValueError("the existing 0.20 N load-bearing threshold must be reused")
    if cfg.state.maximum_palmar_load_bearing_fingers != 2:
        raise ValueError("palmar states may use no more than two load-bearing fingers")
    if cfg.matching.target_pairs != 100:
        raise ValueError("the formal matched target is 100 pairs")
    if cfg.second_grasp.calibration_B_seeds_per_state != 5 or cfg.second_grasp.formal_B_seeds_per_state != 20:
        raise ValueError("Phase 2R B-seed counts must match the supplied protocol")
    if cfg.second_grasp.maximum_trajectory_candidates != 2048:
        raise ValueError("trajectory-only calibration is capped at 2048 candidates")
    return cfg, source
