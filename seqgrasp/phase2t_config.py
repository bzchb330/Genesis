from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import ROOT


PHASE2T_EXPERIMENT_ID = "phase2T_eligible_fingertip_vs_palmar"


@dataclass(frozen=True)
class Phase2TStateSearchConfig:
    seed: int
    fingertip_target: int
    fingertip_minimum: int
    fingertip_attempt_cap: int
    palmar_target: int
    palmar_attempt_cap: int
    maximum_workers: int
    support_pairs: list[list[str]]


@dataclass(frozen=True)
class Phase2TMatchingConfig:
    target_pairs: int
    minimum_pairs: int
    calibration_per_group: int
    covariates: list[str]


@dataclass(frozen=True)
class Phase2TSecondGraspConfig:
    b_only_seed: int
    geometry_seed: int
    calibration_seed: int
    formal_seed: int
    b_only_candidate_cap: int
    b_only_success_target: int
    b_only_hard_minimum: int
    calibration_B_seeds_per_state: int
    formal_B_seeds_per_state: int
    maximum_controller_candidates: int


@dataclass(frozen=True)
class Phase2TConfig:
    experiment_id: str
    output_dir: str
    scene_filename: str
    state_search: Phase2TStateSearchConfig
    matching: Phase2TMatchingConfig
    second_grasp: Phase2TSecondGraspConfig


def load_phase2t_config(path: str | Path | None = None) -> tuple[Phase2TConfig, Path]:
    source = (Path(path) if path is not None else ROOT / "configs" / "phase2T_digit_eligible_control.yaml").resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    cfg = Phase2TConfig(
        experiment_id=str(payload["experiment_id"]),
        output_dir=str(payload["output_dir"]),
        scene_filename=str(payload["scene_filename"]),
        state_search=Phase2TStateSearchConfig(**payload["state_search"]),
        matching=Phase2TMatchingConfig(**payload["matching"]),
        second_grasp=Phase2TSecondGraspConfig(**payload["second_grasp"]),
    )
    expected_pairs = {
        ("index", "middle"), ("index", "ring"), ("index", "thumb"),
        ("middle", "ring"), ("middle", "thumb"), ("ring", "thumb"),
    }
    if cfg.experiment_id != PHASE2T_EXPERIMENT_ID:
        raise ValueError("Phase 2T experiment namespace changed")
    if cfg.scene_filename != "scene_two_object_half_scale.yaml":
        raise ValueError("Phase 2T must preserve the Phase 2S half-scale scene")
    if {tuple(pair) for pair in cfg.state_search.support_pairs} != expected_pairs:
        raise ValueError("Phase 2T must search all six two-finger support pairs")
    if cfg.state_search.fingertip_attempt_cap != 50_000 or cfg.state_search.palmar_attempt_cap != 30_000:
        raise ValueError("Phase 2T state-search caps changed")
    if cfg.state_search.maximum_workers > 8:
        raise ValueError("Phase 2T permits at most eight workers")
    if cfg.matching.minimum_pairs != 60 or cfg.matching.target_pairs != 100:
        raise ValueError("Phase 2T matching limits changed")
    if cfg.second_grasp.formal_B_seeds_per_state != 20:
        raise ValueError("Phase 2T requires twenty formal B seeds")
    seeds = {
        cfg.state_search.seed, cfg.second_grasp.b_only_seed, cfg.second_grasp.geometry_seed,
        cfg.second_grasp.calibration_seed, cfg.second_grasp.formal_seed,
    }
    if len(seeds) != 5:
        raise ValueError("Phase 2T seed namespaces must be distinct")
    return cfg, source
