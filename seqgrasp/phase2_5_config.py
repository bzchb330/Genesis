from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import ROOT
from .phase2_config import load_phase2_config


@dataclass(frozen=True)
class Phase25Seeds:
    geometry_positive_control: int
    b_only_search: int
    sequential_search: int
    calibration_split: int
    calibration_B_namespace: int
    formal_v2_B_namespace: int


@dataclass(frozen=True)
class PositiveControlConfig:
    B_pose_world_m: list[float]
    B_yaw_rad: float
    parked_A_position_m: list[float]
    active_fingers: list[str]
    precontact_anchor_joint_rad: dict[str, list[float]]


@dataclass(frozen=True)
class Phase25Timing:
    approach_steps: int
    precontact_steps: int
    close_steps_bounds: list[int]
    fixture_release_delay_steps_bounds: list[int]
    unsupported_hold_steps: int
    diagnostic_pre_release_steps: int
    diagnostic_post_release_steps: int


@dataclass(frozen=True)
class TrajectorySearchConfig:
    initial_candidate_count: int
    expanded_candidate_count: int
    joint_approach_offset_rad: list[float]
    joint_precontact_offset_rad: list[float]
    joint_closing_offset_rad: list[float]
    joint_hold_offset_rad: list[float]
    per_finger_close_delay_steps: list[int]
    maximum_saved_profiles: int
    required_B_only_successes: int
    preferred_B_only_successes: int


@dataclass(frozen=True)
class SequentialSearchConfig:
    initial_candidate_count: int
    expanded_candidate_count: int
    stage1_A_grasps: int
    expanded_A_grasps: int
    calibration_A_grasps: int
    calibration_B_placements: int
    robustness_A_grasps: int
    robustness_B_placements: int
    required_distinct_successes: int


@dataclass(frozen=True)
class FormalV2Config:
    A_grasp_target: int
    trials_per_grasp: int


@dataclass(frozen=True)
class FrozenCriteria:
    minimum_A_finger_contacts: int
    minimum_A_normal_force_N: float
    minimum_B_free_finger_contacts: int
    minimum_B_hand_contacts: int
    minimum_B_normal_force_N: float
    maximum_penetration_m: float
    maximum_A_translation_m: float
    maximum_A_orientation_rad: float
    maximum_B_translation_m: float
    maximum_B_orientation_rad: float


@dataclass(frozen=True)
class FrozenBDistribution:
    center_x_bounds_m: list[float]
    center_y_bounds_m: list[float]
    center_z_bounds_m: list[float]
    yaw_bounds_rad: list[float]
    roll_pitch_rad: list[float]


@dataclass(frozen=True)
class Phase25Config:
    phase2_5_only: bool
    frozen_phase2_config: str
    output_dir: str
    maximum_workers: int
    seeds: Phase25Seeds
    positive_control: PositiveControlConfig
    timing: Phase25Timing
    trajectory_search: TrajectorySearchConfig
    sequential_search: SequentialSearchConfig
    formal_v2: FormalV2Config
    criteria: FrozenCriteria
    frozen_B_distribution: FrozenBDistribution


def load_phase2_5_config(path: str | Path | None = None) -> tuple[Phase25Config, Path]:
    source = Path(path) if path is not None else ROOT / "configs" / "phase2_5_second_grasp_calibration.yaml"
    source = source.resolve()
    payload: dict[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8"))
    cfg = Phase25Config(
        phase2_5_only=bool(payload["phase2_5_only"]),
        frozen_phase2_config=str(payload["frozen_phase2_config"]),
        output_dir=str(payload["output_dir"]),
        maximum_workers=int(payload["maximum_workers"]),
        seeds=Phase25Seeds(**payload["seeds"]),
        positive_control=PositiveControlConfig(**payload["positive_control"]),
        timing=Phase25Timing(**payload["timing"]),
        trajectory_search=TrajectorySearchConfig(**payload["trajectory_search"]),
        sequential_search=SequentialSearchConfig(**payload["sequential_search"]),
        formal_v2=FormalV2Config(**payload["formal_v2"]),
        criteria=FrozenCriteria(**payload["criteria"]),
        frozen_B_distribution=FrozenBDistribution(**payload["frozen_B_distribution"]),
    )
    if not cfg.phase2_5_only or cfg.maximum_workers > 8:
        raise ValueError("Phase 2.5 must remain isolated and use at most 8 workers")
    if cfg.trajectory_search.initial_candidate_count < 512 or cfg.trajectory_search.expanded_candidate_count < 2048:
        raise ValueError("B-only search budgets must satisfy the supplied Phase 2.5 protocol")
    if cfg.sequential_search.initial_candidate_count < 1024 or cfg.sequential_search.expanded_candidate_count < 4096:
        raise ValueError("sequential search budgets must satisfy the supplied Phase 2.5 protocol")
    if cfg.timing.unsupported_hold_steps != 500 or cfg.formal_v2.trials_per_grasp != 20:
        raise ValueError("the frozen hold and K=20 formal design may not change")
    phase2, _ = load_phase2_config(ROOT / cfg.frozen_phase2_config)
    frozen_pairs = {
        "minimum_A_finger_contacts": phase2.second_grasp.minimum_A_finger_contacts,
        "minimum_A_normal_force_N": phase2.second_grasp.minimum_A_normal_force_N,
        "minimum_B_free_finger_contacts": phase2.second_grasp.minimum_B_free_finger_contacts,
        "minimum_B_hand_contacts": phase2.second_grasp.minimum_B_hand_contacts,
        "minimum_B_normal_force_N": phase2.second_grasp.minimum_B_normal_force_N,
        "maximum_penetration_m": phase2.second_grasp.maximum_penetration_m,
        "maximum_A_translation_m": phase2.second_grasp.maximum_A_translation_m,
        "maximum_A_orientation_rad": phase2.second_grasp.maximum_A_orientation_rad,
        "maximum_B_translation_m": phase2.second_grasp.maximum_B_translation_m,
        "maximum_B_orientation_rad": phase2.second_grasp.maximum_B_orientation_rad,
    }
    if any(getattr(cfg.criteria, key) != value for key, value in frozen_pairs.items()):
        raise ValueError("Phase 2.5 criteria differ from frozen Phase 2")
    expected_bounds = (
        phase2.second_grasp.B_center_x_bounds_m,
        phase2.second_grasp.B_center_y_bounds_m,
        phase2.second_grasp.B_center_z_bounds_m,
        phase2.second_grasp.B_yaw_bounds_rad,
    )
    actual_bounds = (
        cfg.frozen_B_distribution.center_x_bounds_m,
        cfg.frozen_B_distribution.center_y_bounds_m,
        cfg.frozen_B_distribution.center_z_bounds_m,
        cfg.frozen_B_distribution.yaw_bounds_rad,
    )
    if actual_bounds != expected_bounds:
        raise ValueError("Phase 2.5 B distribution differs from frozen Phase 2")
    return cfg, source
