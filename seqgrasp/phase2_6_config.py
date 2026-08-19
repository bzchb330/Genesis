from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import ROOT
from .phase2_5_config import load_phase2_5_config


@dataclass(frozen=True)
class Phase26Seeds:
    workspace: int
    candidate_poses: int
    pose_trajectory_search: int
    perturbations: int
    calibration_B_namespace: int
    formal_v3_B_namespace: int


@dataclass(frozen=True)
class WorkspaceConfig:
    samples_per_finger: int
    batch_size: int
    plot_samples_per_finger: int
    surface_access_tolerance_m: float
    palm_support_tolerance_m: float
    self_collision_tolerance_m: float
    candidate_pose_count: int
    selected_pose_count: int
    opposition_minimum_angle_deg: float
    minimum_joint_margin_rad: float


@dataclass(frozen=True)
class ObjectBConfig:
    vertical: bool
    yaw_bounds_rad: list[float]
    old_center_x_bounds_m: list[float]
    old_center_y_bounds_m: list[float]
    old_center_z_bounds_m: list[float]


@dataclass(frozen=True)
class DynamicSearchConfig:
    initial_candidate_count: int
    expanded_candidate_count: int
    unsupported_hold_steps: int
    target_success_count: int
    robustness_profiles: int
    perturbations_per_profile: int


@dataclass(frozen=True)
class SequentialConfig:
    intersection_A_grasps: int
    initial_candidate_count: int
    expanded_candidate_count: int
    target_success_count: int
    calibration_A_grasps: int
    calibration_B_seeds: int
    formal_A_grasps: int
    formal_B_seeds_per_grasp: int


@dataclass(frozen=True)
class Phase26Config:
    phase2_6_only: bool
    frozen_phase2_config: str
    frozen_phase2_5_config: str
    output_dir: str
    maximum_workers: int
    seeds: Phase26Seeds
    workspace: WorkspaceConfig
    object_B: ObjectBConfig
    dynamic_search: DynamicSearchConfig
    sequential: SequentialConfig


def load_phase2_6_config(path: str | Path | None = None) -> tuple[Phase26Config, Path]:
    source = Path(path) if path is not None else ROOT / "configs" / "phase2_6_b_graspable_workspace.yaml"
    source = source.resolve()
    payload: dict[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8"))
    cfg = Phase26Config(
        phase2_6_only=bool(payload["phase2_6_only"]),
        frozen_phase2_config=str(payload["frozen_phase2_config"]),
        frozen_phase2_5_config=str(payload["frozen_phase2_5_config"]),
        output_dir=str(payload["output_dir"]),
        maximum_workers=int(payload["maximum_workers"]),
        seeds=Phase26Seeds(**payload["seeds"]),
        workspace=WorkspaceConfig(**payload["workspace"]),
        object_B=ObjectBConfig(**payload["object_B"]),
        dynamic_search=DynamicSearchConfig(**payload["dynamic_search"]),
        sequential=SequentialConfig(**payload["sequential"]),
    )
    if not cfg.phase2_6_only or cfg.maximum_workers > 8:
        raise ValueError("Phase 2.6 must remain isolated and use at most 8 workers")
    if cfg.workspace.samples_per_finger < 200000 or cfg.workspace.candidate_pose_count < 10000:
        raise ValueError("Phase 2.6 dense workspace and pose-search budgets are frozen minima")
    if cfg.workspace.samples_per_finger % cfg.workspace.batch_size:
        raise ValueError("workspace samples must divide into deterministic complete batches")
    if cfg.dynamic_search.initial_candidate_count < 4096 or cfg.dynamic_search.expanded_candidate_count < 8192:
        raise ValueError("Phase 2.6 B-only dynamic budgets are frozen minima")
    if cfg.dynamic_search.unsupported_hold_steps != 500:
        raise ValueError("the unsupported hold may not change")
    phase25, _ = load_phase2_5_config(ROOT / cfg.frozen_phase2_5_config)
    old = cfg.object_B
    if (
        old.old_center_x_bounds_m != phase25.frozen_B_distribution.center_x_bounds_m
        or old.old_center_y_bounds_m != phase25.frozen_B_distribution.center_y_bounds_m
        or old.old_center_z_bounds_m != phase25.frozen_B_distribution.center_z_bounds_m
    ):
        raise ValueError("the historical B box must be recorded exactly")
    return cfg, source
