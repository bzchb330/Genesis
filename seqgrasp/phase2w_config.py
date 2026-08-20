from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import ROOT


PHASE2W_EXPERIMENT_ID = "phase2W_static_wrist_matched"
OCCUPIED_FINGERS = ("middle", "ring")
FREE_FINGERS = ("index", "thumb")


@dataclass(frozen=True)
class Phase2WTopologyConfig:
    occupied_fingers: list[str]
    free_fingers: list[str]
    load_bearing_threshold_N: float


@dataclass(frozen=True)
class Phase2WWristSearchConfig:
    coarse_roll_deg: list[float]
    coarse_pitch_deg: list[float]
    coarse_yaw_deg: list[float]
    refinement_offsets_deg: list[float]
    screening_states_per_group: int
    minimum_valid_states_per_group: int
    top_coarse_for_refinement: int
    top_dynamic_candidates: int
    maximum_workers: int


@dataclass(frozen=True)
class Phase2WGeometryConfig:
    workspace_samples_per_group: int
    candidate_B_poses_per_orientation: int
    yaw_bounds_rad: list[float]
    minimum_opposition_angle_deg: float


@dataclass(frozen=True)
class Phase2WSecondGraspConfig:
    initial_candidates_per_wrist: int
    expanded_candidates_per_wrist: int
    global_candidate_cap: int
    hard_minimum_successes: int
    preferred_successes: int
    robustness_trials_per_configuration: int
    robustness_configuration_cap: int


@dataclass(frozen=True)
class Phase2WEndpointPopulationConfig:
    target_per_group: int
    minimum_per_group: int
    additional_attempt_cap_per_group: int
    calibration_states_per_group: int


@dataclass(frozen=True)
class Phase2WCalibrationConfig:
    maximum_controller_candidates: int
    formal_B_seeds_per_pair: int
    target_matched_pairs: int
    minimum_matched_pairs: int


@dataclass(frozen=True)
class Phase2WSeeds:
    screening: int
    geometry: int
    B_only: int
    robustness: int
    endpoint_replay: int
    additional_fingertip: int
    additional_palmar: int
    calibration: int
    formal: int
    bootstrap: int


@dataclass(frozen=True)
class Phase2WConfig:
    experiment_id: str
    output_dir: str
    scene_filename: str
    topology: Phase2WTopologyConfig
    wrist_search: Phase2WWristSearchConfig
    geometry: Phase2WGeometryConfig
    second_grasp: Phase2WSecondGraspConfig
    endpoint_population: Phase2WEndpointPopulationConfig
    calibration: Phase2WCalibrationConfig
    seeds: Phase2WSeeds


def load_phase2w_config(path: str | Path | None = None) -> tuple[Phase2WConfig, Path]:
    source = (Path(path) if path is not None else ROOT / "configs" / "phase2W_static_wrist_feasibility.yaml").resolve()
    payload: dict[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8"))
    cfg = Phase2WConfig(
        experiment_id=str(payload["experiment_id"]),
        output_dir=str(payload["output_dir"]),
        scene_filename=str(payload["scene_filename"]),
        topology=Phase2WTopologyConfig(**payload["topology"]),
        wrist_search=Phase2WWristSearchConfig(**payload["wrist_search"]),
        geometry=Phase2WGeometryConfig(**payload["geometry"]),
        second_grasp=Phase2WSecondGraspConfig(**payload["second_grasp"]),
        endpoint_population=Phase2WEndpointPopulationConfig(**payload["endpoint_population"]),
        calibration=Phase2WCalibrationConfig(**payload["calibration"]),
        seeds=Phase2WSeeds(**payload["seeds"]),
    )
    if cfg.experiment_id != PHASE2W_EXPERIMENT_ID:
        raise ValueError("Phase 2W experiment namespace changed")
    if cfg.scene_filename != "scene_two_object_half_scale.yaml":
        raise ValueError("Phase 2W must preserve the Phase 2S/2T/2TR scene")
    if tuple(cfg.topology.occupied_fingers) != OCCUPIED_FINGERS or tuple(cfg.topology.free_fingers) != FREE_FINGERS:
        raise ValueError("Phase 2W requires middle+ring occupied and index+thumb free")
    if cfg.topology.load_bearing_threshold_N != 0.20:
        raise ValueError("Phase 2W occupied threshold changed")
    wrist = cfg.wrist_search
    expected = [-90.0, -45.0, 0.0, 45.0, 90.0]
    if any([wrist.coarse_roll_deg != expected, wrist.coarse_pitch_deg != expected, wrist.coarse_yaw_deg != expected]):
        raise ValueError("Phase 2W coarse orientation grid changed")
    if wrist.refinement_offsets_deg != [-22.5, 0.0, 22.5]:
        raise ValueError("Phase 2W refinement offsets changed")
    if (wrist.screening_states_per_group, wrist.minimum_valid_states_per_group) != (20, 10):
        raise ValueError("Phase 2W endpoint screening gate changed")
    if wrist.top_coarse_for_refinement != 5 or wrist.top_dynamic_candidates != 10 or wrist.maximum_workers > 8:
        raise ValueError("Phase 2W selection or worker limit changed")
    geometry = cfg.geometry
    if geometry.candidate_B_poses_per_orientation < 5000:
        raise ValueError("Phase 2W requires at least 5000 geometry candidates per promising wrist")
    second = cfg.second_grasp
    if (second.initial_candidates_per_wrist, second.expanded_candidates_per_wrist, second.global_candidate_cap) != (512, 2048, 8192):
        raise ValueError("Phase 2W B-only search caps changed")
    if (second.hard_minimum_successes, second.preferred_successes) != (3, 5):
        raise ValueError("Phase 2W B-only success gate changed")
    if second.robustness_trials_per_configuration < 100 or second.robustness_configuration_cap != 3:
        raise ValueError("Phase 2W robustness protocol changed")
    population = cfg.endpoint_population
    if (population.target_per_group, population.minimum_per_group, population.additional_attempt_cap_per_group) != (80, 50, 20_000):
        raise ValueError("Phase 2W endpoint population gate changed")
    if population.calibration_states_per_group != 20:
        raise ValueError("Phase 2W calibration reservation changed")
    calibration = cfg.calibration
    if (calibration.maximum_controller_candidates, calibration.formal_B_seeds_per_pair) != (2048, 20):
        raise ValueError("Phase 2W controller/formal limits changed")
    if (calibration.target_matched_pairs, calibration.minimum_matched_pairs) != (80, 50):
        raise ValueError("Phase 2W matching gate changed")
    seed_values = tuple(vars(cfg.seeds).values())
    if len(seed_values) != len(set(seed_values)):
        raise ValueError("Phase 2W seed namespaces must be distinct")
    return cfg, source
