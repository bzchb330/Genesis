from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import ROOT


PHASE2TR_EXPERIMENT_ID = "phase2TR_index_thumb_matched"
OCCUPIED_FINGERS = ("middle", "ring")
FREE_FINGERS = ("index", "thumb")


@dataclass(frozen=True)
class Phase2TRTopologyConfig:
    occupied_fingers: list[str]
    free_fingers: list[str]
    load_bearing_threshold_N: float


@dataclass(frozen=True)
class Phase2TRStateSearchConfig:
    seed: int
    fingertip_target: int
    fingertip_minimum: int
    fingertip_attempt_cap: int
    palmar_target: int
    palmar_minimum: int
    palmar_attempt_cap: int
    maximum_workers: int


@dataclass(frozen=True)
class Phase2TRMatchingConfig:
    target_pairs: int
    minimum_pairs: int
    calibration_per_group: int
    covariates: list[str]


@dataclass(frozen=True)
class Phase2TRSecondGraspConfig:
    b_only_seed: int
    geometry_seed: int
    calibration_seed: int
    formal_seed: int
    b_only_candidate_cap: int
    b_only_success_target: int
    b_only_hard_minimum: int
    robustness_trials: int
    maximum_controller_candidates: int
    formal_B_seeds_per_state: int


@dataclass(frozen=True)
class Phase2TRConfig:
    experiment_id: str
    output_dir: str
    scene_filename: str
    topology: Phase2TRTopologyConfig
    state_search: Phase2TRStateSearchConfig
    matching: Phase2TRMatchingConfig
    second_grasp: Phase2TRSecondGraspConfig


def load_phase2tr_config(path: str | Path | None = None) -> tuple[Phase2TRConfig, Path]:
    source = (Path(path) if path is not None else ROOT / "configs" / "phase2TR_index_thumb_control.yaml").resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    cfg = Phase2TRConfig(
        experiment_id=str(payload["experiment_id"]),
        output_dir=str(payload["output_dir"]),
        scene_filename=str(payload["scene_filename"]),
        topology=Phase2TRTopologyConfig(**payload["topology"]),
        state_search=Phase2TRStateSearchConfig(**payload["state_search"]),
        matching=Phase2TRMatchingConfig(**payload["matching"]),
        second_grasp=Phase2TRSecondGraspConfig(**payload["second_grasp"]),
    )
    if cfg.experiment_id != PHASE2TR_EXPERIMENT_ID:
        raise ValueError("Phase 2T-R experiment namespace changed")
    if cfg.scene_filename != "scene_two_object_half_scale.yaml":
        raise ValueError("Phase 2T-R must preserve the Phase 2S/2T scene")
    if tuple(cfg.topology.occupied_fingers) != OCCUPIED_FINGERS or tuple(cfg.topology.free_fingers) != FREE_FINGERS:
        raise ValueError("Phase 2T-R must fix middle+ring occupied and index+thumb free")
    if cfg.topology.load_bearing_threshold_N != 0.20:
        raise ValueError("Phase 2T-R load-bearing threshold changed")
    state = cfg.state_search
    if (state.fingertip_target, state.fingertip_minimum, state.fingertip_attempt_cap) != (100, 50, 50_000):
        raise ValueError("Phase 2T-R fingertip search limits changed")
    if (state.palmar_target, state.palmar_minimum, state.palmar_attempt_cap) != (100, 50, 30_000):
        raise ValueError("Phase 2T-R palmar search limits changed")
    if state.maximum_workers > 8:
        raise ValueError("Phase 2T-R permits at most eight workers")
    matching = cfg.matching
    if (matching.target_pairs, matching.minimum_pairs, matching.calibration_per_group) != (80, 50, 20):
        raise ValueError("Phase 2T-R matching limits changed")
    second = cfg.second_grasp
    if (second.b_only_candidate_cap, second.b_only_success_target, second.b_only_hard_minimum) != (4096, 5, 3):
        raise ValueError("Phase 2T-R B-only gate changed")
    if second.robustness_trials < 100 or second.maximum_controller_candidates != 2048:
        raise ValueError("Phase 2T-R robustness/calibration limits changed")
    if second.formal_B_seeds_per_state != 20:
        raise ValueError("Phase 2T-R requires twenty formal B seeds")
    seeds = {state.seed, second.b_only_seed, second.geometry_seed, second.calibration_seed, second.formal_seed}
    if len(seeds) != 5:
        raise ValueError("Phase 2T-R seed namespaces must be distinct")
    return cfg, source


def assert_index_thumb_free_topology(record: dict) -> None:
    occupied = tuple(finger for finger, flag in zip(("index", "middle", "ring", "thumb"), record["occupied_finger_mask"]) if flag)
    free = tuple(finger for finger, flag in zip(("index", "middle", "ring", "thumb"), record["occupied_finger_mask"]) if not flag)
    if occupied != OCCUPIED_FINGERS or free != FREE_FINGERS:
        raise ValueError(f"ineligible Phase 2T-R topology: occupied={occupied}, free={free}")
