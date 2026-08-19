from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import ROOT


@dataclass(frozen=True)
class PhysicsValidationConfig:
    seed: int
    grasp_profile_path: str
    long_hold_steps: int
    expected_force_order_N: float
    force_order_factor: float
    penetration_tolerance_m: float | None
    maximum_vertical_drift_m: float | None
    maximum_translational_drift_m: float | None
    maximum_orientation_drift_rad: float | None
    minimum_active_object_contacts: int | None
    allow_table_recontact: bool | None
    allow_complete_contact_loss: bool | None


@dataclass(frozen=True)
class ContactSweepConfig:
    target_geom_names: list[str] | None
    friction_vectors: list[list[float]] | None
    solref_values: list[list[float]] | None
    solimp_values: list[list[float]] | None
    timestep_values_s: list[float] | None


@dataclass(frozen=True)
class PersistenceConfig:
    output_dir: str
    lock_timeout_seconds: float
    lock_poll_seconds: float


@dataclass(frozen=True)
class LaterPhaseInputs:
    occupied_finger_force_threshold_N: float | None
    tactile_binary_force_threshold_N: float | None
    short_hold_drift_tolerance_m: float | None
    grasp_acquisition_threshold: float | None
    retained_object_threshold: float | None
    object_loss_drop_threshold: float | None
    invalid_penetration_threshold_m: float | None
    B_placement_low_m: list[float] | None
    B_placement_high_m: list[float] | None
    workspace_monte_carlo_samples: int | None
    workspace_voxel_size_m: float | None
    workspace_collision_tolerance_m: float | None
    free_palm_box_low_m: list[float] | None
    free_palm_box_high_m: list[float] | None
    free_palm_voxel_size_m: float | None
    second_grasp_trials_per_grasp: int | None
    accepted_grasp_target: int


@dataclass(frozen=True)
class GraspDatasetConfig:
    seed: int
    maximum_candidate_attempts: int
    maximum_workers: int
    short_hold_steps: int
    palm_translation_bounds_m: dict[str, list[float]]
    palm_orientation_bounds_deg: dict[str, list[float]]
    object_a_jitter_x_m: list[float]
    object_a_jitter_y_m: list[float]
    object_a_yaw_rad: list[float]
    finger_subset_sizes: list[int]
    active_joint_perturbation_rad: float
    proposal_profile_paths: list[str]
    anchor_profile_path: str
    anchor_commanded_subsets: list[list[str]]
    anchor_palm_translation_half_width_m: float
    anchor_palm_orientation_half_width_deg: float
    anchor_object_jitter_half_width_m: float
    anchor_active_joint_perturbation_rad: float
    maximum_penetration_m: float
    maximum_translation_drift_m: float
    maximum_orientation_drift_rad: float
    allow_table_recontact: bool
    allow_complete_contact_loss: bool
    friction_cone_edges: int
    convex_hull_tolerance: float


@dataclass(frozen=True)
class ResourceExperimentConfig:
    occupied_finger_normal_force_threshold_N: float
    workspace_convergence_samples: list[int]
    workspace_samples: int
    workspace_voxel_size_m: float
    workspace_collision_tolerance_m: float
    free_palm_box_lower_m: list[float]
    free_palm_box_upper_m: list[float]
    free_palm_voxel_size_m: float


@dataclass(frozen=True)
class Phase2TactileConfig:
    binary_contact_threshold_N: float
    zero_normal_epsilon_N: float


@dataclass(frozen=True)
class SecondGraspConfig:
    seed: int
    B_center_x_bounds_m: list[float]
    B_center_y_bounds_m: list[float]
    B_center_z_bounds_m: list[float]
    B_yaw_bounds_rad: list[float]
    placement_preflight_count: int
    representative_grasp_count: int
    geometry_workspace_samples: int
    trials_per_grasp: int
    approach_steps: int
    close_steps: int
    final_hold_steps: int
    minimum_A_finger_contacts: int
    minimum_B_free_finger_contacts: int
    minimum_B_hand_contacts: int
    minimum_A_normal_force_N: float
    minimum_B_normal_force_N: float
    maximum_penetration_m: float
    maximum_A_translation_m: float
    maximum_A_orientation_rad: float
    maximum_B_translation_m: float
    maximum_B_orientation_rad: float
    ik_damping: float
    ik_step_size: float


@dataclass(frozen=True)
class Phase2AnalysisConfig:
    continuous_quantile_bins: int
    confidence_level: float
    greedy_top_fraction: float


@dataclass(frozen=True)
class Phase2Config:
    phase2_only: bool
    validation: PhysicsValidationConfig
    sweep: ContactSweepConfig
    persistence: PersistenceConfig
    required_for_later_parts: LaterPhaseInputs
    dataset: GraspDatasetConfig
    resources: ResourceExperimentConfig
    tactile: Phase2TactileConfig
    second_grasp: SecondGraspConfig
    analysis: Phase2AnalysisConfig


def missing_contact_sweep_inputs(cfg: ContactSweepConfig) -> list[str]:
    return [
        name for name in (
            "target_geom_names", "friction_vectors", "solref_values",
            "solimp_values", "timestep_values_s",
        )
        if not getattr(cfg, name)
    ]


def validate_contact_sweep_shapes(cfg: ContactSweepConfig) -> None:
    if missing_contact_sweep_inputs(cfg):
        return
    if any(len(value) != 3 for value in cfg.friction_vectors):
        raise ValueError("every friction vector must contain slide, spin, and roll")
    if any(len(value) != 2 for value in cfg.solref_values):
        raise ValueError("every solref value must contain mjNREF=2 entries")
    if any(len(value) != 5 for value in cfg.solimp_values):
        raise ValueError("every solimp value must contain mjNIMP=5 entries")
    if any(value <= 0 for value in cfg.timestep_values_s):
        raise ValueError("every sweep timestep must be positive")


def load_phase2_config(path: str | Path | None = None) -> tuple[Phase2Config, Path]:
    source = Path(path) if path is not None else ROOT / "configs" / "phase2_physics_validation.yaml"
    source = source.resolve()
    payload: dict[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8"))
    cfg = Phase2Config(
        phase2_only=bool(payload["phase2_only"]),
        validation=PhysicsValidationConfig(**payload["validation"]),
        sweep=ContactSweepConfig(**payload["sweep"]),
        persistence=PersistenceConfig(**payload["persistence"]),
        required_for_later_parts=LaterPhaseInputs(**payload["required_for_later_parts"]),
        dataset=GraspDatasetConfig(**payload["dataset"]),
        resources=ResourceExperimentConfig(**payload["resources"]),
        tactile=Phase2TactileConfig(**payload["tactile"]),
        second_grasp=SecondGraspConfig(**payload["second_grasp"]),
        analysis=Phase2AnalysisConfig(**payload["analysis"]),
    )
    if not cfg.phase2_only:
        raise ValueError("Phase 2 configuration must remain phase2_only")
    if cfg.validation.long_hold_steps < 1000:
        raise ValueError("Phase 2 long hold must contain at least 1000 simulation steps")
    if cfg.validation.force_order_factor != 10.0:
        raise ValueError("the Phase 2 force sanity factor is specified as one order of magnitude")
    if not 200 <= cfg.required_for_later_parts.accepted_grasp_target <= 500:
        raise ValueError("accepted_grasp_target must remain within the Phase 2 range 200-500")
    if cfg.dataset.maximum_workers > 8:
        raise ValueError("Phase 2 worker count may not exceed 8")
    if cfg.dataset.friction_cone_edges != 8 or cfg.dataset.convex_hull_tolerance != 1e-8:
        raise ValueError("Ferrari-Canny numerical settings must match the PI-supplied values")
    if cfg.dataset.anchor_profile_path not in cfg.dataset.proposal_profile_paths:
        raise ValueError("dataset anchor must be one of the configured proposal centres")
    if cfg.dataset.anchor_active_joint_perturbation_rad > cfg.dataset.active_joint_perturbation_rad:
        raise ValueError("anchor joint width must remain inside the PI-supplied sampling width")
    if any(cfg.dataset.anchor_palm_translation_half_width_m > min(abs(value) for value in cfg.dataset.palm_translation_bounds_m[axis]) for axis in "xyz"):
        raise ValueError("anchor translation widths must remain inside the PI-supplied ranges")
    if any(cfg.dataset.anchor_palm_orientation_half_width_deg > min(abs(value) for value in cfg.dataset.palm_orientation_bounds_deg[axis]) for axis in ("roll", "pitch", "yaw")):
        raise ValueError("anchor orientation widths must remain inside the PI-supplied ranges")
    if cfg.dataset.anchor_object_jitter_half_width_m > min(
        min(abs(value) for value in cfg.dataset.object_a_jitter_x_m),
        min(abs(value) for value in cfg.dataset.object_a_jitter_y_m),
    ):
        raise ValueError("anchor object jitter must remain inside the PI-supplied ranges")
    if cfg.resources.workspace_samples != 10000:
        raise ValueError("production workspace sample count must remain 10000")
    if cfg.second_grasp.trials_per_grasp != 20:
        raise ValueError("Phase 2 requires K=20 second-grasp trials")
    if cfg.second_grasp.placement_preflight_count != 200:
        raise ValueError("the frozen Phase 2 geometry preflight requires 200 B poses")
    if cfg.second_grasp.representative_grasp_count < 20:
        raise ValueError("the geometry preflight requires at least 20 representative A grasps")
    if cfg.second_grasp.minimum_B_free_finger_contacts != 1:
        raise ValueError("the PI-defined B criterion requires one free-finger contact")
    validate_contact_sweep_shapes(cfg.sweep)
    return cfg, source
