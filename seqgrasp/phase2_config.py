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
class Phase2Config:
    phase2_only: bool
    validation: PhysicsValidationConfig
    sweep: ContactSweepConfig
    persistence: PersistenceConfig
    required_for_later_parts: LaterPhaseInputs


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
    )
    if not cfg.phase2_only:
        raise ValueError("Phase 2 configuration must remain phase2_only")
    if cfg.validation.long_hold_steps < 1000:
        raise ValueError("Phase 2 long hold must contain at least 1000 simulation steps")
    if cfg.validation.force_order_factor != 10.0:
        raise ValueError("the Phase 2 force sanity factor is specified as one order of magnitude")
    if not 200 <= cfg.required_for_later_parts.accepted_grasp_target <= 500:
        raise ValueError("accepted_grasp_target must remain within the Phase 2 range 200-500")
    validate_contact_sweep_shapes(cfg.sweep)
    return cfg, source
