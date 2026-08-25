from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import ROOT


PHASE2CM_EXPERIMENT_ID = "phase2CM_contact_model_audit"


@dataclass(frozen=True)
class Phase2CMConfig:
    experiment_id: str
    output_dir: str
    source_scene_filename: str
    source_metric_method_id: str
    runtime_contact_audit_trials: int
    paired_release_state_target: int
    post_release_steps: int
    bootstrap_resamples: int
    bootstrap_seed: int
    variants: dict[str, int]


def load_phase2cm_config(path: str | Path | None = None) -> tuple[Phase2CMConfig, Path]:
    source = (
        Path(path)
        if path is not None
        else ROOT / "configs" / "phase2CM_contact_model_audit.yaml"
    ).resolve()
    payload: dict[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8"))
    cfg = Phase2CMConfig(
        experiment_id=str(payload["experiment_id"]),
        output_dir=str(payload["output_dir"]),
        source_scene_filename=str(payload["source_scene_filename"]),
        source_metric_method_id=str(payload["source_metric_method_id"]),
        runtime_contact_audit_trials=int(payload["runtime_contact_audit_trials"]),
        paired_release_state_target=int(payload["paired_release_state_target"]),
        post_release_steps=int(payload["post_release_steps"]),
        bootstrap_resamples=int(payload["bootstrap_resamples"]),
        bootstrap_seed=int(payload["bootstrap_seed"]),
        variants={str(name): int(value) for name, value in payload["variants"].items()},
    )
    if cfg.experiment_id != PHASE2CM_EXPERIMENT_ID:
        raise ValueError("Phase 2CM experiment namespace changed")
    if cfg.source_scene_filename != "scene_two_object_half_scale.yaml":
        raise ValueError("Phase 2CM must replay the Phase 2W/2H half-scale scene")
    if cfg.paired_release_state_target != 200 or cfg.post_release_steps != 500:
        raise ValueError("Phase 2CM paired target or frozen replay duration changed")
    if cfg.runtime_contact_audit_trials <= 0:
        raise ValueError("Phase 2CM runtime audit sample must be nonempty")
    if cfg.bootstrap_resamples != 10_000:
        raise ValueError("Phase 2CM requires exactly 10000 paired bootstrap resamples")
    if cfg.variants != {"CM3": 3, "CM4": 4, "CM6": 6}:
        raise ValueError("Phase 2CM variants must be exactly CM3/CM4/CM6")
    return cfg, source
