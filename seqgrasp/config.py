from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class HandConfig:
    model_path: str; dof_count: int; actuator_names: list[str]; joint_names: list[str]
    fingertip_bodies: list[str]; palm_body: str; finger_geom_mapping: dict[str, list[str]]
    mount_pos: list[float]; mount_quat: list[float]

@dataclass(frozen=True)
class ObjectConfig:
    name: str; shape: str; size: list[float]; mass: float; friction: list[float]
    pos: list[float]; rgba: list[float]

@dataclass(frozen=True)
class SceneConfig:
    timestep: float; frame_skip: int; render_width: int; render_height: int
    table_size: list[float]; table_pos: list[float]
    workspace_low: list[float]; workspace_high: list[float]; placement_jitter_xy: float
    objects: list[ObjectConfig]

@dataclass(frozen=True)
class TaskConfig:
    control_mode: str; episode_steps: int; impedance_stiffness: float
    impedance_damping: float; torque_limit: float; residual_scale: float
    residual_limit: float; observations: dict[str, bool]; tactile_normalization: float | None
    extra_tactile_feature_dim: int; grasp_contact_force_threshold: float | None
    hold_steps_threshold: int | None; drop_height_threshold: float | None
    reward_weights: dict[str, float]

@dataclass(frozen=True)
class TrainConfig:
    total_timesteps: int; n_envs: int; seed: int; checkpoint_freq: int
    log_dir: str; checkpoint_dir: str

@dataclass(frozen=True)
class DiagnosticConfig:
    diagnostic_only: bool; seed: int; object_name: str
    object_fixture_pos: list[float]; object_fixture_quat: list[float]
    fixture_jitter_xy: float; stage_durations_seconds: dict[str, float]
    kinematic_fixture_stages: list[str]; episode_phase_by_stage: dict[str, int]
    open_joint_fractions: dict[str, float]; closed_joint_fractions: dict[str, float]
    output_dir: str; save_csv: bool; save_npz: bool; save_plots: bool
    render_video: bool; video_filename: str; video_fps: int; render_stride: int
    num_seeds: int

@dataclass(frozen=True)
class ConfigBundle:
    hand: HandConfig; scene: SceneConfig; task: TaskConfig
    train: TrainConfig; diagnostic: DiagnosticConfig

def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f: return yaml.safe_load(f)

def load_configs(config_dir: str | Path | None = None) -> ConfigBundle:
    d = Path(config_dir) if config_dir else ROOT / "configs"
    h, s, t, tr, diag = (_read(d / n) for n in ("hand_allegro.yaml", "scene_two_object.yaml", "task_sequential.yaml", "train_ppo.yaml", "diagnostic_grasp_a.yaml"))
    s["objects"] = [ObjectConfig(**x) for x in s["objects"]]
    diagnostic = DiagnosticConfig(**diag)
    if not diagnostic.diagnostic_only:
        raise ValueError("diagnostic_grasp_a.yaml must remain diagnostic_only")
    return ConfigBundle(HandConfig(**h), SceneConfig(**s), TaskConfig(**t), TrainConfig(**tr), diagnostic)
