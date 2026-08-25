from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..config import ROOT


FINGERS = ("thumb", "index", "middle", "ring", "little")
SUPPORT_SURFACES = (*FINGERS, "palm")


@dataclass(frozen=True)
class ShadowHandConfig:
    model_path: str
    source_repository: str
    source_commit: str
    model_name: str
    palm_body: str
    forearm_body: str
    wrist_joints: tuple[str, ...]
    finger_order: tuple[str, ...]
    finger_bodies: dict[str, tuple[str, ...]]
    finger_joints: dict[str, tuple[str, ...]]
    fingertip_bodies: dict[str, str]
    actuator_groups: dict[str, tuple[str, ...]]
    mount_pos: tuple[float, ...]
    mount_quat: tuple[float, ...]


@dataclass(frozen=True)
class Phase3Config:
    hand: ShadowHandConfig
    raw: dict[str, Any]

    @property
    def object(self) -> dict[str, Any]:
        return self.raw["object"]

    @property
    def diagnostic(self) -> dict[str, Any]:
        return self.raw["diagnostic"]

    @property
    def control(self) -> dict[str, Any]:
        return self.raw["control"]


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def load_phase3_config(config_dir: str | Path | None = None) -> Phase3Config:
    directory = Path(config_dir) if config_dir else ROOT / "configs"
    hand_raw = _read(directory / "hand_shadow_right.yaml")
    phase_raw = _read(directory / "phase3A_shadow_hand.yaml")
    for key in ("wrist_joints", "finger_order", "mount_pos", "mount_quat"):
        hand_raw[key] = tuple(hand_raw[key])
    for key in ("finger_bodies", "finger_joints", "actuator_groups"):
        hand_raw[key] = {name: tuple(values) for name, values in hand_raw[key].items()}
    hand = ShadowHandConfig(**hand_raw)
    if hand.finger_order != FINGERS:
        raise ValueError(f"Shadow semantic finger order must be {FINGERS}")
    if set(hand.fingertip_bodies) != set(FINGERS):
        raise ValueError("all five semantic fingertip bodies must be configured")
    return Phase3Config(hand=hand, raw=phase_raw)
