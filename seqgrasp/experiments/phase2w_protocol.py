from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable

from .static_wrist import FrozenWristBRegion, verify_wrist_b_freeze, wrist_b_integrity_hash


@dataclass(frozen=True)
class FrozenPhase2WController:
    wrist_B_integrity_hash: str
    controller_payload: dict
    calibration_ids: tuple[str, ...]
    integrity_hash: str


def freeze_controller_payload(
    frozen_wrist_B: FrozenWristBRegion,
    controller_payload: dict,
    calibration_ids: Iterable[str],
) -> FrozenPhase2WController:
    verify_wrist_b_freeze(frozen_wrist_B)
    payload = {
        "wrist_B_integrity_hash": frozen_wrist_B.integrity_hash,
        "controller_payload": controller_payload,
        "calibration_ids": tuple(sorted(str(value) for value in calibration_ids)),
    }
    return FrozenPhase2WController(**payload, integrity_hash=wrist_b_integrity_hash(payload))


def verify_controller_freeze(frozen: FrozenPhase2WController) -> None:
    payload = asdict(frozen)
    claimed = payload.pop("integrity_hash")
    if wrist_b_integrity_hash(payload) != claimed:
        raise ValueError("frozen Phase 2W controller integrity check failed")


def calibration_trial_id(calibration_seed: int, state_id: str, B_seed_index: int, controller_index: int) -> str:
    payload = ["phase2W-calibration", int(calibration_seed), state_id, int(B_seed_index), int(controller_index)]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()
