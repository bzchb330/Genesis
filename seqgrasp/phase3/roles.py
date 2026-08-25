from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from .config import FINGERS


class FingerRole(IntEnum):
    FREE = 0
    PROBING = 1
    ACQUIRING = 2
    SUPPORTING = 3
    TRANSFERRING = 4
    RELEASING = 5


class PalmRole(IntEnum):
    NO_CONTACT = 0
    CONTACT = 1
    SECURE_SUPPORT = 2


class ManipulationPhase(IntEnum):
    PROBE = 0
    MINIMAL_ACQUIRE = 1
    RECRUIT = 2
    TRANSFER = 3
    PALMAR_SECURE = 4
    RELEASE_ACQUISITION_FINGERS = 5
    RESOURCE_RECOVERED = 6


@dataclass
class RoleState:
    phase: ManipulationPhase = ManipulationPhase.PROBE
    fingers: dict[str, FingerRole] = field(
        default_factory=lambda: {finger: FingerRole.FREE for finger in FINGERS}
    )
    palm: PalmRole = PalmRole.NO_CONTACT

    def begin_probe(self) -> None:
        self.phase = ManipulationPhase.PROBE
        self.fingers["thumb"] = FingerRole.PROBING
        self.fingers["index"] = FingerRole.PROBING

    def acquisition_contact(self) -> None:
        self.phase = ManipulationPhase.MINIMAL_ACQUIRE
        self.fingers["thumb"] = FingerRole.ACQUIRING
        self.fingers["index"] = FingerRole.ACQUIRING

    def recruit(self, finger: str = "middle") -> None:
        if finger not in FINGERS:
            raise ValueError(f"unknown finger {finger}")
        self.phase = ManipulationPhase.RECRUIT
        self.fingers[finger] = FingerRole.SUPPORTING

    def begin_transfer(self) -> None:
        self.phase = ManipulationPhase.TRANSFER
        for finger in ("thumb", "index"):
            if self.fingers[finger] == FingerRole.ACQUIRING:
                self.fingers[finger] = FingerRole.TRANSFERRING

    def palm_contact(self) -> None:
        self.palm = PalmRole.CONTACT

    def palmar_secure_diagnostic(self) -> None:
        self.phase = ManipulationPhase.PALMAR_SECURE
        self.palm = PalmRole.SECURE_SUPPORT

    def begin_release(self, fingers: tuple[str, ...]) -> None:
        self.phase = ManipulationPhase.RELEASE_ACQUISITION_FINGERS
        for finger in fingers:
            self.fingers[finger] = FingerRole.RELEASING

    def resource_recovered(self, finger: str) -> None:
        self.fingers[finger] = FingerRole.FREE
        self.phase = ManipulationPhase.RESOURCE_RECOVERED
