from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Phase3RewardTerms:
    object_progress_to_palm: float
    valid_support: float
    palm_contact: float
    support_transfer: float
    acquisition_finger_release_after_support: float
    recovered_resource: float
    complete_object_loss: float
    unsafe_penetration: float
    joint_limit: float
    violent_action: float

    def as_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in self.__dict__.items()}


def weighted_reward(terms: Phase3RewardTerms, weights: dict[str, float]) -> float:
    # TODO(PI): scientific reward weights remain zero placeholders in Phase 3A.
    return float(sum(weights[name] * value for name, value in terms.as_dict().items()))
