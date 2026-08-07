from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class GraspCriterionState:
    object_pose: Any
    object_velocity: Any
    table_clearance: float
    active_fingers: Any
    tactile_features: dict
    elapsed_time: float
    workspace_exit: bool

def is_grasp_acquired(state:GraspCriterionState,cfg)->None:
    # TODO(PI): define grasp-acquisition evidence and persistence criteria.
    return None

def is_object_retained(state:GraspCriterionState,cfg)->None:
    # TODO(PI): define unsupported persistent-retention criteria.
    return None

def is_object_lost(state:GraspCriterionState,cfg)->None:
    # TODO(PI): define object loss/drop criteria beyond mechanical workspace exit.
    return None
