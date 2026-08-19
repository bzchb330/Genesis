from __future__ import annotations

from dataclasses import replace
import math

import numpy as np

from .phase2_5_trajectory import BAcquisitionTrajectory, run_b_acquisition_trajectory
from .phase2r import GraspStateType, PHASE2R_OUTCOMES, digit_precheck_outcome
from .resource_components import FINGER_ORDER


ACQUISITION_PAIR_PRIORITY = (
    ("index", "thumb"),
    ("middle", "thumb"),
    ("index", "middle"),
    ("ring", "thumb"),
    ("index", "ring"),
    ("middle", "ring"),
)


def select_static_acquisition_pair(occupied_mask) -> tuple[str, str] | None:
    free = {finger for finger, occupied in zip(FINGER_ORDER, occupied_mask) if not occupied}
    if len(free) < 2:
        return None
    return next(pair for pair in ACQUISITION_PAIR_PRIORITY if set(pair).issubset(free))


def remap_index_thumb_trajectory(
    trajectory: BAcquisitionTrajectory, acquisition_pair: tuple[str, str], baseline_joint_rad,
) -> BAcquisitionTrajectory:
    """Map the frozen index/thumb family to a fixed geometry-selected pair."""

    finger_index = {finger: index for index, finger in enumerate(FINGER_ORDER)}
    if "thumb" in acquisition_pair:
        first = next(finger for finger in acquisition_pair if finger != "thumb")
        mappings = ((first, "index"), ("thumb", "thumb"))
    else:
        mappings = ((acquisition_pair[0], "index"), (acquisition_pair[1], "thumb"))

    def mapped(values):
        source = np.asarray(values, dtype=float)
        target = np.asarray(baseline_joint_rad, dtype=float).copy()
        for destination, origin in mappings:
            d = slice(4 * finger_index[destination], 4 * finger_index[destination] + 4)
            o = slice(4 * finger_index[origin], 4 * finger_index[origin] + 4)
            target[d] = source[o]
        return tuple(float(value) for value in target)

    source_delays = trajectory.per_finger_close_delay_steps
    delays = [0, 0, 0, 0]
    for destination, origin in mappings:
        delays[finger_index[destination]] = source_delays[finger_index[origin]]
    return replace(
        trajectory,
        approach_joint_rad=mapped(trajectory.approach_joint_rad),
        precontact_joint_rad=mapped(trajectory.precontact_joint_rad),
        closing_joint_rad=mapped(trajectory.closing_joint_rad),
        hold_joint_rad=mapped(trajectory.hold_joint_rad),
        per_finger_close_delay_steps=tuple(delays),
    )


def _A_retained(record: dict, arrays: dict[str, np.ndarray], release_step: int, state_cfg) -> tuple[bool, str | None]:
    if not len(arrays["A_position_m"]):
        return False, "A_DESTABILIZED"
    final = slice(release_step, len(arrays["A_position_m"]))
    if np.any(arrays["A_table_contact"]):
        return False, "A_DESTABILIZED"
    if np.max(arrays["A_penetration_m"]) > state_cfg.maximum_penetration_m:
        return False, "A_DESTABILIZED"
    if np.max(arrays["A_displacement_m"]) > state_cfg.maximum_translation_drift_m:
        return False, "A_DESTABILIZED"
    if np.max(arrays["A_rotation_rad"]) > state_cfg.maximum_orientation_drift_rad:
        return False, "A_DESTABILIZED"
    if np.any(arrays["A_hand_contact_count"] == 0):
        return False, "A_DESTABILIZED"
    state_type = GraspStateType(record["grasp_state_type"])
    if state_type is GraspStateType.FINGERTIP:
        if np.any(arrays["A_palm_contact"][final] > 0):
            return False, "A_DESTABILIZED"
        participating = np.mean(arrays["A_per_finger_contact_flag"][final] > 0, axis=0) > 0.0
        return (int(np.sum(participating)) >= state_cfg.minimum_fingertip_contact_fingers, None)
    palm_fraction = float(np.mean(arrays["A_palm_contact"][final] > 0))
    forces = np.mean(arrays["A_all_link_per_finger_normal_force_N"][final], axis=0)
    load_bearing = int(np.sum(forces > state_cfg.load_bearing_force_threshold_N))
    retained = (
        palm_fraction >= state_cfg.palm_contact_fraction_minimum
        and state_cfg.minimum_palmar_load_bearing_fingers <= load_bearing <= state_cfg.maximum_palmar_load_bearing_fingers
    )
    return retained, None if retained else "A_DESTABILIZED"


def run_phase2r_second_grasp_trial(
    cfg25, state_cfg, trajectory: BAcquisitionTrajectory, A_record: dict, placement,
) -> dict:
    precheck = digit_precheck_outcome(A_record)
    pair = select_static_acquisition_pair(A_record["occupied_finger_mask"])
    if precheck is not None:
        return {
            **precheck,
            "A_retained": True,
            "B_acquired": False,
            "acquisition_finger_subset": None,
            "fixture_released": False,
            "initial_invalid_reason": None,
        }
    baseline = np.asarray(
        A_record.get("retaining_joint_target_rad", A_record["final_joint_configuration_rad"]), dtype=float,
    )
    adapted = remap_index_thumb_trajectory(trajectory, pair, baseline)
    summary, arrays = run_b_acquisition_trajectory(
        cfg25, adapted, A_record=A_record,
        occupied_mask=np.asarray(A_record["occupied_finger_mask"], dtype=bool),
        placement=placement, collect_timeseries=True,
    )
    release_step = int(summary["fixture_release_timestep"])
    a_retained, a_subreason = _A_retained(A_record, arrays, release_step, state_cfg)
    b_acquired = bool(summary["B_acquired"])
    if summary.get("invalid_reason"):
        outcome, subreason = "INVALID", "INITIAL_OVERLAP"
    elif a_retained and b_acquired:
        outcome, subreason = "BOTH_RETAINED", None
    elif not a_retained and b_acquired:
        outcome, subreason = "A_DROPPED", a_subreason
    elif a_retained and not b_acquired:
        outcome = "B_NOT_ACQUIRED"
        if summary["maximum_B_hand_contacts_before_release"] == 0:
            subreason = "NO_B_CONTACT"
        elif summary["B_table_contact_after_release"]:
            subreason = "B_SLIP"
        elif summary["first_post_release_contact_loss_step"] is not None:
            subreason = "B_CONTACT_LOST"
        else:
            subreason = "B_NOT_ACQUIRED"
    else:
        outcome, subreason = "BOTH_LOST", a_subreason or "B_CONTACT_LOST"
    if outcome not in PHASE2R_OUTCOMES:
        raise AssertionError(outcome)
    return {
        **summary,
        "outcome": outcome,
        "outcome_subreason": subreason,
        "A_retained": bool(a_retained),
        "B_acquired": b_acquired,
        "BOTH_RETAINED": outcome == "BOTH_RETAINED",
        "dynamic_attempt_executed": True,
        "acquisition_finger_subset": list(pair),
        "controller_source_candidate_index": trajectory.candidate_index,
        "A_final_palm_contact_fraction": float(np.mean(arrays["A_palm_contact"][release_step:] > 0)),
        "A_maximum_penetration_m": float(np.max(arrays["A_penetration_m"])),
    }
