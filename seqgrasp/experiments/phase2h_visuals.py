from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import numpy as np

from ..config import ConfigBundle
from ..phase2_5_config import Phase25Config
from .phase2_5_trajectory import BAcquisitionTrajectory, run_b_acquisition_trajectory
from .second_grasp import BPlacement, _rotation_change
from .static_wrist import compose_mount_quaternion_wxyz


FINGER_NAMES = ("index", "middle", "ring", "thumb")
INDEX, MIDDLE, RING, THUMB = range(4)


def scene_for_trial(base_cfg: ConfigBundle, row: dict) -> ConfigBundle:
    quaternion = compose_mount_quaternion_wxyz(
        base_cfg.hand.mount_quat,
        row["wrist_pose"]["relative_quaternion_wxyz"],
    )
    return replace(base_cfg, hand=replace(base_cfg.hand, mount_quat=quaternion.tolist()))


def trajectory_from_trial(row: dict) -> BAcquisitionTrajectory:
    payload = row["trajectory"]
    return BAcquisitionTrajectory(
        candidate_index=int(payload["candidate_index"]),
        approach_joint_rad=tuple(float(value) for value in payload["approach_joint_rad"]),
        precontact_joint_rad=tuple(float(value) for value in payload["precontact_joint_rad"]),
        closing_joint_rad=tuple(float(value) for value in payload["closing_joint_rad"]),
        hold_joint_rad=tuple(float(value) for value in payload["hold_joint_rad"]),
        close_steps=int(payload["close_steps"]),
        per_finger_close_delay_steps=tuple(int(value) for value in payload["per_finger_close_delay_steps"]),
        fixture_release_delay_steps=int(payload["fixture_release_delay_steps"]),
    )


def placement_from_trial(row: dict) -> BPlacement:
    payload = row["placement"]
    return BPlacement(
        index=int(row["candidate_index"]),
        position_m=tuple(float(value) for value in payload["position_m"]),
        quaternion=tuple(float(value) for value in payload["quaternion"]),
        yaw_rad=float(payload["yaw_rad"]),
    )


def replay_trial(
    row: dict,
    cfg25: Phase25Config,
    base_cfg: ConfigBundle,
    *,
    diagnostic_callback=None,
) -> tuple[dict, dict[str, np.ndarray]]:
    summary, arrays = run_b_acquisition_trajectory(
        cfg25,
        trajectory_from_trial(row),
        placement=placement_from_trial(row),
        scene_cfg=scene_for_trial(base_cfg, row),
        collect_timeseries=True,
        diagnostic_callback=diagnostic_callback,
    )
    if arrays is None:
        raise AssertionError("Phase 2H replay requires timeseries")
    return summary, arrays


def _prefix_length(values: Iterable[bool]) -> int:
    count = 0
    for value in values:
        if not value:
            break
        count += 1
    return count


def _maximum_run(values: np.ndarray) -> int:
    padded = np.r_[False, np.asarray(values, dtype=bool), False].astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return int(np.max(edges[1::2] - edges[::2])) if len(edges) else 0


def diagnostic_series(summary: dict, arrays: dict[str, np.ndarray], cfg25: Phase25Config) -> dict[str, np.ndarray]:
    release = int(summary["fixture_release_timestep"])
    reference_position = arrays["B_position_m"][release - 1]
    reference_quaternion = arrays["B_quaternion"][release - 1]
    translation = np.linalg.norm(arrays["B_position_m"] - reference_position, axis=1)
    rotation = np.asarray([
        _rotation_change(quaternion, reference_quaternion)
        for quaternion in arrays["B_quaternion"]
    ])
    vertical = arrays["B_position_m"][:, 2] - reference_position[2]
    flags = np.asarray(arrays["B_per_finger_contact_flag"], dtype=bool)
    both = flags[:, INDEX] & flags[:, THUMB]
    assist = flags[:, MIDDLE] | flags[:, RING]
    penetration = np.max(np.asarray(arrays["B_penetration_depths_m"]), axis=1)
    strict = np.zeros(len(flags), dtype=bool)
    both_before_release = bool(np.any(both[:release] & ~assist[:release]))
    no_assist_before_release = not bool(np.any(assist[:release]))
    post = slice(release, len(flags))
    table_history = np.maximum.accumulate(np.asarray(arrays["B_table_contact"], dtype=bool)[post])
    penetration_history = np.maximum.accumulate(penetration)[post]
    translation_history = np.maximum.accumulate(translation[post])
    rotation_history = np.maximum.accumulate(rotation[post])
    free_contacts = np.asarray(arrays["B_free_finger_contacts"], dtype=int)
    hand_contacts = np.asarray(arrays["B_hand_contacts"], dtype=int)
    normal_force = np.asarray(arrays["B_hand_normal_force_N"], dtype=float)
    strict[post] = (
        bool(summary["numerically_valid"])
        & both_before_release
        & no_assist_before_release
        & ~assist[post]
        & (free_contacts[post] >= cfg25.criteria.minimum_B_free_finger_contacts)
        & (hand_contacts[post] >= cfg25.criteria.minimum_B_hand_contacts)
        & (normal_force[post] >= cfg25.criteria.minimum_B_normal_force_N)
        & ~table_history
        & (penetration_history <= cfg25.criteria.maximum_penetration_m)
        & (translation_history <= cfg25.criteria.maximum_B_translation_m)
        & (rotation_history <= cfg25.criteria.maximum_B_orientation_rad)
    )
    return {
        "translation_m": translation,
        "rotation_rad": rotation,
        "vertical_displacement_m": vertical,
        "contact_flags": flags,
        "both_contact": both,
        "assist_contact": assist,
        "penetration_m": penetration,
        "free_finger_contacts": free_contacts,
        "hand_contacts": hand_contacts,
        "normal_force_N": normal_force,
        "strict_state": strict,
    }


def first_strict_failure(series: dict[str, np.ndarray], release: int, cfg25: Phase25Config) -> tuple[int | None, str | None]:
    strict_post = series["strict_state"][release:]
    failure = np.flatnonzero(~strict_post)
    if not len(failure):
        return None, None
    step = release + int(failure[0])
    flags = series["contact_flags"][step]
    if series["hand_contacts"][step] < cfg25.criteria.minimum_B_hand_contacts:
        mechanism = "INDEX_AND_THUMB_CONTACT_LOST"
    elif series["free_finger_contacts"][step] < cfg25.criteria.minimum_B_free_finger_contacts:
        mechanism = "FREE_FINGER_CONTACT_LOST"
    elif series["normal_force_N"][step] < cfg25.criteria.minimum_B_normal_force_N:
        mechanism = "NORMAL_FORCE_BELOW_THRESHOLD"
    elif series["assist_contact"][step]:
        mechanism = "MIDDLE_OR_RING_ASSIST"
    elif series["penetration_m"][step] > cfg25.criteria.maximum_penetration_m:
        mechanism = "PENETRATION_THRESHOLD"
    elif series["translation_m"][step] > cfg25.criteria.maximum_B_translation_m:
        mechanism = "TRANSLATION_THRESHOLD"
    elif series["rotation_rad"][step] > cfg25.criteria.maximum_B_orientation_rad:
        mechanism = "ROTATION_THRESHOLD"
    else:
        mechanism = "TABLE_CONTACT"
    return step, mechanism


def trial_metrics(row: dict, summary: dict, arrays: dict[str, np.ndarray], cfg25: Phase25Config) -> dict:
    release = int(summary["fixture_release_timestep"])
    series = diagnostic_series(summary, arrays, cfg25)
    dual_post = series["both_contact"][release:] & ~series["assist_contact"][release:]
    strict_post = series["strict_state"][release:]
    failure_step, failure_mechanism = first_strict_failure(series, release, cfg25)
    flags = series["contact_flags"]
    pre = slice(0, release)
    first_contact_loss = np.flatnonzero(~series["both_contact"][release:])
    first_table = np.flatnonzero(np.asarray(arrays["B_table_contact"][release:], dtype=bool))
    first_rotation = np.flatnonzero(series["rotation_rad"][release:] > cfg25.criteria.maximum_B_orientation_rad)
    return {
        "trial_id": row["trial_id"],
        "source_config_hash": row["config_hash"],
        "wrist_pose_id": row["wrist_pose_id"],
        "wrist_rpy_deg": row["wrist_pose"]["relative_rpy_deg"],
        "trajectory_proposal_center": row["trajectory_proposal_center"],
        "candidate_index": int(row["candidate_index"]),
        "failure_mechanism": row["failure_mechanism"],
        "fixture_release_timestep": release,
        "index_contact_established_pre_release": bool(np.any(flags[pre, INDEX])),
        "thumb_contact_established_pre_release": bool(np.any(flags[pre, THUMB])),
        "dual_contact_pre_release": bool(np.any(series["both_contact"][pre] & ~series["assist_contact"][pre])),
        "dual_contact_at_release": bool(dual_post[0]),
        "dual_contact_survival_steps": _prefix_length(dual_post),
        "maximum_dual_contact_run_steps": _maximum_run(dual_post),
        "strict_survival_steps": _prefix_length(strict_post),
        "first_strict_failure_timestep": failure_step,
        "first_strict_failure_relative_step": None if failure_step is None else failure_step - release,
        "first_strict_failure_mechanism": failure_mechanism,
        "first_dual_contact_loss_relative_step": None if not len(first_contact_loss) else int(first_contact_loss[0]),
        "first_table_contact_relative_step": None if not len(first_table) else int(first_table[0]),
        "first_rotation_violation_relative_step": None if not len(first_rotation) else int(first_rotation[0]),
        "unsupported_any_hand_contact_steps": int(summary["unsupported_contact_steps"]),
        "maximum_B_translation_after_release_m": float(summary["maximum_B_translation_after_release_m"]),
        "maximum_B_orientation_after_release_rad": float(summary["maximum_B_orientation_after_release_rad"]),
        "replay_failure_mechanism": summary["failure_mechanism"],
    }


def assert_replay_matches(row: dict, summary: dict) -> None:
    exact_keys = (
        "candidate_index", "fixture_release_timestep", "failure_mechanism",
        "B_table_contact_after_release", "first_post_release_contact_loss_step",
        "unsupported_contact_steps", "numerically_valid",
    )
    for key in exact_keys:
        if row[key] != summary[key]:
            raise AssertionError(f"Phase 2H replay mismatch for {row['trial_id']} field {key}")
    numeric_keys = (
        "maximum_B_penetration_m", "maximum_B_translation_after_release_m",
        "maximum_B_orientation_after_release_rad",
    )
    for key in numeric_keys:
        if not np.isclose(float(row[key]), float(summary[key]), rtol=0.0, atol=1e-12):
            raise AssertionError(f"Phase 2H replay mismatch for {row['trial_id']} field {key}")
