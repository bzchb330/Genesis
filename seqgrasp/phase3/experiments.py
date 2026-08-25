from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
import json
from pathlib import Path

import mujoco
import numpy as np

from ..config import ROOT
from .config import FINGERS, SUPPORT_SURFACES, Phase3Config, load_phase3_config
from .contacts import (
    ShadowContactState,
    extract_shadow_contacts,
    fingertip_object_penetration,
    object_velocity,
)
from .control import ContactAwareCloser, actuator_target_from_qpos
from .env import load_keyframe_qpos
from .model import ShadowScene, build_shadow_scene, set_fixture, set_object_pose


class AcquisitionClassification(StrEnum):
    THUMB_INDEX_SUCCESS = "THUMB_INDEX_SUCCESS"
    THUMB_INDEX_CONTACT_BUT_UNSTABLE = "THUMB_INDEX_CONTACT_BUT_UNSTABLE"
    THUMB_INDEX_GEOMETRICALLY_INADEQUATE = "THUMB_INDEX_GEOMETRICALLY_INADEQUATE"
    EXCESSIVE_PENETRATION = "EXCESSIVE_PENETRATION"
    CONTACT_LOSS = "CONTACT_LOSS"
    OBJECT_SLIP = "OBJECT_SLIP"
    OTHER = "OTHER"


def _object_floor_contact(scene: ShadowScene, contacts: ShadowContactState | None = None) -> bool:
    contacts = contacts or extract_shadow_contacts(scene)
    obj = scene.config.object["name"]
    floor = scene.config.raw["floor"]["name"]
    return any(
        obj in {record.body1_name, record.body2_name}
        and floor in {record.geom1_name, record.geom2_name}
        for record in contacts.records
    )


def _contact_pairs(contacts: ShadowContactState) -> list[dict]:
    return [
        {
            "geom_pair": [record.geom1_name, record.geom2_name],
            "body_pair": [record.body1_name, record.body2_name],
            "normal_force_n": record.normal_force,
            "tangential_force_n": record.tangential_force,
            "penetration_m": max(0.0, -record.distance),
        }
        for record in contacts.object_records
    ]


def _sample(scene: ShadowScene, step: int, stage: str) -> dict:
    contacts = extract_shadow_contacts(scene)
    linear, angular = object_velocity(scene)
    palm_id = mujoco.mj_name2id(
        scene.model, mujoco.mjtObj.mjOBJ_BODY, scene.config.hand.palm_body
    )
    return {
        "step": step,
        "time_s": float(scene.data.time),
        "stage": stage,
        "object_position": scene.data.xpos[scene.object_body_id].tolist(),
        "object_quaternion": scene.data.xquat[scene.object_body_id].tolist(),
        "palm_position": scene.data.xpos[palm_id].tolist(),
        "object_linear_velocity": linear.tolist(),
        "object_angular_velocity": angular.tolist(),
        "contact_flags": contacts.contact_flags.tolist(),
        "normal_forces_n": contacts.normal_forces.tolist(),
        "tangential_forces_n": contacts.tangential_forces.tolist(),
        "support_vector_n": contacts.support_vector.tolist(),
        "support_load_fraction": contacts.support_load_fraction.tolist(),
        "penetration_by_surface_m": contacts.penetration_by_surface.tolist(),
        "maximum_penetration_m": contacts.maximum_penetration,
        "maximum_penetration_pair": contacts.maximum_penetration_pair,
        "finger_joint_states": {
            finger: scene.data.qpos[scene.model.jnt_qposadr[ids]].tolist()
            for finger, ids in scene.joint_ids.items()
        },
        "hand_qpos": scene.data.qpos[:24].tolist(),
        "hand_qvel": scene.data.qvel[:24].tolist(),
        "actuator_commands": scene.data.ctrl.tolist(),
    }


def _prepare(scene: ShadowScene, object_position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mujoco.mj_resetData(scene.model, scene.data)
    pre_qpos = load_keyframe_qpos("pre grasp")
    scene.data.qpos[:24] = pre_qpos
    scene.data.qvel[:] = 0.0
    set_object_pose(scene, object_position)
    set_fixture(scene, True)
    pre_target = actuator_target_from_qpos(scene, pre_qpos)
    scene.data.ctrl[:] = pre_target
    mujoco.mj_forward(scene.model, scene.data)
    for _ in range(20):
        mujoco.mj_step(scene.model, scene.data)
    return pre_target, actuator_target_from_qpos(scene, load_keyframe_qpos("two finger pinch"))


def _contact_aware_thumb_index_close(
    scene: ShadowScene,
    pre_target: np.ndarray,
    pinch_target: np.ndarray,
    samples: list[dict],
    step_counter: int,
) -> tuple[ContactAwareCloser, int]:
    cfg = scene.config.diagnostic
    closer = ContactAwareCloser(scene, float(cfg["contact_force_n"]))
    acquisition_ids = np.r_[scene.actuator_ids["thumb"], scene.actuator_ids["index"]]
    for close_step in range(int(cfg["close_steps"])):
        alpha = (close_step + 1) / int(cfg["close_steps"])
        proposed = pre_target.copy()
        proposed[acquisition_ids] = (
            (1.0 - alpha) * pre_target[acquisition_ids] + alpha * pinch_target[acquisition_ids]
        )
        scene.data.ctrl[:] = closer.limit_target(proposed)
        mujoco.mj_step(scene.model, scene.data)
        if close_step % 5 == 0:
            samples.append(_sample(scene, step_counter, "CONTACT_AWARE_CLOSE"))
        step_counter += 1
    for settle_step in range(int(cfg["settle_steps"])):
        mujoco.mj_step(scene.model, scene.data)
        if settle_step % 5 == 0:
            samples.append(_sample(scene, step_counter, "FIXTURE_SETTLE"))
        step_counter += 1
    return closer, step_counter


def _classify(
    dual_at_release: bool,
    release_penetration: float,
    reference_penetration: float,
    ever_dual_support_after_release: bool,
    final_dual_support: bool,
    floor_contact: bool,
) -> AcquisitionClassification:
    if release_penetration > reference_penetration:
        return AcquisitionClassification.EXCESSIVE_PENETRATION
    if not dual_at_release:
        return AcquisitionClassification.THUMB_INDEX_GEOMETRICALLY_INADEQUATE
    if floor_contact:
        return AcquisitionClassification.OBJECT_SLIP
    if not ever_dual_support_after_release:
        return AcquisitionClassification.CONTACT_LOSS
    if not final_dual_support:
        return AcquisitionClassification.THUMB_INDEX_CONTACT_BUT_UNSTABLE
    return AcquisitionClassification.THUMB_INDEX_SUCCESS


def run_acquisition_candidate(
    candidate_id: int,
    object_position: np.ndarray,
    *,
    recruit_middle: bool = False,
    config: Phase3Config | None = None,
) -> dict:
    scene = build_shadow_scene(config)
    samples: list[dict] = []
    pre_target, pinch_target = _prepare(scene, object_position)
    step_counter = 0
    initial = _sample(scene, step_counter, "PRE_GRASP")
    closer, step_counter = _contact_aware_thumb_index_close(
        scene, pre_target, pinch_target, samples, step_counter
    )
    middle_recruited = False
    if recruit_middle:
        middle_target = actuator_target_from_qpos(scene, load_keyframe_qpos("three finger pinch"))
        ids = scene.actuator_ids["middle"]
        start = scene.data.ctrl.copy()
        for recruit_step in range(int(scene.config.diagnostic["close_steps"])):
            alpha = (recruit_step + 1) / int(scene.config.diagnostic["close_steps"])
            proposed = scene.data.ctrl.copy()
            proposed[ids] = (1.0 - alpha) * start[ids] + alpha * middle_target[ids]
            scene.data.ctrl[:] = closer.limit_target(proposed)
            mujoco.mj_step(scene.model, scene.data)
            contacts = extract_shadow_contacts(scene)
            middle_recruited |= bool(contacts.contact_flags[2])
            if recruit_step % 5 == 0:
                samples.append(_sample(scene, step_counter, "MIDDLE_RECRUITMENT"))
            step_counter += 1
    release_contacts = extract_shadow_contacts(scene)
    dual_at_release = bool(release_contacts.contact_flags[0] and release_contacts.contact_flags[1])
    release_state = _sample(scene, step_counter, "RELEASE_STATE")
    release_state["contact_pairs"] = _contact_pairs(release_contacts)
    set_fixture(scene, False)
    ever_dual_support = False
    floor_contact = False
    unsupported_steps = int(scene.config.diagnostic["unsupported_steps"])
    for unsupported_step in range(unsupported_steps):
        mujoco.mj_step(scene.model, scene.data)
        contacts = extract_shadow_contacts(scene)
        ever_dual_support |= bool(contacts.contact_flags[0] and contacts.contact_flags[1])
        floor_contact |= _object_floor_contact(scene, contacts)
        if unsupported_step % 5 == 0:
            samples.append(_sample(scene, step_counter, "UNSUPPORTED_HOLD"))
        step_counter += 1
    final_contacts = extract_shadow_contacts(scene)
    final_dual_support = bool(final_contacts.contact_flags[0] and final_contacts.contact_flags[1]) and not floor_contact
    retained_with_recruited_support = bool(
        recruit_middle
        and middle_recruited
        and not floor_contact
        and np.any(final_contacts.contact_flags[:3])
    )
    classification = _classify(
        dual_at_release,
        release_contacts.maximum_penetration,
        float(scene.config.diagnostic["reference_penetration_m"]),
        ever_dual_support,
        final_dual_support,
        floor_contact,
    )
    return {
        "candidate_id": candidate_id,
        "condition": "thumb_index_middle" if recruit_middle else "thumb_index",
        "object_initial_position": object_position.tolist(),
        "classification": classification.value,
        "initial_state": initial,
        "contact_latches": dict(closer.latched),
        "dual_contact_at_release": dual_at_release,
        "middle_contact_achieved": middle_recruited,
        "retained_with_recruited_support": retained_with_recruited_support,
        "release_state": release_state,
        "final_state": _sample(scene, step_counter, "FINAL"),
        "floor_contact_after_release": floor_contact,
        "samples": samples,
        "diagnostic_reference_penetration_m": float(scene.config.diagnostic["reference_penetration_m"]),
    }


def run_minimal_acquisition_cohort(config: Phase3Config | None = None) -> dict:
    cfg = config or load_phase3_config()
    origin = np.asarray(cfg.object["initial_pos"], dtype=np.float64)
    minimal = [
        run_acquisition_candidate(index, origin + np.asarray(offset), config=cfg)
        for index, offset in enumerate(cfg.diagnostic["cohort_offsets_m"])
    ]
    insufficient = [trial for trial in minimal if trial["classification"] != AcquisitionClassification.THUMB_INDEX_SUCCESS]
    recruited = [
        run_acquisition_candidate(
            trial["candidate_id"], np.asarray(trial["object_initial_position"]), recruit_middle=True, config=cfg
        )
        for trial in insufficient
    ]
    release_penetrations = np.asarray(
        [trial["release_state"]["maximum_penetration_m"] for trial in minimal], dtype=np.float64
    )
    counts = {classification.value: 0 for classification in AcquisitionClassification}
    for trial in minimal:
        counts[trial["classification"]] += 1
    recruited_success = sum(trial["retained_with_recruited_support"] for trial in recruited)
    summary = {
        "seed": int(cfg.raw["seed"]),
        "object": dict(cfg.object),
        "minimal_attempts": len(minimal),
        "minimal_classification_counts": counts,
        "release_penetration_m": {
            "median": float(np.median(release_penetrations)),
            "p95": float(np.percentile(release_penetrations, 95)),
            "maximum": float(np.max(release_penetrations)),
            "values": release_penetrations.tolist(),
        },
        "middle_recruitment_attempts": len(recruited),
        "middle_recruitment_diagnostic_successes": recruited_success,
        "minimal_trials": minimal,
        "middle_recruitment_trials": recruited,
    }
    return summary


def run_handoff_diagnostic(
    config: Phase3Config | None = None,
    *,
    object_position: np.ndarray | None = None,
    support_keyframe: str | None = None,
) -> dict:
    """Execute a dynamics-only fingertip-to-palm handoff attempt.

    Object qpos is set only during initial candidate setup while the fixture is
    active. After fixture release, every object motion is produced by MuJoCo.
    """
    cfg = config or load_phase3_config()
    scene = build_shadow_scene(cfg)
    origin = np.asarray(
        cfg.diagnostic["handoff_initial_pos"] if object_position is None else object_position,
        dtype=np.float64,
    )
    support_keyframe = support_keyframe or str(cfg.diagnostic["handoff_support_keyframe"])
    pre_target, pinch_target = _prepare(scene, origin)
    samples: list[dict] = [_sample(scene, 0, "PRE_GRASP")]
    closer, step_counter = _contact_aware_thumb_index_close(scene, pre_target, pinch_target, samples, 1)
    release_state = _sample(scene, step_counter, "FIXTURE_RELEASE")
    set_fixture(scene, False)
    for hold_step in range(50):
        mujoco.mj_step(scene.model, scene.data)
        if hold_step % 2 == 0:
            samples.append(_sample(scene, step_counter, "MINIMAL_UNSUPPORTED_HOLD"))
        step_counter += 1
    support_target = actuator_target_from_qpos(scene, load_keyframe_qpos(support_keyframe))
    reference_penetration = float(cfg.diagnostic["reference_penetration_m"])

    def move_group_toward(group: str, target: np.ndarray, step_size: float = 0.0005) -> None:
        ids = scene.actuator_ids[group]
        scene.data.ctrl[ids] += np.clip(target[ids] - scene.data.ctrl[ids], -step_size, step_size)

    def regulated_acquisition_transfer() -> None:
        for finger in ("thumb", "index"):
            if fingertip_object_penetration(scene, finger) < reference_penetration:
                move_group_toward(finger, support_target)

    # Recruit middle first. Acquisition fingers may continue a slow, penetration-
    # guarded transfer motion; this is not the blind full closure used in Phase 2.
    for transfer_step in range(350):
        regulated_acquisition_transfer()
        move_group_toward("wrist", support_target, step_size=0.0002)
        move_group_toward("middle", support_target)
        mujoco.mj_step(scene.model, scene.data)
        if transfer_step % 2 == 0:
            samples.append(_sample(scene, step_counter, "MIDDLE_FIRST_DYNAMIC_TRANSFER"))
        step_counter += 1
    # Ring and little remain progressive: they move only after the middle stage.
    for support_step in range(350):
        regulated_acquisition_transfer()
        move_group_toward("wrist", support_target, step_size=0.0002)
        move_group_toward("ring", support_target)
        move_group_toward("little", support_target)
        mujoco.mj_step(scene.model, scene.data)
        if support_step % 2 == 0:
            samples.append(_sample(scene, step_counter, "RING_LITTLE_SUPPORT"))
        step_counter += 1
    release_target = actuator_target_from_qpos(scene, load_keyframe_qpos("open hand"))
    release_fingers = tuple(cfg.diagnostic["handoff_release_fingers"])
    acquisition_groups = np.concatenate([scene.actuator_ids[finger] for finger in release_fingers])
    release_start = scene.data.ctrl.copy()
    for release_step in range(300):
        alpha = (release_step + 1) / 300
        scene.data.ctrl[acquisition_groups] = (
            (1.0 - alpha) * release_start[acquisition_groups]
            + alpha * release_target[acquisition_groups]
        )
        mujoco.mj_step(scene.model, scene.data)
        if release_step % 2 == 0:
            samples.append(_sample(scene, step_counter, "ACQUISITION_FINGER_RELEASE"))
        step_counter += 1
    for settle_step in range(150):
        mujoco.mj_step(scene.model, scene.data)
        if settle_step % 2 == 0:
            samples.append(_sample(scene, step_counter, "POST_RELEASE_RETENTION"))
        step_counter += 1
    positions = np.asarray([sample["object_position"] for sample in samples])
    palm_positions = np.asarray([sample["palm_position"] for sample in samples])
    palm_distances = np.linalg.norm(positions - palm_positions, axis=1)
    palm_forces = np.asarray([sample["normal_forces_n"][5] for sample in samples])
    non_acquisition_load = np.asarray([sum(sample["support_load_fraction"][2:]) for sample in samples])
    final = _sample(scene, step_counter, "FINAL")
    alternate_support = bool(any(final["contact_flags"][2:]))
    release_indices = [FINGERS.index(finger) for finger in release_fingers]
    acquisition_released = all(not bool(final["contact_flags"][index]) for index in release_indices)
    available_motion = []
    for finger in release_fingers:
        ids = scene.joint_ids[finger]
        qpos = scene.data.qpos[scene.model.jnt_qposadr[ids]]
        limits = scene.model.jnt_range[ids]
        available_motion.append(float(np.sum(np.minimum(qpos - limits[:, 0], limits[:, 1] - qpos))))
    return {
        "object_initial_position": origin.tolist(),
        "support_keyframe": support_keyframe,
        "fixture_release_state": release_state,
        "samples": samples,
        "summary": {
            "object_motion_m": float(np.linalg.norm(positions[-1] - positions[0])),
            "initial_object_palm_distance_m": float(palm_distances[0]),
            "final_object_palm_distance_m": float(palm_distances[-1]),
            "dynamic_progress_toward_palm_m": float(palm_distances[0] - palm_distances[-1]),
            "palm_contact_achieved": bool(np.any(palm_forces > 0.0)),
            "maximum_palm_normal_force_n": float(np.max(palm_forces)),
            "maximum_non_acquisition_support_load_fraction": float(np.max(non_acquisition_load)),
            "final_contact_flags": final["contact_flags"],
            "final_normal_forces_n": final["normal_forces_n"],
            "final_floor_contact": _object_floor_contact(scene),
            "acquisition_fingers_unloaded": bool(
                final["normal_forces_n"][0] == 0.0 and final["normal_forces_n"][1] == 0.0
            ),
            "configured_release_fingers": list(release_fingers),
            "configured_release_fingers_unloaded": bool(
                all(final["normal_forces_n"][index] == 0.0 for index in release_indices)
            ),
            "configured_release_fingers_released": acquisition_released,
            "alternate_support_present": alternate_support,
            "acquisition_finger_available_motion_raw": available_motion,
            "resource_recovered_diagnostic": bool(
                acquisition_released
                and alternate_support
                and not _object_floor_contact(scene)
                and any(value > 0.0 for value in available_motion)
            ),
            "post_release_object_qpos_was_never_set": True,
        },
    }


def write_phase3a_results(output_dir: Path | None = None) -> tuple[dict, dict]:
    output_dir = output_dir or ROOT / "outputs/phase3A"
    output_dir.mkdir(parents=True, exist_ok=True)
    cohort = run_minimal_acquisition_cohort()
    handoff = run_handoff_diagnostic()
    (output_dir / "acquisition_and_recruitment.json").write_text(
        json.dumps(cohort, indent=2), encoding="utf-8"
    )
    (output_dir / "contact_handoff.json").write_text(json.dumps(handoff, indent=2), encoding="utf-8")
    return cohort, handoff
