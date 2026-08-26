"""Ordered, scripted Phase 3C-0 physics diagnostics (no learning)."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .config import ROOT
from .phase3.config import FINGERS, SUPPORT_SURFACES
from .phase3.contacts import extract_shadow_contacts
from .phase3.control import ContactAwareCloser, actuator_target_from_qpos
from .phase3.env import load_keyframe_qpos
from .phase3.model import build_shadow_scene, set_fixture, set_object_pose
from .phase3b1a import project_feasible_hand_qpos
from .phase3c0 import (
    Phase3CFailure,
    Phase3CFingerRole,
    Phase3CRoles,
    Phase3CState,
    build_phase3c_multiscene,
    configured_storage_region,
    gravity_in_palm_frame,
    load_phase3c0_config,
    multi_object_support_graph,
    object_pose_in_palm,
    open_hand_configuration,
    palm_transform,
    release_phase3c_fixture,
    set_phase3c_object_pose,
    storage_aperture,
    storage_measurement,
    transfer_corridor,
)


def _project(name: str, scene) -> np.ndarray:
    return np.asarray(project_feasible_hand_qpos(load_keyframe_qpos(name), scene).projected_qpos)


def _floor_contact(scene, object_body_name: str) -> bool:
    floor = scene.config.raw["floor"]["name"]
    floor_geom = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, floor)
    object_body = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, object_body_name)
    for index in range(scene.data.ncon):
        contact = scene.data.contact[index]
        bodies = {int(scene.model.geom_bodyid[contact.geom1]), int(scene.model.geom_bodyid[contact.geom2])}
        if object_body in bodies and floor_geom in {int(contact.geom1), int(contact.geom2)}:
            return True
    return False


def _single_sample(scene, step: int, stage: str, roles: Phase3CRoles) -> dict[str, Any]:
    contacts = extract_shadow_contacts(scene)
    storage = storage_measurement(scene, scene.object_body_id, np.asarray(scene.config.object["size"]))
    palm_pos, palm_rot = palm_transform(scene)
    region = configured_storage_region()
    target = palm_pos + palm_rot @ np.asarray(region.center_palm_m)
    corridor = transfer_corridor(
        scene, scene.data.xpos[scene.object_body_id], target,
        object_radius_m=float(np.max(scene.config.object["size"])),
        samples=int(load_phase3c0_config()["corridor"]["samples"]),
    )
    return {
        "step": int(step), "time_s": float(scene.data.time), "stage": stage,
        "state": roles.state.value,
        "roles": {name: role.value for name, role in roles.fingers.items()},
        "hand_qpos": scene.data.qpos[:24].tolist(),
        "hand_qvel": scene.data.qvel[:24].tolist(),
        "object_qpos": scene.data.qpos[scene.model.jnt_qposadr[scene.object_joint_id]:
                                        scene.model.jnt_qposadr[scene.object_joint_id] + 7].tolist(),
        "object_qvel": scene.data.qvel[scene.model.jnt_dofadr[scene.object_joint_id]:
                                        scene.model.jnt_dofadr[scene.object_joint_id] + 6].tolist(),
        "object_position_world_m": scene.data.xpos[scene.object_body_id].tolist(),
        "object_position_palm_m": storage["object_center_palm_m"],
        "palm_position_world_m": palm_pos.tolist(),
        "palm_rotation_world": palm_rot.tolist(),
        "contact_flags": contacts.contact_flags.tolist(),
        "normal_forces_n": contacts.normal_forces.tolist(),
        "support_load_fraction": contacts.support_load_fraction.tolist(),
        "maximum_penetration_m": contacts.maximum_penetration,
        "maximum_penetration_pair": contacts.maximum_penetration_pair,
        "storage": storage,
        "corridor": corridor,
        "aperture": storage_aperture(scene),
        "gravity_in_palm_frame": gravity_in_palm_frame(scene).tolist(),
        "floor_contact": _floor_contact(scene, scene.config.object["name"]),
        "fixture_active": bool(scene.data.eq_active[scene.fixture_eq_id]),
        "ctrl": scene.data.ctrl.tolist(),
    }


def _move(scene, group: str, target: np.ndarray, increment: float) -> None:
    ids = scene.actuator_ids[group]
    scene.data.ctrl[ids] += np.clip(target[ids] - scene.data.ctrl[ids], -increment, increment)


def run_single_object_transfer(
    condition: str,
    object_position: np.ndarray,
    *,
    object_quaternion: np.ndarray | tuple[float, ...] = (1.0, 0.0, 0.0, 0.0),
    acquisition_digits: tuple[str, ...] = ("thumb", "index"),
    collect_dense: bool = False,
) -> dict[str, Any]:
    if condition not in {"old_early_support", "open_corridor"}:
        raise ValueError("unknown Phase 3C transfer condition")
    cfg3c = load_phase3c0_config()
    scene = build_shadow_scene()
    mujoco.mj_resetData(scene.model, scene.data)
    open_qpos, open_projection = open_hand_configuration(scene)
    pre_qpos = _project("pre grasp", scene)
    pinch_qpos = _project("two finger pinch", scene)
    support_qpos = _project("three finger pinch", scene)
    secure_qpos = _project("grasp soft", scene)
    open_target = actuator_target_from_qpos(scene, open_qpos)
    pre_target = actuator_target_from_qpos(scene, pre_qpos)
    pinch_target = actuator_target_from_qpos(scene, pinch_qpos)
    support_target = actuator_target_from_qpos(scene, support_qpos)
    secure_target = actuator_target_from_qpos(scene, secure_qpos)
    scene.data.qpos[:24] = open_qpos
    scene.data.qvel[:] = 0.0
    set_object_pose(scene, object_position, object_quaternion)
    set_fixture(scene, True)
    scene.data.ctrl[:] = open_target
    mujoco.mj_forward(scene.model, scene.data)
    roles = Phase3CRoles()
    roles.history.append({"step": 0, "state": roles.state.value, "reason": "compiled OPEN_HAND keyframe",
                          "roles": {finger: role.value for finger, role in roles.fingers.items()}})
    samples: list[dict[str, Any]] = [_single_sample(scene, 0, "OPEN_HAND", roles)]
    step = 0
    for _ in range(20):
        mujoco.mj_step(scene.model, scene.data)
        step += 1
    # Approach from the explicit open state using only wrist and the minimum
    # acquisition digits.  Middle/ring/little stay exactly at open targets.
    approach_steps = int(scene.config.diagnostic["approach_steps"])
    approach_ids = np.r_[scene.actuator_ids["wrist"], scene.actuator_ids["thumb"], scene.actuator_ids["index"]]
    for local in range(approach_steps):
        alpha = (local + 1) / approach_steps
        scene.data.ctrl[approach_ids] = ((1.0 - alpha) * open_target[approach_ids]
                                         + alpha * pre_target[approach_ids])
        for finger in ("middle", "ring", "little"):
            scene.data.ctrl[scene.actuator_ids[finger]] = open_target[scene.actuator_ids[finger]]
        mujoco.mj_step(scene.model, scene.data)
        step += 1
        if collect_dense or local % 10 == 0:
            samples.append(_single_sample(scene, step, "OPEN_MINIMAL_APPROACH", roles))
    roles.begin_minimal_acquisition("A", step)
    closer = ContactAwareCloser(scene, float(scene.config.diagnostic["contact_force_n"]))
    acquisition_ids = np.r_[scene.actuator_ids["thumb"], scene.actuator_ids["index"]]
    close_steps = int(scene.config.diagnostic["close_steps"])
    for local in range(close_steps):
        alpha = (local + 1) / close_steps
        proposed = scene.data.ctrl.copy()
        proposed[acquisition_ids] = (1.0 - alpha) * pre_target[acquisition_ids] + alpha * pinch_target[acquisition_ids]
        scene.data.ctrl[:] = closer.limit_target(proposed)
        mujoco.mj_step(scene.model, scene.data)
        step += 1
        if collect_dense or local % 10 == 0:
            samples.append(_single_sample(scene, step, "MINIMAL_ACQUIRE_A", roles))
    if "middle" in acquisition_digits:
        roles.transition(Phase3CState.MINIMAL_ACQUIRE_A,
                         {"middle": Phase3CFingerRole.ACQUIRING}, step=step,
                         reason="thumb/index-only transfer was insufficient; permitted three-digit fallback")
        for local in range(close_steps):
            _move(scene, "middle", secure_target, 0.002)
            scene.data.ctrl[:] = closer.limit_target(scene.data.ctrl)
            mujoco.mj_step(scene.model, scene.data)
            step += 1
            if collect_dense or local % 10 == 0:
                samples.append(_single_sample(scene, step, "MINIMAL_ACQUIRE_A_MIDDLE_FALLBACK", roles))
    for local in range(int(scene.config.diagnostic["settle_steps"])):
        mujoco.mj_step(scene.model, scene.data)
        step += 1
        if collect_dense or local % 10 == 0:
            samples.append(_single_sample(scene, step, "FIXTURE_SETTLE", roles))
    release_contacts = extract_shadow_contacts(scene)
    acquired = bool(release_contacts.contact_flags[0] and release_contacts.contact_flags[1])
    if "middle" in acquisition_digits:
        acquired = acquired and bool(release_contacts.contact_flags[2])
    set_fixture(scene, False)
    roles.begin_transfer(step)
    dual_after_hold = False
    for local in range(50):
        mujoco.mj_step(scene.model, scene.data)
        step += 1
        held_contacts = extract_shadow_contacts(scene)
        dual_after_hold |= bool(held_contacts.contact_flags[0] and held_contacts.contact_flags[1])
        if collect_dense or local % 5 == 0:
            samples.append(_single_sample(scene, step, "MINIMAL_UNSUPPORTED_HOLD", roles))
    first_storage_contact_step = None
    recruitment_step = None
    storage_entry_step = None
    ever_storage_entry = False
    for local in range(int(cfg3c["diagnostic"]["transfer_steps"])):
        current_contacts = extract_shadow_contacts(scene)
        reference = float(cfg3c["diagnostic"]["reference_penetration_m"])
        for surface_index, finger in enumerate(acquisition_digits):
            if current_contacts.penetration_by_surface[surface_index] < reference:
                _move(scene, finger, support_target, 0.0005)
        _move(scene, "wrist", support_target, 0.0002)
        if condition == "old_early_support":
            _move(scene, "middle", support_target, 0.0005)
            if recruitment_step is None:
                recruitment_step = step
        else:
            # Explicitly hold unused digits at OPEN_HAND until storage entry.
            for finger in ("middle", "ring", "little"):
                if finger in acquisition_digits:
                    continue
                scene.data.ctrl[scene.actuator_ids[finger]] = open_target[scene.actuator_ids[finger]]
        mujoco.mj_step(scene.model, scene.data)
        step += 1
        contacts = extract_shadow_contacts(scene)
        if first_storage_contact_step is None and np.any(contacts.contact_flags[2:]):
            first_storage_contact_step = step
        storage = storage_measurement(scene, scene.object_body_id, np.asarray(scene.config.object["size"]))
        if storage["center_inside"]:
            ever_storage_entry = True
            if storage_entry_step is None:
                storage_entry_step = step
                if condition == "open_corridor":
                    recruitment_step = step
                    roles.storage_entry(tuple(f for f in ("middle", "ring", "little") if f not in acquisition_digits), step)
        if collect_dense or local % 5 == 0:
            samples.append(_single_sample(scene, step, "TRANSFER_A_TO_PALM", roles))
        if condition == "open_corridor" and storage_entry_step is not None:
            break
    # The old condition continues early middle support; the open condition begins
    # storage closure only after the logged entry event.  No contact threshold is
    # silently substituted for the region event.
    if condition == "old_early_support":
        roles.storage_entry(tuple(f for f in ("middle", "ring", "little") if f not in acquisition_digits), step)
    if recruitment_step is not None:
        roles.transition(Phase3CState.SECURE_A,
                         {finger: Phase3CFingerRole.SECURING_STORAGE for finger in ("middle", "ring", "little") if finger not in acquisition_digits},
                         step=step, reason="delayed storage closure" if condition == "open_corridor" else "early support comparator")
        secure_steps = int(cfg3c["diagnostic"]["secure_steps"])
        # Progressive delayed closure mirrors the established safe controller:
        # middle first, then ring/little, while acquisition digits remain
        # penetration guarded by their contact latches.
        if "middle" not in acquisition_digits:
            for local in range(secure_steps):
                _move(scene, "middle", support_target, 0.002)
                mujoco.mj_step(scene.model, scene.data)
                step += 1
                if collect_dense or local % 5 == 0:
                    samples.append(_single_sample(scene, step, "SECURE_A_MIDDLE", roles))
        for local in range(secure_steps):
            _move(scene, "ring", secure_target, 0.002)
            _move(scene, "little", secure_target, 0.002)
            mujoco.mj_step(scene.model, scene.data)
            step += 1
            if collect_dense or local % 5 == 0:
                samples.append(_single_sample(scene, step, "SECURE_A_RING_LITTLE", roles))
    pre_release = extract_shadow_contacts(scene)
    alternate_support_pre_release = bool(np.any(pre_release.contact_flags[2:]))
    roles.transition(Phase3CState.RELEASE_ACQUISITION_DIGITS,
                     {"thumb": Phase3CFingerRole.RELEASING, "index": Phase3CFingerRole.RELEASING},
                     step=step, reason="attempt recovery only after storage securing")
    release_start = scene.data.ctrl.copy()
    for local in range(int(cfg3c["diagnostic"]["release_steps"])):
        alpha = (local + 1) / int(cfg3c["diagnostic"]["release_steps"])
        scene.data.ctrl[acquisition_ids] = ((1.0 - alpha) * release_start[acquisition_ids]
                                            + alpha * open_target[acquisition_ids])
        mujoco.mj_step(scene.model, scene.data)
        step += 1
        if collect_dense or local % 5 == 0:
            samples.append(_single_sample(scene, step, "RELEASE_ACQUISITION_DIGITS", roles))
    for local in range(int(cfg3c["diagnostic"]["settle_steps"])):
        mujoco.mj_step(scene.model, scene.data)
        step += 1
        if collect_dense or local % 5 == 0:
            samples.append(_single_sample(scene, step, "POST_RELEASE_SETTLE", roles))
    final_contacts = extract_shadow_contacts(scene)
    final_storage = storage_measurement(scene, scene.object_body_id, np.asarray(scene.config.object["size"]))
    thumb_recovered = not bool(final_contacts.contact_flags[0])
    index_recovered = not bool(final_contacts.contact_flags[1])
    retained = not _floor_contact(scene, scene.config.object["name"]) and final_storage["center_inside"]
    secure = retained and bool(np.any(final_contacts.contact_flags[2:]))
    recovered = secure and thumb_recovered and index_recovered
    if recovered:
        roles.transition(Phase3CState.ACQUISITION_RESOURCES_RECOVERED,
                         {"thumb": Phase3CFingerRole.FREE, "index": Phase3CFingerRole.FREE},
                         step=step, reason="both acquisition digits unloaded while A remains supported")
    samples.append(_single_sample(scene, step, "FINAL", roles))
    max_pen = np.asarray([s["maximum_penetration_m"] for s in samples])
    min_clearance = min(s["corridor"]["minimum_clearance_m"] for s in samples)
    initial_distance = np.linalg.norm(np.asarray(samples[0]["object_position_world_m"])
                                      - np.asarray(samples[0]["palm_position_world_m"]))
    final_distance = np.linalg.norm(np.asarray(samples[-1]["object_position_world_m"])
                                    - np.asarray(samples[-1]["palm_position_world_m"]))
    return {
        "condition": condition,
        "object_initial_position_m": np.asarray(object_position).tolist(),
        "object_initial_quaternion_wxyz": np.asarray(object_quaternion).tolist(),
        "open_hand_qpos": open_qpos.tolist(),
        "open_projection": open_projection,
        "acquisition_digits": list(acquisition_digits),
        "unused_digits": [finger for finger in FINGERS if finger not in acquisition_digits],
        "dual_contact_before_release": acquired,
        "dual_contact_during_unsupported_hold": dual_after_hold,
        "fixture_release_step": close_steps + approach_steps + 20 + int(scene.config.diagnostic["settle_steps"])
        + (close_steps if "middle" in acquisition_digits else 0),
        "fixture_never_reactivated": True,
        "storage_entry_step": storage_entry_step,
        "recruitment_step": recruitment_step,
        "first_non_acquisition_contact_step": first_storage_contact_step,
        "alternate_support_before_release": alternate_support_pre_release,
        "storage_entry": ever_storage_entry,
        "secure_storage": secure,
        "thumb_recovered": thumb_recovered,
        "index_recovered": index_recovered,
        "resource_recovered": recovered,
        "floor_contact": _floor_contact(scene, scene.config.object["name"]),
        "gross_collision_steps": int(np.sum(max_pen > float(cfg3c["diagnostic"]["reference_penetration_m"]))),
        "maximum_penetration_m": float(max_pen.max()),
        "minimum_corridor_clearance_m": float(min_clearance),
        "palmward_progress_m": float(initial_distance - final_distance),
        "final_storage": final_storage,
        "final_contact_flags": final_contacts.contact_flags.tolist(),
        "final_normal_forces_n": final_contacts.normal_forces.tolist(),
        "roles": roles.history,
        "samples": samples,
    }


def _aggregate(condition: str, trials: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(trials)
    return {
        "condition": condition,
        "attempts": count,
        "transfer_successes": sum(t["storage_entry"] for t in trials),
        "transfer_success_rate": sum(t["storage_entry"] for t in trials) / count,
        "secure_storage_successes": sum(t["secure_storage"] for t in trials),
        "resource_recovery_successes": sum(t["resource_recovered"] for t in trials),
        "gross_collision_steps_total": sum(t["gross_collision_steps"] for t in trials),
        "minimum_corridor_clearance_m": min(t["minimum_corridor_clearance_m"] for t in trials),
        "median_corridor_clearance_m": float(np.median([t["minimum_corridor_clearance_m"] for t in trials])),
    }


def _state_dependent_wrist_grid(scene, samples_per_axis: int) -> list[tuple[float, float]]:
    ids = [mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, name)
           for name in scene.config.hand.wrist_joints]
    axes = [np.linspace(scene.model.jnt_range[jid, 0], scene.model.jnt_range[jid, 1], samples_per_axis)
            for jid in ids]
    return [(float(a), float(b)) for a in axes[0] for b in axes[1]]


def _multiscene_from_stored_trial(trial: dict[str, Any]):
    scene = build_phase3c_multiscene()
    mujoco.mj_resetData(scene.model, scene.data)
    final = trial["samples"][-1]
    scene.data.qpos[:24] = final["hand_qpos"]
    scene.data.qvel[:24] = final["hand_qvel"]
    scene.data.ctrl[:] = final["ctrl"]
    # Initial-state reconstruction occurs before the diagnostic clock starts.
    # A is immediately released and can never be posed again through this API.
    set_phase3c_object_pose(scene, "A", final["object_qpos"][:3], final["object_qpos"][3:])
    a_dof = scene.model.jnt_dofadr[scene.object_joint_ids["A"]]
    scene.data.qvel[a_dof:a_dof + 6] = final["object_qvel"]
    release_phase3c_fixture(scene, "A")
    b_home = np.asarray(scene.config.object["initial_pos"]) + [0.0, 0.12, 0.0]
    set_phase3c_object_pose(scene, "B", b_home)
    mujoco.mj_forward(scene.model, scene.data)
    return scene


def run_wrist_feasibility(stored_trial: dict[str, Any]) -> dict[str, Any]:
    cfg = load_phase3c0_config()
    base = _multiscene_from_stored_trial(stored_trial)
    wrist_joint_ids = [mujoco.mj_name2id(base.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                       for name in base.config.hand.wrist_joints]
    wrist_act = base.actuator_ids["wrist"]
    candidates = []
    for wrist in _state_dependent_wrist_grid(base, int(cfg["wrist_search"]["samples_per_axis"])):
        scene = _multiscene_from_stored_trial(stored_trial)
        target = scene.data.ctrl.copy()
        # Wrist actuators map one-to-one to the two wrist joints.
        target[wrist_act] = wrist
        initial_a = scene.data.xpos[scene.object_body_ids["A"]].copy()
        retained = True
        maximum_penetration = 0.0
        for _ in range(int(cfg["wrist_search"]["retention_steps"])):
            scene.data.ctrl[wrist_act] += np.clip(target[wrist_act] - scene.data.ctrl[wrist_act], -0.001, 0.001)
            mujoco.mj_step(scene.model, scene.data)
            retained &= not _floor_contact(scene, "phase3c_object_A")
            for index in range(scene.data.ncon):
                c = scene.data.contact[index]
                if scene.object_geom_ids["A"] in {int(c.geom1), int(c.geom2)}:
                    maximum_penetration = max(maximum_penetration, max(0.0, -float(c.dist)))
        storage = storage_measurement(scene, scene.object_body_ids["A"], np.asarray(scene.config.object["size"]))
        retained &= bool(storage["center_inside"])
        aperture = storage_aperture(scene)
        palm_pos, palm_rot = palm_transform(scene)
        aperture_world = palm_pos + palm_rot @ np.asarray(aperture["centroid_palm_m"])
        normal_world = palm_rot @ np.asarray(aperture["normal_palm"])
        start = aperture_world + normal_world * 0.12
        corridor = transfer_corridor(scene, start, aperture_world, object_radius_m=float(max(scene.config.object["size"])),
                                     stored_object_geoms=(scene.object_geom_ids["A"],))
        gravity = gravity_in_palm_frame(scene)
        candidates.append({
            "wrist_qpos_rad": list(wrist), "retained_A": bool(retained),
            "A_displacement_m": float(np.linalg.norm(scene.data.xpos[scene.object_body_ids["A"]] - initial_a)),
            "A_storage": storage, "maximum_A_penetration_m": maximum_penetration,
            "aperture": aperture, "corridor": corridor,
            "gravity_in_palm_frame": gravity.tolist(),
            "candidate_insertion_direction_world": normal_world.tolist(),
            "candidate_insertion_direction_palm": aperture["normal_palm"],
            "feasible_raw": bool(retained and corridor["minimum_clearance_m"] >= 0.0),
        })
    return {
        "wrist_configurations_tested": len(candidates),
        "candidates": candidates,
        "retained_A_count": sum(row["retained_A"] for row in candidates),
        "feasible_insertion_corridor_count": sum(row["feasible_raw"] for row in candidates),
        "world_gravity": base.model.opt.gravity.tolist(),
        "insertion_direction_fixed": False,
    }


def run_aperture_relaxation(stored_trial: dict[str, Any]) -> dict[str, Any]:
    cfg = load_phase3c0_config()
    rows = []
    open_qpos, _ = open_hand_configuration()
    for fraction in cfg["aperture"]["relaxation_fractions"]:
        scene = _multiscene_from_stored_trial(stored_trial)
        before = storage_aperture(scene)
        initial_a_pos = scene.data.xpos[scene.object_body_ids["A"]].copy()
        initial_a_rot = scene.data.xmat[scene.object_body_ids["A"]].reshape(3, 3).copy()
        open_target = actuator_target_from_qpos(scene, open_qpos)
        for _ in range(100):
            for finger in ("middle", "ring", "little"):
                ids = scene.actuator_ids[finger]
                desired = (1.0 - fraction) * np.asarray(stored_trial["samples"][-1]["ctrl"])[ids] + fraction * open_target[ids]
                scene.data.ctrl[ids] += np.clip(desired - scene.data.ctrl[ids], -0.0005, 0.0005)
            mujoco.mj_step(scene.model, scene.data)
        after = storage_aperture(scene)
        current_rot = scene.data.xmat[scene.object_body_ids["A"]].reshape(3, 3)
        angle = float(np.arccos(np.clip((np.trace(initial_a_rot.T @ current_rot) - 1.0) / 2.0, -1.0, 1.0)))
        storage = storage_measurement(scene, scene.object_body_ids["A"], np.asarray(scene.config.object["size"]))
        rows.append({
            "relaxation_fraction": float(fraction), "before": before, "after": after,
            "A_displacement_m": float(np.linalg.norm(scene.data.xpos[scene.object_body_ids["A"]] - initial_a_pos)),
            "A_rotation_rad": angle, "A_retained": bool(storage["center_inside"] and not _floor_contact(scene, "phase3c_object_A")),
            "A_storage": storage,
        })
    return {"attempts": rows, "small_controlled_relaxation_only": True}


def run_b_insertion_diagnostic(stored_trial: dict[str, Any], wrist_result: dict[str, Any]) -> dict[str, Any]:
    feasible = [row for row in wrist_result["candidates"] if row["feasible_raw"]]
    if not feasible:
        return {"status": "SKIPPED", "reason": Phase3CFailure.NO_FEASIBLE_INSERTION_CORRIDOR.value,
                "B_acquisition_attempts": 0, "B_insertion_attempts": 0, "multi_object_storage_successes": 0}
    # A physically feasible corridor is necessary, but Phase 3C-0 intentionally
    # does not synthesize a new multi-object controller. Record one honest
    # mechanics attempt using the first feasible state-dependent direction.
    candidate = feasible[0]
    scene = _multiscene_from_stored_trial(stored_trial)
    wrist_target = np.asarray(candidate["wrist_qpos_rad"])
    for _ in range(75):
        scene.data.ctrl[scene.actuator_ids["wrist"]] += np.clip(
            wrist_target - scene.data.ctrl[scene.actuator_ids["wrist"]], -0.001, 0.001)
        mujoco.mj_step(scene.model, scene.data)
    palm_pos, palm_rot = palm_transform(scene)
    aperture = storage_aperture(scene)
    direction = palm_rot @ np.asarray(aperture["normal_palm"])
    center = palm_pos + palm_rot @ np.asarray(aperture["centroid_palm_m"])
    b_start = center + direction * 0.12
    # B is positioned only while its acquisition fixture is active.
    set_phase3c_object_pose(scene, "B", b_start)
    open_qpos, _ = open_hand_configuration()
    pinch_qpos = _project("two finger pinch", scene)
    open_target = actuator_target_from_qpos(scene, open_qpos)
    pinch_target = actuator_target_from_qpos(scene, pinch_qpos)
    start_ctrl = scene.data.ctrl.copy()
    acquisition_ids = np.r_[scene.actuator_ids["thumb"], scene.actuator_ids["index"]]
    for local in range(180):
        alpha = (local + 1) / 180
        scene.data.ctrl[acquisition_ids] = (1.0 - alpha) * open_target[acquisition_ids] + alpha * pinch_target[acquisition_ids]
        mujoco.mj_step(scene.model, scene.data)
    graph_at_acquire = multi_object_support_graph(scene)
    acquisition_edges = [edge for edge in graph_at_acquire["edges"]
                         if edge["object"] == "B" and edge["support"] in {"thumb", "index"}]
    b_acquired = {edge["support"] for edge in acquisition_edges} == {"thumb", "index"}
    release_phase3c_fixture(scene, "B")
    b_initial = scene.data.xpos[scene.object_body_ids["B"]].copy()
    a_initial = scene.data.xpos[scene.object_body_ids["A"]].copy()
    # Move acquisition digits/wrist toward the aperture. No object qpos or mocap
    # pose is changed after fixture release.
    transfer_target = actuator_target_from_qpos(scene, _project("three finger pinch", scene))
    for _ in range(350):
        for finger in ("thumb", "index", "wrist"):
            _move(scene, finger, transfer_target, 0.0004 if finger != "wrist" else 0.0002)
        mujoco.mj_step(scene.model, scene.data)
    final_graph = multi_object_support_graph(scene)
    a_storage = storage_measurement(scene, scene.object_body_ids["A"], np.asarray(scene.config.object["size"]))
    b_storage = storage_measurement(scene, scene.object_body_ids["B"], np.asarray(scene.config.object["size"]))
    a_retained = bool(a_storage["center_inside"] and not _floor_contact(scene, "phase3c_object_A"))
    b_inserted = bool(b_storage["center_inside"])
    both_supported = {edge["object"] for edge in final_graph["edges"]} == {"A", "B"}
    success = bool(b_acquired and a_retained and b_inserted and both_supported)
    return {
        "status": "COMPLETED", "B_acquisition_attempts": 1, "B_insertion_attempts": 1,
        "B_acquired": b_acquired, "B_insertion_success": b_inserted,
        "A_retained_during_B_insertion": a_retained, "A_B_resecure_success": success,
        "multi_object_storage_successes": int(success),
        "candidate_wrist_qpos_rad": candidate["wrist_qpos_rad"],
        "state_dependent_insertion_direction_world": direction.tolist(),
        "B_dynamic_displacement_m": float(np.linalg.norm(scene.data.xpos[scene.object_body_ids["B"]] - b_initial)),
        "A_dynamic_displacement_m": float(np.linalg.norm(scene.data.xpos[scene.object_body_ids["A"]] - a_initial)),
        "no_post_acquisition_object_pose_set": True, "A_fixture_active_during_attempt": False,
        "B_fixture_active_after_acquisition": False, "support_graph_at_acquisition": graph_at_acquire,
        "final_support_graph": final_graph, "A_storage": a_storage, "B_storage": b_storage,
        "final_hand_qpos": scene.data.qpos[:24].tolist(),
        "final_object_qpos": {label: scene.data.qpos[scene.model.jnt_qposadr[jid]:scene.model.jnt_qposadr[jid]+7].tolist()
                              for label, jid in scene.object_joint_ids.items()},
    }


def run_phase3c0(output_dir: Path | None = None) -> dict[str, Any]:
    output_dir = output_dir or ROOT / "outputs/phase3C0"
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_phase3c0_config()
    origin = np.asarray(cfg["diagnostic"]["open_corridor_initial_pos_m"], dtype=float)
    orientation = np.asarray(cfg["diagnostic"]["open_corridor_initial_quat_wxyz"], dtype=float)
    conditions: dict[str, list[dict[str, Any]]] = {}
    for condition in ("old_early_support", "open_corridor"):
        conditions[condition] = [run_single_object_transfer(condition, origin + np.asarray(offset),
                                                             object_quaternion=orientation)
                                 for offset in cfg["diagnostic"]["matched_offsets_m"]]
    summary = {name: _aggregate(name, trials) for name, trials in conditions.items()}
    validated = [trial for trial in conditions["open_corridor"] if trial["secure_storage"]]
    if validated:
        stored = validated[0]
        wrist = run_wrist_feasibility(stored)
        aperture = run_aperture_relaxation(stored)
        insertion = run_b_insertion_diagnostic(stored, wrist)
        gate = "validated open-corridor A storage state available"
    else:
        wrist = {"status": "SKIPPED", "reason": "no dynamically secure open-corridor A storage state"}
        aperture = {"status": "SKIPPED", "reason": "first-object storage mechanism not validated"}
        insertion = {"status": "SKIPPED", "reason": "C0-C/C0-D gate not passed", "B_acquisition_attempts": 0,
                     "B_insertion_attempts": 0, "multi_object_storage_successes": 0}
        gate = "not passed; two-object physics diagnostics correctly withheld"
    result = {
        "phase": "3C-0", "seed": int(cfg["seed"]), "rl_training_performed": False,
        "world_gravity_changed": False, "official_mjcf_modified": False,
        "experiment_order": ["C0-A", "C0-B", "C0-C", "C0-D", "C0-E", "C0-F"],
        "C0_A_and_B": {"summary": summary, "trials": conditions},
        "first_object_validation_gate": gate,
        "C0_C_wrist_feasibility": wrist,
        "C0_D_aperture_relaxation": aperture,
        "C0_E_F_B_insertion_and_resecure": insertion,
    }
    (output_dir / "phase3c0_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


# Local import at end avoids a circular import in static tooling.
from .phase3.config import load_phase3_config
