from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import warnings

import mujoco
import numpy as np
from scipy.stats import qmc

from ..config import ROOT, load_configs
from ..control import JointImpedanceController, hand_state, resolve_hand_indices
from ..diagnostics.multi_grasp import load_grasp_profile
from ..diagnostics.scripted_grasp import _finger_object_contact_arrays, _joint_target
from ..phase2_5_config import Phase25Config
from ..scene_builder import build_scene
from ..sensing import extract_contacts, group_contacts_by_finger
from .resource_components import FINGER_ORDER
from .second_grasp import BPlacement, _b_hand_state, _rotation_change, _set_b_pose, classify_B_acquisition


@dataclass(frozen=True)
class BAcquisitionTrajectory:
    candidate_index: int
    approach_joint_rad: tuple[float, ...]
    precontact_joint_rad: tuple[float, ...]
    closing_joint_rad: tuple[float, ...]
    hold_joint_rad: tuple[float, ...]
    close_steps: int
    per_finger_close_delay_steps: tuple[int, int, int, int]
    fixture_release_delay_steps: int


def _lhs(cfg: Phase25Config, count: int) -> np.ndarray:
    dimension = 8 + 8 + 8 + 8 + 1 + 2 + 1
    return qmc.LatinHypercube(d=dimension, seed=cfg.seeds.b_only_search).random(count)


def sample_b_only_trajectory(cfg: Phase25Config, candidate_index: int) -> BAcquisitionTrajectory:
    count = cfg.trajectory_search.expanded_candidate_count
    if not 0 <= candidate_index < count:
        raise IndexError(candidate_index)
    unit = _lhs(cfg, count)[candidate_index]
    base = load_configs()
    model, data = build_scene(base)
    indices = resolve_hand_indices(model, base.hand)
    _, profile = load_grasp_profile(ROOT / "configs" / "grasps" / "resource_grasp_A_02.yaml")
    open_q = _joint_target(model, base, indices, profile.open_joint_fractions)
    pre = open_q.copy()
    active_indices = np.r_[0:4, 12:16]
    anchors = np.r_[
        cfg.positive_control.precontact_anchor_joint_rad["index"],
        cfg.positive_control.precontact_anchor_joint_rad["thumb"],
    ]
    cursor = 0
    approach_width = cfg.trajectory_search.joint_approach_offset_rad
    approach = open_q.copy()
    approach[active_indices] = anchors + qmc.scale(
        unit[cursor:cursor + 8][None, :], [approach_width[0]] * 8, [approach_width[1]] * 8,
    )[0]
    cursor += 8
    pre_width = cfg.trajectory_search.joint_precontact_offset_rad
    pre[active_indices] = anchors + qmc.scale(unit[cursor:cursor + 8][None, :], [pre_width[0]] * 8, [pre_width[1]] * 8)[0]
    cursor += 8
    close_width = cfg.trajectory_search.joint_closing_offset_rad
    closing = pre.copy()
    closing[active_indices] += qmc.scale(unit[cursor:cursor + 8][None, :], [close_width[0]] * 8, [close_width[1]] * 8)[0]
    cursor += 8
    hold_width = cfg.trajectory_search.joint_hold_offset_rad
    hold = closing.copy()
    hold[active_indices] += qmc.scale(unit[cursor:cursor + 8][None, :], [hold_width[0]] * 8, [hold_width[1]] * 8)[0]
    cursor += 8
    close_steps = int(round(np.interp(unit[cursor], [0, 1], cfg.timing.close_steps_bounds)))
    cursor += 1
    delay_bounds = cfg.trajectory_search.per_finger_close_delay_steps
    index_delay = int(round(np.interp(unit[cursor], [0, 1], delay_bounds)))
    thumb_delay = int(round(np.interp(unit[cursor + 1], [0, 1], delay_bounds)))
    cursor += 2
    release_delay = int(round(np.interp(unit[cursor], [0, 1], cfg.timing.fixture_release_delay_steps_bounds)))
    ranges = model.jnt_range[indices.joint_ids]
    approach = np.clip(approach, ranges[:, 0], ranges[:, 1])
    pre = np.clip(pre, ranges[:, 0], ranges[:, 1])
    closing = np.clip(closing, ranges[:, 0], ranges[:, 1])
    hold = np.clip(hold, ranges[:, 0], ranges[:, 1])
    return BAcquisitionTrajectory(
        candidate_index=candidate_index,
        approach_joint_rad=tuple(float(x) for x in approach),
        precontact_joint_rad=tuple(float(x) for x in pre),
        closing_joint_rad=tuple(float(x) for x in closing),
        hold_joint_rad=tuple(float(x) for x in hold),
        close_steps=close_steps,
        per_finger_close_delay_steps=(index_delay, 0, 0, thumb_delay),
        fixture_release_delay_steps=release_delay,
    )


def trajectory_profile_dict(trajectory: BAcquisitionTrajectory) -> dict:
    return {"phase2_5_trajectory": asdict(trajectory)}


def _phase_target(
    step: int,
    open_q: np.ndarray,
    trajectory: BAcquisitionTrajectory,
    approach_steps: int,
    precontact_steps: int,
):
    approach = np.asarray(trajectory.approach_joint_rad)
    pre = np.asarray(trajectory.precontact_joint_rad)
    closing = np.asarray(trajectory.closing_joint_rad)
    hold = np.asarray(trajectory.hold_joint_rad)
    if step < approach_steps:
        alpha = (step + 1) / approach_steps
        return "approach", (1 - alpha) * open_q + alpha * approach
    precontact_step = step - approach_steps
    if precontact_step < precontact_steps:
        alpha = (precontact_step + 1) / precontact_steps
        return "precontact", (1 - alpha) * approach + alpha * pre
    close_step = precontact_step - precontact_steps
    if close_step < trajectory.close_steps:
        desired = pre.copy()
        for finger_index in range(4):
            sl = slice(4 * finger_index, 4 * finger_index + 4)
            delay = trajectory.per_finger_close_delay_steps[finger_index]
            alpha = np.clip((close_step + 1 - delay) / max(1, trajectory.close_steps - delay), 0.0, 1.0)
            desired[sl] = (1 - alpha) * pre[sl] + alpha * closing[sl]
        return "close", desired
    if close_step < trajectory.close_steps + trajectory.fixture_release_delay_steps:
        return "pre_release_hold", closing
    return "unsupported_hold", hold


def classify_failure_mechanism(summary: dict) -> str:
    if summary.get("invalid_reason"):
        return "INITIAL_INVALID_CONTACT" if str(summary["invalid_reason"]).startswith("initial") else "OTHER"
    if summary["A_present"] and not summary["A_retained"]:
        return "A_DESTABILIZED_DURING_APPROACH"
    if summary["maximum_B_free_finger_contacts_before_release"] == 0:
        return "NO_B_CONTACT_BEFORE_RELEASE"
    if summary["maximum_B_hand_contacts_before_release"] == 1:
        return "SINGLE_UNOPPOSED_CONTACT"
    if summary["maximum_B_hand_normal_force_before_release_N"] <= 0.20:
        return "CONTACT_FORCE_TOO_LOW"
    if summary["first_post_release_contact_loss_step"] is not None and summary["first_post_release_contact_loss_step"] <= 10:
        return "CONTACT_LOST_IMMEDIATELY_AFTER_RELEASE"
    if summary["B_table_contact_after_release"]:
        return "B_SLIPPED_TO_TABLE"
    if summary["maximum_B_orientation_after_release_rad"] > 0.20:
        return "B_ROTATED_OUT"
    return "OTHER"


def _initial_b_overlap_reason(model, data, b_geom: int, tolerance_m: float) -> str | None:
    """Apply the frozen initial-placement overlap rule before dynamics begin."""

    for geom_id in range(model.ngeom):
        if geom_id == b_geom:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name == "table" or not (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]):
            continue
        distance = float(mujoco.mj_geomDistance(model, data, b_geom, geom_id, 1.0, None))
        invalid_distance = 0.0 if name == "object_a_geom" else -tolerance_m
        if distance < invalid_distance:
            return f"initial_overlap:{name or geom_id}"
    return None


def run_b_acquisition_trajectory(
    cfg25: Phase25Config,
    trajectory: BAcquisitionTrajectory,
    *,
    A_record: dict | None = None,
    occupied_mask: np.ndarray | None = None,
    placement: BPlacement | None = None,
    collect_timeseries: bool = False,
    render_video_path: str | Path | None = None,
    render_stride: int | None = None,
    video_fps: int | None = None,
    scene_cfg=None,
    diagnostic_callback=None,
) -> tuple[dict, dict[str, np.ndarray] | None]:
    """Execute frozen-physics B-only or A-held+B scripted acquisition."""

    if A_record is None:
        cfg = scene_cfg or load_configs()
        model, data = build_scene(cfg)
        indices = resolve_hand_indices(model, cfg.hand)
        _, profile = load_grasp_profile(ROOT / "configs" / "grasps" / "resource_grasp_A_02.yaml")
        open_q = _joint_target(model, cfg, indices, profile.open_joint_fractions)
        A_present = False
        a_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_a_free")
        a_qadr, a_vadr = model.jnt_qposadr[a_joint], model.jnt_dofadr[a_joint]
        data.qpos[a_qadr:a_qadr + 3] = cfg25.positive_control.parked_A_position_m
        data.qvel[a_vadr:a_vadr + 6] = 0.0
        occupied = np.zeros(4, dtype=bool)
    else:
        from .resource_components import reconstruct_grasp
        cfg, model, data, indices = reconstruct_grasp(A_record, scene_cfg)
        open_q = data.qpos[indices.qpos_addresses].copy()
        A_present = True
        occupied = np.asarray(occupied_mask, dtype=bool)
        if "retaining_joint_target_rad" in A_record:
            a_hold = np.asarray(A_record["retaining_joint_target_rad"], dtype=float)
        else:
            _, accepted_profile = load_grasp_profile(ROOT / A_record["proposal_profile_path"])
            a_hold = _joint_target(
                model,
                cfg,
                indices,
                A_record.get(
                    "hold_joint_fractions",
                    accepted_profile.hold_joint_fractions or accepted_profile.closed_joint_fractions,
                ),
            )
    if placement is None:
        yaw = cfg25.positive_control.B_yaw_rad
        placement = BPlacement(
            0, tuple(cfg25.positive_control.B_pose_world_m),
            (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)), yaw,
        )
    b_body, b_geom, b_joint = _set_b_pose(model, data, placement)
    b_qadr, b_vadr = model.jnt_qposadr[b_joint], model.jnt_dofadr[b_joint]
    a_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object_a")
    a_reference_position = data.xpos[a_body].copy()
    a_reference_quaternion = data.xquat[a_body].copy()
    if A_record is None:
        data.qpos[indices.qpos_addresses] = open_q
        data.qvel[indices.qvel_addresses] = 0.0
        mujoco.mj_forward(model, data)
    initial_invalid_reason = _initial_b_overlap_reason(
        model, data, b_geom, cfg25.criteria.maximum_penetration_m,
    )
    ranges = model.jnt_range[indices.joint_ids]
    controller = JointImpedanceController(cfg.task.impedance_stiffness, cfg.task.impedance_damping, cfg.task.torque_limit)
    free = ~occupied
    release_step = (
        cfg25.timing.approach_steps
        + cfg25.timing.precontact_steps
        + trajectory.close_steps
        + trajectory.fixture_release_delay_steps
    )
    total_steps = release_step + cfg25.timing.unsupported_hold_steps
    object_b = next(obj for obj in cfg.scene.objects if obj.name == "object_b")
    table_top = cfg.scene.table_pos[2] + cfg.scene.table_size[2]
    rows = []
    frames = []
    renderer = None
    camera = None
    video_error = None
    if render_video_path is not None:
        if render_stride is None or render_stride <= 0 or video_fps is None or video_fps <= 0:
            raise ValueError("positive render_stride and video_fps are required when rendering")
        try:
            renderer = mujoco.Renderer(model, height=cfg.scene.render_height, width=cfg.scene.render_width)
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = np.asarray(placement.position_m)
            camera.distance, camera.azimuth, camera.elevation = 0.34, 140, -22
        except Exception as exc:  # platform OpenGL availability is not a physics outcome
            video_error = f"renderer_initialization_failed:{type(exc).__name__}:{exc}"
            warnings.warn(video_error)
    max_penetration = 0.0
    numeric = True
    for step in range(total_steps):
        phase, proposed = _phase_target(
            step, open_q, trajectory, cfg25.timing.approach_steps, cfg25.timing.precontact_steps,
        )
        proposed = np.clip(proposed, ranges[:, 0], ranges[:, 1])
        if A_present:
            desired = np.asarray(A_record["final_joint_configuration_rad"], dtype=float).copy()
            desired[~np.repeat(occupied, 4)] = proposed[~np.repeat(occupied, 4)]
            # Exact accepted support target for occupied fingers.
            desired[np.repeat(occupied, 4)] = a_hold[np.repeat(occupied, 4)]
        else:
            desired = proposed
        q, qdot = hand_state(data, indices)
        controls = controller.torque(desired, q, qdot)
        data.ctrl[indices.actuator_ids] = controls
        fixture_active = step < release_step
        if fixture_active:
            data.qpos[b_qadr:b_qadr + 3] = placement.position_m
            data.qpos[b_qadr + 3:b_qadr + 7] = placement.quaternion
            data.qvel[b_vadr:b_vadr + 6] = 0.0
        if A_record is None:
            data.qpos[a_qadr:a_qadr + 3] = cfg25.positive_control.parked_A_position_m
            data.qvel[a_vadr:a_vadr + 6] = 0.0
        mujoco.mj_step(model, data)
        numeric &= all(np.all(np.isfinite(value)) for value in (data.qpos, data.qvel, data.ctrl))
        contacts = extract_contacts(model, data)
        grouped = group_contacts_by_finger(contacts, cfg.hand.finger_geom_mapping)
        b_counts, b_positions, b_normals, b_distances, b_forces = _finger_object_contact_arrays(
            grouped, list(FINGER_ORDER), "object_b", data.xpos[b_body],
        )
        b_tangential = np.asarray([
            sum(row.tangential_force for row in grouped[finger] if "object_b" in {row.body1_name, row.body2_name})
            for finger in FINGER_ORDER
        ])
        hand_contacts, hand_force = _b_hand_state(contacts)
        b_table = any({row.geom1_name, row.geom2_name} == {"object_b_geom", "table"} for row in contacts)
        penetration = max([0.0, *[-row.distance for row in contacts if "object_b" in {row.body1_name, row.body2_name}]])
        max_penetration = max(max_penetration, penetration)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b_body, velocity, 0)
        a_counts, _, _, _, a_forces = _finger_object_contact_arrays(grouped, list(FINGER_ORDER), "object_a", data.xpos[a_body])
        from .phase2r import _object_hand_contacts
        a_all_contacts = _object_hand_contacts(model, data, cfg, "object_a")
        a_table_contact = any({row.geom1_name, row.geom2_name} == {"object_a_geom", "table"} for row in contacts)
        rows.append({
            "timestep": step, "phase": phase, "fixture_active": fixture_active,
            "B_position_m": data.xpos[b_body].copy(), "B_quaternion": data.xquat[b_body].copy(),
            "B_linear_velocity_m_per_s": velocity[3:].copy(), "B_angular_velocity_rad_per_s": velocity[:3].copy(),
            "B_vertical_position_m": float(data.xpos[b_body, 2]),
            "B_table_distance_m": float(data.xpos[b_body, 2] - object_b.size[1] - table_top),
            "B_table_contact": b_table, "B_hand_contacts": hand_contacts,
            "B_hand_normal_force_N": hand_force,
            "B_free_finger_contacts": int(np.sum((b_counts > 0) & free)),
            "B_per_finger_contact_flag": (b_counts > 0).astype(np.int8),
            "B_per_finger_normal_force_N": b_forces,
            "B_per_finger_tangential_force_N": b_tangential,
            "B_per_finger_tangential_normal_ratio": np.divide(b_tangential, b_forces, out=np.zeros(4), where=b_forces > 0),
            "B_contact_positions_m": b_positions, "B_contact_normals": b_normals,
            "B_penetration_depths_m": np.maximum(0.0, -b_distances),
            "A_position_m": data.xpos[a_body].copy(), "A_quaternion": data.xquat[a_body].copy(),
            "A_displacement_m": float(np.linalg.norm(data.xpos[a_body] - a_reference_position)),
            "A_rotation_rad": _rotation_change(data.xquat[a_body], a_reference_quaternion),
            "A_per_finger_contact_flag": (a_counts > 0).astype(np.int8),
            "A_per_finger_normal_force_N": a_forces,
            "A_all_link_per_finger_contact_flag": (a_all_contacts["finger_counts"] > 0).astype(np.int8),
            "A_all_link_per_finger_normal_force_N": a_all_contacts["finger_force_N"],
            "A_palm_contact": int(a_all_contacts["palm_count"] > 0),
            "A_palm_contact_count": a_all_contacts["palm_count"],
            "A_palm_normal_force_N": a_all_contacts["palm_force_N"],
            "A_hand_contact_count": a_all_contacts["hand_count"],
            "A_penetration_m": a_all_contacts["maximum_penetration_m"],
            "A_table_contact": int(a_table_contact),
            "commanded_joint_target_rad": desired.copy(), "actual_joint_rad": q.copy(),
            "joint_velocity_rad_per_s": qdot.copy(), "actuator_controls": controls.copy(),
            "commanded_free_finger_target_rad": desired[np.repeat(free, 4)].copy(),
            "actual_free_finger_joint_rad": q[np.repeat(free, 4)].copy(),
            "free_finger_joint_velocity_rad_per_s": qdot[np.repeat(free, 4)].copy(),
            "free_finger_actuator_controls": controls[np.repeat(free, 4)].copy(),
            "actuator_saturation_count": int(np.sum(np.isclose(np.abs(controls), cfg.task.torque_limit, atol=1e-10))),
        })
        if diagnostic_callback is not None:
            # Phase 2H visualization hook. Callbacks are required to be
            # read-only; the default experiment path never enters this block.
            diagnostic_callback(step, model, data, rows[-1])
        if (
            renderer is not None
            and release_step - cfg25.timing.diagnostic_pre_release_steps <= step < release_step + cfg25.timing.diagnostic_post_release_steps
            and (step - (release_step - cfg25.timing.diagnostic_pre_release_steps)) % render_stride == 0
        ):
            try:
                renderer.update_scene(data, camera=camera)
                frames.append(renderer.render().copy())
            except Exception as exc:
                video_error = f"frame_capture_failed:{type(exc).__name__}:{exc}"
                warnings.warn(video_error)
                renderer.close()
                renderer = None
    if renderer is not None:
        renderer.close()
    video_written = False
    if render_video_path is not None and frames and video_error is None:
        try:
            import imageio.v2 as imageio

            video_path = Path(render_video_path)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(video_path, frames, fps=video_fps)
            video_written = True
        except Exception as exc:
            video_error = f"video_encoding_failed:{type(exc).__name__}:{exc}"
            warnings.warn(video_error)
    arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    post = slice(release_step, total_steps)
    pre = slice(0, release_step)
    b_reference_position = arrays["B_position_m"][release_step - 1]
    b_reference_quaternion = arrays["B_quaternion"][release_step - 1]
    translations = np.linalg.norm(arrays["B_position_m"][post] - b_reference_position, axis=1)
    rotations = np.asarray([_rotation_change(q, b_reference_quaternion) for q in arrays["B_quaternion"][post]])
    contact_loss_indices = np.flatnonzero(arrays["B_hand_contacts"][post] == 0)
    A_translation = float(np.max(arrays["A_displacement_m"])) if A_present else 0.0
    A_rotation = float(np.max(arrays["A_rotation_rad"])) if A_present else 0.0
    A_table = bool(np.any([
        False
    ]))
    A_retained = (not A_present) or bool(
        A_translation <= cfg25.criteria.maximum_A_translation_m
        and A_rotation <= cfg25.criteria.maximum_A_orientation_rad
        and np.all(np.sum(arrays["A_per_finger_contact_flag"], axis=1) >= cfg25.criteria.minimum_A_finger_contacts)
        and arrays["A_per_finger_normal_force_N"][-1].sum() > cfg25.criteria.minimum_A_normal_force_N
    )
    final_free = int(arrays["B_free_finger_contacts"][-1])
    final_hand = int(arrays["B_hand_contacts"][-1])
    final_force = float(arrays["B_hand_normal_force_N"][-1])
    acquired = initial_invalid_reason is None and classify_B_acquisition(
        fixture_released=True,
        final_free_finger_contacts=final_free,
        final_hand_contacts=final_hand,
        final_hand_normal_force_N=final_force,
        table_contact=bool(np.any(arrays["B_table_contact"][post])),
        complete_hand_contact_loss=bool(len(contact_loss_indices)),
        maximum_penetration_m=max_penetration,
        maximum_translation_m=float(np.max(translations)),
        maximum_orientation_rad=float(np.max(rotations)),
        numerically_stable=numeric,
        criteria=cfg25.criteria,
    )
    summary = {
        "candidate_index": trajectory.candidate_index,
        "A_present": A_present,
        "A_retained": A_retained,
        "B_acquired": acquired,
        "BOTH_RETAINED": bool(A_retained and acquired),
        "fixture_released": True,
        "fixture_release_timestep": release_step,
        "maximum_B_free_finger_contacts_before_release": int(np.max(arrays["B_free_finger_contacts"][pre])),
        "maximum_B_hand_contacts_before_release": int(np.max(arrays["B_hand_contacts"][pre])),
        "maximum_B_hand_normal_force_before_release_N": float(np.max(arrays["B_hand_normal_force_N"][pre])),
        "first_post_release_contact_loss_step": None if not len(contact_loss_indices) else int(contact_loss_indices[0]),
        "B_table_contact_after_release": bool(np.any(arrays["B_table_contact"][post])),
        "unsupported_contact_steps": int(np.sum(arrays["B_hand_contacts"][post] > 0)),
        "maximum_B_hand_contacts_after_release": int(np.max(arrays["B_hand_contacts"][post])),
        "maximum_B_penetration_m": max_penetration,
        "maximum_B_translation_after_release_m": float(np.max(translations)),
        "maximum_B_orientation_after_release_rad": float(np.max(rotations)),
        "maximum_A_translation_m": A_translation,
        "maximum_A_orientation_rad": A_rotation,
        "maximum_actuator_saturation_count": int(np.max(arrays["actuator_saturation_count"])),
        "numerically_valid": bool(numeric and initial_invalid_reason is None),
        "invalid_reason": initial_invalid_reason,
        "diagnostic_video_requested": render_video_path is not None,
        "diagnostic_video_written": video_written,
        "diagnostic_video_frame_count": len(frames),
        "diagnostic_video_error": video_error,
        "placement": {"position_m": list(placement.position_m), "quaternion": list(placement.quaternion), "yaw_rad": placement.yaw_rad},
        "trajectory": asdict(trajectory),
    }
    summary["failure_mechanism"] = None if acquired else classify_failure_mechanism(summary)
    return summary, arrays if collect_timeseries else None


def b_only_lexicographic_key(summary: dict) -> tuple:
    return (
        int(summary["numerically_valid"]), int(summary["fixture_released"]),
        int(not summary["B_table_contact_after_release"]),
        int(summary["first_post_release_contact_loss_step"] is None),
        summary["unsupported_contact_steps"], summary["maximum_B_hand_contacts_after_release"],
        -summary["maximum_B_penetration_m"], -summary["maximum_B_translation_after_release_m"],
        -summary["maximum_B_orientation_after_release_rad"], -summary["maximum_actuator_saturation_count"],
    )
