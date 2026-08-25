from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import mujoco
import numpy as np

from ..config import ROOT, load_configs
from ..control import JointImpedanceController, hand_state, resolve_hand_indices
from ..phase2_5_config import load_phase2_5_config
from ..phase2cm_config import Phase2CMConfig, load_phase2cm_config
from ..scene_builder import build_scene
from .phase2cm import (
    FINGER_NAMES,
    _capture_release_state,
    _jsonl,
    _load_state,
    _restore_state,
    _state_filename,
    _write_json,
    eligible_release_trials,
    phase2h_sources,
    scene_for_trial,
    select_frozen_trials,
)
from .resumable import IncrementalJsonlStore


PAIR_ORDER = ("index-B", "thumb-B", "middle-B", "ring-B", "palm-B", "B-table", "B-A", "other")


def _finger_for_body(body_name: str | None) -> str | None:
    prefix = (body_name or "").split("_", 1)[0]
    return {"ff": "index", "mf": "middle", "rf": "ring", "th": "thumb"}.get(prefix)


def classify_b_contact(model: mujoco.MjModel, cfg, geom1: int, geom2: int) -> dict:
    b_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_b_geom")
    if b_geom not in {int(geom1), int(geom2)}:
        raise ValueError("contact does not involve object B geom")
    other = int(geom2) if int(geom1) == b_geom else int(geom1)
    other_geom = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other)
    other_body_id = int(model.geom_bodyid[other])
    other_body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other_body_id)
    configured = {
        name: finger for finger, names in cfg.hand.finger_geom_mapping.items() for name in names
    }
    finger = configured.get(other_geom) or _finger_for_body(other_body)
    is_configured_tip = other_geom in configured
    if finger is not None:
        bucket = f"{finger}-B"
    elif other_geom == "table" or other_body == "world":
        bucket = "B-table"
    elif other_geom == "object_a_geom" or other_body == "object_a":
        bucket = "B-A"
    elif other_body == cfg.hand.palm_body:
        bucket = "palm-B"
    else:
        bucket = "other"
    return {
        "bucket": bucket,
        "finger": finger,
        "is_configured_fingertip": is_configured_tip,
        "other_geom_id": other,
        "other_geom_name": other_geom,
        "other_body_id": other_body_id,
        "other_body_name": other_body,
    }


def measure_b_penetration(model: mujoco.MjModel, data: mujoco.MjData, cfg) -> dict:
    b_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_b_geom")
    contacts = []
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        if b_geom not in {int(contact.geom1), int(contact.geom2)}:
            continue
        classified = classify_b_contact(model, cfg, int(contact.geom1), int(contact.geom2))
        depth = max(0.0, -float(contact.dist))
        body_ids = [int(model.geom_bodyid[int(contact.geom1)]), int(model.geom_bodyid[int(contact.geom2)])]
        contacts.append({
            "contact_index": int(contact_index),
            "penetration_depth_m": depth,
            "signed_distance_m": float(contact.dist),
            "geom_pair_ids": [int(contact.geom1), int(contact.geom2)],
            "geom_pair_names": [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)),
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)),
            ],
            "body_pair_ids": body_ids,
            "body_pair_names": [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) for body_id in body_ids
            ],
            **classified,
        })
    responsible = max(contacts, key=lambda record: record["penetration_depth_m"], default=None)
    per_pair = {
        bucket: max(
            [record["penetration_depth_m"] for record in contacts if record["bucket"] == bucket],
            default=0.0,
        )
        for bucket in PAIR_ORDER
    }
    return {
        "maximum_penetration_m": 0.0 if responsible is None else responsible["penetration_depth_m"],
        "responsible_contact": responsible,
        "per_pair_maximum_penetration_m": per_pair,
        "contact_count": len(contacts),
        "contacts": contacts,
    }


def _baseline_model_and_data(base_cfg, row: dict, frozen: dict[str, np.ndarray]):
    trial_cfg = scene_for_trial(base_cfg, row)
    model, data = build_scene(trial_cfg)
    _restore_state(model, data, frozen)
    return trial_cfg, model, data


def _step_original_phase2h_once(trial_cfg, model, data, frozen, row):
    indices = resolve_hand_indices(model, trial_cfg.hand)
    controller = JointImpedanceController(
        trial_cfg.task.impedance_stiffness,
        trial_cfg.task.impedance_damping,
        trial_cfg.task.torque_limit,
    )
    desired = np.asarray(frozen["post_release_controller_target"], dtype=float)
    q, qvel = hand_state(data, indices)
    data.ctrl[indices.actuator_ids] = controller.torque(desired, q, qvel)
    # Phase 2W/2H B-only behavior keeps parked A fixed on every integration step.
    cfg25, _ = load_phase2_5_config()
    a_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_a_free")
    a_qadr, a_vadr = model.jnt_qposadr[a_joint], model.jnt_dofadr[a_joint]
    data.qpos[a_qadr:a_qadr + 3] = cfg25.positive_control.parked_A_position_m
    data.qvel[a_vadr:a_vadr + 6] = 0.0
    mujoco.mj_step(model, data)


def measure_frozen_release(row: dict, cfg_cm: Phase2CMConfig, base_cfg) -> dict:
    path = ROOT / cfg_cm.output_dir / "frozen_release_states" / _state_filename(row["trial_id"])
    frozen = _load_state(path)
    trial_cfg, model, data = _baseline_model_and_data(base_cfg, row, frozen)
    boundary = measure_b_penetration(model, data, trial_cfg)
    _step_original_phase2h_once(trial_cfg, model, data, frozen, row)
    first_post = measure_b_penetration(model, data, trial_cfg)
    return {
        "trial_id": row["trial_id"],
        "wrist_pose_id": row["wrist_pose_id"],
        "release_boundary": boundary,
        "first_post_release_step": first_post,
    }


def _measure_original_worker(row: dict) -> dict:
    cfg_cm, _ = load_phase2cm_config()
    cfg25, _ = load_phase2_5_config()
    base_cfg = load_configs(scene_filename=cfg_cm.source_scene_filename)
    frozen = _capture_release_state(row, cfg25, base_cfg)
    trial_cfg, model, data = _baseline_model_and_data(base_cfg, row, frozen)
    boundary = measure_b_penetration(model, data, trial_cfg)
    _step_original_phase2h_once(trial_cfg, model, data, frozen, row)
    first_post = measure_b_penetration(model, data, trial_cfg)
    return {
        "trial_id": row["trial_id"],
        "wrist_pose_id": row["wrist_pose_id"],
        "release_boundary_maximum_penetration_m": boundary["maximum_penetration_m"],
        "release_boundary_responsible_bucket": None if boundary["responsible_contact"] is None else boundary["responsible_contact"]["bucket"],
        "release_boundary_responsible_geom_pair": None if boundary["responsible_contact"] is None else boundary["responsible_contact"]["geom_pair_names"],
        "release_boundary_per_pair_maximum_penetration_m": boundary["per_pair_maximum_penetration_m"],
        "first_post_release_maximum_penetration_m": first_post["maximum_penetration_m"],
        "first_post_release_responsible_bucket": None if first_post["responsible_contact"] is None else first_post["responsible_contact"]["bucket"],
        "first_post_release_per_pair_maximum_penetration_m": first_post["per_pair_maximum_penetration_m"],
    }


def reconstruct_original_release_measurements(
    cfg_cm: Phase2CMConfig,
    eligible: list[dict],
    workers: int,
) -> list[dict]:
    if not 1 <= workers <= 8:
        raise ValueError("Phase 2CM-P workers must be in [1, 8]")
    path = ROOT / cfg_cm.output_dir / "penetration_gate_original_release.jsonl"
    store = IncrementalJsonlStore(path, 30.0, 0.05)
    completed = store.completed_ids()
    pending = [row for row in eligible if row["trial_id"] not in completed]
    print(f"Phase 2CM-P original release: {len(completed)} complete, {len(pending)} pending", flush=True)
    batch = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, result in enumerate(executor.map(_measure_original_worker, pending, chunksize=2), start=1):
            batch.append(result)
            if len(batch) >= 16:
                store.append_many(batch)
                batch.clear()
            if index % 50 == 0 or index == len(pending):
                store.append_many(batch)
                batch.clear()
                print(f"Phase 2CM-P original release: {index}/{len(pending)}", flush=True)
    store.append_many(batch)
    records = store.records()
    expected = {row["trial_id"] for row in eligible}
    if {row["trial_id"] for row in records} != expected:
        raise RuntimeError("Phase 2CM-P original release audit is incomplete")
    return records


def _distribution(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "N": len(array),
        "median_m": float(np.median(array)),
        "mean_m": float(np.mean(array)),
        "p90_m": float(np.percentile(array, 90)),
        "p95_m": float(np.percentile(array, 95)),
        "p99_m": float(np.percentile(array, 99)),
        "maximum_m": float(np.max(array)),
        "bins": {
            "less_than_or_equal_1_mm": int(np.sum(array <= 0.001)),
            "greater_1_to_2_mm": int(np.sum((array > 0.001) & (array <= 0.002))),
            "greater_2_to_3_mm": int(np.sum((array > 0.002) & (array <= 0.003))),
            "greater_3_to_4_mm": int(np.sum((array > 0.003) & (array <= 0.004))),
            "greater_4_to_5_mm": int(np.sum((array > 0.004) & (array <= 0.005))),
            "greater_than_5_mm": int(np.sum(array > 0.005)),
        },
    }


def _pair_violation_counts(records: list[dict], threshold: float) -> dict:
    responsible = Counter()
    any_pair = Counter()
    for row in records:
        release = row["release_boundary"]
        if release["maximum_penetration_m"] > threshold and release["responsible_contact"] is not None:
            responsible[release["responsible_contact"]["bucket"]] += 1
        for bucket, depth in release["per_pair_maximum_penetration_m"].items():
            if depth > threshold:
                any_pair[bucket] += 1
    return {
        "responsible_maximum_pair": {bucket: responsible[bucket] for bucket in PAIR_ORDER},
        "states_with_any_violating_contact_by_pair": {bucket: any_pair[bucket] for bucket in PAIR_ORDER},
    }


def _paired_difference(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "N": len(array),
        "median_m": float(np.median(array)),
        "mean_m": float(np.mean(array)),
        "p95_absolute_m": float(np.percentile(np.abs(array), 95)),
        "maximum_absolute_m": float(np.max(np.abs(array))),
        "positive_count": int(np.sum(array > 1e-12)),
        "negative_count": int(np.sum(array < -1e-12)),
        "zero_within_1e-12_count": int(np.sum(np.abs(array) <= 1e-12)),
    }


def _code_definition() -> dict:
    return {
        "threshold_m": 0.003,
        "phase2W": {
            "path": "seqgrasp/experiments/phase2_5_trajectory.py",
            "definition": "maximum over -contact.distance for every contact whose body pair contains object_b, accumulated over pre-release and post-release steps",
            "included_pairs": {bucket: True for bucket in PAIR_ORDER},
            "notes": "Used by the Phase 2W acquisition summary/classifier; table and other-object contacts also have separate gates.",
        },
        "phase2H": {
            "path": "seqgrasp/experiments/phase2h_visuals.py via B_penetration_depths_m from seqgrasp/experiments/phase2_5_trajectory.py",
            "definition": "maximum across configured per-fingertip B contact distances; cumulative maximum begins at the first post-release sample",
            "included_pairs": {
                "index-B": True, "thumb-B": True, "middle-B": True, "ring-B": True,
                "palm-B": False, "B-table": False, "B-A": False, "other": False,
            },
            "pre_release_accumulated": False,
            "first_sample": "step == fixture_release_timestep, after the first unfixture-held mj_step",
        },
        "phase2CM_primary": {
            "path": "seqgrasp/experiments/phase2cm.py",
            "definition": "cumulative maximum over -contact.distance for every contact involving object_b, initialized to zero before the 500-step post-release replay",
            "included_pairs": {bucket: True for bucket in PAIR_ORDER},
            "pre_release_accumulated": False,
            "initial_boundary_checked": False,
            "first_sample": "after the first post-release mj_step",
        },
    }


def _write_markdown(payload: dict) -> None:
    frozen = payload["frozen_200"]
    all_eligible = payload["eligible_1521"]
    distribution = frozen["release_boundary_distribution"]
    pairs = frozen["release_boundary_violation_counts"]["responsible_maximum_pair"]
    wrist_rows = all_eligible["by_wrist_pose"]
    lines = [
        "# Phase 2CM-P penetration-gate audit",
        "",
        "## Result",
        "",
        f"The frozen penetration threshold remains `{payload['threshold_m']} m`. Of the 200 frozen Phase 2CM states, **{frozen['violating_at_release_boundary_count']}** already exceeded the gate at the exact release boundary before any post-release integration, while **{frozen['violating_at_first_post_release_step_count']}** exceeded it after the first post-release step. The primary replay classified {payload['primary_penetration_first_failure_count']} first failures as penetration failures.",
        "",
        "No threshold, physics parameter, contact geometry, controller, or primary result was changed by this audit.",
        "",
        "## Exact implementation audit",
        "",
        "- **Phase 2W:** `seqgrasp/experiments/phase2_5_trajectory.py` takes the maximum of `-contact.distance` over every extracted contact whose body pair contains `object_b`. This includes intended fingertips, other hand links/palm, table, and object A, and it accumulates across the complete pre- and post-release trajectory.",
        "- **Phase 2H:** `seqgrasp/experiments/phase2h_visuals.py` uses `B_penetration_depths_m`, which comes from the configured fingertip groups only. It includes index, middle, ring, and thumb fingertip–B contacts but excludes palm–B, table–B, B–A, and unconfigured other hand geoms. `np.maximum.accumulate` is applied only to the post-release slice beginning at `fixture_release_timestep`.",
        "- **Phase 2CM primary:** `seqgrasp/experiments/phase2cm.py` starts a new cumulative maximum at zero and includes every contact involving B, but it first evaluates the gate after the first post-release `mj_step`; it does not test the saved boundary state itself.",
        "",
        "Therefore intended fingertip–B solver overlap is included by all three paths. Phase 2CM's pair scope matches the broad Phase 2W summary, not Phase 2H's fingertip-only strict-series scope.",
        "",
        "## Frozen 200 release-state distribution",
        "",
        "| statistic | penetration [m] |",
        "|---|---:|",
        *[f"| {label} | {distribution[key]:.12g} |" for label, key in (("median", "median_m"), ("mean", "mean_m"), ("p90", "p90_m"), ("p95", "p95_m"), ("p99", "p99_m"), ("maximum", "maximum_m"))],
        "",
        "| interval | N |",
        "|---|---:|",
        *[f"| {name.replace('_', ' ')} | {count} |" for name, count in distribution["bins"].items()],
        "",
        "### Violating responsible pairs",
        "",
        "| pair | states where pair is maximum and >3 mm | states with any >3 mm contact of pair |",
        "|---|---:|---:|",
        *[f"| {bucket} | {pairs[bucket]} | {frozen['release_boundary_violation_counts']['states_with_any_violating_contact_by_pair'][bucket]} |" for bucket in PAIR_ORDER],
        "",
        "## Original Phase 2H comparison",
        "",
        f"Reconstructed-versus-frozen release-boundary penetration differences had median `{frozen['original_minus_frozen_release_boundary_difference_m']['median_m']:.12g} m`, mean `{frozen['original_minus_frozen_release_boundary_difference_m']['mean_m']:.12g} m`, and maximum absolute difference `{frozen['original_minus_frozen_release_boundary_difference_m']['maximum_absolute_m']:.12g} m`. This tests the same pre-integration boundary in independently reconstructed original Phase 2H trajectories.",
        "",
        "Pre-release pinch penetration is not accumulated into Phase 2H strict survival. It can nevertheless cause an immediate strict failure when the overlap persists into the first post-release sample, which is the sample at array index `fixture_release_timestep`.",
        "",
        "## All 1,521 eligible states",
        "",
        f"At the pre-integration release boundary, **{all_eligible['penetration_valid_at_release_boundary_count']}/{all_eligible['eligible_count']} ({100*all_eligible['penetration_valid_at_release_boundary_fraction']:.3f}%)** satisfy penetration `<= 0.003 m`. There {'are' if all_eligible['at_least_200_release_valid_states'] else 'are not'} at least 200 such states.",
        "",
        "| wrist pose | eligible N | penetration-valid boundary N | valid % | first-post valid N |",
        "|---|---:|---:|---:|---:|",
        *[f"| `{wrist}` | {row['eligible_count']} | {row['penetration_valid_at_release_boundary_count']} | {100*row['penetration_valid_at_release_boundary_fraction']:.3f} | {row['penetration_valid_at_first_post_release_step_count']} |" for wrist, row in sorted(wrist_rows.items())],
        "",
        "## Physical interpretation and next step",
        "",
        "MuJoCo's negative contact distance is solver overlap for the compiled rigid collision primitives. At the configured gripping interface, some negative distance is the intended compliant-contact representation rather than automatically proving gross geometric invalidity. The same signal can also identify invalid overlap with the table, object A, palm, or non-tip hand geometry. The pair identity and magnitude must therefore be kept explicit.",
        "",
        "This audit does not decide whether intended fingertip solver overlap should be inside the scientific penetration gate. The next experiment should be specified only after the PI confirms the intended pair scope. Without changing the 3 mm threshold, a later paired freeze can require release-boundary validity under that confirmed definition, provided the eligible population remains at least 200; no new states are selected here.",
    ]
    (ROOT / "docs" / "PHASE2CM_PENETRATION_GATE_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_penetration_gate_audit(workers: int = 8) -> dict:
    cfg_cm, _ = load_phase2cm_config()
    cfg25, _ = load_phase2_5_config()
    base_cfg = load_configs(scene_filename=cfg_cm.source_scene_filename)
    _, _, source_rows, metrics, _ = phase2h_sources(cfg_cm)
    eligible = eligible_release_trials(source_rows, metrics)
    selected = select_frozen_trials(eligible, cfg_cm.paired_release_state_target)
    threshold = float(cfg25.criteria.maximum_penetration_m)
    if threshold != 0.003:
        raise RuntimeError("Phase 2CM-P penetration threshold changed")

    frozen_records = [measure_frozen_release(row, cfg_cm, base_cfg) for row in selected]
    original_records = reconstruct_original_release_measurements(cfg_cm, eligible, workers)
    original_by_id = {row["trial_id"]: row for row in original_records}

    differences = [
        original_by_id[row["trial_id"]]["release_boundary_maximum_penetration_m"]
        - row["release_boundary"]["maximum_penetration_m"]
        for row in frozen_records
    ]
    boundary_values = [row["release_boundary"]["maximum_penetration_m"] for row in frozen_records]
    first_post_values = [row["first_post_release_step"]["maximum_penetration_m"] for row in frozen_records]
    frozen_payload = {
        "N": len(frozen_records),
        "violating_at_release_boundary_count": sum(value > threshold for value in boundary_values),
        "violating_at_first_post_release_step_count": sum(value > threshold for value in first_post_values),
        "release_boundary_distribution": _distribution(boundary_values),
        "first_post_release_distribution": _distribution(first_post_values),
        "release_boundary_violation_counts": _pair_violation_counts(frozen_records, threshold),
        "original_minus_frozen_release_boundary_difference_m": _paired_difference(differences),
        "records": frozen_records,
    }

    by_wrist: dict[str, list[dict]] = defaultdict(list)
    for row in original_records:
        by_wrist[row["wrist_pose_id"]].append(row)
    wrist_summary = {}
    for wrist, rows in sorted(by_wrist.items()):
        boundary_valid = sum(row["release_boundary_maximum_penetration_m"] <= threshold for row in rows)
        post_valid = sum(row["first_post_release_maximum_penetration_m"] <= threshold for row in rows)
        wrist_summary[wrist] = {
            "eligible_count": len(rows),
            "penetration_valid_at_release_boundary_count": boundary_valid,
            "penetration_valid_at_release_boundary_fraction": boundary_valid / len(rows),
            "penetration_valid_at_first_post_release_step_count": post_valid,
            "penetration_valid_at_first_post_release_step_fraction": post_valid / len(rows),
        }
    valid_boundary = sum(row["release_boundary_maximum_penetration_m"] <= threshold for row in original_records)
    valid_first_post = sum(row["first_post_release_maximum_penetration_m"] <= threshold for row in original_records)
    all_payload = {
        "eligible_count": len(original_records),
        "penetration_valid_at_release_boundary_count": valid_boundary,
        "penetration_valid_at_release_boundary_fraction": valid_boundary / len(original_records),
        "penetration_valid_at_first_post_release_step_count": valid_first_post,
        "penetration_valid_at_first_post_release_step_fraction": valid_first_post / len(original_records),
        "at_least_200_release_valid_states": valid_boundary >= 200,
        "release_boundary_distribution": _distribution([row["release_boundary_maximum_penetration_m"] for row in original_records]),
        "first_post_release_distribution": _distribution([row["first_post_release_maximum_penetration_m"] for row in original_records]),
        "by_wrist_pose": wrist_summary,
    }
    primary = _jsonl(ROOT / cfg_cm.output_dir / "paired_replay" / "candidate_results.jsonl")
    primary_cm3 = [row for row in primary if row["condition"] == "CM3"]
    primary_penetration = sum(row["first_failure_mechanism"] == "PENETRATION_LIMIT" for row in primary_cm3)
    already_boundary = sum(
        row["release_boundary"]["maximum_penetration_m"] > threshold
        and next(item for item in primary_cm3 if item["source_trial_id"] == row["trial_id"])["first_failure_mechanism"] == "PENETRATION_LIMIT"
        for row in frozen_records
    )
    payload = {
        "status": "complete",
        "threshold_m": threshold,
        "code_definition": _code_definition(),
        "frozen_200": frozen_payload,
        "eligible_1521": all_payload,
        "primary_penetration_first_failure_count": primary_penetration,
        "primary_penetration_failures_already_violating_at_release_boundary_count": already_boundary,
        "primary_penetration_failures_not_yet_violating_at_release_boundary_count": primary_penetration - already_boundary,
        "interpretation_guardrail": "No PI decision is made about whether intended fingertip solver overlap belongs in the scientific penetration gate.",
        "recommendation": "Keep the 0.003 m threshold unchanged. After PI confirmation of pair scope, freeze a later cohort that is release-boundary-valid under that confirmed definition and rerun the paired condim comparison; do not select it yet.",
    }
    output = ROOT / cfg_cm.output_dir / "penetration_gate_audit.json"
    _write_json(output, payload)
    _write_markdown(payload)
    return payload
