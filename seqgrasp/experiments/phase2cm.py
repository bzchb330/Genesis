from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np

from ..config import ConfigBundle, ROOT, load_configs
from ..control import JointImpedanceController, hand_state, resolve_hand_indices
from ..phase2_5_config import Phase25Config, load_phase2_5_config
from ..phase2cm_config import Phase2CMConfig, load_phase2cm_config
from ..scene_builder import build_scene
from ..sensing import extract_contacts, group_contacts_by_finger
from .phase2_5_trajectory import BAcquisitionTrajectory
from .phase2h_visuals import (
    FINGER_NAMES,
    INDEX,
    MIDDLE,
    RING,
    THUMB,
    assert_replay_matches,
    placement_from_trial,
    replay_trial,
    scene_for_trial,
    trajectory_from_trial,
)
from .resumable import IncrementalJsonlStore, stable_trial_id
from .second_grasp import _b_hand_state, _rotation_change


STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
HORIZONS = (1, 5, 10, 25, 50, 100, 200, 300, 400, 500)
VARIANT_ORDER = ("CM3", "CM4", "CM6")
PHASE2CM_STATE_VERSION = "phase2cm_release_state_v1"


class Phase2CMStop(RuntimeError):
    """Raised only for an explicit Phase 2CM STOP condition."""


class _ReleaseCaptured(RuntimeError):
    pass


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def phase2h_sources(cfg_cm: Phase2CMConfig) -> tuple[Path, Path, list[dict], dict[str, dict], str]:
    complete = []
    for summary_path in (ROOT / "outputs" / "phase2H" / "trial_metrics").rglob("summary.json"):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            payload.get("status") == "complete"
            and payload.get("trial_count") == 8192
            and payload.get("metric_method_id") == cfg_cm.source_metric_method_id
        ):
            complete.append((summary_path, payload))
    if len(complete) != 1:
        raise RuntimeError(f"expected one complete Phase 2H metric source, found {len(complete)}")
    metric_summary_path, metric_summary = complete[0]
    source_path = ROOT / metric_summary["source_candidate_results"]
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if source_hash != metric_summary["source_sha256"]:
        raise RuntimeError("Phase 2H source hash differs from the completed metric manifest")
    source_rows = _jsonl(source_path)
    metrics = {row["trial_id"]: row for row in _jsonl(metric_summary_path.with_name("metrics.jsonl"))}
    if len(source_rows) != 8192 or len(metrics) != 8192:
        raise RuntimeError("Phase 2H source or metric dataset is incomplete")
    return source_path, metric_summary_path, source_rows, metrics, source_hash


def eligible_release_trials(source_rows: Iterable[dict], metrics: dict[str, dict]) -> list[dict]:
    eligible = []
    for row in source_rows:
        metric = metrics[row["trial_id"]]
        if (
            bool(metric["dual_contact_at_release"])
            and not bool(row["middle_ring_assist"])
            and bool(row["numerically_valid"])
            and row.get("invalid_reason") is None
        ):
            eligible.append(row)
    return eligible


def _trial_hash(row: dict) -> str:
    return hashlib.sha256(str(row["trial_id"]).encode("utf-8")).hexdigest()


def select_frozen_trials(eligible: Iterable[dict], target: int) -> list[dict]:
    """Round-robin wrist strata; SHA-256 trial-ID order within every stratum."""

    by_wrist: dict[str, list[dict]] = defaultdict(list)
    for row in eligible:
        by_wrist[str(row["wrist_pose_id"])].append(row)
    for rows in by_wrist.values():
        rows.sort(key=lambda row: (_trial_hash(row), row["trial_id"]))
    wrist_ids = sorted(by_wrist)
    selected: list[dict] = []
    cursor = 0
    while len(selected) < target:
        appended = False
        for wrist_id in wrist_ids:
            rows = by_wrist[wrist_id]
            if cursor < len(rows):
                selected.append(rows[cursor])
                appended = True
                if len(selected) == target:
                    break
        if not appended:
            break
        cursor += 1
    if len(selected) != target:
        raise RuntimeError(f"could freeze only {len(selected)} of {target} required release states")
    if len({row["trial_id"] for row in selected}) != target:
        raise AssertionError("deterministic Phase 2CM selection reused a source trial")
    return selected


def _geom_type_name(value: int) -> str:
    for name in dir(mujoco.mjtGeom):
        if name.startswith("mjGEOM_") and int(getattr(mujoco.mjtGeom, name)) == int(value):
            return name.removeprefix("mjGEOM_").lower()
    return str(int(value))


def _model_geom_row(model: mujoco.MjModel, geom_id: int, role: str) -> dict:
    body_id = int(model.geom_bodyid[geom_id])
    return {
        "role": role,
        "geom_id": int(geom_id),
        "geom_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id),
        "body_id": body_id,
        "body_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
        "geom_type": _geom_type_name(int(model.geom_type[geom_id])),
        "geom_size": model.geom_size[geom_id].copy(),
        "condim": int(model.geom_condim[geom_id]),
        "friction": model.geom_friction[geom_id, :3].copy(),
        "solref": model.geom_solref[geom_id].copy(),
        "solimp": model.geom_solimp[geom_id].copy(),
        "priority": int(model.geom_priority[geom_id]),
        "solmix": float(model.geom_solmix[geom_id]),
        "contype": int(model.geom_contype[geom_id]),
        "conaffinity": int(model.geom_conaffinity[geom_id]),
    }


def compiled_contact_audit(base_cfg: ConfigBundle, example_trial: dict) -> tuple[dict, mujoco.MjModel]:
    trial_cfg = scene_for_trial(base_cfg, example_trial)
    model, _ = build_scene(trial_cfg)
    rows = []
    for finger in FINGER_NAMES:
        for name in trial_cfg.hand.finger_geom_mapping[finger]:
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            rows.append(_model_geom_row(model, geom_id, f"{finger}_fingertip"))
    palm_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, trial_cfg.hand.palm_body)
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) == palm_body:
            rows.append(_model_geom_row(model, geom_id, "palm"))
    for name, role in (("object_b_geom", "object_B"), ("table", "table")):
        rows.append(_model_geom_row(model, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name), role))
    payload = {
        "scene": base_cfg.scene.objects[1].name,
        "example_wrist_pose_id": example_trial["wrist_pose_id"],
        "model_counts": {"nbody": int(model.nbody), "ngeom": int(model.ngeom), "njnt": int(model.njnt), "nu": int(model.nu)},
        "geoms": rows,
    }
    return payload, model


def _runtime_contact_rows_for_trial(
    row: dict,
    cfg25: Phase25Config,
    base_cfg: ConfigBundle,
) -> tuple[list[dict], dict]:
    records: list[dict] = []
    tip_names = {
        name: finger for finger, names in base_cfg.hand.finger_geom_mapping.items() for name in names
    }

    def callback(step: int, model: mujoco.MjModel, data: mujoco.MjData, _sample: dict) -> None:
        b_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_b_geom")
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if b_geom not in pair:
                continue
            other = int(contact.geom2) if int(contact.geom1) == b_geom else int(contact.geom1)
            other_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or ""
            finger = tip_names.get(other_name)
            if finger not in {"index", "thumb"}:
                continue
            wrench = np.zeros(6, dtype=float)
            mujoco.mj_contactForce(model, data, contact_index, wrench)
            records.append({
                "trial_id": row["trial_id"],
                "wrist_pose_id": row["wrist_pose_id"],
                "step": int(step),
                "contact_index": int(contact_index),
                "finger": finger,
                "geom_pair_ids": [int(contact.geom1), int(contact.geom2)],
                "geom_pair_names": [
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)),
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)),
                ],
                "dim": int(contact.dim),
                "friction": np.asarray(contact.friction, dtype=float).copy(),
                "position_m": np.asarray(contact.pos, dtype=float).copy(),
                "frame_rows_world": np.asarray(contact.frame, dtype=float).reshape(3, 3).copy(),
                "mj_contactForce": wrench,
            })

    summary, _ = replay_trial(row, cfg25, base_cfg, diagnostic_callback=callback)
    assert_replay_matches(row, summary)
    return records, summary


def _markdown_table(rows: list[dict]) -> str:
    columns = ("role", "geom_id", "geom_name", "body_id", "body_name", "geom_type", "geom_size", "condim", "friction", "solref", "solimp", "priority", "solmix", "contype", "conaffinity")
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        values = []
        for key in columns:
            value = row[key]
            if isinstance(value, np.ndarray):
                value = value.tolist()
            if value is None:
                value = "(unnamed)"
            values.append(f"`{value}`" if key not in {"role", "geom_type"} else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def run_contact_audit(cfg_cm: Phase2CMConfig | None = None) -> dict:
    cfg_cm = cfg_cm or load_phase2cm_config()[0]
    cfg25, _ = load_phase2_5_config()
    base_cfg = load_configs(scene_filename=cfg_cm.source_scene_filename)
    source_path, metric_path, source_rows, metrics, source_hash = phase2h_sources(cfg_cm)
    eligible = eligible_release_trials(source_rows, metrics)
    audit_trials = sorted(eligible, key=lambda row: (_trial_hash(row), row["trial_id"]))[:cfg_cm.runtime_contact_audit_trials]
    compiled, model = compiled_contact_audit(base_cfg, audit_trials[0])
    output = ROOT / cfg_cm.output_dir / "contact_model_audit"
    _write_json(output / "compiled_contact_model.json", compiled)
    runtime_records = []
    replay_summaries = []
    for row in audit_trials:
        records, summary = _runtime_contact_rows_for_trial(row, cfg25, base_cfg)
        runtime_records.extend(records)
        replay_summaries.append({"trial_id": row["trial_id"], "record_count": len(records), "replay_failure_mechanism": summary["failure_mechanism"]})
    runtime_path = output / "runtime_contacts.jsonl"
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path.write_text(
        "".join(json.dumps(record, sort_keys=True, default=_json_default, allow_nan=False) + "\n" for record in runtime_records),
        encoding="utf-8",
    )
    dims = {
        finger: sorted({int(record["dim"]) for record in runtime_records if record["finger"] == finger})
        for finger in ("index", "thumb")
    }
    if any(not values for values in dims.values()):
        raise RuntimeError(f"runtime audit did not observe both fingertip-B contacts: {dims}")
    summary = {
        "source_candidate_results": source_path.relative_to(ROOT).as_posix(),
        "source_metric_summary": metric_path.relative_to(ROOT).as_posix(),
        "source_sha256": source_hash,
        "eligible_release_state_count": len(eligible),
        "sampled_trial_ids": [row["trial_id"] for row in audit_trials],
        "sampled_replays": replay_summaries,
        "runtime_record_count": len(runtime_records),
        "runtime_contact_dims": dims,
        "compiled_index_condim": next(row["condim"] for row in compiled["geoms"] if row["role"] == "index_fingertip"),
        "compiled_thumb_condim": next(row["condim"] for row in compiled["geoms"] if row["role"] == "thumb_fingertip"),
        "compiled_fingertip_friction": next(row["friction"] for row in compiled["geoms"] if row["role"] == "index_fingertip"),
        "compiled_B_friction": next(row["friction"] for row in compiled["geoms"] if row["role"] == "object_B"),
    }
    _write_json(output / "summary.json", summary)
    doc = (
        "# Phase 2CM compiled contact audit\n\n"
        "This audit compiles the exact half-scale scene and wrist-transformed model used by the deterministic Phase 2W/2H replay path. Values below are read from `mujoco.MjModel`, not inferred from source XML.\n\n"
        + _markdown_table(compiled["geoms"])
        + "\n\n## Runtime contact audit\n\n"
        + f"A deterministic SHA-256-ordered sample of {len(audit_trials)} eligible Phase 2H trials produced {len(runtime_records)} index–B/thumb–B contact records. Every record, including contact frame and the six-value `mj_contactForce` buffer, is stored at `{runtime_path.relative_to(ROOT).as_posix()}`.\n\n"
        + f"- actual runtime index–B contact dimensions: `{dims['index']}`\n"
        + f"- actual runtime thumb–B contact dimensions: `{dims['thumb']}`\n"
        + "- sample trial IDs: " + ", ".join(f"`{row['trial_id']}`" for row in audit_trials) + "\n"
    )
    (ROOT / "docs" / "PHASE2CM_COMPILED_CONTACT_AUDIT.md").write_text(doc, encoding="utf-8")
    baseline_dims = {summary["compiled_index_condim"], summary["compiled_thumb_condim"], *dims["index"], *dims["thumb"]}
    if any(value >= 4 for value in baseline_dims):
        raise Phase2CMStop("PHASE2CM_BASELINE_ALREADY_TORSIONAL")
    if baseline_dims != {3}:
        raise RuntimeError(f"baseline compiled/runtime contact dimension mismatch: {sorted(baseline_dims)}")
    return summary


def fingertip_geom_ids(model: mujoco.MjModel, cfg: ConfigBundle) -> np.ndarray:
    ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for finger in FINGER_NAMES for name in cfg.hand.finger_geom_mapping[finger]
    ]
    if any(value < 0 for value in ids):
        raise RuntimeError("configured fingertip geom missing from Phase 2CM model")
    return np.asarray(ids, dtype=int)


def construct_contact_variant(base_cfg: ConfigBundle, row: dict, condim: int) -> tuple[ConfigBundle, mujoco.MjModel, mujoco.MjData]:
    if condim not in {3, 4, 6}:
        raise ValueError("Phase 2CM contact dimension must be 3, 4, or 6")
    trial_cfg = scene_for_trial(base_cfg, row)
    model, data = build_scene(trial_cfg)
    model.geom_condim[fingertip_geom_ids(model, trial_cfg)] = condim
    return trial_cfg, model, data


def _array_signature(model: mujoco.MjModel) -> dict[str, np.ndarray]:
    names = (
        "body_pos", "body_quat", "body_ipos", "body_iquat", "body_mass", "body_inertia", "body_gravcomp",
        "jnt_type", "jnt_bodyid", "jnt_qposadr", "jnt_dofadr", "jnt_axis", "jnt_pos", "jnt_range", "jnt_limited", "jnt_solref", "jnt_solimp", "jnt_stiffness", "jnt_margin",
        "actuator_trntype", "actuator_dyntype", "actuator_gaintype", "actuator_biastype", "actuator_trnid", "actuator_dynprm", "actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange", "actuator_forcerange", "actuator_actrange", "actuator_gear", "actuator_cranklength", "actuator_ctrllimited", "actuator_forcelimited", "actuator_actlimited",
        "geom_type", "geom_bodyid", "geom_contype", "geom_conaffinity", "geom_pos", "geom_quat", "geom_size", "geom_friction", "geom_solref", "geom_solimp", "geom_priority", "geom_solmix", "geom_margin", "geom_gap",
    )
    return {name: np.asarray(getattr(model, name)).copy() for name in names if hasattr(model, name)}


def _option_signature(model: mujoco.MjModel) -> dict[str, object]:
    names = (
        "timestep", "apirate", "impratio", "tolerance", "ls_tolerance", "noslip_tolerance", "mpr_tolerance",
        "gravity", "wind", "magnetic", "density", "viscosity", "o_margin", "integrator", "cone", "jacobian", "solver",
        "iterations", "ls_iterations", "noslip_iterations", "mpr_iterations", "disableflags", "enableflags", "disableactuator",
    )
    result = {}
    for name in names:
        if hasattr(model.opt, name):
            value = getattr(model.opt, name)
            result[name] = np.asarray(value).copy() if np.ndim(value) else value
    return result


def verify_model_isolation(
    base_cfg: ConfigBundle,
    row: dict,
    variants: dict[str, int],
    *,
    write_artifact: bool = True,
) -> dict:
    built = {name: construct_contact_variant(base_cfg, row, condim) for name, condim in variants.items()}
    reference_cfg, reference, _ = built["CM3"]
    reference_arrays = _array_signature(reference)
    reference_options = _option_signature(reference)
    tip_ids = fingertip_geom_ids(reference, reference_cfg)
    checks = []
    failures = []
    counts = ("nbody", "njnt", "nu", "ngeom", "nq", "nv", "na")
    for name in VARIANT_ORDER:
        cfg, model, _ = built[name]
        for key in counts:
            passed = int(getattr(model, key)) == int(getattr(reference, key))
            checks.append({"variant": name, "field": key, "equal": passed})
            if not passed:
                failures.append(f"{name}:{key}")
        arrays = _array_signature(model)
        for key, expected in reference_arrays.items():
            passed = np.array_equal(arrays[key], expected)
            checks.append({"variant": name, "field": key, "equal": passed})
            if not passed:
                failures.append(f"{name}:{key}")
        options = _option_signature(model)
        for key, expected in reference_options.items():
            passed = np.array_equal(np.asarray(options[key]), np.asarray(expected))
            checks.append({"variant": name, "field": f"opt.{key}", "equal": passed})
            if not passed:
                failures.append(f"{name}:opt.{key}")
        expected_condim = reference.geom_condim.copy()
        expected_condim[tip_ids] = variants[name]
        passed = np.array_equal(model.geom_condim, expected_condim)
        checks.append({"variant": name, "field": "geom_condim_expected_only", "equal": passed})
        if not passed:
            failures.append(f"{name}:geom_condim")
        if not np.array_equal(fingertip_geom_ids(model, cfg), tip_ids):
            failures.append(f"{name}:fingertip_geom_ids")
    payload = {
        "status": "PASS" if not failures else "FAIL",
        "reference_variant": "CM3",
        "intended_changed_field": "geom_condim on configured fingertip collision geoms only",
        "fingertip_geom_ids": tip_ids,
        "variant_condim": variants,
        "checks": checks,
        "failures": failures,
    }
    if write_artifact:
        _write_json(ROOT / "outputs" / "phase2CM" / "model_isolation.json", payload)
    if failures:
        raise Phase2CMStop("PHASE2CM_MODEL_ISOLATION_FAILED")
    return payload


def write_fingertip_geometry_audit(base_cfg: ConfigBundle, example_trial: dict) -> dict:
    trial_cfg = scene_for_trial(base_cfg, example_trial)
    model, _ = build_scene(trial_cfg)
    rows = []
    for finger in FINGER_NAMES:
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in trial_cfg.hand.finger_geom_mapping[finger]]
        rows.append({
            "finger": finger,
            "collision_geom_count": len(ids),
            "geoms": [{"id": int(geom_id), "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id), "type": _geom_type_name(model.geom_type[geom_id]), "size": model.geom_size[geom_id].copy()} for geom_id in ids],
            "representation": "single primitive" if len(ids) == 1 else "multiple collision primitives",
            "rigid_contact_limitation": "Rigid capsule contact is resolved at MuJoCo contact points; it is not a deformable distributed fingertip patch or taxel array.",
        })
    output = ROOT / "outputs" / "phase2CM" / "contact_model_audit" / "fingertip_geometry.json"
    _write_json(output, {"fingertips": rows})
    lines = ["# Phase 2CM fingertip geometry audit", "", "No geometry is modified by Phase 2CM.", "", "| finger | collision geoms | type | size | representation | limitation |", "|---|---:|---|---|---|---|"]
    for row in rows:
        geom = row["geoms"][0]
        lines.append(f"| {row['finger']} | {row['collision_geom_count']} | {geom['type']} | `{np.asarray(geom['size']).tolist()}` | {row['representation']} | {row['rigid_contact_limitation']} |")
    (ROOT / "docs" / "PHASE2CM_FINGERTIP_GEOMETRY_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"fingertips": rows}


def _state_filename(trial_id: str) -> str:
    return hashlib.sha256(trial_id.encode("utf-8")).hexdigest() + ".npz"


def freeze_trial_ids(
    cfg_cm: Phase2CMConfig,
    eligible: list[dict],
    source_hash: str,
) -> tuple[list[dict], Path]:
    selected = select_frozen_trials(eligible, cfg_cm.paired_release_state_target)
    state_root = ROOT / cfg_cm.output_dir / "frozen_release_states"
    manifest_path = state_root / "frozen_trial_ids.json"
    by_wrist = Counter(row["wrist_pose_id"] for row in selected)
    manifest = {
        "experiment_id": cfg_cm.experiment_id,
        "state_version": PHASE2CM_STATE_VERSION,
        "source_sha256": source_hash,
        "eligibility": {
            "index_contacts_B_at_release": True,
            "thumb_contacts_B_at_release": True,
            "middle_assists_B": False,
            "ring_assists_B": False,
            "numerically_valid": True,
        },
        "selection": "round-robin over sorted wrist IDs; SHA-256(source trial ID) order within each wrist",
        "selection_excludes": ["Phase 2H survival", "Phase 2H failure", "visual appearance", "rotation", "slip", "CM outcome"],
        "eligible_count": len(eligible),
        "target_count": cfg_cm.paired_release_state_target,
        "frozen_count": len(selected),
        "wrist_counts": dict(sorted(by_wrist.items())),
        "trials": [
            {
                "selection_index": index,
                "trial_id": row["trial_id"],
                "trial_id_sha256": _trial_hash(row),
                "wrist_pose_id": row["wrist_pose_id"],
                "release_step": int(row["fixture_release_timestep"]),
                "state_path": (state_root / _state_filename(row["trial_id"])).relative_to(ROOT).as_posix(),
            }
            for index, row in enumerate(selected)
        ],
    }
    # This manifest is deliberately written before any CM3/CM4/CM6 outcome.
    _write_json(manifest_path, manifest)
    lines = [
        "# Phase 2CM paired trial freeze",
        "",
        f"The freeze contains exactly {len(selected)} of {len(eligible)} eligible existing Phase 2H release states. IDs were frozen before any CM3/CM4/CM6 outcome replay.",
        "",
        "Selection uses round-robin wrist strata with SHA-256 trial-ID ordering within each wrist. It does not use Phase 2H survival or failure, appearance, rotation, slip, or any future CM outcome.",
        "",
        "## Wrist distribution",
        "",
        "| wrist pose ID | frozen N |",
        "|---|---:|",
        *[f"| `{wrist}` | {count} |" for wrist, count in sorted(by_wrist.items())],
        "",
        "## Frozen source IDs",
        "",
        "| index | wrist pose ID | source trial ID | SHA-256 order key |",
        "|---:|---|---|---|",
        *[
            f"| {index} | `{row['wrist_pose_id']}` | `{row['trial_id']}` | `{_trial_hash(row)}` |"
            for index, row in enumerate(selected)
        ],
        "",
        f"Machine-readable manifest: `{manifest_path.relative_to(ROOT).as_posix()}`.",
    ]
    (ROOT / "docs" / "PHASE2CM_PAIRED_TRIAL_FREEZE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return selected, manifest_path


def _capture_release_state(
    row: dict,
    cfg25: Phase25Config,
    base_cfg: ConfigBundle,
) -> dict[str, np.ndarray | str | int | float]:
    release_step = int(row["fixture_release_timestep"])
    captured: dict[str, object] = {}

    def callback(step: int, model: mujoco.MjModel, data: mujoco.MjData, sample: dict) -> None:
        if step != release_step - 1:
            return
        state = np.empty(mujoco.mj_stateSize(model, STATE_SPEC), dtype=float)
        mujoco.mj_getState(model, data, state, STATE_SPEC)
        b_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object_b")
        velocity = np.zeros(6, dtype=float)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b_body, velocity, 0)
        palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_cfg.hand.palm_body)
        trajectory = trajectory_from_trial(row)
        captured.update({
            "state_spec": int(STATE_SPEC),
            "integration_state": state,
            "qpos": data.qpos.copy(),
            "qvel": data.qvel.copy(),
            "act": data.act.copy(),
            "ctrl": data.ctrl.copy(),
            "qacc_warmstart": data.qacc_warmstart.copy(),
            "controller_target_at_capture": np.asarray(sample["commanded_joint_target_rad"], dtype=float).copy(),
            "post_release_controller_target": np.asarray(trajectory.hold_joint_rad, dtype=float),
            "B_position": data.xpos[b_body].copy(),
            "B_quaternion": data.xquat[b_body].copy(),
            "B_linear_velocity": velocity[3:].copy(),
            "B_angular_velocity": velocity[:3].copy(),
            "wrist_position": data.xpos[palm].copy(),
            "wrist_quaternion": data.xquat[palm].copy(),
            "release_step": release_step,
            "captured_after_step": int(step),
            "time": float(data.time),
        })
        raise _ReleaseCaptured

    try:
        replay_trial(row, cfg25, base_cfg, diagnostic_callback=callback)
    except _ReleaseCaptured:
        pass
    if not captured:
        raise RuntimeError(f"failed to capture release state for {row['trial_id']}")
    return captured


def _atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def reconstruct_release_states(
    cfg_cm: Phase2CMConfig,
    selected: list[dict],
    cfg25: Phase25Config,
    base_cfg: ConfigBundle,
) -> Path:
    state_root = ROOT / cfg_cm.output_dir / "frozen_release_states"
    store = IncrementalJsonlStore(state_root / "state_manifest.jsonl", 30.0, 0.05)
    completed = store.completed_ids()
    for index, row in enumerate(selected, start=1):
        if row["trial_id"] in completed:
            continue
        captured = _capture_release_state(row, cfg25, base_cfg)
        path = state_root / _state_filename(row["trial_id"])
        metadata = {
            "state_version": PHASE2CM_STATE_VERSION,
            "source_trial_id": row["trial_id"],
            "wrist_pose_id": row["wrist_pose_id"],
            "release_state": "after final fixture-held step and before first post-release integration step",
            "fixture_release_timestep": int(row["fixture_release_timestep"]),
            "controller": {
                "type": "JointImpedanceController",
                "stiffness": base_cfg.task.impedance_stiffness,
                "damping": base_cfg.task.impedance_damping,
                "torque_limit": base_cfg.task.torque_limit,
                "internal_dynamic_state": "stateless",
            },
            "actuator_state_applicable": bool(np.asarray(captured["act"]).size),
            "determinism": {
                "runtime_rng_used": False,
                "source_trial_id": row["trial_id"],
                "source_config_hash": row["config_hash"],
                "trajectory_candidate_index": int(row["candidate_index"]),
                "selection_hash": _trial_hash(row),
            },
        }
        _atomic_savez(path, **captured, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
        store.append({
            "trial_id": row["trial_id"],
            "wrist_pose_id": row["wrist_pose_id"],
            "release_step": int(row["fixture_release_timestep"]),
            "state_path": path.relative_to(ROOT).as_posix(),
            "state_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
        if index % 10 == 0 or index == len(selected):
            print(f"Phase 2CM release states: {index}/{len(selected)}", flush=True)
    records = store.records()
    if len(records) != len(selected):
        raise RuntimeError(f"release state reconstruction incomplete: {len(records)}/{len(selected)}")
    return store.path


def _load_state(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def _restore_state(model: mujoco.MjModel, data: mujoco.MjData, frozen: dict[str, np.ndarray]) -> None:
    if int(frozen["state_spec"]) != int(STATE_SPEC):
        raise RuntimeError("release state specification changed")
    if len(frozen["integration_state"]) != mujoco.mj_stateSize(model, STATE_SPEC):
        raise RuntimeError("release state size differs from counterfactual model")
    mujoco.mj_setState(model, data, frozen["integration_state"], STATE_SPEC)
    mujoco.mj_forward(model, data)


def _object_velocity(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> np.ndarray:
    velocity = np.zeros(6, dtype=float)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0)
    return velocity


def _quaternion_difference(left: np.ndarray, right: np.ndarray) -> float:
    """Angular difference robust to harmless quaternion norm roundoff."""

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if np.array_equal(left, right):
        return 0.0
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return math.inf
    cosine = abs(float(np.dot(left, right))) / denominator
    return float(2.0 * np.arccos(np.clip(cosine, 0.0, 1.0)))


def verify_release_state_equivalence(
    cfg_cm: Phase2CMConfig,
    selected: list[dict],
    base_cfg: ConfigBundle,
) -> tuple[list[dict], list[dict]]:
    state_root = ROOT / cfg_cm.output_dir / "frozen_release_states"
    records = []
    valid = []
    for index, row in enumerate(selected, start=1):
        frozen = _load_state(state_root / _state_filename(row["trial_id"]))
        restored = {}
        for name in VARIANT_ORDER:
            trial_cfg, model, data = construct_contact_variant(base_cfg, row, cfg_cm.variants[name])
            _restore_state(model, data, frozen)
            b_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object_b")
            palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, trial_cfg.hand.palm_body)
            velocity = _object_velocity(model, data, b_body)
            restored[name] = {
                "qpos": data.qpos.copy(),
                "qvel": data.qvel.copy(),
                "B_position": data.xpos[b_body].copy(),
                "B_quaternion": data.xquat[b_body].copy(),
                "B_linear_velocity": velocity[3:].copy(),
                "B_angular_velocity": velocity[:3].copy(),
                "wrist_position": data.xpos[palm].copy(),
                "wrist_quaternion": data.xquat[palm].copy(),
                "controller_target": frozen["post_release_controller_target"].copy(),
            }
        reference = restored["CM3"]
        differences = {}
        triplet_valid = True
        for name in ("CM4", "CM6"):
            candidate = restored[name]
            diff = {
                "qpos_max_abs": float(np.max(np.abs(candidate["qpos"] - reference["qpos"]))),
                "qvel_max_abs": float(np.max(np.abs(candidate["qvel"] - reference["qvel"]))),
                "B_position_max_abs_m": float(np.max(np.abs(candidate["B_position"] - reference["B_position"]))),
                "B_orientation_difference_rad": _quaternion_difference(candidate["B_quaternion"], reference["B_quaternion"]),
                "B_linear_velocity_max_abs_m_per_s": float(np.max(np.abs(candidate["B_linear_velocity"] - reference["B_linear_velocity"]))),
                "B_angular_velocity_max_abs_rad_per_s": float(np.max(np.abs(candidate["B_angular_velocity"] - reference["B_angular_velocity"]))),
                "wrist_position_max_abs_m": float(np.max(np.abs(candidate["wrist_position"] - reference["wrist_position"]))),
                "wrist_orientation_difference_rad": _quaternion_difference(candidate["wrist_quaternion"], reference["wrist_quaternion"]),
                "controller_target_max_abs_rad": float(np.max(np.abs(candidate["controller_target"] - reference["controller_target"]))),
            }
            differences[f"{name}_minus_CM3"] = diff
            triplet_valid &= all(value <= 1e-12 for value in diff.values())
        record = {
            "source_trial_id": row["trial_id"],
            "wrist_pose_id": row["wrist_pose_id"],
            "valid_complete_triplet": bool(triplet_valid),
            "differences": differences,
        }
        records.append(record)
        if triplet_valid:
            valid.append(row)
        if index % 25 == 0 or index == len(selected):
            print(f"Phase 2CM release equivalence: {index}/{len(selected)} valid={len(valid)}", flush=True)
    payload = {
        "frozen_state_count": len(selected),
        "valid_complete_triplet_count": len(valid),
        "rejected_triplet_count": len(selected) - len(valid),
        "absolute_tolerance": 1e-12,
        "records": records,
    }
    _write_json(ROOT / cfg_cm.output_dir / "release_state_equivalence.json", payload)
    return valid, records


def _finger_b_wrenches(model: mujoco.MjModel, data: mujoco.MjData, cfg: ConfigBundle):
    b_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_b_geom")
    by_geom = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name): finger
        for finger, names in cfg.hand.finger_geom_mapping.items() for name in names
    }
    counts = np.zeros(4, dtype=np.int64)
    dims = np.zeros(4, dtype=np.int64)
    normal = np.zeros(4, dtype=float)
    tangential = np.zeros((4, 2), dtype=float)
    torsional = np.full(4, np.nan, dtype=float)
    rolling = np.full((4, 2), np.nan, dtype=float)
    finger_index = {finger: index for index, finger in enumerate(FINGER_NAMES)}
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if b_geom not in pair:
            continue
        other = int(contact.geom2) if int(contact.geom1) == b_geom else int(contact.geom1)
        finger = by_geom.get(other)
        if finger is None:
            continue
        index = finger_index[finger]
        wrench = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(model, data, contact_index, wrench)
        counts[index] += 1
        dims[index] = max(dims[index], int(contact.dim))
        normal[index] += abs(float(wrench[0]))
        tangential[index] += wrench[1:3]
        if int(contact.dim) >= 4:
            torsional[index] = np.nan_to_num(torsional[index]) + wrench[3]
        if int(contact.dim) >= 6:
            rolling[index] = np.nan_to_num(rolling[index]) + wrench[4:6]
    return counts, dims, normal, tangential, torsional, rolling


def _failure_mechanism(
    counts: np.ndarray,
    hand_contacts: int,
    hand_force: float,
    b_table: bool,
    maximum_penetration: float,
    maximum_translation: float,
    maximum_rotation: float,
    numeric: bool,
    criteria,
) -> str | None:
    if not numeric:
        return "OTHER"
    if hand_contacts < criteria.minimum_B_hand_contacts or not (counts[INDEX] and counts[THUMB]):
        return "CONTACT_LOST"
    if hand_force < criteria.minimum_B_normal_force_N:
        return "OTHER"
    if counts[MIDDLE] or counts[RING]:
        return "OTHER"
    if maximum_penetration > criteria.maximum_penetration_m:
        return "PENETRATION_LIMIT"
    if maximum_translation > criteria.maximum_B_translation_m:
        return "TRANSLATION_LIMIT"
    if maximum_rotation > criteria.maximum_B_orientation_rad:
        return "B_ROTATED_OUT"
    if b_table:
        return "B_SLIPPED_TO_TABLE"
    return None


def replay_release_state(
    cfg_cm: Phase2CMConfig,
    cfg25: Phase25Config,
    base_cfg: ConfigBundle,
    row: dict,
    variant: str,
    frozen: dict[str, np.ndarray],
    series_path: Path,
) -> dict:
    trial_cfg, model, data = construct_contact_variant(base_cfg, row, cfg_cm.variants[variant])
    _restore_state(model, data, frozen)
    indices = resolve_hand_indices(model, trial_cfg.hand)
    controller = JointImpedanceController(
        trial_cfg.task.impedance_stiffness,
        trial_cfg.task.impedance_damping,
        trial_cfg.task.torque_limit,
    )
    desired = np.asarray(frozen["post_release_controller_target"], dtype=float)
    b_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object_b")
    reference_position = np.asarray(frozen["B_position"], dtype=float)
    reference_quaternion = np.asarray(frozen["B_quaternion"], dtype=float)
    ranges = model.jnt_range[indices.joint_ids]
    steps = cfg_cm.post_release_steps
    per_finger_counts = np.zeros((steps, 4), dtype=np.int64)
    contact_dims = np.zeros((steps, 4), dtype=np.int64)
    per_finger_normal = np.zeros((steps, 4), dtype=float)
    per_finger_tangential = np.zeros((steps, 4, 2), dtype=float)
    per_finger_torsional = np.full((steps, 4), np.nan, dtype=float)
    per_finger_rolling = np.full((steps, 4, 2), np.nan, dtype=float)
    b_table_series = np.zeros(steps, dtype=bool)
    translation = np.zeros(steps, dtype=float)
    rotation = np.zeros(steps, dtype=float)
    vertical = np.zeros(steps, dtype=float)
    actuator_saturation = np.zeros((steps, model.nu), dtype=bool)
    minimum_joint_margin = np.zeros(steps, dtype=float)
    strict = np.zeros(steps, dtype=bool)
    dual = np.zeros(steps, dtype=bool)
    maximum_penetration = 0.0
    maximum_translation = 0.0
    maximum_rotation = 0.0
    table_seen = False
    numeric = True
    first_failure_step = None
    first_failure_mechanism = None
    first_table_step = None
    first_contact_loss_step = None
    first_rotation_step = None
    for step in range(steps):
        q, qvel = hand_state(data, indices)
        controls = controller.torque(desired, q, qvel)
        data.ctrl[indices.actuator_ids] = controls
        mujoco.mj_step(model, data)
        numeric &= all(np.all(np.isfinite(value)) for value in (data.qpos, data.qvel, data.ctrl))
        contacts = extract_contacts(model, data)
        counts, dims, normal, tangential, torsional, rolling = _finger_b_wrenches(model, data, trial_cfg)
        per_finger_counts[step] = counts
        contact_dims[step] = dims
        per_finger_normal[step] = normal
        per_finger_tangential[step] = tangential
        per_finger_torsional[step] = torsional
        per_finger_rolling[step] = rolling
        b_table = any({record.geom1_name, record.geom2_name} == {"object_b_geom", "table"} for record in contacts)
        b_table_series[step] = b_table
        table_seen |= b_table
        penetration = max([0.0, *[-record.distance for record in contacts if "object_b" in {record.body1_name, record.body2_name}]])
        maximum_penetration = max(maximum_penetration, penetration)
        translation[step] = np.linalg.norm(data.xpos[b_body] - reference_position)
        rotation[step] = _rotation_change(data.xquat[b_body], reference_quaternion)
        vertical[step] = data.xpos[b_body, 2] - reference_position[2]
        maximum_translation = max(maximum_translation, translation[step])
        maximum_rotation = max(maximum_rotation, rotation[step])
        actuator_saturation[step] = np.isclose(np.abs(controls), trial_cfg.task.torque_limit, atol=1e-10)
        q_after = data.qpos[indices.qpos_addresses]
        minimum_joint_margin[step] = np.min(np.minimum(q_after - ranges[:, 0], ranges[:, 1] - q_after))
        hand_contacts, hand_force = _b_hand_state(contacts)
        mechanism = _failure_mechanism(
            counts, hand_contacts, hand_force, table_seen, maximum_penetration,
            maximum_translation, maximum_rotation, numeric, cfg25.criteria,
        )
        strict[step] = mechanism is None
        dual[step] = bool(counts[INDEX] and counts[THUMB] and not counts[MIDDLE] and not counts[RING])
        if first_failure_step is None and mechanism is not None:
            first_failure_step, first_failure_mechanism = step, mechanism
        if first_table_step is None and b_table:
            first_table_step = step
        if first_contact_loss_step is None and not (counts[INDEX] and counts[THUMB]):
            first_contact_loss_step = step
        if first_rotation_step is None and rotation[step] > cfg25.criteria.maximum_B_orientation_rad:
            first_rotation_step = step
    def prefix(values: np.ndarray) -> int:
        failed = np.flatnonzero(~values)
        return int(failed[0]) if len(failed) else len(values)
    strict_survival = prefix(strict)
    dual_survival = prefix(dual)
    actual_dims = sorted({int(value) for value in contact_dims[:, [INDEX, THUMB]].ravel() if value})
    if actual_dims and actual_dims != [cfg_cm.variants[variant]]:
        raise RuntimeError(f"{variant} runtime fingertip-B dimensions differ: {actual_dims}")
    _atomic_savez(
        series_path,
        B_per_finger_contact_count=per_finger_counts,
        B_per_finger_contact_dim=contact_dims,
        B_per_finger_normal_force_N=per_finger_normal,
        B_per_finger_tangential_components_N=per_finger_tangential,
        B_per_finger_torsional_component_Nm=per_finger_torsional,
        B_per_finger_rolling_components_Nm=per_finger_rolling,
        B_table_contact=b_table_series,
        B_translation_from_release_m=translation,
        B_rotation_from_release_rad=rotation,
        B_vertical_displacement_m=vertical,
        actuator_saturation=actuator_saturation,
        minimum_joint_margin_rad=minimum_joint_margin,
        strict_state=strict,
        dual_contact_state=dual,
    )
    result_id = stable_trial_id("phase2CM-paired-replay", {"source_trial_id": row["trial_id"], "condition": variant})
    return {
        "trial_id": result_id,
        "condition": variant,
        "condim": cfg_cm.variants[variant],
        "source_trial_id": row["trial_id"],
        "wrist_pose_id": row["wrist_pose_id"],
        "strict_survival_steps": strict_survival,
        "dual_contact_survival_steps": dual_survival,
        "first_failure_step": first_failure_step,
        "first_failure_mechanism": first_failure_mechanism if first_failure_mechanism is not None else "STRICT_SUCCESS",
        "strict_500_success": strict_survival == steps,
        "first_table_step": first_table_step,
        "first_contact_loss_step": first_contact_loss_step,
        "first_rotation_limit_step": first_rotation_step,
        "maximum_B_translation_m": float(np.max(translation)),
        "maximum_B_rotation_rad": float(np.max(rotation)),
        "minimum_vertical_displacement_m": float(np.min(vertical)),
        "maximum_vertical_displacement_m": float(np.max(vertical)),
        "maximum_actuator_saturation_count": int(np.max(np.sum(actuator_saturation, axis=1))),
        "minimum_joint_margin_rad": float(np.min(minimum_joint_margin)),
        "runtime_fingertip_B_dims": actual_dims,
        "steps_completed": steps,
        "timeseries_path": series_path.relative_to(ROOT).as_posix(),
    }


def run_paired_replays(
    cfg_cm: Phase2CMConfig,
    cfg25: Phase25Config,
    base_cfg: ConfigBundle,
    valid_rows: list[dict],
) -> list[dict]:
    root = ROOT / cfg_cm.output_dir / "paired_replay"
    store = IncrementalJsonlStore(root / "candidate_results.jsonl", 30.0, 0.05)
    completed = store.completed_ids()
    state_root = ROOT / cfg_cm.output_dir / "frozen_release_states"
    total = len(valid_rows) * len(VARIANT_ORDER)
    cursor = 0
    for row in valid_rows:
        frozen = _load_state(state_root / _state_filename(row["trial_id"]))
        for variant in VARIANT_ORDER:
            cursor += 1
            result_id = stable_trial_id("phase2CM-paired-replay", {"source_trial_id": row["trial_id"], "condition": variant})
            if result_id in completed:
                continue
            series_path = root / "timeseries" / variant / (_state_filename(row["trial_id"]))
            result = replay_release_state(cfg_cm, cfg25, base_cfg, row, variant, frozen, series_path)
            store.append(result)
            completed.add(result_id)
            if cursor % 10 == 0 or cursor == total:
                print(f"Phase 2CM paired replay: {cursor}/{total}", flush=True)
    records = store.records()
    expected_ids = {
        stable_trial_id("phase2CM-paired-replay", {"source_trial_id": row["trial_id"], "condition": variant})
        for row in valid_rows for variant in VARIANT_ORDER
    }
    actual_ids = {row["trial_id"] for row in records}
    if actual_ids != expected_ids:
        raise RuntimeError(f"paired replay incomplete: expected {len(expected_ids)}, found {len(actual_ids & expected_ids)}")
    if any(int(row["steps_completed"]) != cfg_cm.post_release_steps for row in records):
        raise RuntimeError("paired replay contains an incomplete post-release trajectory")
    return sorted(records, key=lambda row: (row["source_trial_id"], row["condition"]))


def _describe(values: Iterable[float]) -> dict:
    array = np.asarray(list(values), dtype=float)
    return {
        "count": int(len(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "maximum": float(np.max(array)),
    }


def _event_time_description(values: Iterable[int | None], horizon: int) -> dict:
    values = list(values)
    observed = [int(value) for value in values if value is not None]
    return {
        "observed_count": len(observed),
        "censored_count": len(values) - len(observed),
        "horizon_steps": horizon,
        "observed_steps": _describe(observed) if observed else None,
    }


def _trajectory_descriptions(rows: list[dict], field: str) -> dict[str, dict]:
    result = {}
    for horizon in HORIZONS:
        values = []
        for row in rows:
            with np.load(ROOT / row["timeseries_path"], allow_pickle=False) as series:
                values.append(float(series[field][horizon - 1]))
        result[str(horizon)] = {
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "p95": float(np.percentile(values, 95)),
        }
    return result


def _paired_effect(
    left: np.ndarray,
    right: np.ndarray,
    resamples: int,
    seed: int,
) -> dict:
    differences = np.asarray(right, dtype=float) - np.asarray(left, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(differences)
    bootstrap_means = np.empty(resamples, dtype=float)
    bootstrap_medians = np.empty(resamples, dtype=float)
    for start in range(0, resamples, 1000):
        stop = min(start + 1000, resamples)
        indices = rng.integers(0, n, size=(stop - start, n))
        samples = differences[indices]
        bootstrap_means[start:stop] = np.mean(samples, axis=1)
        bootstrap_medians[start:stop] = np.median(samples, axis=1)
    return {
        "N": n,
        "median_paired_difference_steps": float(np.median(differences)),
        "median_paired_difference_percentile_bootstrap_95_CI_steps": [
            float(np.percentile(bootstrap_medians, 2.5)),
            float(np.percentile(bootstrap_medians, 97.5)),
        ],
        "mean_paired_difference_steps": float(np.mean(differences)),
        "mean_paired_difference_percentile_bootstrap_95_CI_steps": [
            float(np.percentile(bootstrap_means, 2.5)),
            float(np.percentile(bootstrap_means, 97.5)),
        ],
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
    }


def _paired_success_table(left_rows: list[dict], right_rows: list[dict]) -> dict | None:
    left = {row["source_trial_id"]: bool(row["strict_500_success"]) for row in left_rows}
    right = {row["source_trial_id"]: bool(row["strict_500_success"]) for row in right_rows}
    if not any(left.values()) and not any(right.values()):
        return None
    table = Counter((left[key], right[key]) for key in sorted(left))
    return {
        "both_fail": table[(False, False)],
        "left_fails_right_succeeds": table[(False, True)],
        "left_succeeds_right_fails": table[(True, False)],
        "both_succeed": table[(True, True)],
    }


def summarize_primary(
    cfg_cm: Phase2CMConfig,
    audit: dict,
    eligible_count: int,
    frozen_count: int,
    valid_count: int,
    records: list[dict],
) -> dict:
    groups = {name: [row for row in records if row["condition"] == name] for name in VARIANT_ORDER}
    if any(len(rows) != valid_count for rows in groups.values()):
        raise RuntimeError("condition counts differ from valid complete triplet count")
    conditions = {}
    for name, rows in groups.items():
        survival = np.asarray([row["strict_survival_steps"] for row in rows], dtype=int)
        raw_failure = Counter(row["first_failure_mechanism"] for row in rows)
        failures = {
            "B_ROTATED_OUT": raw_failure["B_ROTATED_OUT"],
            "B_SLIPPED_TO_TABLE": raw_failure["B_SLIPPED_TO_TABLE"],
            "CONTACT_LOST": raw_failure["CONTACT_LOST"],
            "TABLE_CONTACT": raw_failure["B_SLIPPED_TO_TABLE"],
            "ROTATION_LIMIT": raw_failure["B_ROTATED_OUT"],
            "TRANSLATION_LIMIT": raw_failure["TRANSLATION_LIMIT"],
            "PENETRATION_LIMIT": raw_failure["PENETRATION_LIMIT"],
            "OTHER": raw_failure["OTHER"],
            "STRICT_SUCCESS": raw_failure["STRICT_SUCCESS"],
        }
        conditions[name] = {
            "N": len(rows),
            "strict_500_success_count": int(np.sum(survival == cfg_cm.post_release_steps)),
            "strict_500_success_rate": float(np.mean(survival == cfg_cm.post_release_steps)),
            "strict_survival_steps": _describe(survival),
            "survival_fraction": {str(horizon): float(np.mean(survival >= horizon)) for horizon in HORIZONS},
            "failure_counts": failures,
            "first_failure_mechanism_raw": dict(sorted(raw_failure.items())),
            "time_to_rotation_threshold": _event_time_description((row["first_rotation_limit_step"] for row in rows), cfg_cm.post_release_steps),
            "time_to_table": _event_time_description((row["first_table_step"] for row in rows), cfg_cm.post_release_steps),
            "time_to_contact_loss": _event_time_description((row["first_contact_loss_step"] for row in rows), cfg_cm.post_release_steps),
            "rotation_trajectory_rad": _trajectory_descriptions(rows, "B_rotation_from_release_rad"),
            "translation_trajectory_m": _trajectory_descriptions(rows, "B_translation_from_release_m"),
        }
    ordered_survival = {
        name: np.asarray([
            row["strict_survival_steps"]
            for row in sorted(groups[name], key=lambda row: row["source_trial_id"])
        ], dtype=float)
        for name in VARIANT_ORDER
    }
    paired = {
        "CM4_minus_CM3": _paired_effect(ordered_survival["CM3"], ordered_survival["CM4"], cfg_cm.bootstrap_resamples, cfg_cm.bootstrap_seed),
        "CM6_minus_CM3": _paired_effect(ordered_survival["CM3"], ordered_survival["CM6"], cfg_cm.bootstrap_resamples, cfg_cm.bootstrap_seed + 1),
        "CM6_minus_CM4": _paired_effect(ordered_survival["CM4"], ordered_survival["CM6"], cfg_cm.bootstrap_resamples, cfg_cm.bootstrap_seed + 2),
    }
    for key, left, right in (("CM4_minus_CM3", "CM3", "CM4"), ("CM6_minus_CM3", "CM3", "CM6"), ("CM6_minus_CM4", "CM4", "CM6")):
        paired[key]["strict_500_success_transition_table"] = _paired_success_table(groups[left], groups[right])
    payload = {
        "status": "complete",
        "experiment_id": cfg_cm.experiment_id,
        "compiled_baseline_index_condim": audit["compiled_index_condim"],
        "compiled_baseline_thumb_condim": audit["compiled_thumb_condim"],
        "runtime_index_B_contact_dims": audit["runtime_contact_dims"]["index"],
        "runtime_thumb_B_contact_dims": audit["runtime_contact_dims"]["thumb"],
        "baseline_fingertip_friction": audit["compiled_fingertip_friction"],
        "baseline_B_friction": audit["compiled_B_friction"],
        "eligible_release_state_count": eligible_count,
        "frozen_paired_state_count": frozen_count,
        "valid_complete_triplet_count": valid_count,
        "simulations_completed": len(records),
        "post_release_steps_per_simulation": cfg_cm.post_release_steps,
        "conditions": conditions,
        "paired_effects": paired,
        "materiality": {
            "CM4_materially_improves_retention": None,
            "CM6_materially_improves_beyond_CM4": None,
            "reason": "No PI-supplied materiality threshold was provided; descriptive paired effects are reported without inventing one.",
        },
    }
    path = ROOT / cfg_cm.output_dir / "paired_replay" / "summary.json"
    _write_json(path, payload)
    write_artifact_manifest(cfg_cm)
    return payload


def write_artifact_manifest(cfg_cm: Phase2CMConfig) -> Path:
    output_root = ROOT / cfg_cm.output_dir
    manifest_path = output_root / "artifact_manifest.json"
    files = [path for path in output_root.rglob("*") if path.is_file() and path != manifest_path]
    files.extend(sorted((ROOT / "docs").glob("PHASE2CM_*.md")))
    files.extend([
        ROOT / "configs" / "phase2CM_contact_model_audit.yaml",
        ROOT / "seqgrasp" / "phase2cm_config.py",
        ROOT / "seqgrasp" / "experiments" / "phase2cm.py",
        ROOT / "scripts" / "audit_phase2cm_contact_model.py",
        ROOT / "scripts" / "run_phase2cm_primary.py",
        ROOT / "tests" / "test_phase2cm_contact_model.py",
    ])
    unique = sorted({path.resolve() for path in files if path.exists()}, key=lambda path: path.as_posix())
    payload = {
        "artifact_count_excluding_this_manifest": len(unique),
        "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
        "artifacts": [
            {"path": path.relative_to(ROOT.resolve()).as_posix(), "size_bytes": path.stat().st_size}
            for path in unique
        ],
    }
    _write_json(manifest_path, payload)
    return manifest_path


def run_primary_phase2cm() -> dict:
    cfg_cm, _ = load_phase2cm_config()
    cfg25, _ = load_phase2_5_config()
    base_cfg = load_configs(scene_filename=cfg_cm.source_scene_filename)
    audit = run_contact_audit(cfg_cm)
    _, _, source_rows, metrics, source_hash = phase2h_sources(cfg_cm)
    eligible = eligible_release_trials(source_rows, metrics)
    selected, _ = freeze_trial_ids(cfg_cm, eligible, source_hash)
    # Isolation is established before reconstructing or running any CM outcome.
    verify_model_isolation(base_cfg, selected[0], cfg_cm.variants)
    write_fingertip_geometry_audit(base_cfg, selected[0])
    reconstruct_release_states(cfg_cm, selected, cfg25, base_cfg)
    valid_rows, _ = verify_release_state_equivalence(cfg_cm, selected, base_cfg)
    records = run_paired_replays(cfg_cm, cfg25, base_cfg, valid_rows)
    summary = summarize_primary(cfg_cm, audit, len(eligible), len(selected), len(valid_rows), records)
    print(json.dumps({
        "status": summary["status"],
        "eligible": len(eligible),
        "frozen": len(selected),
        "valid_triplets": len(valid_rows),
        "simulations": len(records),
        "summary": (ROOT / cfg_cm.output_dir / "paired_replay" / "summary.json").relative_to(ROOT).as_posix(),
    }, indent=2), flush=True)
    return summary
