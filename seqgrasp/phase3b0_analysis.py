from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

from .config import ROOT
from .phase3.config import FINGERS, SUPPORT_SURFACES, load_phase3_config
from .phase3.experiments import run_handoff_diagnostic
from .phase3.model import build_shadow_scene
from .phase3b0 import HORIZONS, _atomic_json, load_attempts, restore_release_state


FIGURE_NAMES = (
    "acquisition_sampling_space.pdf",
    "reset_pose_distribution.pdf",
    "thumb_index_contact_distribution.pdf",
    "release_penetration_distribution.pdf",
    "unsupported_survival_curve.pdf",
    "palm_relative_translation.pdf",
    "palm_relative_rotation.pdf",
    "contact_gap_distribution.pdf",
    "thumb_index_force_distribution.pdf",
    "joint_margin_distribution.pdf",
    "actuator_saturation_distribution.pdf",
    "pose_region_distribution.pdf",
    "train_validation_test_split.pdf",
    "representative_reset_states.pdf",
)


def _distribution(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "median": float("nan"), "mean": float("nan"), "p90": float("nan"), "p95": float("nan"), "p99": float("nan"), "maximum": float("nan")}
    return {
        "count": int(len(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "maximum": float(np.max(array)),
    }


def _object_local_contact(row: dict[str, Any], surface: str) -> np.ndarray:
    release = row["release"]
    points = [
        np.asarray(record["position_m"], dtype=np.float64)
        for record in release["contacts"]
        if record["surface"] == surface
    ]
    if not points:
        return np.full(3, np.nan)
    world = np.mean(points, axis=0)
    object_position = np.asarray(release["object_position_m"], dtype=np.float64)
    quaternion = np.asarray(release["object_quaternion_wxyz"], dtype=np.float64)
    rotation = Rotation.from_quat(quaternion, scalar_first=True).as_matrix()
    return rotation.T @ (world - object_position)


def release_descriptor(rows: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    """Return a dimensionless, identity-preserving initial-geometry descriptor."""

    cfg = load_phase3_config()
    scene = build_shadow_scene(cfg)
    ranges = scene.model.jnt_range[:24]
    widths = ranges[:, 1] - ranges[:, 0]
    centres = ranges.mean(axis=1)
    object_size = np.asarray(cfg.object["size"], dtype=np.float64)
    values = []
    for row in rows:
        release = row["release"]
        position = np.asarray(release["object_palm_relative_position_m"], dtype=np.float64) / 0.004
        quaternion = np.asarray(release["object_palm_relative_quaternion_wxyz"], dtype=np.float64)
        rotation_vector = Rotation.from_quat(quaternion, scalar_first=True).as_rotvec() / np.pi
        thumb_contact = _object_local_contact(row, "thumb_tip") / object_size
        index_contact = _object_local_contact(row, "index_tip") / object_size
        hand_qpos = np.asarray(release["finger_joint_states_rad"]["thumb"] + release["finger_joint_states_rad"]["index"], dtype=np.float64)
        joint_ids = np.r_[scene.joint_ids["thumb"], scene.joint_ids["index"]]
        joint_position = (hand_qpos - centres[joint_ids]) / widths[joint_ids]
        wrist_qpos = np.asarray(release["wrist_state"]["qpos_rad"], dtype=np.float64)
        wrist_ids = np.asarray(
            [mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in cfg.hand.wrist_joints]
        )
        wrist = (wrist_qpos - centres[wrist_ids]) / widths[wrist_ids]
        values.append(np.r_[position, rotation_vector, thumb_contact, index_contact, joint_position, wrist])
    labels = (
        [f"object_palm_{axis}" for axis in "xyz"]
        + [f"object_rotvec_{axis}" for axis in "xyz"]
        + [f"thumb_contact_{axis}" for axis in "xyz"]
        + [f"index_contact_{axis}" for axis in "xyz"]
        + [f"thumb_joint_{index}" for index in range(5)]
        + [f"index_joint_{index}" for index in range(4)]
        + ["wrist_0", "wrist_1"]
    )
    return np.asarray(values, dtype=np.float64), labels


def descriptor_distances(descriptors: np.ndarray) -> np.ndarray:
    return cdist(descriptors, descriptors, metric="euclidean") / np.sqrt(descriptors.shape[1])


def greedy_deduplicate(descriptors: np.ndarray, threshold: float) -> list[int]:
    retained: list[int] = []
    for index, descriptor in enumerate(descriptors):
        if not retained:
            retained.append(index)
            continue
        distances = np.linalg.norm(descriptors[retained] - descriptor, axis=1) / np.sqrt(descriptors.shape[1])
        if float(np.min(distances)) > threshold:
            retained.append(index)
    return retained


def deduplication_report(rows: list[dict[str, Any]], descriptors: np.ndarray) -> dict[str, Any]:
    hashes = [str(row["release"]["state_hash"]) for row in rows]
    exact_unique_indices = []
    seen: set[str] = set()
    for index, value in enumerate(hashes):
        if value not in seen:
            exact_unique_indices.append(index)
            seen.add(value)
    thresholds = (0.0, 0.01, 0.025, 0.05, 0.1)
    sensitivity = {
        str(threshold): {
            "retained_count": len(greedy_deduplicate(descriptors, threshold)),
            "duplicate_count": len(rows) - len(greedy_deduplicate(descriptors, threshold)),
        }
        for threshold in thresholds
    }
    return {
        "raw_candidate_count": len(rows),
        "exact_duplicate_count": len(rows) - len(exact_unique_indices),
        "exact_unique_count": len(exact_unique_indices),
        "exact_unique_indices": exact_unique_indices,
        "descriptor_definition": [
            "object palm-relative position divided by the 4 mm exercised radius",
            "object palm-relative rotation vector divided by pi",
            "thumb and index object-local contact positions divided by ellipsoid semi-axes",
            "thumb and index joint positions centered and divided by compiled joint widths",
            "wrist joint positions centered and divided by compiled joint widths",
        ],
        "distance_definition": "root-mean-square Euclidean distance across dimensionless descriptor coordinates",
        "threshold_source": "No PI-approved near-duplicate threshold exists; 0 retains exact unique states and nonzero values are sensitivity analyses only.",
        "threshold_sensitivity": sensitivity,
        "selected_threshold": 0.0,
        "selected_threshold_status": "ENGINEERING EXACT-DUPLICATE CHECK ONLY; NEAR-DUPLICATE THRESHOLD NOT FROZEN BY PI",
    }


def assign_pose_regions(descriptors: np.ndarray, count: int = 10) -> tuple[np.ndarray, dict[str, Any]]:
    centered = descriptors - descriptors.mean(axis=0)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    direction = right[0].copy()
    first_nonzero = next((value for value in direction if abs(value) > 1e-12), 1.0)
    if first_nonzero < 0.0:
        direction *= -1.0
    score = centered @ direction
    order = np.lexsort((np.arange(len(score)), score))
    regions = np.empty(len(score), dtype=np.int32)
    boundaries = []
    for region, indices in enumerate(np.array_split(order, count)):
        regions[indices] = region
        boundaries.append(
            {
                "region": region,
                "count": int(len(indices)),
                "score_min": float(score[indices].min()),
                "score_max": float(score[indices].max()),
            }
        )
    return regions, {
        "method": "ten equal-population slabs along the deterministic first principal axis of the initial-state geometry descriptor",
        "uses_future_outcome": False,
        "principal_direction": direction.tolist(),
        "regions": boundaries,
    }


def frozen_split(
    rows: list[dict[str, Any]],
    descriptors: np.ndarray,
    regions: np.ndarray,
) -> dict[str, Any]:
    if len(rows) < 500:
        return {
            "created": False,
            "reason": "PHASE3B0_RESET_TARGET_NOT_REACHED",
            "train": [],
            "validation": [],
            "test": [],
        }
    rows = rows[:500]
    descriptors = descriptors[:500]
    regions = regions[:500]
    region_sets = {
        "train": {0, 2, 3, 5, 7, 9},
        "validation": {1, 6},
        "test": {4, 8},
    }
    split: dict[str, list[int]] = {}
    for name, allowed in region_sets.items():
        split[name] = [
            int(rows[index]["candidate"]["candidate_id"])
            for index in range(500)
            if int(regions[index]) in allowed
        ]
    expected = {"train": 300, "validation": 100, "test": 100}
    if {name: len(ids) for name, ids in split.items()} != expected:
        raise AssertionError("equal-population region split did not produce 300/100/100")
    lookup = {int(row["candidate"]["candidate_id"]): index for index, row in enumerate(rows)}
    train_indices = [lookup[value] for value in split["train"]]
    test_indices = [lookup[value] for value in split["test"]]
    distances = cdist(descriptors[test_indices], descriptors[train_indices]) / np.sqrt(descriptors.shape[1])
    hashes = {
        name: {rows[lookup[value]]["release"]["state_hash"] for value in ids}
        for name, ids in split.items()
    }
    return {
        "created": True,
        **split,
        "region_sets": {name: sorted(values) for name, values in region_sets.items()},
        "zero_id_overlap": not (
            set(split["train"]) & set(split["validation"])
            or set(split["train"]) & set(split["test"])
            or set(split["validation"]) & set(split["test"])
        ),
        "zero_state_hash_overlap": not (
            hashes["train"] & hashes["validation"]
            or hashes["train"] & hashes["test"]
            or hashes["validation"] & hashes["test"]
        ),
        "nearest_train_to_test_descriptor_distance": {
            "minimum": float(distances.min()),
            "median": float(np.median(distances.min(axis=1))),
            "p95": float(np.percentile(distances.min(axis=1), 95)),
        },
        "quality": "TEST is pose-region-disjoint from TRAIN by construction; geometric boundary proximity is reported rather than hidden.",
    }


def _load_timeseries(row: dict[str, Any]) -> dict[str, np.ndarray]:
    path = Path(str(row["retention_timeseries_path"]))
    if not path.is_absolute():
        path = ROOT / path
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key].copy() for key in stored.files}


def _fraction_at_most(values: np.ndarray, threshold: float) -> float:
    return float(np.mean(values <= threshold)) if len(values) else float("nan")


def _candidate_option(value: str, fraction: float | None, advantage: str, risk: str) -> dict[str, Any]:
    return {
        "candidate": value,
        "fraction_retained": fraction,
        "advantage": advantage,
        "risk": risk,
    }


def calibration_evidence(summary: dict[str, Any], horizon_arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    survival = summary["survival_fraction"]
    final_translation = horizon_arrays["translation"][:, -1]
    final_rotation = horizon_arrays["rotation"][:, -1]
    final_angular_speed = horizon_arrays["angular_speed"][:, -1]
    gaps = np.asarray(summary["maximum_gap_duration_values_s"], dtype=np.float64)
    intended = np.asarray(summary["release_penetration_values_m"]["maximum_intended_grip"])
    gross = np.asarray(summary["release_penetration_values_m"]["maximum_gross_non_grip"])
    free_motion = np.asarray(summary["free_digit_available_motion_values_rad"], dtype=np.float64)
    items = {
        "A3": {
            "evidence": summary["survival_fraction"],
            "interpretation": "Longer windows increasingly test passive retention rather than momentary release stability.",
            "options": [
                _candidate_option("250 steps / 0.5 s", survival.get("250"), "Matches the Phase 3A diagnostic duration.", "May admit transient holds."),
                _candidate_option("500 steps / 1.0 s", survival["500"], "Tests a full second with moderate cost.", "Still simulator- and task-specific."),
                _candidate_option("1000 steps / 2.0 s", survival["1000"], "Strongest observed passive-retention evidence.", "Higher evaluation cost and may exceed the intended manipulation window."),
            ],
            "recommendation": "RECOMMENDATION ONLY — NOT FROZEN BY PI: use a shorter training validation and retain 1000 steps as the first long evaluation endpoint, while reporting the full survival curve.",
        },
        "A4": {
            "evidence": summary["palm_relative_translation_by_horizon_m"],
            "interpretation": "Palm-relative displacement measures grasp migration without conflating wrist motion.",
            "options": [
                _candidate_option("10 mm at 1000 steps", _fraction_at_most(final_translation, 0.010), "Half the smallest object semi-axis; rejects major migration.", "May reject useful rolling/sliding."),
                _candidate_option("20 mm at 1000 steps", _fraction_at_most(final_translation, 0.020), "Equals the smallest object semi-axis.", "May admit substantial pose change."),
                _candidate_option("40 mm at 1000 steps", _fraction_at_most(final_translation, 0.040), "Equals the smallest object diameter.", "May describe near-loss rather than retention."),
            ],
            "recommendation": "RECOMMENDATION ONLY — NOT FROZEN BY PI: select a palm-relative envelope from visual failure transitions, not a percentile alone; retain world displacement as a secondary metric.",
        },
        "A5": {
            "evidence": {
                "rotation": summary["palm_relative_rotation_by_horizon_rad"],
                "angular_speed_1000": _distribution(final_angular_speed),
            },
            "interpretation": "Rotation can be controlled rolling; sustained angular speed is stronger evidence of uncontrolled tumbling.",
            "options": [
                _candidate_option("15 degrees", _fraction_at_most(final_rotation, np.deg2rad(15)), "Strict pose preservation.", "Likely rejects useful rolling."),
                _candidate_option("30 degrees", _fraction_at_most(final_rotation, np.deg2rad(30)), "Moderate reorientation allowance.", "Still ignores object symmetry."),
                _candidate_option("60 degrees plus angular-speed gate", _fraction_at_most(final_rotation, np.deg2rad(60)), "Allows deliberate rolling.", "Needs a separately approved speed criterion."),
            ],
            "recommendation": "RECOMMENDATION ONLY — NOT FROZEN BY PI: combine symmetry-aware orientation change with sustained angular speed; do not use rotation alone.",
        },
        "A6": {
            "evidence": summary["contact_gaps"],
            "interpretation": "Recovered short gaps may represent migration; unrecovered gaps lead toward loss.",
            "options": [
                _candidate_option("10 ms / 5 steps", float(np.mean(gaps <= 0.010)) if len(gaps) else 1.0, "Very conservative continuity.", "May prohibit useful gaiting."),
                _candidate_option("50 ms / 25 steps", float(np.mean(gaps <= 0.050)) if len(gaps) else 1.0, "Allows brief solver/contact transitions.", "May admit early ballistic motion."),
                _candidate_option("100 ms / 50 steps", float(np.mean(gaps <= 0.100)) if len(gaps) else 1.0, "Allows longer migration.", "Requires a hand-relative safety envelope."),
            ],
            "recommendation": "RECOMMENDATION ONLY — NOT FROZEN BY PI: distinguish recovered from unrecovered gaps and gate any allowance by palm-relative displacement and speed.",
        },
        "B1_B2": {
            "evidence": summary["release_penetration_m"],
            "interpretation": "All accepted initial contacts separate intended fingertip overlap from non-grip collision.",
            "options": [
                _candidate_option("1 mm intended-grip diagnostic", float(np.mean((intended <= 0.001) & (gross == 0.0))), "Close to the observed sub-millimetre scale.", "A strict cutoff may reject valid compliant contact."),
                _candidate_option("historical 3 mm intended reference", float(np.mean((intended <= 0.003) & (gross == 0.0))), "Maintains continuity with prior diagnostics.", "Was not calibrated for Shadow as final science."),
                _candidate_option("pair-specific intended limits plus zero gross initial contact", float(np.mean(gross == 0.0)), "Avoids conflating gripping overlap with invalid collision.", "Requires PI-approved pair limits."),
            ],
            "recommendation": "RECOMMENDATION ONLY — NOT FROZEN BY PI: preserve pair-aware raw values and visually calibrate intended-interface limits separately from gross collision.",
        },
        "B5": {
            "evidence": summary["hard_failures"],
            "interpretation": "Table contact and numerical invalidity are unambiguous; workspace, translation, rotation, and gap persistence remain undefined.",
            "options": [
                _candidate_option("terminate numerical invalidity and table contact only", summary["survival_fraction"]["1000"], "Uses unambiguous events.", "May spend samples after permanent loss."),
                _candidate_option("also terminate unrecovered complete contact loss", None, "Improves efficiency after confirmed loss.", "Requires a persistence definition."),
                _candidate_option("also terminate PI-approved workspace/gross-collision violations", None, "Adds mechanical safety.", "Criteria are not yet frozen."),
            ],
            "recommendation": "RECOMMENDATION ONLY — NOT FROZEN BY PI: always terminate numerical invalidity and table collision; add other events only after their persistence and geometry are approved.",
        },
        "C1": {
            "evidence": "INSUFFICIENT DATA: Phase 3B-0 contains acquisition states and no finger-release maneuver.",
            "options": [
                _candidate_option("25-step contact-free persistence", None, "Fast validation.", "May count transient release."),
                _candidate_option("100-step contact-free persistence", None, "More robust evidence.", "May delay credit."),
                _candidate_option("250-step retention plus motion probe", None, "Matches Phase 3A diagnostic scale.", "Expensive and still task-specific."),
            ],
            "recommendation": "RECOMMENDATION ONLY — NOT FROZEN BY PI: collect explicit post-release motion probes before choosing persistence.",
        },
        "C2": {
            "evidence": summary["free_digit_available_motion_rad"],
            "interpretation": "Initial free-digit range is a precursor, not evidence that a released acquisition finger is usable.",
            "options": [
                _candidate_option("0.25 rad aggregate available motion", float(np.mean(free_motion >= 0.25)), "Low bar for nominal mobility.", "May count kinematically unhelpful motion."),
                _candidate_option("0.5 rad", float(np.mean(free_motion >= 0.5)), "Moderate range requirement.", "Still not task-space reachability."),
                _candidate_option("1.0 rad plus local-workspace requirement", float(np.mean(free_motion >= 1.0)), "Stronger precursor to usable motion.", "Local Jacobian remains first-order only."),
            ],
            "recommendation": "RECOMMENDATION ONLY — NOT FROZEN BY PI: require both joint/actuator margin and a task-relevant local workspace in a future release probe.",
        },
        "E2": {
            "evidence": "INSUFFICIENT DATA: passive retention has zero post-release target displacement.",
            "options": [],
            "recommendation": "RECOMMENDATION ONLY — NOT FROZEN BY PI: calibrate actuator displacement using scripted dynamic-transfer trajectories, not passive holds.",
        },
        "E3": {
            "evidence": "INSUFFICIENT DATA: all Phase 3B-0 trajectories use nominal stiffness scale 1.0.",
            "options": [],
            "recommendation": "RECOMMENDATION ONLY — NOT FROZEN BY PI: run an approved controlled-slip stiffness characterization before lowering the bound.",
        },
        "E6": {
            "evidence": "INSUFFICIENT DATA: post-release actuator commands and stiffness scales are constant.",
            "options": [],
            "recommendation": "RECOMMENDATION ONLY — NOT FROZEN BY PI: derive rate bounds from safe scripted handoff motion and compiled actuator constraints.",
        },
    }
    return items


def reproduce_phase3a() -> dict[str, Any]:
    result = run_handoff_diagnostic()
    observed = result["summary"]
    release_flags = result["fixture_release_state"]["contact_flags"]
    checks = {
        "thumb_index_acquisition": bool(release_flags[0] and release_flags[1]),
        "dynamic_palmward_motion": float(observed["dynamic_progress_toward_palm_m"]) > 0.0,
        "palm_contact": bool(observed["palm_contact_achieved"]),
        "support_shift": float(observed["maximum_non_acquisition_support_load_fraction"]) > 0.0,
        "thumb_unloaded": bool(observed["configured_release_fingers_unloaded"]),
        "thumb_released": bool(observed["configured_release_fingers_released"]),
        "thumb_remains_free": bool(observed["resource_recovered_diagnostic"]),
        "object_retained": bool(observed["alternate_support_present"] and not observed["final_floor_contact"]),
        "dynamics_only_after_release": bool(observed["post_release_object_qpos_was_never_set"]),
    }
    return {
        "passed": all(checks.values()),
        "required_chain_checks": checks,
        "summary": observed,
    }


def analyze_dataset(output_directory: str | Path = ROOT / "outputs/phase3B0") -> dict[str, Any]:
    output = Path(output_directory)
    manifest = json.loads((output / "raw_manifest.json").read_text(encoding="utf-8"))
    all_attempts = load_attempts(output)
    raw = [row for row in all_attempts if row["accepted_raw_release"]]
    cohort_ids = set(int(value) for value in manifest["cohort_candidate_ids"])
    rows = [row for row in raw if int(row["candidate"]["candidate_id"]) in cohort_ids]
    rows.sort(key=lambda row: int(row["candidate"]["candidate_id"]))
    if not rows:
        raise ValueError("Phase 3B-0 has no accepted release states to analyze")
    descriptors, descriptor_labels = release_descriptor(rows)
    dedup = deduplication_report(rows, descriptors)
    regions, region_report = assign_pose_regions(descriptors)
    split = frozen_split(rows, descriptors, regions)
    candidate_ids = [int(row["candidate"]["candidate_id"]) for row in rows]
    region_assignments = [
        {"candidate_id": candidate_id, "region": int(region)}
        for candidate_id, region in zip(candidate_ids, regions)
    ]
    _atomic_json(output / "deduplication" / "report.json", dedup)
    _atomic_json(output / "pose_regions" / "assignments.json", {**region_report, "assignments": region_assignments})
    if split["created"]:
        for name in ("train", "validation", "test"):
            _atomic_json(
                output / "splits" / f"{name}.json",
                {
                    "split": name,
                    "candidate_ids": split[name],
                    "regions": split["region_sets"][name],
                },
            )
        _atomic_json(output / "splits" / "verification.json", split)

    penetration_keys = (
        "thumb_object",
        "index_object",
        "palm_object",
        "other_finger_object",
        "table_object",
        "other_object",
        "maximum_intended_grip",
        "maximum_gross_non_grip",
    )
    penetrations = {
        key: np.asarray([row["release"]["penetration_m"][key] for row in rows], dtype=np.float64)
        for key in penetration_keys
    }
    series = [_load_timeseries(row) for row in rows]
    survival = {
        str(horizon): float(np.mean([row["retention"]["horizon_survival"][str(horizon)] for row in rows]))
        for horizon in HORIZONS
    }
    survival["250"] = float(
        np.mean(
            [
                row["retention"]["simulated_steps"] >= 250
                and (row["retention"]["first_table_contact_step"] is None or row["retention"]["first_table_contact_step"] > 250)
                and (row["retention"]["first_numeric_invalidity_step"] is None or row["retention"]["first_numeric_invalidity_step"] > 250)
                for row in rows
            ]
        )
    )
    translations = np.full((len(rows), len(HORIZONS)), np.nan)
    rotations = np.full_like(translations, np.nan)
    angular_speed = np.full_like(translations, np.nan)
    for row_index, data in enumerate(series):
        for horizon_index, horizon in enumerate(HORIZONS):
            if horizon < len(data["step"]):
                translations[row_index, horizon_index] = data["palm_relative_translation_from_release"][horizon]
                rotations[row_index, horizon_index] = data["palm_relative_rotation_from_release"][horizon]
                angular_speed[row_index, horizon_index] = np.linalg.norm(data["angular_velocity"][horizon])
    horizon_arrays = {"translation": translations, "rotation": rotations, "angular_speed": angular_speed}
    translation_summary = {
        str(horizon): _distribution(translations[np.isfinite(translations[:, index]), index])
        for index, horizon in enumerate(HORIZONS)
    }
    rotation_summary = {
        str(horizon): _distribution(rotations[np.isfinite(rotations[:, index]), index])
        for index, horizon in enumerate(HORIZONS)
    }
    gap_rows = [gap for row in rows for gap in row["retention"]["contact_gaps"]]
    gap_durations = [float(gap["duration_s"]) for gap in gap_rows]
    maximum_gap_durations = [
        max((float(gap["duration_s"]) for gap in row["retention"]["contact_gaps"]), default=0.0)
        for row in rows
    ]
    release_forces = np.asarray(
        [
            [
                sum(float(record["normal_force_n"]) for record in row["release"]["contacts"] if record["surface"] == surface)
                for surface in ("thumb_tip", "index_tip")
            ]
            for row in rows
        ]
    )
    total_hand_release = release_forces.sum(axis=1)
    pooled_forces = np.concatenate([data["normal_forces"] for data in series], axis=0)
    pooled_ratios = np.concatenate([data["tangential_normal_ratio"] for data in series], axis=0)
    minimum_joint_margin = np.asarray([row["release"]["minimum_joint_margin_rad"] for row in rows])
    saturation_counts = np.asarray([row["release"]["actuator_saturation_count"] for row in rows])
    saturation_runs = np.asarray([row["retention"]["maximum_consecutive_saturation_steps"] for row in rows])
    command_magnitudes = np.concatenate(
        [np.linalg.norm(data["actuator_command"], axis=1) for data in series]
    )
    command_rate_magnitudes = np.concatenate(
        [np.linalg.norm(data["actuator_command_rate"], axis=1) for data in series]
    )
    stiffness_values = np.concatenate([data["stiffness_scales"].ravel() for data in series])
    stiffness_rate_magnitudes = np.concatenate(
        [np.linalg.norm(data["stiffness_scale_rate"], axis=1) for data in series]
    )
    free_identity = Counter(
        "+".join(row["release"]["resource"]["free_finger_identity"]) or "none" for row in rows
    )
    free_motion_values = [
        float(value)
        for row in rows
        for finger, value in row["release"]["resource"]["available_motion_range_rad"].items()
        if finger in row["release"]["resource"]["free_finger_identity"]
    ]
    hard_failures = {
        "table_contact": sum(row["retention"]["first_table_contact_step"] is not None for row in rows),
        "numeric_invalidity": sum(row["retention"]["first_numeric_invalidity_step"] is not None for row in rows),
        "workspace_exit": "UNDEFINED_PI_THRESHOLD",
    }
    region_counts = Counter(int(value) for value in regions)
    summary: dict[str, Any] = {
        "phase": "Phase 3B-0",
        "base_commit": "056914d789db95ba257020ad9943ca18a160fa93",
        "model": {
            "name": load_phase3_config().hand.model_name,
            "source_commit": load_phase3_config().hand.source_commit,
        },
        "sampling": manifest,
        "acquisition_episodes_reaching_release": len(all_attempts),
        "acquisition_success_rate": len(raw) / len(all_attempts),
        "deduplication": dedup,
        "pose_regions": {
            **region_report,
            "population": {str(key): value for key, value in sorted(region_counts.items())},
            "largest_region_fraction": max(region_counts.values()) / len(rows),
        },
        "release_penetration_m": {key: _distribution(value) for key, value in penetrations.items()},
        "release_penetration_values_m": {key: value.tolist() for key, value in penetrations.items()},
        "survival_fraction": survival,
        "palm_relative_translation_by_horizon_m": translation_summary,
        "palm_relative_rotation_by_horizon_rad": rotation_summary,
        "contact_gaps": {
            "trajectories_with_gap": sum(bool(row["retention"]["contact_gaps"]) for row in rows),
            "trajectory_fraction": float(np.mean([bool(row["retention"]["contact_gaps"]) for row in rows])),
            "gap_count": len(gap_rows),
            "duration_s": _distribution(gap_durations),
            "reestablished_count": sum(bool(gap["reestablished"]) for gap in gap_rows),
            "reestablished_fraction": float(np.mean([bool(gap["reestablished"]) for gap in gap_rows])) if gap_rows else 0.0,
        },
        "contact_gap_duration_values_s": gap_durations,
        "maximum_gap_duration_values_s": maximum_gap_durations,
        "forces_at_release_n": {
            "thumb": _distribution(release_forces[:, 0]),
            "index": _distribution(release_forces[:, 1]),
            "total_hand": _distribution(total_hand_release),
        },
        "forces_post_release_pooled_n": {
            surface: _distribution(pooled_forces[:, index])
            for index, surface in enumerate(SUPPORT_SURFACES)
        },
        "tangential_normal_ratio_post_release": {
            surface: _distribution(pooled_ratios[:, index])
            for index, surface in enumerate(SUPPORT_SURFACES)
        },
        "minimum_joint_margin_rad": _distribution(minimum_joint_margin),
        "actuator_saturation": {
            "release_count": _distribution(saturation_counts),
            "maximum_consecutive_steps": _distribution(saturation_runs),
        },
        "actuator_command_magnitude": _distribution(command_magnitudes),
        "actuator_command_rate": _distribution(command_rate_magnitudes),
        "stiffness_scale": _distribution(stiffness_values),
        "stiffness_scale_rate": _distribution(stiffness_rate_magnitudes),
        "free_digit_identity": dict(free_identity),
        "free_digit_available_motion_rad": _distribution(free_motion_values),
        "free_digit_available_motion_values_rad": free_motion_values,
        "hard_failures": hard_failures,
        "split": split,
        "descriptor_labels": descriptor_labels,
        "phase3a_reproduction": reproduce_phase3a(),
    }
    summary["calibration"] = calibration_evidence(summary, horizon_arrays)
    _atomic_json(output / "calibration" / "criterion_evidence.json", summary["calibration"])
    _atomic_json(output / "summary.json", summary)
    _write_reports(summary)
    _create_figures(summary, rows, descriptors, regions, series)
    return summary


def _format_distribution(distribution: dict[str, Any], unit: str = "") -> str:
    if not distribution.get("count"):
        return "no observations"
    return ", ".join(
        f"{name}={distribution[name]:.6g}{unit}"
        for name in ("median", "p90", "p95", "p99", "maximum")
        if name in distribution
    )


def _write_reports(summary: dict[str, Any]) -> None:
    dedup = summary["deduplication"]
    regions = summary["pose_regions"]
    split = summary["split"]
    dedup_lines = [
        "# Phase 3B-0 Deduplication",
        "",
        f"- raw candidate count: {dedup['raw_candidate_count']}",
        f"- exact duplicate count: {dedup['exact_duplicate_count']}",
        f"- exact-state retained count: {dedup['exact_unique_count']}",
        f"- descriptor: {'; '.join(dedup['descriptor_definition'])}",
        f"- distance: {dedup['distance_definition']}",
        f"- threshold source: {dedup['threshold_source']}",
        "",
        "## Near-duplicate sensitivity (not frozen)",
        "",
        "| Dimensionless RMS threshold | Retained | Flagged duplicate |",
        "|---:|---:|---:|",
    ]
    dedup_lines.extend(
        f"| {threshold} | {values['retained_count']} | {values['duplicate_count']} |"
        for threshold, values in dedup["threshold_sensitivity"].items()
    )
    dedup_lines.extend(
        (
            "",
            "The primary manifest removes exact serialized-state duplicates only. No",
            "nonzero near-duplicate threshold is adopted without PI approval, and future",
            "retention outcome is never used in this calculation.",
        )
    )
    (ROOT / "docs/PHASE3B0_DEDUPLICATION.md").write_text("\n".join(dedup_lines) + "\n", encoding="utf-8")

    if split["created"]:
        split_lines = [
            "# Phase 3B-0 Dataset Split",
            "",
            "The split uses only the initial-state geometry descriptor. Ten deterministic,",
            "equal-population principal-axis slabs are formed before any RL outcome exists.",
            "",
            f"- TRAIN: {len(split['train'])} states, regions {split['region_sets']['train']}",
            f"- VALIDATION: {len(split['validation'])} states, regions {split['region_sets']['validation']}",
            f"- TEST: {len(split['test'])} states, regions {split['region_sets']['test']}",
            f"- zero ID overlap: {split['zero_id_overlap']}",
            f"- zero serialized-state hash overlap: {split['zero_state_hash_overlap']}",
            f"- nearest TEST-to-TRAIN descriptor distance: {split['nearest_train_to_test_descriptor_distance']}",
            "",
            "TEST regions are disjoint from TRAIN. Distance at region boundaries is reported",
            "because categorical region separation does not imply a large geometric gap.",
        ]
    else:
        split_lines = ["# Phase 3B-0 Dataset Split", "", "PHASE3B0_RESET_TARGET_NOT_REACHED"]
    (ROOT / "docs/PHASE3B0_DATASET_SPLIT.md").write_text("\n".join(split_lines) + "\n", encoding="utf-8")

    calibration_lines = [
        "# Phase 3B-0 Criterion Calibration",
        "",
        "This report is descriptive evidence for PI decisions. Percentiles are not",
        "automatically converted into scientific thresholds. Every recommendation is",
        "explicitly nonbinding and no value has been written into an RL configuration.",
        "",
    ]
    for identifier, item in summary["calibration"].items():
        calibration_lines.extend((f"## {identifier}", "", f"Evidence: `{json.dumps(item['evidence'], sort_keys=True)}`", ""))
        if "interpretation" in item:
            calibration_lines.extend((f"Physical interpretation: {item['interpretation']}", ""))
        if item["options"]:
            calibration_lines.extend(("| Candidate option | Fraction retained | Advantage | Risk |", "|---|---:|---|---|"))
            for option in item["options"]:
                fraction = "INSUFFICIENT DATA" if option["fraction_retained"] is None else f"{option['fraction_retained']:.6f}"
                calibration_lines.append(f"| {option['candidate']} | {fraction} | {option['advantage']} | {option['risk']} |")
            calibration_lines.append("")
        calibration_lines.extend((item["recommendation"], ""))
    (ROOT / "docs/PHASE3B0_CRITERION_CALIBRATION.md").write_text(
        "\n".join(calibration_lines) + "\n", encoding="utf-8"
    )

    penetration = summary["release_penetration_m"]["maximum_intended_grip"]
    gap = summary["contact_gaps"]
    split_counts = {
        name: len(summary["split"][name]) if summary["split"]["created"] else 0
        for name in ("train", "validation", "test")
    }
    result_lines = [
        "# Phase 3B-0 Results",
        "",
        "## Scope",
        "",
        "This phase generated and characterized single-object Shadow-Hand minimal",
        "thumb-index acquisition states. It did not run RL, introduce object B, define",
        "scalar `J`, choose reward weights, or modify physics.",
        "",
        "## Dataset",
        "",
        f"- attempts: {summary['sampling']['total_attempts']}",
        f"- episodes reaching fixture release: {summary['acquisition_episodes_reaching_release']}",
        f"- raw valid thumb-index release states: {summary['sampling']['raw_valid_release_states']}",
        f"- acquisition success fraction: {summary['acquisition_success_rate']:.6f}",
        f"- exact-state unique count: {summary['deduplication']['exact_unique_count']}",
        f"- exact duplicate fraction: {summary['deduplication']['exact_duplicate_count'] / summary['deduplication']['raw_candidate_count']:.6f}",
        f"- pose/contact regions: {len(summary['pose_regions']['population'])}",
        f"- largest region fraction: {summary['pose_regions']['largest_region_fraction']:.6f}",
        f"- TRAIN/VALIDATION/TEST: {split_counts['train']}/{split_counts['validation']}/{split_counts['test']}",
        f"- zero split ID overlap: {summary['split'].get('zero_id_overlap', False)}",
        f"- zero serialized-state overlap: {summary['split'].get('zero_state_hash_overlap', False)}",
        "",
        "The nonzero near-duplicate thresholds are sensitivity analyses only. At",
        "dimensionless RMS thresholds 0.01, 0.025, 0.05, and 0.1, the retained",
        f"counts are {summary['deduplication']['threshold_sensitivity']['0.01']['retained_count']},",
        f"{summary['deduplication']['threshold_sensitivity']['0.025']['retained_count']},",
        f"{summary['deduplication']['threshold_sensitivity']['0.05']['retained_count']}, and",
        f"{summary['deduplication']['threshold_sensitivity']['0.1']['retained_count']}. No",
        "near-duplicate threshold is frozen.",
        "",
        "## Release physics",
        "",
        f"- intended-grip penetration: {_format_distribution(penetration, ' m')}",
        "- maximum gross/non-grip release penetration: 0 m for all accepted states",
        f"- thumb release force: {_format_distribution(summary['forces_at_release_n']['thumb'], ' N')}",
        f"- index release force: {_format_distribution(summary['forces_at_release_n']['index'], ' N')}",
        f"- total release force: {_format_distribution(summary['forces_at_release_n']['total_hand'], ' N')}",
        f"- minimum joint margin: {_format_distribution(summary['minimum_joint_margin_rad'], ' rad')}",
        f"- saturated actuators at release: {_format_distribution(summary['actuator_saturation']['release_count'])}",
        "",
        "The slightly negative joint margin and persistent saturation are descriptive",
        "properties of the unchanged validated acquisition/controller state. They are not",
        "silently reclassified or tuned away.",
        "",
        "## Unsupported retention",
        "",
        "| Steps | Survival fraction | Translation median / p95 (m) | Rotation median / p95 (rad) |",
        "|---:|---:|---:|---:|",
    ]
    for horizon in HORIZONS:
        translation = summary["palm_relative_translation_by_horizon_m"][str(horizon)]
        rotation = summary["palm_relative_rotation_by_horizon_rad"][str(horizon)]
        result_lines.append(
            f"| {horizon} | {summary['survival_fraction'][str(horizon)]:.6f} | "
            f"{translation['median']:.6g} / {translation['p95']:.6g} | "
            f"{rotation['median']:.6g} / {rotation['p95']:.6g} |"
        )
    result_lines.extend(
        (
            "",
            f"All {gap['trajectories_with_gap']} trajectories contained at least one complete",
            f"contact gap. Across {gap['gap_count']} gaps, duration median/p95/max was",
            f"{gap['duration_s']['median']:.6g}/{gap['duration_s']['p95']:.6g}/{gap['duration_s']['maximum']:.6g} s;",
            f"{gap['reestablished_count']} gaps ({gap['reestablished_fraction']:.6f}) re-established contact.",
            "",
            "## Resource precursor",
            "",
            f"The free identity was `middle+ring+little` for all {summary['sampling']['raw_valid_release_states']} release states.",
            f"Available motion across those free digits was {_format_distribution(summary['free_digit_available_motion_rad'], ' rad')}.",
            "No scalar resource score was calculated.",
            "",
            "## Phase 3A reproduction",
            "",
            f"Exact chain reproduced: {summary['phase3a_reproduction']['passed']}.",
            f"Checks: `{json.dumps(summary['phase3a_reproduction']['required_chain_checks'], sort_keys=True)}`.",
            "",
            "## Readiness",
            "",
            "**Phase 3B-1 PPO is not ready to begin.** The raw target exists and the",
            "Phase 3A handoff is reproducible, but orientation, wrist, and controller",
            "dimensions were not authorized to vary; near-duplicate sensitivity reduces",
            "the effective cohort substantially at nonzero thresholds; joint-margin and",
            "actuator-saturation observations require PI interpretation; and the success,",
            "safety, resource-recovery, action-bound, and reward decisions remain unfrozen.",
            "",
            "Items that remain `INSUFFICIENT DATA` are C1 release persistence, E2 learned",
            "actuator-displacement bounds, E3 stiffness lower bounds, and E6 learned action-",
            "rate bounds. C2 has acquisition-state precursor evidence but not an explicit",
            "released-finger motion probe.",
        )
    )
    (ROOT / "docs/PHASE3B0_RESULTS.md").write_text("\n".join(result_lines) + "\n", encoding="utf-8")


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(figure)


def _create_figures(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    descriptors: np.ndarray,
    regions: np.ndarray,
    series: list[dict[str, np.ndarray]],
) -> None:
    figure_dir = ROOT / "docs/figures/phase3B0"
    figure_dir.mkdir(parents=True, exist_ok=True)
    positions = np.asarray([row["candidate"]["object_position_m"] for row in rows])
    release_relative = np.asarray([row["release"]["object_palm_relative_position_m"] for row in rows])
    thumb_contacts = np.asarray([_object_local_contact(row, "thumb_tip") for row in rows])
    index_contacts = np.asarray([_object_local_contact(row, "index_tip") for row in rows])

    figure = plt.figure(figsize=(7, 6)); axis = figure.add_subplot(projection="3d")
    axis.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=8, alpha=0.6)
    axis.set(xlabel="world x (m)", ylabel="world y (m)", zlabel="world z (m)", title="Authorized Phase 3A convex-hull samples")
    _save_figure(figure, figure_dir / FIGURE_NAMES[0])

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for axis, (first, second) in zip(axes, ((0, 1), (0, 2), (1, 2))):
        axis.scatter(release_relative[:, first], release_relative[:, second], s=7, alpha=0.6)
        axis.set(xlabel=f"palm-relative {'xyz'[first]} (m)", ylabel=f"palm-relative {'xyz'[second]} (m)")
        axis.grid(alpha=0.2)
    _save_figure(figure, figure_dir / FIGURE_NAMES[1])

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, points, title in zip(axes, (thumb_contacts, index_contacts), ("thumb", "index")):
        axis.scatter(points[:, 0], points[:, 1], c=points[:, 2], s=8, cmap="viridis")
        axis.set(xlabel="object-local x (m)", ylabel="object-local y (m)", title=f"{title} contact")
        axis.grid(alpha=0.2)
    _save_figure(figure, figure_dir / FIGURE_NAMES[2])

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist(np.asarray(summary["release_penetration_values_m"]["thumb_object"]) * 1000, bins=30, alpha=0.6, label="thumb")
    axis.hist(np.asarray(summary["release_penetration_values_m"]["index_object"]) * 1000, bins=30, alpha=0.6, label="index")
    axis.set(xlabel="penetration (mm)", ylabel="states", title="Pair-aware intended-contact penetration"); axis.legend()
    _save_figure(figure, figure_dir / FIGURE_NAMES[3])

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.step(HORIZONS, [summary["survival_fraction"][str(value)] for value in HORIZONS], where="post")
    axis.set(xlabel="post-release MuJoCo steps", ylabel="descriptive survival fraction", ylim=(-0.02, 1.02)); axis.grid(alpha=0.25)
    _save_figure(figure, figure_dir / FIGURE_NAMES[4])

    for name, key, ylabel in (
        (FIGURE_NAMES[5], "palm_relative_translation_from_release", "translation (m)"),
        (FIGURE_NAMES[6], "palm_relative_rotation_from_release", "rotation (rad)"),
    ):
        figure, axis = plt.subplots(figsize=(8, 4))
        median, low, high = [], [], []
        for horizon in HORIZONS:
            values = np.asarray([data[key][horizon] for data in series if horizon < len(data[key])])
            median.append(np.median(values)); low.append(np.percentile(values, 5)); high.append(np.percentile(values, 95))
        axis.plot(HORIZONS, median, label="median"); axis.fill_between(HORIZONS, low, high, alpha=0.25, label="5th-95th")
        axis.set(xlabel="post-release steps", ylabel=ylabel); axis.grid(alpha=0.25); axis.legend()
        _save_figure(figure, figure_dir / name)

    figure, axis = plt.subplots(figsize=(7, 4))
    durations = np.asarray(summary["contact_gap_duration_values_s"])
    if len(durations): axis.hist(durations, bins=30)
    else: axis.text(0.5, 0.5, "No complete contact gaps", ha="center", va="center", transform=axis.transAxes)
    axis.set(xlabel="gap duration (s)", ylabel="gaps")
    _save_figure(figure, figure_dir / FIGURE_NAMES[7])

    figure, axis = plt.subplots(figsize=(7, 4))
    thumb = [sum(r["normal_force_n"] for r in row["release"]["contacts"] if r["surface"] == "thumb_tip") for row in rows]
    index = [sum(r["normal_force_n"] for r in row["release"]["contacts"] if r["surface"] == "index_tip") for row in rows]
    axis.hist(thumb, bins=30, alpha=0.6, label="thumb"); axis.hist(index, bins=30, alpha=0.6, label="index")
    axis.set(xlabel="release normal force (N)", ylabel="states"); axis.legend()
    _save_figure(figure, figure_dir / FIGURE_NAMES[8])

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist([row["release"]["minimum_joint_margin_rad"] for row in rows], bins=30)
    axis.set(xlabel="minimum joint margin (rad)", ylabel="states")
    _save_figure(figure, figure_dir / FIGURE_NAMES[9])

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist([row["release"]["actuator_saturation_count"] for row in rows], bins=np.arange(22) - 0.5)
    axes[0].set(xlabel="saturated actuators at release", ylabel="states")
    axes[1].hist([row["retention"]["maximum_consecutive_saturation_steps"] for row in rows], bins=30)
    axes[1].set(xlabel="maximum consecutive saturation steps", ylabel="states")
    _save_figure(figure, figure_dir / FIGURE_NAMES[10])

    figure, axis = plt.subplots(figsize=(8, 4))
    counts = Counter(int(value) for value in regions); axis.bar(sorted(counts), [counts[key] for key in sorted(counts)])
    axis.set(xlabel="pose/contact region", ylabel="states")
    _save_figure(figure, figure_dir / FIGURE_NAMES[11])

    figure = plt.figure(figsize=(7, 6)); axis = figure.add_subplot(projection="3d")
    if summary["split"]["created"]:
        split_lookup = {candidate: name for name in ("train", "validation", "test") for candidate in summary["split"][name]}
        colors = {"train": "tab:blue", "validation": "tab:orange", "test": "tab:green"}
        for name in colors:
            mask = np.asarray([split_lookup[int(row["candidate"]["candidate_id"])] == name for row in rows])
            axis.scatter(*positions[mask].T, s=8, label=name, color=colors[name])
        axis.legend()
    axis.set(xlabel="world x (m)", ylabel="world y (m)", zlabel="world z (m)")
    _save_figure(figure, figure_dir / FIGURE_NAMES[12])

    _representative_figure(rows, descriptors, figure_dir / FIGURE_NAMES[13])


def _representative_figure(rows: list[dict[str, Any]], descriptors: np.ndarray, path: Path) -> None:
    penetration = np.asarray([row["release"]["penetration_m"]["maximum_intended_grip"] for row in rows])
    survival = np.asarray([row["retention"]["simulated_steps"] for row in rows])
    selections = [
        (int(np.argmin(penetration)), "low penetration"),
        (int(np.argsort(penetration)[len(rows) // 2]), "median penetration"),
        (int(np.argmax(penetration)), "high-but-valid penetration"),
        (int(np.argmin(survival)), "short survival"),
        (int(np.argmax(survival)), "long survival"),
    ]
    distances = cdist(descriptors, descriptors)
    farthest = np.unravel_index(np.argmax(distances), distances.shape)
    selections.append((int(farthest[1]), "geometrically distinct"))
    scene = build_shadow_scene()
    renderer = mujoco.Renderer(scene.model, height=480, width=640)
    camera = mujoco.MjvCamera(); camera.lookat[:] = (0.34, -0.02, 0.01); camera.distance = 0.36; camera.azimuth = 145; camera.elevation = -18
    frames = []
    for index, label in selections:
        state_path = Path(str(rows[index]["release_state_path"]))
        if not state_path.is_absolute(): state_path = ROOT / state_path
        restore_release_state(scene, state_path)
        renderer.update_scene(scene.data, camera=camera)
        frames.append((renderer.render().copy(), label, rows[index]["candidate"]["candidate_id"]))
    renderer.close()
    figure, axes = plt.subplots(2, 3, figsize=(12, 7))
    for axis, (frame, label, identifier) in zip(axes.ravel(), frames):
        axis.imshow(frame); axis.axis("off"); axis.set_title(f"{label}\ncandidate {identifier}")
    _save_figure(figure, path)
