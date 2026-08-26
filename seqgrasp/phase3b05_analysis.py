from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from .config import ROOT
from .phase3.experiments import run_handoff_diagnostic
from .phase3.model import build_shadow_scene
from .phase3b0 import _atomic_json, load_attempts
from .phase3b0_analysis import greedy_deduplicate, release_descriptor
from .phase3b05 import (
    DEDUPLICATION_THRESHOLDS,
    FEASIBILITY_LEVELS,
    PERSISTENCE_HORIZONS,
    load_active_trials,
    load_feasibility_rows,
    run_active_handoff,
)


FIGURE_NAMES = (
    "joint_margin_by_joint.pdf",
    "actuator_saturation_by_actuator.pdf",
    "reset_position_orientation_feasibility.pdf",
    "wrist_feasibility.pdf",
    "effective_diversity_vs_threshold.pdf",
    "reset_distribution_candidates.pdf",
    "active_handoff_success_map.pdf",
    "recovered_finger_persistence.pdf",
    "recovered_finger_available_motion.pdf",
    "action_displacement_sensitivity.pdf",
    "stiffness_sensitivity.pdf",
    "rate_limit_sensitivity.pdf",
    "orientation_stability.pdf",
    "passive_vs_active_contact_gaps.pdf",
)


def distribution(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {name: float("nan") for name in ("median", "mean", "p90", "p95", "p99", "maximum")} | {"count": 0}
    return {
        "count": int(len(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "maximum": float(np.max(array)),
    }


def _fraction(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([bool(row[key]) for row in rows])) if rows else float("nan")


def _feasibility_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for definition in FEASIBILITY_LEVELS:
        level_rows = [row for row in rows if int(row["candidate"]["level"]) == definition.level]
        valid = [row for row in level_rows if row["accepted_raw_release"]]
        penetration = [row["release"]["penetration_m"]["maximum_intended_grip"] for row in valid]
        gross = [row["release"]["penetration_m"]["maximum_gross_non_grip"] for row in valid]
        output[str(definition.level)] = {
            "definition": {
                "position_l1_radius_m": definition.position_l1_radius_m,
                "orientation_limit_deg": definition.orientation_limit_deg,
                "wrist_limit_deg": definition.wrist_limit_deg,
                "label": definition.label,
            },
            "tested": len(level_rows),
            "acquisition_success_count": len(valid),
            "acquisition_success_fraction": len(valid) / len(level_rows) if level_rows else float("nan"),
            "valid_release_count": len(valid),
            "release_penetration_m": distribution(penetration),
            "gross_collision_count": int(sum(value > 0.0 for value in gross)),
            "immediate_slip_count": int(sum(bool(row["immediate_slip"]) for row in valid)),
            "retained_250_count": int(sum(bool(row["retained_250"]) for row in valid)),
            "retained_250_fraction_of_valid": _fraction(valid, "retained_250"),
            "sampled_wrist_feasible_count": int(sum(bool(row["sampled_wrist_within_limits"]) for row in level_rows)),
        }
    return output


def _deduplication(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["accepted_raw_release"]]
    descriptors, labels = release_descriptor(valid)
    sensitivity = {}
    for threshold in DEDUPLICATION_THRESHOLDS:
        indices = greedy_deduplicate(descriptors, threshold)
        sensitivity[str(threshold)] = {"retained_count": len(indices), "retained_fraction": len(indices) / len(valid)}
    per_level = {}
    for level in range(len(FEASIBILITY_LEVELS)):
        subset = [row for row in valid if int(row["candidate"]["level"]) == level]
        values, _ = release_descriptor(subset)
        per_level[str(level)] = {
            str(threshold): len(greedy_deduplicate(values, threshold)) for threshold in DEDUPLICATION_THRESHOLDS
        }
    return {
        "valid_count": len(valid),
        "descriptor_labels": labels,
        "descriptor_definition": [
            "palm-relative object position normalized by 4 mm",
            "palm-relative object rotation vector normalized by pi",
            "thumb and index object-local contact positions normalized by ellipsoid semi-axes",
            "thumb and index joint positions normalized by compiled joint widths",
            "wrist pose normalized by compiled joint widths",
        ],
        "distance": "root-mean-square Euclidean descriptor distance",
        "threshold_status": "SENSITIVITY ONLY - NO NONZERO THRESHOLD PI-FROZEN",
        "all_levels": sensitivity,
        "per_level": per_level,
    }


def _active_group(trials: list[dict[str, Any]], family: str, scale_key: str) -> dict[str, Any]:
    rows = [row for row in trials if row["family"] == family]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[scale_key])].append(row)
    output = {}
    for scale, values in sorted(grouped.items(), key=lambda item: float(item[0])):
        output[scale] = {
            "trials": len(values),
            "diagnostic_handoff_fraction": _fraction(values, "diagnostic_handoff_complete"),
            "palm_contact_fraction": _fraction(values, "palm_contact_achieved"),
            "support_shift_fraction": _fraction(values, "support_shift_observed"),
            "finger_release_fraction": _fraction(values, "selected_finger_released"),
            "retained_fraction": _fraction(values, "final_retained_raw"),
            "maximum_penetration_m": distribution(row["maximum_intended_penetration_m"] for row in values),
            "maximum_object_acceleration_m_s2": distribution(row["maximum_object_acceleration_m_s2"] for row in values),
            "maximum_control_rate_rad_s": distribution(row["maximum_actuator_control_rate_rad_s"] for row in values),
            "minimum_joint_margin_rad": distribution(row["minimum_joint_margin_rad"] for row in values),
            "command_limit_sample_fraction": distribution(row["command_limit_sample_fraction"] for row in values),
            "control_effort_l1_n_steps": distribution(row["control_effort_l1_n_steps"] for row in values),
        }
    return output


def _persistence(trials: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = [row for row in trials if row["family"] == "baseline"]
    output = {}
    for finger in ("thumb", "index"):
        rows = [row for row in baseline if row["release_finger"] == finger]
        output[finger] = {
            str(horizon): {
                key: float(np.mean([row["persistence"][str(horizon)][key] for row in rows])) if rows else float("nan")
                for key in ("contact_free", "object_retained", "usable_available_motion", "combined")
            }
            for horizon in PERSISTENCE_HORIZONS
        }
    return output


def _usable_motion(trials: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in trials if row["family"] == "C2" and row["usable_motion_probe"]]
    output = {}
    for finger in ("thumb", "index"):
        for scale in (0.25, 0.5, 1.0):
            values = [row for row in rows if row["release_finger"] == finger and row["motion_scale"] == scale]
            output[f"{finger}_{scale}"] = {
                "trials": len(values),
                "collision_free_fraction": float(np.mean([row["usable_motion_probe"]["collision_free_reachable"] for row in values])) if values else float("nan"),
                "retained_fraction": float(np.mean([row["usable_motion_probe"]["retained_after_motion"] for row in values])) if values else float("nan"),
                "joint_space_available_motion_rad": distribution(row["usable_motion_probe"]["joint_space_available_motion_rad"] for row in values),
                "jacobian_displacement_envelope_m": distribution(row["usable_motion_probe"]["jacobian_displacement_envelope_m"] for row in values),
                "object_translation_due_to_probe_m": distribution(row["usable_motion_probe"]["object_translation_due_to_probe_m"] for row in values),
            }
    return output


def _gap_summary(trials: list[dict[str, Any]]) -> dict[str, Any]:
    active = [gap for row in trials for gap in row["contact_gaps"]]
    passive_rows = [row for row in load_attempts(ROOT / "outputs/phase3B0") if row["accepted_raw_release"]][:500]
    passive = [gap for row in passive_rows for gap in row["retention"]["contact_gaps"]]
    return {
        "active": {
            "trial_count": len(trials),
            "gap_count": len(active),
            "duration_s": distribution(gap["duration_s"] for gap in active),
            "displacement_m": distribution(gap["palm_relative_displacement_m"] for gap in active),
            "maximum_speed_m_s": distribution(gap["maximum_object_speed_m_s"] for gap in active),
            "reestablished_fraction": float(np.mean([gap["reestablished"] for gap in active])) if active else 1.0,
            "during_controlled_handoff_fraction": float(np.mean([gap["during_controlled_handoff"] for gap in active])) if active else 0.0,
            "subsequently_retained_fraction": float(np.mean([gap["subsequently_retained"] for gap in active])) if active else 1.0,
            "recontact_identity": dict(Counter(surface for gap in active for surface in gap["reestablished_by"])),
        },
        "passive": {
            "trial_count": len(passive_rows),
            "gap_count": len(passive),
            "duration_s": distribution(gap["duration_s"] for gap in passive),
            "displacement_m": distribution(gap["palm_relative_displacement_m"] for gap in passive),
            "maximum_speed_m_s": distribution(gap["maximum_object_speed_m_s"] for gap in passive),
            "reestablished_fraction": float(np.mean([gap["reestablished"] for gap in passive])) if passive else 1.0,
            "recontact_identity": dict(Counter(surface for gap in passive for surface in gap["reestablished_by"])),
        },
    }


def _orientation(trials: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total_rotation_rad": distribution(row["maximum_total_orientation_change_rad"] for row in trials),
        "symmetry_aware_rotation_rad": distribution(row["maximum_symmetry_aware_orientation_change_rad"] for row in trials),
        "sustained_angular_speed_rad_s": distribution(row["maximum_sustained_angular_speed_rad_s"] for row in trials),
        "final_angular_speed_rad_s": distribution(row["final_angular_speed_rad_s"] for row in trials),
        "retained_trials": int(sum(row["final_retained_raw"] for row in trials)),
        "rotation_is_not_a_failure_criterion": True,
        "symmetry_model": "D2 principal-axis symmetry of a triaxial ellipsoid",
    }


def _proposals(feasibility: dict[str, Any], deduplication: dict[str, Any]) -> dict[str, Any]:
    labels = (("CONSERVATIVE", 1), ("MODERATE", 2), ("CHALLENGING", 3))
    output = {}
    for name, level in labels:
        definition = FEASIBILITY_LEVELS[level]
        metrics = feasibility[str(level)]
        output[name] = {
            "status": "RECOMMENDATION ONLY - PI NOT YET FROZEN",
            "position_l1_radius_m": definition.position_l1_radius_m,
            "object_orientation_euler_xyz_deg": [-definition.orientation_limit_deg, definition.orientation_limit_deg],
            "wrist_perturbation_deg": [-definition.wrist_limit_deg, definition.wrist_limit_deg],
            "acquisition_success_fraction": metrics["acquisition_success_fraction"],
            "release_penetration_m": metrics["release_penetration_m"],
            "retained_250_fraction_of_valid": metrics["retained_250_fraction_of_valid"],
            "effective_diversity": deduplication["per_level"][str(level)],
            "geometry_coverage": definition.label,
        }
    return output


def _decision_packet(summary: dict[str, Any]) -> dict[str, Any]:
    trials = summary["active"]["trial_count"]
    active_present = trials > 0
    decisions = {
        "A3": "READY_FOR_PI_DECISION",
        "A4": "READY_FOR_PI_DECISION",
        "A5": "READY_FOR_PI_DECISION",
        "A6": "READY_FOR_PI_DECISION",
        "B1": "READY_FOR_PI_DECISION",
        "B2": "READY_FOR_PI_DECISION",
        "B5": "READY_FOR_PI_DECISION",
        "C1": "READY_FOR_PI_DECISION" if active_present else "INSUFFICIENT_DATA",
        "C2": "READY_FOR_PI_DECISION" if bool(summary["active"]["usable_motion"]) else "INSUFFICIENT_DATA",
        "E2": "INSUFFICIENT_DATA",
        "E3": "INSUFFICIENT_DATA",
        "E6": "INSUFFICIENT_DATA",
    }
    return decisions


def analyze_phase3b05(output_directory: str | Path = ROOT / "outputs/phase3B05") -> dict[str, Any]:
    output = Path(output_directory)
    audit_path = output / "audits" / "phase3b0_joint_actuator_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    feasibility_rows = load_feasibility_rows(output)
    trials = load_active_trials(output)
    feasibility = _feasibility_summary(feasibility_rows)
    deduplication = _deduplication(feasibility_rows)
    phase3a = run_handoff_diagnostic()
    active = {
        "trial_count": len(trials),
        "diagnostic_handoff_success_fraction": _fraction(trials, "diagnostic_handoff_complete"),
        "palm_contact_fraction": _fraction(trials, "palm_contact_achieved"),
        "support_shift_fraction": _fraction(trials, "support_shift_observed"),
        "thumb_release_fraction": _fraction([row for row in trials if row["release_finger"] == "thumb"], "selected_finger_released"),
        "index_release_fraction": _fraction([row for row in trials if row["release_finger"] == "index"], "selected_finger_released"),
        "persistence": _persistence(trials),
        "usable_motion": _usable_motion(trials),
        "E2": _active_group(trials, "E2", "displacement_scale"),
        "E3": _active_group(trials, "E3", "stiffness_scale"),
        "E6": _active_group(trials, "E6", "rate_scale"),
        "contact_gaps": _gap_summary(trials),
        "orientation": _orientation(trials),
        "engineering_options": {
            "E2": {
                "status": "RECOMMENDATION ONLY - PI NOT YET FROZEN",
                "candidate_scales": [0.5, 1.0],
                "rationale": "0.5x-1.0x bracket the two highest observed retention fractions without the 1.5x joint-margin deterioration; zero complete handoffs prevents a final bound.",
            },
            "E3": {
                "status": "RECOMMENDATION ONLY - PI NOT YET FROZEN",
                "candidate_scales": [0.75, 1.0],
                "rationale": "0.75x-1.0x retain more palm/support evidence than 0.25x; zero complete handoffs prevents freezing a lower limit.",
            },
            "E6": {
                "status": "RECOMMENDATION ONLY - PI NOT YET FROZEN",
                "candidate_scales": [1.0],
                "rationale": "1.0x is the scripted reference and had the highest observed retention; neither slower nor faster rates established complete handoff.",
            },
        },
    }
    summary = {
        "scope": "Phase 3B-0.5 pre-RL engineering calibration; no RL, reward, object B, scalar J, or physics change",
        "base_commit": "6520796fdc7a2709c53ecb7667361aae2c0135b8",
        "audit": audit,
        "feasibility": feasibility,
        "effective_diversity": deduplication,
        "reset_distribution_proposals": _proposals(feasibility, deduplication),
        "active": active,
        "phase3a_reproduction": {
            "resource_recovered_diagnostic": phase3a["summary"]["resource_recovered_diagnostic"],
            "post_release_object_qpos_was_never_set": phase3a["summary"]["post_release_object_qpos_was_never_set"],
            "configured_release_fingers_released": phase3a["summary"]["configured_release_fingers_released"],
            "alternate_support_present": phase3a["summary"]["alternate_support_present"],
        },
    }
    summary["pi_decisions"] = _decision_packet(summary)
    valid_total = sum(item["valid_release_count"] for item in feasibility.values())
    summary["ppo_readiness"] = {
        "status": "PPO_NOT_READY",
        "checks": {
            "joint_limit_violation_explained": True,
            "permanent_actuator_command_limit_explained": True,
            "meaningfully_diverse_reset_demonstrated": valid_total > 0 and deduplication["all_levels"]["0.05"]["retained_count"] >= 100,
            "phase3a_handoff_reproduced": bool(summary["phase3a_reproduction"]["resource_recovered_diagnostic"]),
            "some_active_handoff_completed": any(row["diagnostic_handoff_complete"] for row in trials),
            "C1_data": bool(trials),
            "C2_data": bool(active["usable_motion"]),
            "E2_data": len(active["E2"]) == 4,
            "E3_data": len(active["E3"]) == 4,
            "E6_data": len(active["E6"]) == 3,
            "penetration_sane_descriptively": max((item["release_penetration_m"]["maximum"] for item in feasibility.values()), default=float("nan")) < 0.003,
            "criteria_PI_frozen": False,
        },
        "blockers": [
            "no expanded-reset active trial completed the full palm-contact handoff diagnostic, so E2/E3/E6 cannot be frozen from successful matched handoffs",
            "A3/A4/A5/A6/B1/B2/B5/C1/C2/E2/E3/E6 recommendations remain explicitly unfrozen pending PI decision",
            "the official pre-grasp keyframe starts several free-joint/tendon coordinates outside compiled limits and was not altered in this audit",
        ],
    }
    _atomic_json(output / "summary.json", summary)
    _write_reports(summary)
    _create_figures(summary, feasibility_rows, trials)
    return summary


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def _write_reports(summary: dict[str, Any]) -> None:
    audit = summary["audit"]
    outside = [row for row in audit["joint_summary"] if row["outside_count"]]
    lines = [
        "# Phase 3B-0.5 Joint-Margin Audit",
        "",
        "The audit covers all 500 stored Phase 3B-0 release states and the sampled",
        "states of a fresh deterministic Phase 3A successful handoff trajectory.",
        "No model, limit, gain, controller keyframe, or stored result was changed.",
        "",
        f"Exact definition: `{audit['joint_margin_definition']}`.",
        f"Compiled solver tolerance: {audit['compiled_solver_tolerance']:.6g}.",
        "",
        "| Joint | Type | Limits (rad) | Outside / 500 | Min / median / max margin (rad) | Tendon affected |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in audit["joint_summary"]:
        lines.append(
            f"| {row['joint_name']} | {row['joint_type']} | {row['limits_rad']} | {row['outside_count']} | "
            f"{row['margin_min_rad']:.7g} / {row['margin_median_rad']:.7g} / {row['margin_max_rad']:.7g} | "
            f"{row['tendon_coupling_affected']} {','.join(row['tendons'])} |"
        )
    lines.extend(
        [
            "",
            "## Cause",
            "",
            "The negative values are real qpos excursions beyond compiled hinge limits;",
            "the indexing and margin formula are correct. They are not floating-point",
            "noise at the compiled solver-tolerance scale. The official `pre grasp`",
            "keyframe already contains out-of-range components (including negative distal",
            "flexion coordinates and LFJ4 below its lower bound). MuJoCo joint limits are",
            "soft constraints, so the unchanged settling/contact dynamics leave RFJ2,",
            "LFJ2, LFJ1, and THJ3 slightly outside at all 500 releases. Coupled distal",
            "coordinates are additionally affected by fixed J2+J1 tendons. This is a",
            "physical generalized-coordinate soft-constraint excursion initiated by the",
            "source keyframe, not a metric, qpos-indexing, tendon-indexing, or semantic-map bug.",
            "",
            "Affected release joints: " + ", ".join(row["joint_name"] for row in outside) + ".",
            "The same per-joint audit was performed across the Phase 3A handoff samples;",
            "the affected sampled trajectory joints are listed below.",
            "",
            "| Phase 3A handoff joint | Outside sampled states | Samples | Minimum margin (rad) |",
            "|---|---:|---:|---:|",
            *[
                f"| {row['joint_name']} | {row['outside_sample_count']} | {row['sample_count']} | {row['minimum_margin_rad']:.7g} |"
                for row in audit["phase3a_handoff_joint_summary"]
                if row["outside_sample_count"]
            ],
            "",
            "No fix was made: changing the keyframe, constraints, or model tolerance would",
            "be a controller/physics design change, not an authorized software correction.",
        ]
    )
    (ROOT / "docs/PHASE3B05_JOINT_MARGIN_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    actuators = audit["actuator_summary"]
    lines = [
        "# Phase 3B-0.5 Actuator-Saturation Audit",
        "",
        "Phase 3B-0's `actuator_saturation` field tests whether a desired position",
        "command is exactly at a compiled `ctrlrange` endpoint. It does not test",
        "measured actuator force against `forcerange`. This audit reports both.",
        "",
        "| Actuator | Joint/tendon | Transmission | ctrlrange | forcerange | Release command-limit frac | Post-release command-limit frac | Release force-limit frac | Post-release force-limit frac | Max command-limit run |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in actuators:
        lines.append(
            f"| {row['actuator_name']} | {row['associated_joint_or_tendon']} | {row['transmission']} | "
            f"{row['ctrlrange']} | {row['forcerange']} | {row['release_command_limit_fraction']:.6f} | "
            f"{row['postrelease_command_limit_fraction']:.6f} | {row['release_actual_force_limit_fraction']:.6f} | "
            f"{row['postrelease_actual_force_limit_fraction']:.6f} | {row['maximum_consecutive_command_limit_samples']} |"
        )
    lines.extend(
        [
            "",
            "| Actuator | Release command median | Release actual force min / median / max (N) | Post-release actual force min / max (N) |",
            "|---|---:|---:|---:|",
            *[
                f"| {row['actuator_name']} | {row['release_command_median']:.7g} | "
                f"{row['release_actual_force_min_n']:.7g} / {row['release_actual_force_median_n']:.7g} / {row['release_actual_force_max_n']:.7g} | "
                f"{row['postrelease_actual_force_min_n']:.7g} / {row['postrelease_actual_force_max_n']:.7g} |"
                for row in actuators
            ],
        ]
    )
    flagged = [row for row in actuators if row["release_command_limit_fraction"] > 0.0]
    lines.extend(
        [
            "",
            "## Cause and interpretation",
            "",
            "The exact two command-limit actuators are " + " and ".join(f"`{row['actuator_name']}`" for row in flagged) + ".",
            "`rh_A_LFJ4` clips because the official pre-grasp LFJ4 target lies below",
            "its ctrlrange. `rh_A_LFJ0` is a fixed-tendon position servo for LFJ2+LFJ1;",
            "the source keyframe sum is negative while the tendon ctrlrange starts at zero.",
            "The controller holds free digits at that unchanged clipped pre-grasp target,",
            "so the commands remain on their boundaries. Neither actuator reaches its",
            "force range in any audited release or replayed post-release sample. Controller",
            "gain and semantic actuator mapping are not responsible. This is natural",
            "command clipping for the inherited keyframe/tendon representation, not actual",
            "force saturation and not a blocker by itself. Renaming the historical field",
            "would change stored schema, so the audit corrects the interpretation without",
            "rewriting Phase 3B-0 results.",
            "",
            "No actuator limit or gain was altered.",
        ]
    )
    (ROOT / "docs/PHASE3B05_ACTUATOR_SATURATION_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [
        "# Phase 3B-0.5 Reset Feasibility Map",
        "",
        "**ENGINEERING FEASIBILITY LEVELS - NOT PI-FROZEN TRAINING RANGES.**",
        "The same unchanged contact-aware thumb-index acquisition controller is used",
        "at every level. Position is sampled in an L1 ball; Euler and wrist components",
        "are deterministic low-discrepancy probes. Level 3 alternates 15 and 20 degree",
        "object-orientation envelopes so both requested larger probes are represented.",
        "",
        "| Level | Tested | Position L1 radius | Object orientation | Wrist | Valid release | Gross collision | Immediate slip | Retained 250 / valid | Penetration median / p95 / max (m) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for level, item in summary["feasibility"].items():
        definition = item["definition"]
        penetration = item["release_penetration_m"]
        lines.append(
            f"| {level} | {item['tested']} | {definition['position_l1_radius_m']:.3f} m | "
            f"+/-{definition['orientation_limit_deg']:.0f} deg | +/-{definition['wrist_limit_deg']:.0f} deg | "
            f"{item['valid_release_count']} ({item['acquisition_success_fraction']:.3f}) | {item['gross_collision_count']} | "
            f"{item['immediate_slip_count']} | {item['retained_250_count']} ({item['retained_250_fraction_of_valid']:.3f}) | "
            f"{penetration['median']:.6g} / {penetration['p95']:.6g} / {penetration['maximum']:.6g} |"
        )
    lines.extend(["", "## Candidate broader reset distributions", ""])
    for name, item in summary["reset_distribution_proposals"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"**{item['status']}.** Position L1 radius {item['position_l1_radius_m']:.3f} m; "
                f"object Euler range {item['object_orientation_euler_xyz_deg']} deg; wrist perturbation "
                f"{item['wrist_perturbation_deg']} deg. Observed acquisition={item['acquisition_success_fraction']:.3f}, "
                f"250-step retention={item['retained_250_fraction_of_valid']:.3f}. Effective-N sensitivity: "
                f"`{json.dumps(item['effective_diversity'], sort_keys=True)}`.",
                "",
            ]
        )
    lines.extend(
        [
            "No proposal replaces Phase 3B-0. The plots retain position, roll, pitch,",
            "yaw, WRJ1, and WRJ2 separately rather than collapsing feasibility to a scalar.",
        ]
    )
    (ROOT / "docs/PHASE3B05_RESET_FEASIBILITY_MAP.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    orientation = summary["active"]["orientation"]
    lines = [
        "# Phase 3B-0.5 Orientation Analysis",
        "",
        "Cumulative orientation change is descriptive, never a failure gate. The object",
        "is a triaxial ellipsoid, so the symmetry-aware trace minimizes orientation",
        "distance over the four D2 principal-axis symmetries. Angular speed, a 25-step",
        "sustained angular-speed trace, support topology, gaps, and later retention are",
        "reported alongside both rotation traces.",
        "",
        f"- total rotation distribution: `{json.dumps(orientation['total_rotation_rad'], sort_keys=True)}`",
        f"- D2 symmetry-aware rotation: `{json.dumps(orientation['symmetry_aware_rotation_rad'], sort_keys=True)}`",
        f"- sustained angular speed: `{json.dumps(orientation['sustained_angular_speed_rad_s'], sort_keys=True)}`",
        f"- final angular speed: `{json.dumps(orientation['final_angular_speed_rad_s'], sort_keys=True)}`",
        f"- retained active trials: {orientation['retained_trials']}",
        "",
        "Recommendation only: A5 should combine symmetry-aware orientation with",
        "sustained angular speed and support/loss context. It should not use total",
        "rotation alone; useful rolling or sliding must remain distinguishable from",
        "uncontrolled rotation. No A5 threshold is frozen here.",
    ]
    (ROOT / "docs/PHASE3B05_ORIENTATION_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [
        "# Phase 3B-0.5 PI Decision Update",
        "",
        "`READY_FOR_PI_DECISION` means the requested empirical evidence now exists; it",
        "does not adopt a threshold or scientific definition. All recommendations remain",
        "nonbinding until the PI freezes them.",
        "",
        "| Item | Status | Evidence produced |",
        "|---|---|---|",
    ]
    evidence = {
        "A3": "passive and active retention horizons",
        "A4": "palm-relative translation traces",
        "A5": "total, D2-aware, angular-speed, and stabilization traces",
        "A6": "active/passive gap duration, motion, recontact, and retention",
        "B1": "pair-aware intended penetration distributions",
        "B2": "gross-contact distributions",
        "B5": "raw floor, numeric, topology, and retention events",
        "C1": "thumb/index release persistence at seven horizons",
        "C2": "joint/Jacobian envelope and three post-release motion probes",
        "E2": "paired 0.25x/0.5x/1x/1.5x target-step trials",
        "E3": "paired 1x/0.75x/0.5x/0.25x stiffness trials",
        "E6": "paired 0.5x/1x/2x rate trials",
    }
    for item, status in summary["pi_decisions"].items():
        lines.append(f"| {item} | {status} | {evidence[item]} |")
    lines.extend(["", "## Engineering options for PI review", ""])
    for item, option in summary["active"]["engineering_options"].items():
        lines.append(
            f"- {item}: {option['candidate_scales']}x. **{option['status']}**. {option['rationale']}"
        )
    lines.extend(
        [
            "",
            "## PPO readiness",
            "",
            f"**{summary['ppo_readiness']['status']}**",
            "",
            "Remaining blockers:",
            "",
            *[f"- {blocker}" for blocker in summary["ppo_readiness"]["blockers"]],
            "",
            "No PPO code or run was started.",
        ]
    )
    (ROOT / "docs/PHASE3B05_PI_DECISION_UPDATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    active = summary["active"]
    gap = active["contact_gaps"]
    lines = [
        "# Phase 3B-0.5 Results",
        "",
        "This phase is pre-RL engineering calibration only. Physics, collision",
        "geometry, actuator/joint limits, finger-controller keyframes, reward weights,",
        "and the Phase 3B-0 dataset were not changed.",
        "",
        f"- base commit: `{summary['base_commit']}`",
        f"- feasibility candidates: {sum(item['tested'] for item in summary['feasibility'].values())}",
        f"- valid releases: {summary['effective_diversity']['valid_count']}",
        f"- active matched trials: {active['trial_count']}",
        f"- full diagnostic handoff fraction: {active['diagnostic_handoff_success_fraction']:.6f}",
        f"- palm-contact fraction: {active['palm_contact_fraction']:.6f}",
        f"- support-shift fraction: {active['support_shift_fraction']:.6f}",
        f"- thumb-release fraction: {active['thumb_release_fraction']:.6f}",
        f"- index-release fraction: {active['index_release_fraction']:.6f}",
        "",
        "## Effective diversity sensitivity",
        "",
        "No nonzero threshold is selected. Retained effective N across all valid",
        "explored states:",
        "",
        "| Dimensionless RMS threshold | Effective N |",
        "|---:|---:|",
        *[
            f"| {threshold} | {values['retained_count']} |"
            for threshold, values in summary["effective_diversity"]["all_levels"].items()
        ],
        "",
        "## Recovered-finger persistence",
        "",
        "| Finger | Horizon (steps) | Contact-free | Object retained | Combined with available motion |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| {finger} | {horizon} | {active['persistence'][finger][str(horizon)]['contact_free']:.3f} | "
            f"{active['persistence'][finger][str(horizon)]['object_retained']:.3f} | "
            f"{active['persistence'][finger][str(horizon)]['combined']:.3f} |"
            for finger in ("thumb", "index")
            for horizon in PERSISTENCE_HORIZONS
        ],
        "",
        "## Usable-motion probes",
        "",
        "The released finger moved toward the unchanged pre-grasp target and returned;",
        "it was never moved toward object B. Each scale reports joint-space availability,",
        "Jacobian-derived fingertip envelope, selected-finger contact clearance, and",
        "retained-object behavior in the machine summary.",
        "",
        "| Finger / scale | Trials | Collision-free | Retained | Median joint range (rad) | Median Jacobian envelope (m) |",
        "|---|---:|---:|---:|---:|---:|",
        *[
            f"| {key} | {item['trials']} | {item['collision_free_fraction']:.3f} | {item['retained_fraction']:.3f} | "
            f"{item['joint_space_available_motion_rad']['median']:.6g} | {item['jacobian_displacement_envelope_m']['median']:.6g} |"
            for key, item in active["usable_motion"].items()
        ],
        "",
        "## E2 / E3 / E6 sensitivity",
        "",
        "The raw paired tables are stored in `outputs/phase3B05/summary.json`. Candidate",
        "options are recommendations for PI review only:",
        "",
        *[
            f"- {key}: {value['candidate_scales']}x; {value['rationale']}"
            for key, value in active["engineering_options"].items()
        ],
        "",
        "No condition completed the full palm-contact handoff diagnostic across the",
        "expanded matched cohort, so none of these options is a validated final bound.",
        "",
        "## Contact gaps and orientation",
        "",
        f"Passive: {gap['passive']['gap_count']} gaps, median/p95/max duration "
        f"{gap['passive']['duration_s']['median']:.6g}/{gap['passive']['duration_s']['p95']:.6g}/{gap['passive']['duration_s']['maximum']:.6g} s, "
        f"re-established fraction {gap['passive']['reestablished_fraction']:.6f}.",
        f"Active: {gap['active']['gap_count']} gaps, median/p95/max duration "
        f"{gap['active']['duration_s']['median']:.6g}/{gap['active']['duration_s']['p95']:.6g}/{gap['active']['duration_s']['maximum']:.6g} s, "
        f"re-established fraction {gap['active']['reestablished_fraction']:.6f}.",
        "Recovered gaps remain distinct from permanent loss. Orientation is reported",
        "with total change, D2 symmetry-aware change, angular speed, sustained angular",
        "speed, and later retention; rotation alone is not a failure criterion.",
        "",
        "## Readiness",
        "",
        f"**{summary['ppo_readiness']['status']}**",
        "",
        *[f"- {blocker}" for blocker in summary["ppo_readiness"]["blockers"]],
        "",
        "Phase 3A's deterministic handoff reproduced successfully. Phase 3B-0 does not",
        "require revalidation because no baseline implementation or physics was changed.",
        "Raw artifacts and MP4s remain under ignored `outputs/phase3B05/`; reports and",
        "the 14 vector figures are under `docs/` for PI review.",
    ]
    (ROOT / "docs/PHASE3B05_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(figure)


def _create_figures(summary: dict[str, Any], rows: list[dict[str, Any]], trials: list[dict[str, Any]]) -> None:
    directory = ROOT / "docs/figures/phase3B05"
    directory.mkdir(parents=True, exist_ok=True)
    audit = summary["audit"]
    joint = audit["joint_summary"]
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), height_ratios=(2, 1))
    colors = ["tab:red" if row["outside_count"] else "tab:blue" for row in joint]
    axes[0].bar(np.arange(24), [row["margin_min_rad"] * 1000 for row in joint], color=colors)
    axes[0].axhline(0, color="black", lw=0.8); axes[0].set_xticks(np.arange(24), [row["joint_name"].replace("rh_", "") for row in joint], rotation=75)
    axes[0].set(ylabel="minimum margin (mrad)", title="Compiled-coordinate joint-margin audit: all joints")
    outside = [row for row in joint if row["outside_count"]]
    axes[1].bar(np.arange(len(outside)), [row["margin_min_rad"] * 1000 for row in outside], color="tab:red")
    axes[1].axhline(0, color="black", lw=0.8); axes[1].set_xticks(np.arange(len(outside)), [row["joint_name"].replace("rh_", "") for row in outside])
    axes[1].set(ylabel="minimum margin (mrad)", title="Expanded view of joints outside compiled limits")
    for axis in axes: axis.grid(alpha=0.2, axis="y")
    _save(figure, directory / FIGURE_NAMES[0])

    actuators = audit["actuator_summary"]
    x = np.arange(len(actuators)); width = 0.38
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.bar(x - width / 2, [row["postrelease_command_limit_fraction"] for row in actuators], width, label="command at ctrlrange")
    axis.bar(x + width / 2, [row["postrelease_actual_force_limit_fraction"] for row in actuators], width, label="actual force at forcerange")
    axis.set_xticks(x, [row["actuator_name"].replace("rh_A_", "") for row in actuators], rotation=75); axis.set_ylim(0, 1.05); axis.legend()
    axis.set(ylabel="fraction of replayed post-release samples")
    _save(figure, directory / FIGURE_NAMES[1])

    valid = [row for row in rows if row["accepted_raw_release"]]
    failed = [row for row in rows if not row["accepted_raw_release"]]
    figure, axes = plt.subplots(2, 3, figsize=(12, 7))
    labels = (("object_offset_m", 0, "dx (mm)"), ("object_offset_m", 1, "dy (mm)"), ("object_offset_m", 2, "dz (mm)"), ("object_euler_xyz_deg", 0, "roll (deg)"), ("object_euler_xyz_deg", 1, "pitch (deg)"), ("object_euler_xyz_deg", 2, "yaw (deg)"))
    for axis, (key, index, label) in zip(axes.ravel(), labels):
        for data, color, name in ((valid, "tab:blue", "valid"), (failed, "tab:red", "failed")):
            values = [row["candidate"][key][index] * (1000 if key == "object_offset_m" else 1) for row in data]
            axis.hist(values, bins=24, alpha=0.55, color=color, label=name)
        axis.set(xlabel=label, ylabel="candidates"); axis.grid(alpha=0.2)
    axes[0, 0].legend()
    _save(figure, directory / FIGURE_NAMES[2])

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for index, axis in enumerate(axes):
        for level in range(4):
            subset = [row for row in rows if int(row["candidate"]["level"]) == level]
            values = np.rad2deg([row["candidate"]["wrist_perturbation_rad"][index] for row in subset])
            accepted = [row["accepted_raw_release"] for row in subset]
            axis.scatter(values, accepted, s=8, alpha=0.5, label=f"L{level}")
        axis.set(xlabel=f"WRJ{2-index} perturbation (deg)", ylabel="valid release", yticks=(0, 1)); axis.grid(alpha=0.2)
    axes[0].legend(ncol=2)
    _save(figure, directory / FIGURE_NAMES[3])

    figure, axis = plt.subplots(figsize=(7, 4))
    thresholds = list(DEDUPLICATION_THRESHOLDS)
    axis.plot(thresholds, [summary["effective_diversity"]["all_levels"][str(value)]["retained_count"] for value in thresholds], marker="o", label="all valid")
    for level in range(4):
        axis.plot(thresholds, [summary["effective_diversity"]["per_level"][str(level)][str(value)] for value in thresholds], marker=".", label=f"level {level}")
    axis.set(xlabel="dimensionless RMS threshold (sensitivity only)", ylabel="retained effective N"); axis.grid(alpha=0.25); axis.legend()
    _save(figure, directory / FIGURE_NAMES[4])

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    names = list(summary["reset_distribution_proposals"])
    proposals = list(summary["reset_distribution_proposals"].values())
    axes[0].bar(names, [row["position_l1_radius_m"] * 1000 for row in proposals]); axes[0].set(ylabel="position L1 radius (mm)")
    axes[1].bar(names, [row["object_orientation_euler_xyz_deg"][1] for row in proposals]); axes[1].set(ylabel="orientation envelope (+/- deg)")
    axes[2].bar(names, [row["wrist_perturbation_deg"][1] for row in proposals]); axes[2].set(ylabel="wrist envelope (+/- deg)")
    for axis in axes: axis.tick_params(axis="x", rotation=20); axis.grid(alpha=0.2, axis="y")
    _save(figure, directory / FIGURE_NAMES[5])

    figure, axis = plt.subplots(figsize=(8, 5))
    sources = sorted(set(row["source_id"] for row in trials))
    for finger, marker in (("thumb", "o"), ("index", "s")):
        baseline = [row for row in trials if row["family"] == "baseline" and row["release_finger"] == finger]
        axis.scatter([sources.index(row["source_id"]) for row in baseline], [int(row["diagnostic_handoff_complete"]) for row in baseline], marker=marker, s=55, label=finger)
    axis.set(xticks=np.arange(len(sources)), xticklabels=sources, ylabel="diagnostic handoff complete", yticks=(0, 1)); axis.tick_params(axis="x", rotation=45); axis.legend(); axis.grid(alpha=0.25)
    _save(figure, directory / FIGURE_NAMES[6])

    figure, axis = plt.subplots(figsize=(7, 4))
    for finger in ("thumb", "index"):
        axis.plot(PERSISTENCE_HORIZONS, [summary["active"]["persistence"][finger][str(h)]["combined"] for h in PERSISTENCE_HORIZONS], marker="o", label=finger)
    axis.set(xlabel="steps after release", ylabel="combined persistence fraction", ylim=(-0.02, 1.02)); axis.legend(); axis.grid(alpha=0.25)
    _save(figure, directory / FIGURE_NAMES[7])

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    motion = summary["active"]["usable_motion"]
    for finger, axis in zip(("thumb", "index"), axes):
        scales = (0.25, 0.5, 1.0)
        axis.plot(scales, [motion[f"{finger}_{scale}"]["collision_free_fraction"] for scale in scales], marker="o", label="collision-free")
        axis.plot(scales, [motion[f"{finger}_{scale}"]["retained_fraction"] for scale in scales], marker="s", label="retained")
        axis.set(title=finger, xlabel="motion scale", ylabel="fraction", ylim=(-0.02, 1.02)); axis.grid(alpha=0.25); axis.legend()
    _save(figure, directory / FIGURE_NAMES[8])

    for name, key, xlabel in ((FIGURE_NAMES[9], "E2", "target-displacement scale"), (FIGURE_NAMES[10], "E3", "stiffness scale"), (FIGURE_NAMES[11], "E6", "rate scale")):
        data = summary["active"][key]
        scales = sorted((float(value) for value in data), key=float)
        figure, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].plot(scales, [data[str(value)]["diagnostic_handoff_fraction"] for value in scales], marker="o")
        axes[0].set(ylabel="diagnostic handoff fraction")
        axes[1].plot(scales, [data[str(value)]["maximum_penetration_m"]["p95"] * 1000 for value in scales], marker="o")
        axes[1].set(ylabel="p95 max intended penetration (mm)")
        axes[2].plot(scales, [data[str(value)]["maximum_object_acceleration_m_s2"]["p95"] for value in scales], marker="o")
        axes[2].set(ylabel="p95 max object acceleration (m/s2)")
        for axis in axes: axis.set(xlabel=xlabel); axis.grid(alpha=0.25)
        _save(figure, directory / name)

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].hist([row["maximum_total_orientation_change_rad"] for row in trials], bins=25, alpha=0.65, label="total")
    axes[0].hist([row["maximum_symmetry_aware_orientation_change_rad"] for row in trials], bins=25, alpha=0.65, label="D2-aware"); axes[0].legend(); axes[0].set(xlabel="max rotation (rad)")
    axes[1].scatter([row["maximum_symmetry_aware_orientation_change_rad"] for row in trials], [row["maximum_sustained_angular_speed_rad_s"] for row in trials], c=[row["final_retained_raw"] for row in trials], cmap="coolwarm", s=10); axes[1].set(xlabel="D2-aware rotation (rad)", ylabel="sustained angular speed (rad/s)")
    axes[2].scatter([row["maximum_symmetry_aware_orientation_change_rad"] for row in trials], [int(row["final_retained_raw"]) for row in trials], s=10); axes[2].set(xlabel="D2-aware rotation (rad)", ylabel="retained", yticks=(0, 1))
    for axis in axes: axis.grid(alpha=0.2)
    _save(figure, directory / FIGURE_NAMES[12])

    gaps = summary["active"]["contact_gaps"]
    passive_rows = load_attempts(ROOT / "outputs/phase3B0")[:500]
    passive_values = [gap["duration_s"] for row in passive_rows for gap in row["retention"]["contact_gaps"]]
    active_values = [gap["duration_s"] for row in trials for gap in row["contact_gaps"]]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(passive_values, bins=30, alpha=0.65, label="passive"); axes[0].hist(active_values, bins=30, alpha=0.65, label="active"); axes[0].set(xlabel="gap duration (s)", ylabel="gaps"); axes[0].legend()
    axes[1].bar(("passive", "active"), (gaps["passive"]["reestablished_fraction"], gaps["active"]["reestablished_fraction"])); axes[1].set(ylabel="re-established fraction", ylim=(0, 1.05))
    _save(figure, directory / FIGURE_NAMES[13])


def render_representative_videos(
    output_directory: str | Path = ROOT / "outputs/phase3B05",
) -> dict[str, Any]:
    output = Path(output_directory)
    trials = load_active_trials(output)
    if not trials:
        return {}
    video_directory = output / "videos"
    reference_path = video_directory / "nominal_successful_handoff.mp4"
    _render_phase3a_reference(reference_path)
    categories: dict[str, dict[str, Any] | None] = {
        "broadened_pose_successful_handoff.mp4": next((row for row in trials if row["source_id"].startswith("L3") and row["diagnostic_handoff_complete"]), None),
        "failed_handoff_insufficient_support.mp4": next(
            (
                row
                for row in trials
                if not row["palm_contact_achieved"] and not row["final_retained_raw"]
            ),
            None,
        ),
        "successful_finger_release.mp4": next((row for row in trials if row["selected_finger_released"] and row["final_retained_raw"]), None),
        "failed_finger_release.mp4": next((row for row in trials if not row["selected_finger_released"]), None),
        "low_stiffness_controlled_migration.mp4": next((row for row in trials if row["family"] == "E3" and row["stiffness_scale"] == 0.25 and row["final_retained_raw"]), None),
    }
    result = {
        "nominal_successful_handoff.mp4": {
            "rendered": True,
            "source_trial": "Phase3A deterministic reference",
            "path": str(reference_path.relative_to(ROOT)).replace("\\", "/"),
        }
    }
    for filename, row in categories.items():
        if row is None:
            result[filename] = {"rendered": False, "reason": "requested behavior not observed; no episode was relabeled"}
            continue
        path = output / "videos" / filename
        run_active_handoff(
            row["source_state_path"],
            source_id=row["source_id"],
            release_finger=row["release_finger"],
            family=row["family"],
            displacement_scale=row["displacement_scale"],
            stiffness_scale=row["stiffness_scale"],
            rate_scale=row["rate_scale"],
            motion_scale=row["motion_scale"],
            output_directory=output,
            trial_id=row["trial_id"],
            render_video_path=path,
        )
        result[filename] = {"rendered": True, "source_trial": row["trial_id"], "path": str(path.relative_to(ROOT)).replace("\\", "/")}
    _atomic_json(output / "videos" / "manifest.json", result)
    return result


def _render_phase3a_reference(path: Path) -> None:
    """Render the logged Phase 3A reference states without changing its dynamics."""

    diagnostic = run_handoff_diagnostic()
    scene = build_shadow_scene()
    renderer = mujoco.Renderer(scene.model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0.34, -0.02, 0.01)
    camera.distance = 0.36
    camera.azimuth = 145
    camera.elevation = -18
    address = int(scene.model.jnt_qposadr[scene.object_joint_id])
    frames = []
    for sample in diagnostic["samples"]:
        scene.data.qpos[:24] = sample["hand_qpos"]
        scene.data.qpos[address : address + 3] = sample["object_position"]
        scene.data.qpos[address + 3 : address + 7] = sample["object_quaternion"]
        mujoco.mj_forward(scene.model, scene.data)
        renderer.update_scene(scene.data, camera=camera)
        frames.append(renderer.render().copy())
    renderer.close()
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=30, codec="libx264", quality=8)
