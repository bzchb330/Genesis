#!/usr/bin/env python
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import statistics

import matplotlib.pyplot as plt
import numpy as np

from seqgrasp.config import ROOT, load_configs
from seqgrasp.experiments.metadata import config_hash, git_commit_sha
from seqgrasp.experiments.phase2w_diagnostics import palm_space_diagnostics
from seqgrasp.phase2_config import load_phase2_config
from seqgrasp.phase2w_config import load_phase2w_config


STOP_CODE = "PHASE2W_NO_STATIC_WRIST_B_CONTROL"
GROUPS = ("FINGERTIP", "PALMAR_SECURED")


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _summaries(root: Path, predicate) -> list[tuple[int, Path, dict]]:
    result = []
    for path in root.rglob("summary.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if predicate(payload):
            result.append((path.stat().st_mtime_ns, path, payload))
    return result


def _latest(root: Path, predicate) -> tuple[Path, dict]:
    rows = _summaries(root, predicate)
    if not rows:
        raise FileNotFoundError(f"no qualifying summary under {root}")
    _, path, payload = max(rows)
    return path, payload


def _source_records() -> tuple[dict[str, dict], list[Path]]:
    records, paths = {}, []
    for group_dir in ("fingertip_states", "palmar_states"):
        candidates = list((ROOT / "outputs" / "phase2TR" / group_dir).rglob("accepted_states.jsonl"))
        path = max(candidates, key=lambda item: (len(_jsonl(item)), item.stat().st_mtime_ns))
        paths.append(path)
        for row in _jsonl(path):
            records[row["grasp_state_id"]] = row
    return records, paths


def _screen_record(source: dict, screened: dict) -> dict:
    record = dict(source)
    for key in (
        "initial_palm_position_m", "initial_palm_quaternion",
        "initial_object_position_m", "initial_object_quaternion",
        "final_object_position_m", "final_object_quaternion",
        "final_joint_configuration_rad", "occupied_finger_mask", "free_finger_mask",
        "A_translation_drift_m", "A_rotation_drift_rad", "ferrari_canny_epsilon",
        "total_A_normal_force_N", "minimum_joint_margin_rad", "palm_A_contact_fraction",
        "COM_to_palm_surface_distance_m", "object_A_COM_palm_reference_m",
        "mean_per_finger_normal_force_N",
    ):
        if screened.get(key) is not None:
            record[key] = screened[key]
    record["grasp_id"] = screened["source_state_id"]
    return record


def _mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(statistics.fmean(values)) if values else None


def _group_spatial(rows: list[dict], sources: dict[str, dict], phase2, base_cfg) -> dict:
    enriched = []
    for index, screened in enumerate(rows):
        record = _screen_record(sources[screened["source_state_id"]], screened)
        diagnostics = palm_space_diagnostics(record, phase2.resources, base_cfg)
        enriched.append({**screened, **diagnostics})
        if (index + 1) % 10 == 0 or index + 1 == len(rows):
            print(f"Phase 2W palm diagnostics: {index + 1}/{len(rows)}", flush=True)
    force = np.asarray([row["mean_per_finger_normal_force_N"] for row in enriched], dtype=float)
    return {
        "state_count": len(enriched),
        "free_palm_volume_m3_mean": _mean(enriched, "free_palm_volume_m3"),
        "COM_to_palm_surface_distance_m_mean": _mean(enriched, "COM_to_palm_surface_distance_m"),
        "minimum_object_to_palm_boundary_margin_m_mean": _mean(enriched, "minimum_object_to_palm_boundary_margin_m"),
        "occupied_palm_voxel_fraction_mean": _mean(enriched, "occupied_palm_voxel_fraction"),
        "largest_connected_free_palm_component_m3_mean": _mean(enriched, "largest_connected_free_palm_component_m3"),
        "largest_inscribed_free_space_radius_m_mean": _mean(enriched, "largest_inscribed_free_space_radius_m"),
        "palm_contact_fraction_mean": _mean(enriched, "palm_A_contact_fraction"),
        "ferrari_canny_epsilon_mean": _mean(enriched, "ferrari_canny_epsilon"),
        "A_load_distribution_N_mean": np.mean(force, axis=0).tolist(),
    }


def _rpy(row: dict) -> tuple[float, float, float]:
    return tuple(float(value) for value in row["pose"]["relative_rpy_deg"])


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 12, "axes.labelsize": 9,
        "figure.facecolor": "white", "axes.facecolor": "#F7F9FB",
        "axes.edgecolor": "#778899", "axes.grid": True,
        "grid.alpha": 0.2, "pdf.fonttype": 42,
    })


def _wrist_search_pdf(path: Path, coarse_endpoint: dict, refined_endpoint: dict, coarse_geometry: dict, refined_geometry: dict):
    _style()
    fig = plt.figure(figsize=(11, 7.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.1, 0.9])
    for ax, payload, title in (
        (fig.add_subplot(grid[0, 0], projection="3d"), coarse_endpoint, "Coarse static wrist endpoint survival"),
        (fig.add_subplot(grid[0, 1], projection="3d"), refined_endpoint, "Refined static wrist endpoint survival"),
    ):
        xyz = np.asarray([_rpy(row) for row in payload["poses"]])
        values = np.asarray([row["minimum_group_survival"] for row in payload["poses"]])
        scatter = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=values, cmap="viridis", vmin=0, vmax=20, s=28)
        ax.set(xlabel="roll [deg]", ylabel="pitch [deg]", zlabel="yaw [deg]", title=title)
        fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.12, label="min valid / 20")
    ax = fig.add_subplot(grid[1, 0])
    geometry_rows = coarse_geometry["poses"] + refined_geometry["poses"]
    access = [min(row["common_access"][group]["access_fraction"] for group in GROUPS) for row in geometry_rows]
    opposition = [row["collision_free_opposition_count"] for row in geometry_rows]
    survival = [row["endpoint_summary"]["minimum_group_survival"] for row in geometry_rows]
    scatter = ax.scatter(access, opposition, c=survival, cmap="viridis", s=28, alpha=0.8)
    ax.set(xlabel="min common access fraction", ylabel="collision-free opposition candidates / 5000", title="Pre-outcome common B geometry")
    fig.colorbar(scatter, ax=ax, label="min endpoint survival / 20")
    ax = fig.add_subplot(grid[1, 1])
    ax.axis("off")
    lines = [
        "Phase 2W search summary",
        f"Coarse: {coarse_endpoint['orientation_count']} unique poses; {coarse_endpoint['eligible_orientation_count']} endpoint-eligible",
        f"Refined: {refined_endpoint['orientation_count']} unique poses; {refined_endpoint['eligible_orientation_count']} endpoint-eligible",
        f"Geometry mapped: {len(geometry_rows)} poses x 5000 B candidates",
        "All mapped poses had nonzero common opposition access.",
        "Two-contact Ferrari-Canny epsilon remained zero (descriptive).",
        "No A+B outcomes were used in search or ranking.",
    ]
    for index, line in enumerate(lines):
        ax.text(0.02, 0.92 - index * 0.12, line, fontsize=13 if index == 0 else 10, color="#16324F", transform=ax.transAxes)
    fig.suptitle("Phase 2W static wrist feasibility map", fontsize=17, color="#0B1F33")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _geometry_pdf(path: Path, evidence: dict):
    _style()
    top = evidence["highest_ranked_failed_candidate"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.6))
    colors = {"FINGERTIP": "#2F80C1", "PALMAR_SECURED": "#8E5EA2"}
    for index, group in enumerate(GROUPS):
        ax = axes[index]
        ax.set_aspect("equal")
        ax.add_patch(plt.Rectangle((-0.18, -0.10), 0.08, 0.20, color="#8797A5", alpha=0.8))
        ax.add_patch(plt.Rectangle((-0.02, -0.025), 0.05, 0.05, color="#D8B384"))
        ax.scatter([0.10, 0.10], [0.055, -0.055], s=110, color=colors[group], label="middle + ring")
        ax.scatter([0.16, 0.16], [0.085, -0.085], s=90, facecolors="none", edgecolors="#2A9D8F", linewidth=2, label="index + thumb free")
        ax.add_patch(plt.Circle((0.23, 0.0), 0.025, facecolor="#4267B2", alpha=0.75))
        ax.arrow(-0.14, 0.0, 0.06, 0.03, width=0.003, color="#D1495B", length_includes_head=True)
        ax.arrow(-0.14, 0.0, -0.02, -0.07, width=0.003, color="#333333", length_includes_head=True)
        ax.text(-0.08, 0.045, "palm normal", color="#D1495B", fontsize=8)
        ax.text(-0.17, -0.095, "gravity", color="#333333", fontsize=8)
        access = top["common_access"][group]["access_fraction"]
        ax.set(title=f"{group}\nendpoint survival {top['endpoint_summary']['groups'][group]['valid']}/20; access {access:.3f}", xlim=(-0.22, 0.30), ylim=(-0.16, 0.16))
        ax.axis("off")
    ax = axes[2]
    ax.axis("off")
    lines = [
        "Highest-ranked candidate (NOT FROZEN)",
        f"relative RPY: {top['pose']['relative_rpy_deg']} deg",
        f"quaternion wxyz: {[round(v, 6) for v in top['pose']['relative_quaternion_wxyz']]}",
        f"palm normal world: {[round(v, 6) for v in top['endpoint_summary']['nominal_palm_normal_world']]}",
        f"gravity palm: {[round(v, 6) for v in top['endpoint_summary']['nominal_gravity_palm_m_per_s2']]}",
        "hand overlap: 0 in both groups",
        "A overlap: 0 in both groups",
        "B-only strict success: 0/2048",
        "No wrist/B freeze was created.",
    ]
    for index, line in enumerate(lines):
        ax.text(0.02, 0.94 - index * 0.095, line, fontsize=12 if index == 0 else 9.5, color="#16324F", transform=ax.transAxes)
    fig.suptitle("Static wrist geometry diagnosis - gate stopped before freeze", fontsize=16, color="#0B1F33")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _b_only_pdf(path: Path, evidence: dict):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].bar(["Phase 2TR native", "Phase 2W static wrist"], [5 / 12, 0], color=["#2E8B57", "#B5473C"])
    axes[0].set(ylim=(0, 0.5), ylabel="strict success fraction", title="Index + thumb B-only positive-control gate")
    axes[0].text(0, 5 / 12 + 0.025, "5/12", ha="center", fontweight="bold")
    axes[0].text(1, 0.02, "0/8192", ha="center", fontweight="bold", color="#B5473C")
    failure = evidence["B_only_failure_mechanisms"]
    ordered = sorted(failure.items(), key=lambda item: item[1], reverse=True)
    axes[1].barh([name.replace("_", " ") for name, _ in ordered][::-1], [value for _, value in ordered][::-1], color="#D46A4C")
    axes[1].set(xlabel="trial count", title="Phase 2W failure mechanism")
    fig.suptitle("No Phase 2W B-only positive control at a common static wrist region", fontsize=15, color="#0B1F33")
    fig.text(0.5, 0.01, "Geometry anchors established pre-release contact, but no trial met the unchanged 500-step strict hold gate.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _failure_pdf(path: Path, evidence: dict):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    failure = evidence["B_only_failure_mechanisms"]
    ordered = sorted(failure.items(), key=lambda item: item[1], reverse=True)
    axes[0].barh([name.replace("_", " ") for name, _ in ordered][::-1], [value for _, value in ordered][::-1], color="#C85A54")
    axes[0].set(xlabel="B-only trial count", title="Dynamic B-only failures (N=8192)")
    proposal = evidence["proposal_center_diagnostics"]
    centers = list(proposal)
    both = [proposal[name]["both_before_release_fraction"] for name in centers]
    acquired = [proposal[name]["strict_success_fraction"] for name in centers]
    x = np.arange(len(centers))
    axes[1].bar(x - 0.18, both, 0.36, label="both digits before release", color="#2F80C1")
    axes[1].bar(x + 0.18, acquired, 0.36, label="strict success", color="#B5473C")
    axes[1].set_xticks(x, [name.replace("_", "\n") for name in centers])
    axes[1].set(ylim=(0, 1), ylabel="fraction", title="Proposal-center diagnostic")
    axes[1].legend(fontsize=8)
    fig.suptitle("Phase 2W stop diagnosis", fontsize=16, color="#0B1F33")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    cfg_w, cfg_w_path = load_phase2w_config()
    phase2, phase2_path = load_phase2_config()
    base_cfg = load_configs(scene_filename=cfg_w.scene_filename)
    coarse_endpoint_path, coarse_endpoint = _latest(
        ROOT / cfg_w.output_dir / "endpoint_screen" / "coarse",
        lambda payload: payload.get("orientation_count") == 93,
    )
    coarse_geometry_path, coarse_geometry = _latest(
        ROOT / cfg_w.output_dir / "wrist_geometry" / "coarse",
        lambda payload: payload.get("collision_free_opposition_orientation_count") == 18,
    )
    refined_geometry_path, refined_geometry = _latest(
        ROOT / cfg_w.output_dir / "wrist_geometry" / "refined",
        lambda payload: payload.get("collision_free_opposition_orientation_count") == 60,
    )
    b_path, b_summary = _latest(
        ROOT / cfg_w.output_dir / "b_only_dynamic",
        lambda payload: payload.get("status") == STOP_CODE and payload.get("total_B_only_candidates") == 8192,
    )
    top_pose_id = b_summary["poses"][0]["wrist_pose_id"]
    refined_endpoint_path, refined_endpoint = _latest(
        ROOT / cfg_w.output_dir / "endpoint_screen" / "refined",
        lambda payload: payload.get("orientation_count") == 114 and any(row["pose"]["pose_id"] == top_pose_id for row in payload.get("poses", [])),
    )
    refined_screen_rows = _jsonl(refined_endpoint_path.parent / "endpoint_trials.jsonl")
    sources, source_paths = _source_records()
    top_screen = {
        group: [
            row for row in refined_screen_rows
            if row["pose_id"] == top_pose_id and row["group"] == group and row["accepted"]
        ]
        for group in GROUPS
    }
    spatial = {
        group: _group_spatial(top_screen[group], sources, phase2, base_cfg)
        for group in GROUPS
    }
    geometry_rows = coarse_geometry["poses"] + refined_geometry["poses"]
    top_geometry = next(row for row in geometry_rows if row["pose"]["pose_id"] == top_pose_id)
    b_rows = _jsonl(b_path.parent / "candidate_results.jsonl")
    failure = Counter(row["failure_mechanism"] for row in b_rows if not row["strict_index_thumb_success"])
    proposal_diagnostics = {}
    for center in ("native_phase2TR_success", "FINGERTIP_geometry", "PALMAR_SECURED_geometry"):
        rows = [row for row in b_rows if row["trajectory_proposal_center"] == center]
        proposal_diagnostics[center] = {
            "count": len(rows),
            "both_before_release_count": sum(row["both_index_thumb_contact_before_release"] for row in rows),
            "both_before_release_fraction": sum(row["both_index_thumb_contact_before_release"] for row in rows) / len(rows),
            "strict_success_count": sum(row["strict_index_thumb_success"] for row in rows),
            "strict_success_fraction": sum(row["strict_index_thumb_success"] for row in rows) / len(rows),
            "failure_mechanisms": dict(Counter(row["failure_mechanism"] for row in rows if not row["strict_index_thumb_success"])),
        }
    cfg_hash = config_hash([
        cfg_w_path, phase2_path, coarse_endpoint_path, refined_endpoint_path,
        coarse_geometry_path, refined_geometry_path, b_path, *source_paths,
        ROOT / "scripts" / "analyze_phase2w_stop.py",
        ROOT / "seqgrasp" / "experiments" / "phase2w_diagnostics.py",
    ])
    evidence = {
        "status": STOP_CODE,
        "interpretation": "W1",
        "dynamic_wrist_reorientation_simulated": False,
        "coarse_orientation_count": coarse_endpoint["orientation_count"],
        "coarse_endpoint_eligible_count": coarse_endpoint["eligible_orientation_count"],
        "refined_orientation_count": refined_endpoint["orientation_count"],
        "refined_endpoint_eligible_count": refined_endpoint["eligible_orientation_count"],
        "geometry_orientation_count": len(geometry_rows),
        "candidate_B_pose_count": sum(row["candidate_B_pose_count"] for row in geometry_rows),
        "collision_free_opposition_orientation_count": sum(row["collision_free_opposition_count"] > 0 for row in geometry_rows),
        "positive_ferrari_canny_orientation_count": sum(row["positive_ferrari_canny_count"] > 0 for row in geometry_rows),
        "final_wrist_candidate_count": b_summary["selected_wrist_candidate_count"],
        "highest_ranked_failed_candidate": top_geometry,
        "B_only_total_candidates": b_summary["total_B_only_candidates"],
        "B_only_strict_successes_by_pose": {row["wrist_pose_id"]: row["strict_success_count"] for row in b_summary["poses"]},
        "B_only_failure_mechanisms": dict(failure),
        "proposal_center_diagnostics": proposal_diagnostics,
        "B_only_robustness": "not run - no wrist pose reached three strict B-only successes",
        "wrist_B_freeze": "not created - B-only gate failed",
        "full_endpoint_replay": "not run - freeze gate not reached",
        "calibration": "not run - B-only gate failed",
        "controller_freeze": "not created",
        "matching": "not run",
        "formal": "not run",
        "spatial_diagnostics_at_highest_ranked_failed_candidate": spatial,
        "config_hash": cfg_hash,
        "git_commit_sha": git_commit_sha(ROOT),
        "resume_command": "python scripts/analyze_phase2w_stop.py",
    }
    analysis_dir = ROOT / cfg_w.output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    top = top_geometry
    F, P = top["endpoint_summary"]["groups"]["FINGERTIP"], top["endpoint_summary"]["groups"]["PALMAR_SECURED"]
    wf, wp = top["workspace"]["FINGERTIP"], top["workspace"]["PALMAR_SECURED"]
    af, ap = top["common_access"]["FINGERTIP"], top["common_access"]["PALMAR_SECURED"]
    results = f"""# Phase 2W static wrist results

## Outcome

Phase 2W stopped at `{STOP_CODE}`. Static orientation changed the collision geometry enough to create common index+thumb opposition regions, but none of the ten pre-outcome-selected wrist/B candidates produced the required strict B-only positive control in {b_summary['total_B_only_candidates']:,} dynamic candidates. No wrist or B region was frozen, and no A+B calibration or formal comparison was run.

Dynamic wrist reorientation was **not** simulated. Candidate endpoint states were initialized directly at each static rigid root orientation while preserving the finger configuration and object-A pose relative to the palm, then revalidated under unchanged world-fixed gravity. This experiment tests only whether a static post-reorientation endpoint removes the Phase 2T-R geometric blockage.

## Wrist search and endpoint stability

- Coarse grid: 125 Euler combinations deduplicated to {coarse_endpoint['orientation_count']} physical orientations; {coarse_endpoint['eligible_orientation_count']} passed both groups at >=10/20.
- Refined grid around the five pre-outcome-ranked coarse poses: {refined_endpoint['orientation_count']} deduplicated orientations; {refined_endpoint['eligible_orientation_count']} passed.
- Highest-ranked failed candidate (diagnostic, not frozen): relative RPY `{top['pose']['relative_rpy_deg']}` degrees; quaternion wxyz `{top['pose']['relative_quaternion_wxyz']}`.
- Palm normal world: `{top['endpoint_summary']['nominal_palm_normal_world']}`.
- Gravity in palm frame: `{top['endpoint_summary']['nominal_gravity_palm_m_per_s2']}`; palm-normal/gravity angle {top['endpoint_summary']['nominal_palm_normal_gravity_angle_deg']:.3f} degrees.
- At that candidate, FINGERTIP survival was {F['valid']}/{F['screened']} and PALMAR survival was {P['valid']}/{P['screened']}.

## Transformed index/thumb workspace and common B geometry

- FINGERTIP index/thumb volumes: {wf['index_reachable_volume_m3']:.10g} / {wf['thumb_reachable_volume_m3']:.10g} m^3.
- PALMAR index/thumb volumes: {wp['index_reachable_volume_m3']:.10g} / {wp['thumb_reachable_volume_m3']:.10g} m^3.
- Opposition-region volume at the highest-ranked failed candidate: {top['opposition_region_volume_m3']:.10g} m^3.
- Common access fractions: FINGERTIP {af['access_fraction']:.6f}; PALMAR {ap['access_fraction']:.6f}.
- Initial hand/A overlap fractions: FINGERTIP {af['initial_hand_overlap_fraction']:.6f}/{af['initial_A_overlap_fraction']:.6f}; PALMAR {ap['initial_hand_overlap_fraction']:.6f}/{ap['initial_A_overlap_fraction']:.6f}.
- Final geometry mapping evaluated {evidence['candidate_B_pose_count']:,} B poses across {len(geometry_rows)} endpoint-eligible wrist orientations. All had nonzero collision-free opposition candidates; the two-point Ferrari-Canny approximation remained zero and was retained as a limitation rather than misused as the explicit geometric stop condition.

## B-only dynamic control

- Ten wrist/B candidates were selected without A+B outcomes.
- Total candidates: {b_summary['total_B_only_candidates']:,}; strict successes: 0.
- The two highest-ranked candidates were expanded to 2,048 each; the other eight used 512 each.
- Failure mechanisms: `{json.dumps(dict(failure), sort_keys=True)}`.
- Geometry-centered proposals established both index and thumb before release in {proposal_diagnostics['FINGERTIP_geometry']['both_before_release_count']}/{proposal_diagnostics['FINGERTIP_geometry']['count']} F-centered and {proposal_diagnostics['PALMAR_SECURED_geometry']['both_before_release_count']}/{proposal_diagnostics['PALMAR_SECURED_geometry']['count']} P-centered trials, but none survived the unchanged strict 500-step gate.
- Robustness was not run because no wrist pose reached three strict successes.

## Stopped stages

No wrist/B freeze, full-population replay, additional endpoint sampling, calibration split, A+B controller calibration, controller freeze, matching, formal trials, McNemar test, bootstrap, or representative videos were produced. No Phase 2U experiment, scalar J, transfer, dynamic wrist controller, finger gaiting, object C, or RL training was implemented.

## Exploratory palm-space diagnostics

At the highest-ranked failed candidate only (not a formal endpoint comparison):

- FINGERTIP: `{json.dumps(spatial['FINGERTIP'], sort_keys=True)}`
- PALMAR: `{json.dumps(spatial['PALMAR_SECURED'], sort_keys=True)}`

The existing scalar free-palm volume remains nearly identical between groups, while signed COM-to-palm distance and occupied-palm spatial structure differ. These descriptors remain exploratory and were not used to tune wrist/B selection or define J.

## Limitations

Static feasibility does not establish dynamic wrist planning or control. Workspace access used collision-filtered Monte Carlo fingertip samples and a two-point force-closure approximation; dynamic B-only trials remained the decisive positive-control gate. The stopped design provides no PALMAR-versus-FINGERTIP sequential outcome estimate.
"""
    preliminary = f"""# Phase 2W preliminary evidence

Phase 2W evaluated {coarse_endpoint['orientation_count']} coarse and {refined_endpoint['orientation_count']} refined static wrist orientations under unchanged world-fixed gravity. Static wrist configuration removed the systematic initial-overlap blockage observed in Phase 2T-R: the highest-ranked failed candidate had common geometry access {af['access_fraction']:.3f} for FINGERTIP and {ap['access_fraction']:.3f} for PALMAR, with zero initial hand and A overlap in both groups.

However, ten pre-outcome-selected wrist/B candidates produced 0 strict index+thumb B-only successes in {b_summary['total_B_only_candidates']:,} dynamic candidates. Geometry-centered proposals frequently established both contacts before release, but objects then slipped, rotated out, or lost contact. Phase 2W therefore stopped at `{STOP_CODE}` before any freeze or A+B outcome.

This is evidence that wrist configuration changes the collision-free future-acquisition workspace and is a candidate manipulation-resource state descriptor. It does **not** show that wrist control has been solved: no wrist trajectory was simulated, and dynamic wrist planning/control remains proposed work.
"""
    interpretation = f"""# Phase 2W interpretation

## Classification: W1

No tested static wrist orientation produced a common region that was both collision-free and dynamically index+thumb-graspable under the unchanged strict positive-control gate. Geometry-only opposition regions did exist, so the measured obstruction shifted from initial overlap to post-release dynamic retention; nevertheless, no eligible wrist/B configuration existed for freezing.

- W1 is supported by 0 strict successes in {b_summary['total_B_only_candidates']:,} B-only candidates across ten pre-outcome-selected wrist/B configurations.
- W2 is not supported: endpoint populations survived at many coarse and refined orientations.
- W3 is not reached: strict B-only control did not pass, so sequential A+B calibration was not run.
- W4-W7 require formal execution and are not applicable.

The classification does not imply that every possible wrist orientation or controller has been disproved; it applies to the authorized deterministic search and unchanged thresholds.
"""
    wrist_evidence = f"""# Phase 2W wrist-resource evidence

The evidence sequence now separates four resource descriptions:

1. Phase 2T: free-digit count alone was insufficient.
2. Phase 2T-R: digit identity and opposition topology mattered; index+thumb worked in a native B-only region, but the fixed endpoint orientation caused systematic overlap.
3. Phase 2W geometry: changing one static root orientation created collision-free common opposition regions with nonzero access in both endpoint groups.
4. Phase 2W dynamics: geometric access did not imply strict dynamic B retention; the authorized search produced 0/{b_summary['total_B_only_candidates']:,} successes.

Wrist orientation and gravity direction relative to the palm are therefore candidate structured state/resource descriptors. They are not automatically scalar J components, and no weights or normalization are selected here.
"""
    consolidated = f"""# Phase 2R/S/T/T-R/W consolidated evidence

- **Phase 2R:** palmar securing strongly restored digit availability.
- **Phase 2S:** half-scale objects made palmar storage and B access more realistic under the revalidated physics.
- **Phase 2T:** free-digit count alone was insufficient because index+middle produced no strict B-only acquisition.
- **Phase 2T-R:** exact digit identity/opposition topology mattered; index+thumb acquired B in isolation, but the fixed wrist blocked a common endpoint region through systematic hand overlap.
- **Phase 2W:** static wrist orientation removed that initial geometric blockage for many candidates, but 0/{b_summary['total_B_only_candidates']:,} strict B-only dynamic successes meant no wrist/B freeze and no sequential comparison.

Together these phases support a structured, topology- and configuration-aware representation of manipulation resources. They do not define scalar J.
"""
    J_evidence = f"""# J PI decision evidence after Phase 2W

| Candidate descriptor | Units/type | Physical interpretation | Observed Phase 2W variation | Apparent redundancy | Relationship to future acquisition | Sim-to-real observability |
|---|---|---|---|---|---|---|
| Free digit count | integer | Number of digits below the load-bearing threshold | Fixed at 2 | Does not encode identity | Insufficient alone in Phase 2T | Tactile/contact estimation |
| Free digit identity | categorical set | Which digits remain available | Fixed at index+thumb | Not reducible to count | Phase 2T/T-R distinguishes acquisition topology | Tactile/contact estimation |
| Free-digit reachable workspace | m^3 plus spatial set | Collision-filtered index/thumb reach | F index {wf['index_reachable_volume_m3']:.6g}, thumb {wf['thumb_reachable_volume_m3']:.6g}; P index {wp['index_reachable_volume_m3']:.6g}, thumb {wp['thumb_reachable_volume_m3']:.6g} | Scalar volume loses location | Necessary but not sufficient for B-only retention | Kinematics plus scene perception |
| Opposition-capable workspace | m^3 plus spatial set | B centers admitting index/thumb opposition | {top['opposition_region_volume_m3']:.6g} at highest-ranked failed candidate | Related to reachable workspace, but adds topology | Produced collision-free candidates; did not guarantee dynamic retention | Kinematics and object-pose perception |
| Free-palm volume | m^3 | Unoccupied voxels in fixed palm box | F {spatial['FINGERTIP']['free_palm_volume_m3_mean']:.6g}; P {spatial['PALMAR_SECURED']['free_palm_volume_m3_mean']:.6g} | Nearly redundant under equal topology | Did not distinguish the groups strongly here | Hand/object pose reconstruction |
| Object COM relative to palm | m, signed vector/distance | Location of retained A relative to palm | Signed surface means F {spatial['FINGERTIP']['COM_to_palm_surface_distance_m_mean']:.6g}; P {spatial['PALMAR_SECURED']['COM_to_palm_surface_distance_m_mean']:.6g} | Not captured by scalar free-palm volume | Describes endpoint topology; future acquisition relation not identified here | Object pose and hand pose |
| Palm contact | binary/fraction/force | Whether and how persistently A loads the palm | F {spatial['FINGERTIP']['palm_contact_fraction_mean']:.3f}; P {spatial['PALMAR_SECURED']['palm_contact_fraction_mean']:.3f} | Related to COM/palm distance | Defines endpoint class; no Phase 2W formal outcome relation | Tactile array |
| Wrist orientation | normalized quaternion or SO(3) element | Rigid hand mount orientation in world | 93 coarse and 114 refined orientations tested | Not reducible to palm contact or digit identity | Changed overlap and common geometric access | Robot encoders/kinematics |
| Gravity relative to palm | m/s^2 vector | Load direction in the hand frame | Highest-ranked failed candidate `{top['endpoint_summary']['nominal_gravity_palm_m_per_s2']}` | Determined by wrist orientation when world gravity is fixed | Affected endpoint survival; acquisition relationship not formally identified | IMU plus kinematics |
| Ferrari-Canny epsilon | dimensionless under current normalization | Geometric force-closure margin | Two-contact mapped approximation remained 0; A endpoint epsilon varied | Partly related to contact topology | Did not identify dynamic B success in Phase 2W | Contact geometry/force estimation |

No descriptor is normalized, weighted, or collapsed into an arbitrary scalar. Categorical digit topology remains categorical.

**TODO(PI): define whether future resource representation should be scalar, vector, structured/topological, or task-conditioned.**
"""
    for path, text in (
        (ROOT / "docs" / "PHASE2W_STATIC_WRIST_RESULTS.md", results),
        (ROOT / "docs" / "PHASE2W_PRELIMINARY_EVIDENCE.md", preliminary),
        (ROOT / "docs" / "PHASE2W_INTERPRETATION.md", interpretation),
        (ROOT / "docs" / "PHASE2W_WRIST_RESOURCE_EVIDENCE.md", wrist_evidence),
        (ROOT / "docs" / "PHASE2RSTW_CONSOLIDATED_EVIDENCE.md", consolidated),
        (ROOT / "docs" / "J_PI_DECISION_EVIDENCE_PHASE2W.md", J_evidence),
    ):
        path.write_text(text, encoding="utf-8")
    figures = ROOT / "docs" / "figures" / "phase2W"
    figures.mkdir(parents=True, exist_ok=True)
    _wrist_search_pdf(figures / "wrist_pose_search_map.pdf", coarse_endpoint, refined_endpoint, coarse_geometry, refined_geometry)
    _geometry_pdf(figures / "frozen_wrist_geometry.pdf", evidence)
    _b_only_pdf(figures / "index_thumb_b_only_positive_control.pdf", evidence)
    _failure_pdf(figures / "failure_modes.pdf", evidence)
    print(json.dumps({
        "status": STOP_CODE,
        "analysis": str(analysis_dir / "evidence.json"),
        "reports": 6,
        "figures": 4,
        "formal_main_result_generated": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
