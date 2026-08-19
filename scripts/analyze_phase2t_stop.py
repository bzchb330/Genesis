#!/usr/bin/env python
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mujoco
import numpy as np

from seqgrasp.config import ROOT, load_configs
from seqgrasp.experiments.metadata import config_hash, git_commit_sha
from seqgrasp.experiments.resource_components import FINGER_ORDER, compute_resource_components, reconstruct_grasp
from seqgrasp.experiments.resumable import IncrementalJsonlStore, stable_trial_id
from seqgrasp.phase2_config import load_phase2_config
from seqgrasp.phase2t_config import load_phase2t_config


STOP_CODE = "PHASE2T_NO_PAIR_SPECIFIC_B_CONTROL"
GROUPS = ("FINGERTIP", "PALMAR_SECURED")
COLORS = {"FINGERTIP": "#0072B2", "PALMAR_SECURED": "#D55E00"}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _latest(root: Path, filename: str) -> Path:
    candidates = [(path.stat().st_mtime_ns, path) for path in root.rglob(filename)]
    if not candidates:
        raise FileNotFoundError(f"{filename} under {root}")
    return max(candidates)[1]


def _largest(root: Path, filename: str) -> Path:
    candidates = [(len(_jsonl(path)), path.stat().st_mtime_ns, path) for path in root.rglob(filename)]
    if not candidates:
        raise FileNotFoundError(f"{filename} under {root}")
    return max(candidates)[2]


def _describe(values) -> dict:
    array = np.asarray(list(values), dtype=float)
    return {
        "count": len(array), "minimum": float(np.min(array)), "maximum": float(np.max(array)),
        "mean": float(np.mean(array)), "standard_deviation": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
    }


def _effect(fingertip, palmar) -> dict:
    left, right = np.asarray(fingertip, dtype=float), np.asarray(palmar, dtype=float)
    pooled = math.sqrt((np.var(left, ddof=1) + np.var(right, ddof=1)) / 2.0)
    return {
        "FINGERTIP": _describe(left), "PALMAR_SECURED": _describe(right),
        "mean_difference_palmar_minus_fingertip": float(np.mean(right) - np.mean(left)),
        "standardized_mean_difference": None if pooled == 0 else float((np.mean(right) - np.mean(left)) / pooled),
    }


def _resource_task(payload):
    row, resources, seed, base_cfg, cfg_hash, commit = payload
    enriched = dict(row)
    enriched["grasp_id"] = row["grasp_state_id"]
    enriched.setdefault("mean_per_finger_normal_force_N", row["per_finger_A_normal_force_N"])
    result = compute_resource_components(enriched, resources, seed, base_cfg)
    return {
        "trial_id": stable_trial_id("phase2T-resource-state", row["grasp_state_id"]),
        "grasp_state_id": row["grasp_state_id"], **asdict(result),
        "free_finger_count": 4 - result.occupied_finger_count,
        "config_hash": cfg_hash, "git_commit_sha": commit,
    }


def _render(row: dict, scene_cfg, azimuth=140, elevation=-22):
    _, model, data, _ = reconstruct_grasp(row, scene_cfg)
    renderer = mujoco.Renderer(model, height=420, width=520)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object_a")
    camera.lookat[:] = data.xpos[object_id]
    camera.distance, camera.azimuth, camera.elevation = 0.26, azimuth, elevation
    renderer.update_scene(data, camera=camera)
    image = renderer.render().copy()
    renderer.close()
    return image


def _state_label(row):
    occupied = "+".join(f for f, flag in zip(FINGER_ORDER, row["occupied_finger_mask"]) if flag)
    free = "+".join(row["free_finger_set"])
    return f"occupied: {occupied} | free: {free}"


def _plot_outputs(states, results, figures):
    plt.style.use(ROOT / "configs" / "phase2_publication.mplstyle")
    figures.mkdir(parents=True, exist_ok=True)
    proposal = ROOT / "docs" / "figures" / "proposal"
    proposal.mkdir(parents=True, exist_ok=True)
    fingertip = [row for row in states if row["grasp_state_type"] == "FINGERTIP"]
    palmar = [row for row in states if row["grasp_state_type"] == "PALMAR_SECURED"]
    selected_f = sorted(fingertip, key=lambda row: row["grasp_state_id"])[::max(1, len(fingertip) // 6)][:6]

    with PdfPages(figures / "eligible_fingertip_examples.pdf") as pdf:
        fig, axes = plt.subplots(2, 3, figsize=(11, 7.2))
        for ax, row in zip(axes.flat, selected_f):
            ax.imshow(_render(row, load_configs(scene_filename="scene_two_object_half_scale.yaml")))
            ax.set(xticks=[], yticks=[])
            ax.set_title(row["grasp_state_id"].replace("phase2T_", ""), fontsize=8)
            ax.text(.02, .02, _state_label(row), transform=ax.transAxes, fontsize=6, color="white", bbox={"facecolor": "black", "alpha": .6})
        fig.suptitle("Phase 2T strict FINGERTIP_ELIGIBLE_2FREE examples")
        fig.tight_layout(rect=(0, 0, 1, .95)); pdf.savefig(fig); plt.close(fig)

    covariates = results["endpoint_covariate_distributions"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.8))
    for ax, key in zip(axes.flat[:5], (
        "ferrari_canny_epsilon", "total_A_normal_force_N", "A_translation_drift_m",
        "A_rotation_drift_rad", "minimum_joint_margin_rad",
    )):
        ax.boxplot([[row[key] for row in fingertip], [row[key] for row in palmar]], tick_labels=("Tip", "Palm"))
        ax.set_title(key.replace("_", " "), fontsize=8)
    axes.flat[5].axis("off")
    axes.flat[5].text(.5, .58, "FORMAL MATCHING NOT PERFORMED", ha="center", weight="bold", color="#B22222")
    axes.flat[5].text(.5, .36, f"Stop gate: {STOP_CODE}\nPair-specific B-only: 0 / 4096", ha="center")
    fig.suptitle("Phase 2T candidate endpoint balance before the mandatory B-control gate")
    fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(figures / "matched_endpoint_states.pdf"); plt.close(fig)

    representative_f = sorted(fingertip, key=lambda row: row["COM_to_palm_origin_distance_m"])[len(fingertip) // 2]
    representative_p = sorted(palmar, key=lambda row: row["COM_to_palm_origin_distance_m"])[len(palmar) // 2]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.7))
    scene_cfg = load_configs(scene_filename="scene_two_object_half_scale.yaml")
    for ax, row, title in ((axes[0, 0], representative_f, "A. Eligible fingertip"), (axes[0, 1], representative_p, "B. Two-free palmar")):
        ax.imshow(_render(row, scene_cfg)); ax.set(xticks=[], yticks=[]); ax.set_title(title, weight="bold")
        ax.text(.02, .02, _state_label(row), transform=ax.transAxes, fontsize=6, color="white", bbox={"facecolor": "black", "alpha": .6})
    axes[0, 2].bar((0, 1), (2, 2), color=(COLORS["FINGERTIP"], COLORS["PALMAR_SECURED"]))
    axes[0, 2].set(title="C. Exact free-digit count", xticks=(0, 1), xticklabels=("Tip", "Palm"), ylim=(0, 4), ylabel="free fingers")
    axes[0, 3].text(.5, .65, "Identical free set", ha="center", weight="bold", fontsize=12)
    axes[0, 3].text(.5, .42, "index + middle", ha="center", fontsize=12)
    axes[0, 3].axis("off")
    for ax, key, title in (
        (axes[1, 0], "COM_to_palm_origin_distance_m", "D. COM-to-palm distance"),
        (axes[1, 1], "free_finger_workspace_vol_m3", "E. Free-finger workspace"),
        (axes[1, 2], "free_palm_volume_m3", "F. Free-palm volume"),
    ):
        values = [np.mean([row[key] for row in group]) for group in (fingertip, palmar)]
        ax.bar((0, 1), values, color=(COLORS["FINGERTIP"], COLORS["PALMAR_SECURED"]))
        ax.set(title=title, xticks=(0, 1), xticklabels=("Tip", "Palm"))
    axes[1, 3].bar((0, 1), (4096, 0), color=("#999999", "#B22222"))
    axes[1, 3].set(title="G. Pair-specific B-only gate", xticks=(0, 1), xticklabels=("failed", "strict success"), ylabel="candidates")
    fig.suptitle("Phase 2T equal-digit control: endpoint construction succeeded; dynamic control gate failed")
    fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(figures / "phase2T_main_result.pdf"); plt.close(fig)

    failures = results["failure_modes"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    state_items = sorted(failures["fingertip_state_search"].items(), key=lambda item: item[1], reverse=True)
    axes[0].barh([item[0] for item in state_items], [item[1] for item in state_items]); axes[0].invert_yaxis(); axes[0].set_title("A. Fingertip-state rejection")
    b_items = sorted(failures["pair_specific_B_only"].items(), key=lambda item: item[1], reverse=True)
    axes[1].barh([item[0] for item in b_items], [item[1] for item in b_items]); axes[1].invert_yaxis(); axes[1].set_title("B. index+middle B-only failure")
    fig.suptitle(f"Phase 2T stopped at {STOP_CODE}"); fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(figures / "failure_modes.pdf"); plt.close(fig)

    phase2s = json.loads(_latest(ROOT / "outputs" / "phase2S" / "formal", "analysis_results.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.5))
    for ax, row, title in ((axes[0, 0], representative_f, "A. ACQUIRE: eligible fingertip"), (axes[0, 1], representative_p, "B. SECURE: palmar")):
        ax.imshow(_render(row, scene_cfg)); ax.set(xticks=[], yticks=[]); ax.set_title(title, weight="bold")
    elig = phase2s["eligibility"]
    axes[0, 2].bar((0, 1), (elig["FINGERTIP"]["fraction"], elig["PALMAR_SECURED"]["fraction"]), color=(COLORS["FINGERTIP"], COLORS["PALMAR_SECURED"]))
    axes[0, 2].set(title="C. Phase 2S eligibility recovery", xticks=(0, 1), xticklabels=("Tip", "Palm"), ylim=(0, 1))
    axes[1, 0].bar((0, 1), (len(fingertip), len(palmar)), color=(COLORS["FINGERTIP"], COLORS["PALMAR_SECURED"]))
    axes[1, 0].set(title="D. Phase 2T equal-digit endpoints", xticks=(0, 1), xticklabels=("Tip", "Palm"), ylabel="valid states")
    axes[1, 1].text(.5, .62, "Dynamic control not identified", ha="center", weight="bold", color="#B22222")
    axes[1, 1].text(.5, .40, "pair-specific B-only: 0/4096", ha="center"); axes[1, 1].axis("off")
    axes[1, 2].text(.5, .65, "Phase 2U not run", ha="center", weight="bold")
    axes[1, 2].text(.5, .43, "no frozen Phase 2T B control", ha="center")
    axes[1, 2].text(.5, .20, "transfer/secure controller: proposed work", ha="center", style="italic"); axes[1, 2].axis("off")
    fig.suptitle("Preliminary resource evidence hierarchy (measured evidence only)")
    fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(proposal / "preliminary_resource_evidence.pdf"); plt.close(fig)


def main() -> int:
    phase2t, phase2t_path = load_phase2t_config()
    phase2, phase2_path = load_phase2_config()
    fingertip_path = _largest(ROOT / phase2t.output_dir / "fingertip_states", "accepted_states.jsonl")
    palmar_path = _largest(ROOT / phase2t.output_dir / "palmar_states", "accepted_states.jsonl")
    fingertip_all, palmar_all = _jsonl(fingertip_path), _jsonl(palmar_path)
    fingertip = [row for row in fingertip_all if row["free_finger_set"] == ["index", "middle"]]
    palmar = [row for row in palmar_all if row["free_finger_set"] == ["index", "middle"]]
    states = fingertip + palmar
    b_summary_path = _latest(ROOT / phase2t.output_dir / "b_only_pair", "summary.json")
    b_summary = json.loads(b_summary_path.read_text(encoding="utf-8"))
    if b_summary["status"] != STOP_CODE or b_summary["strict_success_count"] != 0:
        raise RuntimeError("this analysis is only valid for the recorded Phase 2T B-control stop")
    if any(row["free_finger_count"] != 2 or row["free_finger_set"] != ["index", "middle"] for row in states):
        raise RuntimeError("Phase 2T endpoint identity invariant failed")
    base_cfg = load_configs(scene_filename=phase2t.scene_filename)
    cfg_hash = config_hash([phase2t_path, phase2_path, fingertip_path, palmar_path, b_summary_path])
    output = ROOT / phase2t.output_dir / "analysis" / cfg_hash[:12]
    store = IncrementalJsonlStore(output / "resource_states.jsonl", 30.0, 0.05)
    completed = store.completed_ids()
    pending = [row for row in states if stable_trial_id("phase2T-resource-state", row["grasp_state_id"]) not in completed]
    workers = min(max(1, (os.cpu_count() or 1) // 2), phase2t.state_search.maximum_workers)
    commit = git_commit_sha(ROOT)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        payloads = ((row, phase2.resources, phase2t.state_search.seed, base_cfg, cfg_hash, commit) for row in pending)
        for count, result in enumerate(executor.map(_resource_task, payloads), start=1):
            store.append(result)
            if count % workers == 0 or count == len(pending):
                print(f"Phase 2T resources: {len(completed) + count}/{len(states)}", flush=True)
    resource = {row["grasp_state_id"]: row for row in store.records()}
    states = [{**row, **resource[row["grasp_state_id"]]} for row in states]
    keys = (
        "free_finger_workspace_vol_m3", "free_palm_volume_m3", "COM_to_palm_origin_distance_m",
        "palm_A_contact_fraction", "ferrari_canny_epsilon", "total_A_normal_force_N",
        "A_translation_drift_m", "A_rotation_drift_rad", "minimum_joint_margin_rad",
    )
    comparisons = {
        key: _effect(
            [row[key] for row in states if row["grasp_state_type"] == "FINGERTIP"],
            [row[key] for row in states if row["grasp_state_type"] == "PALMAR_SECURED"],
        ) for key in keys
    }
    f_summary = json.loads((fingertip_path.parent / "summary.json").read_text(encoding="utf-8"))
    p_summary = json.loads((palmar_path.parent / "summary.json").read_text(encoding="utf-8"))
    results = {
        "experiment_id": phase2t.experiment_id,
        "stop_code": STOP_CODE,
        "interpretation_case": "T4",
        "interpretation": "Eligible equal-digit endpoints exist, but the mandatory pair-specific B-only positive control failed; dynamic comparison is unidentifiable.",
        "fingertip_state_search": f_summary,
        "palmar_state_search": p_summary,
        "candidate_endpoint_counts": {"FINGERTIP": len(fingertip), "PALMAR_SECURED": len(palmar)},
        "all_valid_endpoint_counts": {"FINGERTIP": len(fingertip_all), "PALMAR_SECURED": len(palmar_all)},
        "selected_candidate_free_finger_set": ["index", "middle"],
        "formal_topology_selected": None,
        "pair_specific_B_only": b_summary,
        "formal_matching_performed": False,
        "formal_trials": 0,
        "endpoint_covariate_distributions": comparisons,
        "failure_modes": {
            "fingertip_state_search": f_summary["rejection_reasons"],
            "palmar_state_search": p_summary["rejection_reasons"],
            "pair_specific_B_only": b_summary["failure_mechanisms"],
        },
        "config_hash": cfg_hash,
        "git_commit_sha": commit,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    report = ROOT / "docs" / "PHASE2T_ELIGIBLE_FINGERTIP_CONTROL_RESULTS.md"
    report.write_text(
        "# Phase 2T digit-eligible fingertip control results\n\n"
        f"**Stop code: `{STOP_CODE}`**\n\n"
        "## Motivation\n\nPhase 2S established digit eligibility but could not isolate a conditional dynamic palmar effect. Phase 2T held digit count and identity fixed. Physics and all thresholds remained identical to Phase 2S.\n\n"
        "## Endpoint search\n\n```json\n" + json.dumps({"FINGERTIP": f_summary, "PALMAR_SECURED": p_summary}, indent=2) + "\n```\n\n"
        "The only topology satisfying the 50-state endpoint-population requirement in both groups had occupied `ring+thumb` and free `index+middle`. Both groups therefore had exactly two occupied and two free fingers with identical free-finger identity.\n\n"
        "## Mandatory pair-specific B-only control\n\n"
        "Only index and middle were commanded/permitted as acquisition digits. Phase 2S regions and successful trajectories were tried first, followed by deterministic bounded trajectory/pose refinement. The authorized cap was exhausted: 0 strict successes in 4,096 candidates.\n\n```json\n"
        + json.dumps(b_summary, indent=2) + "\n```\n\n"
        "## Resource descriptors under equal digit count\n\n```json\n" + json.dumps(comparisons, indent=2) + "\n```\n\n"
        "## Formal inference\n\nNo B distribution or controller was frozen, no calibration A+B outcomes were run, no formal matching dataset was created, and no A+B formal outcomes were inspected. `INSUFFICIENT_FREE_DIGITS_PRECHECK` is absent from the constructed endpoint populations, but the dynamic experiment is unidentifiable because its mandatory positive control failed.\n\n"
        "## Interpretation\n\nCase T4. Eligible FINGERTIP states exist, but the pair-specific second-acquisition control failed. No palmar-versus-fingertip dynamic effect is estimated.\n\n"
        "## Limitations\n\nThe failure is specific to the tested half-scale B, index+middle topology, existing Phase 2S proposal regions, and bounded 4,096-candidate search. It does not prove all possible two-digit B acquisition is impossible. No scalar J, transfer, gaiting, wrist control, third object, reward change, or RL was introduced.\n",
        encoding="utf-8",
    )
    (ROOT / "docs" / "PHASE2T_INTERPRETATION.md").write_text(
        f"# Phase 2T interpretation\n\n**Case T4 - dynamic experiment remains unidentifiable.**\n\nStable two-finger FINGERTIP endpoints exist, so T1 does not apply. However, the only free-finger topology with sufficient endpoint states in both groups (`index+middle`) produced 0 strict pair-specific B-only successes after the full 4,096-candidate cap. The mandatory control gate therefore fired as `{STOP_CODE}`. T2 and T3 cannot be evaluated because no A+B formal experiment was authorized.\n",
        encoding="utf-8",
    )
    (ROOT / "docs" / "PHASE2T_PRELIMINARY_EVIDENCE.md").write_text(
        "# Phase 2T preliminary evidence\n\n"
        "## Result A - digit eligibility\n\nPhase 2S measured 0/100 eligible FINGERTIP matched states and 100/100 eligible PALMAR_SECURED states. Phase 2T then constructed 150 strict two-free FINGERTIP endpoints and 151 strict two-free PALMAR endpoints with the same `index+middle` free set.\n\n"
        "## Result B - conditional dynamic effect\n\nNot identified. Pair-specific index+middle B-only acquisition produced 0/4,096 strict successes, triggering the mandatory stop gate before controller freeze, matching, calibration, or formal trials. No dynamic group difference is reported.\n",
        encoding="utf-8",
    )
    (ROOT / "docs" / "PHASE2STU_CONSOLIDATED_EVIDENCE.md").write_text(
        "# Consolidated Phase 2S/2T/2U evidence\n\n"
        "## Primary evidence - Phase 2S\n\nPalmar securing restored digit/resource eligibility: FINGERTIP left one free digit, whereas PALMAR_SECURED left at least two. The Phase 2S dynamic result is eligibility-confounded.\n\n"
        "## Control experiment - Phase 2T\n\nEqual-digit, identical-topology endpoints were constructed, but the mandatory pair-specific B-only positive control failed (0/4,096). The conditional dynamic palmar effect remains unidentified (T4).\n\n"
        "## Sensitivity experiment - Phase 2U\n\nNot run. Phase 2T did not produce the frozen B distribution and controller required for a mass-only replay. Creating substitutes would confound the specified sensitivity and require a new PI decision.\n\n"
        "These experiments are not pooled into one statistical population. No transfer is demonstrated.\n",
        encoding="utf-8",
    )
    component_notes = {
        "occupied_finger_count": ("load-bearing digits above 0.20 N", "count"),
        "free_finger_count": ("digits below the occupied threshold", "count"),
        "free_finger_workspace_vol_m3": ("collision-free sampled fingertip reachability", "m^3"),
        "free_palm_volume_m3": ("unoccupied configured palm-frame voxel volume", "m^3"),
        "COM_to_palm_origin_distance_m": ("object COM location relative to palm origin", "m"),
        "palm_A_contact_fraction": ("fraction of unsupported hold with real palm-A contact", "unitless"),
        "ferrari_canny_epsilon": ("force-closure quality descriptor", "normalized epsilon"),
    }
    lines = ["# Updated evidence for a future PI decision about J", "", "State descriptors are measurements of an endpoint. Candidate resource-metric components are descriptors the PI may later choose to include in J; Phase 2T does not make that choice.", ""]
    for key, (meaning, units) in component_notes.items():
        if key in ("occupied_finger_count", "free_finger_count"):
            effect = {"FINGERTIP": {"minimum": 2, "maximum": 2, "mean": 2}, "PALMAR_SECURED": {"minimum": 2, "maximum": 2, "mean": 2}, "mean_difference_palmar_minus_fingertip": 0}
        else:
            effect = comparisons[key]
        lines.extend([
            f"## {key}", "", f"- Physical meaning: {meaning}.", f"- Units: {units}.",
            f"- Measured Phase 2T range/group effect: `{json.dumps(effect, sort_keys=True)}`.",
            "- Relationship with second-grasp success: not identifiable in Phase 2T because the pair-specific B-only gate failed before formal outcomes.",
            "- Redundancy/correlation: descriptive overlap with the other endpoint components was retained for PI review; no component was collapsed or weighted.",
            "- Sim-to-real measurability: digit contacts and palm contact require tactile/contact sensing; COM pose requires object tracking; workspace/volume requires a calibrated geometric model; Ferrari-Canny requires contact geometry and friction assumptions.", "",
        ])
    lines.extend([
        "## TODO(PI)", "", "Choose whether J should represent: (a) digit availability, (b) future reachable manipulation volume, (c) palm storage capacity, (d) a combination, and define functional form / normalization only after PI review.",
    ])
    (ROOT / "docs" / "J_PI_DECISION_EVIDENCE_UPDATED.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _plot_outputs(states, results, ROOT / "docs" / "figures" / "phase2T")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
