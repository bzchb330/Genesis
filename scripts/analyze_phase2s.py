#!/usr/bin/env python
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pointbiserialr
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_phase2r import (
    COLORS,
    GROUPS,
    _describe,
    _effect,
    _paired,
    _rate,
    _safe_odds,
    _safe_relative,
    _wilson,
)
from seqgrasp.config import ROOT
from seqgrasp.experiments.phase2r import PHASE2R_OUTCOMES
from seqgrasp.phase2s_config import load_phase2s_config


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _latest(root: Path, filename: str):
    candidates = [(path.stat().st_mtime_ns, path) for path in root.rglob(filename)]
    if not candidates:
        raise FileNotFoundError(filename)
    return max(candidates)[1]


def _state_comparisons(states):
    keys = (
        "occupied_finger_count",
        "free_finger_count",
        "free_finger_workspace_vol_m3",
        "free_palm_volume_m3",
        "COM_to_palm_origin_distance_m",
        "palm_A_contact_fraction",
        "palm_A_normal_force_N",
        "ferrari_canny_epsilon",
        "total_A_normal_force_N",
        "A_translation_drift_m",
        "A_rotation_drift_rad",
    )
    return {
        key: _effect(
            [row[key] for row in states if row["grasp_state_type"] == "FINGERTIP"],
            [row[key] for row in states if row["grasp_state_type"] == "PALMAR_SECURED"],
        )
        for key in keys
    }


def _phase2r_results():
    path = _latest(ROOT / "outputs" / "phase2R" / "formal", "analysis_results.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _large_vs_half(phase2s_results, states):
    phase2r = _phase2r_results()
    large_freeze = yaml.safe_load((ROOT / "configs" / "phase2R_frozen_B_distribution.yaml").read_text(encoding="utf-8"))
    small_freeze = yaml.safe_load((ROOT / "configs" / "phase2S_frozen_B_distribution.yaml").read_text(encoding="utf-8"))
    phase2s_state_summary = json.loads(_latest(ROOT / "outputs" / "phase2S" / "palmar_states", "summary.json").read_text(encoding="utf-8"))
    large_palmar_acceptance = {"accepted": 150, "attempts": 4672, "rate": 150 / 4672}
    small_palmar_acceptance = {
        "accepted": phase2s_state_summary["accepted_states"],
        "attempts": phase2s_state_summary["attempts"],
        "rate": phase2s_state_summary["accepted_states"] / phase2s_state_summary["attempts"],
    }
    small_resources = phase2s_results["resource_component_comparisons"]
    large_resources = phase2r["resource_component_comparisons"]
    return {
        "interpretation": "cross-experiment descriptive comparison only; objects were not randomized within one experiment",
        "palmar_state_acceptance": {"Phase2R_large": large_palmar_acceptance, "Phase2S_half_scale": small_palmar_acceptance},
        "palmar_resource_means": {
            key: {
                "Phase2R_large": large_resources[key]["PALMAR_SECURED_mean"],
                "Phase2S_half_scale": small_resources[key]["PALMAR_SECURED_mean"],
            }
            for key in ("occupied_finger_count", "free_finger_count", "free_palm_volume_m3")
        },
        "digit_eligibility": {
            "Phase2R_large": phase2r["eligibility"],
            "Phase2S_half_scale": phase2s_results["eligibility"],
        },
        "B_geometry_access": {
            "Phase2R_large": large_freeze["geometry_access"],
            "Phase2S_half_scale": small_freeze["geometry_access"],
        },
        "BOTH_RETAINED": {
            "Phase2R_large": phase2r["primary_comparison"],
            "Phase2S_half_scale": phase2s_results["primary_comparison"],
        },
        "failure_modes": {
            "Phase2R_large": phase2r["failure_modes"],
            "Phase2S_half_scale": phase2s_results["failure_modes"],
        },
    }


def _plot_main(states, results, figures):
    plt.style.use(ROOT / "configs" / "phase2_publication.mplstyle")
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.8))
    rendered = ROOT / "outputs" / "phase2S" / "diagnostics" / "rendered_states"
    for ax, image_name, title in (
        (axes[0, 0], "main_fingertip.png", "A. FINGERTIP endpoint"),
        (axes[0, 1], "main_palmar.png", "B. PALMAR_SECURED endpoint"),
    ):
        image_path = rendered / image_name
        if image_path.exists():
            ax.imshow(plt.imread(image_path))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, weight="bold")
    categories = np.arange(2)
    occupied = [
        np.mean([row["occupied_finger_count"] for row in states if row["grasp_state_type"] == group])
        for group in GROUPS
    ]
    free = [4.0 - value for value in occupied]
    axes[0, 2].bar(categories - .18, occupied, .36, label="occupied")
    axes[0, 2].bar(categories + .18, free, .36, label="free")
    axes[0, 2].set(title="C. Digit resources", xticks=categories, xticklabels=("Tip", "Palm"), ylabel="mean fingers")
    axes[0, 2].legend(fontsize=7)
    eligibility = results["eligibility"]
    axes[1, 0].bar(categories, [eligibility[group]["fraction"] for group in GROUPS], color=[COLORS[group] for group in GROUPS])
    axes[1, 0].set(title="D. Second-grasp eligibility", xticks=categories, xticklabels=("Tip", "Palm"), ylim=(0, 1), ylabel="fraction")
    rates = results["primary_comparison"]
    means = [rates[group]["rate"] for group in GROUPS]
    lows = [rates[group]["wilson_95_CI"][0] for group in GROUPS]
    highs = [rates[group]["wilson_95_CI"][1] for group in GROUPS]
    axes[1, 1].bar(categories, means, color=[COLORS[group] for group in GROUPS])
    axes[1, 1].errorbar(
        categories,
        means,
        yerr=[
            np.maximum(0.0, np.asarray(means) - np.asarray(lows)),
            np.maximum(0.0, np.asarray(highs) - np.asarray(means)),
        ],
        fmt="none",
        color="black",
        capsize=3,
    )
    axes[1, 1].set(title="E. BOTH_RETAINED", xticks=categories, xticklabels=("Tip", "Palm"), ylim=(0, max(.05, max(highs) * 1.2)), ylabel="rate")
    failures = results["failure_modes"]
    names = ("BOTH_RETAINED", "A_DROPPED", "B_NOT_ACQUIRED", "BOTH_LOST", "INVALID")
    bottom = np.zeros(2)
    for name in names:
        values = np.asarray([failures[group]["outcomes"][name] for group in GROUPS])
        axes[1, 2].bar(categories, values, bottom=bottom, label=name)
        bottom += values
    axes[1, 2].set(title="F. Formal outcomes", xticks=categories, xticklabels=("Tip", "Palm"), ylabel="trials")
    axes[1, 2].legend(fontsize=6, loc="upper right")
    fig.suptitle("Phase 2S half-scale endpoint comparison", weight="bold")
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(figures / "phase2S_main_result.pdf")
    plt.close(fig)


def _plot_large_vs_half(comparison, figures):
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.4))
    labels = ("LARGE\nPhase 2R", "HALF-SCALE\nPhase 2S")
    acceptance = comparison["palmar_state_acceptance"]
    axes[0, 0].bar(labels, [acceptance["Phase2R_large"]["rate"], acceptance["Phase2S_half_scale"]["rate"]])
    axes[0, 0].set(title="A. Palmar-state acceptance", ylabel="accepted / attempts")
    for column, key in enumerate(("occupied_finger_count", "free_finger_count"), start=1):
        values = comparison["palmar_resource_means"][key]
        axes[0, column].bar(labels, [values["Phase2R_large"], values["Phase2S_half_scale"]])
        axes[0, column].set(title=f"{chr(65 + column)}. {key.replace('_', ' ')}", ylabel="palmar-state mean")
    eligibility = comparison["digit_eligibility"]
    axes[1, 0].bar(labels, [
        eligibility["Phase2R_large"]["PALMAR_SECURED"]["fraction"],
        eligibility["Phase2S_half_scale"]["PALMAR_SECURED"]["fraction"],
    ])
    axes[1, 0].set(title="D. Palmar digit eligibility", ylabel="fraction")
    access = comparison["B_geometry_access"]
    axes[1, 1].bar(labels, [
        access["Phase2R_large"]["PALMAR_SECURED"]["access_fraction"],
        access["Phase2S_half_scale"]["PALMAR_SECURED"]["access_fraction"],
    ])
    axes[1, 1].set(title="E. Palmar B geometry access", ylabel="fraction")
    retained = comparison["BOTH_RETAINED"]
    axes[1, 2].bar(labels, [
        retained["Phase2R_large"]["PALMAR_SECURED"]["rate"],
        retained["Phase2S_half_scale"]["PALMAR_SECURED"]["rate"],
    ])
    axes[1, 2].set(title="F. Palmar BOTH_RETAINED", ylabel="formal rate")
    fig.suptitle("Descriptive large-vs-half-scale comparison (not a randomized size effect)", weight="bold")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(figures / "large_vs_half_scale.pdf")
    plt.close(fig)


def main() -> int:
    phase2s, _ = load_phase2s_config()
    formal_path = _latest(ROOT / phase2s.output_dir / "formal", "trials.jsonl")
    matched_path = _latest(ROOT / phase2s.output_dir / "matching", "matched_states.jsonl")
    rows = [row for row in _jsonl(formal_path) if not row.get("pilot_only") and not row.get("calibration_only")]
    states = _jsonl(matched_path)
    if len(rows) != 4000:
        raise RuntimeError(f"formal analysis requires 4000 trials, found {len(rows)}")
    primary = {group: _rate([row for row in rows if row["grasp_state_type"] == group]) for group in GROUPS}
    ps, pn = primary["PALMAR_SECURED"]["successes"], primary["PALMAR_SECURED"]["valid_trials"]
    fs, fn = primary["FINGERTIP"]["successes"], primary["FINGERTIP"]["valid_trials"]
    comparison = {
        **primary,
        "absolute_percentage_point_difference_palmar_minus_fingertip": 100 * (ps / pn - fs / fn),
        "relative_risk_palmar_over_fingertip": _safe_relative(ps, pn, fs, fn),
        "odds_ratio_palmar_over_fingertip": _safe_odds(ps, pn, fs, fn),
    }
    eligibility = {}
    for group in GROUPS:
        group_states = [row for row in states if row["grasp_state_type"] == group]
        eligible = sum(int(row["free_finger_count"]) >= 2 for row in group_states)
        eligibility[group] = {
            "eligible_states": eligible,
            "state_count": len(group_states),
            "fraction": eligible / len(group_states),
            "wilson_95_CI": _wilson(eligible, len(group_states)),
        }
    conditional = {
        group: _rate([
            row for row in rows
            if row["grasp_state_type"] == group
            and row["second_grasp_digit_eligible"]
            and row["B_geometrically_reachable"]
        ])
        for group in GROUPS
    }
    unique_states = {row["grasp_state_id"]: row for row in states}
    epsilon = np.asarray([row["ferrari_canny_epsilon"] for row in rows if row["outcome"] != "INVALID"])
    outcomes = np.asarray([row["outcome"] == "BOTH_RETAINED" for row in rows if row["outcome"] != "INVALID"], dtype=float)
    correlation = pointbiserialr(outcomes, epsilon) if len(np.unique(outcomes)) > 1 else None
    cutoff = float(np.quantile([row["ferrari_canny_epsilon"] for row in states], .9))
    top = [
        row for row in rows
        if row["outcome"] != "INVALID"
        and unique_states[row["grasp_state_id"]]["ferrari_canny_epsilon"] >= cutoff
    ]
    ferrari = {
        "matched_group_distributions": {
            group: _describe([row["ferrari_canny_epsilon"] for row in states if row["grasp_state_type"] == group])
            for group in GROUPS
        },
        "top_decile_cutoff": cutoff,
        "top_decile_success_rate": float(np.mean([row["outcome"] == "BOTH_RETAINED" for row in top])) if top else None,
        "full_population_success_rate": float(np.mean(outcomes)),
        "point_biserial_correlation": None if correlation is None else {"r": float(correlation.statistic), "p_value": float(correlation.pvalue)},
    }
    failures = {
        group: {
            "outcomes": {
                outcome: sum(row["outcome"] == outcome for row in rows if row["grasp_state_type"] == group)
                for outcome in PHASE2R_OUTCOMES
            },
            "subreasons": dict(Counter(
                str(row.get("outcome_subreason")) for row in rows if row["grasp_state_type"] == group
            )),
        }
        for group in GROUPS
    }
    results = {
        "experiment_id": phase2s.formal_experiment_id,
        "formal_trials": len(rows),
        "valid_trials": sum(row["outcome"] != "INVALID" for row in rows),
        "primary_comparison": comparison,
        "paired_analysis": _paired(rows, phase2s.second_grasp.formal_seed),
        "eligibility": eligibility,
        "conditional_eligible_reachable": conditional,
        "resource_component_comparisons": _state_comparisons(states),
        "ferrari_canny_baseline": ferrari,
        "failure_modes": failures,
    }
    large_vs_half = _large_vs_half(results, states)
    results["large_vs_half_scale_descriptive"] = large_vs_half
    output = formal_path.parent
    (output / "analysis_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    figures = ROOT / "docs" / "figures" / "phase2S"
    figures.mkdir(parents=True, exist_ok=True)
    _plot_main(states, results, figures)
    _plot_large_vs_half(large_vs_half, figures)
    report = ROOT / "docs" / "PHASE2S_HALF_SCALE_OBJECT_RESULTS.md"
    geometry = (ROOT / "docs" / "PHASE2S_GEOMETRY_VALIDATION.md").read_text(encoding="utf-8")
    matching = (ROOT / "docs" / "PHASE2S_MATCHING_REPORT.md").read_text(encoding="utf-8")
    report.write_text(
        "# Phase 2S half-scale object results\n\n"
        "## Scope and geometry revision\n\n"
        "Phase 2S reduces every physical linear dimension of A and B to exactly 50% of the Phase 2R geometry while retaining each mass at 0.08 kg and retaining friction, gravity, timestep, contact parameters, actuation, gains, thresholds, and outcome semantics. This isolates geometry rather than material density. Phase 2R remains the historical large-object baseline. No transfer trajectory is simulated.\n\n"
        + geometry + "\n\n"
        "## Regenerated endpoint populations, resources, and matching\n\n"
        + matching + "\n\n"
        "## Small-B graspability, common region, and controller calibration\n\n"
        "The small-B map evaluated 8,192 geometry candidates. The strict dynamic search found 24 successful pose/trajectory combinations after 4,392 evaluated candidates. Ten correctly paired profiles were locally perturbed; the frozen common distribution and controller are recorded in `PHASE2S_B_DISTRIBUTION_FREEZE.md` and `PHASE2S_CONTROLLER_FREEZE.md`.\n\n"
        "## Formal, paired, eligibility, resource, and Ferrari–Canny results\n\n```json\n"
        + json.dumps({
            "primary": comparison,
            "paired": results["paired_analysis"],
            "eligibility": eligibility,
            "eligible_reachable": conditional,
            "resources": results["resource_component_comparisons"],
            "Ferrari_Canny": ferrari,
            "failures": failures,
        }, indent=2) + "\n```\n\n"
        "## Large-versus-half-scale descriptive comparison\n\n"
        "This cross-experiment comparison is descriptive only; raw trials are not pooled as though size were randomized within one experiment, and it does not establish causal significance.\n\n```json\n"
        + json.dumps(large_vs_half, indent=2) + "\n```\n\n"
        "## Limitations\n\n"
        "Endpoint initialization does not establish transfer controllability. The fixed-mass scale change increases density. One scripted B controller and one pre-frozen region probe only a narrow acquisition family. No scalar resource score, wrist controller, three-object task, transfer controller, or RL policy is defined.\n",
        encoding="utf-8",
    )
    paired_p = results["paired_analysis"]["McNemar_exact_two_sided_p_value"]
    if ps + fs == 0:
        evidence = "C: dynamically/statistically unidentifiable because both groups had zero BOTH_RETAINED trials."
    elif ps / pn > fs / fn and paired_p is not None and paired_p < .05:
        evidence = "A: observed evidence supports a PALMAR_SECURED advantage in the paired formal experiment."
    else:
        evidence = "B: no statistically detectable PALMAR_SECURED advantage was established."
    (ROOT / "docs" / "PHASE2S_PRELIMINARY_EVIDENCE.md").write_text(
        "# Phase 2S preliminary evidence\n\n"
        f"**Observed classification:** {evidence}\n\n"
        f"The formal experiment completed {len(rows)} trials. FINGERTIP achieved {fs}/{fn} valid-trial BOTH_RETAINED outcomes and PALMAR_SECURED achieved {ps}/{pn}. The absolute difference was {comparison['absolute_percentage_point_difference_palmar_minus_fingertip']:.4f} percentage points; the matched-pair bootstrap 95% interval was {results['paired_analysis']['absolute_rate_difference_bootstrap_95_CI']}, and exact McNemar p was {paired_p}.\n\n"
        "These are directly initialized endpoint states. No transfer dynamics, scalar J, wrist control, third object, or RL training was implemented.\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
