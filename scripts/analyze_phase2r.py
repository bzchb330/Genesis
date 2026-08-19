#!/usr/bin/env python
from __future__ import annotations

from collections import Counter
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "seqgrasp-matplotlib"))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, pointbiserialr
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint

from seqgrasp.config import ROOT
from seqgrasp.experiments.phase2r import PHASE2R_OUTCOMES
from seqgrasp.phase2r_config import load_phase2r_config


COLORS = {"FINGERTIP": "#0072B2", "PALMAR_SECURED": "#D55E00"}
GROUPS = ("FINGERTIP", "PALMAR_SECURED")


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _latest(root: Path, filename: str) -> Path:
    candidates = [(path.stat().st_mtime_ns, path) for path in root.rglob(filename)]
    if not candidates:
        raise FileNotFoundError(filename)
    return max(candidates)[1]


def _wilson(successes: int, count: int) -> list[float] | None:
    if count == 0:
        return None
    low, high = proportion_confint(successes, count, alpha=0.05, method="wilson")
    return [float(low), float(high)]


def _rate(rows: list[dict]) -> dict:
    valid = [row for row in rows if row["outcome"] != "INVALID"]
    successes = sum(row["outcome"] == "BOTH_RETAINED" for row in valid)
    return {
        "successes": successes, "valid_trials": len(valid),
        "rate": successes / len(valid) if valid else None,
        "wilson_95_CI": _wilson(successes, len(valid)),
    }


def _safe_relative(a: int, an: int, b: int, bn: int) -> float | str | None:
    if not an or not bn:
        return None
    if b == 0:
        return "infinite" if a > 0 else None
    return (a / an) / (b / bn)


def _safe_odds(a: int, an: int, b: int, bn: int) -> float | str | None:
    if min(an, bn) == 0:
        return None
    if b == 0 and 0 < a < an:
        return "infinite"
    if a in (0, an) or b in (0, bn):
        return None
    return (a / (an - a)) / (b / (bn - b))


def _paired(rows: list[dict], seed: int) -> dict:
    cells = {}
    for row in rows:
        cells.setdefault((row["matched_pair_id"], int(row["B_seed_index"])), {})[row["grasp_state_type"]] = row
    counts = Counter()
    for values in cells.values():
        if set(values) != set(GROUPS) or any(values[group]["outcome"] == "INVALID" for group in GROUPS):
            counts["excluded_invalid_or_incomplete"] += 1
            continue
        finger = values["FINGERTIP"]["outcome"] == "BOTH_RETAINED"
        palm = values["PALMAR_SECURED"]["outcome"] == "BOTH_RETAINED"
        counts[("both_succeed" if finger and palm else "palmar_succeeds_fingertip_fails" if palm else "fingertip_succeeds_palmar_fails" if finger else "both_fail")] += 1
    b = counts["palmar_succeeds_fingertip_fails"]
    c = counts["fingertip_succeeds_palmar_fails"]
    p_value = float(binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue) if b + c else None
    by_pair = {}
    for pair_id in sorted({row["matched_pair_id"] for row in rows}):
        pair = [row for row in rows if row["matched_pair_id"] == pair_id and row["outcome"] != "INVALID"]
        group_rows = {group: [row for row in pair if row["grasp_state_type"] == group] for group in GROUPS}
        if any(not group_rows[group] for group in GROUPS):
            continue
        rates = {group: np.mean([row["outcome"] == "BOTH_RETAINED" for row in group_rows[group]]) for group in GROUPS}
        by_pair[pair_id] = float(rates["PALMAR_SECURED"] - rates["FINGERTIP"])
    rng = np.random.default_rng(seed)
    pair_values = np.asarray(list(by_pair.values()))
    bootstrap = np.asarray([
        np.mean(rng.choice(pair_values, size=len(pair_values), replace=True))
        for _ in range(10_000)
    ]) if len(pair_values) else np.asarray([])
    return {
        "paired_cells": len(cells),
        "palmar_succeeds_fingertip_fails": b,
        "fingertip_succeeds_palmar_fails": c,
        "both_succeed": counts["both_succeed"],
        "both_fail": counts["both_fail"],
        "excluded_invalid_or_incomplete": counts["excluded_invalid_or_incomplete"],
        "McNemar_exact_two_sided_p_value": p_value,
        "pair_level_bootstrap_replicates": 10_000,
        "pair_level_bootstrap_pairs_with_valid_trials_in_both_groups": len(pair_values),
        "absolute_rate_difference_bootstrap_95_CI": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))] if len(bootstrap) else None,
    }


def _regression(frame: pd.DataFrame, adjusted: bool) -> dict:
    if frame["both_retained"].nunique() < 2:
        return {"status": "NOT_IDENTIFIABLE", "reason": "requires both binary outcome classes"}
    predictors = ["palmar"]
    if adjusted:
        predictors += ["ferrari_canny_epsilon", "A_translation_drift_m", "A_rotation_drift_rad", "minimum_joint_margin_rad"]
    design = frame[predictors].astype(float).copy()
    transforms = {}
    for key in predictors[1:]:
        mean, std = float(design[key].mean()), float(design[key].std(ddof=0))
        transforms[key] = {"mean": mean, "standard_deviation": std}
        design[key] = (design[key] - mean) / std if std > 0 else 0.0
    design = sm.add_constant(design, has_constant="add")
    try:
        fit = sm.GLM(frame["both_retained"], design, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": frame["matched_pair_id"]}, maxiter=100,
        )
        interval = fit.conf_int(alpha=0.05)
        terms = {
            name: {
                "coefficient": float(fit.params[name]), "cluster_robust_standard_error": float(fit.bse[name]),
                "z": float(fit.tvalues[name]), "p_value": float(fit.pvalues[name]),
                "confidence_interval_95": [float(interval.loc[name, 0]), float(interval.loc[name, 1])],
                "odds_ratio": float(np.exp(fit.params[name])),
            }
            for name in fit.params.index
        }
        return {
            "status": "FIT", "formula": "BOTH_RETAINED ~ " + " + ".join(predictors),
            "cluster": "matched_pair_id", "N": int(fit.nobs), "terms": terms,
            "standardized_adjustment_covariates": transforms,
        }
    except Exception as exc:
        return {"status": "FIT_FAILED", "error": str(exc)}


def _effect(left: list[float], right: list[float]) -> dict:
    x, y = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    pooled = math.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2.0) if len(x) > 1 and len(y) > 1 else 0.0
    return {
        "FINGERTIP_mean": float(np.mean(x)), "FINGERTIP_standard_deviation": float(np.std(x, ddof=1)),
        "PALMAR_SECURED_mean": float(np.mean(y)), "PALMAR_SECURED_standard_deviation": float(np.std(y, ddof=1)),
        "mean_difference_palmar_minus_fingertip": float(np.mean(y) - np.mean(x)),
        "standardized_mean_difference": float((np.mean(y) - np.mean(x)) / pooled) if pooled else None,
    }


def _describe(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "count": len(array), "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "minimum": float(np.min(array)), "median": float(np.median(array)), "maximum": float(np.max(array)),
    }


def _state_comparisons(states: list[dict]) -> dict:
    keys = (
        "occupied_finger_count", "free_finger_count", "free_finger_workspace_vol_m3", "free_palm_volume_m3",
        "COM_to_palm_origin_distance_m", "palm_A_contact_fraction", "palm_A_normal_force_N",
        "ferrari_canny_epsilon", "total_A_normal_force_N", "A_translation_drift_m", "A_rotation_drift_rad",
    )
    return {
        key: _effect(
            [row[key] for row in states if row["grasp_state_type"] == "FINGERTIP"],
            [row[key] for row in states if row["grasp_state_type"] == "PALMAR_SECURED"],
        )
        for key in keys
    }


def _plot_main(states, results, figures: Path):
    plt.style.use(ROOT / "configs" / "phase2_publication.mplstyle")
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.2))
    rendered = ROOT / "outputs" / "phase2R" / "diagnostics" / "rendered_states"
    for ax, group, label, image_name in ((axes[0, 0], "FINGERTIP", "A. FINGERTIP endpoint", "main_fingertip.png"), (axes[0, 1], "PALMAR_SECURED", "B. PALMAR_SECURED endpoint", "main_palmar.png")):
        row = next(state for state in states if state["grasp_state_type"] == group)
        image_path = rendered / image_name
        if image_path.exists():
            ax.imshow(plt.imread(image_path)); ax.set_xticks([]); ax.set_yticks([])
        else:
            ax.axis("off")
        ax.set_title(label, weight="bold")
        ax.text(0.02, 0.02, "occupied: " + ", ".join(f for f, flag in zip(("index", "middle", "ring", "thumb"), row["occupied_finger_mask"]) if flag) + "\nfree: " + ", ".join(f for f, flag in zip(("index", "middle", "ring", "thumb"), row["occupied_finger_mask"]) if not flag), fontsize=6, color="white", bbox={"facecolor": "black", "alpha": .55}, transform=ax.transAxes)
    for group in GROUPS:
        values = [row["COM_to_palm_origin_distance_m"] for row in states if row["grasp_state_type"] == group]
        axes[0, 2].hist(values, bins=15, alpha=0.65, label=group, color=COLORS[group])
    axes[0, 2].set(title="C. A COM-to-palm distance", xlabel="distance [m]", ylabel="states"); axes[0, 2].legend(fontsize=7)
    categories = np.arange(2)
    occupied = [np.mean([row["occupied_finger_count"] for row in states if row["grasp_state_type"] == g]) for g in GROUPS]
    free = [np.mean([row["free_finger_count"] for row in states if row["grasp_state_type"] == g]) for g in GROUPS]
    axes[0, 3].bar(categories - .18, occupied, .36, label="occupied"); axes[0, 3].bar(categories + .18, free, .36, label="free")
    axes[0, 3].set(title="D. Digit occupancy", xticks=categories, xticklabels=("Tip", "Palm"), ylabel="mean fingers"); axes[0, 3].legend(fontsize=7)
    eligibility = results["eligibility"]
    axes[1, 0].bar(categories, [eligibility[g]["fraction"] for g in GROUPS], color=[COLORS[g] for g in GROUPS])
    axes[1, 0].set(title="E. Second-grasp eligibility", xticks=categories, xticklabels=("Tip", "Palm"), ylim=(0, 1), ylabel="fraction")
    rates = results["primary_comparison"]
    means = [rates[g]["rate"] for g in GROUPS]
    lows = [rates[g]["wilson_95_CI"][0] for g in GROUPS]
    highs = [rates[g]["wilson_95_CI"][1] for g in GROUPS]
    axes[1, 1].bar(categories, means, color=[COLORS[g] for g in GROUPS]); axes[1, 1].errorbar(categories, means, yerr=[np.asarray(means)-lows, np.asarray(highs)-means], fmt="none", color="black", capsize=3)
    axes[1, 1].set(title="F. BOTH_RETAINED", xticks=categories, xticklabels=("Tip", "Palm"), ylim=(0, max(.05, max(highs)*1.2)), ylabel="rate")
    conditional = results["conditional_eligible_reachable"]
    cmeans = [np.nan if conditional[g]["rate"] is None else conditional[g]["rate"] for g in GROUPS]
    finite_cmeans = [value for value in cmeans if np.isfinite(value)]
    axes[1, 2].bar(categories, cmeans, color=[COLORS[g] for g in GROUPS]); axes[1, 2].set(title="G. Eligible + reachable", xticks=categories, xticklabels=("Tip", "Palm"), ylim=(0, max(.05, max(finite_cmeans, default=0.0)*1.2)), ylabel="success rate")
    for index, value in enumerate(cmeans):
        if not np.isfinite(value):
            axes[1, 2].text(
                index, .025, "not\nidentifiable", ha="center", va="center", fontsize=7,
                bbox={"facecolor": "white", "edgecolor": "0.7", "boxstyle": "round,pad=.25"},
            )
    axes[1, 3].axis("off"); axes[1, 3].text(0.02, .88, "H. Endpoint comparison", weight="bold", transform=axes[1, 3].transAxes)
    axes[1, 3].text(.02, .58, "ACQUIRE A → transfer\n(not simulated) → SECURE A", transform=axes[1, 3].transAxes)
    axes[1, 3].text(.02, .27, "Then acquire B with the\nsame frozen distribution/controller", transform=axes[1, 3].transAxes)
    fig.tight_layout(); fig.savefig(figures / "phase2R_main_result.pdf"); plt.close(fig)


def _plot_concept(figures: Path):
    fig, ax = plt.subplots(figsize=(10.5, 2.4)); ax.axis("off")
    labels = ["ACQUIRE", "TRANSFER", "SECURE", "FREE DIGITS", "ACQUIRE NEXT"]
    for index, label in enumerate(labels):
        x = 0.02 + index * 0.195
        ax.text(x + .075, .62, label, ha="center", va="center", weight="bold", bbox={"boxstyle": "round,pad=.5", "facecolor": "#E6F2F8"}, transform=ax.transAxes)
        if index < len(labels) - 1:
            ax.annotate("", xy=(x + .19, .62), xytext=(x + .155, .62), arrowprops={"arrowstyle": "->"}, xycoords=ax.transAxes)
    ax.text(.29, .23, "proposed future control problem", ha="center", color="#D55E00", transform=ax.transAxes)
    ax.text(.50, .06, "current preliminary experiment: endpoint states sampled directly", ha="center", transform=ax.transAxes)
    fig.savefig(figures / "acquire_transfer_secure_free_concept.pdf"); plt.close(fig)


def main() -> int:
    phase2r, _ = load_phase2r_config()
    formal_path = _latest(ROOT / phase2r.output_dir / "formal", "trials.jsonl")
    matched_path = _latest(ROOT / phase2r.output_dir / "matching", "matched_states.jsonl")
    rows, states = _jsonl(formal_path), _jsonl(matched_path)
    rows = [row for row in rows if not row.get("pilot_only") and not row.get("calibration_only")]
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
        eligibility[group] = {"eligible_states": eligible, "state_count": len(group_states), "fraction": eligible / len(group_states), "wilson_95_CI": _wilson(eligible, len(group_states))}
    conditional = {
        group: _rate([row for row in rows if row["grasp_state_type"] == group and row["second_grasp_digit_eligible"] and row["B_geometrically_reachable"]])
        for group in GROUPS
    }
    frame = pd.DataFrame([{
        "both_retained": int(row["outcome"] == "BOTH_RETAINED"), "palmar": int(row["grasp_state_type"] == "PALMAR_SECURED"),
        "matched_pair_id": row["matched_pair_id"], "ferrari_canny_epsilon": row["ferrari_canny_epsilon"],
        "A_translation_drift_m": row["A_translation_drift_m"], "A_rotation_drift_rad": row["A_rotation_drift_rad"],
        "minimum_joint_margin_rad": row["minimum_joint_margin_rad"],
    } for row in rows if row["outcome"] != "INVALID"])
    regression = {"primary": _regression(frame, False), "adjusted": _regression(frame, True)}
    unique_state = {row["grasp_state_id"]: row for row in states}
    epsilon = np.asarray([row["ferrari_canny_epsilon"] for row in rows if row["outcome"] != "INVALID"])
    outcome = np.asarray([row["outcome"] == "BOTH_RETAINED" for row in rows if row["outcome"] != "INVALID"], dtype=float)
    corr = pointbiserialr(outcome, epsilon) if len(np.unique(outcome)) > 1 else None
    state_eps = np.asarray([row["ferrari_canny_epsilon"] for row in states])
    cutoff = float(np.quantile(state_eps, .9))
    top = [row for row in rows if row["outcome"] != "INVALID" and unique_state[row["grasp_state_id"]]["ferrari_canny_epsilon"] >= cutoff]
    ferrari = {
        "matched_group_distributions": {group: _describe(
            [row["ferrari_canny_epsilon"] for row in states if row["grasp_state_type"] == group]
        ) for group in GROUPS},
        "top_decile_cutoff": cutoff,
        "top_decile_success_rate": np.mean([row["outcome"] == "BOTH_RETAINED" for row in top]) if top else None,
        "full_population_success_rate": np.mean(outcome),
        "point_biserial_correlation": None if corr is None else {"r": float(corr.statistic), "p_value": float(corr.pvalue)},
    }
    resource_success = {}
    for component in ("occupied_finger_count", "free_finger_workspace_vol_m3", "free_palm_volume_m3"):
        values = np.asarray([row["resource_components"][component] for row in rows if row["outcome"] != "INVALID"], dtype=float)
        if len(np.unique(outcome)) < 2 or np.std(values) == 0:
            resource_success[component] = {"status": "NOT_IDENTIFIABLE", "reason": "requires both outcome classes and component variation"}
        else:
            association = pointbiserialr(outcome, values)
            resource_success[component] = {"point_biserial_r": float(association.statistic), "p_value": float(association.pvalue)}
    failure = {
        group: {
            "outcomes": {outcome_name: sum(row["outcome"] == outcome_name for row in rows if row["grasp_state_type"] == group) for outcome_name in PHASE2R_OUTCOMES},
            "subreasons": dict(Counter(str(row.get("outcome_subreason")) for row in rows if row["grasp_state_type"] == group)),
        }
        for group in GROUPS
    }
    results = {
        "experiment_id": phase2r.experiment_id,
        "formal_trials": len(rows), "valid_trials": sum(row["outcome"] != "INVALID" for row in rows),
        "primary_comparison": comparison,
        "paired_analysis": _paired(rows, phase2r.second_grasp.formal_seed),
        "eligibility": eligibility,
        "conditional_eligible_reachable": conditional,
        "clustered_logistic_regression": regression,
        "resource_component_comparisons": _state_comparisons(states),
        "secondary_resource_vs_success": resource_success,
        "ferrari_canny_baseline": ferrari,
        "failure_modes": failure,
        "scalar_J": None,
    }
    output = formal_path.parent
    (output / "analysis_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    figures = ROOT / "docs" / "figures" / "phase2R"; figures.mkdir(parents=True, exist_ok=True)
    _plot_main(states, results, figures); _plot_concept(figures)
    report = ROOT / "docs" / "PHASE2R_PALMAR_VS_FINGERTIP_RESULTS.md"
    report.write_text(
        "# Phase 2R palmar-secured versus fingertip results\n\n"
        "## Interpretation boundary\n\n"
        "The previous Phase 2 state was an acquisition-state grasp. Phase 2R directly initializes and validates a post-transfer palmar-secured endpoint for comparison; no transfer dynamics are simulated and no weld/equality constraint is present during validation or formal trials. Physics and thresholds remain frozen.\n\n"
        "The experiment does not demonstrate the control process required to transfer an object from fingertip acquisition to palmar secure storage. It evaluates whether the post-transfer endpoint state provides greater capability for subsequent acquisition.\n\n"
        "## Endpoint definitions and physical validation\n\n"
        "FINGERTIP states replay accepted Phase 2 acquisition grasps and require at least two participating finger contacts, no persistent palm contact, no table support, and the frozen unsupported-hold gate. PALMAR_SECURED states are sampled directly in the existing palm region, require persistent physical palm contact for at least 80% of the stable window, one or two load-bearing fingers at the frozen 0.20 N threshold, no table support, and the same unsupported-hold gate.\n\n"
        "Palmar candidates use a temporary free-joint pose fixture only during initialization and closure. The fixture is removed before validation; accepted states have a free object joint, zero equality constraints, and retention only through palm/finger contact, friction, gravity, and unchanged MuJoCo dynamics.\n\n"
        "The FINGERTIP filter accepted 221 of 227 replayed states. The deterministic PALMAR_SECURED search reached 150 accepted states after 4,672 of the authorized 30,000 attempts without relaxing criteria. Palmar load-bearing topologies were thumb 76, middle 13, middle+thumb 53, ring+thumb 6, and middle+ring 2.\n\n"
        "## Dataset, matching, and freezes\n\n"
        f"Validated endpoint states: FINGERTIP 221 and PALMAR_SECURED 150. Formal matching used 100 non-reused pairs after reserving 20+20 calibration states. See `PHASE2R_MATCHING_REPORT.md`. The common B region and generic controller were frozen before formal outcomes; see `PHASE2R_B_DISTRIBUTION_FREEZE.md` and `PHASE2R_CONTROLLER_FREEZE.md`.\n\n"
        "Matching used standardized Ferrari–Canny epsilon, total A normal force, translation drift, rotation drift, and minimum joint margin only. It did not use B outcomes or the resource variables hypothesized to differ between endpoint states. Residual matching imbalance is disclosed below and in the matching report.\n\n"
        "## Common B distribution and controller calibration\n\n"
        "Geometry-only selection froze the Phase 2.6 `index_thumb_region` for both groups: x=[0.0453599871, 0.0473599871] m, y=[0.0840990493, 0.0860990493] m, z=[0.2239900112, 0.2259900112] m, and yaw=[-0.1, 0.1] rad. Matched-state geometric access was 0.19 for FINGERTIP and 0.83585 for PALMAR_SECURED, with zero initial A-overlap pairs in either group.\n\n"
        "The separate 20+20-state calibration evaluated three existing Phase 2.6 controller families with five B seeds per state (200 planned records per candidate). Candidate 01 produced 0 BOTH_RETAINED, candidate 02 produced 2 across one A state and two B seeds, and candidate 03 produced 0. The pooled lexicographic rule selected `phase2_6_b_only_02`; no group-rate difference was used and no trajectory-search expansion was needed. The fixed acquisition digits are assigned once by geometry before motion and never reassigned.\n\n"
        "## Formal paired experiment\n\n"
        "All 4,000 planned records completed: 100 matched pairs × two endpoint types × 20 shared formal B seeds. Records are deterministic, incremental, resumable, and use the separate `phase2R_palmar_vs_fingertip_formal` experiment ID. Ineligible states are recorded as B_NOT_ACQUIRED with `INSUFFICIENT_FREE_DIGITS_PRECHECK` without a meaningless dynamic attempt.\n\n"
        "## Eligibility and formal results\n\n```json\n" + json.dumps({"eligibility": eligibility, "primary": comparison, "conditional_eligible_reachable": conditional}, indent=2) + "\n```\n\n"
        "The conditional eligible-and-reachable FINGERTIP success rate is not identifiable because no FINGERTIP trial entered that analysis set; it must not be interpreted as zero.\n\n"
        "## Paired comparison\n\n```json\n" + json.dumps(results["paired_analysis"], indent=2) + "\n```\n\n"
        "## Clustered models\n\nThe adjusted model includes only the prespecified baseline stability covariates and does not control away digit occupancy, workspace, palm contact, COM position, or free-palm volume. Because FINGERTIP had zero successes, both logistic fits exhibit complete or quasi-complete separation and very large coefficient uncertainty. These numerical fits are unstable and are not the primary inferential result; the prespecified paired exact comparison is primary.\n\n```json\n" + json.dumps(regression, indent=2) + "\n```\n\n"
        "## Resource distributions and Ferrari–Canny baseline\n\n```json\n" + json.dumps({"resources": results["resource_component_comparisons"], "Ferrari_Canny": ferrari}, indent=2) + "\n```\n\n"
        "## Failure modes\n\n```json\n" + json.dumps(failure, indent=2) + "\n```\n\n"
        "## Limitations\n\nDirect endpoint initialization does not establish transfer controllability. Scripted B acquisition probes one frozen controller and region. The matched groups retain residual imbalance in baseline translation drift, rotation drift, and minimum joint margin, reported transparently in the matching report. No scalar J, wrist controller, three-object task, or RL policy is defined.\n",
        encoding="utf-8",
    )
    paired_p = results["paired_analysis"]["McNemar_exact_two_sided_p_value"]
    if ps + fs == 0:
        evidence_case = "C: dynamically unidentifiable because both groups had zero BOTH_RETAINED trials."
    elif ps / pn > fs / fn and paired_p is not None and paired_p < 0.05:
        evidence_case = "A: PALMAR_SECURED outperformed FINGERTIP in the paired formal experiment."
    else:
        evidence_case = "B: no statistically detectable paired difference was established."
    preliminary = ROOT / "docs" / "PHASE2R_PRELIMINARY_EVIDENCE.md"
    preliminary.write_text(
        "# Phase 2R proposal-ready preliminary evidence\n\n"
        f"**Observed evidence classification:** {evidence_case}\n\n"
        f"The formal experiment completed {len(rows)} trials. FINGERTIP achieved {fs}/{fn} valid-trial successes; PALMAR_SECURED achieved {ps}/{pn}. The absolute difference was {comparison['absolute_percentage_point_difference_palmar_minus_fingertip']:.4f} percentage points, with pair-bootstrap 95% CI {results['paired_analysis']['absolute_rate_difference_bootstrap_95_CI']}. McNemar's exact paired p-value was {paired_p}.\n\n"
        f"Digit eligibility was {eligibility['FINGERTIP']['eligible_states']}/{eligibility['FINGERTIP']['state_count']} for FINGERTIP and {eligibility['PALMAR_SECURED']['eligible_states']}/{eligibility['PALMAR_SECURED']['state_count']} for PALMAR_SECURED.\n\n"
        "These data evaluate directly sampled endpoint states. They do not demonstrate the transfer controller needed to reach a palmar-secured state from a fingertip acquisition state.\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
