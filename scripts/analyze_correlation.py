#!/usr/bin/env python
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "seqgrasp-matplotlib"))
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint

from seqgrasp.config import ROOT
from seqgrasp.experiments.resource_components import RESOURCE_METHOD_ID, RESOURCE_RECORDS_FILENAME, reconstruct_grasp
from seqgrasp.experiments.second_grasp import OUTCOMES
from seqgrasp.phase2_config import load_phase2_config


PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]
COMPONENTS = ["occupied_finger_count", "free_finger_workspace_vol_m3", "free_palm_volume_m3"]


def _dataset_dir(root: Path) -> Path:
    candidates = []
    for path in root.glob("*/accepted_grasps.jsonl"):
        candidates.append((len(path.read_text(encoding="utf-8").splitlines()), path.parent))
    if not candidates:
        raise FileNotFoundError("no grasp dataset found")
    return max(candidates, key=lambda item: item[0])[1]


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _flatten_trials(rows: list[dict]) -> pd.DataFrame:
    flat = []
    for row in rows:
        item = {key: row.get(key) for key in ("trial_id", "grasp_id", "outcome", "ferrari_canny_epsilon")}
        item.update(row["resource_components"])
        item["both_retained"] = int(row["outcome"] == "BOTH_RETAINED")
        flat.append(item)
    return pd.DataFrame(flat)


def _wilson(successes: int, count: int, alpha: float) -> tuple[float, float]:
    if count == 0:
        return float("nan"), float("nan")
    low, high = proportion_confint(successes, count, alpha=alpha, method="wilson")
    return float(low), float(high)


def _binned(df: pd.DataFrame, component: str, bins: int, alpha: float) -> list[dict]:
    if component == "occupied_finger_count":
        categories = df[component].astype(str)
    else:
        distinct = df[component].nunique()
        if distinct >= bins:
            categories = pd.qcut(df[component], q=bins, duplicates="drop")
        else:
            categories = df[component].astype(str)
    result = []
    for category, group in df.groupby(categories, observed=True):
        successes, count = int(group["both_retained"].sum()), len(group)
        low, high = _wilson(successes, count, alpha)
        result.append({
            "category": str(category), "x_mean": float(group[component].mean()), "successes": successes,
            "count": count, "rate": successes / count, "wilson_low": low, "wilson_high": high,
        })
    return result


def _fit_logit(valid: pd.DataFrame, standardized: bool, clustered: bool) -> dict:
    predictors = valid[COMPONENTS].astype(float).copy()
    scale = {}
    output_multipliers = {"const": 1.0, **{key: 1.0 for key in COMPONENTS}}
    if standardized:
        for key in COMPONENTS:
            mean, std = float(predictors[key].mean()), float(predictors[key].std(ddof=0))
            scale[key] = {"mean": mean, "std": std}
            predictors[key] = (predictors[key] - mean) / std if std > 0 else 0.0
    else:
        # Optimize numerically scaled but uncentered columns, then transform
        # coefficients and standard errors exactly back to raw physical units.
        for key in COMPONENTS:
            numerical_scale = float(predictors[key].std(ddof=0))
            if numerical_scale > 0:
                predictors[key] /= numerical_scale
                output_multipliers[key] = 1.0 / numerical_scale
                scale[key] = {"optimization_divisor": numerical_scale, "reported_units": "raw physical units"}
    design = sm.add_constant(predictors, has_constant="add")
    model = sm.Logit(valid["both_retained"].astype(float), design)
    fit = (
        model.fit(disp=False, maxiter=100, cov_type="cluster", cov_kwds={"groups": valid["grasp_id"]})
        if clustered else model.fit(disp=False, maxiter=100)
    )
    interval = fit.conf_int(alpha=0.05)
    terms = {}
    for name in fit.params.index:
        multiplier = output_multipliers[name]
        terms[name] = {
            "coefficient": float(fit.params[name] * multiplier), "standard_error": float(fit.bse[name] * multiplier),
            "z": float(fit.tvalues[name]), "p_value": float(fit.pvalues[name]),
            "confidence_interval_95": [float(interval.loc[name, 0] * multiplier), float(interval.loc[name, 1] * multiplier)],
        }
    return {
        "terms": terms, "predictor_transform": scale, "N_valid_trials": int(fit.nobs),
        "McFadden_pseudo_R_squared": float(fit.prsquared), "clustered_by_grasp": clustered,
    }


def _safe_regressions(valid: pd.DataFrame) -> dict:
    if len(valid) < 10 or valid["both_retained"].nunique() < 2:
        return {"status": "NOT_IDENTIFIABLE", "reason": "requires >=10 valid trials and both outcome classes"}
    result = {}
    for name, standardized, clustered in (
        ("raw_physical_units", False, False),
        ("standardized_predictors", True, False),
        ("raw_clustered_robust", False, True),
        ("standardized_clustered_robust", True, True),
    ):
        try:
            result[name] = _fit_logit(valid, standardized, clustered)
        except Exception as exc:
            result[name] = {"status": "FIT_FAILED", "error": str(exc)}
    return result


def _plot_success(binned: dict, output: Path) -> None:
    labels = {
        "occupied_finger_count": "Occupied finger count",
        "free_finger_workspace_vol_m3": "Free-finger workspace volume [m³]",
        "free_palm_volume_m3": "Free-palm volume [m³]",
    }
    filenames = {
        "occupied_finger_count": "occupied_fingers_vs_success.pdf",
        "free_finger_workspace_vol_m3": "free_finger_workspace_vs_success.pdf",
        "free_palm_volume_m3": "free_palm_volume_vs_success.pdf",
    }
    for component, rows in binned.items():
        fig, ax = plt.subplots(figsize=(3.4, 2.7))
        x = np.arange(len(rows)) if component == "occupied_finger_count" else np.asarray([row["x_mean"] for row in rows])
        rate = np.asarray([row["rate"] for row in rows])
        low = np.asarray([row["wilson_low"] for row in rows])
        high = np.asarray([row["wilson_high"] for row in rows])
        ax.errorbar(x, rate, yerr=[rate - low, high - rate], fmt="o-", color=PALETTE[0], capsize=3)
        if component == "occupied_finger_count":
            ax.set_xticks(x, [row["category"] for row in rows])
        ax.set(xlabel=labels[component], ylabel="BOTH_RETAINED rate", ylim=(0, 1))
        fig.savefig(output / filenames[component])
        plt.close(fig)


def _plot_failure_modes(df: pd.DataFrame, output: Path, bins: int) -> dict:
    result = {}
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.0))
    for ax, component in zip(axes, COMPONENTS):
        if component == "occupied_finger_count":
            category = df[component].astype(str)
        else:
            category = pd.qcut(df[component], q=min(4, max(1, df[component].nunique())), duplicates="drop")
        table = pd.crosstab(category, df["outcome"], normalize="index").reindex(columns=OUTCOMES, fill_value=0)
        table.plot(kind="bar", stacked=True, ax=ax, color=PALETTE, legend=False, width=0.85)
        ax.set(xlabel=component.replace("_", " "), ylabel="Outcome proportion", ylim=(0, 1))
        ax.tick_params(axis="x", rotation=25)
        result[component] = {str(index): {outcome: float(table.loc[index, outcome]) for outcome in OUTCOMES} for index in table.index}
    fig.legend(OUTCOMES, loc="upper center", ncol=5, frameon=False)
    fig.subplots_adjust(top=0.78)
    fig.savefig(output / "outcomes_by_resource_component.pdf")
    plt.close(fig)
    return result


def _plot_resource_histograms(resources: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.7))
    for ax, component in zip(axes, COMPONENTS):
        ax.hist([row[component] for row in resources], bins="auto", color=PALETTE[0], alpha=0.85)
        ax.set(xlabel=component.replace("_", " "), ylabel="Grasp count")
    fig.savefig(output / "resource_component_histograms.pdf")
    plt.close(fig)


def _render_grasp(record: dict) -> np.ndarray:
    _, model, data, _ = reconstruct_grasp(record)
    renderer = mujoco.Renderer(model, height=360, width=360)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object_a")]
    camera.distance, camera.azimuth, camera.elevation = 0.32, 140, -22
    renderer.update_scene(data, camera=camera)
    image = renderer.render().copy()
    renderer.close()
    return image


def _representative_figure(accepted: list[dict], output: Path) -> str:
    if not accepted:
        return "not generated: no accepted grasps"
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 7.2))
    for row_index, component in enumerate(COMPONENTS):
        ordered = sorted(accepted, key=lambda row: row[component] if component in row else row["occupied_finger_count"])
        for column, (label, quantile) in enumerate((("low", 0.0), ("middle", 0.5), ("high", 1.0))):
            record = ordered[round(quantile * (len(ordered) - 1))]
            axes[row_index, column].imshow(_render_grasp(record))
            axes[row_index, column].set_xlabel(f"{label}: {record.get(component, float('nan')):.4g}")
            axes[row_index, column].set_xticks([]); axes[row_index, column].set_yticks([])
        axes[row_index, 0].set_ylabel(component.replace("_", " "))
    fig.savefig(output / "representative_actual_grasps.pdf")
    plt.close(fig)
    return "docs/figures/phase2/representative_actual_grasps.pdf"


def main() -> int:
    phase2, config_path = load_phase2_config()
    dataset_dir = _dataset_dir(ROOT / phase2.persistence.output_dir / "grasp_dataset")
    accepted = _jsonl(dataset_dir / "accepted_grasps.jsonl")
    resources = _jsonl(dataset_dir / RESOURCE_RECORDS_FILENAME)
    trials = _jsonl(dataset_dir / "correlation" / "trials.jsonl")
    resource_by_id = {row["grasp_id"]: row for row in resources}
    enriched_accepted = [{**row, **resource_by_id.get(row["grasp_id"], {})} for row in accepted if row["grasp_id"] in resource_by_id]
    figures = ROOT / "docs" / "figures" / "phase2"
    figures.mkdir(parents=True, exist_ok=True)
    plt.style.use(ROOT / "configs" / "phase2_publication.mplstyle")
    df = _flatten_trials(trials)
    valid = df[df["outcome"] != "INVALID"].copy() if not df.empty else df.copy()
    alpha = 1 - phase2.analysis.confidence_level
    binned = {component: _binned(valid, component, phase2.analysis.continuous_quantile_bins, alpha) for component in COMPONENTS} if not valid.empty else {}
    if binned:
        _plot_success(binned, figures)
        failure_modes = _plot_failure_modes(df, figures, phase2.analysis.continuous_quantile_bins)
    else:
        failure_modes = {}
    if resources:
        _plot_resource_histograms(resources, figures)
    representative_status = "not generated: resource components unavailable"
    if enriched_accepted:
        try:
            representative_status = _representative_figure(enriched_accepted, figures)
        except Exception as exc:
            representative_status = f"render failed: {exc}"
    convergence_source = dataset_dir / f"workspace_convergence_{RESOURCE_METHOD_ID}.json"
    if convergence_source.exists():
        convergence = json.loads(convergence_source.read_text(encoding="utf-8"))
        if convergence:
            fig, ax = plt.subplots(figsize=(3.5, 2.8))
            for grasp_id, group in pd.DataFrame(convergence).groupby("grasp_id"):
                ax.plot(group["samples"], group["volume_m3"], marker="o", alpha=0.6)
            ax.set(xlabel="Monte Carlo samples", ylabel="Workspace volume [m³]")
            fig.savefig(figures / "workspace_convergence.pdf")
            plt.close(fig)
    regression = _safe_regressions(valid) if not valid.empty else {"status": "NOT_IDENTIFIABLE", "reason": "no trials"}
    outcomes = dict(Counter(row["outcome"] for row in trials))
    overall_success = (outcomes.get("BOTH_RETAINED", 0) / len(valid)) if len(valid) else None
    grasp_quality = {}
    if trials:
        epsilon = pd.Series({row["grasp_id"]: row["ferrari_canny_epsilon"] for row in trials})
        cutoff = epsilon.quantile(1 - phase2.analysis.greedy_top_fraction)
        top_ids = set(epsilon[epsilon >= cutoff].index)
        top = valid[valid["grasp_id"].isin(top_ids)]
        remainder = valid[~valid["grasp_id"].isin(top_ids)]
        grasp_quality = {
            "top_decile_epsilon_cutoff": float(cutoff), "top_decile_grasps": len(top_ids),
            "top_decile_BOTH_RETAINED_rate": float(top["both_retained"].mean()) if len(top) else None,
            "full_population_rate": overall_success,
            "remaining_90_percent_rate": float(remainder["both_retained"].mean()) if len(remainder) else None,
        }
    preflight_path = dataset_dir / "correlation" / "preflight" / "geometry_preflight_summary.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8")) if preflight_path.exists() else {}
    baseline_path = ROOT / phase2.persistence.output_dir / "physics_validation" / "physics_validation_summary.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else {}
    dataset_summary_path = dataset_dir / "dataset_summary.json"
    dataset_summary = json.loads(dataset_summary_path.read_text(encoding="utf-8")) if dataset_summary_path.exists() else {}
    epsilon_values = np.asarray([row["ferrari_canny_epsilon"] for row in accepted], dtype=float)
    grasp_statistics = {
        "accepted": len(accepted),
        "candidate_attempts": dataset_summary.get("candidate_attempts"),
        "commanded_subset_distribution": dict(Counter("+".join(row["commanded_finger_subset"]) for row in accepted)),
        "occupied_finger_count_distribution": dict(Counter(str(row["occupied_finger_count"]) for row in accepted)),
        "ferrari_canny_epsilon": {
            "min": float(np.min(epsilon_values)), "max": float(np.max(epsilon_values)),
            "mean": float(np.mean(epsilon_values)), "std": float(np.std(epsilon_values)),
        } if len(epsilon_values) else {},
    }
    figure_paths = sorted(path.relative_to(ROOT).as_posix() for path in figures.glob("*.pdf"))
    results = {
        "dataset_dir": dataset_dir.relative_to(ROOT).as_posix(),
        "planned_trials": phase2.required_for_later_parts.accepted_grasp_target * phase2.second_grasp.trials_per_grasp,
        "completed_trials": len(trials), "valid_trials": len(valid), "invalid_trials": outcomes.get("INVALID", 0),
        "outcome_counts": outcomes, "BOTH_RETAINED_rate_valid": overall_success,
        "success_bins_with_95_percent_Wilson_intervals": binned,
        "logistic_regression": regression, "failure_modes": failure_modes,
        "greedy_Ferrari_Canny_baseline": grasp_quality,
        "grasp_dataset_statistics": grasp_statistics,
        "B_geometry_preflight": preflight,
        "baseline_physics": baseline,
        "figure_paths": figure_paths,
        "representative_render_status": representative_status,
        "scalar_J": None,
    }
    (dataset_dir / "correlation" / "analysis_results.json").parent.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "correlation" / "analysis_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    report = ROOT / "docs" / "PHASE2_RESOURCE_CORRELATION_RESULTS.md"
    sweep_report = (ROOT / "docs" / "PHASE2_PHYSICS_SWEEP.md").read_text(encoding="utf-8") if (ROOT / "docs" / "PHASE2_PHYSICS_SWEEP.md").exists() else "Unavailable."
    resource_summary_path = dataset_dir / f"resource_summary_{RESOURCE_METHOD_ID}.json"
    resource_summary = json.loads(resource_summary_path.read_text(encoding="utf-8")) if resource_summary_path.exists() else {}
    execution_status = (
        "PHASE2_B_WORKSPACE_GEOMETRY_BLOCKED" if preflight.get("status") == "PHASE2_B_WORKSPACE_GEOMETRY_BLOCKED"
        else ("complete" if len(trials) == results["planned_trials"] else "resumable batch incomplete")
    )
    resume_text = (
        "No correlation-batch resume command is authorized under the current config: the required geometry preflight returned `PHASE2_B_WORKSPACE_GEOMETRY_BLOCKED`. PI input changing the B-placement geometry would be required before a new experiment namespace may be run."
        if execution_status == "PHASE2_B_WORKSPACE_GEOMETRY_BLOCKED" else
        "Resume with `python scripts/build_grasp_dataset.py --workers 8`, then `python scripts/compute_resource_components.py --workers 8`, then `python scripts/run_correlation_experiment.py --workers 8`, and rerun this script."
    )
    report.write_text(
        "# Phase 2 resource-component correlation results\n\n"
        f"> Status: {execution_status}. No scalar J is defined or evaluated.\n\n"
        "## Part A physics gate and sensitivity\n\n"
        "The original baseline passed the PI-supplied hard gate and was retained a priori. The 81-condition sweep is a sensitivity study only; it did not select production physics. Baseline measurements and checks:\n\n"
        f"```json\n{json.dumps(baseline, indent=2)}\n```\n\n"
        + sweep_report + "\n\n"
        "## Parts B, C, and E\n\n"
        f"Accepted-grasp generation statistics:\n\n```json\n{json.dumps(grasp_statistics, indent=2)}\n```\n\n"
        f"Resource records: {len(resources)}. Raw component and convergence summary:\n\n```json\n{json.dumps(resource_summary, indent=2)}\n```\n\n"
        "Occupied fingers use summed A normal force >0.20 N. Free-finger workspace uses 10,000 Monte Carlo joint samples, actual MuJoCo collision geometry, and 0.005 m voxels. Free-palm volume uses the supplied palm-frame AABB and actual box/capsule collision geometry. The components are not combined.\n\n"
        "The three unnormalised tactile features per finger are: binary contact (>0.05 N), total normal force [N], and tangential/normal force ratio. Ratio zero at zero normal force means no slip-proxy signal, not a physical loaded ratio of zero.\n\n"
        "## Parts D and F\n\n"
        f"The B centre distribution is x={phase2.second_grasp.B_center_x_bounds_m} m, y={phase2.second_grasp.B_center_y_bounds_m} m, table-resting z from actual cylinder geometry, and uniform yaw={phase2.second_grasp.B_yaw_bounds_rad} rad. Geometry preflight:\n\n```json\n{json.dumps(preflight, indent=2)}\n```\n\n"
        f"Completed {len(trials)} / {results['planned_trials']} trials. Outcome counts: `{json.dumps(outcomes, sort_keys=True)}`. BOTH_RETAINED rate among valid trials: `{overall_success}`.\n\n"
        "Binned intervals are 95% Wilson binomial intervals. Continuous components use five equal-frequency bins when enough distinct values exist; occupied count uses integer categories. INVALID is reported separately and excluded from the primary logistic model.\n\n"
        f"```json\n{json.dumps(results, indent=2)}\n```\n\n"
        "## Interpretation and limitations\n\n"
        "These analyses test association between raw resource components and sequential acquisition outcomes. They do not establish causality and do not claim that J predicts success; J remains PI-blocked. Incomplete batches must not be interpreted as final estimates.\n\n"
        "The prescribed 10,000-sample workspace budget was retained even though the representative convergence study still changed materially between 5,000, 10,000, and 20,000 samples. This limitation is reported rather than used to tune the production budget.\n\n"
        "## Remaining PI decisions\n\n"
        "Every active scientific placeholder is enumerated with file and line in `docs/PI_DECISIONS.md`. Scalar J, general task transition/drop criteria, reward design, and closed-loop retention remain unresolved.\n\n"
        "## Reproducibility\n\n"
        f"Phase-2 config: `{config_path.relative_to(ROOT).as_posix()}`. Dataset/config hashes and source git SHAs are stored in every incremental JSONL record. {resume_text}\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2))
    print(f"report: {report.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
