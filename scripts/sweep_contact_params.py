#!/usr/bin/env python
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path

from seqgrasp.config import ROOT
from seqgrasp.experiments.metadata import config_hash, git_commit_sha
from seqgrasp.experiments.physics_validation import parameter_sensitivity, run_physics_validation, validation_bundle, validation_config_paths
from seqgrasp.experiments.resumable import IncrementalJsonlStore, stable_trial_id
from seqgrasp.phase2_config import load_phase2_config, missing_contact_sweep_inputs
from seqgrasp.scene_builder import ContactParameterOverride


def _write_report(path: Path, status: str, rows: list[dict], missing: list[str]) -> None:
    lines = [
        "# Phase 2 Contact-Parameter Sweep",
        "",
        "The sweep is diagnostic and does not automatically select a physics configuration.",
        "",
        f"Status: **{status}**",
        "",
    ]
    if missing:
        lines.extend([
            "No sweep was executed because the following PI inputs are absent:",
            "",
            *[f"- `{name}`" for name in missing],
            "",
            "Consequently, no strongest-effect statement or configuration selection is available.",
        ])
    else:
        lines.extend([
            "| Rank | Parameters | Gate | Force [N] | Penetration [m] | Translation drift [m] | Numerical |",
            "|---:|---|---|---:|---:|---:|---|",
        ])
        for index, row in enumerate(rows, 1):
            measure = row["summary"]["measurements"]
            lines.append(
                f"| {index} | `{json.dumps(row['parameters'], sort_keys=True)}` | {row['summary']['verdict']} "
                f"| {measure['mean_total_normal_force_N']:.6g} | {measure['maximum_penetration_m']:.6g} "
                f"| {measure['maximum_translational_drift_m']:.6g} | {measure['numerical_validity']} |"
            )
        lines.extend([
            "",
            "Rows are ordered for inspection by numerical validity, configured gate state, penetration, and drift. This ordering is not a final physics selection.",
            "",
            "## Descriptive sensitivity",
            "",
            "Between-level spans compare the mean measurement at each configured level. They do not select a configuration.",
            "",
            "| Parameter | Force span [N] | Penetration span [m] | Translation span [m] | Rotation span [rad] |",
            "|---|---:|---:|---:|---:|",
        ])
        sensitivity = parameter_sensitivity(rows)
        for parameter, values in sensitivity.items():
            span = values["between_level_span"]
            lines.append(
                f"| {parameter} | {span['mean_total_normal_force_N']:.6g} "
                f"| {span['maximum_penetration_m']:.6g} | {span['maximum_translational_drift_m']:.6g} "
                f"| {span['maximum_orientation_drift_rad']:.6g} |"
            )
        metrics = (
            "mean_total_normal_force_N", "maximum_penetration_m",
            "maximum_translational_drift_m", "maximum_orientation_drift_rad",
        )
        strongest = {
            metric: max(sensitivity, key=lambda parameter: sensitivity[parameter]["between_level_span"][metric])
            for metric in metrics
        }
        lines.extend(["", "Largest descriptive spans: " + ", ".join(f"{metric} -> {parameter}" for metric, parameter in strongest.items()) + "."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the resumable Phase 2 contact-parameter sweep")
    parser.add_argument("--config", default="configs/phase2_physics_validation.yaml")
    args = parser.parse_args()
    phase2, config_path = load_phase2_config(ROOT / args.config)
    output = ROOT / phase2.persistence.output_dir / "contact_sweep"
    output.mkdir(parents=True, exist_ok=True)
    missing = missing_contact_sweep_inputs(phase2.sweep)
    status_path = output / "sweep_status.json"
    report_path = ROOT / "docs" / "PHASE2_PHYSICS_SWEEP.md"
    if missing:
        payload = {
            "status": "PI_INPUT_REQUIRED",
            "missing_pi_inputs": missing,
            "seed": phase2.validation.seed,
            "config_hash": config_hash([config_path]),
            "git_commit_sha": git_commit_sha(ROOT),
        }
        status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _write_report(report_path, payload["status"], [], missing)
        print(json.dumps(payload, indent=2))
        return 0

    _, _, profile_path = validation_bundle(phase2)
    metadata_hash = config_hash(validation_config_paths(config_path, profile_path))
    store = IncrementalJsonlStore(
        output / "trials.jsonl",
        phase2.persistence.lock_timeout_seconds,
        phase2.persistence.lock_poll_seconds,
    )
    completed = store.completed_ids()
    combinations = product(
        phase2.sweep.friction_vectors,
        phase2.sweep.solref_values,
        phase2.sweep.solimp_values,
        phase2.sweep.timestep_values_s,
    )
    for friction, solref, solimp, timestep in combinations:
        parameters = {
            "target_geom_names": phase2.sweep.target_geom_names,
            "friction": friction,
            "solref": solref,
            "solimp": solimp,
            "timestep_s": timestep,
        }
        identity = {"seed": phase2.validation.seed, "config_hash": metadata_hash, "parameters": parameters}
        trial_id = stable_trial_id("phase2-contact-sweep", identity)
        if trial_id in completed:
            continue
        override = ContactParameterOverride(
            geom_names=tuple(phase2.sweep.target_geom_names),
            friction=tuple(friction),
            solref=tuple(solref),
            solimp=tuple(solimp),
            timestep=timestep,
        )
        _, summary = run_physics_validation(
            phase2,
            config_path,
            output / "trial_summaries" / trial_id.split(":", 1)[1],
            contact_override=override,
            write_plot=False,
        )
        store.append({
            "trial_id": trial_id,
            "seed": phase2.validation.seed,
            "config_hash": metadata_hash,
            "git_commit_sha": git_commit_sha(ROOT),
            "parameters": parameters,
            "summary": summary,
        })
    rows = store.records()
    verdict_order = {"PASS": 0, "PI_INPUT_REQUIRED": 1, "FAIL": 2}
    ranked = sorted(
        rows,
        key=lambda row: (
            not row["summary"]["measurements"]["numerical_validity"],
            verdict_order[row["summary"]["verdict"]],
            row["summary"]["measurements"]["maximum_penetration_m"],
            row["summary"]["measurements"]["maximum_translational_drift_m"],
        ),
    )
    _write_report(report_path, "COMPLETE_AWAITING_PI_SELECTION", ranked, [])
    print(f"completed sweep trials: {len(ranked)}")
    for index, row in enumerate(ranked, 1):
        measure = row["summary"]["measurements"]
        print(index, row["summary"]["verdict"], measure["mean_total_normal_force_N"], measure["maximum_penetration_m"], row["parameters"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
