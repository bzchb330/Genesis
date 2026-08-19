from __future__ import annotations

from dataclasses import asdict, replace
import csv
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from ..config import ROOT, ConfigBundle, DiagnosticProfile, load_configs
from ..diagnostics.multi_grasp import bundle_for_profile, load_grasp_profile
from ..diagnostics.scripted_grasp import DiagnosticRun, run_scripted_grasp
from ..phase2_config import Phase2Config, PhysicsValidationConfig
from ..scene_builder import ContactParameterOverride
from .metadata import config_hash, git_commit_sha


def validation_config_paths(config_path: Path, profile_path: Path) -> list[Path]:
    return [
        config_path,
        profile_path,
        ROOT / "configs" / "hand_allegro.yaml",
        ROOT / "configs" / "scene_two_object.yaml",
        ROOT / "configs" / "task_sequential.yaml",
        ROOT / "configs" / "diagnostic_grasp_a.yaml",
    ]


def validation_bundle(
    phase2: Phase2Config,
    base: ConfigBundle | None = None,
    timestep: float | None = None,
) -> tuple[ConfigBundle, DiagnosticProfile, Path]:
    base = load_configs() if base is None else base
    profile_path = (ROOT / phase2.validation.grasp_profile_path).resolve()
    _, profile = load_grasp_profile(profile_path)
    effective_timestep = base.scene.timestep if timestep is None else timestep
    durations = dict(profile.stage_durations_seconds)
    durations["hold"] = phase2.validation.long_hold_steps * effective_timestep
    profile = replace(profile, stage_durations_seconds=durations)
    return bundle_for_profile(base, "phase2_physics_validation", profile), profile, profile_path


def _orientation_distance(quaternions: np.ndarray, reference: np.ndarray) -> np.ndarray:
    dots = np.abs(quaternions @ reference)
    return 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))


def summarize_physics_run(run: DiagnosticRun, cfg: PhysicsValidationConfig) -> dict:
    arrays = run.arrays
    hold = np.flatnonzero(arrays["diagnostic_stage"] == "hold")
    if hold.size < cfg.long_hold_steps:
        return {
            "verdict": "FAIL",
            "reason": "long hold did not complete",
            "completed_hold_steps": int(hold.size),
            "required_hold_steps": cfg.long_hold_steps,
            "missing_pi_inputs": [],
        }

    hold = hold[-cfg.long_hold_steps:]
    positions = arrays["object_position"][hold]
    orientations = arrays["object_orientation"][hold]
    forces = arrays["finger_object_normal_force_raw"][hold]
    counts = arrays["finger_object_contact_count"][hold]
    distances = arrays["finger_object_contact_distance_m"][hold]
    penetration = np.where(counts > 0, np.maximum(0.0, -distances), 0.0)
    displacement = positions - positions[0]
    translational = np.linalg.norm(displacement, axis=1)
    orientation = _orientation_distance(orientations, orientations[0])
    totals = forces.sum(axis=1)
    contact_counts = counts.sum(axis=1)
    numeric_keys = (
        "object_position", "object_orientation", "object_linear_velocity",
        "object_angular_velocity", "joint_positions", "joint_velocities",
        "actuator_controls", "finger_object_normal_force_raw",
    )
    numerical_validity = all(np.all(np.isfinite(arrays[key][hold])) for key in numeric_keys)
    force_lower = cfg.expected_force_order_N / cfg.force_order_factor
    force_upper = cfg.expected_force_order_N * cfg.force_order_factor
    mean_total_force = float(np.mean(totals))
    force_sanity = force_lower <= mean_total_force <= force_upper
    table_recontact = bool(np.any(arrays["object_table_contact"][hold] > 0))
    complete_loss = bool(np.any(arrays["active_object_finger_count"][hold] == 0))

    missing = []
    configured = {
        "penetration_tolerance_m": cfg.penetration_tolerance_m,
        "maximum_vertical_drift_m": cfg.maximum_vertical_drift_m,
        "maximum_translational_drift_m": cfg.maximum_translational_drift_m,
        "maximum_orientation_drift_rad": cfg.maximum_orientation_drift_rad,
        "minimum_active_object_contacts": cfg.minimum_active_object_contacts,
        "allow_table_recontact": cfg.allow_table_recontact,
        "allow_complete_contact_loss": cfg.allow_complete_contact_loss,
    }
    missing.extend(name for name, value in configured.items() if value is None)
    measurements = {
        "maximum_penetration_m": float(np.max(penetration)),
        "maximum_vertical_drift_m": float(np.max(np.abs(displacement[:, 2]))),
        "final_vertical_drift_m": float(displacement[-1, 2]),
        "maximum_translational_drift_m": float(np.max(translational)),
        "final_translational_drift_m": float(translational[-1]),
        "maximum_orientation_drift_rad": float(np.max(orientation)),
        "final_orientation_drift_rad": float(orientation[-1]),
        "mean_force_per_finger_N": np.mean(forces, axis=0).tolist(),
        "final_force_per_finger_N": forces[-1].tolist(),
        "mean_total_normal_force_N": mean_total_force,
        "final_total_normal_force_N": float(totals[-1]),
        "total_normal_force_std_N": float(np.std(totals)),
        "minimum_active_object_contacts": int(np.min(contact_counts)),
        "maximum_active_object_contacts": int(np.max(contact_counts)),
        "table_recontact": table_recontact,
        "complete_object_hand_contact_loss": complete_loss,
        "numerical_validity": numerical_validity,
    }

    checks: dict[str, bool | None] = {
        "force_order_of_magnitude": force_sanity,
        "penetration": None if cfg.penetration_tolerance_m is None else measurements["maximum_penetration_m"] <= cfg.penetration_tolerance_m,
        "vertical_drift": None if cfg.maximum_vertical_drift_m is None else measurements["maximum_vertical_drift_m"] <= cfg.maximum_vertical_drift_m,
        "translational_drift": None if cfg.maximum_translational_drift_m is None else measurements["maximum_translational_drift_m"] <= cfg.maximum_translational_drift_m,
        "orientation_drift": None if cfg.maximum_orientation_drift_rad is None else measurements["maximum_orientation_drift_rad"] <= cfg.maximum_orientation_drift_rad,
        "active_contacts": None if cfg.minimum_active_object_contacts is None else measurements["minimum_active_object_contacts"] >= cfg.minimum_active_object_contacts,
        "table_recontact": None if cfg.allow_table_recontact is None else cfg.allow_table_recontact or not table_recontact,
        "complete_contact_loss": None if cfg.allow_complete_contact_loss is None else cfg.allow_complete_contact_loss or not complete_loss,
        "numerical_validity": numerical_validity,
    }
    if not numerical_validity or not force_sanity:
        verdict = "FAIL"
    elif missing:
        verdict = "PI_INPUT_REQUIRED"
    else:
        verdict = "PASS" if all(bool(value) for value in checks.values()) else "FAIL"
    return {
        "verdict": verdict,
        "completed_hold_steps": int(hold.size),
        "required_hold_steps": cfg.long_hold_steps,
        "force_sanity_range_N": [force_lower, force_upper],
        "measurements": measurements,
        "checks": checks,
        "missing_pi_inputs": missing,
    }


def run_physics_validation(
    phase2: Phase2Config,
    config_path: Path,
    output_dir: str | Path,
    contact_override: ContactParameterOverride | None = None,
    write_plot: bool = True,
) -> tuple[DiagnosticRun, dict]:
    timestep = None if contact_override is None else contact_override.timestep
    bundle, _, profile_path = validation_bundle(phase2, timestep=timestep)
    run = run_scripted_grasp(
        bundle,
        seed=phase2.validation.seed,
        save_outputs=False,
        profile_name="phase2_physics_validation",
        contact_override=contact_override,
    )
    summary = summarize_physics_run(run, phase2.validation)
    summary["metadata"] = {
        "seed": phase2.validation.seed,
        "config_hash": config_hash(validation_config_paths(config_path, profile_path)),
        "git_commit_sha": git_commit_sha(ROOT),
        "fixed_base_interpretation": "palm fixed; lift replayed by removing external object support",
        "profile_path": str(profile_path.relative_to(ROOT)).replace("\\", "/"),
        "contact_override": None if contact_override is None else asdict(contact_override),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "physics_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_timeseries_csv(run, output / "physics_validation_timeseries.csv")
    if write_plot and "measurements" in summary:
        _plot_validation(run, phase2.validation.long_hold_steps, output / "physics_validation.pdf")
    return run, summary


def _write_timeseries_csv(run: DiagnosticRun, path: Path) -> None:
    arrays = run.arrays
    fingers = run.metadata["finger_order"]
    fieldnames = ["time_s", "stage"]
    fieldnames.extend(f"normal_force_{finger}_N" for finger in fingers)
    fieldnames.extend([
        "total_object_normal_force_N", "active_object_contact_count",
        "maximum_penetration_m", "object_x_m", "object_y_m", "object_z_m",
        "object_qw", "object_qx", "object_qy", "object_qz",
        "object_vx_m_s", "object_vy_m_s", "object_vz_m_s",
        "object_wx_rad_s", "object_wy_rad_s", "object_wz_rad_s",
        "object_table_contact",
    ])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, time_s in enumerate(arrays["time"]):
            counts = arrays["finger_object_contact_count"][index]
            distances = arrays["finger_object_contact_distance_m"][index]
            penetration = np.where(counts > 0, np.maximum(0.0, -distances), 0.0)
            row = {"time_s": float(time_s), "stage": str(arrays["diagnostic_stage"][index])}
            row.update({f"normal_force_{finger}_N": float(arrays["finger_object_normal_force_raw"][index, finger_index]) for finger_index, finger in enumerate(fingers)})
            row.update({
                "total_object_normal_force_N": float(arrays["finger_object_normal_force_raw"][index].sum()),
                "active_object_contact_count": int(counts.sum()),
                "maximum_penetration_m": float(penetration.max()),
                **dict(zip(("object_x_m", "object_y_m", "object_z_m"), arrays["object_position"][index].astype(float))),
                **dict(zip(("object_qw", "object_qx", "object_qy", "object_qz"), arrays["object_orientation"][index].astype(float))),
                **dict(zip(("object_vx_m_s", "object_vy_m_s", "object_vz_m_s"), arrays["object_linear_velocity"][index].astype(float))),
                **dict(zip(("object_wx_rad_s", "object_wy_rad_s", "object_wz_rad_s"), arrays["object_angular_velocity"][index].astype(float))),
                "object_table_contact": int(arrays["object_table_contact"][index]),
            })
            writer.writerow(row)


def _plot_validation(run: DiagnosticRun, hold_steps: int, path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "seqgrasp-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arrays = run.arrays
    hold = np.flatnonzero(arrays["diagnostic_stage"] == "hold")[-hold_steps:]
    time_axis = arrays["time"][hold] - arrays["time"][hold[0]]
    counts = arrays["finger_object_contact_count"][hold]
    distances = arrays["finger_object_contact_distance_m"][hold]
    penetration = np.max(np.where(counts > 0, np.maximum(0.0, -distances), 0.0), axis=1)
    displacement = arrays["object_position"][hold] - arrays["object_position"][hold[0]]
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
    axes[0, 0].plot(time_axis, arrays["finger_object_normal_force_raw"][hold])
    axes[0, 0].set_ylabel("normal force [N]")
    axes[0, 0].legend(run.metadata["finger_order"], fontsize=7)
    axes[0, 1].plot(time_axis, penetration)
    axes[0, 1].set_ylabel("maximum penetration [m]")
    axes[1, 0].plot(time_axis, displacement)
    axes[1, 0].set_ylabel("object displacement [m]")
    axes[1, 0].legend(("x", "y", "z"), fontsize=7)
    axes[1, 1].plot(time_axis, counts.sum(axis=1))
    axes[1, 1].set_ylabel("active object contacts [count]")
    for axis in axes[1]:
        axis.set_xlabel("long-hold time [s]")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def diagnostic_rows(summary: dict, fingers: list[str]) -> list[tuple[str, str, str]]:
    if "measurements" not in summary:
        return [("long hold", str(summary.get("completed_hold_steps", 0)), summary["verdict"])]
    measurements = summary["measurements"]
    rows = [
        (finger, f"{force:.6g} N mean", "raw")
        for finger, force in zip(fingers, measurements["mean_force_per_finger_N"])
    ]
    rows.extend([
        ("total normal force", f"{measurements['mean_total_normal_force_N']:.6g} N mean", str(summary["checks"]["force_order_of_magnitude"])),
        ("maximum penetration", f"{measurements['maximum_penetration_m']:.6g} m", str(summary["checks"]["penetration"])),
        ("translation drift", f"{measurements['maximum_translational_drift_m']:.6g} m", str(summary["checks"]["translational_drift"])),
        ("orientation drift", f"{measurements['maximum_orientation_drift_rad']:.6g} rad", str(summary["checks"]["orientation_drift"])),
        ("active contacts", f"{measurements['minimum_active_object_contacts']}..{measurements['maximum_active_object_contacts']}", str(summary["checks"]["active_contacts"])),
        ("table re-contact", str(measurements["table_recontact"]), str(summary["checks"]["table_recontact"])),
        ("complete contact loss", str(measurements["complete_object_hand_contact_loss"]), str(summary["checks"]["complete_contact_loss"])),
        ("numerical validity", str(measurements["numerical_validity"]), str(summary["checks"]["numerical_validity"])),
    ])
    return rows


def parameter_sensitivity(rows: list[dict]) -> dict:
    """Describe between-level measurement spans without selecting parameters."""

    parameters = ("friction", "solref", "solimp", "timestep_s")
    metrics = (
        "mean_total_normal_force_N", "maximum_penetration_m",
        "maximum_translational_drift_m", "maximum_orientation_drift_rad",
    )
    result: dict[str, dict] = {}
    for parameter in parameters:
        levels: dict[str, list[dict]] = {}
        for row in rows:
            level = json.dumps(row["parameters"][parameter], sort_keys=True)
            levels.setdefault(level, []).append(row["summary"]["measurements"])
        means = {
            level: {metric: float(np.mean([measurement[metric] for measurement in values])) for metric in metrics}
            for level, values in levels.items()
        }
        result[parameter] = {
            "level_means": means,
            "between_level_span": {
                metric: float(np.ptp([values[metric] for values in means.values()])) if means else 0.0
                for metric in metrics
            },
        }
    return result
