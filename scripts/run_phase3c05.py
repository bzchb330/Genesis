from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from seqgrasp.config import ROOT
from seqgrasp.phase3.model import build_shadow_scene
from seqgrasp.phase3c05 import (
    CaptureOutcome,
    CaptureStrategy,
    capture_trial,
    load_frozen_states,
    load_phase3c05_config,
    release_trial,
)


def _save_capture(path: Path, trial: dict) -> dict:
    samples = trial.pop("samples")
    snapshot = trial.pop("final_snapshot")
    np.savez_compressed(
        path,
        step=np.asarray([row["step"] for row in samples], dtype=np.int32),
        normal_forces_n=np.asarray([row["normal_forces_n"] for row in samples]),
        tangential_forces_n=np.asarray([row["tangential_forces_n"] for row in samples]),
        tangential_normal_ratio_by_surface=np.asarray(
            [row["tangential_normal_ratio_by_surface"] for row in samples]
        ),
        penetration_by_surface_m=np.asarray([row["penetration_by_surface_m"] for row in samples]),
        contact_details_json=np.asarray([json.dumps(row["contact_details"]) for row in samples]),
        load_share=np.asarray([[row["acquisition_force_n"], row["alternate_force_n"],
                                row["total_force_n"], row["alternate_fraction"]] for row in samples]),
        contact_flags=np.asarray([row["contact_flags"] for row in samples], dtype=np.int8),
        maximum_penetration_m=np.asarray([row["maximum_penetration_m"] for row in samples]),
        maximum_penetration_pair=np.asarray([
            list(row["maximum_penetration_pair"] or ("", "")) for row in samples
        ]),
        A_position_palm_m=np.asarray([row["A_position_palm_m"] for row in samples]),
        A_orientation_palm=np.asarray([row["A_orientation_palm"] for row in samples]),
        A_displacement_from_capture_start_m=np.asarray(
            [row["A_displacement_from_capture_start_m"] for row in samples]
        ),
        A_linear_speed_mps=np.asarray([row["A_linear_speed_mps"] for row in samples]),
        A_angular_speed_radps=np.asarray([row["A_angular_speed_radps"] for row in samples]),
        gravity_in_palm_frame=np.asarray([row["gravity_in_palm_frame"] for row in samples]),
        wrist_qpos=np.asarray([row["wrist_qpos"] for row in samples]),
        minimum_joint_margin_rad=np.asarray([row["minimum_joint_margin_rad"] for row in samples]),
        actuator_clipping_count=np.asarray([row["actuator_clipping_count"] for row in samples]),
        floor_contact=np.asarray([row["floor_contact"] for row in samples], dtype=np.int8),
    )
    trial["timeseries_path"] = str(path)
    trial["minimum_joint_margin_rad"] = float(min(row["minimum_joint_margin_rad"] for row in samples))
    trial["maximum_penetration_m"] = float(max(row["maximum_penetration_m"] for row in samples))
    trial["maximum_actuator_clipping_count"] = int(max(row["actuator_clipping_count"] for row in samples))
    trial["palm_contact"] = CaptureOutcome.PALM_CONTACT.value in trial["outcomes"]
    trial["storage_finger_contact"] = CaptureOutcome.STORAGE_FINGER_CONTACT.value in trial["outcomes"]
    trial["coordinated_capture"] = CaptureOutcome.COORDINATED_CAPTURE_ESTABLISHED.value in trial["outcomes"]
    trial["_snapshot"] = snapshot
    return trial


def _public(row: dict) -> dict:
    return {key: value for key, value in row.items() if key != "_snapshot"}


def _aggregate(rows: list[dict]) -> dict:
    count = len(rows)
    return {
        "trials": count,
        "A_retained": sum(row["A_retained"] for row in rows),
        "storage_finger_contact": sum(row["storage_finger_contact"] for row in rows),
        "palm_contact": sum(row["palm_contact"] for row in rows),
        "coordinated_capture": sum(row["coordinated_capture"] for row in rows),
        "maximum_alternate_fraction": float(max((row["maximum_alternate_fraction"] for row in rows), default=0.0)),
        "gate_persistence_counts": {
            key: sum(row["persistence_first_reached"][key] is not None for row in rows)
            for key in rows[0]["persistence_first_reached"]
        } if rows else {},
    }


def main() -> None:
    cfg = load_phase3c05_config()
    states = load_frozen_states()
    if len(states) != int(cfg["matched_states"]["count"]):
        raise RuntimeError("frozen matched-state cohort size changed")
    output = ROOT / "outputs/phase3C05"
    series_dir = output / "timeseries"
    series_dir.mkdir(parents=True, exist_ok=True)
    scene = build_shadow_scene()
    subsets = [tuple(value) for value in cfg["capture"]["storage_subsets"]]
    rows: list[dict] = []

    for strategy in (CaptureStrategy.SERIAL, CaptureStrategy.SIMULTANEOUS):
        for subset_index, subset in enumerate(subsets):
            for state_index, state in enumerate(states):
                name = f"{strategy.value}_{subset_index}_{state.state_id}.npz"
                row = capture_trial(scene, state, subset, strategy)
                rows.append(_save_capture(series_dir / name, row))
            print(f"completed {strategy.value} subset {subset_index + 1}/{len(subsets)}", flush=True)

    w1_commands = [tuple(float(v) for v in value) for value in cfg["wrist_probes"]["W1_commands_deg"]]
    w1_rows: list[dict] = []
    safe_w1: list[tuple] = []
    for command_index, command in enumerate(w1_commands):
        for subset_index, subset in enumerate(subsets):
            for state in states:
                name = f"C05-WRIST_W1_{command_index}_{subset_index}_{state.state_id}.npz"
                row = capture_trial(scene, state, subset, CaptureStrategy.WRIST, wrist_delta_deg=command)
                saved = _save_capture(series_dir / name, row)
                rows.append(saved); w1_rows.append(saved)
                if saved["A_retained"]:
                    safe_w1.append((state, subset, command, command_index, subset_index))
        print(f"completed C05-WRIST W1 command {command_index + 1}/{len(w1_commands)}", flush=True)

    w2_rows: list[dict] = []
    for state, subset, command, command_index, subset_index in safe_w1:
        w2_command = tuple(10.0 * np.sign(value) for value in command)
        name = f"C05-WRIST_W2_{command_index}_{subset_index}_{state.state_id}.npz"
        row = capture_trial(scene, state, subset, CaptureStrategy.WRIST, wrist_delta_deg=w2_command)
        saved = _save_capture(series_dir / name, row)
        rows.append(saved); w2_rows.append(saved)
    print(f"completed conditional W2 probes: {len(w2_rows)}", flush=True)

    # Predefined full-subset load-transfer condition, not selected by state outcome.
    load_rows: list[dict] = []
    full_subset = ("middle", "ring", "little")
    load_wrist = (-5.0, -5.0)
    for state in states:
        name = f"C05-WRIST-LOAD-TRANSFER_{state.state_id}.npz"
        row = capture_trial(scene, state, full_subset, CaptureStrategy.WRIST_LOAD_TRANSFER,
                            wrist_delta_deg=load_wrist)
        saved = _save_capture(series_dir / name, row)
        rows.append(saved); load_rows.append(saved)
    print("completed load-transfer capture states", flush=True)

    release_rows = []
    for capture in load_rows:
        for finger in cfg["release"]["fingers"]:
            for ramp in cfg["release"]["ramp_steps"]:
                release_input = dict(capture)
                release_input["final_snapshot"] = capture["_snapshot"]
                result = release_trial(scene, release_input, str(finger), int(ramp))
                result.update({"state_id": capture["state_id"], "capture_strategy": capture["strategy"],
                               "subset": capture["subset"], "wrist_delta_command_deg": capture["wrist_delta_command_deg"]})
                if result.get("samples"):
                    samples = result.pop("samples")
                    path = series_dir / f"RELEASE_{finger}_{ramp}_{capture['state_id']}.npz"
                    np.savez_compressed(path,
                        step=np.asarray([sample["step"] for sample in samples]),
                        normal_forces_n=np.asarray([sample["normal_forces_n"] for sample in samples]),
                        alternate_fraction=np.asarray([sample["alternate_fraction"] for sample in samples]),
                        floor_contact=np.asarray([sample["floor_contact"] for sample in samples], dtype=np.int8))
                    result["timeseries_path"] = str(path)
                release_rows.append(result)
    print("completed gated release matrix", flush=True)

    # Optional second release is allowed only after robust one-finger recovery.
    successful_first = [row for row in release_rows if row.get("one_resource_recovered")]
    second_release = {"executed": False, "reason": "one-finger recovery not robust across multiple matched states"}
    if len({row["state_id"] for row in successful_first}) >= 2:
        second_release = {"executed": False, "reason": "mechanism met robustness precondition; PI review required before optional expansion"}

    by_strategy_subset = {}
    for strategy in (CaptureStrategy.SERIAL.value, CaptureStrategy.SIMULTANEOUS.value):
        for subset in subsets:
            selected = [row for row in rows if row["strategy"] == strategy and tuple(row["subset"]) == subset]
            by_strategy_subset[f"{strategy}|{'+'.join(subset)}"] = _aggregate(selected)
    result = {
        "phase": "3C-0.5", "matched_state_count": len(states),
        "matched_state_ids_frozen_before_formal_conditions": [state.state_id for state in states],
        "physics_changed": False, "world_gravity_changed": False, "object_B_instantiated": False,
        "rl_training_performed": False,
        "condition_summary": {
            "serial": _aggregate([row for row in rows if row["strategy"] == CaptureStrategy.SERIAL.value]),
            "simultaneous": _aggregate([row for row in rows if row["strategy"] == CaptureStrategy.SIMULTANEOUS.value]),
            "wrist_W1": _aggregate(w1_rows), "wrist_W2": _aggregate(w2_rows),
            "wrist_load_transfer_capture": _aggregate(load_rows),
            "by_strategy_subset": by_strategy_subset,
        },
        "capture_trials": [_public(row) for row in rows],
        "release_trials": release_rows,
        "optional_second_release": second_release,
    }
    (output / "phase3c05_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"condition_summary": result["condition_summary"],
                      "release_executed": sum(row["executed"] for row in release_rows),
                      "one_resource_recovered": sum(row.get("one_resource_recovered", False) for row in release_rows),
                      "optional_second_release": second_release}, indent=2), flush=True)


if __name__ == "__main__":
    main()
