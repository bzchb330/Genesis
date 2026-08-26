from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from seqgrasp.config import ROOT
from seqgrasp.phase3c06 import (
    POCKET_NAMES,
    PRESHAPE_CONDITIONS,
    build_sphere_scene,
    construct_palmodigital_pockets,
    freeze_acquisition_states,
    load_acquisition_states,
    load_phase3c06_config,
    no_object_b_or_rl_contract,
    pocket_geometry,
    progression_allowed,
    run_storage_trial,
    sphere_scale,
    wrist_commands,
)


def _save_trial(path: Path, row: dict[str, Any]) -> dict[str, Any]:
    samples = row.pop("samples")
    np.savez_compressed(
        path,
        step=np.asarray([sample["step"] for sample in samples], dtype=np.int32),
        stage=np.asarray([sample["stage"] for sample in samples]),
        center_palm_m=np.asarray([sample["center_palm_m"] for sample in samples]),
        contact_flags=np.asarray([sample["contact_flags"] for sample in samples], dtype=np.int8),
        normal_forces_n=np.asarray([sample["normal_forces_n"] for sample in samples]),
        penetration_by_surface_m=np.asarray([sample["penetration_by_surface_m"] for sample in samples]),
        penetration_by_surface_over_radius=np.asarray([
            sample["penetration_by_surface_over_radius"] for sample in samples
        ]),
        maximum_penetration_m=np.asarray([sample["maximum_penetration_m"] for sample in samples]),
        maximum_penetration_over_radius=np.asarray([
            sample["maximum_penetration_over_radius"] for sample in samples
        ]),
        gravity_in_palm_mps2=np.asarray([sample["gravity_in_palm_mps2"] for sample in samples]),
        unused_finger_clearance_m=np.asarray([sample["unused_finger_clearance_m"] for sample in samples]),
        floor_contact=np.asarray([sample["floor_contact"] for sample in samples], dtype=np.int8),
        storage_json=np.asarray([json.dumps(sample["storage"]) for sample in samples]),
        contact_pairs_json=np.asarray([json.dumps(sample["contact_pairs"]) for sample in samples]),
        minimum_joint_margin_rad=np.asarray([sample["minimum_joint_margin_rad"] for sample in samples]),
    )
    row["timeseries_path"] = str(path)
    return row


def _condition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    checkpoints = load_phase3c06_config()["experiment"]["survival_checkpoints"]
    return {
        "N": count,
        "corridor_cleared": sum(row["corridor_cleared"] for row in rows),
        "pocket_entry": sum(row["pocket_entry_step"] is not None for row in rows),
        "stable_capture": sum(row["stable_capture"] for row in rows),
        "ring_contact": sum(row["ring_contact"] for row in rows),
        "little_contact": sum(row["little_contact"] for row in rows),
        "palm_root_contact": sum(row["palm_contact"] for row in rows),
        "alternate_support": sum(row["alternate_support"] for row in rows),
        "thumb_release_attempts": sum(row["thumb_release_attempted"] for row in rows),
        "thumb_recovered": sum(row["thumb_recovered"] for row in rows),
        "survival": {
            str(step): sum(row.get("survival", {}).get(str(step), False) for row in rows)
            for step in checkpoints
        },
        "maximum_penetration_m": float(max((row["maximum_penetration_m"] for row in rows), default=0.0)),
        "maximum_penetration_over_radius": float(max((row["maximum_penetration_over_radius"] for row in rows), default=0.0)),
        "gross_overlap_warnings": sum(row["gross_overlap_warning"] is True for row in rows),
        "penetration_acceptability_pending_PI": sum(
            row.get("penetration_valid_for_progression") is None for row in rows
        ),
    }


def main() -> None:
    output = ROOT / "outputs/phase3C06"
    states_dir = output / "matched_states"
    series_dir = output / "timeseries"
    output.mkdir(parents=True, exist_ok=True)
    series_dir.mkdir(parents=True, exist_ok=True)
    if not (states_dir / "manifest.json").exists():
        freeze_acquisition_states(states_dir)
    states = load_acquisition_states(states_dir)
    cfg = load_phase3c06_config()
    if len(states) != int(cfg["matched_states"]["count"]):
        raise RuntimeError("frozen Phase 3C-0.6 cohort count changed")
    scene = build_sphere_scene()
    scale = sphere_scale()
    pockets = construct_palmodigital_pockets(scene, scale.radius_m)
    pocket_audit = {name: pocket_geometry(scene, pocket, scale.radius_m) for name, pocket in pockets.items()}
    rows: list[dict[str, Any]] = []
    commands = (("W0", command) for command in wrist_commands("W0"))
    commands = list(commands) + [("W1", command) for command in wrist_commands("W1")]
    for pocket_name in POCKET_NAMES:
        pocket = pockets[pocket_name]
        for preshape in PRESHAPE_CONDITIONS:
            for wrist_level, command in commands:
                for state in states:
                    row = run_storage_trial(scene, state, pocket, preshape, command)
                    row["wrist_level"] = wrist_level
                    name = f"D0_{pocket_name}_{preshape}_{wrist_level}_{command[0]:+g}_{command[1]:+g}_{state.state_id}.npz"
                    rows.append(_save_trial(series_dir / name, row))
                print(f"completed D0 {pocket_name} {preshape} {wrist_level} {command}", flush=True)

    # The protocol authorizes wider wrists and larger spheres only after the
    # D0 physical gate. This decision uses only the stated structural meaning
    # of multiple distinct successful states (at least two), not a new rate.
    d0_gate = progression_allowed(rows)
    wider_wrist_rows: list[dict[str, Any]] = []
    size_rows: list[dict[str, Any]] = []
    if d0_gate:
        successful_conditions = {
            (row["pocket"], row["preshape"]) for row in rows
            if row["thumb_recovered"] and row.get("survival", {}).get("1000")
            and row.get("penetration_valid_for_progression") is True
        }
        for level in ("W2", "W3"):
            for pocket_name, preshape in sorted(successful_conditions):
                for command in wrist_commands(level):
                    for state in states:
                        row = run_storage_trial(scene, state, pockets[pocket_name], preshape, command)
                        row["wrist_level"] = level
                        name = f"D0_{pocket_name}_{preshape}_{level}_{command[0]:+g}_{command[1]:+g}_{state.state_id}.npz"
                        wider_wrist_rows.append(_save_trial(series_dir / name, row))
            if not progression_allowed(wider_wrist_rows):
                break
        # A size increase is similarly sequential. If a scale loses the gate,
        # no larger scale is run.
        for size in (1.25, 1.5, 1.75, 2.0):
            sized_scene = build_sphere_scene(size)
            sized_scale = sphere_scale(size)
            sized_pockets = construct_palmodigital_pockets(sized_scene, sized_scale.radius_m)
            current: list[dict[str, Any]] = []
            # Acquisition states are not silently reused across sizes. A full
            # size-specific state freeze would be required; stop and report the
            # physically unmatched state issue for PI review.
            _ = sized_pockets
            size_rows.extend(current)
            break

    all_d0 = rows + wider_wrist_rows
    by_condition = {}
    for pocket_name in POCKET_NAMES:
        for preshape in PRESHAPE_CONDITIONS:
            for level in ("W0", "W1", "W2", "W3"):
                selected = [row for row in all_d0 if row["pocket"] == pocket_name
                            and row["preshape"] == preshape and row["wrist_level"] == level]
                if selected:
                    by_condition[f"{pocket_name}|{preshape}|{level}"] = _condition_summary(selected)
    result = {
        "phase": "3C-0.6", "branch": "codex/phase3C06-sphere-palmodigital-storage",
        "base_commit": "7baac924a14ff863c7d1b0bb9bfc67734390609d",
        "matched_state_count": len(states),
        "matched_state_ids_frozen_before_outcomes": [state.state_id for state in states],
        "sphere_scale": scale.__dict__,
        "compiled_mass_kg": float(scene.model.body_mass[scene.object_body_id]),
        "pocket_geometry": pocket_audit,
        "D0_progression_gate_reached": d0_gate,
        "W2_executed": any(row["wrist_level"] == "W2" for row in wider_wrist_rows),
        "W3_executed": any(row["wrist_level"] == "W3" for row in wider_wrist_rows),
        "size_curriculum_executed": bool(size_rows),
        "size_curriculum_stop_reason": (
            "D0 physical progression gate not reached" if not d0_gate
            else "size-specific matched acquisition cohort requires PI review before outcomes"
        ),
        "condition_summary": by_condition,
        "D0_summary": _condition_summary(all_d0),
        "trials": all_d0,
        "wider_wrist_trials": wider_wrist_rows,
        "size_trials": size_rows,
        "contract": no_object_b_or_rl_contract() | {
            "physics_changed": False, "friction_changed": False, "compliance_changed": False,
            "world_gravity_changed": False, "official_MJCF_modified": False,
        },
    }
    (output / "phase3c06_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "matched_states": len(states), "D0_trials": len(all_d0),
        "D0_summary": result["D0_summary"], "D0_progression_gate_reached": d0_gate,
        "W2_executed": result["W2_executed"], "W3_executed": result["W3_executed"],
        "size_curriculum_executed": result["size_curriculum_executed"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
