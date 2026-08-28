"""Run the frozen Phase 3C-0.7 mechanistic experiment in protocol order."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from seqgrasp.config import ROOT
from seqgrasp.phase3.config import SUPPORT_SURFACES
from seqgrasp.phase3.contacts import extract_shadow_contacts
from seqgrasp.phase3c07 import (
    PreshapeCondition,
    TransportStrategy,
    build_c07_scene,
    build_static_reachability_map,
    contract,
    forearm_dof_audit,
    floor_contact,
    freeze_acquisition_states,
    load_acquisition_states,
    load_phase3c07_config,
    plan_transport,
    pocket_volume_from_audit,
    restore_acquisition_state,
    run_transport_trial,
    wrist_commands,
)
from seqgrasp.phase3c0 import object_pose_in_palm


OUTPUT = ROOT / "outputs/phase3C07"


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def run_audit() -> dict[str, Any]:
    audit = build_static_reachability_map()
    _json(OUTPUT / "static_reachability.json", audit)
    print(json.dumps({key: audit[key] for key in ("sphere", "candidate_count", "feasible_count", "pocket_volume")}, indent=2), flush=True)
    return audit


def run_acquisition() -> dict[str, Any]:
    result = freeze_acquisition_states(OUTPUT / "matched_states")
    states = result["states"]
    retention = audit_acquisition_retention(states)
    summary = {
        "N": len(states),
        "candidate_attempts": result["manifest"]["candidates_attempted"],
        "thumb_contact": sum(bool(row.contact_flags[0]) for row in states),
        "index_contact": sum(bool(row.contact_flags[1]) for row in states),
        "unused_finger_accidental_contact": sum(any(row.contact_flags[2:5]) for row in states),
        "initial_penetration_by_surface_m": {
            surface: {
                "median": float(np.median([row.penetration_by_surface_m[index] for row in states])),
                "maximum": float(max(row.penetration_by_surface_m[index] for row in states)),
            }
            for index, surface in enumerate(SUPPORT_SURFACES)
        },
        "fixture_off_retention_audit": retention,
        "acquisition_retention": retention["retained_dual_contact_no_floor"],
        "frozen_before_transport_outcomes": True,
    }
    _json(OUTPUT / "acquisition_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def audit_acquisition_retention(states=None) -> dict[str, Any]:
    """Replay the frozen snapshots for the inherited 50-step hold, fixture off."""
    states = states or load_acquisition_states(OUTPUT / "matched_states")
    scene = build_c07_scene()
    rows = []
    for state in states:
        restore_acquisition_state(scene, state)
        initial = extract_shadow_contacts(scene)
        dual_contact_all_steps = bool(initial.contact_flags[0] and initial.contact_flags[1])
        floor_seen = floor_contact(scene)
        for _ in range(50):
            mujoco.mj_step(scene.model, scene.data)
            contacts = extract_shadow_contacts(scene)
            dual_contact_all_steps &= bool(contacts.contact_flags[0] and contacts.contact_flags[1])
            floor_seen |= floor_contact(scene)
        final = extract_shadow_contacts(scene)
        rows.append({
            "state_id": state.state_id,
            "initial_dual_contact_fixture_off": bool(initial.contact_flags[0] and initial.contact_flags[1]),
            "dual_contact_all_50_steps": dual_contact_all_steps,
            "final_dual_contact": bool(final.contact_flags[0] and final.contact_flags[1]),
            "floor_contact_seen": floor_seen,
        })
    result = {
        "N": len(rows), "hold_steps": 50, "fixture_active": False,
        "initial_dual_contact": sum(row["initial_dual_contact_fixture_off"] for row in rows),
        "dual_contact_all_50_steps": sum(row["dual_contact_all_50_steps"] for row in rows),
        "final_dual_contact": sum(row["final_dual_contact"] for row in rows),
        "retained_dual_contact_no_floor": sum(row["final_dual_contact"] and not row["floor_contact_seen"] for row in rows),
        "rows": rows,
    }
    _json(OUTPUT / "acquisition_retention_audit.json", result)
    summary_path = OUTPUT / "acquisition_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["fixture_off_retention_audit"] = result
        summary["acquisition_retention"] = result["retained_dual_contact_no_floor"]
        _json(summary_path, summary)
    return result


def _save_trial(row: dict[str, Any], serial: int) -> dict[str, Any]:
    samples = row.pop("samples")
    path = OUTPUT / "timeseries" / f"trial_{serial:04d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        step=np.asarray([sample["step"] for sample in samples], dtype=np.int32),
        stage=np.asarray([sample["stage"] for sample in samples]),
        center_palm_m=np.asarray([sample["sphere_center_palm_m"] for sample in samples]),
        transport=np.asarray([list(sample["transport"].values()) for sample in samples]),
        gravity_in_palm_mps2=np.asarray([sample["gravity_in_palm_mps2"] for sample in samples]),
        speed=np.asarray([[sample["linear_speed_mps"], sample["angular_speed_radps"]] for sample in samples]),
        contact_geometry_json=np.asarray([json.dumps(sample["contact_geometry"]) for sample in samples]),
        inside_pocket=np.asarray([sample["inside_pocket"] for sample in samples], dtype=np.int8),
        near_pocket=np.asarray([sample["near_pocket"] for sample in samples], dtype=np.int8),
        floor_contact=np.asarray([sample["floor_contact"] for sample in samples], dtype=np.int8),
        unused_finger_clearance_m=np.asarray([sample["unused_finger_clearance_m"] for sample in samples]),
        joint_boundary_events_json=np.asarray([json.dumps(sample["joint_boundary_events"]) for sample in samples]),
    )
    row["timeseries_path"] = str(path)
    row["sample_count"] = len(samples)
    row["entered_or_near_pocket"] = any(sample["inside_pocket"] or sample["near_pocket"] for sample in samples)
    row["near_pocket"] = any(sample["near_pocket"] for sample in samples)
    row["ring_contact"] = any("ring" in sample["contact_geometry"]["contact_topology"] for sample in samples)
    row["little_contact"] = any("little" in sample["contact_geometry"]["contact_topology"] for sample in samples)
    row["palm_contact"] = any("palm" in sample["contact_geometry"]["contact_topology"] for sample in samples)
    row["maximum_lambda_storage"] = float(max(sample["contact_geometry"]["lambda_storage"] for sample in samples))
    row["maximum_storage_force_n"] = float(max(sample["contact_geometry"]["storage_force_n"] for sample in samples))
    row["final_transport"] = samples[-1]["transport"]
    row["final_contact_topology"] = samples[-1]["contact_geometry"]["contact_topology"]
    row["final_load_bearing_topology"] = samples[-1]["contact_geometry"]["load_bearing_topology"]
    row["final_exit_speed"] = {
        "linear_mps": samples[-1]["linear_speed_mps"],
        "angular_radps": samples[-1]["angular_speed_radps"],
    }
    return row


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "N": len(rows),
        "pocket_entry": sum(row["first_pocket_entry_step"] is not None for row in rows),
        "near_pocket": sum(row["near_pocket"] for row in rows),
        "corridor_clear": sum(row["corridor_clear"] for row in rows),
        "lateral_transport_success": sum(row["lateral_transport_success"] for row in rows),
        "inward_transport_success": sum(row["inward_transport_success"] for row in rows),
        "ring_contact": sum(row["ring_contact"] for row in rows),
        "little_contact": sum(row["little_contact"] for row in rows),
        "palm_contact": sum(row["palm_contact"] for row in rows),
        "cage_formed": sum(row["cage_formed"] for row in rows),
        "unique_cage_states": len({row["state_id"] for row in rows if row["cage_formed"]}),
        "hold_survival": {
            step: sum(row["hold_survival"].get(step, False) for row in rows)
            for step in ("10", "25", "50", "100", "200", "300", "500", "750", "1000")
        },
        "thumb_contact_retained": sum(row["thumb_contact_retained"] for row in rows),
        "index_contact_retained": sum(row["index_contact_retained"] for row in rows),
        "maximum_lambda_storage": float(max((row["maximum_lambda_storage"] for row in rows), default=0.0)),
        "maximum_penetration_by_surface_m": (
            np.max([row["maximum_penetration_by_surface_m"] for row in rows], axis=0).tolist() if rows else [0.0] * 6
        ),
        "joint_boundary_event_count": sum(len(row["joint_boundary_events"]) for row in rows),
        "failure_counts": {
            label: sum(label in row["failures"] for row in rows)
            for label in sorted({failure for row in rows for failure in row["failures"]})
        },
    }


def run_transport(audit: dict[str, Any] | None = None) -> dict[str, Any]:
    audit = audit or json.loads((OUTPUT / "static_reachability.json").read_text(encoding="utf-8"))
    pocket = pocket_volume_from_audit(audit)
    states = load_acquisition_states(OUTPUT / "matched_states")
    if len(states) != 50:
        raise RuntimeError("Phase 3C-0.7 frozen acquisition cohort is not N=50")
    scene = build_c07_scene()
    rows: list[dict[str, Any]] = []
    serial = 0
    plan_cache = {}
    for state in states:
        for strategy in (TransportStrategy.T0_OLD_DIRECT, TransportStrategy.T1_POCKET_DIRECTED):
            key = (state.state_id, strategy.value)
            plan_cache[key] = plan_transport(scene, state, pocket, strategy)
            row = run_transport_trial(scene, state, pocket, strategy, transport_plan=plan_cache[key])
            row["wrist_level"] = "W0"
            rows.append(_save_trial(row, serial)); serial += 1
        print(f"C07-C {state.state_id}", flush=True)

    # W1 is required first. The pocket-directed IK path is identical for every
    # wrist direction; only the concurrently commanded native wrist differs.
    for state in states:
        plan = plan_transport(scene, state, pocket, TransportStrategy.T2_WRIST_ASSISTED)
        plan_cache[(state.state_id, TransportStrategy.T2_WRIST_ASSISTED.value)] = plan
        for command in wrist_commands("W1"):
            row = run_transport_trial(
                scene, state, pocket, TransportStrategy.T2_WRIST_ASSISTED,
                wrist_delta_deg=command, transport_plan=plan,
            )
            row["wrist_level"] = "W1"
            rows.append(_save_trial(row, serial)); serial += 1
        print(f"C07-D W1 {state.state_id}", flush=True)

    restore_acquisition_state(scene, states[0])
    start = object_pose_in_palm(scene, scene.object_body_id)[0]
    forearm = forearm_dof_audit(scene, start, pocket)
    # Protocol requires stopping angle expansion on a native-DOF orientation
    # limit. W2/W3 are therefore never silently attempted after this finding.
    wider_executed = []
    if not forearm["stop_wrist_expansion"]:
        for level in ("W2", "W3"):
            level_rows = []
            for state in states:
                plan = plan_cache[(state.state_id, TransportStrategy.T2_WRIST_ASSISTED.value)]
                for command in wrist_commands(level):
                    row = run_transport_trial(
                        scene, state, pocket, TransportStrategy.T2_WRIST_ASSISTED,
                        wrist_delta_deg=command, transport_plan=plan,
                    )
                    row["wrist_level"] = level
                    saved = _save_trial(row, serial); serial += 1
                    rows.append(saved); level_rows.append(saved)
            wider_executed.append(level)
            if not any(row["first_pocket_entry_step"] is not None for row in level_rows):
                break

    # Preshape only the best transport condition that actually reaches or
    # approaches the independently constructed volume.
    eligible = [row for row in rows if row["strategy"] != TransportStrategy.T0_OLD_DIRECT.value and row["entered_or_near_pocket"]]
    preshape_condition = None
    if eligible:
        keys = {(row["wrist_level"], tuple(row["wrist_delta_command_deg"])) for row in eligible}
        preshape_condition = max(
            keys,
            key=lambda key: (
                sum(row["first_pocket_entry_step"] is not None for row in rows if row["wrist_level"] == key[0] and tuple(row["wrist_delta_command_deg"]) == key[1]),
                sum(row["near_pocket"] for row in rows if row["wrist_level"] == key[0] and tuple(row["wrist_delta_command_deg"]) == key[1]),
            ),
        )
        level, command = preshape_condition
        strategy = TransportStrategy.T1_POCKET_DIRECTED if level == "W0" else TransportStrategy.T2_WRIST_ASSISTED
        for preshape in PreshapeCondition:
            for state in states:
                plan = plan_cache[(state.state_id, strategy.value)]
                row = run_transport_trial(
                    scene, state, pocket, strategy, wrist_delta_deg=command,
                    preshape=preshape, transport_plan=plan,
                )
                row["wrist_level"] = level
                rows.append(_save_trial(row, serial)); serial += 1
            print(f"C07-E {preshape.value} {level} {command}", flush=True)

    by_condition = {}
    for key in sorted({(row["strategy"], row["wrist_level"], tuple(row["wrist_delta_command_deg"]), row["preshape"]) for row in rows}, key=str):
        selected = [row for row in rows if (row["strategy"], row["wrist_level"], tuple(row["wrist_delta_command_deg"]), row["preshape"]) == key]
        by_condition["|".join((key[0], key[1], str(key[2]), str(key[3])))] = _summary(selected)
    result = {
        "phase": "3C-0.7", "branch": "codex/phase3C07-pocket-reachability-cage",
        "base_commit": "9aa135f0012ca9e6b845741355ce73aeadec0f09",
        "matched_state_count": len(states), "static_audit_path": str(OUTPUT / "static_reachability.json"),
        "forearm_dof_audit": forearm, "wider_wrist_levels_executed": wider_executed,
        "preshape_selected_transport_condition": preshape_condition,
        "condition_summary": by_condition, "overall_summary": _summary(rows),
        "contract": contract() | {"physics_changed": False, "friction_changed": False,
                                   "contact_parameters_changed": False, "joint_limits_changed": False,
                                   "actuator_limits_changed": False},
        "trials": rows,
    }
    _json(OUTPUT / "phase3c07_results.json", result)
    print(json.dumps({key: result[key] for key in ("forearm_dof_audit", "wider_wrist_levels_executed", "preshape_selected_transport_condition", "overall_summary")}, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("audit", "acquisition", "acquisition-retention", "transport", "all"), default="all")
    args = parser.parse_args()
    audit = None
    if args.stage in {"audit", "all"}:
        audit = run_audit()
    if args.stage in {"acquisition", "all"}:
        run_acquisition()
    if args.stage == "acquisition-retention":
        print(json.dumps(audit_acquisition_retention(), indent=2), flush=True)
    if args.stage in {"transport", "all"}:
        run_transport(audit)


if __name__ == "__main__":
    main()
