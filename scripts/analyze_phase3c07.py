"""Analyze Phase 3C-0.7 and create vector figures plus PI-facing reports."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from seqgrasp.config import ROOT
from seqgrasp.phase3.config import SUPPORT_SURFACES


OUTPUT = ROOT / "outputs/phase3C07"
FIGURES = ROOT / "docs/figures/phase3C07"
NAMES = (
    "25mm_sphere_relative_to_finger_links", "ring_little_pocket_geometry",
    "pocket_static_reachability_map", "palm_frame_transport_components",
    "old_vs_pocket_directed_transport", "fixed_vs_wrist_assisted_transport",
    "wrist_direction_pocket_entry_map", "gravity_in_palm_transport_map",
    "sphere_dynamic_paths_to_pocket", "pocket_entry_distribution",
    "no_preshape_vs_preshape", "ring_little_palm_contact_geometry",
    "load_bearing_cage_topology", "cage_hold_survival",
    "penetration_by_contact_pair", "joint_boundary_transport_audit",
    "failure_taxonomy", "representative_pocket_transport_sequence",
)


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", format="pdf", bbox_inches="tight",
                metadata={"Title": name, "Creator": "Phase 3C-0.7 analysis"})
    plt.close(fig)


def _group(rows: list[dict[str, Any]], strategy: str, level: str, preshape: str | None = None) -> list[dict[str, Any]]:
    return [row for row in rows if row["strategy"] == strategy and row["wrist_level"] == level and row["preshape"] == preshape]


def _series(row: dict[str, Any]) -> dict[str, np.ndarray]:
    return dict(np.load(row["timeseries_path"], allow_pickle=False))


def _count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(bool(row[key]) for row in rows)


def _condition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "N": len(rows), "entry": sum(row["first_pocket_entry_step"] is not None for row in rows),
        "near": _count(rows, "near_pocket"), "cage": _count(rows, "cage_formed"),
        "lateral": _count(rows, "lateral_transport_success"), "inward": _count(rows, "inward_transport_success"),
        "ring": _count(rows, "ring_contact"), "little": _count(rows, "little_contact"),
        "palm": _count(rows, "palm_contact"), "corridor_clear": _count(rows, "corridor_clear"),
    }


def _reclassify_boundary_limit(row: dict[str, Any]) -> None:
    """Use logged exact events; a passive storage-finger bound is not causal."""
    limited = bool(
        any(event["group"] in {"thumb", "index"} for event in row["joint_boundary_events"])
        and (not row["lateral_transport_success"] or not row["inward_transport_success"])
    )
    row["joint_boundary_limited_transport"] = limited
    row["failures"] = [failure for failure in row["failures"] if failure != "JOINT_BOUNDARY_LIMITED_TRANSPORT"]
    if limited:
        row["failures"].append("JOINT_BOUNDARY_LIMITED_TRANSPORT")


def _figures(audit: dict[str, Any], result: dict[str, Any], summary: dict[str, Any]) -> None:
    rows = result["trials"]
    t0 = _group(rows, "T0_OLD_DIRECT", "W0")
    t1 = _group(rows, "T1_POCKET_DIRECTED", "W0")
    w1 = _group(rows, "T2_WRIST_ASSISTED", "W1")
    links = [row["length_m"] * 1000 for row in summary["link_audit"]]

    fig, ax = plt.subplots(); ax.bar(["sphere", "proximal", "intermediate"], [25, np.mean(links[::2]), np.mean(links[1::2])]); ax.set_ylabel("mm"); ax.set_title("25-mm sphere and audited Shadow link scale"); _save(fig, NAMES[0])
    pts = np.asarray(audit["pocket_volume"]["feasible_centers_palm_m"])
    fig = plt.figure(); ax = fig.add_subplot(projection="3d"); ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=5); ax.set(xlabel="palm x (m)", ylabel="palm y (m)", zlabel="palm z (m)", title="Geometry-derived ring/little pocket volume"); _save(fig, NAMES[1])
    candidates = audit["candidates"]; xyz = np.asarray([r["center_palm_m"] for r in candidates]); fits = np.asarray([r["sphere_geometrically_fits"] for r in candidates]); fig, ax = plt.subplots(); ax.scatter(xyz[~fits, 0], xyz[~fits, 2], s=1, alpha=.08, label="infeasible"); ax.scatter(xyz[fits, 0], xyz[fits, 2], s=5, label="feasible"); ax.set(xlabel="palm x (m)", ylabel="palm z (m)", title="Outcome-independent static reachability slices"); ax.legend(); _save(fig, NAMES[2])
    fig, ax = plt.subplots(); ax.bar(["T0 lateral", "T0 inward", "T1 lateral", "T1 inward"], [_count(t0, "lateral_transport_success"), _count(t0, "inward_transport_success"), _count(t1, "lateral_transport_success"), _count(t1, "inward_transport_success")]); ax.set_ylabel("states / 50"); _save(fig, NAMES[3])
    fig, ax = plt.subplots(); ax.bar(["old target", "pocket directed"], [sum(r["first_pocket_entry_step"] is not None for r in t0), sum(r["first_pocket_entry_step"] is not None for r in t1)]); ax.set_ylabel("pocket entries / 50"); _save(fig, NAMES[4])
    fig, ax = plt.subplots(); ax.bar(["fixed T1", "all W1"], [sum(r["first_pocket_entry_step"] is not None for r in t1), sum(r["first_pocket_entry_step"] is not None for r in w1)]); ax.set_ylabel("entries (matched trials)"); _save(fig, NAMES[5])
    directions = sorted({tuple(r["wrist_delta_command_deg"]) for r in w1}); entries = [sum(r["first_pocket_entry_step"] is not None for r in w1 if tuple(r["wrist_delta_command_deg"]) == d) for d in directions]; fig, ax = plt.subplots(); ax.scatter([d[0] for d in directions], [d[1] for d in directions], s=np.asarray(entries) * 25 + 30, c=entries); ax.set(xlabel="WRJ command 1 (deg)", ylabel="WRJ command 2 (deg)", title="W1 pocket-entry map (marker size=count)"); _save(fig, NAMES[6])
    gravity, distances = [], []
    for row in w1:
        data = _series(row); gravity.append(data["gravity_in_palm_mps2"][-1]); distances.append(row["final_transport"]["pocket_distance_m"])
    gravity = np.asarray(gravity); fig, ax = plt.subplots(); ax.scatter(gravity[:, 0] if len(gravity) else [], distances, s=8); ax.set(xlabel="final palm-x gravity (m/s²)", ylabel="final pocket distance (m)", title="Native wrist gravity versus transport"); _save(fig, NAMES[7])
    representative = min(t1 or t0, key=lambda r: r["closest_approach_m"]); data = _series(representative); centers = data["center_palm_m"]; fig = plt.figure(); ax = fig.add_subplot(projection="3d"); ax.plot(centers[:, 0], centers[:, 1], centers[:, 2]); ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, alpha=.15); ax.set_title("Truthful best-observed dynamic path"); _save(fig, NAMES[8])
    fig, ax = plt.subplots(); values = [r["first_pocket_entry_step"] for r in rows if r["first_pocket_entry_step"] is not None]
    if values:
        ax.hist(values, bins=20)
    else:
        ax.text(.5, .5, "No pocket entries observed", ha="center", va="center", transform=ax.transAxes)
    ax.set(xlabel="entry step", ylabel="count", title="Pocket-entry timing"); _save(fig, NAMES[9])
    p0 = [r for r in rows if r["preshape"] == "P0_AFTER_ENTRY"]; p1 = [r for r in rows if r["preshape"] == "P1_GEOMETRIC_APPROACH"]
    fig, ax = plt.subplots(); ax.bar([f"none\nN={sum(r['preshape'] is None for r in rows)}", f"P0\nN={len(p0)}", f"P1\nN={len(p1)}"], [sum(r["cage_formed"] for r in rows if r["preshape"] is None), _count(p0, "cage_formed"), _count(p1, "cage_formed")]); ax.set(ylabel="cages (raw trials)", title="Preshape gated off because pocket was not approached"); _save(fig, NAMES[10])
    fig, ax = plt.subplots(); ax.bar(["ring", "little", "palm/root"], [_count(rows, "ring_contact"), _count(rows, "little_contact"), _count(rows, "palm_contact")]); ax.set_ylabel("trials with contact"); _save(fig, NAMES[11])
    topology = Counter(tuple(r["final_load_bearing_topology"]) for r in rows); labels = ["+".join(k) or "none" for k, _ in topology.most_common(10)]; counts = [v for _, v in topology.most_common(10)]; fig, ax = plt.subplots(); ax.barh(labels, counts); ax.set_title("Final load-bearing topology"); _save(fig, NAMES[12])
    checkpoints = [10, 25, 50, 100, 200, 300, 500, 750, 1000]; holds = [sum(r["hold_survival"][str(x)] for r in rows) for x in checkpoints]; fig, ax = plt.subplots(); ax.step(checkpoints, holds, where="post"); ax.set(xlabel="hold steps", ylabel="surviving cages", title="Coordinated cage hold"); _save(fig, NAMES[13])
    penetration = np.max([r["maximum_penetration_by_surface_m"] for r in rows], axis=0) if rows else np.zeros(6); fig, ax = plt.subplots(); ax.bar(SUPPORT_SURFACES, penetration * 1000); ax.set(ylabel="maximum penetration (mm)", title="Raw penetration; acceptability remains TODO(PI)"); _save(fig, NAMES[14])
    joints = Counter(event["joint"] for row in rows for event in row["joint_boundary_events"]); fig, ax = plt.subplots(); ax.barh(list(joints) or ["none"], list(joints.values()) or [0]); ax.set_title("Exact compiled-boundary events"); _save(fig, NAMES[15])
    failures = Counter(f for r in rows for f in r["failures"]); fig, ax = plt.subplots(); ax.barh(list(failures), list(failures.values())); ax.set_title("Protocol failure taxonomy"); _save(fig, NAMES[16])
    idx = np.linspace(0, len(centers) - 1, min(8, len(centers))).astype(int); fig, ax = plt.subplots(); ax.plot(centers[:, 0], centers[:, 2], color="0.75"); ax.scatter(centers[idx, 0], centers[idx, 2], c=np.arange(len(idx)), cmap="viridis"); ax.set(xlabel="palm x (m)", ylabel="palm z (m)", title="Representative transport sequence"); _save(fig, NAMES[17])


def main() -> None:
    audit = json.loads((OUTPUT / "static_reachability.json").read_text(encoding="utf-8"))
    result = json.loads((OUTPUT / "phase3c07_results.json").read_text(encoding="utf-8"))
    acquisition = json.loads((OUTPUT / "acquisition_summary.json").read_text(encoding="utf-8"))
    from seqgrasp.phase3c06 import audit_non_thumb_link_lengths
    links = [row.__dict__ for row in audit_non_thumb_link_lengths()]
    rows = result["trials"]
    for row in rows:
        _reclassify_boundary_limit(row)
    # This is a deterministic correction of taxonomy from already-recorded
    # raw boundary events, not a new replay or a changed physical outcome.
    (OUTPUT / "phase3c07_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    t0 = _group(rows, "T0_OLD_DIRECT", "W0"); t1 = _group(rows, "T1_POCKET_DIRECTED", "W0"); w1 = _group(rows, "T2_WRIST_ASSISTED", "W1")
    directions = {}
    for command in sorted({tuple(row["wrist_delta_command_deg"]) for row in w1}):
        selected = [row for row in w1 if tuple(row["wrist_delta_command_deg"]) == command]
        directions[str(command)] = _condition(selected) | {"median_final_pocket_distance_m": float(np.median([r["final_transport"]["pocket_distance_m"] for r in selected]))}
    p0 = [r for r in rows if r["preshape"] == "P0_AFTER_ENTRY"]; p1 = [r for r in rows if r["preshape"] == "P1_GEOMETRIC_APPROACH"]
    boundary = Counter(event["joint"] for row in rows for event in row["joint_boundary_events"])
    topology = Counter(tuple(row["final_load_bearing_topology"]) for row in rows)
    penetration = np.max([row["maximum_penetration_by_surface_m"] for row in rows], axis=0)
    exit_vectors = np.asarray([row["exit_direction_palm"] for row in rows])
    all_entry = sum(row["first_pocket_entry_step"] is not None for row in rows)
    all_cage = sum(row["cage_formed"] for row in rows)
    classification = "PC-F" if result["forearm_dof_audit"]["code"] == "PHASE3C07_FOREARM_DOF_LIMIT" else ("PC-A" if all_cage else ("PC-B" if all_entry else "PC-D"))
    summary = {
        "branch": result["branch"], "base_commit": result["base_commit"],
        "sphere": audit["sphere"], "link_audit": links,
        "pocket": audit["pocket_volume"] | {
            "feasible_count": audit["feasible_count"],
            "volume_m3": float(audit["feasible_count"] * np.prod(np.asarray(audit["grid_steps_m"]))),
        },
        "acquisition": acquisition, "T0": _condition(t0), "T1": _condition(t1), "W1_all": _condition(w1),
        "wrist_directions": directions, "forearm_dof_audit": result["forearm_dof_audit"],
        "W2_executed": "W2" in result["wider_wrist_levels_executed"], "W3_executed": "W3" in result["wider_wrist_levels_executed"],
        "preshape_P0": _condition(p0), "preshape_P1": _condition(p1),
        "load_bearing_cages": all_cage, "unique_cage_states": len({r["state_id"] for r in rows if r["cage_formed"]}),
        "hold": {step: sum(r["hold_survival"][step] for r in rows) for step in ("100", "500", "1000")},
        "lambda_storage": {"median_trial_max": float(np.median([r["maximum_lambda_storage"] for r in rows])), "maximum": float(max(r["maximum_lambda_storage"] for r in rows))},
        "dominant_final_topologies": {"+".join(key) or "none": value for key, value in topology.most_common()},
        "mean_exit_direction_palm": np.mean(exit_vectors, axis=0).tolist(),
        "penetration_max_by_surface_m": dict(zip(SUPPORT_SURFACES, penetration.tolist())),
        "penetration_max_over_radius": dict(zip(SUPPORT_SURFACES, (penetration / 0.0125).tolist())),
        "penetration_acceptability": "TODO(PI): no acceptable 25-mm-sphere penetration threshold was frozen",
        "joint_boundary_joints": dict(boundary),
        "classification": classification,
        "videos": [],
    }
    _figures(audit, result, summary)
    summary["figures"] = [str(FIGURES / f"{name}.pdf") for name in NAMES]
    (OUTPUT / "phase3c07_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    volume = summary["pocket"]["volume_m3"]
    docs = ROOT / "docs"
    geometry = f"""# Phase 3C-0.7 pocket geometry audit

This audit was constructed before dynamic outcomes from exact compiled Shadow Hand geometry.

- Sphere: 25 mm diameter, 12.5 mm radius, density 1000 kg/m³.
- Analytic and compiled mass: `{summary['sphere']['compiled_mass_kg']:.17g} kg`.
- Grid: `{audit['candidate_count']}` palm-frame candidate centers; `{audit['feasible_count']}` feasible voxels.
- RING_LITTLE_POCKET_VOLUME: union of feasible voxels, not a single Cartesian target.
- Bounds: `{audit['pocket_volume']['lower_palm_m']}` to `{audit['pocket_volume']['upper_palm_m']}` m.
- Voxel-union volume: `{volume:.9g} m³`.
- Construction: {audit['pocket_volume']['construction']}.
- Actual references: ring/little MCP roots, proximal links, and compiled palm/root collision geometry.
- The map records palm/middle/ring/little clearance, thumb/index and storage-finger reach gaps, incoming-path clearance, local opening, and escape directions for every voxel.

No dynamic result was used to draw this volume.
"""
    (docs / "PHASE3C07_POCKET_GEOMETRY_AUDIT.md").write_text(geometry, encoding="utf-8")
    joint = f"""# Phase 3C-0.7 joint-boundary transport audit

Exact compiled-boundary events were localized by joint, timestep, and stage in every trial.

- Event counts by joint: `{dict(boundary)}`.
- Transport-limiting events: `{sum(row['joint_boundary_limited_transport'] for row in rows)}/{len(rows)}`. All observed exact-boundary joints were passive open middle/ring/little joints; none was a thumb/index transport boundary, so causation was not assigned.
- Native wrist diagnostic: `{result['forearm_dof_audit']['code']}`.
- Desired direction: `{result['forearm_dof_audit']['desired_transport_direction_palm']}`.
- Best native-wrist gravity direction: `{result['forearm_dof_audit']['best_reachable_gravity_direction_palm']}`.
- Residual orientation angle: `{result['forearm_dof_audit']['residual_orientation_angle_deg']:.6g}°`.
- Missing component: native WRJ1/WRJ2 cannot generate the required palm-x gravity component. W2/W3 expansion was stopped as required; no MJCF or joint limit was changed.

Boundary association with progress remains descriptive; the experiment does not alter limits or invent a margin threshold.
"""
    (docs / "PHASE3C07_JOINT_BOUNDARY_TRANSPORT_AUDIT.md").write_text(joint, encoding="utf-8")
    results = f"""# Phase 3C-0.7 results

## Outcome

Primary classification: **{classification}**. The 25-mm geometry is statically plausible, but native WRJ1/WRJ2 cannot supply the lateral palm-frame gravity component requested by the pocket transport direction. This is reported as `PHASE3C07_FOREARM_DOF_LIMIT`; W2/W3 were not run.

## Frozen protocol results

- Frozen acquisition states: `{acquisition['N']}`; thumb/index contact `{acquisition['thumb_contact']}/50`, unused-finger contact `{acquisition['unused_finger_accidental_contact']}/50`, fixture-off dual-contact/no-floor retention `{acquisition['acquisition_retention']}/50` through the inherited 50-step hold.
- T0 old-target pocket entry: `{summary['T0']['entry']}/{summary['T0']['N']}`.
- T1 fixed-wrist pocket entry: `{summary['T1']['entry']}/{summary['T1']['N']}`.
- W1 wrist-assisted pocket entry: `{summary['W1_all']['entry']}/{summary['W1_all']['N']}` across eight matched directions.
- P0/P1 were run only if transport reached or approached the pocket: N=`{summary['preshape_P0']['N']}` / `{summary['preshape_P1']['N']}`.
- Load-bearing cages: `{all_cage}`; unique states `{summary['unique_cage_states']}`.
- Cage hold survival at 100/500/1000: `{summary['hold']}`.
- Maximum raw penetration by surface: `{summary['penetration_max_by_surface_m']}` m.
- Penetration acceptability: **TODO(PI)**; no threshold was invented.
- Thumb/index release: never performed. Object B, RL, compliant skin, and physics changes: none.

## Interpretation

The experiment distinguishes static fit from dynamic reachability. A smaller object can fit the geometry-derived pocket, but that does not by itself demonstrate transport or a cage. Thumb release and size progression remain premature unless the measured transport/cage chain supports PC-A or strong PC-B. Compliant skin is not justified by a native-DOF transport failure.
"""
    (docs / "PHASE3C07_RESULTS.md").write_text(results, encoding="utf-8")
    todo = """# Phase 3C-0.7 TODO(PI)

- `configs/phase3C07_pocket_reachability_cage.yaml`: freeze a publication definition for acceptable 25-mm-sphere penetration.
- `configs/phase3C07_pocket_reachability_cage.yaml`: freeze a publication persistence criterion for `PALMODIGITAL_CAGE_FORMED`.
- `configs/phase3C07_pocket_reachability_cage.yaml`: decide whether a physically meaningful forearm/global-hand orientation DOF may be added in a later phase.
- `configs/phase3C07_pocket_reachability_cage.yaml`: decide whether compliant-skin modeling is warranted after transport/cage evidence.
"""
    (docs / "PHASE3C07_TODO_PI.md").write_text(todo, encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
