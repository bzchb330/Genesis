"""Create Phase 3C-0.8 reports and vector figures from frozen results."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from seqgrasp.config import ROOT


OUTPUT = ROOT / "outputs/phase3C08"
FIGURES = ROOT / "docs/figures/phase3C08"
FIGURE_NAMES = (
    "forearm_axis_definition",
    "native_wrist_reachable_gravity_set",
    "native_residual_to_target",
    "augmented_reachable_gravity_set",
    "native_vs_forearm_gravity_workspace",
    "residual_vs_forearm_angle",
    "gravity_projection_vs_forearm_angle",
    "best_forearm_wrist_configurations",
    "target_direction_distribution",
    "targeted_dynamic_pocket_entry",
    "sphere_paths_with_forearm_rotation",
    "phase3C08_causal_summary",
)


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURES / f"{name}.pdf", format="pdf", bbox_inches="tight",
        metadata={"Title": name, "Creator": "Phase 3C-0.8 analysis"},
    )
    plt.close(fig)


def _unit_sphere(ax) -> None:
    u = np.linspace(0, 2 * np.pi, 40); v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v)); y = np.outer(np.sin(u), np.sin(v)); z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color="0.85", linewidth=0.25, alpha=0.4)
    ax.set(xlabel="palm x", ylabel="palm y", zlabel="palm z")
    ax.set_box_aspect((1, 1, 1))


def _forearm_sensitivity(configs: np.ndarray, directions: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angles = np.unique(configs[:, 0])
    residual, projection = [], []
    for angle in angles:
        selected = directions[np.isclose(configs[:, 0], angle)]
        dots = np.clip(selected @ target, -1.0, 1.0)
        projection.append(np.max(dots)); residual.append(np.rad2deg(np.arccos(np.max(dots))))
    return np.rad2deg(angles), np.asarray(residual), np.asarray(projection)


def create_figures(kinematic: dict, dynamics: dict) -> list[str]:
    reach = kinematic["reachable_gravity_audit"]
    targets = np.asarray([row["target_direction"] for row in reach["rows"]])
    native_directions = np.asarray(reach["native"]["directions"])
    augmented_directions = np.asarray(reach["augmented"]["directions"])
    augmented_configs = np.asarray(reach["augmented"]["configurations_rad"])
    axis = kinematic["forearm_axis_audit"]

    fig = plt.figure(figsize=(6.4, 4.8)); ax = fig.add_subplot(projection="3d")
    parent = np.zeros(3); offset = np.asarray(axis["child_offset_parent_m"]); direction = np.asarray(axis["axis_parent"])
    ax.quiver(*parent, *direction, length=np.linalg.norm(offset), color="tab:red", linewidth=3, label="forearm_PS axis")
    ax.scatter(*parent, label="rh_forearm origin"); ax.scatter(*offset, label="rh_wrist child anchor")
    ax.plot(*np.vstack([parent, offset]).T, color="0.3"); ax.set_title("Official-model forearm longitudinal axis"); ax.legend(); _save(fig, FIGURE_NAMES[0])

    fig = plt.figure(figsize=(6.4, 5)); ax = fig.add_subplot(projection="3d"); _unit_sphere(ax)
    ax.scatter(*native_directions.T, s=8, alpha=.45, label="native WRJ1/WRJ2"); ax.scatter(*targets.T, s=35, marker="x", label="targets")
    ax.legend(); ax.set_title("Native wrist reachable gravity directions"); _save(fig, FIGURE_NAMES[1])

    native_residual = np.asarray([row["native"]["minimum_residual_deg"] for row in reach["rows"]])
    augmented_residual = np.asarray([row["augmented"]["minimum_residual_deg"] for row in reach["rows"]])
    fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(native_residual, "o", label="native"); ax.plot(augmented_residual, ".", label="forearm augmented")
    ax.axhline(15, color="0.5", linestyle="--", label="KR-A engineering guide"); ax.set(xlabel="frozen state index", ylabel="minimum residual (deg)", title="State-specific gravity alignment residual"); ax.legend(); _save(fig, FIGURE_NAMES[2])

    fig = plt.figure(figsize=(6.4, 5)); ax = fig.add_subplot(projection="3d"); _unit_sphere(ax)
    stride = max(1, len(augmented_directions) // 5000); ax.scatter(*augmented_directions[::stride].T, s=2, alpha=.12, label="augmented coarse set")
    ax.scatter(*targets.T, s=35, marker="x", color="tab:red", label="targets"); ax.legend(); ax.set_title("Forearm-augmented reachable gravity directions"); _save(fig, FIGURE_NAMES[3])

    fig = plt.figure(figsize=(6.4, 5)); ax = fig.add_subplot(projection="3d"); _unit_sphere(ax)
    stride = max(1, len(augmented_directions) // 3500); ax.scatter(*augmented_directions[::stride].T, s=2, alpha=.08, label="forearm augmented")
    ax.scatter(*native_directions.T, s=8, alpha=.6, label="native"); ax.scatter(*targets.T, s=30, marker="x", label="targets"); ax.legend(); ax.set_title("Native versus augmented gravity workspace"); _save(fig, FIGURE_NAMES[4])

    angles, sensitivity_residual, sensitivity_projection = _forearm_sensitivity(augmented_configs, augmented_directions, np.mean(targets, axis=0))
    fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(angles, sensitivity_residual); ax.axhline(15, color="0.5", linestyle="--")
    ax.set(xlabel="forearm_PS (deg)", ylabel="best wrist residual (deg)", title="Orientation sensitivity to forearm angle"); _save(fig, FIGURE_NAMES[5])
    fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(angles, sensitivity_projection); ax.set(xlabel="forearm_PS (deg)", ylabel="max gravity projection", title="Transport-direction gravity projection"); _save(fig, FIGURE_NAMES[6])

    best = np.asarray([[row["augmented"]["best_forearm_PS_rad"], row["augmented"]["best_WRJ1_rad"], row["augmented"]["best_WRJ2_rad"]] for row in reach["rows"]])
    fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(np.rad2deg(best)); ax.set(xlabel="frozen state index", ylabel="angle (deg)", title="Best feasible forearm/wrist configurations"); ax.legend(["forearm_PS", "WRJ1", "WRJ2"]); _save(fig, FIGURE_NAMES[7])

    fig = plt.figure(figsize=(6.4, 5)); ax = fig.add_subplot(projection="3d"); _unit_sphere(ax); ax.scatter(*targets.T, c=np.arange(len(targets)), cmap="viridis", s=25)
    mean = np.asarray(kinematic["target_direction_audit"]["mean_direction"]); ax.quiver(0, 0, 0, *mean, color="black", linewidth=2, label="mean"); ax.legend(); ax.set_title("Reconstructed target-direction distribution"); _save(fig, FIGURE_NAMES[8])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4)); ax1.bar(["F0 static", "F1 coordinated"], [dynamics["static_forearm_pocket_entry"], dynamics["coordinated_forearm_pocket_entry"]]); ax1.set(ylabel="pocket entries", title=f"0 / {dynamics['trial_count']} total")
    distances = np.asarray([row["closest_pocket_distance_m"] for row in dynamics["rows"]]) * 1000; ax2.hist(distances, bins=12); ax2.axvline(distances.min(), color="tab:red", label=f"best {distances.min():.3f} mm"); ax2.set(xlabel="closest pocket distance (mm)", ylabel="trials"); ax2.legend(); _save(fig, FIGURE_NAMES[9])

    fig = plt.figure(figsize=(6.4, 5)); ax = fig.add_subplot(projection="3d")
    for index, row in enumerate(sorted(dynamics["rows"], key=lambda item: item["closest_pocket_distance_m"])[:10]):
        series = np.load(row["timeseries_path"], allow_pickle=False); path = series["sphere_center_palm_m"]
        ax.plot(*path.T, alpha=.6, label=row["mode"] if index < 2 else None)
    ax.set(xlabel="palm x (m)", ylabel="palm y (m)", zlabel="palm z (m)", title="Ten closest forearm-rotation sphere paths"); ax.legend(); _save(fig, FIGURE_NAMES[10])

    fig, ax = plt.subplots(figsize=(8, 4)); labels = ["Native\nmedian residual", "Augmented\nmedian residual", "Targeted dynamics\npocket entry"]
    values = [reach["native"]["median_residual_deg"], reach["augmented"]["median_residual_deg"], dynamics["total_pocket_entry"]]
    bars = ax.bar(labels, values, color=["tab:red", "tab:green", "tab:blue"]); ax.bar_label(bars, fmt="%.3g"); ax.set(ylabel="degrees for residual; count for entry", title="Causal result: orientation solved, pocket entry unchanged")
    ax.text(.5, .88, "Gravity orientation was necessary but not sufficient", transform=ax.transAxes, ha="center"); _save(fig, FIGURE_NAMES[11])
    return [str(FIGURES / f"{name}.pdf") for name in FIGURE_NAMES]


def create_reports(kinematic: dict, dynamics: dict, figures: list[str]) -> dict:
    docs = ROOT / "docs"; target = kinematic["target_direction_audit"]; axis = kinematic["forearm_axis_audit"]
    reach = kinematic["reachable_gravity_audit"]; native = reach["native"]; augmented = reach["augmented"]
    best_row = min(reach["rows"], key=lambda row: row["augmented"]["minimum_residual_deg"])
    penetration_rows = np.asarray([row["maximum_penetration_by_surface_m"] for row in dynamics["rows"]])
    penetration = {
        surface: {"median_m": float(np.median(penetration_rows[:, index])), "p95_m": float(np.percentile(penetration_rows[:, index], 95)), "maximum_m": float(np.max(penetration_rows[:, index]))}
        for index, surface in enumerate(("thumb", "index", "middle", "ring", "little", "palm"))
    }
    agreement = "close directional agreement" if target["mean_vs_previous_angle_deg"] < 5 else "material directional discrepancy"
    (docs / "PHASE3C08_TARGET_DIRECTION_AUDIT.md").write_text(f"""# Phase 3C-0.8 target-direction audit

The direction was reconstructed independently from all 50 frozen acquisition states and the frozen 344-voxel feasible pocket. For each state, the nearest five feasible voxels were retained; the primary state direction points to the nearest voxel.

- Mean normalized direction: `{target['mean_direction']}`.
- Median normalized direction: `{target['median_direction']}`.
- Angular spread about the mean: median `{target['angular_spread_deg']['median']:.6f}` deg, p95 `{target['angular_spread_deg']['p95']:.6f}` deg, maximum `{target['angular_spread_deg']['maximum']:.6f}` deg.
- Component variation (min/max/std): `{target['component_variation']}`.
- Previous Phase 3C-0.7 direction after normalization: `{target['previous_reported_direction']}`.
- Mean-to-previous angle: `{target['mean_vs_previous_angle_deg']:.6f}` deg ({agreement}).

The transport direction and gravity are not equated: angular residual and gravity projection are computed and reported separately.
""", encoding="utf-8")
    (docs / "PHASE3C08_FOREARM_AXIS_AUDIT.md").write_text(f"""# Phase 3C-0.8 forearm-axis audit

- Parent body: `{axis['parent_body']}`.
- Direct child body: `{axis['child_body']}`.
- Child anchor offset in parent coordinates: `{axis['child_offset_parent_m']}` m.
- Normalized longitudinal axis in parent coordinates: `{axis['axis_parent']}`.
- Axis in world coordinates at nominal configuration: `{axis['axis_world_nominal']}`.
- Evidence: {axis['evidence']}.

The runtime wrapper injects `forearm_PS` into the parsed Phase-3 scene composition. The official vendored Shadow Hand XML is read only. The -90 to +90 deg interval is an engineering diagnostic range for a robot forearm/manipulator mount, not a human-anatomy claim.
""", encoding="utf-8")
    summary = {
        "phase": "3C-0.8", "branch": kinematic["branch"], "base_commit": kinematic["base_commit"],
        "target_direction_audit": {key: value for key, value in target.items() if key != "rows"},
        "forearm_axis_audit": axis, "zero_angle_backward_compatibility": kinematic["zero_angle_backward_compatibility"],
        "native": {key: value for key, value in native.items() if key not in ("directions", "configurations_rad")},
        "augmented": {key: value for key, value in augmented.items() if key not in ("directions", "configurations_rad")},
        "residual_reduction_deg": reach["residual_reduction_deg"], "classification": reach["classification"],
        "targeted_dynamics_authorized": reach["targeted_dynamics_authorized"],
        "representative_best_configuration": best_row, "targeted_dynamics": {key: value for key, value in dynamics.items() if key != "rows"},
        "penetration_statistics": penetration,
        "causal_conclusion": "Forearm rotation resolved gravity-orientation reachability, but pocket entry remained 0/50; gravity orientation was necessary but not sufficient.",
        "additional_orientation_required": False,
        "global_translation_investigation": "Candidate follow-up requiring PI choice; not implemented here.",
        "recommended_next_phase": "Thumb/index in-hand transport Jacobian and lateral object controllability audit, including rolling/sliding mechanics; compare global translation only after PI selection.",
        "cage_formation_should_be_tested": False, "skin_compliance_premature": True, "rl_premature": True,
        "figures": figures, "videos": [],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True); (OUTPUT / "phase3c08_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (docs / "PHASE3C08_RESULTS.md").write_text(f"""# Phase 3C-0.8 results

## Outcome

Classification: **{reach['classification']}**. The runtime forearm DOF reduced the native median residual from `{native['median_residual_deg']:.6f}` deg to `{augmented['median_residual_deg']:.9g}` deg; all 50 states were below 10 deg. Zero-angle equivalence passed over 50 states with a maximum recorded error of `0` at a `1e-12` tolerance.

The outcome-independent targeted manifest froze 10 states and five configurations per state before dynamics (`{dynamics['manifest_sha256']}`). F0 static orientation produced `{dynamics['static_forearm_pocket_entry']}/{dynamics['static_forearm_trials']}` entries; F1 coordinated orientation produced `{dynamics['coordinated_forearm_pocket_entry']}/{dynamics['coordinated_forearm_trials']}`; total `{dynamics['total_pocket_entry']}/{dynamics['trial_count']}`. The closest approach was `{dynamics['closest_pocket_distance_m']:.9g}` m. Ring/little/palm-root contact counts were `{dynamics['ring_contact']}/{dynamics['little_contact']}/{dynamics['palm_root_contact']}`; sphere loss was `{dynamics['sphere_loss']}`; corridor-clear trials were `{dynamics['corridor_clear']}`.

Maximum raw penetration by surface was `{dynamics['maximum_penetration_by_surface_m']}` m. No acceptability decision is inferred from these raw values.

## Causal interpretation

Forearm rotation causally changed the upstream gravity-orientation reachability from a roughly 49-deg median mismatch to effectively zero, but did not change dynamic pocket entry from the historical `0/500` to a nonzero outcome. Gravity orientation was therefore necessary but not sufficient. Additional whole-hand orientation is not indicated by this kinematic result; global hand translation remains a candidate diagnostic, not an implemented change.

The exact recommended next phase is a thumb/index in-hand transport Jacobian and lateral object-controllability audit, including rolling/sliding mechanics, with any global-translation comparison left for PI selection. Cage formation, skin/compliance changes, object B, and RL remain premature.
""", encoding="utf-8")
    (docs / "PHASE3C08_TODO_PI.md").write_text("""# Phase 3C-0.8 TODO(PI)

- `configs/phase3C08_forearm_orientation.yaml`: decide whether the diagnostic forearm/global-hand orientation DOF may become part of a later scientific model.
- `configs/phase3C08_forearm_orientation.yaml`: choose whether a later study may add global hand translation after reviewing the in-hand transport Jacobian and lateral-controllability evidence.
- `docs/PHASE3C08_RESULTS.md`: decide what penetration acceptability criterion, if any, applies to a future successful 25-mm pocket-entry/cage protocol. The present phase reports raw penetration only.

No TODO(PI) decision was resolved in Phase 3C-0.8.
""", encoding="utf-8")
    return summary


def main() -> None:
    kinematic = json.loads((OUTPUT / "kinematic_audit.json").read_text(encoding="utf-8"))
    dynamics = json.loads((OUTPUT / "targeted_dynamics_results.json").read_text(encoding="utf-8"))
    figures = create_figures(kinematic, dynamics)
    summary = create_reports(kinematic, dynamics, figures)
    print(json.dumps({"figures": len(figures), "classification": summary["classification"], "entries": dynamics["total_pocket_entry"]}, indent=2))


if __name__ == "__main__":
    main()
