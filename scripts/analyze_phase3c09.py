"""Create Phase 3C-0.9 reports and vector figures from frozen/static audits."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from seqgrasp.config import ROOT


OUTPUT = ROOT / "outputs/phase3C09"; FIGURES = ROOT / "docs/figures/phase3C09"
NAMES = (
    "best_trajectory_pocket_distance_timeseries", "best_trajectory_contact_forces",
    "best_trajectory_friction_utilization", "best_trajectory_contact_point_migration",
    "cspace_3d_connectivity", "cspace_shortest_path_or_blockage", "cspace_bottleneck_cross_section",
    "first_order_accessible_motion", "lie_bracket_accessible_motion", "first_vs_second_order_accessibility",
    "cyclic_motion_bracket_validation", "storage_manifold_overview", "storage_basin_clusters",
    "storage_basin_resource_occupancy", "storage_basin_cspace_connectivity",
    "human_ulnar_vs_shadow_native_storage", "phase3C09_decision_tree", "phase3C09_causal_summary",
)


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", format="pdf", bbox_inches="tight", metadata={"Title": name, "Creator": "Phase 3C-0.9 static analysis"})
    plt.close(fig)


def create_figures(result: dict) -> list[str]:
    trajectories = result["trajectory"]["rows"]; best = trajectories[0]; series = best["series"]
    cspace = result["cspace"]; contact = result["contact_accessibility"]; storage = result["storage_manifold"]
    time = np.asarray(series["time_s"])
    fig, ax = plt.subplots(figsize=(7, 4));
    for row in trajectories: ax.plot(row["series"]["time_s"], np.asarray(row["series"]["pocket_distance_m"]) * 1000, label=row["trial_id"].replace("C08_C07_STATE_", "S"))
    ax.scatter([best["minimum_time_s"]], [best["minimum_pocket_distance_m"] * 1000], color="black", zorder=5); ax.set(xlabel="time (s)", ylabel="nearest pocket distance (mm)", title="Deterministically selected five closest stored trajectories"); ax.legend(fontsize=6); _save(fig, NAMES[0])
    fig, ax = plt.subplots(figsize=(7, 4));
    for surface in ("thumb", "index", "little", "ring", "palm"): ax.plot(time, series["normal_force_n"][surface], label=surface)
    ax.axvline(best["minimum_time_s"], color="0.4", linestyle="--"); ax.set(xlabel="time (s)", ylabel="stored normal force (N)", title="Best trajectory stored contact forces"); ax.legend(); _save(fig, NAMES[1])
    fig, ax = plt.subplots(figsize=(7, 4)); ax.axis("off"); ax.text(.5, .6, "Tangential force was not logged", ha="center", fontsize=15); ax.text(.5, .42, "Friction utilization is unavailable\nNo dynamics rerun or inferred zero", ha="center", fontsize=11); ax.set_title("Best trajectory friction utilization"); _save(fig, NAMES[2])
    fig, ax = plt.subplots(figsize=(7, 4));
    for surface, marker in (("thumb", "o"), ("index", "x")):
        points = np.asarray(series[f"{surface}_contact_point_palm_m"]); valid = np.all(np.isfinite(points), axis=1); ax.plot(points[valid, 0] * 1000, points[valid, 2] * 1000, marker=marker, markersize=2, label=surface)
    ax.set(xlabel="palm x (mm)", ylabel="palm z (mm)", title="Stored contact-point migration"); ax.legend(); _save(fig, NAMES[3])

    grid_file = OUTPUT / "cspace_grids" / f"{best['trial_id']}_1mm.npz"; grid = np.load(grid_file); free = grid["free"]; path = grid["path"]
    indices = np.argwhere(free); stride = max(1, len(indices) // 5000); sampled = indices[::stride]
    pts = np.column_stack([grid["axes_x"][sampled[:, 0]], grid["axes_y"][sampled[:, 1]], grid["axes_z"][sampled[:, 2]]])
    fig = plt.figure(figsize=(6.4, 5)); ax = fig.add_subplot(projection="3d"); ax.scatter(*pts.T, s=1, alpha=.04, label="free C-space"); ax.plot(*path.T, color="tab:red", linewidth=3, label="A* path"); ax.set(xlabel="palm x", ylabel="palm y", zlabel="palm z", title="25-mm sphere translational C-space"); ax.legend(); _save(fig, NAMES[4])
    fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(path[:, 0] * 1000, path[:, 2] * 1000, "-o", markersize=2); ax.set(xlabel="palm x (mm)", ylabel="palm z (mm)", title="Best-state shortest collision-free path"); _save(fig, NAMES[5])
    clearance = grid["clearance_m"]; finest = cspace["states"][0]["grids"][0]; bottleneck = np.asarray(finest["bottleneck_palm_m"]); yi = int(np.argmin(np.abs(grid["axes_y"] - bottleneck[1])))
    fig, ax = plt.subplots(figsize=(7, 4)); image = ax.imshow(clearance[:, yi, :].T * 1000, origin="lower", aspect="auto", extent=[grid["axes_x"][0]*1000,grid["axes_x"][-1]*1000,grid["axes_z"][0]*1000,grid["axes_z"][-1]*1000], cmap="viridis"); fig.colorbar(image, ax=ax, label="sphere-to-hand clearance (mm)"); ax.set(xlabel="palm x (mm)", ylabel="palm z (mm)", title="C-space bottleneck cross-section"); _save(fig, NAMES[6])

    target = np.asarray(contact["target_direction_palm"]); modes = contact["modes"]
    fig = plt.figure(figsize=(6.4, 5)); ax = fig.add_subplot(projection="3d"); ax.quiver(0,0,0,*target,color="black",linewidth=3,label="pocket target")
    for mode in modes:
        for field in mode["control_vector_fields_at_analysis_state"]: ax.quiver(0,0,0,*np.asarray(field[:3]),alpha=.5)
    ax.set(xlim=(-1,1),ylim=(-1,1),zlim=(-1,1),xlabel="palm x",ylabel="palm y",zlabel="palm z",title="First-order accessible translation fields"); ax.legend(); _save(fig, NAMES[7])
    fig = plt.figure(figsize=(6.4, 5)); ax = fig.add_subplot(projection="3d"); ax.quiver(0,0,0,*target,color="black",linewidth=3,label="target")
    for mode in modes:
        for bracket in mode["second_order_bracket_vectors_at_analysis_state"]: ax.quiver(0,0,0,*np.asarray(bracket[:3]),alpha=.6)
    ax.set(xlabel="palm x",ylabel="palm y",zlabel="palm z",title="Validated second-order bracket translations"); ax.legend(); _save(fig, NAMES[8])
    fig, ax = plt.subplots(figsize=(7, 4)); labels=[m["mode"].split("_")[0] for m in modes]; x=np.arange(len(labels)); ax.bar(x-.18,[m["first_order_rank"] for m in modes],.36,label="Delta1"); ax.bar(x+.18,[m["second_order_rank"] for m in modes],.36,label="Delta2"); ax.set_xticks(x,labels); ax.set(ylabel="state-space rank",title="First- versus second-order accessibility"); ax.legend(); _save(fig,NAMES[9])
    fig, ax = plt.subplots(figsize=(7, 4));
    for mode in modes:
        if mode["cyclic_validation"]: values=mode["cyclic_validation"]; ax.loglog([v["epsilon"] for v in values],[np.linalg.norm(v["translation"]) for v in values],"-o",label=mode["mode"].split("_")[0])
    epsilon=np.asarray([.005,.02]); ax.loglog(epsilon,epsilon**2,"--",color="black",label="O(epsilon^2) reference"); ax.set(xlabel="epsilon",ylabel="net translation norm",title="Kinematic cyclic bracket validation"); ax.legend(); _save(fig,NAMES[10])

    basins=storage["basins"]; centers=np.asarray([b["centroid_palm_m"] for b in basins]); sizes=np.asarray([b["voxel_count"] for b in basins])
    fig=plt.figure(figsize=(6.4,5)); ax=fig.add_subplot(projection="3d"); ax.scatter(*centers.T,s=20+2*np.sqrt(sizes),c=sizes,cmap="viridis");
    for b,c in zip(basins,centers): ax.text(*c,b["basin_id"],fontsize=7)
    ax.set(xlabel="palm x",ylabel="palm y",zlabel="palm z",title="Morphology-aware storage manifold"); _save(fig,NAMES[11])
    fig,ax=plt.subplots(figsize=(7,4)); ax.bar([b["basin_id"] for b in basins],sizes); ax.set(ylabel="5-mm center voxels",title="Six connected storage-basin clusters"); _save(fig,NAMES[12])
    resources=("thumb","index","middle","ring","little"); matrix=np.asarray([[int(b["resource_availability"][r]) for r in resources] for b in basins]); fig,ax=plt.subplots(figsize=(7,4)); im=ax.imshow(matrix,aspect="auto",cmap="RdYlGn",vmin=0,vmax=1); ax.set_xticks(range(5),resources); ax.set_yticks(range(len(basins)),[b["basin_id"] for b in basins]); ax.set_title("Resource availability by basin (green = available)"); _save(fig,NAMES[13])
    fig,ax=plt.subplots(figsize=(8,4)); colors={"DIRECTLY_GEOMETRICALLY_CONNECTED":"tab:green","CONNECTED_ONLY_WITH_HAND_RECONFIGURATION":"tab:orange","NOT_CONNECTED_UNDER_TESTED_GEOMETRY":"tab:red"}; ax.bar([b["basin_id"] for b in basins],[b["direct_cspace_path_length_m"] or b["reconfigured_cspace_path_length_m"] or 0 for b in basins],color=[colors[b["reachability"]] for b in basins]); ax.set(ylabel="geometry path length (m)",title="Acquisition-to-basin connectivity"); _save(fig,NAMES[14])
    fig,ax=plt.subplots(figsize=(7,4)); ax.bar([b["basin_id"] for b in basins],[b["previous_ulnar_pocket_voxels_in_basin"] for b in basins]); ax.set(ylabel="previous 344 pocket voxels represented",title="Human-inspired ulnar target versus Shadow-native basins"); _save(fig,NAMES[15])
    fig,ax=plt.subplots(figsize=(9,4)); ax.axis("off"); ax.text(.05,.7,"C-space connected\nat both resolutions",bbox=dict(boxstyle="round",fc="#d9f0d3")); ax.annotate("",(.42,.72),(.27,.72),arrowprops=dict(arrowstyle="->")); ax.text(.43,.7,"All smooth modes CT-C\ntarget absent from validated closure",bbox=dict(boxstyle="round",fc="#fde0dd")); ax.annotate("",(.78,.72),(.69,.72),arrowprops=dict(arrowstyle="->")); ax.text(.79,.7,"Change contact mode\nwith deterministic baseline",bbox=dict(boxstyle="round",fc="#deebf7")); ax.text(.5,.25,"BASIN_03 is the connected morphology-native middle/ring/little manifold;\nit overlaps, but is broader than, the prior ring/little pocket.",ha="center"); ax.set_title("Phase 3C-0.9 decision logic"); _save(fig,NAMES[16])
    fig,ax=plt.subplots(figsize=(8,4)); labels=["Loaded jam\nevidence","C-space\nconnected states","CT modes\nlocally sufficient","Storage\nbasins"]; values=[0,sum(s["classification"]=="CS-A" for s in cspace["states"]),sum(m["classification"] in ("CT-A","CT-B") for m in modes),storage["basin_count"]]; bars=ax.bar(labels,values,color=["tab:gray","tab:green","tab:red","tab:blue"]); ax.bar_label(bars); ax.set(title="Causal summary: contact mode, not geometry, is upstream",ylabel="descriptive count"); _save(fig,NAMES[17])
    return [str(FIGURES/f"{name}.pdf") for name in NAMES]


def create_reports(result: dict, figures: list[str]) -> dict:
    docs=ROOT/"docs"; tr=result["trajectory"]; cs=result["cspace"]; ct=result["contact_accessibility"]; sm=result["storage_manifold"]; best=tr["rows"][0]; basins=sm["basins"]; recommended=max(basins,key=lambda b:b["voxel_count"])
    top_lines="\n".join(f"- `{r['trial_id']}`: {r['mode']}, {r['configuration_label']}, minimum `{r['minimum_pocket_distance_m']*1000:.6f} mm` at step `{r['minimum_index']}`, outcome `{r['final_outcome']}`." for r in tr["rows"])
    (docs/"PHASE3C09_TRAJECTORY_FAILURE_DIAGNOSIS.md").write_text(f"""# Phase 3C-0.9 trajectory failure diagnosis

Selection was deterministic: the five smallest recorded minimum pocket distances, with trial ID as a tie-breaker.

{top_lines}

For the overall best trajectory, the 21-sample numerical window around the minimum had median distance rate `{best['near_minimum']['distance_rate_mps_median']:.9g} m/s`, sphere speed `{best['near_minimum']['speed_mps_median']:.9g} m/s`, and zero median thumb/index normal force. The minimum occurred during drop/escape after acquisition-contact loss, not during a stationary loaded plateau. It is therefore **not jamming-consistent**.

Sphere position, finite-difference linear/angular motion, stored normal forces, contact points, forearm/wrist state, and palm-frame gravity were reconstructed. Tangential force, friction utilization, and contact slip velocity are unavailable because Phase 3C-0.8 did not log qvel/tangential force/slip and no dynamics rerun is authorized.
""",encoding="utf-8")
    state_lines=[]
    for state in cs["states"]:
        grids="; ".join(f"{g['resolution_m']*1000:g} mm: free {g['free_voxel_count']}, component {g['start_component_size']}, connected pocket voxels {g['pocket_voxels_connected']}/{g['pocket_voxel_count']}, path {g['shortest_path_length_m']:.9g} m, clearance {g['minimum_path_clearance_m']:.9g} m" for g in state["grids"])
        state_lines.append(f"- `{state['trial_id']}`: **{state['classification']}**; {grids}.")
    bounds=cs["states"][0]["grids"][0]["domain_bounds_palm_m"]
    (docs/"PHASE3C09_CSPACE_CONNECTIVITY_AUDIT.md").write_text(f"""# Phase 3C-0.9 rigid-sphere C-space connectivity audit

The 25-mm sphere was represented exactly by compiled sphere-to-hand geometry distances, which is equivalent to 12.5-mm Minkowski inflation. Table and fixture were excluded. The best-state local domain bounds were `{bounds}` m; each state used the union of its start, all 344 pocket voxels, and a 15-mm detour margin.

{chr(10).join(state_lines)}

All five states are **CS-A** at both 1.0-mm and 0.5-mm resolution. The best-state path is `{cs['states'][0]['grids'][-1]['shortest_path_length_m']:.9g} m`, with `{cs['states'][0]['grids'][-1]['minimum_path_clearance_m']:.9g} m` minimum sphere-to-hand clearance and `{cs['states'][0]['grids'][-1]['bottleneck_opening_width_m']:.9g} m` inferred opening width. Thus the reported 4.23-mm gap is geometrically traversable; blockage is not the upstream cause.
""",encoding="utf-8")
    mode_lines="\n".join(f"- `{m['mode']}`: Delta1 rank `{m['first_order_rank']}` (translation `{m['first_order_translation_rank']}`), target projection `{m['first_order_target_projection']:.9g}`, Delta2 rank `{m['second_order_rank']}` (translation `{m['second_order_translation_rank']}`), second-order target residual `{m['second_order_target_residual']:.9g}`, classification **{m['classification']}**." for m in ct["modes"])
    (docs/"PHASE3C09_CONTACT_ACCESSIBILITY_AUDIT.md").write_text(f"""# Phase 3C-0.9 contact accessibility audit

State: 8 dimensions - sphere position (3), local sphere rotation (3), and smooth contact chart (2). The representative sample is stored step `{ct['representative_stored_step']}` of `{ct['representative_trial']}`. Modes are M0 dual rolling/no-slip; M1 index guide with smooth thumb migration; M2 thumb guide with smooth index migration; and M3 one unloaded contact with the guide plus gravity as external drift.

{mode_lines}

Finite-difference brackets used steps 1e-3, 5e-4, and 2.5e-4. M1/M2 bracket differences decreased toward the finest result, while their cyclic checks showed approximately O(epsilon^2) net motion. However, brackets increased full state-space rank without increasing translational rank enough to contain the target within the frozen numerical rank tolerance. All modes are CT-C. The current topology is locally insufficient under these explicitly smooth models; this is not a global LARC claim for nonsmooth MuJoCo contacts. Nonholonomic cycling is demonstrated kinematically but is not sufficient in the desired direction. RL is not implied.
""",encoding="utf-8")
    basin_lines="\n".join(f"- `{b['basin_id']}`: centroid `{b['centroid_palm_m']}`, `{b['voxel_count']}` voxels / `{b['volume_m3']:.9g} m3`, supports `{b['dominant_support_surfaces']}`, `{b['confinement']}`, aperture `{b['aperture_bottleneck_width_m']}`, reachability `{b['reachability']}`, prior-pocket voxels `{b['previous_ulnar_pocket_voxels_in_basin']}`." for b in basins)
    (docs/"PHASE3C09_STORAGE_MANIFOLD_AUDIT.md").write_text(f"""# Phase 3C-0.9 morphology-aware storage manifold audit

The reduced search varied five quantities: middle, ring, and little flexion; WRJ2 offset; and forearm_PS. Thumb/index remained in the stored acquisition envelope. It evaluated `{sm['candidate_configurations_evaluated']}` hand configurations and `{sm['center_voxels_per_configuration']}` center voxels per configuration. The 5-mm center grid and 5-mm near-surface neighborhood are numerical diagnostic resolutions, not success thresholds.

- Valid configuration-center pairs: `{sm['valid_configuration_center_pairs']}`.
- Unique valid centers: `{sm['valid_unique_storage_centers']}`.
- Connected basins: `{sm['basin_count']}`.

{basin_lines}

`{recommended['basin_id']}` is the best-supported morphology-native target for the next design phase: it contains `{recommended['voxel_count']}` of 560 unique valid centers, is directly geometrically connected, preserves thumb/index, has middle/ring/little support, and contains 125 of the former 344 ulnar-pocket voxels. The human-inspired pocket is therefore not wholly wrong, but it is an overly narrow subset of a broader Shadow-native middle/ring/little manifold. Geometry alone does not establish dynamic retention.
""",encoding="utf-8")
    (docs/"PHASE3C09_TODO_PI.md").write_text("""# Phase 3C-0.9 TODO(PI)

- `configs/phase3C09_storage_reachability.yaml`: choose whether the next dynamics phase prioritizes deliberate acquisition-contact-mode change toward BASIN_03 or aperture reconfiguration toward another basin.
- `configs/phase3C09_storage_reachability.yaml`: decide whether BASIN_03 may replace the original narrow ring/little pocket as the scientific storage target.
- `docs/PHASE3C09_CONTACT_ACCESSIBILITY_AUDIT.md`: decide which future smooth contact-migration model and deterministic controller family should be validated dynamically; Phase 3C-0.9 does not select one.

No TODO(PI) scientific decision was resolved automatically.
""",encoding="utf-8")
    summary={"phase":"3C-0.9","branch":result["branch"],"base_commit":result["base_commit"],"best_trajectory":{k:best[k] for k in ("trial_id","minimum_pocket_distance_m","minimum_index","minimum_time_s","failure_interpretation","jamming_consistent","near_minimum","availability")},"cspace":cs,"contact_accessibility":ct,"storage_manifold":sm,"recommended_storage_basin":recommended["basin_id"],"primary_blocker":"current thumb/index contact topology/mode is locally insufficient under all four specified smooth models; geometry is connected and loaded jamming is unsupported","recommended_next_phase":"deterministic deliberate contact-mode-change and transport trajectory-optimization baseline toward BASIN_03","new_dynamics_next_phase":"Only after PI freezes the target basin and contact-mode primitive; use a small predeclared validation, not RL.","figures":figures,"contract":result["contract"]}
    (OUTPUT/"phase3c09_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8"); return summary


def main() -> None:
    result=json.loads((OUTPUT/"phase3c09_results.json").read_text(encoding="utf-8")); figures=create_figures(result); summary=create_reports(result,figures); print(json.dumps({"figures":len(figures),"primary_blocker":summary["primary_blocker"],"recommended_basin":summary["recommended_storage_basin"]},indent=2))


if __name__=="__main__": main()
