"""Create the Phase 3C-1.0 gated-stop report and vector figures."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from seqgrasp.config import ROOT
from seqgrasp.phase3c10 import phase3c10_contract, phase_gate, scripted_stage_specification


OUTPUT = ROOT / "outputs/phase3C10"
FIGURES = ROOT / "docs/figures/phase3C10"
NAMES = (
    "old_vs_support_gated_progress",
    "phase3C08_flyby_metric_failure",
    "transfer_clearance_vs_receiver_readiness",
    "B03_actual_validation_states",
    "B03_gravity_orientation_hold_map",
    "B03_hold_survival",
    "B03_support_topology",
    "thumb_workspace_open_vs_B03",
    "index_workspace_open_vs_B03",
    "thumb_index_joint_acquisition_workspace",
    "six_stage_contact_handoff_sequence",
    "gravity_transport_receiver_decomposition",
    "scripted_sphere_speed_profile",
    "scripted_support_transfer_profile",
    "scripted_contact_topology_timeline",
    "Cspace_reference_vs_actual_sphere_path",
    "B03_storage_handoff_result",
    "phase3C10_causal_summary",
)


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURES / f"{name}.pdf",
        format="pdf",
        bbox_inches="tight",
        metadata={"Title": name, "Creator": "Phase 3C-1.0 frozen-protocol analysis"},
    )
    plt.close(fig)


def _unavailable(name: str, title: str, reason: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    ax.text(.5, .62, "NOT EXECUTED", ha="center", va="center", fontsize=22, weight="bold", color="tab:red")
    ax.text(.5, .39, reason, ha="center", va="center", fontsize=11, wrap=True)
    ax.set_title(title)
    _save(fig, name)


def _load() -> tuple[dict, dict, dict]:
    metric = json.loads((OUTPUT / "metric_repair_audit.json").read_text(encoding="utf-8"))
    manifest = json.loads((OUTPUT / "B03_validation_manifest.json").read_text(encoding="utf-8"))
    results = json.loads((OUTPUT / "B03_validation_results.json").read_text(encoding="utf-8"))
    return metric, manifest, results


def summarize(metric: dict, manifest: dict, result: dict) -> dict:
    rows = result["rows"]
    dominant = Counter(tuple(row["dominant_support_topology"]) for row in rows)
    all_penetration = []
    maximum_speeds = []
    for row in rows:
        series = np.load(row["timeseries_path"], allow_pickle=False)
        all_penetration.extend(np.asarray(series["maximum_penetration_m"], dtype=float).tolist())
        maximum_speeds.append(float(np.max(series["linear_speed_mps"])))
    penetration = np.asarray(all_penetration)
    gate = phase_gate(result["classification"])
    summary = {
        "phase": "3C-1.0",
        "branch": "codex/phase3C10-b03-scripted-contact-handoff",
        "base_commit": "834b2deefe9d7e04447bb2bb792d1322b86f70c8",
        "metric_repair": metric,
        "B03_validation": {
            "manifest_sha256": manifest["sha256"],
            "candidates": manifest["candidates"]["selected"],
            "orientations": manifest["orientations"],
            "trial_count": result["trial_count"],
            "survival_counts": result["survival_counts"],
            "classification": result["classification"],
            "approved_as_transport_target": result["approved_as_transport_target"],
            "contact_trial_counts": {
                surface: sum(bool(row[f"{surface}_contact"]) for row in rows)
                for surface in ("middle", "ring", "little", "palm")
            },
            "dominant_support_topology": list(dominant.most_common(1)[0][0]),
            "dominant_support_topology_trial_count": dominant.most_common(1)[0][1],
            "displacement_m": {
                "median": float(np.median([row["maximum_displacement_m"] for row in rows])),
                "maximum": float(np.max([row["maximum_displacement_m"] for row in rows])),
            },
            "penetration_m": {
                "initial_maximum": float(max(row["initial_maximum_penetration_m"] for row in rows)),
                "dynamic_median": float(np.median(penetration)),
                "dynamic_p95": float(np.quantile(penetration, .95)),
                "dynamic_p99": float(np.quantile(penetration, .99)),
                "dynamic_maximum": float(np.max(penetration)),
            },
            "maximum_linear_speed_mps": float(max(maximum_speeds)),
            "escape_modes": {
                "B03_STATIC_INSTABILITY": result["trial_count"],
                "B03_GRAVITY_ESCAPE": sum(not row["survival"]["1000"] for row in rows),
                "B03_GROSS_OVERLAP": sum(row["gross_overlap"] for row in rows),
            },
        },
        "phase_gate": gate,
        "resource_recovery": {
            "status": "NOT_RUN_BY_FROZEN_PROTOCOL",
            "reason": gate["reason"],
            "baseline_thumb_workspace": None,
            "B03_stored_thumb_workspace": None,
            "baseline_index_workspace": None,
            "B03_stored_index_workspace": None,
            "joint_workspace": None,
        },
        "scripted_handoff": {
            "status": "NOT_RUN_BY_FROZEN_PROTOCOL",
            "reason": gate["reason"],
            "primary_mapping": {"mode": "M2", "guide": "thumb", "unload_and_migrate": "index"},
            "stage_specification": scripted_stage_specification(),
            "dynamic_trials": 0,
            "failure_mode": "B03_STATIC_INSTABILITY",
        },
        "total_dynamic_trials": result["trial_count"],
        "contract": phase3c10_contract(),
        "recommended_next_phase": (
            "PI review of the B03-C direct-hold failure and selection of whether to revise the "
            "storage target/contact-conformity hypothesis; do not run handoff, optimization, RL, or object B yet"
        ),
    }
    (OUTPUT / "phase3c10_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def create_figures(metric: dict, manifest: dict, result: dict, summary: dict) -> list[str]:
    c08 = json.loads((ROOT / "outputs/phase3C08/targeted_dynamics_results.json").read_text(encoding="utf-8"))
    best = min(c08["rows"], key=lambda row: row["closest_pocket_distance_m"])
    series = np.load(best["timeseries_path"], allow_pickle=False)
    distance = np.asarray(series["pocket_distance_m"])
    gate_values = metric["diagnostic_speed_gates_mps"]
    valid_min = [metric["sensitivity"][str(g)]["valid_supported_minimum_distance_m"] for g in gate_values]

    fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(distance * 1000, color="0.65", label="raw pocket distance")
    ax.axhline(valid_min[0] * 1000, color="tab:blue", label="supported controlled minimum")
    ax.scatter([metric["old_raw_minimum_index"]], [metric["old_raw_minimum_distance_m"] * 1000], color="tab:red", label="unsupported fly-by")
    ax.set(xlabel="stored trajectory step", ylabel="distance (mm)", title="Raw proximity versus support-gated transport metric"); ax.legend(); _save(fig, NAMES[0])

    fig, ax1 = plt.subplots(figsize=(7, 4)); ax1.plot(distance * 1000, color="tab:blue", label="distance"); ax1.set(xlabel="stored trajectory step", ylabel="distance (mm)")
    ax2 = ax1.twinx(); ax2.scatter([metric["old_raw_minimum_index"]], [metric["raw_speed_at_minimum_mps"]], color="tab:red", label="speed at fly-by"); ax2.set_ylabel("sphere speed relative to palm (m/s)")
    ax1.set_title("Phase 3C-0.8 4.226-mm fly-by is unsupported and rejected"); _save(fig, NAMES[1])

    fig, ax = plt.subplots(figsize=(8, 4)); ax.axis("off")
    ax.text(.18,.65,"TRANSFER_CLEARANCE\nMRL clear early",ha="center",bbox=dict(boxstyle="round",fc="#deebf7")); ax.annotate("",(.47,.65),(.31,.65),arrowprops=dict(arrowstyle="->")); ax.text(.57,.65,"APPROACH\nhand-supported",ha="center",bbox=dict(boxstyle="round",fc="#fff7bc")); ax.annotate("",(.79,.65),(.69,.65),arrowprops=dict(arrowstyle="->")); ax.text(.88,.65,"RECEIVER_READY\nB03 geometry + >=2 opportunities",ha="center",bbox=dict(boxstyle="round",fc="#d9f0d3")); ax.text(.5,.25,"Clearance and receiver readiness are separate diagnostics; neither is a timer-only gate.",ha="center"); ax.set_title("Receiver-first geometry diagnostics"); _save(fig,NAMES[2])

    centers = np.asarray(manifest["candidates"]["B03_centers_palm_m"]); selected=np.asarray([c["center_palm_m"] for c in manifest["candidates"]["selected"]])
    fig=plt.figure(figsize=(6.4,5)); ax=fig.add_subplot(projection="3d"); ax.scatter(*centers.T,s=3,alpha=.15,label="B03 manifold centers"); ax.scatter(*selected.T,s=80,c=np.arange(3),cmap="tab10",label="preselected actual pairs"); ax.set(xlabel="palm x (m)",ylabel="palm y (m)",zlabel="palm z (m)",title="Actual B03 validation states"); ax.legend(); _save(fig,NAMES[3])

    checkpoints=np.asarray([10,25,50,100,200,500,1000]); held=np.asarray([[int(row["survival"][str(cp)]) for cp in checkpoints] for row in result["rows"]]); last=np.asarray([max([cp for cp in checkpoints if row["survival"][str(cp)]] or [0]) for row in result["rows"]]).reshape(3,4)
    fig,ax=plt.subplots(figsize=(8,4)); im=ax.imshow(last,aspect="auto",cmap="viridis",vmin=0,vmax=1000); fig.colorbar(im,ax=ax,label="largest passed checkpoint (steps)"); ax.set_xticks(range(4),[o["orientation_id"].replace("_","\n") for o in manifest["orientations"]]); ax.set_yticks(range(3),[c["candidate_id"] for c in manifest["candidates"]["selected"]]); ax.set_title("Frozen B03 candidate × gravity-orientation hold map"); _save(fig,NAMES[4])

    fig,ax=plt.subplots(figsize=(7,4)); counts=np.asarray([result["survival_counts"][str(cp)] for cp in checkpoints]); ax.semilogx(checkpoints,counts,"-o"); ax.set(xticks=checkpoints,ylim=(-.5,12.5),xlabel="hold checkpoint (steps)",ylabel="trials retained / 12",title="B03 direct-placement survival"); ax.grid(alpha=.25); _save(fig,NAMES[5])

    matrix=np.asarray([[int(row[f"{s}_contact"]) for s in ("middle","ring","little","palm")] for row in result["rows"]]); fig,ax=plt.subplots(figsize=(7,5)); ax.imshow(matrix,aspect="auto",cmap="Blues",vmin=0,vmax=1); ax.set_xticks(range(4),("middle","ring","little","palm")); ax.set_yticks(range(12),[row["trial_id"].replace("B03_CANDIDATE_","") for row in result["rows"]],fontsize=6); ax.set_title("Any observed B03 contact during each hold"); _save(fig,NAMES[6])

    reason="B03-C: no frozen candidate/orientation survived 100 steps; no dynamically validated stored-A state exists."
    _unavailable(NAMES[7],"Thumb workspace: open versus B03-stored A",reason)
    _unavailable(NAMES[8],"Index workspace: open versus B03-stored A",reason)
    _unavailable(NAMES[9],"Joint thumb-index acquisition workspace",reason)

    stages=scripted_stage_specification(); fig,ax=plt.subplots(figsize=(13,4.2)); ax.axis("off")
    short_names=("STABLE\nACQUISITION","WHOLE-HAND\nREORIENTATION","RECEIVER\nPRESHAPE","INDEX UNLOAD\nTHUMB GUIDE","CONTROLLED\nTRANSPORT","STORAGE\nTAKEOVER")
    positions=np.linspace(.08,.92,6)
    for i,(stage,label) in enumerate(zip(stages,short_names)):
        x=positions[i]; ax.text(x,.57,f"{stage['stage']}\n{label}",ha="center",va="center",fontsize=8,bbox=dict(boxstyle="round",fc="#eeeeee"));
        if i<5: ax.annotate("",(positions[i+1]-.065,.57),(x+.065,.57),arrowprops=dict(arrowstyle="->"))
    ax.text(.5,.16,"Specified but not executed: B03 failed the mandatory dynamic-validation gate.",ha="center",color="tab:red",weight="bold"); ax.set_title("Frozen six-stage receiver-first M2 sequence"); _save(fig,NAMES[10])

    directions=[o for o in manifest["orientations"] if o["gravity_direction_target_palm"] is not None]; fig=plt.figure(figsize=(6.4,5)); ax=fig.add_subplot(projection="3d")
    for row in directions: ax.quiver(0,0,0,*row["gravity_direction_target_palm"],label=row["orientation_id"])
    ax.set(xlim=(-1,1),ylim=(-1,1),zlim=(-1,1),xlabel="palm x",ylabel="palm y",zlabel="palm z",title="Predeclared gravity decomposition targets"); ax.legend(); _save(fig,NAMES[11])

    for index,title in ((12,"Scripted sphere-speed profile"),(13,"Scripted support-transfer profile"),(14,"Scripted contact-topology timeline"),(15,"C-space reference versus actual sphere path")):
        _unavailable(NAMES[index],title,reason+" Scripted dynamics were therefore not run.")

    fig,ax=plt.subplots(figsize=(8,4)); ax.axis("off"); ax.text(.2,.65,"STATIC B03\n544 voxels\n0 initial overlap",ha="center",bbox=dict(boxstyle="round",fc="#deebf7")); ax.annotate("",(.47,.65),(.32,.65),arrowprops=dict(arrowstyle="->")); ax.text(.6,.65,"DYNAMIC HOLD\n0/12 at 100 steps\n0/12 at 1000",ha="center",bbox=dict(boxstyle="round",fc="#fee0d2")); ax.annotate("",(.83,.65),(.73,.65),arrowprops=dict(arrowstyle="->")); ax.text(.9,.65,"B03-C\nNOT APPROVED",ha="center",bbox=dict(boxstyle="round",fc="#fcbba1")); ax.text(.5,.22,"No storage handoff was executed or claimed.",ha="center",weight="bold"); ax.set_title("B03 storage/handoff result"); _save(fig,NAMES[16])

    fig,ax=plt.subplots(figsize=(9,4)); ax.axis("off"); ax.text(.13,.66,"C08 proximity\n4.226 mm",ha="center",bbox=dict(boxstyle="round",fc="#fee0d2")); ax.annotate("rejected",(.36,.66),(.23,.66),arrowprops=dict(arrowstyle="->")); ax.text(.48,.66,"SUPPORTED_PROGRESS\nmax 0.038 mm",ha="center",bbox=dict(boxstyle="round",fc="#deebf7")); ax.annotate("validate target",(.76,.66),(.62,.66),arrowprops=dict(arrowstyle="->")); ax.text(.87,.66,"B03-C\n0/12 at 100",ha="center",bbox=dict(boxstyle="round",fc="#fee0d2")); ax.text(.5,.22,"Dynamic storage validity is upstream of workspace and scripted handoff.\nBoth downstream experiments remain unexecuted by the frozen protocol.",ha="center"); ax.set_title("Phase 3C-1.0 causal summary"); _save(fig,NAMES[17])
    return [str(FIGURES / f"{name}.pdf") for name in NAMES]


def create_report(summary: dict, figures: list[str]) -> None:
    b = summary["B03_validation"]; m = summary["metric_repair"]
    candidates = "\n".join(
        f"- `{row['candidate_id']}`: center `{row['center_palm_m']}` m; configuration `{row['configuration']}`; full qpos is frozen in the manifest."
        for row in b["candidates"]
    )
    orientations = "\n".join(
        f"- `{row['orientation_id']}`: forearm/WRJ1/WRJ2 = `{row['forearm_PS_rad']}`, `{row['WRJ1_rad']}`, `{row['WRJ2_rad']}` rad; basis `{row['target_basis']}`."
        for row in b["orientations"]
    )
    figure_lines = "\n".join(f"- `{Path(path).relative_to(ROOT).as_posix()}`" for path in figures)
    text = f"""# Phase 3C-1.0 results

## Outcome

The support-gated metric rejects the Phase 3C-0.8 ballistic fly-by. Direct validation classified B03 as **{b['classification']}**: 0/12 frozen trials survived 100, 500, or 1000 steps. B03 is not approved as a transport target. Consequently, the frozen protocol forbids the B03-stored workspace experiment and the scripted handoff; neither was run or inferred.

## Metric repair

`SUPPORTED_PROGRESS` evaluates raw progress only while hand normal-force support is positive, excludes table/floor/fixture support by construction, and requires palm-relative sphere speed below a configurable engineering diagnostic gate. The predeclared sensitivity gates were `{m['diagnostic_speed_gates_mps']}` m/s; none is a publication threshold. Raw minimum distance remains descriptive only.

The Phase 3C-0.8 best trajectory had raw minimum `{m['old_raw_minimum_distance_m']*1000:.6f} mm`, speed `{m['raw_speed_at_minimum_mps']:.9g} m/s`, and hand force `{m['hand_force_at_minimum_n']:.9g} N` at that sample, so the fly-by is rejected. Across all three gates, the valid supported minimum was `{m['sensitivity']['0.02']['valid_supported_minimum_distance_m']*1000:.6f} mm` and maximum supported progress was `{m['sensitivity']['0.02']['maximum_valid_progress_m']*1000:.6f} mm`.

`TRANSFER_CLEARANCE` is the minimum sphere clearance from middle/ring/little during early transfer. `RECEIVER_READY` is separate: receiver joints near an actual B03 configuration, clear sphere C-space, and at least two storage-side contact opportunities. The joint tolerance is configurable and labeled an engineering geometry diagnostic.

## Frozen B03 direct validation

Manifest SHA-256: `{b['manifest_sha256']}`. Candidate selection (medoid then deterministic farthest-point coverage) and four orientations were frozen before dynamic outcomes.

{candidates}

{orientations}

All 12 states had zero initial maximum penetration, so B03-D is rejected. Survival counts were `{b['survival_counts']}`. Median/maximum displacement were `{b['displacement_m']['median']:.9g}` / `{b['displacement_m']['maximum']:.9g} m`. Dynamic penetration median/p95/p99/max were `{b['penetration_m']['dynamic_median']:.9g}` / `{b['penetration_m']['dynamic_p95']:.9g}` / `{b['penetration_m']['dynamic_p99']:.9g}` / `{b['penetration_m']['dynamic_maximum']:.9g} m`.

Any-contact trial counts were `{b['contact_trial_counts']}`. The dominant load-bearing topology was `{b['dominant_support_topology']}` in `{b['dominant_support_topology_trial_count']}` trials. The measured outcome is static/gravity escape from a geometry-only cage, not gross initial overlap.

## Gated downstream experiments

Workspace: **NOT RUN**. There is no dynamically validated retained B03 state in which to fix/store A, so open-versus-stored workspace values, retained fractions, apertures, and collision fractions are unavailable rather than zero.

Scripted handoff: **NOT RUN**. The frozen M2 mapping remains thumb guide + index unload/migration, and the six receiver-first stages are specified in code, but target validation is an upstream prerequisite. Therefore no nominal initial state, gravity decomposition, receiver-ready time, unload time, force history, path trace, B03 entry, post-handoff hold, or handoff video exists.

## Interpretation and next step

The receiver-first hypothesis is untested, not rejected. Old preshape conclusions remain non-discriminative under failed transport and should be reconsidered only after a viable receiving target exists. Trajectory optimization, RL, compliant skin, and object B remain premature. The exact next phase is PI review of the B03-C failure and a PI decision about revising the storage target or contact-conformity hypothesis; no scientific criterion or physics change is selected here.

## Figures

{figure_lines}
"""
    (ROOT / "docs/PHASE3C10_RESULTS.md").write_text(text, encoding="utf-8")
    (ROOT / "docs/PHASE3C10_TODO_PI.md").write_text(
        """# Phase 3C-1.0 TODO(PI)

- `configs/phase3C10_b03_handoff.yaml`: decide whether the B03-C direct-hold result warrants revising the storage target, the contact-conformity hypothesis, or both.
- `configs/phase3C10_b03_handoff.yaml`: no scientific quasi-static speed threshold is frozen; decide whether and how a later publication criterion should be defined.
- `configs/phase3C10_b03_handoff.yaml`: decide whether a future optimizer may use a successful scripted handoff as its warm start; no successful handoff exists yet.

No TODO(PI) scientific decision was resolved automatically.
""",
        encoding="utf-8",
    )


def main() -> None:
    metric, manifest, result = _load(); summary = summarize(metric, manifest, result)
    figures = create_figures(metric, manifest, result, summary); summary["figures"] = figures
    (OUTPUT / "phase3c10_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    create_report(summary, figures)
    print(json.dumps({"classification": result["classification"], "figures": len(figures), "workspace": "NOT_RUN", "scripted_handoff": "NOT_RUN"}, indent=2))


if __name__ == "__main__":
    main()
