from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from seqgrasp.config import ROOT
from seqgrasp.phase3c06 import (
    FAILURE_TAXONOMY,
    POCKET_NAMES,
    SUPPORT_SURFACES,
    audit_non_thumb_link_lengths,
    build_sphere_scene,
    construct_palmodigital_pockets,
    pocket_geometry,
    size_curriculum,
    sphere_scale,
)


FIGURE_NAMES = (
    "shadow_finger_link_length_audit.pdf",
    "D0_sphere_scale_relative_to_hand.pdf",
    "old_palm_center_vs_palmodigital_pocket.pdf",
    "palmodigital_candidate_regions.pdf",
    "ring_little_storage_pocket.pdf",
    "open_corridor_sphere_transfer.pdf",
    "storage_preshape_sequence.pdf",
    "no_preshape_vs_preshape.pdf",
    "pocket_support_topology.pdf",
    "wrist_assisted_sphere_settling.pdf",
    "gravity_in_palm_vs_storage.pdf",
    "penetration_normalized_by_sphere_radius.pdf",
    "thumb_recovery_survival.pdf",
    "sphere_size_curriculum.pdf",
    "sphere_size_vs_wrist_range.pdf",
    "representative_storage_sequence.pdf",
)
COLORS = {"old_palm_center": "#64748B", "middle_ring": "#0EA5E9",
          "ring_little": "#8B5CF6", "ulnar_palmodigital": "#F97316"}


def _style(ax, title: str, subtitle: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=13, weight="bold", color="#0F172A", pad=26)
    if subtitle:
        ax.text(0, 1.01, subtitle, transform=ax.transAxes, va="bottom", color="#475569", fontsize=8)
    ax.grid(alpha=.18, linewidth=.7)
    for edge in ("top", "right"):
        ax.spines[edge].set_visible(False)


def _save(fig, path: Path) -> None:
    fig.savefig(path, format="pdf", bbox_inches="tight", metadata={"Title": path.stem, "Creator": "Phase 3C-0.6 analysis"})
    plt.close(fig)


def _counts(rows, predicate) -> dict[str, int]:
    return {pocket: sum(predicate(row) for row in rows if row["pocket"] == pocket) for pocket in POCKET_NAMES}


def _summary(rows: list[dict]) -> dict:
    by_pocket = {}
    for pocket in POCKET_NAMES:
        selected = [row for row in rows if row["pocket"] == pocket]
        by_pocket[pocket] = {
            "N": len(selected), "pocket_entry": sum(r["pocket_entry_step"] is not None for r in selected),
            "stable_capture": sum(r["stable_capture"] for r in selected),
            "thumb_release_attempts": sum(r["thumb_release_attempted"] for r in selected),
            "thumb_recovered": sum(r["thumb_recovered"] for r in selected),
            "ring_contact": sum(r["ring_contact"] for r in selected),
            "little_contact": sum(r["little_contact"] for r in selected),
            "palm_root_contact": sum(r["palm_contact"] for r in selected),
            "alternate_support": sum(r["alternate_support"] for r in selected),
            "retention_1000": sum(r.get("survival", {}).get("1000", False) for r in selected),
            "maximum_penetration_m": max(r["maximum_penetration_m"] for r in selected),
        }
    by_preshape = {}
    for condition in ("NO_PRESHAPE", "PRESHAPE"):
        selected = [row for row in rows if row["preshape"] == condition]
        by_preshape[condition] = {
            "N": len(selected), "pocket_entry": sum(r["pocket_entry_step"] is not None for r in selected),
            "stable_capture": sum(r["stable_capture"] for r in selected),
            "thumb_recovered": sum(r["thumb_recovered"] for r in selected),
        }
    by_wrist = {}
    for command in sorted({tuple(row["wrist_delta_command_deg"]) for row in rows}):
        selected = [row for row in rows if tuple(row["wrist_delta_command_deg"]) == command]
        by_wrist[str(list(command))] = {
            "N": len(selected), "pocket_entry": sum(r["pocket_entry_step"] is not None for r in selected),
            "stable_capture": sum(r["stable_capture"] for r in selected),
            "thumb_recovered": sum(r["thumb_recovered"] for r in selected),
        }
    first_steps = [r["first_storage_finger_contact_step"] for r in rows if r["first_storage_finger_contact_step"] is not None]
    trigger_steps = [r["preshape_trigger_step"] for r in rows if r["preshape"] == "PRESHAPE" and r["preshape_trigger_step"] is not None]
    topologies = Counter()
    for row in rows:
        arrays = np.load(row["timeseries_path"])
        for raw in arrays["storage_json"]:
            state = json.loads(str(raw))
            if state["load_bearing_topology"]:
                topologies["+".join(state["load_bearing_topology"])] += 1
    maximum_by_surface = np.max([row["maximum_penetration_by_surface_m"] for row in rows], axis=0)
    normalized_by_surface = np.max([row["maximum_penetration_by_surface_over_radius"] for row in rows], axis=0)
    return {
        "by_pocket": by_pocket, "by_preshape": by_preshape, "by_wrist_command": by_wrist,
        "first_storage_contact_step": {
            "count": len(first_steps), "median": float(np.median(first_steps)) if first_steps else None,
            "minimum": min(first_steps, default=None), "maximum": max(first_steps, default=None),
        },
        "preshape_trigger_step": {
            "count": len(trigger_steps), "median": float(np.median(trigger_steps)) if trigger_steps else None,
            "minimum": min(trigger_steps, default=None), "maximum": max(trigger_steps, default=None),
        },
        "dominant_load_bearing_topologies": dict(topologies.most_common(12)),
        "maximum_penetration_by_surface_m": dict(zip(SUPPORT_SURFACES, maximum_by_surface.tolist())),
        "maximum_penetration_by_surface_over_radius": dict(zip(SUPPORT_SURFACES, normalized_by_surface.tolist())),
        "failure_counts": dict(Counter(failure for row in rows for failure in row["failures"])),
    }


def _figures(rows: list[dict], summary: dict, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    scale = sphere_scale(); links = audit_non_thumb_link_lengths(); scene = build_sphere_scene()
    pockets = construct_palmodigital_pockets(scene)

    fig, ax = plt.subplots(figsize=(9, 5)); labels=[f"{r.finger}\n{r.segment}" for r in links]
    ax.bar(labels, [1e3*r.length_m for r in links], color=["#0EA5E9" if r.segment=="proximal" else "#8B5CF6" for r in links])
    ax.axhline(1e3*scale.diameter_m, color="#DC2626", ls="--", label="median L_ref = D0 = 35 mm")
    ax.set_ylabel("joint-to-joint length (mm)"); ax.legend(); _style(ax,"Official Shadow finger-link audit","Four proximal 45 mm and four intermediate 25 mm links; no single unique length")
    _save(fig, figure_dir/FIGURE_NAMES[0])

    fig, ax = plt.subplots(figsize=(8,5)); y=np.arange(4); ax.barh(y,[45,45,45,45],height=.25,color="#0EA5E9",label="proximal"); ax.barh(y-.28,[25]*4,height=.25,color="#8B5CF6",label="intermediate")
    for i in y: ax.add_patch(Circle((17.5,i+.28),17.5/45*.25,fill=False,color="#DC2626",lw=2))
    ax.set(yticks=y,yticklabels=["index","middle","ring","little"],xlabel="length scale (mm)"); ax.legend(); _style(ax,"D0 sphere scale relative to the hand","Circle glyph is schematic; exact D0 diameter is 35 mm")
    _save(fig, figure_dir/FIGURE_NAMES[1])

    fig, ax=plt.subplots(figsize=(8,5)); x=np.arange(4); entry=[summary["by_pocket"][p]["pocket_entry"]/5 for p in POCKET_NAMES]; stable=[summary["by_pocket"][p]["stable_capture"]/5 for p in POCKET_NAMES]
    ax.bar(x-.18,entry,.36,label="pocket entry",color="#0EA5E9"); ax.bar(x+.18,stable,.36,label="stable capture",color="#F97316"); ax.set(xticks=x,xticklabels=[p.replace('_','\n') for p in POCKET_NAMES],ylabel="percent of 500 matched trials",ylim=(0,60)); ax.legend(); _style(ax,"Old palm center vs palmodigital pockets","Same 50 states, preshape conditions, W0 and four W1 commands per target")
    _save(fig,figure_dir/FIGURE_NAMES[2])

    fig,ax=plt.subplots(figsize=(8,7));
    for name,p in pockets.items():
        c=np.asarray(p.center_palm_m); h=np.asarray(p.half_extents_m); ax.add_patch(Rectangle((c[0]-h[0],c[2]-h[2]),2*h[0],2*h[2],alpha=.25,color=COLORS[name],label=name)); ax.scatter(c[0],c[2],color=COLORS[name])
    ax.set(xlabel="palm x (m; ulnar left)",ylabel="palm z (m; finger roots upward)",aspect="equal"); ax.legend(fontsize=8); _style(ax,"Candidate storage volumes from compiled root geometry","Rectangles are x-z projections; each condition uses the full 3-D volume")
    _save(fig,figure_dir/FIGURE_NAMES[3])

    fig,ax=plt.subplots(figsize=(8,6)); p=pockets["ring_little"]; c=np.asarray(p.center_palm_m); r=scale.radius_m
    roots={"ring":(-.011,.095),"little":(-.033,.0865)}; ax.scatter([v[0] for v in roots.values()],[v[1] for v in roots.values()],s=100,color="#334155")
    for name,(xv,zv) in roots.items(): ax.text(xv,zv+.004,name,ha="center")
    ax.add_patch(Circle((c[0],c[2]),r,fill=False,color="#8B5CF6",lw=2,label="D0 sphere at region center")); ax.plot([-.011,-.033],[.095,.0865],ls="--",color="#64748B",label=f"root aperture {1e3*p.local_aperture_m:.1f} mm")
    ax.set(xlabel="palm x (m)",ylabel="palm z (m)",aspect="equal");ax.legend();_style(ax,"Ring/little candidate pocket","Geometry-derived target; formal trajectories reached it 0/500 times")
    _save(fig,figure_dir/FIGURE_NAMES[4])

    fig,ax=plt.subplots(figsize=(8,5)); clear=np.array([r["corridor_cleared"] for r in rows]); ax.bar(["clear","blocked"],[clear.sum(),len(clear)-clear.sum()],color=["#10B981","#EF4444"]); ax.set_ylabel("trials"); _style(ax,"Open-corridor sphere transfer","1,689/2,000 trials preserved nonnegative exact unused-finger clearance until preshape")
    _save(fig,figure_dir/FIGURE_NAMES[5])

    fig,ax=plt.subplots(figsize=(9,4)); trigger=summary["preshape_trigger_step"]; contact=summary["first_storage_contact_step"]
    ax.hlines(1,0,300,color="#CBD5E1"); ax.scatter([0,40,trigger["median"],contact["median"],300],[1]*5,s=80,color=["#334155","#0EA5E9","#8B5CF6","#F97316","#334155"]); ax.set(yticks=[],xticks=[],xlabel="script step",xlim=(-5,305),ylim=(.70,1.30));
    labels=["acquired","transfer starts","median geometric\npreshape trigger","median first\nstorage contact","capture end"]
    for index,(xval,label) in enumerate(zip([0,40,trigger["median"],contact["median"],300],labels)):
        above = index % 2 == 0
        ax.text(xval,1.07 if above else .93,label,ha="center",va="bottom" if above else "top",fontsize=8)
    _style(ax,"Storage preshape sequence","Trigger depends on bottleneck passage and exact current/predicted clearance, not a fixed-time publication rule")
    _save(fig,figure_dir/FIGURE_NAMES[6])

    fig,ax=plt.subplots(figsize=(7,5)); names=["NO_PRESHAPE","PRESHAPE"]; x=np.arange(2); entry=[summary["by_preshape"][n]["pocket_entry"]/10 for n in names]; stable=[summary["by_preshape"][n]["stable_capture"]/10 for n in names]
    ax.bar(x-.18,entry,.36,label="pocket entry",color="#0EA5E9"); ax.bar(x+.18,stable,.36,label="stable capture",color="#F97316"); ax.set(xticks=x,xticklabels=names,ylabel="percent of 1,000 matched trials");ax.legend();_style(ax,"No-preshape vs preshape","Identical entry (195/1,000) and transient capture (2/1,000) counts")
    _save(fig,figure_dir/FIGURE_NAMES[7])

    fig,ax=plt.subplots(figsize=(10,5)); topo=summary["dominant_load_bearing_topologies"]; labels=list(topo)[:10]; vals=[topo[k] for k in labels]; ax.barh(labels[::-1],vals[::-1],color="#14B8A6"); ax.set_xlabel("sample occurrences");_style(ax,"Observed load-bearing support topology","Raw positive normal-force topology; occurrences are samples, not independent trials")
    _save(fig,figure_dir/FIGURE_NAMES[8])

    fig,ax=plt.subplots(figsize=(7,6)); grid=np.full((3,3),np.nan); commands=summary["by_wrist_command"]
    for key,value in commands.items(): a,b=json.loads(key.replace("'",'"')); grid[int(a/5)+1,int(b/5)+1]=value["stable_capture"]
    im=ax.imshow(grid,origin="lower",cmap="Blues",vmin=0,vmax=max(4,np.nanmax(grid))); ax.set(xticks=range(3),xticklabels=[-5,0,5],yticks=range(3),yticklabels=[-5,0,5],xlabel="WRJ1 command (deg)",ylabel="WRJ2 command (deg)");
    for i in range(3):
        for j in range(3):
            if np.isfinite(grid[i,j]): ax.text(j,i,int(grid[i,j]),ha="center",va="center")
    fig.colorbar(im,ax=ax,label="stable captures / 400 command trials");_style(ax,"Direction-specific wrist-assisted settling","Only [+5,+5] produced transient stable captures; zero command recovered the thumb")
    _save(fig,figure_dir/FIGURE_NAMES[9])

    fig,ax=plt.subplots(figsize=(8,5)); gx=[];gz=[];colors=[]
    for row in rows: gx.append(row["gravity_in_palm_final_mps2"][0]);gz.append(row["gravity_in_palm_final_mps2"][2]);colors.append("#F97316" if row["stable_capture"] else "#94A3B8")
    ax.scatter(gx,gz,c=colors,s=8,alpha=.35);ax.set(xlabel="gravity palm-x (m/s^2)",ylabel="gravity palm-z (m/s^2)");_style(ax,"Gravity in palm coordinates vs storage","Orange: four transient stable captures; world gravity remained [0,0,-9.81] m/s^2")
    _save(fig,figure_dir/FIGURE_NAMES[10])

    fig,ax=plt.subplots(figsize=(8,5)); vals=[summary["maximum_penetration_by_surface_over_radius"][s] for s in SUPPORT_SURFACES]; ax.bar(SUPPORT_SURFACES,vals,color="#DC2626");ax.set_ylabel("maximum penetration / D0 radius");_style(ax,"Penetration normalized by sphere radius",f"Maximum across 2,000 trials; global maximum {max(vals):.3f} R0; acceptability remains TODO(PI)")
    _save(fig,figure_dir/FIGURE_NAMES[11])

    fig,ax=plt.subplots(figsize=(8,5)); checkpoints=[10,25,50,100,200,300,500,750,1000]; ax.step(checkpoints,[0]*len(checkpoints),where="post",color="#DC2626",lw=2);ax.set(xscale="log",xlabel="post-release steps",ylabel="valid thumb-recovery survivals",ylim=(-.1,1));_style(ax,"Thumb-recovery survival","Four release attempts; all lost during the release ramp, so every checkpoint is 0/4")
    _save(fig,figure_dir/FIGURE_NAMES[12])

    fig,ax=plt.subplots(figsize=(8,5)); scales=size_curriculum(); labels=[s.scale_id for s in scales]; ax.bar(labels,[2000,0,0,0,0],color=["#0EA5E9"]+["#CBD5E1"]*4);ax.set_ylabel("formal storage trials");_style(ax,"Sphere-size curriculum","D0 did not pass the physical progression gate; D1-D4 were correctly not run")
    _save(fig,figure_dir/FIGURE_NAMES[13])

    fig,ax=plt.subplots(figsize=(9,4)); matrix=np.full((5,4),np.nan); matrix[0,0]=0;matrix[0,1]=4; im=ax.imshow(matrix,cmap="Blues",vmin=0,vmax=4,aspect="auto");ax.set(xticks=range(4),xticklabels=["W0","W1","W2","W3"],yticks=range(5),yticklabels=["D0","D1","D2","D3","D4"]);
    for i in range(5):
        for j in range(4): ax.text(j,i,"not run" if np.isnan(matrix[i,j]) else f"{int(matrix[i,j])} captures",ha="center",va="center",fontsize=8)
    fig.colorbar(im,ax=ax,label="transient stable captures");_style(ax,"Sphere size vs wrist range","Descriptive gate map; no unexecuted cell is imputed")
    _save(fig,figure_dir/FIGURE_NAMES[14])

    representative=next(row for row in rows if row["stable_capture"]); arrays=np.load(representative["timeseries_path"]); pos=arrays["center_palm_m"]; flags=arrays["contact_flags"]
    fig,axes=plt.subplots(1,2,figsize=(11,5)); axes[0].plot(pos[:,0],pos[:,2],color="#0EA5E9");axes[0].scatter(pos[0,0],pos[0,2],label="start",color="#10B981");axes[0].scatter(pos[-1,0],pos[-1,2],label="after failed release",color="#DC2626");axes[0].set(xlabel="palm x (m)",ylabel="palm z (m)");axes[0].legend();_style(axes[0],"Sphere path")
    axes[1].imshow(flags.T,aspect="auto",interpolation="nearest",cmap="Blues");axes[1].set(yticks=range(6),yticklabels=SUPPORT_SURFACES,xlabel="recorded sample",ylabel="contact surface");_style(axes[1],"Contact sequence",f"{representative['state_id']}, old center, [+5,+5]; loss during thumb release")
    _save(fig,figure_dir/FIGURE_NAMES[15])


def main() -> None:
    output=ROOT/"outputs/phase3C06"; result=json.loads((output/"phase3c06_results.json").read_text(encoding="utf-8")); rows=result["trials"]
    summary=_summary(rows); scale=sphere_scale(); links=audit_non_thumb_link_lengths(); scene=build_sphere_scene(); pockets=construct_palmodigital_pockets(scene)
    summary.update({
        "phase_classification": {"primary":"SP-C","qualifiers":[],"reason":"sphere reached candidate volumes but rigid-contact scripted capture produced zero valid thumb recoveries"},
        "palmodigital_hypothesis_supported": False, "preshaping_beneficial": False,
        "wrist_assistance": "direction-specific transient-capture benefit at [+5,+5], no thumb-recovery benefit",
        "rigid_contact_geometry_sufficient": False,
        "future_compliant_skin_justified": False,
        "future_compliant_skin_reason":"ring/little and ulnar targets were not reached, so gross transfer geometry must be solved before a conformity ablation",
        "object_B_may_be_introduced_next": False, "RL_remains_premature": True,
        "sizes_tested":["D0"], "largest_physically_demonstrated_storable_sphere":"D0 transient only; no post-release storage",
    })
    (output/"analysis_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    figure_dir=ROOT/"docs/figures/phase3C06";_figures(rows,summary,figure_dir)

    table="\n".join(f"| {r.finger} | {r.segment} | `{r.parent_body}` | `{r.child_body}` | {r.vector_m} | {1000*r.length_m:.3f} |" for r in links)
    audit=f"""# Phase 3C-0.6 finger-link size audit

## Source and method

The audit reads the official Shadow Hand source `assets/hands/shadow_right/right_hand.xml` without modifying it. For index, middle, ring, and little fingers, the proximal joint-to-next-joint length is the norm of the child middle-body `pos`; the intermediate length is the norm of the child distal-body `pos`.

| finger | segment | parent body | child body | MJCF vector (m) | length (mm) |
|---|---|---|---|---|---:|
{table}

There is no unique single link length: the audited set contains four 45 mm proximal links and four 25 mm intermediate links. Following the PI-approved rule, `L_ref` is the median of all eight corresponding non-thumb lengths: `(25 + 45) / 2 = 35 mm`.

- `D0 = L_ref = {scale.diameter_m:.6f} m` (35 mm)
- `R0 = D0 / 2 = {scale.radius_m:.6f} m` (17.5 mm)
- inherited material density: `{scale.density_kg_m3:.1f} kg/m^3`
- sphere volume: `{scale.mass_kg/scale.density_kg_m3:.12f} m^3`
- analytic and compiled sphere mass: `{scale.mass_kg:.12f} kg`

The previous ellipsoid remains the default Phase 3A/3C object; the sphere is a separate Phase 3C-0.6 runtime configuration.
"""
    (ROOT/"docs/PHASE3C06_FINGER_LINK_SIZE_AUDIT.md").write_text(audit,encoding="utf-8")

    pocket_lines="\n".join(f"| {name} | {tuple(round(float(v),6) for v in pocket.center_palm_m)} | {tuple(round(float(v),6) for v in pocket.half_extents_m)} | {summary['by_pocket'][name]['pocket_entry']}/500 | {summary['by_pocket'][name]['stable_capture']}/500 | {summary['by_pocket'][name]['thumb_recovered']}/500 |" for name,pocket in pockets.items())
    failure_lines="\n".join(f"| {name} | {summary['failure_counts'].get(name,0)} |" for name in FAILURE_TAXONOMY)
    results=f"""# Phase 3C-0.6 results

## Outcome

Primary classification: **SP-C**. The D0 sphere reached configured volumes in 390/2,000 trials, but only four transient storage states were detected and all four lost the sphere during thumb release. Valid thumb recovery and every 10-1,000-step survival checkpoint were 0. W2, W3, and D1-D4 were not run because the stated D0 progression gate was not reached.

## Frozen physical setup

- Branch: `codex/phase3C06-sphere-palmodigital-storage`
- Base commit: `7baac924a14ff863c7d1b0bb9bfc67734390609d`
- Frozen acquisition cohort: 50 IDs (`C06_D0_STATE_00000` through `C06_D0_STATE_00049`) before storage outcomes
- D0: diameter {scale.diameter_m:.6f} m, radius {scale.radius_m:.6f} m, density {scale.density_kg_m3:.1f} kg/m3, compiled mass {scale.mass_kg:.12f} kg
- Matrix: 4 pockets x 2 preshape conditions x (W0 + 4 W1 commands) x 50 states = 2,000 trials
- World gravity, friction, compliance, official MJCF, collision geometry, and joint limits were unchanged. Object B, RL, rewards, and scalar J were absent.

## Matched pocket comparison

| target | center in palm (m) | half extents (m) | entry | transient stable capture | thumb recovery |
|---|---|---|---:|---:|---:|
{pocket_lines}

The old palm-center control produced all four transient captures. Middle/ring had 122 entries but no stable capture. Ring/little and adjacent ulnar-palmodigital targets had zero entries. Thus the tested palmodigital hypothesis is not supported by this controller/geometry result.

## Preshape and wrist

NO_PRESHAPE and PRESHAPE each produced 195/1,000 entries, 2/1,000 transient captures, and 0/1,000 thumb recoveries. Preshaping was not beneficial. W0 produced 0/400 transient captures. W1 produced 4/1,600, all under `[+5,+5]`; this is a direction-specific temporary settling effect, not a recovery benefit. Native wrist insufficiency and forearm-rotation necessity are not established because the transfer controller failed to reach the ulnar targets.

## Contacts, penetration, and survival

- First storage-finger contact: N={summary['first_storage_contact_step']['count']}, median step {summary['first_storage_contact_step']['median']}, range {summary['first_storage_contact_step']['minimum']}-{summary['first_storage_contact_step']['maximum']}.
- Ring contact: {sum(r['ring_contact'] for r in rows)}/2,000; little contact: {sum(r['little_contact'] for r in rows)}/2,000; palm/root contact: {sum(r['palm_contact'] for r in rows)}/2,000; alternate support: {sum(r['alternate_support'] for r in rows)}/2,000.
- Thumb release attempts: 4; valid thumb recoveries: 0; index release was not attempted because the primary thumb milestone failed.
- Maximum penetration across the full matrix: {max(r['maximum_penetration_m'] for r in rows):.9f} m = {max(r['maximum_penetration_over_radius'] for r in rows):.6f} R0. Penetration acceptability remains `TODO(PI)`; no new threshold or automatic gross-overlap label is applied.
- MuJoCo contact penetration is solver overlap and is not biological skin deformation. Multi-millimeter overlap is reported as a model warning, not justified as human compliance.
- Survival at 10, 25, 50, 100, 200, 300, 500, 750, and 1,000 steps: all 0/4 release attempts; losses occurred during the thumb-release ramp.

## Failure taxonomy

| label | trial count |
|---|---:|
{failure_lines}

The joint-boundary diagnostic fired in all trials because the official open/target keyframes touch compiled joint bounds; the raw minimum margin is retained and the label is not treated as a new scientific exclusion rule.

## Decision

Rigid-contact geometry alone was insufficient for reproducible palmodigital storage and thumb recovery under this bounded protocol. A compliant-skin ablation is not yet justified: the intended ring/little and ulnar geometries were not reached, so transfer geometry/control must be resolved first. Object B must not be introduced next, and RL remains premature. The recommended next step is PI review of the unreachable ulnar transfer geometry and whether an explicit forearm reorientation DOF or a different scripted transfer family should be tested; no physics criterion should be relaxed.
"""
    (ROOT/"docs/PHASE3C06_RESULTS.md").write_text(results,encoding="utf-8")

    evidence={
        "link_audit":[r.__dict__ for r in links], "sphere_scale":scale.__dict__,
        "pockets":{name:pocket_geometry(scene,pocket,scale.radius_m) for name,pocket in pockets.items()},
        "analysis_summary":summary,
        "criteria_note":"No new publication threshold was frozen; progression used only the protocol's multiple-distinct-state structural gate.",
        "contact_model_note":"MuJoCo solver penetration is not biological skin deformation.",
    }
    (output/"mechanism_evidence.json").write_text(json.dumps(evidence,indent=2),encoding="utf-8")
    print(json.dumps({"summary":str(output/"analysis_summary.json"),"figures":[str(figure_dir/name) for name in FIGURE_NAMES],"classification":"SP-C"},indent=2))


if __name__ == "__main__":
    main()
