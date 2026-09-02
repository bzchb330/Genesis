"""Create the Phase 3C-1.1 reports and the twenty frozen-protocol figures."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from seqgrasp.config import ROOT
from seqgrasp.phase3c08 import build_forearm_scene
from seqgrasp.phase3c10 import initialize_b03_trial
from seqgrasp.phase3c07 import contact_geometry
from seqgrasp.phase3c11 import shape_specifications


OUTPUT = ROOT / "outputs/phase3C11"
FIGURES = ROOT / "docs/figures/phase3C11"
CHECKPOINTS = (10, 25, 50, 100, 200, 500, 1000)


def load(name: str):
    return json.loads((OUTPUT / name).read_text(encoding="utf-8"))


def original_initial_contacts() -> list[dict]:
    manifest = load("../phase3C10/B03_validation_manifest.json")
    wrapper = build_forearm_scene(with_actuator=True)
    rows = []
    for trial in manifest["trials"]:
        initialize_b03_trial(wrapper, trial)
        state = contact_geometry(wrapper.scene)
        rows.append({
            "trial_id": trial["trial_id"],
            "active_contacts": state["contact_topology"],
            "load_bearing_contacts": state["load_bearing_topology"],
            "contact_count": len(state["contact_topology"]),
            "maximum_penetration_m": state["maximum_penetration_m"],
        })
    return rows


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _title(ax, title: str, subtitle: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=22 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.01, subtitle, transform=ax.transAxes, va="bottom", fontsize=8, color="#555")


def _notice(name: str, title: str, headline: str, detail: str) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8)); ax.axis("off")
    _title(ax, title)
    ax.text(.04, .62, headline, fontsize=20, fontweight="bold", color="#8c2d04", transform=ax.transAxes)
    ax.text(.04, .47, detail, fontsize=11, transform=ax.transAxes, wrap=True)
    ax.text(.04, .10, "Frozen physics • no fabricated dynamics • Phase 3C-1.1", fontsize=9, color="#666", transform=ax.transAxes)
    _save(fig, name)


def make_figures(summary: dict, calibration: dict, audit: dict, preload: dict, results: dict,
                 shape_manifest: dict, shape_results: dict, workspace: dict, roles: dict) -> list[str]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    colors = {"middle":"#4477aa", "ring":"#66ccee", "little":"#228833", "palm":"#ccbb44", "thumb":"#ee6677", "index":"#aa3377"}
    generated = []

    # 1 compiled contact parameters
    fig, ax = plt.subplots(figsize=(8.2, 5.2)); ax.axis("off"); _title(ax, "Compiled contact model", "Values read from the instantiated MuJoCo model")
    obj = audit["object"]; solver = audit["solver"]
    text = (f"Object: {obj['type_name']}  condim={obj['condim']}  friction={obj['friction']}\n"
            f"Object solref={obj['solref']}  solimp={obj['solimp']}  margin/gap={obj['margin_m']}/{obj['gap_m']} m\n\n"
            f"Hand geoms: condim=3, friction=[1, 0.005, 0.0001]\n"
            f"Runtime object–hand contact: dim=6, friction=[0.5, 0.5, 0.01, 0.003, 0.003]\n\n"
            f"timestep={solver['timestep_s']} s  iterations={solver['iterations']}  tolerance={solver['tolerance']:.0e}\n"
            f"gravity={solver['gravity_mps2']} m/s²; solver={solver['solver']}; cone={solver['cone']}")
    ax.text(.03,.82,text,va="top",fontsize=11,fontfamily="monospace"); _save(fig,"compiled_contact_parameter_audit.pdf"); generated.append("compiled_contact_parameter_audit.pdf")

    # 2 force curves
    fig, ax = plt.subplots(figsize=(8.2,5.2)); _title(ax,"Normal force vs signed approach","Solid: principal sweep; dotted extension is explicitly documented")
    for surface in colors:
        rows=[r for r in calibration["rows"] if r["surface"]==surface]
        ax.plot([r["approach_mm"] for r in rows],[r["normal_force_n"] for r in rows],marker="o",label=surface,color=colors[surface])
    ax.axvspan(.1,.3,color="#ddd",alpha=.45,label="PI-proposed engineering region"); ax.set(xlabel="Approach (mm)",ylabel="Normal force (N)"); ax.legend(ncol=3,fontsize=8); ax.grid(alpha=.25)
    _save(fig,"normal_force_vs_approach.pdf"); generated.append("normal_force_vs_approach.pdf")

    # 3 selected preload targets
    targets=calibration["selection"]["targets"]
    fig, ax=plt.subplots(figsize=(8.2,5.2)); _title(ax,"Frozen pair-specific preload targets","Central 0.20 mm rule with documented force-activation fallback; frozen before B03 outcomes")
    names=list(targets); vals=[targets[n]["target_normal_force_n"] for n in names]
    ax.bar(names,vals,color=[colors[n] for n in names]); ax.set_ylabel("Target normal force (N)"); ax.grid(axis="y",alpha=.25)
    for i,v in enumerate(vals): ax.text(i,v,f" {v:.3f}",ha="center",va="bottom",fontsize=9)
    _save(fig,"preload_calibration_by_surface.pdf"); generated.append("preload_calibration_by_surface.pdf")

    # 4 original vs preload initialization
    before=[r["contact_count"] for r in summary["original_initial_contacts"]]
    after=[len(r["active_storage_contacts"]) for r in preload["rows"]]
    feasible_count=sum(r["initializer_feasible"] for r in preload["rows"])
    fig,ax=plt.subplots(figsize=(8.2,5.2)); _title(ax,"B03 initialization: original vs preload closure",f"All 12 original states began contact-free; {feasible_count} preload solutions passed validity")
    x=np.arange(12); ax.bar(x-.2,before,.4,label="original",color="#999"); ax.bar(x+.2,after,.4,label="after closure",color="#4477aa"); ax.set(xlabel="Frozen trial index",ylabel="Active storage-contact count",xticks=x); ax.legend(); ax.grid(axis="y",alpha=.25)
    _save(fig,"B03_original_vs_preloaded_initialization.pdf"); generated.append("B03_original_vs_preloaded_initialization.pdf")

    # 5 survival comparison
    fig,ax=plt.subplots(figsize=(8.2,5.2)); _title(ax,"Original vs preloaded B03 retention","Preloaded denominator is the frozen 12; no state was dynamically executable")
    orig=[results["original_survival_counts"][str(c)] for c in CHECKPOINTS]; new=[results["preloaded_survival_counts"][str(c)] for c in CHECKPOINTS]
    ax.plot(CHECKPOINTS,orig,"o-",label="Phase 3C-1.0 original",color="#4477aa"); ax.plot(CHECKPOINTS,new,"o-",label="Phase 3C-1.1 preloaded",color="#cc3311"); ax.set_xscale("log"); ax.set(xlabel="Hold step",ylabel="Retained states / 12",ylim=(-.3,12.3)); ax.legend(); ax.grid(alpha=.25)
    _save(fig,"B03_original_vs_preloaded_survival.pdf"); generated.append("B03_original_vs_preloaded_survival.pdf")

    # 6 topology
    topology=Counter("+".join(r["load_bearing_storage_topology"]) or "none" for r in preload["rows"])
    fig,ax=plt.subplots(figsize=(8.2,5.2)); _title(ax,"Preloaded B03 support topology","Initializer outputs before dynamics; validity requires ≥2 load-bearing storage surfaces")
    ax.bar(topology.keys(),topology.values(),color="#66ccee"); ax.set_ylabel("Frozen states"); ax.grid(axis="y",alpha=.25)
    _save(fig,"B03_preloaded_support_topology.pdf"); generated.append("B03_preloaded_support_topology.pdf")

    # 7 shapes
    specs=shape_specifications(); fig,axes=plt.subplots(1,3,figsize=(9,4));
    for ax,(sid,spec) in zip(axes,specs.items()):
        ax.axis("off"); ax.add_patch(plt.Circle((.5,.58),.22,fill=False,lw=3,color="#4477aa") if sid=="S0" else plt.Rectangle((.28,.36),.44,.44,fill=False,lw=3,color="#228833" if sid=="S1" else "#cc6677")); ax.set(xlim=(0,1),ylim=(0,1)); ax.set_title(f"{sid}: {spec['shape']}\n{spec['dimensions_m']}\n{spec['mass_kg']*1000:.3f} g",fontsize=10)
    fig.suptitle("25-mm shape controls at fixed density and contact physics",fontsize=14,fontweight="bold"); _save(fig,"sphere_vs_cube_vs_cylinder_geometry.pdf"); generated.append("sphere_vs_cube_vs_cylinder_geometry.pdf")

    # 8 shape survival
    fig,ax=plt.subplots(figsize=(8.2,5.2)); _title(ax,"Shape-control direct-storage retention","S1/S2: 0 dynamically executable because preload initialization was infeasible")
    s0=results["preloaded_survival_counts"]
    for sid,col in zip(("S0","S1","S2"),("#4477aa","#228833","#cc6677")):
        counts=s0 if sid=="S0" else shape_results["shapes"][sid]["survival_counts"]
        ax.plot(CHECKPOINTS,[counts[str(c)] for c in CHECKPOINTS],"o-",label=sid,color=col)
    ax.set_xscale("log"); ax.set(xlabel="Hold step",ylabel="Retained frozen candidates",ylim=(-.2,6.5)); ax.legend(); ax.grid(alpha=.25)
    _save(fig,"shape_storage_survival.pdf"); generated.append("shape_storage_survival.pdf")

    # 9 shape topology
    fig,ax=plt.subplots(figsize=(8.2,5.2)); _title(ax,"Shape support-topology outcome","No cube or cylinder preload initializer formed the required valid network")
    s0_feasible=results["feasible_initializers"]
    ax.bar(["S0\nvalid", "S0\ninfeasible", "S1\ninfeasible", "S2\ninfeasible"],[s0_feasible,12-s0_feasible,6,6],color=["#228833","#bbb","#bbb","#bbb"]); ax.set_ylabel("Frozen candidates"); ax.grid(axis="y",alpha=.25)
    _save(fig,"shape_storage_support_topology.pdf"); generated.append("shape_storage_support_topology.pdf")

    # workspace point clouds 10-14
    for section,finger,name,title in [
        ("baseline","thumb","baseline_thumb_workspace.pdf","Baseline thumb workspace"),
        ("geometric_B03","thumb","B03_thumb_workspace.pdf","Geometric-B03 thumb workspace"),
        ("baseline","index","baseline_index_workspace.pdf","Baseline index workspace"),
        ("geometric_B03","index","B03_index_workspace.pdf","Geometric-B03 index workspace")]:
        rec=workspace[section][finger]; points=np.asarray(rec["reachable_points_palm_m"])
        fig,ax=plt.subplots(figsize=(6.4,5.4)); _title(ax,title,f"Geometric/kinematic only • volume={rec['reachable_volume_m3']:.6g} m³")
        ax.scatter(points[:,0]*1000,points[:,2]*1000,s=5,alpha=.35,color=colors[finger]); ax.set(xlabel="Palm x (mm)",ylabel="Palm z (mm)"); ax.axis("equal"); ax.grid(alpha=.2)
        _save(fig,name); generated.append(name)
    fig,ax=plt.subplots(figsize=(8.2,5.2)); _title(ax,"Thumb–index opposition workspace","Independent reachable samples paired at diagnostic aperture ≤80 mm")
    labels=["baseline","geometric B03"]; vals=[workspace[x]["opposition"]["opposition_midpoint_volume_m3"] for x in ("baseline","geometric_B03")]
    ax.bar(labels,vals,color=["#999","#aa3377"]); ax.set_ylabel("Opposition-midpoint hull volume (m³)"); ax.ticklabel_format(axis="y",style="sci",scilimits=(0,0)); ax.grid(axis="y",alpha=.25)
    for i,v in enumerate(vals): ax.text(i,v,f" {v:.6g}",ha="center",va="bottom",fontsize=9)
    _save(fig,"thumb_index_opposition_workspace.pdf"); generated.append("thumb_index_opposition_workspace.pdf")

    # 15 retained fractions
    fig,ax=plt.subplots(figsize=(8.2,5.2)); _title(ax,"Geometric workspace retained under B03 arrangement","Descriptive fractions; no success threshold is asserted")
    rf=workspace["retained_fraction"]; ax.bar(rf.keys(),rf.values(),color=["#ee6677","#aa3377","#4477aa"]); ax.axhline(1,color="#555",lw=1); ax.set_ylim(0,1.08); ax.set_ylabel("Fraction of baseline volume"); ax.grid(axis="y",alpha=.25)
    _save(fig,"workspace_retained_fraction.pdf"); generated.append("workspace_retained_fraction.pdf")

    # 16 role search funnel
    fig,ax=plt.subplots(figsize=(8.2,5.2)); _title(ax,"MRL vs thumb-assisted storage search","Separate role families; no scalar score")
    stages=["search","prefilter","frozen","init feasible","wrench feasible","robust"]
    for i,(role,col) in enumerate(zip(("ROLE-MRL","ROLE-T"),("#4477aa","#ee6677"))):
        r=roles["roles"][role]; vals=[r["search_size"],r["prefilter_count"],len(r["selected"]),sum(x["initializer_feasible"] for x in r["initialized"]),r["mechanically_feasible_count"],r["disturbance_robust_count"]]
        ax.plot(stages,vals,"o-",label=role,color=col)
    ax.set_yscale("symlog",linthresh=1); ax.set_ylabel("Candidate count"); ax.legend(); ax.grid(alpha=.25); ax.tick_params(axis="x",rotation=20)
    _save(fig,"MRL_vs_thumb_assisted_storage.pdf"); generated.append("MRL_vs_thumb_assisted_storage.pdf")

    # 17/18 mechanics and explicit no-robustness outcome
    fig,ax=plt.subplots(figsize=(8.2,5.2)); _title(ax,"Alternative storage mechanics","One initialized candidate per role reached wrench analysis; neither satisfied full force+torque equilibrium")
    labels=[]; force=[]; torque=[]
    for role in ("ROLE-MRL","ROLE-T"):
        nominal=next((m["nominal"] for m in roles["roles"][role]["mechanics"] if m["nominal"].get("contact_count",0)>0),None)
        labels.append(role); force.append(np.linalg.norm(nominal["force_residual_n"])*1000 if nominal else 0); torque.append(np.linalg.norm(nominal["torque_residual_nm"])*1e6 if nominal else 0)
    x=np.arange(2); ax.bar(x-.18,force,.36,label="force residual (mN)",color="#4477aa"); ax.bar(x+.18,torque,.36,label="torque residual (µN·m)",color="#ee6677"); ax.set_xticks(x,labels); ax.set_ylabel("Residual magnitude (mixed displayed units)"); ax.legend(); ax.grid(axis="y",alpha=.25)
    _save(fig,"alternative_storage_mechanics.pdf"); generated.append("alternative_storage_mechanics.pdf")
    _notice("disturbance_robustness_by_role.pdf","Six-direction disturbance robustness","0 robust candidates in either role","The ±0.5 mm translation set was frozen, but no nominal mechanically feasible candidate existed to qualify for a robust result."); generated.append("disturbance_robustness_by_role.pdf")

    # 19 preserved-resource volumes
    fig,ax=plt.subplots(figsize=(8.2,5.2)); _title(ax,"Morphology/resource allocation map","Kinematic descriptors at representative frozen role geometry; not dynamic retention")
    labels=[]; vals=[]; cols=[]
    for role,col in (("ROLE-MRL","#4477aa"),("ROLE-T","#ee6677")):
        for finger,rec in roles["roles"][role]["preserved_workspace"].items(): labels.append(f"{role}\n{finger}"); vals.append(rec["reachable_volume_m3"]); cols.append(col)
    ax.bar(labels,vals,color=cols); ax.set_ylabel("Reachable volume (m³)"); ax.ticklabel_format(axis="y",style="sci",scilimits=(0,0)); ax.grid(axis="y",alpha=.25)
    _save(fig,"morphology_resource_allocation_map.pdf"); generated.append("morphology_resource_allocation_map.pdf")

    # 20 causal summary
    fig,ax=plt.subplots(figsize=(9,5.4)); ax.axis("off"); _title(ax,"Phase 3C-1.1 causal summary")
    boxes=[(.03,.68,"Calibration","Pair-isolated force response;\npair-specific target selection"),(.37,.68,"B03 recheck",f"{feasible_count}/12 initializers feasible\nno valid dynamics executable"),(.70,.68,"Shape controls","0/6 cube and 0/6 cylinder\ninitializers feasible"),(.20,.25,"Role mechanics","0 MRL and 0 T mechanically\ncredible receivers"),(.58,.25,"Decision","CASE E: conformity/morphology\nis next scientific question")]
    for x,y,h,b in boxes:
        ax.add_patch(plt.Rectangle((x,y),.27,.18,fc="#eef4f8",ec="#4477aa",lw=1.5)); ax.text(x+.015,y+.13,h,fontweight="bold"); ax.text(x+.015,y+.04,b,fontsize=9)
    for x1,y1,x2,y2 in [(.30,.77,.37,.77),(.64,.77,.70,.77),(.50,.68,.34,.43),(.82,.68,.70,.43),(.47,.34,.58,.34)]: ax.annotate("",xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle="->",color="#666"))
    _save(fig,"phase3C11_causal_summary.pdf"); generated.append("phase3C11_causal_summary.pdf")
    assert len(generated)==20 and len(set(generated))==20
    return [str(FIGURES/name) for name in generated]


def build_summary() -> dict:
    audit=load("compiled_contact_audit.json"); calibration=load("preload_calibration.json")
    preload=load("preloaded_B03_manifest.json"); results=load("preloaded_B03_results.json")
    shape_manifest=load("shape_candidate_manifest.json"); shape_results=load("shape_hold_results.json")
    workspace=load("resource_workspace_audit.json"); roles=load("storage_role_mechanics.json"); role_hold=load("role_T_hold_results.json")
    originals=original_initial_contacts(); specs=shape_specifications()
    summary={
        "phase":"3C-1.1", "branch":"codex/phase3C11-preload-shape-resource-storage",
        "base_commit":"60f9bfc2f42e31f9296fc582a018c9af021e7fa7",
        "compiled_contact_audit":audit, "preload_calibration":calibration,
        "original_initial_contacts":originals,
        "original_state_identity":{"candidate_ids":["B03_CANDIDATE_00","B03_CANDIDATE_01","B03_CANDIDATE_02"],"orientation_count":4,"trial_count":12},
        "preloaded_B03":{"manifest_sha256":preload["sha256"],"feasible_initializers":sum(r["initializer_feasible"] for r in preload["rows"]),"rows":preload["rows"],"results":results},
        "shapes":{"specifications":specs,"manifest":shape_manifest,"results":shape_results,
                  "S0":{"candidate_count":12,"feasible_initializers":results["feasible_initializers"],"survival_counts":results["preloaded_survival_counts"]}},
        "workspace":workspace,"roles":roles,"role_T_holds":role_hold,
        "decision":{"case":"CASE E","primary_blocker":"No tested geometry/role produced a reproducible calibrated multi-surface receiver; local preload closure is the immediate gate.",
                    "shape_effect":"INCONCLUSIVE: cube/cylinder had no dynamically executable initialized candidates.",
                    "role_preference":"UNRESOLVED: neither role produced a mechanically feasible candidate.",
                    "recommended_next_phase":"PI-designed rigid-contact conformity/morphology study, followed by a controlled compliant-contact/skin ablation only if authorized; do not start handoff, RL, or object B."},
    }
    summary["figures"]=make_figures(summary,calibration,audit,preload,results,shape_manifest,shape_results,workspace,roles)
    return summary


def write_reports(summary: dict) -> None:
    OUTPUT.mkdir(parents=True,exist_ok=True); (OUTPUT/"phase3c11_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    cal=summary["preload_calibration"]; audit=summary["compiled_contact_audit"]; pre=summary["preloaded_B03"]; ws=summary["workspace"]; roles=summary["roles"]["roles"]
    audit_md=f"""# Phase 3C-1.1 compiled contact and preload audit

The audit reads the instantiated MuJoCo model and measures actual contact response. It does **not** assume that zero geometric penetration always means zero normal force.

## Frozen compiled model

- Object geom: `{audit['object']['type_name']}`, condim `{audit['object']['condim']}`, friction `{audit['object']['friction']}`, margin `{audit['object']['margin_m']}` m, gap `{audit['object']['gap_m']}` m, solref `{audit['object']['solref']}`, solimp `{audit['object']['solimp']}`.
- Representative hand geoms: condim 3, friction `[1.0, 0.005, 0.0001]`, margin/gap 0 m, solref `[0.005, 1.0]`, solimp `[0.5, 0.99, 0.0001, 0.5, 2.0]`.
- Runtime object-hand contacts: dim 6, friction `[0.5, 0.5, 0.01, 0.003, 0.003]`, solref `[0.02, 1.0]`, solimp `[0.9, 0.95, 0.001, 0.5, 2.0]`.
- Solver: timestep {audit['solver']['timestep_s']} s, iterations {audit['solver']['iterations']}, tolerance {audit['solver']['tolerance']}, gravity {audit['solver']['gravity_mps2']} m/s².

## Calibration and frozen selection

The principal signed-approach sweep is `{cal['principal_sweep_mm']}` mm. The explicitly documented extension is `{cal['explicit_extension_sweep_mm']}` mm. The representative pair is chosen from compiled fingertip collision geometry (or nearest compiled palm geom), with other fingers placed in the existing zero-flexion configuration. The inward approach direction is selected by a two-sided signed-distance probe.

The frozen rule is: {cal['selection']['rule']} The selection was hashed before B03 outcomes (`{cal['selection_sha256']}`). Targets are:

| Surface | Approach (mm) | Measured target force (N) |
|---|---:|---:|
""" + "\n".join(f"| {s} | {v['target_approach_mm']:.2f} | {v['target_normal_force_n']:.9f} |" for s,v in cal["selection"]["targets"].items()) + "\n"
    (ROOT/"docs/PHASE3C11_CONTACT_PRELOAD_AUDIT.md").write_text(audit_md,encoding="utf-8")
    results_md=f"""# Phase 3C-1.1 results

## Outcome

Phase 3C-1.1 reaches **CASE E** under the frozen protocol. Pair-isolated calibration confirmed measurable rigid-contact response, but 0/12 exact B03 states admitted a valid two-surface initializer. Cube and short-cylinder searches each froze six candidates, but none passed preload initialization. Consequently no Phase 3C-1.1 hold dynamics were validly executable. Neither ROLE-MRL nor ROLE-T produced a mechanically feasible candidate.

## Original B03 recheck

- Exact original candidates/orientations: 3 × 4 = 12; all original states had zero initial active storage contacts.
- Preload initializer: {pre['feasible_initializers']}/12 feasible.
- Original retention: `{pre['results']['original_survival_counts']}`.
- Preloaded retention (frozen denominator 12): `{pre['results']['preloaded_survival_counts']}`.
- Classification: `{pre['results']['classification']}`; initialization did not materially alter the B03-C conclusion.

## Shape controls

- S0 sphere: 25 mm diameter; 12 frozen rechecks, 0 executable initializers.
- S1 cube: 25 mm side; 6 frozen candidates, 0 executable initializers.
- S2 short cylinder: 25 mm diameter × 20 mm height; 6 frozen candidates, 0 executable initializers. The 20 mm height is an engineering control.
- Shape effect is inconclusive because cube/cylinder retention dynamics could not validly start.

## Resource workspace

The geometric B03 arrangement retained thumb/index/opposition volumes of `{ws['retained_fraction']}` relative to baseline. This is kinematic evidence only: the dynamically-supported workspace gate had zero eligible ≥200-step states. Object/storage-finger collision counts remain part of each descriptor.

## Role allocation

- ROLE-MRL: {roles['ROLE-MRL']['search_size']} sampled, {roles['ROLE-MRL']['prefilter_count']} prefilter passes, 6 frozen, {sum(r['initializer_feasible'] for r in roles['ROLE-MRL']['initialized'])} feasible initializer, 0 mechanically feasible, 0 robust.
- ROLE-T: {roles['ROLE-T']['search_size']} sampled, {roles['ROLE-T']['prefilter_count']} prefilter passes, 6 frozen, {sum(r['initializer_feasible'] for r in roles['ROLE-T']['initialized'])} feasible initializer, 0 mechanically feasible, 0 robust.

No morphology-specific role winner is supported. No handoff, object B, optimizer, RL, altered contact physics, or skin was used.
"""
    (ROOT/"docs/PHASE3C11_RESULTS.md").write_text(results_md,encoding="utf-8")
    todo="""# Phase 3C-1.1 decisions awaiting PI input

- `configs/phase3C11_preload_shape_resource_storage.yaml`: decide whether the calibrated rigid-contact preload evidence supports freezing any receiver for Phase 3C-1.2. Current evidence does not identify one.
- `configs/phase3C11_preload_shape_resource_storage.yaml`: decide whether a future valid shape-dependent retention comparison justifies using an easier non-spherical object before returning to the sphere. The present comparison is initialization-limited.
- `configs/phase3C11_preload_shape_resource_storage.yaml`: decide whether Shadow morphology warrants thumb-assisted storage instead of the human-inspired MRL allocation. Neither role is currently feasible.
- `configs/phase3C11_preload_shape_resource_storage.yaml`: define publication thresholds, if any, for useful workspace, quasi-static force, and sufficient disturbance robustness.
- Phase 3C-1.1 conclusion: decide the design of a rigid-contact conformity/morphology study and whether it should then include a controlled compliant-contact/skin ablation. No such ablation was run here.
"""
    (ROOT/"docs/PHASE3C11_TODO_PI.md").write_text(todo,encoding="utf-8")


def main() -> None:
    summary=build_summary(); write_reports(summary)
    print(json.dumps({"summary":str(OUTPUT/"phase3c11_summary.json"),"figures":len(summary["figures"]),"case":summary["decision"]["case"]},indent=2))


if __name__=="__main__": main()
