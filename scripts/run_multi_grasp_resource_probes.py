#!/usr/bin/env python
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import csv,json,os,tempfile
import numpy as np
os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"seqgrasp-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from seqgrasp import load_configs
from seqgrasp.diagnostics.grasp_search import candidate_profile,diagnostic_metrics,evaluate_candidate,load_search_config
from seqgrasp.diagnostics.multi_grasp import bundle_for_profile,load_grasp_profile,load_resource_probe_config,local_variants,pearson_correlations,penetration_summary,reachability_cloud,refinement_objective,resource_rows,run_b_probe,write_json,write_profile,write_rows_csv
from seqgrasp.diagnostics.scripted_grasp import run_scripted_grasp

def describe(values):
    a=np.asarray(values,float); return {"minimum":float(a.min()),"maximum":float(a.max()),"mean":float(a.mean()),"median":float(np.median(a))}

def profile_runs(cfg,name,profile,seeds):
    local=bundle_for_profile(cfg,name,profile); return [run_scripted_grasp(local,seed=seed,save_outputs=False) for seed in seeds]

def aggregate_penetration(rows,fingers):
    return {"runs":rows,"maximum_penetration_m":describe([row["maximum_penetration_m"] for row in rows]),"steady_hold_penetration_m":describe([row["maximum_steady_hold_penetration_m"] for row in rows]),"per_finger_maximum_penetration_m":{finger:describe([row["per_finger"][finger]["maximum_penetration_m"] for row in rows]) for finger in fingers},"deepest_finger_counts":{finger:sum(row["deepest_finger"]==finger for row in rows) for finger in fingers},"relaxation_observed_runs":sum(row["penetration_relaxes_after_release"] for row in rows)}

def plot_all(root,summaries,probe_rows,clouds,fingers,joints):
    out=root/"figures"; out.mkdir(parents=True,exist_ok=True); names=list(summaries); written=[]
    def save(fig,name): path=out/name; fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig); written.append(path)
    active=np.asarray([[summaries[g]["A_contact_fraction_per_finger"][f] for f in fingers] for g in names]); fig,ax=plt.subplots(); im=ax.imshow(active,vmin=0,vmax=1,cmap="viridis"); ax.set(xticks=range(len(fingers)),xticklabels=fingers,yticks=range(len(names)),yticklabels=names,title="Object-A finger contact occupancy"); fig.colorbar(im,ax=ax,label="sample fraction"); save(fig,"01_contacting_finger_patterns.png")
    occ=np.asarray([summaries[g]["mean_normalized_joint_position"] for g in names]); fig,ax=plt.subplots(figsize=(10,4)); im=ax.imshow(occ,vmin=0,vmax=1,aspect="auto"); ax.set(xticks=range(len(joints)),xticklabels=joints,yticks=range(len(names)),yticklabels=names,title="Joint-range occupancy"); ax.tick_params(axis="x",rotation=60); fig.colorbar(im,ax=ax); save(fig,"02_joint_range_occupancy.png")
    fig,ax=plt.subplots(); ax.boxplot([summaries[g]["joint_margin_samples_rad"] for g in names],tick_labels=names); ax.set(title="Joint-limit margins",ylabel="minimum two-sided margin [rad]"); save(fig,"03_joint_limit_margins.png")
    fig,ax=plt.subplots(); x=np.arange(len(names)); util=[summaries[g]["mean_actuator_utilization"] for g in names]; reserve=[summaries[g]["mean_actuator_reserve_Nm"] for g in names]; ax.bar(x-.18,util,.36,label="utilization [1]"); ax.bar(x+.18,reserve,.36,label="reserve [N m]"); ax.set(xticks=x,xticklabels=names,title="Actuator utilization and reserve"); ax.legend(); save(fig,"04_actuator_utilization_reserve.png")
    fig,ax=plt.subplots(); width=.18
    for i,f in enumerate(fingers): ax.bar(x+(i-1.5)*width,[summaries[g]["mean_A_force_per_finger_N"][f] for g in names],width,label=f)
    ax.set(xticks=x,xticklabels=names,title="Object-A normal-force distribution",ylabel="mean force [N]"); ax.legend(); save(fig,"05_A_contact_forces.png")
    fig=plt.figure(); ax=fig.add_subplot(projection="3d")
    for key,cloud in clouds.items(): ax.scatter(*cloud["points"][:,3:6].T,s=5,alpha=.3,label=key)
    ax.set(title="Fingertip reachability relative to palm",xlabel="x [m]",ylabel="y [m]",zlabel="z [m]"); ax.legend(fontsize=6); save(fig,"06_fingertip_reachable_workspace.png")
    grouped=[[row["minimum_fingertip_to_B_signed_distance_m"] for row in probe_rows if row["grasp"]==g] for g in names]; fig,ax=plt.subplots(); ax.boxplot(grouped,tick_labels=names); ax.set(title="Minimum fingertip-to-B distance",ylabel="signed distance [m]"); save(fig,"07_B_minimum_distance.png")
    fig,ax=plt.subplots(); ax.bar(names,[sum(row["B_contact_occurred"] for row in probe_rows if row["grasp"]==g) for g in names]); ax.set(title="Physical fingertip-B contact occurrences",ylabel="count across 80 probes"); save(fig,"08_B_contact_counts.png")
    for number,key,title,ylabel in ((9,"A_maximum_translation_m","A translation during B probing","translation [m]"),(10,"A_maximum_rotation_rad","A rotation during B probing","rotation [rad]"),(11,"A_maximum_force_redistribution_N","A force redistribution during B probing","force-vector change [N]")):
        fig,ax=plt.subplots(); ax.boxplot([[row[key] for row in probe_rows if row["grasp"]==g] for g in names],tick_labels=names); ax.set(title=title,ylabel=ylabel); save(fig,f"{number:02d}_{key}.png")
    fig=plt.figure(figsize=(9,6)); ax=fig.add_subplot(projection="3d")
    for key,cloud in clouds.items(): ax.scatter(*cloud["points"][:,:3].T,s=4,alpha=.15,label=key)
    b=np.asarray([row["B_initial_position_m"] for row in probe_rows]); ax.scatter(*b.T,c="black",marker="x",label="B placements"); ax.set(title="A-grasp fingertip workspaces and B placements",xlabel="world x [m]",ylabel="world y [m]",zlabel="world z [m]"); ax.legend(fontsize=6); save(fig,"12_grasp_workspace_B_positions.png"); return written

def main():
    cfg=load_configs(); probe=load_resource_probe_config(); search=load_search_config(); root=Path(probe["output_dir"]); root.mkdir(parents=True,exist_ok=True); grasp_dir=Path("configs/grasps"); seeds=range(int(probe["validation_seed_count"])); fingers=list(cfg.hand.finger_geom_mapping); source_data={}
    for filename in probe["source_candidates"]:
        payload,profile=load_grasp_profile(grasp_dir/filename); source=Path(filename).stem; runs=profile_runs(cfg,source,profile,seeds); penetration=[penetration_summary(run,cfg) for run in runs]; source_data[source]={"payload":payload,"profile":profile,"runs":runs,"penetration":aggregate_penetration(penetration,fingers)}
    write_json(root/"contact_penetration_originals.json",{name:data["penetration"] for name,data in source_data.items()})
    refinement={}; refined_profiles={}
    for source,data in source_data.items():
        variants=local_variants(cfg,data["profile"],source,int(probe["local_variants_per_candidate"]),probe); screened=[]
        for candidate in variants:
            run,metrics=evaluate_candidate(cfg,search,candidate,0); pen=penetration_summary(run,cfg); screened.append({"candidate":candidate,"metrics":metrics,"penetration":pen,"objective":refinement_objective(metrics,probe)})
        screened.sort(key=lambda row:row["objective"],reverse=True); validated=[]
        for row in screened[:3]:
            profile=candidate_profile(data["profile"],row["candidate"]); runs=profile_runs(cfg,row["candidate"].candidate_id,profile,seeds); metrics=[diagnostic_metrics(run,bundle_for_profile(cfg,row["candidate"].candidate_id,profile),search) for run in runs]; pens=[penetration_summary(run,cfg) for run in runs]; retained=all(m["fingertip_contact_duration_s"]>=0.999 and m["table_recontact_time_s"] is None and m["safety_valid"] for m in metrics); validated.append({"candidate":row["candidate"],"profile":profile,"metrics":metrics,"penetration":pens,"retained_all_seeds":retained,"mean_objective":float(np.mean([refinement_objective(m,probe) for m in metrics]))})
        viable=[row for row in validated if row["retained_all_seeds"]]; chosen=max(viable,key=lambda row:row["mean_objective"]) if viable else None; refined_profiles[source]=(chosen["profile"],chosen["candidate"].candidate_id,chosen) if chosen else (data["profile"],source,None); refinement[source]={"screened_variants":len(screened),"top_screening":[{"candidate_id":row["candidate"].candidate_id,"objective":row["objective"],"penetration_m":row["penetration"]["maximum_penetration_m"]} for row in screened[:5]],"validated":[{"candidate_id":row["candidate"].candidate_id,"retained_all_seeds":row["retained_all_seeds"],"mean_objective":row["mean_objective"],"penetration":aggregate_penetration(row["penetration"],fingers)} for row in validated],"selected":None if chosen is None else chosen["candidate"].candidate_id}
    write_json(root/"local_refinement.json",refinement)
    selected_sources=["grasp_A_candidate_01","grasp_A_candidate_02","grasp_A_candidate_03"][:int(probe["selected_grasp_count"])]; selected={}
    for i,source in enumerate(selected_sources,1):
        profile,variant_id,chosen=refined_profiles[source]; name=f"resource_grasp_A_{i:02d}"; meta={"source_grasp":source,"variant_id":variant_id,"retained_all_20_engineering_windows":chosen is not None and chosen["retained_all_seeds"],"engineering_only":True}; write_profile(grasp_dir/f"{name}.yaml",name,profile,source,meta); selected[name]=profile
    all_rows=[]; summaries={}; selected_runs={}
    for name,profile in selected.items():
        runs=profile_runs(cfg,name,profile,seeds); selected_runs[name]=runs
        for seed,run in enumerate(runs): all_rows.extend(resource_rows(name,seed,run,int(probe["resource_sample_stride"])))
        post=[int(np.searchsorted(run.arrays["time"],run.metadata["support_release_event"]["support_release_time"],side="right")) for run in runs]; normalized=np.concatenate([run.arrays["normalized_joint_range_position"][idx:] for run,idx in zip(runs,post)]); margins=np.concatenate([np.min(run.arrays["distance_to_joint_limits"][idx:],axis=(1,2)) for run,idx in zip(runs,post)]); util=np.concatenate([run.arrays["absolute_control_utilization"][idx:] for run,idx in zip(runs,post)]); reserves=np.concatenate([np.minimum(run.arrays["remaining_positive_control_range"][idx:],run.arrays["remaining_negative_control_range"][idx:]) for run,idx in zip(runs,post)]); active=np.concatenate([run.arrays["active_object_fingers"][idx:] for run,idx in zip(runs,post)]); forces=np.concatenate([run.arrays["finger_object_normal_force_raw"][idx:] for run,idx in zip(runs,post)]); summaries[name]={"mean_normalized_joint_position":np.mean(normalized,axis=0).tolist(),"joint_margin_samples_rad":margins.tolist(),"minimum_joint_margin_rad":float(margins.min()),"mean_actuator_utilization":float(util.mean()),"mean_actuator_reserve_Nm":float(reserves.mean()),"A_contact_fraction_per_finger":dict(zip(fingers,np.mean(active,axis=0).astype(float))),"mean_A_force_per_finger_N":dict(zip(fingers,np.mean(forces,axis=0).astype(float))),"mean_active_A_fingers":float(np.mean(np.sum(active,axis=1)))}
    write_rows_csv(root/"resource_raw_samples.csv",all_rows); write_json(root/"grasp_resource_summaries.json",summaries)
    clouds={}
    for name,profile in selected.items():
        for finger in fingers:
            cloud=reachability_cloud(cfg,profile,finger,int(probe["reachability_samples_per_finger"]),float(probe["reachability_fraction_amplitude"])); clouds[f"{name}:{finger}"]=cloud; path=root/"reachability"/f"{name}_{finger}.npz"; path.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(path,points=cloud["points"],targets=cloud["targets"],columns=np.asarray(cloud["columns"]))
    probe_rows=[]
    for name,profile in selected.items():
        for seed in seeds:
            for finger in fingers:
                _,metrics=run_b_probe(cfg,name,profile,finger,seed,int(probe["reachability_samples_per_finger"]),float(probe["reachability_fraction_amplitude"])); probe_rows.append(metrics)
    write_json(root/"B_probe_records.json",probe_rows); serial=[{key:(json.dumps(value) if isinstance(value,(list,dict)) else value) for key,value in row.items()} for row in probe_rows]; write_rows_csv(root/"B_probe_records.csv",serial)
    correlation_rows=[]
    for row in probe_rows:
        summary=summaries[row["grasp"]]; cloud=clouds[f"{row['grasp']}:{row['finger']}"]; ext=np.ptp(cloud["points"][:,3:6],axis=0); correlation_rows.append({**row,"non_A_contacting_fingers":4-summary["mean_active_A_fingers"],"minimum_joint_margin_rad":summary["minimum_joint_margin_rad"],"mean_actuator_reserve_Nm":summary["mean_actuator_reserve_Nm"],"reachable_workspace_bbox_volume_m3":float(np.prod(ext)),"B_contact_numeric":float(row["B_contact_occurred"])})
    correlations=pearson_correlations(correlation_rows,["non_A_contacting_fingers","minimum_joint_margin_rad","mean_actuator_reserve_Nm","reachable_workspace_bbox_volume_m3"],["minimum_fingertip_to_B_signed_distance_m","B_contact_numeric","A_maximum_translation_m","A_maximum_rotation_rad"]); write_json(root/"exploratory_correlations.json",correlations)
    figures=plot_all(root,summaries,probe_rows,clouds,fingers,list(cfg.hand.joint_names)); write_json(root/"manifest.json",{"engineering_only":True,"selected_grasps":list(selected),"resource_rows":len(all_rows),"B_probe_rows":len(probe_rows),"figures":[str(path) for path in figures],"resource_metric_J_defined":False}); print(json.dumps({"selected_grasps":list(selected),"resource_rows":len(all_rows),"B_probes":len(probe_rows),"B_contacts":sum(row["B_contact_occurred"] for row in probe_rows),"figures":len(figures)},indent=2)); return 0

if __name__=="__main__": raise SystemExit(main())
