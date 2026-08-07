from __future__ import annotations

from dataclasses import asdict,dataclass,replace
from pathlib import Path
import json
import math
import tempfile
import os
import numpy as np
import yaml

os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"seqgrasp-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..config import ConfigBundle,DiagnosticProfile,ROOT
from .scripted_grasp import DiagnosticRun,run_scripted_grasp

@dataclass(frozen=True)
class SearchCandidate:
    candidate_id: str
    closed_joint_fractions: dict[str,float]
    hold_joint_fractions: dict[str,float]
    actuator_close_delay_seconds: dict[str,float]
    close_duration_seconds: float
    establish_contact_duration_seconds: float

def load_search_config(path: str|Path|None=None)->dict:
    source=Path(path) if path else ROOT/"configs"/"grasp_search_a.yaml"
    cfg=yaml.safe_load(source.read_text(encoding="utf-8"))
    if cfg.get("engineering_search_only") is not True: raise ValueError("grasp search must remain engineering_search_only")
    if cfg.get("method")!="latin_hypercube": raise ValueError("only the configured reproducible Latin-hypercube method is supported")
    return cfg

def latin_hypercube(count:int,dimensions:int,seed:int)->np.ndarray:
    if count<1 or dimensions<1: raise ValueError("Latin-hypercube dimensions and count must be positive")
    rng=np.random.default_rng(seed); result=np.empty((count,dimensions))
    for j in range(dimensions): result[:,j]=(rng.permutation(count)+rng.random(count))/count
    return result

def generate_candidates(cfg:ConfigBundle,search_cfg:dict,count:int|None=None)->list[SearchCandidate]:
    names=list(cfg.hand.actuator_names); bounds=search_cfg["joint_fraction_bounds"]
    if set(bounds)!=set(names): raise ValueError("search joint bounds must match configured actuator names")
    groups=search_cfg["finger_groups"]
    if set(name for members in groups.values() for name in members)!=set(names): raise ValueError("search finger groups must cover configured actuators exactly once")
    n=int(count or search_cfg["candidate_count"]); group_names=list(groups); dimensions=len(names)+2*len(group_names)+2; lhs=latin_hypercube(n,dimensions,int(search_cfg["search_seed"])); candidates=[]
    hold_bounds=search_cfg["hold_fraction_delta_bounds"]; delay_bounds=search_cfg["close_delay_bounds_seconds"]; close_bounds=search_cfg["close_duration_bounds_seconds"]; establish_bounds=search_cfg["establish_contact_duration_bounds_seconds"]
    for i,row in enumerate(lhs):
        cursor=0; closed={}
        for name in names:
            low,high=bounds[name]; closed[name]=float(low+(high-low)*row[cursor]); cursor+=1
        hold=dict(closed); delays={}
        for group in group_names:
            low,high=hold_bounds; delta=low+(high-low)*row[cursor]; cursor+=1
            for name in groups[group]: hold[name]=float(np.clip(hold[name]+delta,0.0,1.0))
        for group in group_names:
            low,high=delay_bounds; delay=float(low+(high-low)*row[cursor]); cursor+=1
            for name in groups[group]: delays[name]=delay
        low,high=close_bounds; close=float(low+(high-low)*row[cursor]); cursor+=1
        low,high=establish_bounds; establish=float(low+(high-low)*row[cursor])
        candidates.append(SearchCandidate(f"candidate_{i+1:04d}",closed,hold,delays,close,establish))
    return candidates

def candidate_profile(base:DiagnosticProfile,candidate:SearchCandidate)->DiagnosticProfile:
    durations=dict(base.stage_durations_seconds); durations["close"]=candidate.close_duration_seconds; durations["establish_contact"]=candidate.establish_contact_duration_seconds
    return replace(base,stage_durations_seconds=durations,closed_joint_fractions=candidate.closed_joint_fractions,hold_joint_fractions=candidate.hold_joint_fractions,actuator_close_delay_seconds=candidate.actuator_close_delay_seconds)

def candidate_bundle(cfg:ConfigBundle,candidate:SearchCandidate)->ConfigBundle:
    profile=candidate_profile(cfg.diagnostic.profiles[cfg.diagnostic.active_profile],candidate); profiles=dict(cfg.diagnostic.profiles); profiles[candidate.candidate_id]=profile
    return replace(cfg,diagnostic=replace(cfg.diagnostic,profiles=profiles,active_profile=candidate.candidate_id,save_plots=False,save_csv=False,save_npz=False,render_video=False))

def _first_elapsed(mask:np.ndarray,dt:float)->float|None:
    indices=np.flatnonzero(mask); return None if not len(indices) else float(indices[0]*dt)

def diagnostic_metrics(run:DiagnosticRun,cfg:ConfigBundle,search_cfg:dict)->dict:
    a=run.arrays; event=run.metadata["support_release_event"]; release_index=int(np.searchsorted(a["time"],event["support_release_time"],side="right")); post=slice(release_index,None); dt=cfg.scene.timestep
    release_z=float(event["before"]["object_position"][2]); resting=float(run.metadata["table_resting_center_z_m"]); span=max(release_z-resting,np.finfo(float).eps); height_fraction=np.clip((a["object_position"][post,2]-resting)/span,0.0,1.0)
    active=a["active_object_finger_count"][post]; table=a["object_table_contact"][post].astype(bool); saturation=np.any(a["actuator_saturated"][post]>0,axis=1); translation=a["object_translational_displacement_after_release"][post]; rotation=a["object_orientation_change_after_release"][post]
    contact_fraction=float(np.mean(active>0)); active_fraction=float(np.mean(active)/len(run.metadata["finger_order"])); translation_norm=float(np.max(translation)/search_cfg["object_characteristic_length_m"]); rotation_norm=float(np.max(rotation)/math.pi); saturation_fraction=float(np.mean(saturation)); clear_fraction=float(np.mean(~table)); weights=search_cfg["score_weights"]
    terms={"terminal_height_retention":float(height_fraction[-1]),"mean_height_retention":float(np.mean(height_fraction)),"clear_time_fraction":clear_fraction,"fingertip_contact_fraction":contact_fraction,"active_finger_fraction":active_fraction,"normalized_translation_penalty":translation_norm,"normalized_rotation_penalty":rotation_norm,"actuator_saturation_penalty":saturation_fraction}
    score=sum(weights[key]*terms[key]*( -1.0 if key.endswith("penalty") else 1.0) for key in terms); maximum_joint_limit_excess=float(np.max(a["maximum_joint_limit_excess_rad"])); safety_valid=maximum_joint_limit_excess<=float(search_cfg["joint_limit_tolerance_rad"])
    if not safety_valid: score=-1_000_000.0-maximum_joint_limit_excess
    forces=a["finger_object_normal_force_raw"][post]; positions=a["finger_object_contact_position_world"][release_index]; normals=a["finger_object_contact_normal_world"][release_index]; present=a["finger_object_contact_count"][release_index]>0; pairwise=[float(np.dot(normals[i],normals[j])) for i in range(len(normals)) for j in range(i+1,len(normals)) if present[i] and present[j]]
    active_distances=a["finger_object_contact_distance_m"][post][a["finger_object_contact_count"][post]>0]; minimum_contact_distance=None if not len(active_distances) else float(np.min(active_distances))
    return {"engineering_search_only":True,"engineering_retention_score":float(score),"score_terms":terms,"safety_valid":safety_valid,"support_release_time_s":float(event["support_release_time"]),"final_vertical_displacement_m":float(event["final_vertical_displacement_m"]),"minimum_post_release_height_m":float(np.min(a["object_position"][post,2])),"maximum_post_release_height_m":float(np.max(a["object_position"][post,2])),"maximum_post_release_translation_m":float(np.max(translation)),"maximum_post_release_orientation_change_rad":float(np.max(rotation)),"fingertip_contact_duration_s":float(np.sum(active>0)*dt),"first_complete_fingertip_contact_loss_s":_first_elapsed(active==0,dt),"table_recontact_time_s":_first_elapsed(table,dt),"mean_active_object_fingers":float(np.mean(active)),"peak_force_per_finger_N":dict(zip(run.metadata["finger_order"],np.max(forces,axis=0).astype(float))),"mean_force_per_finger_N":dict(zip(run.metadata["finger_order"],np.mean(forces,axis=0).astype(float))),"contacting_fingers_at_release":dict(zip(run.metadata["finger_order"],present.astype(bool).tolist())),"contact_positions_at_release_m":dict(zip(run.metadata["finger_order"],positions.tolist())),"inward_contact_normals_at_release":dict(zip(run.metadata["finger_order"],normals.tolist())),"minimum_pairwise_contact_normal_dot":None if not pairwise else min(pairwise),"minimum_object_fingertip_contact_distance_m":minimum_contact_distance,"final_object_fingertip_contact_distance_m":dict(zip(run.metadata["finger_order"],a["finger_object_contact_distance_m"][-1].astype(float))),"actuator_saturation_fraction":saturation_fraction,"maximum_saturated_actuators":int(np.max(a["actuator_saturation_count"][post])),"maximum_joint_limit_excess_rad":maximum_joint_limit_excess,"joint_limit_violation_observed":not safety_valid,"terminated_early":run.metadata["terminated_early"],"termination_reason":run.metadata["termination_reason"],"scientific_label_assigned":False}

def evaluate_candidate(cfg:ConfigBundle,search_cfg:dict,candidate:SearchCandidate,seed:int)->tuple[DiagnosticRun,dict]:
    candidate_cfg=candidate_bundle(cfg,candidate); run=run_scripted_grasp(candidate_cfg,seed=seed,save_outputs=False); return run,diagnostic_metrics(run,candidate_cfg,search_cfg)

def candidate_vector(candidate:SearchCandidate,names:list[str])->np.ndarray:
    return np.asarray([candidate.closed_joint_fractions[name] for name in names]+[candidate.hold_joint_fractions[name] for name in names])

def select_distinct(results:list[dict],candidates:dict[str,SearchCandidate],names:list[str],count:int,min_distance:float)->list[dict]:
    selected=[]; vectors=[]
    for result in sorted(results,key=lambda item:item["metrics"]["engineering_retention_score"],reverse=True):
        vector=candidate_vector(candidates[result["candidate_id"]],names)
        if all(np.linalg.norm(vector-other)>=min_distance for other in vectors): selected.append(result); vectors.append(vector)
        if len(selected)==count: break
    if len(selected)<count:
        for result in sorted(results,key=lambda item:item["metrics"]["engineering_retention_score"],reverse=True):
            if result not in selected: selected.append(result)
            if len(selected)==count: break
    return selected

def save_raw_trajectory(path:Path,run:DiagnosticRun)->None:
    keys=("time","object_position","object_orientation","object_linear_velocity","object_angular_velocity","table_clearance","object_table_contact","object_translational_displacement_after_release","object_orientation_change_after_release","active_object_fingers","inactive_object_fingers","active_object_finger_count","finger_object_contact_count","finger_object_contact_position_world","finger_object_contact_normal_world","finger_object_contact_distance_m","finger_object_normal_force_raw","finger_total_normal_force_raw","actuator_controls","actuator_saturated","joint_positions","maximum_joint_limit_excess_rad")
    path.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(path,**{key:run.arrays[key] for key in keys})

def write_candidate_yaml(path:Path,candidate:SearchCandidate,metrics:dict,profile:DiagnosticProfile)->None:
    payload={"engineering_search_only":True,"candidate_id":candidate.candidate_id,"engineering_retention_score":metrics["engineering_retention_score"],"scientific_success_assigned":False,"diagnostic_profile":asdict(profile)}
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(yaml.safe_dump(payload,sort_keys=False),encoding="utf-8")

def write_json(path:Path,payload)->None:
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,indent=2),encoding="utf-8")

def comparison_plots(output_dir:Path,baseline:DiagnosticRun,best:DiagnosticRun,validation_runs:dict[str,list[DiagnosticRun]])->list[Path]:
    output_dir.mkdir(parents=True,exist_ok=True); written=[]
    specs=(("object_position",2,"baseline_vs_top_height.png","Object center height","z [m]"),("object_linear_velocity",2,"baseline_vs_top_vertical_velocity.png","Object vertical velocity","v-z [m/s]"),("active_object_finger_count",None,"baseline_vs_top_active_fingers.png","Active object-contacting fingers","count"),("total_configured_fingertip_normal_force",None,"baseline_vs_top_total_force.png","Configured-fingertip normal force","force [N]"))
    for key,column,name,title,ylabel in specs:
        fig,ax=plt.subplots(figsize=(8,4.5))
        for label,run in (("baseline",baseline),("top candidate",best)):
            values=run.arrays[key] if column is None else run.arrays[key][:,column]; release=run.metadata["support_release_event"]["support_release_time"]; ax.plot(run.arrays["time"]-release,values,label=label)
        ax.axvline(0,color="black",linestyle="--",label="support release"); ax.set(title=title,xlabel="time relative to support release [s]",ylabel=ylabel); ax.grid(True,alpha=.25); ax.legend(); fig.tight_layout(); path=output_dir/name; fig.savefig(path,dpi=160); plt.close(fig); written.append(path)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for candidate_id,runs in validation_runs.items():
        release=runs[0].metadata["support_release_event"]["support_release_time"]
        for run in runs: ax.plot(run.arrays["time"]-release,run.arrays["object_position"][:,2],alpha=.18,linewidth=.7)
        n=min(len(run.arrays["time"]) for run in runs); mean=np.stack([run.arrays["object_position"][:n,2] for run in runs]).mean(axis=0); ax.plot(runs[0].arrays["time"][:n]-release,mean,linewidth=2,label=candidate_id)
    ax.axvline(0,color="black",linestyle="--",label="support release"); ax.set(title="Top-candidate multi-seed object height",xlabel="time relative to support release [s]",ylabel="z [m]"); ax.grid(True,alpha=.25); ax.legend(fontsize="small"); fig.tight_layout(); path=output_dir/"top_candidates_multiseed_height.png"; fig.savefig(path,dpi=160); plt.close(fig); written.append(path)
    labels=list(validation_runs); values=[[run.metadata["support_release_event"]["final_vertical_displacement_m"] for run in validation_runs[label]] for label in labels]; fig,ax=plt.subplots(figsize=(8,4.5)); ax.boxplot(values,tick_labels=labels); ax.set(title="Top-candidate post-release displacement",ylabel="final vertical displacement [m]"); ax.grid(True,axis="y",alpha=.25); fig.autofmt_xdate(rotation=25); fig.tight_layout(); path=output_dir/"top_candidates_vertical_displacement_distribution.png"; fig.savefig(path,dpi=160); plt.close(fig); written.append(path)
    return written
