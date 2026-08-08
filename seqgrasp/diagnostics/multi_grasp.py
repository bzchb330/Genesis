from __future__ import annotations

from dataclasses import asdict,replace
from pathlib import Path
import csv
import json
import numpy as np
import yaml
import mujoco

from ..config import ConfigBundle,DiagnosticProfile,ROOT
from ..control import resolve_hand_indices
from ..scene_builder import build_scene,randomize_objects
from .grasp_search import SearchCandidate,candidate_bundle,diagnostic_metrics,load_search_config
from .scripted_grasp import DiagnosticRun,run_scripted_grasp

def load_resource_probe_config(path: str|Path|None=None)->dict:
    source=Path(path) if path else ROOT/"configs"/"resource_probing_a.yaml"; cfg=yaml.safe_load(source.read_text(encoding="utf-8"))
    if cfg.get("engineering_only") is not True: raise ValueError("resource probing must remain engineering_only")
    return cfg

def load_grasp_profile(path: str|Path)->tuple[dict,DiagnosticProfile]:
    payload=yaml.safe_load(Path(path).read_text(encoding="utf-8")); return payload,DiagnosticProfile(**payload["diagnostic_profile"])

def bundle_for_profile(cfg:ConfigBundle,name:str,profile:DiagnosticProfile)->ConfigBundle:
    profiles=dict(cfg.diagnostic.profiles); profiles[name]=profile; return replace(cfg,diagnostic=replace(cfg.diagnostic,profiles=profiles,active_profile=name,save_plots=False,save_csv=False,save_npz=False,render_video=False))

def penetration_summary(run:DiagnosticRun,cfg:ConfigBundle)->dict:
    a=run.arrays; event=run.metadata["support_release_event"]; release=int(np.searchsorted(a["time"],event["support_release_time"],side="right")); before=max(0,release-1); after=release; steady=np.flatnonzero(a["time"]>=a["time"][-1]-0.2); fingers=run.metadata["finger_order"]; distances=a["finger_object_contact_distance_m"]; counts=a["finger_object_contact_count"]; active=np.where(counts>0,distances,np.inf); flat=int(np.argmin(active)); ti,fi=np.unravel_index(flat,active.shape); per_finger={}
    for i,finger in enumerate(fingers):
        valid=np.flatnonzero(counts[:,i]>0)
        if len(valid):
            j=int(valid[np.argmin(distances[valid,i])]); per_finger[finger]={"maximum_penetration_m":float(max(0,-distances[j,i])),"time_s":float(a["time"][j]),"normal_force_N":float(a["finger_object_normal_force_raw"][j,i])}
        else: per_finger[finger]={"maximum_penetration_m":0.0,"time_s":None,"normal_force_N":0.0}
    def sample(index): return {finger:float(max(0,-distances[index,i])) if counts[index,i]>0 else 0.0 for i,finger in enumerate(fingers)}
    steady_values=active[steady]; steady_values=steady_values[np.isfinite(steady_values)]
    after_min=float(np.min(active[after])); steady_min=float(np.min(active[steady])); relaxes=bool(np.isfinite(after_min) and steady_min>after_min)
    return {"seed":run.metadata["seed"],"maximum_negative_geom_distance_m":float(active[ti,fi]),"maximum_penetration_m":float(-active[ti,fi]),"deepest_finger":fingers[fi],"deepest_configured_geom":cfg.hand.finger_geom_mapping[fingers[fi]],"time_of_maximum_penetration_s":float(a["time"][ti]),"penetration_before_release_m":sample(before),"penetration_after_release_m":sample(after),"maximum_steady_hold_penetration_m":0.0 if not len(steady_values) else float(max(0,-np.min(steady_values))),"normal_force_at_maximum_penetration_N":float(a["finger_object_normal_force_raw"][ti,fi]),"object_linear_velocity_at_maximum_m_per_s":a["object_linear_velocity"][ti].tolist(),"finger_joint_velocity_at_maximum_rad_per_s":a["joint_velocities"][ti].tolist(),"penetration_relaxes_after_release":relaxes,"per_finger":per_finger,"contacting_geom_count":int(sum(np.any(counts[:,i]>0) for i in range(len(fingers))))}

def local_variants(cfg:ConfigBundle,profile:DiagnosticProfile,source_id:str,count:int,probe_cfg:dict)->list[SearchCandidate]:
    rng=np.random.default_rng(int(probe_cfg["local_search_seed"])+sum(map(ord,source_id))); names=list(cfg.hand.actuator_names); low,high=probe_cfg["joint_fraction_perturbation"]; dlo,dhi=probe_cfg["delay_perturbation_seconds"]; tlo,thi=probe_cfg["timing_scale_bounds"]; base_hold=profile.hold_joint_fractions or profile.closed_joint_fractions; base_delay=profile.actuator_close_delay_seconds or {}; out=[]
    for i in range(count):
        closed={name:float(np.clip(profile.closed_joint_fractions[name]+rng.uniform(low,high),0,1)) for name in names}; hold={name:float(np.clip(base_hold[name]+rng.uniform(low,high),0,1)) for name in names}; delays={name:float(max(0,base_delay.get(name,0)+rng.uniform(dlo,dhi))) for name in names}; out.append(SearchCandidate(f"{source_id}_local_{i+1:03d}",closed,hold,delays,float(profile.stage_durations_seconds["close"]*rng.uniform(tlo,thi)),float(profile.stage_durations_seconds["establish_contact"]*rng.uniform(tlo,thi))))
    return out

def refinement_objective(metrics:dict,probe_cfg:dict)->float:
    penetration=max(0.0,-(metrics["minimum_object_fingertip_contact_distance_m"] or 0.0)); return float(metrics["engineering_retention_score"]-probe_cfg["penetration_penalty_weight"]*penetration/0.05)

def resource_rows(grasp_name:str,seed:int,run:DiagnosticRun,stride:int)->list[dict]:
    a=run.arrays; release=int(np.searchsorted(a["time"],run.metadata["support_release_event"]["support_release_time"],side="right")); fingers=run.metadata["finger_order"]; joints=run.metadata["joint_order"]; actuators=run.metadata["actuator_order"]; rows=[]
    for i in range(release,len(a["time"]),stride):
        row={"grasp":grasp_name,"seed":seed,"time_s":float(a["time"][i]),"time_after_release_s":float(a["time"][i]-run.metadata["support_release_event"]["support_release_time"]),"active_A_fingers":int(a["active_object_finger_count"][i]),"inactive_A_fingers":int(len(fingers)-a["active_object_finger_count"][i]),"A_clearance_m":float(a["table_clearance"][i]),"A_orientation_change_rad":float(a["object_orientation_change_after_release"][i])}
        for j,name in enumerate(joints):
            row.update({f"q_{name}_rad":float(a["joint_positions"][i,j]),f"qdot_{name}_rad_s":float(a["joint_velocities"][i,j]),f"joint_lower_{name}_rad":float(a["joint_limits"][i,j,0]),f"joint_upper_{name}_rad":float(a["joint_limits"][i,j,1]),f"lower_margin_{name}_rad":float(a["distance_to_joint_limits"][i,j,0]),f"upper_margin_{name}_rad":float(a["distance_to_joint_limits"][i,j,1]),f"normalized_range_{name}":float(a["normalized_joint_range_position"][i,j])})
        for j,name in enumerate(actuators): row.update({f"control_{name}_Nm":float(a["actuator_controls"][i,j]),f"control_lower_{name}_Nm":float(a["actuator_control_limits"][i,j,0]),f"control_upper_{name}_Nm":float(a["actuator_control_limits"][i,j,1]),f"control_utilization_{name}":float(a["absolute_control_utilization"][i,j]),f"positive_reserve_{name}_Nm":float(a["remaining_positive_control_range"][i,j]),f"negative_reserve_{name}_Nm":float(a["remaining_negative_control_range"][i,j])})
        for j,finger in enumerate(fingers):
            row.update({f"A_contact_{finger}":int(a["active_object_fingers"][i,j]),f"A_contact_count_{finger}":int(a["finger_object_contact_count"][i,j]),f"A_normal_force_{finger}_N":float(a["finger_object_normal_force_raw"][i,j]),f"tactile_contact_{finger}":float(a["tactile_contact_flags"][i,j]),f"tactile_force_{finger}_N":float(a["tactile_normal_force"][i,j])})
            for axis,label in enumerate("xyz"): row[f"A_contact_position_{finger}_{label}_m"]=float(a["finger_object_contact_position_world"][i,j,axis]); row[f"A_contact_normal_{finger}_{label}"]=float(a["finger_object_contact_normal_world"][i,j,axis])
        for axis,label in enumerate("xyz"): row[f"A_position_{label}_m"]=float(a["object_position"][i,axis]); row[f"A_linear_velocity_{label}_m_s"]=float(a["object_linear_velocity"][i,axis]); row[f"A_angular_velocity_{label}_rad_s"]=float(a["object_angular_velocity"][i,axis]); row[f"palm_position_{label}_m"]=float(a["palm_pose"][i,axis])
        for q in range(4): row[f"A_quaternion_{q}"]=float(a["object_orientation"][i,q]); row[f"palm_quaternion_{q}"]=float(a["palm_pose"][i,3+q])
        rows.append(row)
    return rows

def write_rows_csv(path:Path,rows:list[dict])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f: writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

def reachability_cloud(cfg:ConfigBundle,profile:DiagnosticProfile,finger:str,samples:int,amplitude:float,seed:int=0)->dict:
    model,data=build_scene(cfg); randomize_objects(model,data,cfg,np.random.default_rng(seed)); indices=resolve_hand_indices(model,cfg.hand); names=list(cfg.hand.actuator_names); fingers=list(cfg.hand.finger_geom_mapping); fi=fingers.index(finger); configured_groups=load_search_config()["finger_groups"]; group=np.asarray([names.index(name) for name in configured_groups[finger]]); limits=model.jnt_range[indices.joint_ids]; base=profile.hold_joint_fractions or profile.closed_joint_fractions; q=limits[:,0]+np.asarray([base[name] for name in names])*(limits[:,1]-limits[:,0]); palm=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,cfg.hand.palm_body); tip=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,cfg.hand.fingertip_bodies[fi]); aid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,cfg.diagnostic.object_name); bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"object_b"); rng=np.random.default_rng(seed+fi); points=[]; targets=[]
    for _ in range(samples):
        candidate=q.copy(); span=limits[group,1]-limits[group,0]; candidate[group]=np.clip(candidate[group]+rng.uniform(-amplitude,amplitude,len(group))*span,limits[group,0],limits[group,1]); data.qpos[indices.qpos_addresses]=candidate; mujoco.mj_forward(model,data); world=data.xpos[tip].copy(); points.append(np.r_[world,world-data.xpos[palm],world-data.xpos[aid],world-data.xpos[bid]]); targets.append(candidate)
    return {"finger":finger,"columns":["world_x","world_y","world_z","palm_dx","palm_dy","palm_dz","A_dx","A_dy","A_dz","B_dx","B_dy","B_dz"],"points":np.asarray(points),"targets":np.asarray(targets)}

def choose_b_approach(cfg:ConfigBundle,profile:DiagnosticProfile,finger:str,seed:int,samples:int,amplitude:float)->tuple[np.ndarray,float]:
    cloud=reachability_cloud(cfg,profile,finger,samples,amplitude,seed); distances=np.linalg.norm(cloud["points"][:,9:12],axis=1); index=int(np.argmin(distances)); return cloud["targets"][index],float(distances[index])

def run_b_probe(cfg:ConfigBundle,name:str,profile:DiagnosticProfile,finger:str,seed:int,samples:int,amplitude:float)->tuple[DiagnosticRun,dict]:
    local=bundle_for_profile(cfg,name,profile); target,kinematic_distance=choose_b_approach(cfg,profile,finger,seed,samples,amplitude); fingers=list(cfg.hand.finger_geom_mapping); fi=fingers.index(finger); names=list(cfg.hand.actuator_names); group=np.asarray([names.index(actuator) for actuator in load_search_config()["finger_groups"][finger]])
    def transform(stage,elapsed,desired,limits):
        if stage!="hold": return desired
        blend=0.5-0.5*np.cos(np.pi*min(1.0,elapsed/0.4)); out=desired.copy(); out[group]=(1-blend)*desired[group]+blend*target[group]; return out
    run=run_scripted_grasp(local,seed=seed,save_outputs=False,target_transform=transform); a=run.arrays; hold=np.flatnonzero(a["diagnostic_stage"]=="hold"); start=int(hold[0]); base_pos=a["object_position"][start]; base_quat=a["object_orientation"][start]; translations=np.linalg.norm(a["object_position"][hold]-base_pos,axis=1); rotations=2*np.arccos(np.clip(np.abs(a["object_orientation"][hold]@base_quat),0,1)); forces=a["finger_object_normal_force_raw"][hold]; redistribution=np.linalg.norm(forces-forces[0],axis=1); bcontact=a["finger_b_contact_count"][hold,fi]>0; first=np.flatnonzero(bcontact)
    metrics={"engineering_only":True,"scientific_success_assigned":False,"grasp":name,"seed":seed,"finger":finger,"B_initial_position_m":a["object_b_position"][0].tolist(),"B_initial_orientation":a["object_b_orientation"][0].tolist(),"kinematic_sample_closest_center_distance_m":kinematic_distance,"minimum_fingertip_to_B_signed_distance_m":float(np.min(a["finger_b_signed_distance_m"][hold,fi])),"B_contact_occurred":bool(np.any(bcontact)),"first_contact_finger":finger if np.any(bcontact) else None,"joint_configuration_at_first_B_contact_rad":None if not len(first) else a["joint_positions"][hold[first[0]]].tolist(),"A_maximum_translation_m":float(np.max(translations)),"A_maximum_rotation_rad":float(np.max(rotations)),"A_vertical_displacement_m":float(a["object_position"][hold[-1],2]-base_pos[2]),"A_maximum_force_redistribution_N":float(np.max(redistribution)),"A_contact_pattern_changes":int(np.sum(np.any(a["active_object_fingers"][hold][1:]!=a["active_object_fingers"][hold][:-1],axis=1))),"A_complete_contact_loss_event":bool(np.any(a["active_object_finger_count"][hold]==0)),"A_table_contact_event":bool(np.any(a["object_table_contact"][hold]>0)),"maximum_actuator_utilization":float(np.max(a["absolute_control_utilization"][hold])),"minimum_joint_margin_after_probe_rad":float(np.min(a["distance_to_joint_limits"][hold[-1]]))}
    return run,metrics

def pearson_correlations(rows:list[dict],predictors:list[str],outcomes:list[str])->dict:
    result={"exploratory_only":True,"correlations":{}}
    for x in predictors:
        result["correlations"][x]={}
        for y in outcomes:
            a=np.asarray([row[x] for row in rows],float); b=np.asarray([row[y] for row in rows],float); result["correlations"][x][y]=None if np.std(a)==0 or np.std(b)==0 else float(np.corrcoef(a,b)[0,1])
    return result

def write_json(path:Path,payload)->None: path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,indent=2),encoding="utf-8")

def write_profile(path:Path,name:str,profile:DiagnosticProfile,source:str,metadata:dict)->None:
    payload={"engineering_only":True,"name":name,"source":source,"scientific_success_assigned":False,"selection_metadata":metadata,"diagnostic_profile":asdict(profile)}; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(yaml.safe_dump(payload,sort_keys=False),encoding="utf-8")
