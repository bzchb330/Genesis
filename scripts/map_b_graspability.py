#!/usr/bin/env python
from __future__ import annotations

import json, math
import mujoco
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc
import yaml

from seqgrasp.config import ROOT, load_configs
from seqgrasp.control import resolve_hand_indices
from seqgrasp.diagnostics.multi_grasp import load_grasp_profile
from seqgrasp.diagnostics.scripted_grasp import _joint_target
from seqgrasp.experiments.grasp_sampling import ferrari_canny_epsilon
from seqgrasp.experiments.phase2_6_workspace import accessible_surface_samples, contact_opposition_angle_deg, lexicographic_pose_key, pairwise_envelope_boxes
from seqgrasp.experiments.resource_components import FINGER_ORDER
from seqgrasp.experiments.second_grasp import BPlacement, _set_b_pose
from seqgrasp.phase2_6_config import load_phase2_6_config
from seqgrasp.phase2_config import load_phase2_config
from seqgrasp.scene_builder import build_scene


def structured_centers(boxes, count, seed):
    names=sorted(boxes); per=math.ceil(count/len(names)); rows=[]
    for i,name in enumerate(names):
        low,high=boxes[name]; unit=qmc.LatinHypercube(3,seed=np.random.default_rng(np.random.SeedSequence([seed,i]))).random(per)
        rows.extend(qmc.scale(unit,low,high))
    rows.sort(key=lambda row:tuple(row)); return np.asarray(rows[:count])


def select_diverse(records,count):
    promising=[r for r in records if r["valid_initial_geometry"] and r["accessible_finger_count"]>=2 and r["opposition_available"]]
    promising.sort(key=lexicographic_pose_key,reverse=True); selected=[]; remaining=promising.copy()
    while remaining and len(selected)<count:
        unseen=[r for r in remaining if tuple(r["accessible_fingers"]) not in {tuple(x["accessible_fingers"]) for x in selected}]
        pool=unseen or remaining
        if not selected: choice=pool[0]
        else:
            prior=np.asarray([r["position_m"] for r in selected]); spans=np.maximum(np.ptp(np.asarray([r["position_m"] for r in promising]),axis=0),1e-12)
            choice=max(pool,key=lambda r:float(np.min(np.linalg.norm((np.asarray(r["position_m"])-prior)/spans,axis=1))))
        selected.append(choice); remaining.remove(choice)
    return selected


def main():
    p26,_=load_phase2_6_config(); p2,_=load_phase2_config(ROOT/p26.frozen_phase2_config); cfg=load_configs(); model,data=build_scene(cfg); indices=resolve_hand_indices(model,cfg.hand)
    _,profile=load_grasp_profile(ROOT/"configs"/"grasps"/"resource_grasp_A_02.yaml"); data.qpos[indices.qpos_addresses]=_joint_target(model,cfg,indices,profile.open_joint_fractions); mujoco.mj_forward(model,data)
    wdir=ROOT/p26.output_dir/"workspace"; clouds={}; joints={}; margins={}; radii={}; trees={}
    for finger in FINGER_ORDER:
        a=np.load(wdir/f"{finger}_workspace.npz"); clouds[finger]=a["positions_world_m"]; joints[finger]=a["joint_rad"]; margins[finger]=a["joint_margin_rad"]; trees[finger]=cKDTree(clouds[finger])
        gid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,cfg.hand.finger_geom_mapping[finger][0]); radii[finger]=float(model.geom_size[gid,0])
    obj=next(x for x in cfg.scene.objects if x.name=="object_b"); radius,half=obj.size[0],obj.size[1]; table=cfg.scene.table_pos[2]+cfg.scene.table_size[2]
    boxes=pairwise_envelope_boxes(clouds,radii,radius,half); boxes={n:(np.maximum(lo,[-np.inf,-np.inf,table+half]),hi) for n,(lo,hi) in boxes.items() if hi[2]>table+half}
    centers=structured_centers(boxes,p26.workspace.candidate_pose_count,p26.seeds.candidate_poses); bgeom=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"object_b_geom"); palm=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,cfg.hand.palm_body); pgeoms=[g for g in range(model.ngeom) if int(model.geom_bodyid[g])==palm]
    search=max(math.sqrt((radius+r+p26.workspace.surface_access_tolerance_m)**2+(half+r+p26.workspace.surface_access_tolerance_m)**2) for r in radii.values()); records=[]
    for ci,center in enumerate(centers):
        _set_b_pose(model,data,BPlacement(ci,tuple(center),(1.,0.,0.,0.),0.)); mind=float("inf"); valid=center[2]-half>=table
        for g in range(model.ngeom):
            if g==bgeom or (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g) or "")=="table": continue
            if model.geom_contype[g] or model.geom_conaffinity[g]: mind=min(mind,float(mujoco.mj_geomDistance(model,data,bgeom,g,1.,None)))
        valid &= mind>=-p2.second_grasp.maximum_penetration_m; cps=[]; normals=[]; access=[]; rq={}; rm=[]; pred=0.
        for finger in FINGER_ORDER:
            nearby=trees[finger].query_ball_point(center,search)
            if not nearby: continue
            pts=clouds[finger][nearby]; mask,contacts,inward=accessible_surface_samples(pts,center,radius,half,radii[finger],p26.workspace.surface_access_tolerance_m)
            if not np.any(mask): continue
            src=np.asarray(nearby)[mask]; d=np.linalg.norm(pts[mask]-contacts,axis=1); k=int(np.argmin(np.abs(d-radii[finger]))); si=int(src[k])
            access.append(finger); cps.append(contacts[k]); normals.append(inward[k]); rq[finger]=joints[finger][si].tolist(); rm.append(float(margins[finger][si])); pred=max(pred,max(0.,radii[finger]-float(d[k])))
        angle=contact_opposition_angle_deg(np.asarray(normals)); eps=ferrari_canny_epsilon(np.asarray(cps),np.asarray(normals),center,obj.friction[0],p2.dataset.friction_cone_edges,float(np.linalg.norm(obj.size)),p2.dataset.convex_hull_tolerance)
        pd=min((float(mujoco.mj_geomDistance(model,data,bgeom,g,1.,None)) for g in pgeoms),default=float("inf"))
        records.append({"candidate_index":ci,"position_m":center.tolist(),"yaw_rad":0.,"valid_initial_geometry":bool(valid),"minimum_initial_distance_m":mind,"accessible_fingers":access,"accessible_finger_count":len(access),"representative_joint_rad":rq,"contact_positions_m":[x.tolist() for x in cps],"inward_contact_normals":[x.tolist() for x in normals],"maximum_opposition_angle_deg":angle,"opposition_available":angle>=p26.workspace.opposition_minimum_angle_deg,"ferrari_canny_epsilon":eps,"palm_support_available":abs(pd)<=p26.workspace.palm_support_tolerance_m,"predicted_penetration_m":pred,"minimum_joint_margin_rad":min(rm,default=0.)})
        if (ci+1)%1000==0: print(f"B-pose map: {ci+1}/{len(centers)}",flush=True)
    selected=select_diverse(records,p26.workspace.selected_pose_count); out=ROOT/p26.output_dir/"b_pose_graspability"; out.mkdir(parents=True,exist_ok=True)
    with (out/"candidate_poses.jsonl").open("w",encoding="utf-8") as f:
        for r in records:f.write(json.dumps(r,separators=(",",":"))+"\n")
    (out/"selected_poses.yaml").write_text(yaml.safe_dump({"selected_poses":selected},sort_keys=False),encoding="utf-8")
    summary={"candidate_pose_count":len(records),"derived_pairwise_envelope_boxes_m":{n:[lo.tolist(),hi.tolist()] for n,(lo,hi) in boxes.items()},"valid_initial_geometry_count":sum(r["valid_initial_geometry"] for r in records),"multi_finger_access_count":sum(r["accessible_finger_count"]>=2 for r in records),"opposing_contact_count":sum(r["opposition_available"] for r in records),"positive_ferrari_canny_count":sum(r["ferrari_canny_epsilon"]>p2.dataset.convex_hull_tolerance for r in records),"selected_pose_count":len(selected),"selected_topologies":sorted({"+".join(r["accessible_fingers"]) for r in selected})}
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8"); print(json.dumps(summary,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
