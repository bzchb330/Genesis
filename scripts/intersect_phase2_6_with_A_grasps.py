#!/usr/bin/env python
from __future__ import annotations

import json, math
import numpy as np
from scipy.stats import qmc
import yaml

from seqgrasp.config import ROOT
from seqgrasp.experiments.b_workspace import analyze_B_geometry_state, free_fingertip_workspace_clouds, stratified_representative_ids
from seqgrasp.experiments.resource_components import RESOURCE_RECORDS_FILENAME
from seqgrasp.experiments.second_grasp import BPlacement
from seqgrasp.phase2_6_config import load_phase2_6_config
from seqgrasp.phase2_config import load_phase2_config


def main():
    cfg26,_=load_phase2_6_config(); phase2,_=load_phase2_config(ROOT/cfg26.frozen_phase2_config); regions=yaml.safe_load((ROOT/"configs"/"phase2_6_b_only_graspable_regions.yaml").read_text(encoding="utf-8"))["regions"]
    dataset=ROOT/phase2.persistence.output_dir/"grasp_dataset"/"fcc7835446a4"; accepted=[json.loads(x) for x in (dataset/"accepted_grasps.jsonl").read_text(encoding="utf-8").splitlines() if x]; resources=[json.loads(x) for x in (dataset/RESOURCE_RECORDS_FILENAME).read_text(encoding="utf-8").splitlines() if x]
    ids=stratified_representative_ids(accepted,resources,cfg26.sequential.intersection_A_grasps); by_id={r["grasp_id"]:r for r in accepted}; selected=[by_id[x] for x in ids]
    poses=[]
    for ri,region in enumerate(regions):
        b=region["center_bounds_m"]; low=np.asarray([b["x"][0],b["y"][0],b["z"][0],region["yaw_bounds_rad"][0]]); high=np.asarray([b["x"][1],b["y"][1],b["z"][1],region["yaw_bounds_rad"][1]])
        unit=qmc.LatinHypercube(4,seed=np.random.default_rng(np.random.SeedSequence([cfg26.seeds.calibration_B_namespace,ri]))).random(20)
        for pi,row in enumerate(qmc.scale(unit,low,high)):
            yaw=float(row[3]); poses.append((region["name"],BPlacement(ri*20+pi,tuple(row[:3]),(math.cos(yaw/2),0.,0.,math.sin(yaw/2)),yaw)))
    rows=[]
    for gi,record in enumerate(selected):
        cfg,model,data,clouds,radii=free_fingertip_workspace_clouds(record,phase2.resources,phase2.resources.workspace_samples,cfg26.seeds.calibration_B_namespace)
        for region_name,placement in poses:
            result=analyze_B_geometry_state(cfg,model,data,clouds,radii,phase2.resources,placement); rows.append({"grasp_id":record["grasp_id"],"occupied_finger_count":record["occupied_finger_count"],"occupied_finger_mask":record["occupied_finger_mask"],"region":region_name,"pose_index":placement.index,"position_m":list(placement.position_m),"yaw_rad":placement.yaw_rad,**result})
        print(f"A-held geometry: {gi+1}/{len(selected)}",flush=True)
    out=ROOT/cfg26.output_dir/"A_held_intersection"; out.mkdir(parents=True,exist_ok=True)
    with (out/"geometry_trials.jsonl").open("w",encoding="utf-8") as f:
        for row in rows:f.write(json.dumps(row,separators=(",",":"))+"\n")
    region_summary={}
    for region in regions:
        subset=[r for r in rows if r["region"]==region["name"]]; per_grasp={g: any(r["reachable"] for r in subset if r["grasp_id"]==g) for g in ids}; pose_fractions=[]
        for pose_index in sorted({r["pose_index"] for r in subset}):
            p=[r for r in subset if r["pose_index"]==pose_index]; pose_fractions.append(sum(r["reachable"] for r in p)/len(p))
        region_summary[region["name"]]={"A_grasps_with_any_access":sum(per_grasp.values()),"A_grasp_any_access_fraction":sum(per_grasp.values())/len(per_grasp),"mean_typical_pose_access_fraction":float(np.mean(pose_fractions)),"minimum_pose_access_fraction":float(np.min(pose_fractions)),"maximum_pose_access_fraction":float(np.max(pose_fractions)),"initial_A_overlap_count":sum(r["initial_collision_A"] for r in subset),"B_only_robustness_fraction":region["B_only_robustness_fraction"]}
    safe=[r for r in regions if region_summary[r["name"]]["mean_typical_pose_access_fraction"]>0 and region_summary[r["name"]]["initial_A_overlap_count"]==0]
    eligible=[r for r in safe if .2<=region_summary[r["name"]]["mean_typical_pose_access_fraction"]<=.8]
    if not safe: raise RuntimeError("no B-only graspable region has nonzero A-held access without initial A overlap")
    choice=max(eligible,key=lambda r:r["B_only_robustness_fraction"]) if eligible else max(safe,key=lambda r:(region_summary[r["name"]]["mean_typical_pose_access_fraction"],r["B_only_robustness_fraction"])); frozen={"experiment_id":"phase2_6_formal_v3","selection_basis":"B-only robustness plus A-held geometry only","source_region":choice["name"],"center_bounds_m":choice["center_bounds_m"],"yaw_bounds_rad":choice["yaw_bounds_rad"],"vertical_cylinder":True,"formal_seed_namespace":cfg26.seeds.formal_v3_B_namespace}
    (ROOT/"configs"/"phase2_6_frozen_B_distribution.yaml").write_text(yaml.safe_dump(frozen,sort_keys=False),encoding="utf-8"); summary={"A_grasp_count":len(selected),"B_poses_per_region":20,"regions":region_summary,"selected_frozen_region":frozen}; (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8"); print(json.dumps(summary,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
