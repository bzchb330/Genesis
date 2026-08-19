#!/usr/bin/env python
from __future__ import annotations

import argparse, json, math, os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import numpy as np
from scipy.stats import qmc
import yaml

from seqgrasp.config import ROOT, load_configs
from seqgrasp.control import resolve_hand_indices
from seqgrasp.experiments.metadata import config_hash, git_commit_sha
from seqgrasp.experiments.phase2_5_trajectory import run_b_acquisition_trajectory
from seqgrasp.experiments.phase2_6_dynamic import trajectory_from_unit
from seqgrasp.experiments.resumable import IncrementalJsonlStore, stable_trial_id
from seqgrasp.experiments.resource_components import RESOURCE_RECORDS_FILENAME
from seqgrasp.experiments.second_grasp import BPlacement
from seqgrasp.phase2_5_config import load_phase2_5_config
from seqgrasp.phase2_6_config import load_phase2_6_config
from seqgrasp.phase2_config import load_phase2_config
from seqgrasp.scene_builder import build_scene


def evaluate(payload):
    cfg25,traj,A,placement,index,cfg_hash,commit=payload; summary,_=run_b_acquisition_trajectory(cfg25,traj,A_record=A,occupied_mask=np.asarray(A["occupied_finger_mask"],dtype=bool),placement=placement)
    summary.update({"trial_id":stable_trial_id("phase2.6-sequential",{"candidate_index":index,"config_hash":cfg_hash}),"candidate_index":index,"A_grasp_id":A["grasp_id"],"occupied_finger_count":A["occupied_finger_count"],"B_placement_index":placement.index,"experiment_id":"phase2_6_sequential_calibration","calibration_only":True,"config_hash":cfg_hash,"git_commit_sha":commit}); return summary


def main():
    pa=argparse.ArgumentParser(); pa.add_argument("--candidates",type=int); pa.add_argument("--workers",type=int); args=pa.parse_args(); cfg26,source=load_phase2_6_config(); cfg25,_=load_phase2_5_config(ROOT/cfg26.frozen_phase2_5_config); phase2,_=load_phase2_config(ROOT/cfg26.frozen_phase2_config); params=yaml.safe_load((ROOT/"configs"/"phase2_6_sequential_search.yaml").read_text(encoding="utf-8")); frozen=yaml.safe_load((ROOT/"configs"/"phase2_6_frozen_B_distribution.yaml").read_text(encoding="utf-8")); count=args.candidates or cfg26.sequential.initial_candidate_count
    if count not in (cfg26.sequential.initial_candidate_count,cfg26.sequential.expanded_candidate_count): raise ValueError("invalid sequential budget")
    dataset=ROOT/phase2.persistence.output_dir/"grasp_dataset"/"fcc7835446a4"; accepted=[json.loads(x) for x in (dataset/"accepted_grasps.jsonl").read_text(encoding="utf-8").splitlines() if x]; by_id={r["grasp_id"]:r for r in accepted}; resources={r["grasp_id"]:r for r in [json.loads(x) for x in (dataset/RESOURCE_RECORDS_FILENAME).read_text(encoding="utf-8").splitlines() if x]}; geo=[json.loads(x) for x in (ROOT/cfg26.output_dir/"A_held_intersection"/"geometry_trials.jsonl").read_text(encoding="utf-8").splitlines() if x]; eligible={r["grasp_id"] for r in geo if r["region"]==frozen["source_region"] and r["reachable"]}; A_rows=[by_id[x] for x in eligible]
    A_rows.sort(key=lambda r:(int(r["occupied_finger_count"])!=2,-float(resources[r["grasp_id"]]["free_finger_workspace_vol_m3"]),-float(r["ferrari_canny_epsilon"]),r["grasp_id"])); A_rows=A_rows[:int(params["calibration_A_grasps_to_use"])]
    pose_rows=yaml.safe_load((ROOT/cfg26.output_dir/"b_pose_graspability"/"selected_poses.yaml").read_text(encoding="utf-8"))["selected_poses"]; source_index=int(params["source_B_only_candidate_index"]); pose=pose_rows[source_index%len(pose_rows)]; base_unit=qmc.LatinHypercube(70,seed=cfg26.seeds.pose_trajectory_search).random(cfg26.dynamic_search.expanded_candidate_count); base=trajectory_from_unit(cfg25,pose,source_index,base_unit[source_index])
    cfg=load_configs(); model,_=build_scene(cfg); indices=resolve_hand_indices(model,cfg.hand); ranges=model.jnt_range[indices.joint_ids]; unit=qmc.LatinHypercube(70,seed=cfg26.seeds.sequential if hasattr(cfg26.seeds,"sequential") else cfg26.seeds.pose_trajectory_search+1).random(cfg26.sequential.expanded_candidate_count)
    b=frozen["center_bounds_m"]; low=np.asarray([b["x"][0],b["y"][0],b["z"][0],frozen["yaw_bounds_rad"][0]]); high=np.asarray([b["x"][1],b["y"][1],b["z"][1],frozen["yaw_bounds_rad"][1]]); pu=qmc.LatinHypercube(4,seed=cfg26.seeds.calibration_B_namespace).random(int(params["B_poses_per_cycle"])); placements=[]
    for i,row in enumerate(qmc.scale(pu,low,high)):
        yaw=float(row[3]); placements.append(BPlacement(i,tuple(row[:3]),(math.cos(yaw/2),0.,0.,math.sin(yaw/2)),yaw))
    def variant(i):
        row=unit[i]; cursor=0; vals=[]
        for original in (base.approach_joint_rad,base.precontact_joint_rad,base.closing_joint_rad,base.hold_joint_rad):
            delta=np.interp(row[cursor:cursor+16],[0,1],params["joint_target_delta_rad"]); cursor+=16; vals.append(tuple(np.clip(np.asarray(original)+delta,ranges[:,0],ranges[:,1])))
        close=max(1,base.close_steps+int(round(np.interp(row[cursor],[0,1],params["close_steps_delta"])))); cursor+=1; delays=tuple(max(0,x+int(round(np.interp(v,[0,1],params["per_finger_delay_delta_steps"])))) for x,v in zip(base.per_finger_close_delay_steps,row[cursor:cursor+4])); cursor+=4; release=max(0,base.fixture_release_delay_steps+int(round(np.interp(row[cursor],[0,1],params["fixture_release_delay_delta_steps"])))); return replace(base,candidate_index=i,approach_joint_rad=vals[0],precontact_joint_rad=vals[1],closing_joint_rad=vals[2],hold_joint_rad=vals[3],close_steps=close,per_finger_close_delay_steps=delays,fixture_release_delay_steps=release)
    cfg_hash=config_hash([source,ROOT/"configs"/"phase2_6_frozen_B_distribution.yaml",ROOT/"configs"/"phase2_6_sequential_search.yaml",ROOT/cfg26.frozen_phase2_5_config,ROOT/cfg26.output_dir/"A_held_intersection"/"geometry_trials.jsonl"]); out=ROOT/cfg26.output_dir/"sequential_search"/cfg_hash[:12]; store=IncrementalJsonlStore(out/"candidate_results.jsonl",30.,.05); completed=store.completed_ids(); pending=[i for i in range(count) if stable_trial_id("phase2.6-sequential",{"candidate_index":i,"config_hash":cfg_hash}) not in completed]; commit=git_commit_sha(ROOT); workers=min(args.workers or 8,cfg26.maximum_workers)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        buffer=[]; payloads=((cfg25,variant(i),A_rows[i%len(A_rows)],placements[(i//len(A_rows))%len(placements)],i,cfg_hash,commit) for i in pending)
        for n,result in enumerate(ex.map(evaluate,payloads),1):
            buffer.append(result)
            if len(buffer)==workers or n==len(pending):store.append_many(buffer);buffer.clear()
            if n%(workers*4)==0 or n==len(pending):print(f"Phase 2.6 sequential: {len(completed)+n}/{count}",flush=True)
    rows=[r for r in store.records() if int(r["candidate_index"])<count]; success=[r for r in rows if r["BOTH_RETAINED"]]; summary={"status":"PASS" if success else ("PHASE2_6_SEQUENTIAL_INTERSECTION_FAILED" if count==cfg26.sequential.expanded_candidate_count else "EXPANSION_REQUIRED"),"candidate_count":count,"A_grasps_used":len(A_rows),"B_poses_used":len(placements),"BOTH_RETAINED_count":len(success),"distinct_A_grasps":len({r["A_grasp_id"] for r in success}),"distinct_B_poses":len({r["B_placement_index"] for r in success}),"failure_mechanisms":dict(Counter(r["failure_mechanism"] for r in rows if not r["BOTH_RETAINED"])),"config_hash":cfg_hash}; out.mkdir(parents=True,exist_ok=True); (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8"); print(json.dumps(summary,indent=2)); return 0 if success else (3 if count==cfg26.sequential.expanded_candidate_count else 2)


if __name__=="__main__": raise SystemExit(main())
