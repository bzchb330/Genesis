#!/usr/bin/env python
from __future__ import annotations

import argparse, json, os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.stats import qmc
import yaml

from seqgrasp.config import ROOT
from seqgrasp.experiments.metadata import config_hash, git_commit_sha
from seqgrasp.experiments.phase2_5_trajectory import b_only_lexicographic_key, run_b_acquisition_trajectory
from seqgrasp.experiments.phase2_6_dynamic import trajectory_from_unit
from seqgrasp.experiments.resumable import IncrementalJsonlStore, stable_trial_id
from seqgrasp.experiments.second_grasp import BPlacement
from seqgrasp.phase2_5_config import load_phase2_5_config
from seqgrasp.phase2_6_config import load_phase2_6_config


def evaluate(payload):
    cfg25,pose,index,unit,cfg_hash,commit=payload; trajectory=trajectory_from_unit(cfg25,pose,index,unit)
    center=tuple(pose["position_m"]); placement=BPlacement(int(pose["candidate_index"]),center,(1.,0.,0.,0.),0.)
    summary,_=run_b_acquisition_trajectory(cfg25,trajectory,placement=placement)
    summary.update({"pose_candidate_index":int(pose["candidate_index"]),"geometry_accessible_fingers":pose["accessible_fingers"],"geometry_topology":"+".join(pose["accessible_fingers"]),"trial_id":stable_trial_id("phase2.6-b-only",{"candidate_index":index,"config_hash":cfg_hash}),"experiment_id":"phase2_6_b_only_calibration","calibration_only":True,"config_hash":cfg_hash,"git_commit_sha":commit})
    return summary


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--candidates",type=int); parser.add_argument("--workers",type=int); args=parser.parse_args()
    cfg26,source=load_phase2_6_config(); cfg25,_=load_phase2_5_config(ROOT/cfg26.frozen_phase2_5_config)
    count=args.candidates or cfg26.dynamic_search.initial_candidate_count
    if count not in (cfg26.dynamic_search.initial_candidate_count,cfg26.dynamic_search.expanded_candidate_count): raise ValueError("candidate count must equal configured budget")
    pose_path=ROOT/cfg26.output_dir/"b_pose_graspability"/"selected_poses.yaml"; poses=yaml.safe_load(pose_path.read_text(encoding="utf-8"))["selected_poses"]
    if len(poses)<cfg26.workspace.selected_pose_count: raise RuntimeError("geometry stage did not select 50 poses")
    cfg_hash=config_hash([source,ROOT/cfg26.frozen_phase2_config,ROOT/cfg26.frozen_phase2_5_config,pose_path,ROOT/"configs"/"hand_allegro.yaml",ROOT/"configs"/"scene_two_object.yaml",ROOT/"configs"/"task_sequential.yaml"])
    out=ROOT/cfg26.output_dir/"b_only_dynamic"/cfg_hash[:12]; store=IncrementalJsonlStore(out/"candidate_results.jsonl",30.,.05); completed=store.completed_ids()
    unit=qmc.LatinHypercube(70,seed=cfg26.seeds.pose_trajectory_search).random(cfg26.dynamic_search.expanded_candidate_count)
    pending=[i for i in range(count) if stable_trial_id("phase2.6-b-only",{"candidate_index":i,"config_hash":cfg_hash}) not in completed]
    workers=min(args.workers or max(1,(os.cpu_count() or 1)//2),cfg26.maximum_workers); commit=git_commit_sha(ROOT)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        buffer=[]
        payloads=((cfg25,poses[i%len(poses)],i,unit[i],cfg_hash,commit) for i in pending)
        for n,result in enumerate(executor.map(evaluate,payloads),1):
            buffer.append(result)
            if len(buffer)==workers or n==len(pending): store.append_many(buffer); buffer.clear()
            if n%(workers*4)==0 or n==len(pending): print(f"Phase 2.6 B-only: {len(completed)+n}/{count}",flush=True)
    rows=[r for r in store.records() if r.get("config_hash")==cfg_hash and int(r["candidate_index"])<count]; ranked=sorted(rows,key=b_only_lexicographic_key,reverse=True); successes=[r for r in ranked if r["B_acquired"]]
    summary={"status":"PASS" if successes else ("PHASE2_6_B_ONLY_DYNAMIC_GRASP_FAILED" if count==cfg26.dynamic_search.expanded_candidate_count else "EXPANSION_REQUIRED"),"candidate_count":count,"completed_candidates":len(rows),"successful_B_only_trajectories":len(successes),"successful_topologies":dict(Counter(r["geometry_topology"] for r in successes)),"failure_mechanisms":dict(Counter(r["failure_mechanism"] for r in rows if not r["B_acquired"])),"best":ranked[0] if ranked else None,"config_hash":cfg_hash}
    out.mkdir(parents=True,exist_ok=True); (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    for n,row in enumerate(successes[:20],1):
        pose=poses[int(row["candidate_index"])%len(poses)]; trajectory=trajectory_from_unit(cfg25,pose,int(row["candidate_index"]),unit[int(row["candidate_index"])]); (ROOT/"configs"/"grasps"/f"phase2_6_b_only_{n:02d}.yaml").write_text(yaml.safe_dump({"pose":pose,"trajectory":trajectory.__dict__,"source_candidate_index":row["candidate_index"],"calibration_only":True},sort_keys=False),encoding="utf-8")
    print(json.dumps(summary,indent=2)); return 0 if successes else (3 if count==cfg26.dynamic_search.expanded_candidate_count else 2)


if __name__=="__main__": raise SystemExit(main())
