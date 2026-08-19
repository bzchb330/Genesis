#!/usr/bin/env python
from __future__ import annotations

import json, math
import numpy as np
from scipy.stats import qmc
import yaml

from seqgrasp.config import ROOT
from seqgrasp.experiments.phase2_5_trajectory import b_only_lexicographic_key, run_b_acquisition_trajectory
from seqgrasp.experiments.phase2_6_dynamic import trajectory_from_unit
from seqgrasp.experiments.resumable import IncrementalJsonlStore, stable_trial_id
from seqgrasp.experiments.second_grasp import BPlacement
from seqgrasp.phase2_5_config import load_phase2_5_config
from seqgrasp.phase2_6_config import load_phase2_6_config


def main():
    cfg26,_=load_phase2_6_config(); cfg25,_=load_phase2_5_config(ROOT/cfg26.frozen_phase2_5_config)
    robust=yaml.safe_load((ROOT/"configs"/"phase2_6_robustness.yaml").read_text(encoding="utf-8")); root=ROOT/cfg26.output_dir/"b_only_dynamic"
    result_file=next(root.glob("*/candidate_results.jsonl")); rows=[json.loads(x) for x in result_file.read_text(encoding="utf-8").splitlines() if x]; successes=sorted([r for r in rows if r["B_acquired"]],key=b_only_lexicographic_key,reverse=True)
    poses=yaml.safe_load((ROOT/cfg26.output_dir/"b_pose_graspability"/"selected_poses.yaml").read_text(encoding="utf-8"))["selected_poses"]
    unit=qmc.LatinHypercube(70,seed=cfg26.seeds.pose_trajectory_search).random(cfg26.dynamic_search.expanded_candidate_count); selected=successes[:int(robust["profiles_to_validate"])]
    out=ROOT/cfg26.output_dir/"b_only_robustness"; store=IncrementalJsonlStore(out/"trials.jsonl",30.,.05); completed=store.completed_ids(); width=np.asarray(robust["position_half_width_m"])
    for profile_index,row in enumerate(selected):
        pose=poses[int(row["candidate_index"])%len(poses)]; trajectory=trajectory_from_unit(cfg25,pose,int(row["candidate_index"]),unit[int(row["candidate_index"])]); samples=qmc.LatinHypercube(4,seed=np.random.default_rng(np.random.SeedSequence([cfg26.seeds.perturbations,profile_index]))).random(int(robust["trials_per_profile"]))
        for trial_index,sample in enumerate(samples):
            trial_id=stable_trial_id("phase2.6-b-only-robustness",{"source":row["candidate_index"],"trial":trial_index})
            if trial_id in completed: continue
            delta=(2.0*sample[:3]-1.0)*width; yaw=float(np.interp(sample[3],[0,1],[-robust["yaw_half_width_rad"],robust["yaw_half_width_rad"]])); center=np.asarray(pose["position_m"])+delta
            placement=BPlacement(trial_index,tuple(center),(math.cos(yaw/2),0.,0.,math.sin(yaw/2)),yaw); summary,arrays=run_b_acquisition_trajectory(cfg25,trajectory,placement=placement,collect_timeseries=True); final=np.asarray(arrays["B_per_finger_contact_flag"][-1])
            summary.update({"trial_id":trial_id,"source_candidate_index":row["candidate_index"],"robustness_trial_index":trial_index,"position_delta_m":delta.tolist(),"yaw_delta_rad":yaw,"final_contact_topology":"+".join(name for name,active in zip(("index","middle","ring","thumb"),final) if active),"final_per_finger_normal_force_N":arrays["B_per_finger_normal_force_N"][-1].tolist(),"calibration_only":True,"experiment_id":"phase2_6_b_only_robustness"}); store.append(summary)
        print(f"robustness profile {profile_index+1}/{len(selected)}",flush=True)
    trials=store.records(); by_profile={}
    for row in selected:
        subset=[x for x in trials if x["source_candidate_index"]==row["candidate_index"]]; by_profile[str(row["candidate_index"]) ]={"trials":len(subset),"successes":sum(x["B_acquired"] for x in subset),"success_fraction":sum(x["B_acquired"] for x in subset)/len(subset) if subset else 0.,"contact_topologies":sorted({x["final_contact_topology"] for x in subset if x["B_acquired"]})}
    summary={"validated_profiles":len(selected),"trials":len(trials),"successes":sum(x["B_acquired"] for x in trials),"overall_success_fraction":sum(x["B_acquired"] for x in trials)/len(trials) if trials else 0.,"by_profile":by_profile}; (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8"); print(json.dumps(summary,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
