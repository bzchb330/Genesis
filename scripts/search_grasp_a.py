#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np

from seqgrasp import load_configs
from seqgrasp.diagnostics import analyze_contact_sequence,run_scripted_grasp
from seqgrasp.diagnostics.grasp_search import candidate_bundle,candidate_profile,comparison_plots,diagnostic_metrics,evaluate_candidate,generate_candidates,load_search_config,save_raw_trajectory,select_distinct,write_candidate_yaml,write_json

def _describe(values):
    a=np.asarray(values,dtype=float); return {"minimum":float(a.min()),"maximum":float(a.max()),"mean":float(a.mean()),"median":float(np.median(a))}

def _validation_summary(records,fingers):
    scalar_keys=("engineering_retention_score","final_vertical_displacement_m","minimum_post_release_height_m","maximum_post_release_height_m","maximum_post_release_translation_m","maximum_post_release_orientation_change_rad","fingertip_contact_duration_s","mean_active_object_fingers","minimum_object_fingertip_contact_distance_m","actuator_saturation_fraction","maximum_saturated_actuators")
    summary={key:_describe([record[key] for record in records]) for key in scalar_keys}; summary["table_recontact_time_s"]=_describe([record["table_recontact_time_s"] for record in records if record["table_recontact_time_s"] is not None]) if any(record["table_recontact_time_s"] is not None for record in records) else None; summary["runs_without_observed_table_recontact"]=sum(record["table_recontact_time_s"] is None for record in records); summary["runs_without_observed_complete_contact_loss"]=sum(record["first_complete_fingertip_contact_loss_s"] is None for record in records); summary["joint_limit_violation_runs"]=sum(record["joint_limit_violation_observed"] for record in records); summary["early_termination_reasons"]=[record["termination_reason"] for record in records if record["terminated_early"]]
    summary["peak_force_per_finger_N"]={finger:_describe([record["peak_force_per_finger_N"][finger] for record in records]) for finger in fingers}; summary["mean_force_per_finger_N"]={finger:_describe([record["mean_force_per_finger_N"][finger] for record in records]) for finger in fingers}; return summary

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--num-candidates",type=int); parser.add_argument("--validation-seeds",type=int); parser.add_argument("--video",action="store_true"); parser.add_argument("--search-config"); args=parser.parse_args()
    cfg=load_configs(); search_cfg=load_search_config(args.search_config); count=args.num_candidates or int(search_cfg["candidate_count"]); validation_seeds=args.validation_seeds or int(search_cfg["validation_seed_count"]); root=Path(search_cfg["output_dir"]); root.mkdir(parents=True,exist_ok=True)
    baseline=run_scripted_grasp(replace(cfg,diagnostic=replace(cfg.diagnostic,save_plots=False)),seed=int(search_cfg["screening_seed"]),save_outputs=False); baseline_metrics=diagnostic_metrics(baseline,cfg,search_cfg); failure=analyze_contact_sequence(baseline,cfg); write_json(root/"baseline_failure_analysis.json",{"metrics":baseline_metrics,"mechanics":failure})
    candidates=generate_candidates(cfg,search_cfg,count); by_id={candidate.candidate_id:candidate for candidate in candidates}; results=[]
    for index,candidate in enumerate(candidates,1):
        run,metrics=evaluate_candidate(cfg,search_cfg,candidate,int(search_cfg["screening_seed"])); save_raw_trajectory(root/"screening"/f"{candidate.candidate_id}.npz",run); results.append({"candidate_id":candidate.candidate_id,"parameters":asdict(candidate),"metrics":metrics})
        if index%10==0 or index==count: print(f"screened {index}/{count}; current score={metrics['engineering_retention_score']:.6f}",flush=True)
    top=select_distinct(results,by_id,list(cfg.hand.actuator_names),int(search_cfg["top_candidate_count"]),float(search_cfg["minimum_distinct_fraction_l2"])); write_json(root/"screening_summary.json",{"engineering_search_only":True,"method":search_cfg["method"],"search_seed":search_cfg["search_seed"],"candidate_count":count,"baseline":baseline_metrics,"ranked_candidates":sorted(results,key=lambda item:item["metrics"]["engineering_retention_score"],reverse=True),"selected_top_candidate_ids":[item["candidate_id"] for item in top]})
    top_runs={}; validation_payload={"engineering_search_only":True,"scientific_labels_assigned":False,"validation_seed_count":validation_seeds,"candidates":{}}
    config_dir=Path(search_cfg["candidate_config_dir"]); config_dir.mkdir(parents=True,exist_ok=True)
    for rank,result in enumerate(top,1):
        candidate=by_id[result["candidate_id"]]; profile=candidate_profile(cfg.diagnostic.profiles[cfg.diagnostic.active_profile],candidate); write_candidate_yaml(config_dir/f"grasp_A_candidate_{rank:02d}.yaml",candidate,result["metrics"],profile)
        candidate_cfg=candidate_bundle(cfg,candidate); candidate_cfg=replace(candidate_cfg,diagnostic=replace(candidate_cfg.diagnostic,save_plots=True,save_csv=True,save_npz=True,render_video=args.video,video_filename=f"grasp_A_candidate_{rank:02d}.mp4")); run=run_scripted_grasp(candidate_cfg,seed=int(search_cfg["screening_seed"]),output_dir=root/"top_candidates"/f"grasp_A_candidate_{rank:02d}",render_video=args.video); top_runs[candidate.candidate_id]=[]; records=[]
        for seed in range(validation_seeds):
            validation_run,metrics=evaluate_candidate(cfg,search_cfg,candidate,seed); save_raw_trajectory(root/"validation"/f"grasp_A_candidate_{rank:02d}"/f"seed_{seed:04d}.npz",validation_run); top_runs[candidate.candidate_id].append(validation_run); records.append({"seed":seed,**metrics})
        validation_payload["candidates"][f"grasp_A_candidate_{rank:02d}"]={"source_candidate_id":candidate.candidate_id,"screening_metrics":result["metrics"],"parameters":asdict(candidate),"runs":records,"descriptive_summary":_validation_summary(records,run.metadata["finger_order"]),"video_requested":args.video,"video_written":run.metadata["video_written"]}; print(f"validated top candidate {rank}/{len(top)} over {validation_seeds} seeds",flush=True)
    write_json(root/"validation_summary.json",validation_payload); best_id=top[0]["candidate_id"]; comparison_plots(root/"comparison_plots",baseline,top_runs[best_id][0],top_runs); write_json(root/"run_manifest.json",{"engineering_search_only":True,"candidate_count":count,"top_candidate_count":len(top),"validation_seed_count":validation_seeds,"baseline_score":baseline_metrics["engineering_retention_score"],"top_candidate_ids":[item["candidate_id"] for item in top],"candidate_config_files":[f"configs/grasps/grasp_A_candidate_{rank:02d}.yaml" for rank in range(1,len(top)+1)],"scientific_success_assigned":False})
    print(f"search complete: {count} candidates; top {len(top)} validated across {validation_seeds} seeds",flush=True); return 0

if __name__=="__main__": raise SystemExit(main())
