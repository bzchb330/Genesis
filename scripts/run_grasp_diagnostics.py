#!/usr/bin/env python
import argparse, json
from pathlib import Path
import numpy as np
from seqgrasp import load_configs
from seqgrasp.diagnostics import run_scripted_grasp

def main():
    p=argparse.ArgumentParser(); p.add_argument("--num-seeds",type=int); p.add_argument("--output-dir"); args=p.parse_args(); cfg=load_configs()
    n=cfg.diagnostic.num_seeds if args.num_seeds is None else args.num_seeds; root=Path(args.output_dir or cfg.diagnostic.output_dir)/"seeded_runs"; summaries=[]
    for seed in range(n):
        run=run_scripted_grasp(cfg,seed=seed,output_dir=root/f"seed_{seed:04d}",render_video=False); a=run.arrays; forces=a["finger_total_normal_force_raw"]
        summaries.append({"seed":seed,"final_object_pose":np.r_[a["object_position"][-1],a["object_orientation"][-1]].tolist(),"maximum_object_height_m":float(a["object_position"][:,2].max()),"minimum_finger_normal_force_N":float(forces.min()),"maximum_finger_normal_force_N":float(forces.max()),"fingertip_contact_duration_s":float((a["finger_contact_count"].sum(axis=1)>0).sum()*cfg.scene.timestep),"per_finger_contact_samples":dict(zip(run.metadata["finger_order"],(a["finger_contact_count"]>0).sum(axis=0).astype(int).tolist())),"completed":not run.metadata["terminated_early"],"terminated":run.metadata["terminated_early"],"termination_reason":run.metadata["termination_reason"]})
    root.mkdir(parents=True,exist_ok=True); (root/"summary.json").write_text(json.dumps(summaries,indent=2),encoding="utf-8"); print(json.dumps(summaries,indent=2))
if __name__=="__main__": main()
