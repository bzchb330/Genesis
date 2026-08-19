#!/usr/bin/env python
from __future__ import annotations
import json
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc
import yaml
from seqgrasp.config import ROOT
from seqgrasp.experiments.phase2_5_trajectory import run_b_acquisition_trajectory
from seqgrasp.experiments.phase2_6_dynamic import trajectory_from_unit
from seqgrasp.experiments.second_grasp import BPlacement
from seqgrasp.phase2_5_config import load_phase2_5_config
from seqgrasp.phase2_6_config import load_phase2_6_config

def main():
    c26,_=load_phase2_6_config(); c25,_=load_phase2_5_config(ROOT/c26.frozen_phase2_5_config); result=next((ROOT/c26.output_dir/"b_only_dynamic").glob("*/candidate_results.jsonl")); rows=[json.loads(x) for x in result.read_text(encoding="utf-8").splitlines() if x]; success=[r for r in rows if r["B_acquired"]]; poses=yaml.safe_load((ROOT/c26.output_dir/"b_pose_graspability"/"selected_poses.yaml").read_text(encoding="utf-8"))["selected_poses"]; unit=qmc.LatinHypercube(70,seed=c26.seeds.pose_trajectory_search).random(c26.dynamic_search.expanded_candidate_count); out=ROOT/c26.output_dir/"videos"; out.mkdir(parents=True,exist_ok=True); figdir=ROOT/"docs"/"figures"/"phase2_6"; plt.style.use(ROOT/"configs"/"phase2_publication.mplstyle"); fig,axes=plt.subplots(len(success),1,figsize=(8,2.5*len(success)),squeeze=False); frames=[]
    for i,row in enumerate(success):
        ci=int(row["candidate_index"]); pose=poses[ci%len(poses)]; traj=trajectory_from_unit(c25,pose,ci,unit[ci]); p=row["placement"]; placement=BPlacement(int(row["pose_candidate_index"]),tuple(p["position_m"]),tuple(p["quaternion"]),float(p["yaw_rad"])); video=out/f"b_only_success_{i+1:02d}.mp4"; summary,a=run_b_acquisition_trajectory(c25,traj,placement=placement,collect_timeseries=True,render_video_path=video,render_stride=5,video_fps=50); rel=a["timestep"]-summary["fixture_release_timestep"]; axes[i,0].plot(rel,a["B_hand_normal_force_N"],label="hand normal force [N]"); axes[i,0].plot(rel,a["B_hand_contacts"],label="hand contacts"); axes[i,0].axvline(0,color="black",linestyle="--"); axes[i,0].set(title=f"candidate {ci}",xlabel="step relative to release"); axes[i,0].legend(fontsize=7); frames.append(imageio.get_reader(video).get_data(0))
    fig.tight_layout(); fig.savefig(figdir/"b_only_success_release_traces.pdf"); plt.close(fig); fig,axes=plt.subplots(1,len(frames),figsize=(4*len(frames),4),squeeze=False)
    for i,frame in enumerate(frames): axes[0,i].imshow(frame); axes[0,i].set(title=f"B-only success {i+1}",xticks=[],yticks=[])
    fig.tight_layout(); fig.savefig(figdir/"b_only_successful_pose_examples.pdf"); plt.close(fig); print(json.dumps({"successes":len(success),"videos":[str(out/f'b_only_success_{i+1:02d}.mp4') for i in range(len(success))]},indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
