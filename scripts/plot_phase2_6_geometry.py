#!/usr/bin/env python
from __future__ import annotations
import json
import matplotlib.pyplot as plt
import numpy as np
import yaml
from seqgrasp.config import ROOT
from seqgrasp.phase2_6_config import load_phase2_6_config

def main():
    cfg,_=load_phase2_6_config(); root=ROOT/cfg.output_dir; rows=[json.loads(x) for x in (root/"b_pose_graspability"/"candidate_poses.jsonl").read_text(encoding="utf-8").splitlines() if x]; p=np.asarray([r["position_m"] for r in rows]); access=np.asarray([r["accessible_finger_count"] for r in rows]); opposition=np.asarray([r["opposition_available"] for r in rows]); epsilon=np.asarray([r["ferrari_canny_epsilon"] for r in rows]); selected=yaml.safe_load((root/"b_pose_graspability"/"selected_poses.yaml").read_text(encoding="utf-8"))["selected_poses"]; sp=np.asarray([r["position_m"] for r in selected]); frozen=yaml.safe_load((ROOT/"configs"/"phase2_6_frozen_B_distribution.yaml").read_text(encoding="utf-8")); b=frozen["center_bounds_m"]
    out=ROOT/"docs"/"figures"/"phase2_6"; plt.style.use(ROOT/"configs"/"phase2_publication.mplstyle"); fig,axes=plt.subplots(1,3,figsize=(11,3.3))
    for ax,(a,c,title) in zip(axes,((0,1,"XY"),(0,2,"XZ"),(1,2,"YZ"))):
        sc=ax.scatter(p[:,a],p[:,c],c=access,s=2,alpha=.35,cmap="viridis"); ax.scatter(sp[:,a],sp[:,c],marker="x",s=12,color="red"); ax.set(title=title,xlabel="xyz"[a]+" [m]",ylabel="xyz"[c]+" [m]")
    fig.colorbar(sc,ax=axes,label="accessible fingers"); fig.savefig(out/"multi_finger_workspace_intersection.pdf"); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8,3.5)); axes[0].scatter(p[:,0],p[:,1],c=opposition,s=2,cmap="coolwarm",alpha=.4); axes[0].set(xlabel="x [m]",ylabel="y [m]",title="Contact opposition"); sc=axes[1].scatter(p[:,0],p[:,2],c=np.log10(np.maximum(epsilon,1e-12)),s=2,cmap="plasma",alpha=.4); axes[1].set(xlabel="x [m]",ylabel="z [m]",title="Ferrari-Canny evidence"); fig.colorbar(sc,ax=axes[1],label="log10 epsilon"); fig.savefig(out/"B_pose_graspability_map.pdf"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,4)); ax.scatter(p[:,0],p[:,1],s=1,alpha=.08,color="gray"); ax.scatter(sp[:,0],sp[:,1],s=12,color="red",label="50 geometry poses"); ax.add_patch(plt.Rectangle((b["x"][0],b["y"][0]),b["x"][1]-b["x"][0],b["y"][1]-b["y"][0],fill=False,color="blue",linewidth=2,label="frozen sequential box")); ax.set(xlabel="x [m]",ylabel="y [m]",title="Phase 2.6 B-region selection"); ax.legend(); fig.savefig(out/"new_B_region_visualization.pdf"); plt.close(fig); return 0
if __name__=="__main__": raise SystemExit(main())
