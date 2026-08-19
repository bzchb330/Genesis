#!/usr/bin/env python
from __future__ import annotations
import json
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from seqgrasp.config import ROOT
from seqgrasp.phase2_6_config import load_phase2_6_config

CATEGORIES=("A_BLOCKS_B_APPROACH","FREE_FINGER_SET_INSUFFICIENT","PALM_SUPPORT_OCCUPIED","A_DESTABILIZED","B_CONTACT_LOST","B_SLIP","OTHER")

def interaction_failure(row):
    if str(row.get("invalid_reason") or "").startswith("initial_overlap:object_a"): return "A_BLOCKS_B_APPROACH"
    if not row.get("A_retained",True): return "A_DESTABILIZED"
    mechanism=row.get("failure_mechanism")
    if mechanism=="B_SLIPPED_TO_TABLE": return "B_SLIP"
    if mechanism=="CONTACT_LOST_IMMEDIATELY_AFTER_RELEASE": return "B_CONTACT_LOST"
    if mechanism in {"NO_B_CONTACT_BEFORE_RELEASE","SINGLE_UNOPPOSED_CONTACT","CONTACT_FORCE_TOO_LOW"}: return "FREE_FINGER_SET_INSUFFICIENT"
    return "OTHER"

def main():
    cfg,_=load_phase2_6_config(); root=ROOT/cfg.output_dir/"sequential_search"; summary_path=next(p for p in root.glob("*/summary.json") if json.loads(p.read_text())["candidate_count"]==8192 and json.loads(p.read_text())["config_hash"].startswith("41771a1d7470")); rows=[json.loads(x) for x in (summary_path.parent/"candidate_results.jsonl").read_text(encoding="utf-8").splitlines() if x]
    for row in rows: row["interaction_failure"]=interaction_failure(row)
    counts=Counter(row["interaction_failure"] for row in rows); by_A={}
    for grasp in sorted({row["A_grasp_id"] for row in rows}): by_A[grasp]=dict(Counter(row["interaction_failure"] for row in rows if row["A_grasp_id"]==grasp))
    out=ROOT/cfg.output_dir/"sequential_failure_analysis"; out.mkdir(parents=True,exist_ok=True); (out/"summary.json").write_text(json.dumps({"status":"PHASE2_6_SEQUENTIAL_INTERSECTION_FAILED","trials":len(rows),"BOTH_RETAINED":sum(row["BOTH_RETAINED"] for row in rows),"interaction_failure_counts":{key:counts.get(key,0) for key in CATEGORIES},"by_A_grasp":by_A},indent=2),encoding="utf-8")
    figdir=ROOT/"docs"/"figures"/"phase2_6"; figdir.mkdir(parents=True,exist_ok=True); plt.style.use(ROOT/"configs"/"phase2_publication.mplstyle"); fig,axes=plt.subplots(1,2,figsize=(10,3.8)); values=[counts.get(x,0) for x in CATEGORIES]; axes[0].bar(np.arange(len(CATEGORIES)),values); axes[0].set(xticks=np.arange(len(CATEGORIES)),xticklabels=CATEGORIES,ylabel="Trials",title="Sequential interaction failures"); axes[0].tick_params(axis="x",rotation=55)
    matrix=np.asarray([[by_A[g].get(c,0) for c in CATEGORIES] for g in by_A],dtype=float); matrix/=np.maximum(matrix.sum(axis=1,keepdims=True),1); bottom=np.zeros(len(by_A)); x=np.arange(len(by_A))
    for ci,c in enumerate(CATEGORIES): axes[1].bar(x,matrix[:,ci],bottom=bottom,label=c); bottom+=matrix[:,ci]
    axes[1].set(xlabel="Calibration A grasp",ylabel="Failure proportion",title="Failures by A grasp",ylim=(0,1)); axes[1].legend(fontsize=5,loc="upper right"); fig.tight_layout(); fig.savefig(figdir/"sequential_interaction_failures.pdf"); fig.savefig(figdir/"sequential_interaction_failures.png",dpi=180); plt.close(fig); print(json.dumps({"trials":len(rows),"counts":counts},default=dict,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
