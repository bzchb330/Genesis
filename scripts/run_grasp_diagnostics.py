#!/usr/bin/env python
import argparse
from pathlib import Path
from seqgrasp import load_configs
from seqgrasp.diagnostics import aggregate_summaries,plot_aggregate,run_scripted_grasp,summarize_run
from seqgrasp.diagnostics.characterization import write_summary

def main():
    p=argparse.ArgumentParser(); p.add_argument("--num-seeds",type=int); p.add_argument("--output-dir"); p.add_argument("--profile"); args=p.parse_args(); cfg=load_configs()
    n=cfg.diagnostic.num_seeds if args.num_seeds is None else args.num_seeds; root=Path(args.output_dir or cfg.diagnostic.output_dir)/"seeded_runs"; runs=[]; summaries=[]
    for seed in range(n):
        run=run_scripted_grasp(cfg,seed=seed,output_dir=root/f"seed_{seed:04d}",render_video=False,profile_name=args.profile); runs.append(run); summaries.append(summarize_run(run,cfg))
    aggregate=aggregate_summaries(runs,summaries); root.mkdir(parents=True,exist_ok=True); write_summary(root/"summary.json",runs,summaries,aggregate); plot_aggregate(runs,root/"aggregate_plots")
    import json; print(json.dumps({"runs":summaries,"aggregate":aggregate},indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
