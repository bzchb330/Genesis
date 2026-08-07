#!/usr/bin/env python
import argparse, json
from seqgrasp import load_configs
from seqgrasp.diagnostics import run_scripted_grasp

def main():
    p=argparse.ArgumentParser(); p.add_argument("--seed",type=int); p.add_argument("--output-dir"); p.add_argument("--video",action="store_true"); args=p.parse_args()
    run=run_scripted_grasp(load_configs(),seed=args.seed,output_dir=args.output_dir,render_video=args.video or None)
    print(json.dumps(run.metadata,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
