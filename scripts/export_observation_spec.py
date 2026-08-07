#!/usr/bin/env python
import argparse, json
from pathlib import Path
from seqgrasp import load_configs
from seqgrasp.env.observations import observation_spec

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",default="docs/observation_spec.json"); args=p.parse_args()
    spec=observation_spec(load_configs()); path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(spec,indent=2),encoding="utf-8"); print(path)
if __name__=="__main__": main()
